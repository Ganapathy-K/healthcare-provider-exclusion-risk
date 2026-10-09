"""Serves the agent on Cloud Run as one /ask endpoint, so any program can call it over HTTP.

1. Its own Cloud Run service, apart from serving/: the agent's heavy packages cannot break the scorer.
2. Qdrant embedded in the Docker image: no second service to run, pay for and secure.
3. Models load at startup, not on the first question: Cloud Run reads a slow start as a failure.

Run locally:  QDRANT_PATH=../data/qdrant_store uvicorn app:app --reload
"""

import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# src/ sits beside app.py in the Docker image and one level up in the repo; both are checked.
for candidate in (Path(__file__).resolve().parent / "src",
                  Path(__file__).resolve().parent.parent / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))
        break

from agent import build_agent  # noqa: E402
from agent import ask as run_agent  # noqa: E402
from rbac import get_role  # noqa: E402
from retrieve import format_sources, retrieve  # noqa: E402
from vectorstore import get_client, get_embeddings  # noqa: E402
from config import QDRANT_COLLECTION_NAME  # noqa: E402


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3, max_length=500,
        description="A question about OIG exclusions, or a request for one provider's risk score.",
        examples=["Which acupuncturists in New York were excluded?"],
    )
    role: str = Field(
        default="public",
        description=(
            "Which records and fields the caller may see. `investigator` (everything), "
            "`analyst` (de-identified: no names, no NPIs), `auditor` (organisations only), "
            "`public` (nothing). Unknown roles fall back to `public`."
        ),
        examples=["analyst"],
    )

    # No login here: any caller can claim a role. A real deployment takes role from a verified token.


class Source(BaseModel):
    """One cited record; every field is optional because a role-filtered record has fewer."""

    npi: int | None = None
    name: str | None = None
    specialty: str | None = None
    state: str | None = None
    excluded_on: str | None = None
    category: str | None = None


class AskResponse(BaseModel):
    """The typed answer: tool tells a model prediction from a record; a refusal is still HTTP 200."""

    tool: Literal["query_leie_rag", "score_provider_risk"] = Field(
        description="The tool whose answer this is.")
    tools_run: list[Literal["query_leie_rag", "score_provider_risk"]] = Field(
        description="Every tool that ran, in order. Two entries mean the first answer failed the "
                    "agent's check and the other tool ran.")
    status: Literal["answered", "refused"]
    role: str = Field(description="The role actually applied, after unknown names fall back "
                                  "to `public`. Echoed so a caller can see a typo took effect.")
    answer: str
    npi: str | None = Field(description="The NPI extracted from the question, if any.")
    sources: list[Source] = Field(
        description="Empty on a refusal and on the scoring branch. Records are only evidence "
                    "for an answer that was actually drawn from them.")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    retriever_ready: bool
    indexed_records: int | None


ready = False
indexed = None
_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the embedding model and Qdrant before the first request, so the first caller does not wait."""
    global ready, indexed, _agent
    try:
        def warm():
            get_embeddings()
            count = get_client().get_collection(QDRANT_COLLECTION_NAME).points_count
            retrieve("warm up", top_k=1)
            return count

        indexed = await asyncio.to_thread(warm)
        _agent = build_agent()
        ready = True
    except Exception as error:
        logging.exception("warm-up failed: %s", error)
    yield


app = FastAPI(
    title="Healthcare Provider Exclusion Agent",
    version="1.0.0",
    summary="Routes a question to the exclusion-risk model or to grounded retrieval over the "
            "OIG LEIE.",
    description=(
        "Two tools behind one endpoint. Ask for a provider's risk score and it runs the "
        "XGBoost model; ask about the exclusion records and it answers from retrieved records "
        "only, citing the NPI, and refuses when the records do not support an answer.\n\n"
        "Decision support: scores prioritise human review and are not findings about anyone."
    ),
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    """Reports whether the retriever loaded, not only that the server is up."""
    return HealthResponse(
        status="ok" if ready else "degraded",
        retriever_ready=ready,
        indexed_records=indexed,
    )


@app.post("/ask", response_model=AskResponse, tags=["agent"])
async def ask(request: AskRequest):
    """Runs the agent in a thread, so one slow question does not block the other callers."""
    if not ready:
        raise HTTPException(status_code=503, detail="Retriever is not loaded yet.")

    role = get_role(request.role)

    try:
        result = await asyncio.to_thread(run_agent, request.question, role.name, _agent)
    except Exception as error:
        raise HTTPException(status_code=502,
                            detail=f"Agent failed: {type(error).__name__}: {error}") from error

    npi = result["npi"]
    documents = result["documents"]
    # Sources go out as typed fields, so the text "Sources:" part is taken off the answer.
    answer = result["answer"].removesuffix(f"\n\nSources:\n{format_sources(documents)}")

    # A hidden field is left out, never sent as "None".
    sources = [
        Source(**{key: value for key, value in (
            ("npi", doc.metadata.get("NPI")),
            ("name", doc.metadata.get("NAME")),
            ("specialty", doc.metadata.get("SPECIALTY")),
            ("state", doc.metadata.get("STATE")),
            ("excluded_on", doc.metadata.get("EXCLDATE")),
            ("category", doc.metadata.get("GENERAL")),
        ) if value is not None})
        for doc in documents
    ]

    return AskResponse(
        tool=result["tool_used"],
        tools_run=result["tools_run"],
        status="refused" if result["refused"] else "answered",
        role=role.name,
        answer=answer,
        npi=npi if "NPI" in role.visible_fields and npi else None,
        sources=sources,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
