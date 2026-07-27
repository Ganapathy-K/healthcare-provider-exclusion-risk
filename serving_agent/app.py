"""The agent as a service: one endpoint that routes to the risk model or to retrieval.

The scoring model has been deployed since June; the agent has only ever run in a notebook or
on a laptop. That gap is the whole point of this file -- a routing agent that cannot be called
by anything is a demonstration, not a system, and "production-ready, not POCs" is a phrase
that appears in the job descriptions this project exists to answer.

Deliberately a SEPARATE Cloud Run service from `serving/` rather than more endpoints on it.
The scorer's image is small and its dependencies are pandas and xgboost; the agent needs
torch, sentence-transformers, langchain, langgraph and the Gemini SDK. Folding them together
would put a working, deployed service at risk of a dependency it does not need, to save one
deployment.

Qdrant runs EMBEDDED here -- a directory baked into the image, no server, no network. Cloud
Run gives one container and one port, so a Qdrant server would mean a second service to run,
pay for and secure, for a read-only index of 8,482 records. Its exclusive file lock is fine
because each instance is its own single-process container, but it does mean this service
cannot reindex itself: rebuilding is an ingest-time job.

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

# In the image, src/ sits beside app.py at /app/src. In the repo, app.py is in serving_agent/
# and src/ is its sibling one level up. Both are checked so the same file runs in both places
# -- a service that can only be exercised after a container build is a service nobody tests.
for candidate in (Path(__file__).resolve().parent / "src",
                  Path(__file__).resolve().parent.parent / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))
        break

from agent import build_agent, query_leie_rag, score_provider_risk  # noqa: E402
from agent import classify_intent, extract_npi  # noqa: E402
from generate import REFUSAL_TEXT  # noqa: E402
from rbac import get_role  # noqa: E402
from retrieve import retrieve  # noqa: E402
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

    # ⚠️ A ROLE IN THE REQUEST BODY IS NOT AUTHENTICATION, and this API has none. Any caller
    # can claim to be an investigator, so what is demonstrated here is the ENFORCEMENT
    # mechanism -- filtering before the model sees anything -- not the identity check in front
    # of it. In a real deployment the role comes from a verified token, never from the caller.
    # Saying so is better than letting a reviewer assume this is access control end to end.


class Source(BaseModel):
    """One cited record. Every field is optional because a role-filtered record genuinely has
    fewer of them -- an analyst's records carry no name and no NPI."""

    npi: int | None = None
    name: str | None = None
    specialty: str | None = None
    state: str | None = None
    excluded_on: str | None = None
    category: str | None = None


class AskResponse(BaseModel):
    """The typed answer.

    `tool` is what makes this API usable by another program: the two branches answer
    fundamentally different questions -- one is a model PREDICTION about a provider, the other
    is a statement of RECORD about the past -- and a caller must be able to tell them apart
    without parsing prose. Conflating a prediction with a record is the mistake this whole
    project exists to avoid making about real, named people.

    `status` distinguishes a grounded answer from a refusal. A refusal returns HTTP 200: it is
    the system working correctly, not a malformed request.
    """

    tool: Literal["query_leie_rag", "score_provider_risk"]
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
    """Load the embedding model and open the index before the first request.

    Cold-starting inside the first request looks like a hang to whoever opened the link, and
    Cloud Run treats a container that is slow to accept traffic as a failing one. The warm-up
    belongs in startup, where the platform is still waiting for the port.
    """
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
    """Reports whether the RETRIEVER is loaded, not merely that the server is up.

    This process can answer HTTP perfectly while every question fails because the embedding
    model never loaded or the index directory is missing from the image.
    """
    return HealthResponse(
        status="ok" if ready else "degraded",
        retriever_ready=ready,
        indexed_records=indexed,
    )


@app.post("/ask", response_model=AskResponse, tags=["agent"])
async def ask(request: AskRequest):
    """Route the question, run the chosen tool, and say which one ran.

    `asyncio.to_thread` is load-bearing: the router call, the embedding pass and the answer
    call all block for seconds, and awaiting them on the event loop would serialise every
    other caller behind the first.
    """
    if not ready:
        raise HTTPException(status_code=503, detail="Retriever is not loaded yet.")

    role = get_role(request.role)

    try:
        decision = await asyncio.to_thread(classify_intent, request.question)
    except Exception as error:
        raise HTTPException(status_code=502,
                            detail=f"Router failed: {type(error).__name__}: {error}") from error

    npi = decision["npi"] or extract_npi(request.question)

    if decision["intent"] == "risk":
        # The scoring branch is investigator-only. A risk score is about ONE named provider by
        # construction, so there is no de-identified version of it: returning a score for an
        # NPI the caller supplied confirms that provider is in the dataset, which is itself
        # disclosure. Roles that cannot see identities cannot use this branch at all.
        if "NPI" not in role.visible_fields:
            return AskResponse(
                tool="score_provider_risk", status="refused", role=role.name,
                answer=f"The '{role.name}' role cannot retrieve provider-level risk scores. "
                       "Scoring identifies a specific provider.",
                npi=None, sources=[])

        answer = await asyncio.to_thread(score_provider_risk, npi)
        return AskResponse(tool="score_provider_risk", status="answered", role=role.name,
                           answer=answer, npi=npi or None, sources=[])

    try:
        from generate import answer_question
        answer, documents = await asyncio.to_thread(
            answer_question, request.question, role=role)
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Retrieval pipeline failed: {type(error).__name__}: {error}") from error

    refused = REFUSAL_TEXT.lower() in answer.lower()
    # Built from what each record actually carries, so a redacted field is absent rather than
    # rendered as the string "None" -- which would leak the SHAPE of what was withheld.
    sources = [] if refused else [
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
        tool="query_leie_rag",
        status="refused" if refused else "answered",
        role=role.name,
        answer=answer,
        npi=npi if "NPI" in role.visible_fields and npi else None,
        sources=sources,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
