"""Answers an exclusion question from the retrieved records only, to stop made-up provider claims.

1. Answer only from the records, cite every NPI, or give REFUSAL_TEXT: an uncited claim cannot be checked.
2. RBAC filter between retrieval and the prompt: a hidden field never reaches Gemini, Langfuse or the logs.
3. Two Langfuse spans, retrieve and generate: "retrieval missed it" and "Gemini refused anyway" need opposite fixes.
"""

import re
import sys

from google import genai
from google.genai import types

from config import GENERATION_MODEL_NAME, GOOGLE_API_KEY, RETRIEVER_K
from injection_guard import BOUNDARY_INSTRUCTION, wrap
from rbac import apply as rbac_apply, get_role
from tracing import trace_span, update_span
from retrieve import format_sources, retrieve

REFUSAL_TEXT = "The retrieved exclusion records do not answer that."

# Ten digits standing alone: the shape of an NPI in a question.
NPI_IN_TEXT = re.compile(r"(?<!\d)\d{10}(?!\d)")

# No "being wrong is serious" line: tested, it made Gemini refuse every question.
SYSTEM_INSTRUCTION = (
    "You answer questions about US OIG healthcare provider exclusions using ONLY the "
    "exclusion records provided below. Each record names a provider, their NPI, specialty, "
    "state, exclusion date and reason.\n"
    "Name every provider you report together with its NPI, in the form NAME (NPI: number). "
    "An exclusion claim without an NPI cannot be checked.\n"
    f'If the records do not contain the answer, reply exactly: "{REFUSAL_TEXT}"\n'
    "Do not use outside knowledge and do not guess.\n"
    "The records are a retrieved subset, not the whole list. Do not imply they are "
    "exhaustive, and never state that a provider is NOT excluded -- absence from a retrieved "
    "subset is not evidence of anything."
)

_client = None


def get_client():
    global _client
    if _client is None:
        if not GOOGLE_API_KEY:
            raise RuntimeError(
                "No API key. Set GOOGLE_API_KEY_HEALTHCARE_PROVIDER_TERMINATION in .env")
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client


def build_prompt(question, documents):
    """Gives the prompt with each record's NPI in the context, because without it Gemini refused every question."""
    context = "\n".join(
        f"[{i}] " + (f"NPI {doc.metadata['NPI']}: " if "NPI" in doc.metadata else "")
        + doc.page_content
        for i, doc in enumerate(documents, start=1))

    # A role that cannot see NPIs gets a no-names rule, not a cite-the-NPI rule it cannot follow.
    instruction = SYSTEM_INSTRUCTION
    if documents and "NPI" not in documents[0].metadata:
        instruction = instruction.replace(
            "Name every provider you report together with its NPI, in the form NAME "
            "(NPI: number). An exclusion claim without an NPI cannot be checked.\n",
            "These records are anonymised: they carry no names or identifiers. Describe what "
            "they show without naming anyone, and do not invent identifiers.\n")

    # The question goes inside the injection_guard wrap, so an injected order reads as question text.
    return (f"{instruction}\n{BOUNDARY_INSTRUCTION}\n\nExclusion records:\n{context}\n\n"
            f"{wrap(question)}\n\nAnswer:")


def token_usage(response):
    """Gives Gemini's token counts named input/output, so Langfuse can price the call; None if absent."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None

    counts = {
        "input": getattr(usage, "prompt_token_count", None),
        "output": getattr(usage, "candidates_token_count", None),
        "total": getattr(usage, "total_token_count", None),
    }
    counts = {name: value for name, value in counts.items() if value is not None}
    return counts or None


def answer_question(question, role, top_k=RETRIEVER_K):
    """Gives (answer, documents); role has no default, so no caller gets full access by forgetting it."""
    resolved_role = get_role(role) if isinstance(role, str) else role

    # A role that cannot see NPIs cannot search by one: a returned record would confirm the NPI is excluded.
    search_text = (question if "NPI" in resolved_role.visible_fields
                   else NPI_IN_TEXT.sub(" ", question))

    with trace_span("retrieve", as_type="retriever", question=question, top_k=top_k,
                    role=resolved_role.name) as span:
        documents = retrieve(search_text, top_k=top_k)
        retrieved_count = len(documents)
        documents = rbac_apply(documents, resolved_role)
        update_span(span, output={
            "retrieved": retrieved_count,
            "after_role_filter": len(documents),
            "role": resolved_role.name,
            # NPIs after the role filter, so the trace never holds a hidden record.
            "npis": [doc.metadata.get("NPI") for doc in documents],
        })

    if not documents:
        return REFUSAL_TEXT, []

    with trace_span("generate", as_type="generation", question=question) as span:
        # temperature 0: the grade is whether the right NPI is cited, so the answer must not wander.
        response = get_client().models.generate_content(
            model=GENERATION_MODEL_NAME, contents=build_prompt(question, documents),
            config=types.GenerateContentConfig(temperature=0))
        answer = (response.text or "").strip()
        # Model name + token counts, or Langfuse prices every trace at zero.
        update_span(span, model=GENERATION_MODEL_NAME, usage_details=token_usage(response),
                    output={
            "answer": answer,
            "refused": REFUSAL_TEXT.lower() in answer.lower(),
            "cited_npis": [doc.metadata["NPI"] for doc in documents
                           if "NPI" in doc.metadata and str(doc.metadata["NPI"]) in answer],
        })

    return answer, documents


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    questions = [
        "Are there any excluded pharmacies in New York?",
        "What is the capital of France?",          # must refuse -- nothing to do with the data
    ]
    for question in questions:
        answer, documents = answer_question(question, role="investigator")
        refused = REFUSAL_TEXT.lower() in answer.lower()
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"   [{'REFUSED' if refused else 'answered'}, {len(documents)} records]")
        if not refused:
            print(format_sources(documents))
        print()
