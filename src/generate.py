"""Answer a question about exclusions from the retrieved records, and nothing else.

Extracted from notebook 04, with one change that is called out below rather than slipped in.

⚠️ WHAT THE NOTEBOOK'S PROMPT DID, AND WHY IT IS NOT KEPT VERBATIM. It was:

    "Based on the following LEIE exclusion records:\n{context}\n\nAnswer this question: {query}"

That instructs the model to *consider* the records. It does not forbid using its own knowledge,
does not require citing an NPI, and gives it no way to say the records do not answer the
question -- so the one behaviour that makes a grounded system trustworthy, refusing, is
unavailable to it. On exclusion data that is the expensive failure: a fluent sentence naming a
real provider as excluded, assembled from the model's memory rather than the retrieved rows.

The insurance project measured this failure class and it is not hypothetical -- there, a model
turned "22.5% of the policy premium" into "₹22,500" while citing the correct page.

The grounding instruction below is therefore a deliberate correction, not an extraction. It is
also NOT yet measured: there is no golden set for this half of the project, so nothing here
proves the refusal fires when it should. That is the next piece of work.
"""

import sys

from google import genai

from config import GENERATION_MODEL_NAME, GOOGLE_API_KEY, RETRIEVER_K
from rbac import apply as rbac_apply, get_role
from tracing import trace_span, update_span
from retrieve import format_sources, retrieve

REFUSAL_TEXT = "The retrieved exclusion records do not answer that."

# ⚠️ A WARNING THAT BACKFIRED, KEPT HERE BECAUSE IT COST AN HOUR AND WILL RECUR.
# This instruction first contained the line: "These records concern real, named people and
# organisations, and a provider wrongly described as excluded is a serious error." It reads
# like responsible prompt engineering. It made the model refuse EVERYTHING -- including
# "are there any excluded pharmacies in New York?", with three New York pharmacies sitting in
# the context. A/B tested against the same records: with the line, refusal; without it, a
# correct cited answer. Telling a model that being wrong is dangerous does not make it more
# careful, it makes it decline. Constrain what it may SAY; never editorialise about stakes.
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
    """Render the retrieved records, INCLUDING the NPI, as the prompt's context.

    ⚠️ THE BUG THIS FIXES, WHICH LOOKED LIKE FOUR OTHER THINGS FIRST. The indexed sentence
    (`vectorstore.to_sentence`) names the provider, specialty, state, date and code -- but NOT
    the NPI, which lives only in metadata. Rendering only `page_content` therefore produced a
    context with no NPI in it, while the instruction demanded an NPI for every provider named.

    The model resolved that contradiction the only way it could: it decided the records did
    not answer the question, and took the refusal. So a grounded, correctly-retrieved,
    obviously-answerable question ("are there any excluded pharmacies in New York?", with
    three New York pharmacies in context) refused every single time.

    The tell was that removing the refusal option made it answer perfectly. A model given an
    impossible instruction and an escape hatch will use the escape hatch -- and the resulting
    refusal is indistinguishable from correct grounded behaviour, which is what makes this
    class of bug expensive. An unmeasured refusal rate hides it completely.
    """
    context = "\n".join(
        f"[{i}] " + (f"NPI {doc.metadata['NPI']}: " if "NPI" in doc.metadata else "")
        + doc.page_content
        for i, doc in enumerate(documents, start=1))

    # The citation rule has to bend for roles that cannot see the field it names. Demanding an
    # NPI from a context that contains none is the exact contradiction that made this model
    # refuse everything once already -- see the warning above. So the instruction is relaxed
    # rather than the context padded.
    instruction = SYSTEM_INSTRUCTION
    if documents and "NPI" not in documents[0].metadata:
        instruction = instruction.replace(
            "Name every provider you report together with its NPI, in the form NAME "
            "(NPI: number). An exclusion claim without an NPI cannot be checked.\n",
            "These records are anonymised: they carry no names or identifiers. Describe what "
            "they show without naming anyone, and do not invent identifiers.\n")

    return (f"{instruction}\n\nExclusion records:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:")


def answer_question(question, top_k=RETRIEVER_K, role="investigator"):
    """Retrieve, enforce the role, ground, answer. Returns (answer, documents).

    The role is applied BETWEEN retrieval and the prompt, which is the only position where it
    means anything: once a record is in the prompt it has reached the model provider, the
    trace and the logs, and asking the model to keep quiet about it is a request, not a
    control. See rbac.py.

    The default is `investigator` -- full access -- because every internal caller here (the
    eval scripts, the CLI) is doing exclusion review. The API defaults the other way, to
    `public`, because there the caller is unknown. Defaults should follow who is asking.

    Traced in two separate spans on purpose. "Retrieval missed the record" and "retrieval
    found it and the model refused anyway" look identical from the outside and need opposite
    fixes -- one is a retriever problem, the other a prompt problem. Recording the retrieved
    NPIs alongside the final answer is what makes them tellable apart afterwards instead of
    reproducible only by hand.
    """
    resolved_role = get_role(role) if isinstance(role, str) else role

    with trace_span("retrieve", as_type="retriever", question=question, top_k=top_k,
                    role=resolved_role.name) as span:
        documents = retrieve(question, top_k=top_k)
        retrieved_count = len(documents)
        documents = rbac_apply(documents, resolved_role)
        update_span(span, output={
            "retrieved": retrieved_count,
            "after_role_filter": len(documents),
            "role": resolved_role.name,
            # The NPIs AFTER filtering. Logging the pre-filter list would put records the role
            # may not see into the trace -- which is the same leak the filter exists to stop.
            "npis": [doc.metadata.get("NPI") for doc in documents],
        })

    if not documents:
        return REFUSAL_TEXT, []

    with trace_span("generate", as_type="generation", question=question) as span:
        response = get_client().models.generate_content(
            model=GENERATION_MODEL_NAME, contents=build_prompt(question, documents))
        answer = (response.text or "").strip()
        update_span(span, output={
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
        answer, documents = answer_question(question)
        refused = REFUSAL_TEXT.lower() in answer.lower()
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"   [{'REFUSED' if refused else 'answered'}, {len(documents)} records]")
        if not refused:
            print(format_sources(documents))
        print()
