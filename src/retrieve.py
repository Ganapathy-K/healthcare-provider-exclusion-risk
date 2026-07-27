"""Fetch the exclusion records most similar in meaning to a question.

Extracted from notebook 04. Dense retrieval only: the question is embedded, and the k nearest
record-sentences come back.

⚠️ TWO KNOWN WEAKNESSES, recorded rather than quietly fixed. Both are answered in the sibling
insurance project, and both fixes port over almost unchanged:

1. **k = 3, and no measurement says it should be.** Three records is very few for questions
   like "which providers were excluded in Texas?", where the honest answer is a list. There is
   no golden set here yet, so there is no evidence for any value of k -- `src/retrieval_eval.py`
   in the insurance project computes hit-rate and MRR for free and needs only a set of
   questions with known answers.

2. **No keyword leg.** Dense embeddings blur exact tokens, and this corpus is full of them:
   NPIs, exclusion codes like `1128b8`, state abbreviations. Asking about a specific NPI is the
   case where meaning-search is weakest and BM25 is strongest. The insurance project measured
   this: dense-only caused 16 wrong refusals out of 37 questions.

Neither is fixed here because extraction should reproduce behaviour first. Fixing them is the
next piece of real work on this half of the project.
"""

import sys

from config import RETRIEVER_K
from vectorstore import get_vector_store

_retriever_cache = {}


def get_retriever(top_k=RETRIEVER_K):
    """Cached per k -- rebuilding the handle reloads nothing, but the embedding model behind
    it is shared, and a cache keeps that explicit."""
    if top_k not in _retriever_cache:
        _retriever_cache[top_k] = get_vector_store().as_retriever(
            search_kwargs={"k": top_k})
    return _retriever_cache[top_k]


def retrieve(question, top_k=RETRIEVER_K):
    return get_retriever(top_k=top_k).invoke(question)


def format_sources(documents):
    """One checkable line per record. The NPI is the point -- it is the only field that
    identifies a provider unambiguously, and an exclusion claim without it cannot be verified."""
    lines = []
    for index, document in enumerate(documents, start=1):
        meta = document.metadata
        lines.append(f"  {index}. {meta['NAME']} | NPI: {meta['NPI']} | {meta['SPECIALTY']} "
                     f"| {meta['STATE']} | excluded: {meta['EXCLDATE']} | {meta['GENERAL']}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    for question in ["Are there any excluded pharmacies in New York?",
                     "Which providers were excluded in Texas?",
                     "Who was excluded for patient abuse?"]:
        documents = retrieve(question)
        print(f"Q: {question}  ({len(documents)} records)")
        print(format_sources(documents))
        print()
