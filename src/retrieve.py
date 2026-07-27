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

import re
import sys

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from config import HYBRID_WEIGHTS, RETRIEVER_K, USE_HYBRID
from vectorstore import build_documents, get_vector_store

_dense_cache = {}
_bm25_index = None


def get_dense_retriever(top_k=RETRIEVER_K):
    """Matches by MEANING. Cached per k."""
    if top_k not in _dense_cache:
        _dense_cache[top_k] = get_vector_store().as_retriever(search_kwargs={"k": top_k})
    return _dense_cache[top_k]


def tokenize(text):
    """Lowercase, and split on anything that is not a letter or digit.

    ⚠️ THIS IS NOT OPTIONAL, AND ITS ABSENCE IS INVISIBLE. LangChain's BM25Retriever
    preprocesses with `text.split()` -- no lowercasing, no punctuation handling. This corpus
    is UPPERCASE ("PROCTOLOGY is a ... (OTHER BUSINESS).") and questions are lowercase
    ("was a proctologist excluded?"), so with the default every meaningful query token missed.

    Measured 2026-07-27: BM25 could not find the single PROCTOLOGY record from the query
    "proctology", but found it instantly from "PROCTOLOGY". The retriever returned results the
    whole time -- just useless ones -- so an ablation run against it looked like evidence that
    keyword search does not help here. It was evidence that it was never switched on.

    `text.split()` also leaves punctuation attached: "BUSINESS)." is a different token from
    "business", so parenthesised fields were unreachable too.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def get_keyword_retriever(top_k=RETRIEVER_K):
    """Matches by WORD (BM25), over exactly the text that was indexed.

    Built once per process: tokenising 8,482 records on every question is wasted work, since
    the corpus only changes at ingest. `k` is set on the shared index rather than baked into
    a cache key -- it is a different read of the same index, not a different index.
    """
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Retriever.from_documents(
            build_documents(), preprocess_func=tokenize)
    _bm25_index.k = top_k
    return _bm25_index


def get_retriever(top_k=RETRIEVER_K, hybrid=USE_HYBRID, weights=HYBRID_WEIGHTS):
    """Dense alone, or dense merged with BM25.

    Each leg fetches top_k so the merged list can still yield top_k good records once the two
    rankings disagree. An even weighting is the honest default -- tune it against the golden
    set, never by looking at one answer.
    """
    if not hybrid:
        return get_dense_retriever(top_k=top_k)
    return EnsembleRetriever(
        retrievers=[get_dense_retriever(top_k=top_k), get_keyword_retriever(top_k=top_k)],
        weights=list(weights),
    )


def retrieve(question, top_k=RETRIEVER_K, hybrid=USE_HYBRID):
    return get_retriever(top_k=top_k, hybrid=hybrid).invoke(question)


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
