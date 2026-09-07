"""Fetch the exclusion records most similar in meaning to a question.

Extracted from notebook 04. Hybrid retrieval: the question goes to a dense leg (embedding
similarity, Qdrant) and a keyword leg (BM25) at the same time, and an EnsembleRetriever
combines the two rankings. `RETRIEVER_K` is 10.

Both halves of that started as inherited defaults -- dense only, k=3 -- and neither was
changed until `src/golden_set.py` made it measurable. What the measurement said, over the
20 answerable questions:

    dense only,  k=3    hit 0.650   MRR 0.525   record recall 0.545
    dense only,  k=10   hit 0.800   MRR 0.556   record recall 0.697
    dense+BM25,  k=10   hit 1.000   MRR 0.827   record recall 1.000   <- shipped

k mattered most for record recall, because these questions have several correct answers and
returning one of three scores as a hit while missing the rest. BM25 mattered for reaching a
record by its literal words: at k=10 the dense leg still misses 4 of the 20 answerable
questions and the hybrid misses none.

Those literal words only exist in the index because of `src/vocabulary.py`, which writes each
record's specialty and state in the words a person would use as well as the words the LEIE
uses. Before that, three questions were unreachable by either leg -- see that module.

⚠️ The keyword leg only works because of `tokenize` below. With LangChain's default it
returns results for every query, silently useless ones, and an ablation against it reads as
evidence that keyword search does not help here.
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


SOURCE_FIELDS = [("NAME", ""), ("NPI", "NPI: "), ("SPECIALTY", ""), ("STATE", ""),
                 ("EXCLDATE", "excluded: "), ("GENERAL", "")]


def format_sources(documents):
    """One checkable line per record, showing only the fields present on it.

    Fields are looked up rather than assumed, because a role-filtered record genuinely does
    not have them all -- an analyst's records carry no NAME. The first version indexed the
    metadata directly and raised KeyError the moment RBAC was wired in, which is the right
    kind of failure (loud) but the wrong place for it: a formatter should render what it is
    given, not dictate what it must be given.
    """
    lines = []
    for index, document in enumerate(documents, start=1):
        parts = [f"{label}{document.metadata[key]}"
                 for key, label in SOURCE_FIELDS if key in document.metadata]
        lines.append(f"  {index}. " + " | ".join(parts))
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
