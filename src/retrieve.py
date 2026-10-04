"""Hybrid retrieval: dense (Qdrant) and keyword (BM25) legs merged by an EnsembleRetriever, k=10."""

import re
import sys

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from rank_bm25 import BM25Okapi

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
    """Lowercase, split on non-letters and non-digits: "PROCTOLOGY" matches "proctology"."""
    return re.findall(r"[a-z0-9]+", text.lower())


def keyword_tokens(document):
    """The words BM25 matches a record on: its sentence, plus its NPI."""
    return tokenize(document.page_content) + [str(document.metadata["NPI"])]


def get_keyword_retriever(top_k=RETRIEVER_K):
    """Matches by WORD (BM25). Built once, with the same record text as the dense leg."""
    global _bm25_index
    if _bm25_index is None:
        documents = build_documents()
        _bm25_index = BM25Retriever(
            vectorizer=BM25Okapi([keyword_tokens(document) for document in documents]),
            docs=documents, preprocess_func=tokenize)
    _bm25_index.k = top_k
    return _bm25_index


def get_retriever(top_k=RETRIEVER_K, hybrid=USE_HYBRID, weights=HYBRID_WEIGHTS):
    """Dense alone, or dense merged with BM25."""
    if not hybrid:
        return get_dense_retriever(top_k=top_k)
    return EnsembleRetriever(
        retrievers=[get_dense_retriever(top_k=top_k), get_keyword_retriever(top_k=top_k)],
        weights=list(weights),
    )


def retrieve(question, top_k=RETRIEVER_K, hybrid=USE_HYBRID):
    """The top_k exclusion records for one question."""
    return get_retriever(top_k=top_k, hybrid=hybrid).invoke(question)


SOURCE_FIELDS = [("NAME", ""), ("NPI", "NPI: "), ("SPECIALTY", ""), ("STATE", ""),
                 ("EXCLDATE", "excluded: "), ("GENERAL", "")]


def format_sources(documents):
    """One line per record, showing only the fields present (an analyst's records have no NAME)."""
    lines = []
    for index, document in enumerate(documents, start=1):
        parts = [f"{label}{document.metadata[key]}"
                 for key, label in SOURCE_FIELDS if key in document.metadata]
        lines.append(f"  {index}. " + " | ".join(parts))
    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    question = "Which providers were excluded in Texas?"
    documents = retrieve(question)
    print(f"Q: {question}  ({len(documents)} records)")
    print(format_sources(documents))
