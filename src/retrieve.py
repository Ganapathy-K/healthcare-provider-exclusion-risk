"""Hybrid retrieval, dense (Qdrant) + BM25, to find records by meaning and by exact word."""

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
    """The Qdrant retriever, cached per k, to match by meaning."""
    if top_k not in _dense_cache:
        _dense_cache[top_k] = get_vector_store().as_retriever(search_kwargs={"k": top_k})
    return _dense_cache[top_k]


def tokenize(text):
    """Lowercase letter-and-digit words, to make "PROCTOLOGY" match "proctology"."""
    return re.findall(r"[a-z0-9]+", text.lower())


def keyword_tokens(document):
    """A record's sentence words plus its NPI, to let BM25 find a record by NPI."""
    return tokenize(document.page_content) + [str(document.metadata["NPI"])]


def get_keyword_retriever(top_k=RETRIEVER_K):
    """The BM25 retriever, built once, to match by exact word."""
    global _bm25_index
    if _bm25_index is None:
        documents = build_documents()
        _bm25_index = BM25Retriever(
            vectorizer=BM25Okapi([keyword_tokens(document) for document in documents]),
            docs=documents, preprocess_func=tokenize)
    _bm25_index.k = top_k
    return _bm25_index


def get_retriever(top_k=RETRIEVER_K, hybrid=USE_HYBRID, weights=HYBRID_WEIGHTS):
    """Dense alone or dense + BM25 merged, to switch hybrid on or off."""
    if not hybrid:
        return get_dense_retriever(top_k=top_k)
    return EnsembleRetriever(
        retrievers=[get_dense_retriever(top_k=top_k), get_keyword_retriever(top_k=top_k)],
        weights=list(weights),
    )


def retrieve(question, top_k=RETRIEVER_K, hybrid=USE_HYBRID):
    """The top_k exclusion records for a question, to ground the answer."""
    return get_retriever(top_k=top_k, hybrid=hybrid).invoke(question)


SOURCE_FIELDS = [("NAME", ""), ("NPI", "NPI: "), ("SPECIALTY", ""), ("STATE", ""),
                 ("EXCLDATE", "excluded: "), ("GENERAL", "")]


def format_sources(documents):
    """One line per record with only the fields it has, to show sources any role may see."""
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
