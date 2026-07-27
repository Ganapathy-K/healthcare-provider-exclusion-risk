"""Turn LEIE exclusion records into searchable documents and index them in Qdrant.

Extracted from notebook 04. Each excluded provider becomes one short sentence -- name,
specialty, state, date, reason -- because that is what an embedding model can compare. A
table row cannot be searched by meaning; a sentence can.

Only the 8,306 LEIE records carrying a valid NPI are indexed, matching the labelled dataset:
the two halves of this project must agree on who counts as excluded, or the agent will answer
questions about providers the model has never scored.

⚠️ Unlike the insurance project, this Qdrant is a SERVER (Docker, localhost:6333), not an
embedded file. Nothing here works with the container stopped, and the failure is a connection
error rather than an empty result -- which is the better of the two.
"""

import sys

import pandas as pd
from langchain_community.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from config import (EMBEDDING_DIM, EMBEDDING_MODEL_NAME, QDRANT_COLLECTION_NAME,
                    QDRANT_PATH, QDRANT_URL)
from ingest import load_leie

DISTANCE_METRIC = Distance.COSINE

# The columns that describe an exclusion. Everything else in the LEIE (addresses, dates of
# birth, reinstatement fields) is either identifying or empty for most rows.
RAG_COLUMNS = ["NPI", "LASTNAME", "FIRSTNAME", "BUSNAME", "SPECIALTY", "STATE",
               "EXCLTYPE", "EXCLDATE", "GENERAL"]

_embeddings = None


def get_embeddings():
    """One embedding model per process -- loading it costs ~90 MB and a few seconds."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings


_client = None


def get_client():
    """The Qdrant handle: embedded when QDRANT_PATH is set, otherwise the server.

    One client per process, because the embedded store takes an exclusive file lock and a
    second client on the same directory raises AlreadyLocked. Anything that builds two
    retrievers -- the hybrid retriever does -- would hit that immediately.
    """
    global _client
    if _client is None:
        _client = (QdrantClient(path=QDRANT_PATH) if QDRANT_PATH
                   else QdrantClient(url=QDRANT_URL))
    return _client


def provider_name(row):
    """Organisations carry BUSNAME; individuals carry FIRSTNAME/LASTNAME. Never both."""
    if pd.notna(row["BUSNAME"]):
        return row["BUSNAME"]
    return f"{row['FIRSTNAME']} {row['LASTNAME']}"


def to_sentence(row):
    """One record as a sentence an embedding model can compare against a question."""
    return (f"{provider_name(row)} is a {row['SPECIALTY']} in {row['STATE']} "
            f"who was excluded on {row['EXCLDATE']} "
            f"for {row['EXCLTYPE']} ({row['GENERAL']}).")


def build_documents(leie=None):
    """The indexable documents, with the structured fields kept as metadata.

    The metadata matters as much as the text: an answer that cannot name the NPI it came from
    is not checkable, and this is exclusion data -- being wrong about a named provider is the
    expensive kind of wrong.
    """
    if leie is None:
        leie = load_leie()
    records = leie.loc[leie["NPI"] != 0, RAG_COLUMNS].to_dict(orient="records")

    return [
        Document(
            page_content=to_sentence(row),
            metadata={
                "NPI": row["NPI"],
                "NAME": provider_name(row),
                "SPECIALTY": row["SPECIALTY"] if pd.notna(row["SPECIALTY"]) else "Unknown",
                "STATE": row["STATE"],
                "EXCLTYPE": row["EXCLTYPE"],
                "EXCLDATE": str(row["EXCLDATE"]),
                "GENERAL": row["GENERAL"],
            },
        )
        for row in records
    ]


def get_vector_store(client=None):
    """A handle on the existing collection. Does not build or modify it."""
    return QdrantVectorStore(
        client=client or get_client(),
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=get_embeddings(),
    )


def build_collection(documents=None):
    """(Re)create the collection and index every document. Destructive by design.

    The collection is dropped first rather than appended to. Appending is how an index
    silently ends up holding two copies of everything, which shows up later as duplicate
    citations that look like a retrieval bug.
    """
    documents = documents if documents is not None else build_documents()
    client = get_client()

    if client.collection_exists(QDRANT_COLLECTION_NAME):
        client.delete_collection(QDRANT_COLLECTION_NAME)
    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=DISTANCE_METRIC),
    )

    get_vector_store(client).add_documents(documents)
    return client.get_collection(QDRANT_COLLECTION_NAME).points_count


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    rebuilding = "--rebuild" in sys.argv

    if rebuilding:
        print("rebuilding the collection (drops the existing one first)...")
        print(f"indexed {build_collection():,} vectors")
    else:
        client = get_client()
        if not client.collection_exists(QDRANT_COLLECTION_NAME):
            raise SystemExit(f"collection '{QDRANT_COLLECTION_NAME}' does not exist; "
                             "run with --rebuild")
        info = client.get_collection(QDRANT_COLLECTION_NAME)
        documents = build_documents()
        print(f"collection : {QDRANT_COLLECTION_NAME}")
        print(f"indexed    : {info.points_count:,} vectors ({info.status})")
        print(f"would index: {len(documents):,} documents from the LEIE")
        print(f"in sync    : {info.points_count == len(documents)}")
        print(f"\nsample     : {documents[0].page_content}")
        print("\nnothing changed. re-run with --rebuild to reindex.")
