"""Turns each LEIE record into one sentence and stores it in Qdrant, so RAG can search by meaning.

1. A sentence, not a table row: the embedding model compares sentences.
2. Only the 8,482 records with an NPI: the same providers the XGBoost model scored.
3. "Also written as" words added at index time: a question says cardiologist, the LEIE says CARDIOLOGY.
4. No NPI in the sentence: it blurred the meaning search (MRR 0.8267 -> 0.7800); BM25 carries the NPI.
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

# The LEIE columns that describe an exclusion; the rest identify a person or are mostly empty.
RAG_COLUMNS = ["NPI", "LASTNAME", "FIRSTNAME", "BUSNAME", "SPECIALTY", "STATE",
               "EXCLTYPE", "EXCLDATE", "GENERAL"]

# --- Vocabulary: each record also in the words a person would type ---
# Only clear LEIE short forms; an unsure one is left out, because a wrong one puts a lie in the index.
ABBREVIATIONS = {
    "ACF": "adult care facility",
    "CNTR": "center",
    "CO": "company",
    "COMM": "community",
    "CONGLOM": "conglomerate",
    "CTR": "center",
    "DME": "durable medical equipment",
    "EQ": "equipment",
    "FAC": "facility",
    "FACI": "facility",
    "FACIL": "facility",
    "FP": "family practice",
    "GEN": "general",
    "GOV": "government",
    "GYN": "gynecology",
    "HC": "healthcare",
    "HE": "health",
    "HLTH": "health",
    "IDTF": "independent diagnostic testing facility",
    "MANUF": "manufacturer",
    "MGMT": "management",
    "MNTL": "mental",
    "OBS": "obstetrics",
    "ORGANIZAT": "organization",
    "PHYS": "physician",
    "PHYSIATRIS": "physiatrist",
    "PRACT": "practice",
    "PRACTITIONE": "practitioner",
    "PROSTHETIS": "prosthetist",
    "PROVID": "provider",
    "PROVIDE": "provider",
    "RECIPT": "recipient",
    "REHA": "rehabilitation",
    "REHAB": "rehabilitation",
    "SUPLIER": "supplier",
    "SUPP": "supplier",
    "SUPPL": "supplier",
    "SVCS": "services",
    "TRANS": "transportation",
    "UNK": "unknown",
}

# Person words no suffix rule gets right (PHARMACY -> pharmacist).
IRREGULAR_PERSON_FORMS = {
    "ACUPUNCTURE": "acupuncturist",
    "CHIROPRACTIC": "chiropractor",
    "COUNSELING": "counselor",
    "DENTAL": "dentist",
    "GENETICS": "geneticist",
    "MEDICINE": "physician",
    "NURSING": "nurse",
    "ORTHOPEDICS": "orthopedist",
    "PEDIATRICS": "pediatrician",
    "PHARMACY": "pharmacist",
    "SURGERY": "surgeon",
    "THERAPY": "therapist",
}

# Person words by suffix (CARDIOLOGY -> cardiologist).
PERSON_FORM_SUFFIXES = (
    ("OMETRY", "ometrist"),
    ("OTOMY", "otomist"),
    ("IATRY", "iatrist"),
    ("OLOGY", "ologist"),
    ("PATHY", "path"),
)

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
    "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
    "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "AS": "American Samoa", "GU": "Guam", "MP": "Northern Mariana Islands",
    "PR": "Puerto Rico", "VI": "Virgin Islands",
}


def _text(value):
    """Gives the field as text; an empty LEIE field comes back from pandas as NaN."""
    return value if isinstance(value, str) else ""


def person_form(specialty):
    """Gives the person word for a whole specialty (PROCTOLOGY -> proctologist), or None."""
    key = _text(specialty).strip().upper()
    if key in IRREGULAR_PERSON_FORMS:
        return IRREGULAR_PERSON_FORMS[key]
    for suffix, replacement in PERSON_FORM_SUFFIXES:
        if key.endswith(suffix) and len(key) > len(suffix):
            return key[: -len(suffix)].lower() + replacement
    return None


def spell_out(specialty):
    """The specialty with its truncated words written out. None when nothing was truncated."""
    tokens = _text(specialty).replace("/", " ").split()
    if not tokens:
        return None
    expanded = [ABBREVIATIONS.get(token.upper(), token.lower()) for token in tokens]
    spelled = " ".join(expanded)
    return spelled if spelled != " ".join(token.lower() for token in tokens) else None


def also_written_as(specialty, state):
    """Gives the other ways a person writes this specialty and state; empty for most records."""
    forms = []

    spelled = spell_out(specialty)
    if spelled:
        forms.append(spelled)

    person = person_form(specialty)
    if person:
        forms.append(person)
        forms.append(person + "s")

    state_name = STATE_NAMES.get(_text(state).strip().upper())
    if state_name:
        forms.append(state_name)

    return forms


_embeddings = None


def get_embeddings():
    """One embedding model per process -- loading it costs ~90 MB and a few seconds."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings


_client = None


def get_client():
    """Gives one Qdrant client per process: embedded when QDRANT_PATH is set, else the server."""
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
    """Gives one record as a sentence, plus an "Also written as" part in everyday words."""
    sentence = (f"{provider_name(row)} is a {row['SPECIALTY']} in {row['STATE']} "
                f"who was excluded on {row['EXCLDATE']} "
                f"for {row['EXCLTYPE']} ({row['GENERAL']}).")

    forms = also_written_as(row["SPECIALTY"], row["STATE"])
    if forms:
        sentence += " Also written as: " + ", ".join(forms) + "."
    return sentence


def build_documents(leie=None):
    """Gives one Document per record, with the fields kept as metadata so each answer can cite its NPI."""
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
    """Drops and rebuilds the Qdrant collection, so it can never hold two copies of a record."""
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
