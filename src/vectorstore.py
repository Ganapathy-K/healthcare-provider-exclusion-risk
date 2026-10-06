"""Turn LEIE exclusion records into searchable documents and index them in Qdrant.

Each excluded provider becomes one short sentence -- name,
specialty, state, date, reason -- because that is what an embedding model can compare. A
table row cannot be searched by meaning; a sentence can.

Only the 8,482 LEIE records carrying a valid NPI are indexed, matching the labelled dataset:
the two halves of this project must agree on who counts as excluded, or the agent will answer
questions about providers the model has never scored. Those 8,482 rows cover 8,306 unique NPIs
-- 176 providers hold two exclusion records each. Retrieval indexes the rows, because each row
is a separate exclusion event; the labelling join uses the unique NPIs, because a provider is
excluded or not. Both counts are correct and they are not interchangeable.

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

# --- Vocabulary: say each record in the words a person would use, as well as the LEIE's. ---
# Measured 2026-09-07: three golden-set questions were refused with the answer in the file,
# because question and record used different words for the same thing:
#     "cardiologists" vs CARDIOLOGY · "proctologist" vs PROCTOLOGY ·
#     "community mental health centers" vs COMM MNTL HLTH CNTR
# Rewording the question found every one (0/2 -> 2/2, 0/1 -> 1/1), so it is a vocabulary
# problem, not a ranking one. Neither leg closes it alone (no stemmer in BM25; -ology and
# -ologist are two words), so both forms are written into the sentence at INDEX time.

# LEIE specialty fields are truncated to fit a fixed width. Only unambiguous ones are listed;
# tokens whose intent is not obvious from the corpus (T, K, BELO, GRADE) are left alone,
# because a wrong expansion indexes a lie.
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

# A question names the PERSON ("was a proctologist excluded?"); the file names the FIELD
# (PROCTOLOGY). These are the ones no suffix rule gets right.
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

# Everything else that follows a rule. Ordered longest-first so OMETRY beats OTOMY-style
# overlaps and nothing matches on a shorter tail by accident.
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
    """The field as a string. LEIE leaves SPECIALTY and STATE empty on some rows, and pandas
    hands those back as a float NaN rather than as a missing string."""
    return value if isinstance(value, str) else ""


def person_form(specialty):
    """The word for the PERSON, given the file's word for the field. None when there isn't one.

    Only whole specialties are converted, not tokens inside them: "MENTAL/BEHAVIORAL HE" has
    no person form, and inventing one puts a word in the index that no record supports.
    """
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
    """Every other way a person might write this record's specialty and state.

    Returned as a list so the caller decides the phrasing, and empty when the file's own
    wording is already the wording a person would use -- most records need nothing.
    """
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
    """One record as a sentence an embedding model can compare against a question.

    A second sentence carries the same specialty and state in the words a person would use --
    see the Vocabulary block above for the three golden-set questions that made this necessary. It is
    appended rather than substituted because the file's own wording has to stay searchable
    too: someone who types COMM MNTL HLTH CNTR must still find the record.

    The NPI is deliberately NOT in the sentence. A ten-digit number means nothing to the
    meaning search, and adding it blurred it (golden set MRR 0.8267 -> 0.7800). The keyword
    search carries the NPI instead -- see `retrieve.keyword_tokens`.
    """
    sentence = (f"{provider_name(row)} is a {row['SPECIALTY']} in {row['STATE']} "
                f"who was excluded on {row['EXCLDATE']} "
                f"for {row['EXCLTYPE']} ({row['GENERAL']}).")

    forms = also_written_as(row["SPECIALTY"], row["STATE"])
    if forms:
        sentence += " Also written as: " + ", ".join(forms) + "."
    return sentence


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
