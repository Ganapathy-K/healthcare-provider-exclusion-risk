"""Holds every path, setting and model number in one place, so no two files can disagree.

1. Paths built from this file's location, not typed out: the repo runs on any machine and in Docker.
2. Each name defined once: the RAG code and the agent code cannot point at different Qdrant collections.
3. One API key name, with GOOGLE_API_KEY as the fallback.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SERVING_DIR = PROJECT_ROOT / "serving"
LOG_DIR = PROJECT_ROOT / "logs"

# --- source data -------------------------------------------------------------------------
# NPPES = every provider; LEIE = the excluded ones. NPPES_DIR env var moves the 1 GB zip.
NPPES_DIR = Path(os.getenv("NPPES_DIR", r"D:/Data Science/Datasets/Medical/NPPES"))
NPPES_ZIP_PATH = NPPES_DIR / "raw" / "NPPES_Data_Dissemination_March_2026_V2.zip"
NPPES_EXTRACT_DIR = NPPES_DIR / "extracted"
NPPES_FILENAME = "npidata_pfile_20050523-20260308.csv"
NPPES_PATH = NPPES_EXTRACT_DIR / NPPES_FILENAME

# Every number in docs/baseline.json uses the first 500,000 rows; changing this breaks the baseline.
NPPES_SAMPLE_ROWS = 500_000

LEIE_PATH = RAW_DIR / "oig_leie_202602.csv"
LABELLED_DATASET_PATH = PROCESSED_DIR / "labelled_dataset.parquet"

# The 12 columns the risk scorer needs, out of 331: 60 MB down to 8.3 MB, so the Docker image starts faster.
PROVIDER_LOOKUP_PATH = PROCESSED_DIR / "provider_lookup.parquet"
LOOKUP_COLUMNS = [
    "NPI", "Entity Type Code", "Provider Business Mailing Address Telephone Number",
    "Provider Enumeration Date", "Last Update Date", "Provider Sex Code",
    "Healthcare Provider Primary Taxonomy Switch_1", "Is Sole Proprietor",
    "Healthcare Provider Taxonomy Code_1", "Provider Business Mailing Address State Name",
    "Provider Business Practice Location Address State Name",
    "Provider License Number State Code_1",
]

NUCC_TAXONOMY_URL = "https://nucc.org/images/stories/CSV/nucc_taxonomy_251.csv"

# --- model -------------------------------------------------------------------------------
TARGET_COLUMN = "excluded"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Drop a column over 30% empty; treat a text column with over 1000 values as an ID, not a category.
MAX_NULL_FRACTION = 0.30
HIGH_CARDINALITY_LIMIT = 1000

# 422 = not-excluded rows per excluded row in the train rows, so missing one excluded provider costs 422x.
SCALE_POS_WEIGHT = 422
N_ESTIMATORS = 100
MAX_DEPTH = 6
LEARNING_RATE = 0.1

MODEL_PATH = SERVING_DIR / "model.ubj"
ENCODING_MAPS_PATH = SERVING_DIR / "encoding_maps.json"
MLFLOW_TRACKING_URI = (PROJECT_ROOT / "notebooks" / "mlruns").resolve().as_uri()

# 0.5 is checked, not a default: with SCALE_POS_WEIGHT = 422 it is already the cost-best cut-off (threshold.py).
RISK_THRESHOLD = 0.5

# --- retrieval ---------------------------------------------------------------------------
# Qdrant server on localhost by default; set QDRANT_PATH for the embedded store inside the Cloud Run image.
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_PATH = os.getenv("QDRANT_PATH") or None
QDRANT_COLLECTION_NAME = "leie_exclusions"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
# 10, not 3: record recall 0.545 -> 0.697 on the golden set, because list questions have several right NPIs.
RETRIEVER_K = 10

# Dense + BM25 at k=10: hit rate 0.800 -> 1.000, MRR 0.556 -> 0.827, record recall 0.697 -> 1.000.
USE_HYBRID = True
HYBRID_WEIGHTS = (0.5, 0.5)

# --- generation --------------------------------------------------------------------------
GENERATION_MODEL_NAME = "gemini-2.5-flash"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY_HEALTHCARE_PROVIDER_TERMINATION") or os.getenv(
    "GOOGLE_API_KEY"
)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    print(f"project root : {PROJECT_ROOT}")
    print(f"API key set  : {GOOGLE_API_KEY is not None}")
    print("\nexpected inputs:")
    for label, path in [("NPPES zip", NPPES_ZIP_PATH), ("NPPES csv", NPPES_PATH),
                        ("LEIE", LEIE_PATH),
                        ("labelled dataset", LABELLED_DATASET_PATH),
                        ("model", MODEL_PATH), ("encoding maps", ENCODING_MAPS_PATH)]:
        print(f"  {'OK     ' if path.exists() else 'MISSING'}  {label:<18} {path}")
