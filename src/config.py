"""Central configuration for the provider-exclusion pipeline.

Everything the six notebooks hard-coded now lives here once. Three things this fixes, none
of them cosmetic:

1. **Absolute paths.** Notebooks 01, 02, 03 and 04 each contained
   `Path("D:/Data Science/Visual Studio Code/healthcare-provider-exclusion-risk/data/...")`.
   That path exists on exactly one machine, so the repo cannot run for anyone who clones it
   -- including inside a container, which is where it is meant to end up. Paths are now
   derived from this file's own location.

2. **Constants defined twice.** `qdrant_collection_name`, `embedding_model_name` and
   `gemini_model_name` were declared independently in notebooks 04 and 05. They agree today.
   Nothing was keeping them in agreement, and a retrieval pipeline whose two halves point at
   different collections fails in the least obvious way possible: it answers, fluently, from
   the wrong index.

3. **Two different API key names.** Notebook 04 read `GOOGLE_API_KEY`; notebook 05 read
   `GOOGLE_API_KEY_HEALTHCARE_PROVIDER_TERMINATION`. Whichever was set decided which
   notebook worked. The project-specific name wins here, with the generic one as a fallback.
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
# NPPES is the national registry of every provider; LEIE is the OIG's list of the excluded.
# The join of the two is the labelled dataset everything downstream trains and answers on.
#
# NPPES ships as a ~1 GB zip and lives OUTSIDE the repo, in the shared datasets folder, by
# convention: raw third-party data is never copied into a project. That makes its location
# machine-specific, which is exactly what this file exists to avoid -- so it is an
# environment variable with a documented default rather than a bare hard-coded path. Set
# NPPES_DIR to relocate it; nothing else needs to change.
NPPES_DIR = Path(os.getenv("NPPES_DIR", r"D:/Data Science/Datasets/Medical/NPPES"))
NPPES_ZIP_PATH = NPPES_DIR / "raw" / "NPPES_Data_Dissemination_March_2026_V2.zip"
NPPES_EXTRACT_DIR = NPPES_DIR / "extracted"
NPPES_FILENAME = "npidata_pfile_20050523-20260308.csv"
NPPES_PATH = NPPES_EXTRACT_DIR / NPPES_FILENAME

# The registry is ~9M rows. Every number in docs/baseline.json describes the first 500,000,
# so changing this invalidates the recorded baseline and the model with it.
NPPES_SAMPLE_ROWS = 500_000

LEIE_PATH = RAW_DIR / "oig_leie_202602.csv"
LABELLED_DATASET_PATH = PROCESSED_DIR / "labelled_dataset.parquet"

# The 12 columns the risk scorer needs, split out of the 331-column labelled dataset. 60 MB
# down to 8.3 MB -- worth doing because this is what ships inside the agent's container image,
# and cold-start time scales with image size. Written by `python src/ingest.py --save`; the
# agent falls back to the full dataset when it is absent.
PROVIDER_LOOKUP_PATH = PROCESSED_DIR / "provider_lookup.parquet"

NUCC_TAXONOMY_URL = "https://nucc.org/images/stories/CSV/nucc_taxonomy_251.csv"

# --- model -------------------------------------------------------------------------------
TARGET_COLUMN = "excluded"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Preparation thresholds, inherited from notebook 03. A column more than 30% empty is dropped
# rather than imputed; an object column with over 1000 distinct values is treated as an
# identifier rather than a category.
MAX_NULL_FRACTION = 0.30
HIGH_CARDINALITY_LIMIT = 1000

# Excluded providers are rare, so an unweighted model can score well by predicting "not
# excluded" for everyone. 422 is the observed negative:positive ratio in the training split,
# which is what makes recall -- catching the rare positive -- the metric that matters here.
SCALE_POS_WEIGHT = 422
N_ESTIMATORS = 100
MAX_DEPTH = 6
LEARNING_RATE = 0.1
EVAL_METRIC = "aucpr"

MODEL_PATH = SERVING_DIR / "model.ubj"
ENCODING_MAPS_PATH = SERVING_DIR / "encoding_maps.json"
MLFLOW_TRACKING_URI = (PROJECT_ROOT / "notebooks" / "mlruns").resolve().as_uri()

# The probability above which a provider is called high-risk. 0.5 is the inherited default
# and has never been tuned against the precision/recall curve notebook 03 already plots --
# for a queue that prioritises human review, the right threshold is a capacity decision, not
# a statistical one.
RISK_THRESHOLD = 0.5

# --- retrieval ---------------------------------------------------------------------------
# Two ways to reach Qdrant, and which one is in use is an environment decision, not a code one.
#
#   server    (default, local dev)  the Docker container on localhost:6333
#   embedded  (set QDRANT_PATH)     a plain directory, no server, no network
#
# Cloud Run runs one container per instance and gives it one port, so a separate Qdrant server
# would mean a second service to run, pay for and secure. The embedded store is a directory
# that can be baked into the image at build time, which for a read-only index of 8,482 records
# is the right shape: nothing to start, nothing to connect to, nothing to fall over.
# Its limitation is real and worth naming: the embedded store takes an exclusive file lock, so
# only one process per container may touch it, and it cannot be written to by a running
# service. Both are fine for an index rebuilt at ingest.
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_PATH = os.getenv("QDRANT_PATH") or None
QDRANT_COLLECTION_NAME = "leie_exclusions"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
# Was 3, inherited from notebook 04 with nothing behind it. Measured on the golden set
# (src/ablation.py, 2026-07-27): hit@k is flat from 3 to 10, but RECORD RECALL -- how many of
# the correct records a list-style question actually gets back -- climbs 0.600 -> 0.867. The
# questions this corpus attracts ("which adult homes in Texas were excluded?") have several
# correct answers, and returning one of three is a wrong answer that scores as a hit.
RETRIEVER_K = 10

# Dense (meaning) + BM25 (exact words), measured and kept: at k=10 it improves hit rate
# 0.778 -> 0.889, MRR 0.667 -> 0.778 and record recall 0.800 -> 0.867, and it is the only
# thing that reaches the single PHLEBOTOMY record.
# ⚠️ BM25 is worthless here without src/retrieve.tokenize -- see the warning in that function.
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
