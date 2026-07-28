# Container for the LangGraph AGENT service (serving_agent/app.py) on Cloud Run.
#
# This sits at the project root rather than beside the app it builds, and that is deliberate:
# the agent needs src/, the LEIE csv, the provider lookup and the model artefacts, all of
# which live above serving_agent/. A Dockerfile can only COPY from its build context, so the
# context has to be the project root -- which means the file has to be here.
#
# The OTHER service, the XGBoost scorer, has its own Dockerfile in serving/ and is deployed
# from that directory. Two services, two Dockerfiles, deliberately not merged: the scorer's
# dependencies are pandas and xgboost, while this one needs torch, sentence-transformers,
# langchain, langgraph and the Gemini SDK. Folding them together would put a working, deployed
# service at risk of dependencies it has no use for.
#
# Three things are baked in rather than fetched at boot:
#   the MiniLM embedding weights  ~90 MB; a cold start that downloads them looks like a hang.
#   the prebuilt Qdrant directory the embedded store, read-only at runtime. Rebuilding it from
#                                 the LEIE costs ~1 minute of CPU that every instance would
#                                 otherwise repeat.
#   the LEIE csv                  BM25 has no persisted index -- it reads the record text at
#                                 startup to build one, so the source file must ship.
FROM python:3.13-slim

WORKDIR /app

# CPU-only torch, installed FIRST so sentence-transformers finds it already satisfied. The
# default PyPI wheel bundles the CUDA runtime -- gigabytes of NVIDIA libraries for a service
# with no GPU. Cold-start time scales with image size, so this is not only disk.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY serving_agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Baking the weights in is not enough on its own: at startup sentence-transformers still sends a
# HEAD request to huggingface.co to check whether the cached copy is stale. On 2026-07-28 that
# check was rate-limited (HTTP 429, "wait 190s") and cold starts served 429 to callers -- an
# outage caused by someone else's server saying no. Offline mode reads the cache and never asks.
ENV HF_HUB_OFFLINE=1

COPY serving_agent/app.py ./
COPY src/ ./src/
COPY data/raw/ ./data/raw/
COPY data/processed/ ./data/processed/
COPY data/qdrant_store/ ./data/qdrant_store/
COPY serving/model.ubj serving/encoding_maps.json ./serving/

# Embedded Qdrant: a directory, not a server. Cloud Run gives one container and one port, so a
# Qdrant server would be a second service to run, pay for and secure -- for a read-only index
# of 8,482 records. The exclusive file lock it takes is fine here because each Cloud Run
# instance is its own single-process container.
ENV QDRANT_PATH=/app/data/qdrant_store
ENV PORT=8080
EXPOSE 8080

CMD ["python", "app.py"]
