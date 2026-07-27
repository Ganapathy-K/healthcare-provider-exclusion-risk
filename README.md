# Healthcare Provider Exclusion Risk

This is a portfolio project I built to predict which healthcare providers are at risk of being excluded (terminated) by the US OIG, and to let someone ask questions about the exclusion data in plain English.

The problem it's aimed at: a health plan that pays a claim to a provider who has been excluded by the OIG can end up on the hook for that money. Today a lot of that checking is reactive. The idea here is to score providers up front so the risky ones get looked at first, instead of everyone getting reviewed in the order they happen to come in.

## Data

Two public US government datasets, both free to download:

- **NPPES** – the national registry of every provider and their NPI number.
- **OIG LEIE** – the List of Excluded Individuals/Entities, i.e. providers the OIG has already excluded.

I worked with a ~500K sample of NPPES and cross-referenced it against the full LEIE to build a labelled dataset (excluded vs not). Everything shown in the notebook outputs comes from these public files, there's no private or client data anywhere in here.

The raw and processed data files are not committed to the repo (they're large and you can pull them from the source), so `data/` is gitignored.

## What's in here

The work is split across six notebooks that run in order:

1. `01_data_ingestion` – load NPPES + LEIE, join them, build the labelled dataset.
2. `02_eda` – exploring the data and where the exclusion signal actually is.
3. `03_modelling` – training the risk model.
4. `04_rag_pipeline` – embedding the LEIE records and setting up retrieval so you can ask questions over them.
5. `05_langgraph` – a small agent that decides whether a question needs a lookup over the exclusion data or a risk score for a specific provider.
6. `06_serving` – wrapping the model as an API.

`serving/` holds the deployable version of the scorer (the model file, a FastAPI app, and a Dockerfile).

The notebooks are the record of how it was built. The runnable code now lives in `src/`, one module per stage — `config` · `ingest` · `features` · `model` · `vectorstore` · `retrieve` · `generate` · `agent` — plus `baseline.py` and `smoke_test.py`, which are the two files this README is really about.

## Freezing behaviour before refactoring, and what it found

Before moving anything out of the notebooks I recorded what the pipeline currently *did* — dataset shape, label balance, feature order, encoding-map sizes, and the model's test metrics — into `docs/baseline.json`. `python src/baseline.py --check` re-runs it and exits non-zero on any drift.

The point of a characterization test is that it asserts behaviour is **unchanged**, including behaviour that is wrong. Without one, the only check available after a refactor is "the output still looks about right", which catches nothing subtle. It found three things in its first hour, all of which had been live:

**1. The deployed model was not the model this README described.**

| | what the notebook selected | what `serving/model.ubj` actually was |
|---|---|---|
| class weighting | `scale_pos_weight=422` | **none** |
| learning rate | 0.1 | **0.3** (the XGBoost default) |

An earlier, unweighted run had been copied out of MLflow in notebook 06 and shipped. On a problem with one excluded provider per 422, a model never told the classes are imbalanced learns that answering "not excluded" is almost always right. Measured on the held-out split, it caught **42 of 237** excluded providers.

**2. The target encodings leaked.** The four target-encoded columns were fitted on the whole dataset before the train/test split, so each test row's own label contributed to the feature it was later scored on. `features.fit_encoding_maps()` now fits on the training rows only.

**3. The agent scored providers with a third, different model.** Notebook 05 loaded XGBoost straight from an MLflow artifact path rather than the deployed file — so the agent was answering with a model nobody had validated or deployed.

## The model

XGBoost, with `scale_pos_weight=422` — the ratio of negatives to positives, which is how the model is told that excluded providers are rare. Accuracy is meaningless here (99.76% is the score for predicting "not excluded" every time); recall is what the system exists to produce.

| | before | after weighting | after removing the leak |
|---|---|---|---|
| **excluded caught (of 237)** | **42** | 165 | **157** |
| recall | 0.177 | 0.696 | **0.662** |
| ROC-AUC | 0.765 | 0.800 | **0.754** |
| average precision | 0.0055 | 0.0102 | **0.0084** |
| providers flagged | 9,027 | 25,883 | **26,149** |

The third column is the one deployed. It is lower than the second on every metric, and that is the point: the second column was partly measuring its own answer sheet. The leak was worth about 8 of the 157, so most of the gain is real — and ROC-AUC had been overstated by roughly 6%.

**The cost, stated plainly: it flags 26,149 providers for review instead of 9,027.** Someone has to work that queue. The trade is right for this problem — a missed exclusion is an unrecoverable claim payment, a false flag is one review — but the threshold that sets it is a capacity decision belonging to whoever staffs the queue, not a statistical one.

A few things from the EDA that stuck around as real signal:

- Providers in **Pain Management and Addiction Medicine** get excluded at roughly 10x the baseline rate.
- **Individual providers** (NPPES Entity Type 1) are excluded about 5x more often than organisations (Type 2).
- Exclusion rates vary a lot by state (Kentucky came out around 2x the national average in the sample).

**A limitation worth stating before someone else finds it:** only 8,306 of the 82,749 LEIE records carry a valid NPI, and 1,183 of those fall inside the 500K NPPES sample. So the positive class is not "excluded providers" — it is the ~10% of exclusions with an NPI recorded, intersected with the sample. And the label itself is "already caught by the OIG", which is not the same as "committing fraud": anyone never investigated is labelled 0 whatever they have done.

## The RAG + agent part

On top of the scoring model I added a retrieval layer over the LEIE records. The exclusion records get embedded with a sentence-transformers model and stored in Qdrant, and a question gets answered from the retrieved records using Gemini, grounded only in what was retrieved rather than the model's own memory.

The agent (built with LangGraph) sits in front of that. It reads the question and routes it: if you're asking for a provider's risk score it pulls the NPI and runs the model, if you're asking about the exclusion data generally it goes to retrieval. The Qdrant store is rebuilt from the data by `src/vectorstore.py`, so it isn't committed either.

### The refusal bug, which is the most useful thing in this repo

A grounded system that refuses when the documents don't support an answer is working correctly. That is also what makes a *wrong* refusal so expensive: it is indistinguishable from a right one.

This pipeline refused "are there any excluded pharmacies in New York?" every single time — with three New York pharmacies sitting in the retrieved context.

The cause was not the retriever, the temperature, or the wording of the guardrail (I blamed all three, in that order, and was wrong each time). **The indexed sentence never contained the NPI** — it lives only in the document metadata — while the instruction demanded an NPI for every provider named. The model could not satisfy the instruction from the context it was given, so it took the refusal as an exit.

The tell was that *removing the refusal option made it answer perfectly*. Two lessons came out of it:

- **Never editorialise about stakes in a prompt.** An earlier version of the instruction said "these records concern real, named people, and describing someone as excluded wrongly is a serious error." It reads like responsible prompt engineering. Telling a model that being wrong is dangerous does not make it careful, it makes it decline. Constrain what it may *say*; don't tell it how much trouble it's in.
- **An impossible instruction plus an escape hatch equals the escape hatch** — silently, and looking exactly like correct behaviour.

None of this would have been visible with a measured refusal rate in place. There isn't one yet, which is the honest state of this half of the project.

## What's actually deployed vs a prototype

I want to be straight about this because it matters:

- The **XGBoost scoring service is deployed** to Google Cloud Run (FastAPI + Docker), and `src/smoke_test.py` asserts the deployed artefact is the weighted model.
- The **RAG pipeline and the LangGraph agent run locally** against a Qdrant container. They aren't deployed as a service.

## What is not measured (the honest gap)

The model half has a frozen baseline and a smoke test. The retrieval half has neither, and everything below follows from that:

- **No golden set**, so no hit rate, no MRR, and no refusal rate — which is exactly what hid the bug above.
- **`RETRIEVER_K = 3` has no evidence behind it.** Three records is very few for "which providers were excluded in Texas?", where the honest answer is a list.
- **Dense retrieval only, no keyword leg.** This corpus is full of exact tokens — NPIs, exclusion codes like `1128b8`, state abbreviations — which is precisely where embeddings are weakest.
- **Router accuracy is 6 correct out of 6 hand-written cases.** That is a demo, not a measurement.

## Still in progress

- A RAGAS evaluation harness with a golden question set, to actually measure the RAG answers instead of eyeballing them.
- Role-based access on retrieval (RBAC), so different users only see what they're allowed to.

## Running it

```bash
docker start qdrant-healthcare       # the RAG half needs Qdrant on localhost:6333

python src/ingest.py --save          # rebuild the labelled dataset from NPPES + LEIE
python src/model.py                  # train and score; --save replaces the serving artefacts
python src/baseline.py --check       # assert behaviour has not drifted
python src/vectorstore.py            # collection status; --rebuild reindexes the LEIE
python src/agent.py                  # route six questions through the agent
python src/smoke_test.py             # all of the above, as pass/fail
```

`src/model.py --save` writes **both** `serving/model.ubj` and `serving/encoding_maps.json`, and keeps the previous pair as `*_superseded`. They have to travel together: a model trained on training-only category means, served with full-data means, would score every provider on numbers it had never seen — with identical column names and order, so nothing would error.

Dependencies in `serving/requirements.txt` are pinned. They weren't until this work, which meant every Cloud Run build pulled whatever was newest that day, on a live service.

**One theme runs through every bug above: the same thing defined in more than one place, with nothing checking the copies agree.** The feature column order existed in three files, the encoding maps in two, the model path in three. XGBoost scores by column *position* and validates no names, so all three failure modes are silent — a misaligned model returns a confident number, not an error. `features.check_serving_alignment()` guards the one duplication that remains by necessity.

## Stack

Python, pandas, XGBoost, MLflow, LangChain, LangGraph, Qdrant, sentence-transformers, Gemini, FastAPI, Docker, Cloud Run.

## A note / disclaimer

This is a portfolio and learning project. It's built on public data and it's meant to *prioritise* human review, not to make a final decision about any provider. Any real use would need proper validation and a human in the loop.
