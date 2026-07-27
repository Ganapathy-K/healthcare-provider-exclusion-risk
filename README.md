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

The runnable code lives in `src/`. The six notebooks below are the record of how it was built and are no longer the source of truth:

1. `01_data_ingestion` – load NPPES + LEIE, join them, build the labelled dataset.
2. `02_eda` – exploring the data and where the exclusion signal actually is.
3. `03_modelling` – training the risk model.
4. `04_rag_pipeline` – embedding the LEIE records and setting up retrieval so you can ask questions over them.
5. `05_langgraph` – a small agent that decides whether a question needs a lookup over the exclusion data or a risk score for a specific provider.
6. `06_serving` – wrapping the model as an API.

`src/` is one module per stage — `config` · `ingest` · `features` · `model` · `vectorstore` · `retrieve` · `generate` · `rbac` · `agent` · `tracing` — alongside the evaluation suite (`golden_set` · `retrieval_eval` · `ablation` · `answer_eval` · `router_eval`) and the two files this README is really about, `baseline.py` and `smoke_test.py`.

`serving/` is the scorer's deployable (model file, FastAPI app, Dockerfile); `serving_agent/` is the agent's.

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

None of this was visible without a measured refusal rate — which is why the next section exists.

## Measuring retrieval, for free

`src/golden_set.py` holds 15 questions built **backwards from the LEIE**: a record was chosen first, then a question written that only those records answer. Every expected NPI is therefore re-derivable from the source file rather than remembered, and `verify()` re-checks them so the set cannot silently rot.

Rare combinations are used on purpose. *"Which providers were excluded in California?"* has 1,071 correct answers, so any three records score a hit and the question measures nothing. *"Which acupuncturists in New York were excluded?"* has exactly two.

Six of the fifteen must be **refused**, and three of those are **traps** — records that look relevant but do not support the question asked. *"How much money did the excluded pharmacies in New York defraud Medicare of?"* retrieves the pharmacies; the LEIE contains no monetary figures at all. Grounding is only proven where it is tempted.

Because the ground truth is an NPI, retrieval can be scored by string comparison: `src/retrieval_eval.py` gives hit rate, MRR and record recall in seconds, with no LLM and no cost.

### What the measurement changed

| config | hit@10 | MRR | record recall |
|---|---|---|---|
| dense only, k=3 (as inherited) | 0.778 | 0.667 | 0.533 |
| **dense + BM25, k=10 (shipped)** | **0.889** | **0.778** | **0.867** |

Two decisions that had been inherited rather than justified:

- **`RETRIEVER_K` 3 → 10.** Hit rate is flat from k=3 to k=10, but *record recall* climbs 0.533 → 0.800. These questions have several correct answers, and returning one of three is a wrong answer that scores as a hit.
- **BM25 added.** Hit rate being flat under increasing k is the signature of a vocabulary problem, not a depth problem — more results can't reach a record the embedding never places nearby.

**The BM25 result reversed once, and the reason is the most transferable thing here.** The first ablation said BM25 made things *worse* (MRR 0.667 → 0.630) and rescued nothing. That was measuring a BM25 that had never worked: LangChain's `BM25Retriever` preprocesses with `text.split()` — no lowercasing, no punctuation handling. This corpus is uppercase (`PROCTOLOGY`) and the questions are lowercase (`proctologist`), so every meaningful token missed. It returned three records for every query the whole time; they were simply useless ones. With a real tokenizer (`retrieve.tokenize`) it improves every metric at every k.

**A broken component returns results, not errors.** That sentence describes four separate bugs in this repo.

### End-to-end: 14/15, zero hallucinations

`src/answer_eval.py` grades the whole pipeline without a judge, crossing answered/refused with should-have: 8 answered correctly with NPIs cited, 6 refused correctly **including all three traps**, and one wrong refusal — the `PROCTOLOGY` question, where retrieval missed the record, so refusing was correct behaviour given the context. That one is a retrieval failure, not a grounding failure: no exact-match retriever bridges "proctologist" to "PROCTOLOGY".

### The router: 17/17, and one myth

`src/router_eval.py` replaced the six hand-written cases in `agent.py` — written by the person who had just written the router, using the examples already in its prompt — with 17 adversarial ones, half sitting on the boundary. It reports intent accuracy and NPI accuracy separately, because a router can pick the right tool every time and still be useless if it drops the identifier.

**Temperature 0 does not give determinism.** Two consecutive runs at `temperature=0` with identical prompts disagreed about which questions they got right. Server-side batching and routing mean identical inputs need not give identical outputs — "set temperature to 0 for reproducibility" is folklore, and a router that answers differently to the same question cannot be measured at all.

So NPI extraction was taken away from the model entirely. It is a ten-digit number in a string; a regex does it perfectly, for free, identically every time. The model keeps only the judgement it is needed for — *what does the user want* — and the deterministic half stopped being a source of variance.

The remaining errors were real prompt gaps, not noise: *"which specialties are the riskiest overall?"* contains risk vocabulary but asks about a population, and the model scores one provider at a time. The prompt now states that test explicitly. 17/17 on intent and NPI, stable across three consecutive runs — with the caveat that two of those cases are now few-shot examples in the prompt, so they encode a rule rather than being independent evidence.

## Role-based access on retrieval

`src/rbac.py` filters **before the model sees anything**, which is the only position where access control means something. The tempting alternative — retrieve everything, then instruct the model not to mention what this user may not see — is a request, not a control: the restricted text still reaches the provider, the traces and the logs, and a model told to withhold something can be talked out of it.

| role | sees |
|---|---|
| `investigator` | everything |
| `analyst` | de-identified — no names, **no NPIs** |
| `auditor` | organisations only; individual practitioners are not searched |
| `public` (default) | nothing |

Unknown roles fall back to `public`, so a typo narrows access rather than widening it. The risk-scoring branch is investigator-only: a score for a supplied NPI confirms that provider is in the dataset, which is itself disclosure.

**Three bugs in my own implementation, all of which looked like success:**

1. **The entity filter was never called.** The auditor got answers — just the wrong ones, including individuals.
2. **Redaction mutated the shared corpus.** BM25 holds the very `Document` objects the retriever returns, so one analyst query stripped the names out of the index *permanently*, and the next investigator query came back pre-redacted. It presented as the redaction working.
3. **The analyst kept the NPI.** Names removed, answers containing none — and an NPI is one public NPPES lookup away from a name, address and phone number. **"No names but keeps the identifier" is not de-identification**, and the output gave no hint, because there genuinely were no names in it.

Two limits, stated rather than left to be found: **a role in the request body is not authentication** — this API has none, so what is demonstrated is the enforcement mechanism, not the identity check in front of it. And **the LEIE is public data**, so nothing here protects a secret; it is the mechanism shown on data that is safe to demonstrate with.

## What's actually deployed

Two Cloud Run services, deliberately separate — the scorer's dependencies are pandas and xgboost, the agent needs torch, langgraph and the Gemini SDK, and folding them together would risk a working deployment to save one deploy.

- **Scorer** — `serving/`, FastAPI + Docker. `src/smoke_test.py` asserts the deployed artefact is the weighted model.
- **Agent** — `serving_agent/`, `POST /ask` and `GET /health`. Routes to the model or to grounded retrieval, enforces roles, refuses when the records don't support an answer, and traces every request.

Qdrant runs **embedded** in the agent's container: a read-only directory baked into the image, selected by `QDRANT_PATH`, with the Docker server still the default locally. Cloud Run gives one container and one port, so a Qdrant server would have meant a second service to run, pay for and secure, for an index of 8,482 records.

## Observability

`src/tracing.py` (Langfuse) records `route → retrieve → generate` as one linked trace. It is instrumented where this project's failures actually happened, not everywhere.

Both RAG bugs above presented identically — as "it refused." One was retrieval missing the record; the other was retrieval working fine and the prompt demanding a citation the context couldn't supply. Telling them apart cost an hour of manual bisection. The trace records the retrieved NPIs on one line and the refusal on the next: empty list means retrieval, records present but refused means the prompt.

Tracing **fails open** — every function no-ops when the keys are unset, so the service answers normally when Langfuse is unreachable. Observability that can take down the thing it observes is a liability, not a safeguard.

## What is still not measured

- **No RAGAS.** Deliberate: the golden set gives ground truth, so `retrieval_eval` and `answer_eval` grade the same pipeline deterministically and for free. RAGAS's strength is judging *without* ground truth, and on this project's sibling its faithfulness metric turned out to be ~30% the app's own mandated disclaimer being scored as unfaithful.
- **No held-out golden set.** Fifteen questions, all used for every decision above, so they measure fit rather than generalisation.
- **The `PROCTOLOGY` miss is unfixed.** It needs stemming or query expansion; neither is measured yet.
- **No authentication, no rate limiting** on either service.

## Running it

```bash
docker start qdrant-healthcare       # local Qdrant server, or set QDRANT_PATH for embedded

python src/ingest.py --save          # rebuild the labelled dataset from NPPES + LEIE
python src/model.py                  # train and score; --save replaces the serving artefacts
python src/baseline.py --check       # assert behaviour has not drifted
python src/vectorstore.py            # collection status; --rebuild reindexes the LEIE

# free, no LLM, seconds each
python src/golden_set.py             # re-verify every expected NPI against the LEIE
python src/retrieval_eval.py 3 5 10  # hit rate, MRR, record recall at each k
python src/ablation.py               # dense vs dense+BM25, at every k
python src/rbac.py                   # what each role sees for the same question

# each costs a handful of Gemini calls
python src/answer_eval.py            # answered/refused vs should-have, over the golden set
python src/router_eval.py            # intent and NPI accuracy on 17 adversarial cases
python src/smoke_test.py             # everything above, as pass/fail (11 checks)
```

The agent service runs either way — `QDRANT_PATH=data/qdrant_store python serving_agent/app.py` uses the embedded index with no container running at all.

`src/model.py --save` writes **both** `serving/model.ubj` and `serving/encoding_maps.json`, and keeps the previous pair as `*_superseded`. They have to travel together: a model trained on training-only category means, served with full-data means, would score every provider on numbers it had never seen — with identical column names and order, so nothing would error.

Dependencies in `serving/requirements.txt` are pinned. They weren't until this work, which meant every Cloud Run build pulled whatever was newest that day, on a live service.

**Two themes run through every bug in this repo.**

*The same thing defined in more than one place, with nothing checking the copies agree.* The feature column order existed in three files, the encoding maps in two, the model path in three. XGBoost scores by column *position* and validates no names, so all three failures are silent — a misaligned model returns a confident number, not an error. `features.check_serving_alignment()` guards the one duplication that remains by necessity.

*A broken component returns results, not errors.* The unweighted model returned scores. BM25 with the wrong tokenizer returned records. The RAG refusing everything returned a valid refusal. In-place redaction returned correctly redacted text while destroying the index. None of them raised anything, and each was found only by measuring what it produced against something known.

## Stack

Python, pandas, XGBoost, MLflow · LangChain, LangGraph, Qdrant, sentence-transformers, BM25, Gemini · FastAPI, Pydantic, Docker, Cloud Run, Secret Manager · Langfuse.

## A note / disclaimer

This is a portfolio and learning project. It's built on public data and it's meant to *prioritise* human review, not to make a final decision about any provider. Any real use would need proper validation and a human in the loop.
