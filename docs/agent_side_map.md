# Agent side — the complete map

The healthcare project has two halves. This file is the **whole** of the second one.
Built 2026-08-30 from a full file sweep, not from memory. If it isn't here, it isn't agent side.

> ⛔ Don't name this list by its count. It was called "the 17-point list" before the audit;
> the real count is 24 and it will move again. Call it **the agent-side map**.

> ⚠️ **The 1–24 below is this file's own numbering. It is NOT the notebook numbering.**
> Item 1 happens to be notebook **04**. Notebooks 01, 02, 03 and 06 exist and are ML side.

**ML side (not this file):** `ingest.py` · `features.py` · `model.py` · `baseline.py` ·
`serving/app.py` · **notebooks 01 (ingestion) · 02 (EDA) · 03 (modelling) · 06 (serving)**
**Across both:** `config.py` · `smoke_test.py`

---

## Where it was built — 2
| # | Thing | What it is |
|---|---|---|
| 1 | `notebooks/04_rag_pipeline.ipynb` | where retrieval was built |
| 2 | `notebooks/05_langgraph.ipynb` | where the router was built |

## Finding things — 3
| # | Thing | What it is |
|---|---|---|
| 3 | `src/vectorstore.py` | each banned provider becomes one short sentence, indexed in Qdrant (8,482) |
| 4 | `src/retrieve.py` | embeds the question, returns the k nearest records. Dense only |
| 5 | `src/ablation.py` | measures whether adding keyword search earns its place. Hit rate flat 0.778, k=3→10 |

## Answering — 3
| # | Thing | What it is |
|---|---|---|
| 6 | `src/generate.py` | answers from the retrieved records and nothing else |
| 7 | `src/agent.py` | the router — one question, two tools: `query_leie_rag` or `score_provider_risk` |
| 8 | `src/agent_columns.py` | the columns the scorer reads. The seam between the two halves |

## Guardrails — 3
| # | Thing | What it is |
|---|---|---|
| 9 | `src/rbac.py` | filters the index **before** the model sees anything |
| 10 | `src/injection_guard.py` | layered defence, with an honest account of what each layer can do |
| 11 | `src/injection_eval.py` | the attack corpus that proves the guard works. A defence you haven't attacked is a claim |

## Grading — 6
| # | Thing | What it is |
|---|---|---|
| 12 | `src/golden_set.py` | questions with known answers, built backwards from the LEIE itself |
| 13 | `src/retrieval_eval.py` | scores retrieval alone, deterministic, free |
| 14 | `src/answer_eval.py` | grades the whole pipeline: did it answer, refuse, and cite correctly |
| 15 | `src/router_eval.py` | measures the router instead of demoing it (`agent.py`'s "6/6" is a demo) |
| 16 | `src/eval_gate.py` | fails the commit when quality drops below the recorded baseline |
| 17 | `eval_baseline.json` · `docs/baseline.json` | the recorded baseline the gate compares against |

## Knowing what happened — 2
| # | Thing | What it is |
|---|---|---|
| 18 | `src/tracing.py` | Langfuse — every answer replayable after the fact |
| 19 | `src/trace_metrics.py` | real cost and real latency, from real traces |

## Shipping — 3
| # | Thing | What it is |
|---|---|---|
| 20 | `src/mcp_server.py` | exposes the two tools over MCP so other programs can call them |
| 21 | `serving_agent/app.py` | FastAPI, one `/ask` endpoint, two tools behind it |
| 22 | `Dockerfile` | how it ships |

## The models — 2
| # | Thing | What it is |
|---|---|---|
| 23 | Gemini 2.5 Flash | the LLM that routes and writes answers |
| 24 | the embedding model + chunking | how a record becomes a searchable vector |

---

## The shape of it — L1 / L2 / L3

**L1** = the agent side · **L2** = the 8 groups below · **L3** = the 24 items above.
Read as a `value_counts()` — biggest group first:

| L2 group | items |
|---|---|
| Grading | **6** |
| Finding things | 3 |
| Answering | 3 |
| Guardrails | 3 |
| Shipping | 3 |
| Where it was built | 2 |
| Knowing what happened | 2 |
| The models | 2 |
| **Total** | **24** |

6+3+3+3+3+2+2+2 = 24 ✓

**What the shape says:** a quarter of the agent side is measurement. Most people build 3-9
and stop. Grading and guardrails together are 9 of 24.
