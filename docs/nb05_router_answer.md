# nb5 — the router. The answer to give, verbatim.

Pinned 2026-08-31 at his request: *"put a pin on that proper answer for nb 5, i will keep asking it."*
Senior-engineer register. Read this file when he asks; do not rebuild it from memory.

---

## The 30-second version

A LangGraph `StateGraph`: two nodes, `router` → `tool` → END.

**The design call worth defending.** The router is an LLM, not keyword rules. *"Tell me about NPI X"*
and *"who else did what X did"* share every token that matters and need different tools — words cannot
separate them. But the NPI itself is pulled by regex, not the model. **Deterministic where it can be,
LLM only where it must be.**

**Two bugs the extraction caught.** The notebook loaded XGBoost from an MLflow artifact path — the old
unweighted model — so the agent was answering with something never validated or deployed. It now loads
the same `serving/model.ubj` the live API serves, so agent and endpoint cannot disagree. Separately the
feature encoding existed in three copies; now one, in `features.encode_provider_record`.

**What it measures at.** `router_eval.py`, 2026-09-03: **17/17 intent, 17/17 NPI extraction**, on a set
built adversarially -- roughly half the cases sit on the boundary (a bare NPI with no verb; an NPI
inside a records question; the word "risk" inside a records question; malformed identifiers; an
off-topic question). The honest limit: 17 cases, written by the author. The six in `agent.py` are a
demo, not the measurement.
That is a demo, not evidence — and saying so is worth more in an interview than the six.

---

## The two tools

| Tool | Question it answers | What it does |
|---|---|---|
| `score_provider_risk` | *"what is the risk score for NPI 1871596098?"* | validates the NPI → looks the row up in the slim parquet → `features.encode_provider_record` → XGBoost → probability + risk tier |
| `query_leie_rag` | *"are there any excluded pharmacies in NY?"* | retrieval over the LEIE sentences → Gemini writes the answer → refuses if the records don't support it |

## The plain-words ladder (use the rung he asks for, not the one below it)

| Rung | Answer |
|---|---|
| One line | It decides which of two tools answers the question. |
| Kid / uncle | A flowchart you can actually run. |
| Three beats | 1. A question comes in. 2. The router decides: banned list, or one specific provider? 3. Banned list → RAG. Specific provider → fetch them, run the score. |
| Senior | the 30-second version above |

⚠️ He asked for ONE rung at a time. Never give two.

## ⭐ THE MAP — his own ask, 2026-08-31: *"trying to map all these things so i could reproduce them myself"*

| His anchor | Technical name | What it does |
|---|---|---|
| The recipe card | LangGraph `StateGraph` | the flowchart it all runs on |
| The receptionist | the router — an LLM | decides which corridor |
| Corridor 1 | `query_leie_rag` | search the banned list |
| Corridor 2 | `score_provider_risk` | fetch one provider, score them |
| The notepad | state | what each box writes for the next |

**Confusions he hit, resolved — do not re-blur them:**
- **receptionist = router = LLM.** Same thing, three names. Confirmed by him.
- **LangGraph is NOT the receptionist.** It is the CARD. The card says "receptionist first, then corridor."
- **State is not a log.** *A log is read by you, afterwards. State is read by the next step, during.*
- **Keyword rules = a sign on the wall.** An LLM router = someone who actually listens. Same as Ctrl+F vs meaning.

**The two beats he must be able to reproduce (not the paragraph):**
1. **LLM for meaning, pattern-matching for pattern.**
2. **Proof:** *"tell me about NPI X"* → `score_provider_risk`; *"who else did what X did"* → `query_leie_rag`.
   Same words, different tools.

## Files
`src/agent.py` (router + both tools) · `notebooks/05_langgraph.ipynb` (where it was built) ·
`serving_agent/app.py` (the `/ask` endpoint that fronts it)
