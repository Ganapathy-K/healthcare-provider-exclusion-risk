# Provider Review Queue — Risk Ranking Addition (Pilot Proposal)

**To:** [Manager]
**From:** Ganapathi
**Re:** Optional scoring step on top of our existing SSIS pipeline

---

## What this is (in one line)

A small scoring step that adds a `risk_score` column to a SQL table our team already uses. Analysts can sort by it or ignore it. SSIS unchanged. Workflow unchanged.

---

## The pain point

Our analysts work through provider review queues in roughly first-in-first-out order. Low-risk and high-risk providers get the same upfront attention. By the time the high-risk ones surface, the cost of acting on them has gone up.

If we could give the team a *suggested* priority order, the same review hours produce more catches.

---

## Current pipeline (which works — nothing here changes)

SSIS extracts NPPES (9.4M providers) and OIG LEIE (82K excluded providers) → loads into SQL tables → analysts pull worklist → manual review.

This is the foundation. The proposal sits **on top of** this, not in place of it.

---

## The addition I'm proposing

A scoring step that reads the SQL table SSIS already lands, adds two columns:

| Column | Type | Meaning |
|---|---|---|
| `risk_score` | float (0–1) | Likelihood of future exclusion |
| `risk_tier` | varchar | `high` / `low` cutoff for sorting |

Output writes back to a **new** SQL table. Analysts can join it to their existing query. They can also ignore it entirely — it's an additional column, not a workflow change.

The score comes from patterns found in a 500K NPPES sample cross-referenced against OIG LEIE:

- Providers in **Pain Management / Addiction Medicine specialties** exclude at ~10x the baseline rate
- **Kentucky** providers exclude at 0.55% (2x the national average)
- **Individual providers** (Entity Type 1) exclude 5x more often than organizations (Type 2)

The scoring service runs in Python, lives in a container, executes on a schedule. To the SQL team it is just another upstream job that lands a table.

---

## Pilot — what I'm asking for

- **Duration:** 2–4 weeks
- **Data:** Run the scoring on **the last 3 months of already-reviewed cases** (no production impact, no live queue affected)
- **Output:** A one-page summary covering:
  - **Agreement rate**: how often the top-tier flagged cases were also manually escalated
  - **Miss list**: cases the score missed (so we can see where it fails)
  - **False positives**: cases the score flagged that analysts cleared

---

## What I'm NOT asking for

- ❌ Budget, headcount, or new tools
- ❌ Changes to SSIS pipelines
- ❌ Changes to anyone's workflow during the pilot
- ❌ Adoption commitment — this is purely diagnostic

---

## What success looks like

| Outcome | Next step |
|---|---|
| Top-tier flags overlap meaningfully with manual escalations | Discuss adding the column to one analyst's daily worklist as a pilot-2 |
| No meaningful overlap | I close the loop, share what I learned, we move on |

Either way, the team learns something. There is no path where the two weeks were wasted.

---

## Why I'm bringing this to you first

This sits in your team's domain. I don't want to take it anywhere else without your read. If the framing is off, I'd rather hear it now than after I've spent more weekends on it.

Happy to walk through technical details if useful, or skip them entirely. Your call.
