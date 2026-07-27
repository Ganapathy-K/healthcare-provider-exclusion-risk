"""Score retrieval alone, deterministically, for free.

Ported from the insurance project, where it replaced most paid eval runs. RAGAS-style
LLM-judged evaluation costs minutes and money per run, which makes it a milestone tool -- but
most changes worth making are retrieval changes, and retrieval has a ground truth already
sitting in the golden set: the NPI each question's answer belongs to.

  hit rate @ k   fraction of questions whose correct record appears in the top k
  MRR            1/rank of the first correct record, averaged. Rewards ranking it FIRST,
                 not merely including it.

No model, no API call, runs in seconds. This is the per-change signal.

Expected refusals are excluded from both numbers and reported separately. They have no
correct record to retrieve, so scoring them here would punish the system for questions it is
supposed to decline -- the same measurement defect the insurance project had to fix after
paying for the run that revealed it.

Run:  python src/retrieval_eval.py [k ...]
"""

import sys

from config import RETRIEVER_K
from golden_set import answerable_items, refusal_items
from retrieve import retrieve


def evaluate(top_k=RETRIEVER_K):
    """Hit rate and MRR over the answerable golden questions."""
    hits = 0
    reciprocal_ranks = []
    misses = []

    for item in answerable_items():
        wanted = set(item["expected_npis"])
        documents = retrieve(item["question"], top_k=top_k)
        retrieved = [doc.metadata["NPI"] for doc in documents]

        found_at = next((rank for rank, npi in enumerate(retrieved, start=1)
                         if npi in wanted), None)
        if found_at:
            hits += 1
            reciprocal_ranks.append(1 / found_at)
        else:
            reciprocal_ranks.append(0.0)
            misses.append(item["question"])

    # How many of ALL the correct records were surfaced, not just the first. A question with
    # three correct answers and one retrieved scores a hit, which flatters a system asked to
    # "list the providers" -- this is the number that notices.
    recalled, expected_total = 0, 0
    for item in answerable_items():
        wanted = set(item["expected_npis"])
        retrieved = {doc.metadata["NPI"] for doc in retrieve(item["question"], top_k=top_k)}
        recalled += len(wanted & retrieved)
        expected_total += len(wanted)

    total = len(answerable_items())
    return {
        "k": top_k,
        "questions": total,
        "hit_rate": hits / total if total else 0.0,
        "mrr": sum(reciprocal_ranks) / total if total else 0.0,
        "record_recall": recalled / expected_total if expected_total else 0.0,
        "records_found": recalled,
        "records_expected": expected_total,
        "misses": misses,
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ks = [int(argument) for argument in sys.argv[1:]] or [3, 5, 8, 10]

    header = (f"{'k':>4}{'hit@k':>9}{'MRR':>8}{'record recall':>16}{'misses':>9}")
    print(header)
    print("-" * len(header))

    results = []
    for k in ks:
        result = evaluate(top_k=k)
        results.append(result)
        print(f"{k:>4}{result['hit_rate']:>9.3f}{result['mrr']:>8.3f}"
              f"{result['record_recall']:>10.3f} "
              f"({result['records_found']}/{result['records_expected']})"
              f"{len(result['misses']):>7}")

    default = next((r for r in results if r["k"] == RETRIEVER_K), None)
    if default and default["misses"]:
        print(f"\nmissed at the shipped k={RETRIEVER_K}:")
        for question in default["misses"]:
            print(f"  - {question}")

    print(f"\n{len(refusal_items())} expected-refusal questions are excluded from these "
          "numbers: they have no correct record to retrieve, and scoring them here would "
          "penalise the system for refusing correctly.")
