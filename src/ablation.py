"""Does a keyword leg earn its place here? Measure before deciding. Free.

The 2026-07-27 retrieval baseline showed hit rate flat at 0.778 from k=3 to k=10, with the
same two questions failing at every k:

    "Was a proctologist ever excluded?"        (PROCTOLOGY, one record in the whole file)
    "Has anyone working in phlebotomy been excluded?"   (PHLEBOTOMY, one record)

More results never helped, which is the signature of a vocabulary problem rather than a depth
problem: the embedding does not place those records near those words, so no amount of k
reaches them. BM25 matches literal tokens and is the obvious candidate.

"Obvious candidate" is not evidence. In the sibling insurance project the same reasoning
produced a hybrid retriever that, when finally ablated, turned out to be carried entirely by
its dense leg -- BM25 there costs MRR and buys 0.029 hit rate. So it gets measured here too,
and the honest outcome may be that it does nothing.

Run:  python src/ablation.py
"""

import sys

from config import RETRIEVER_K
from golden_set import answerable_items
from retrieve import retrieve

ARMS = [
    ("dense only (shipped)", dict(hybrid=False)),
    ("dense + BM25 (hybrid)", dict(hybrid=True)),
]
KS = [3, 5, 10]


def evaluate(top_k, hybrid):
    hits = 0
    reciprocal_ranks = []
    recalled = expected_total = 0
    misses = []

    for item in answerable_items():
        wanted = set(item["expected_npis"])
        documents = retrieve(item["question"], top_k=top_k, hybrid=hybrid)
        retrieved = [doc.metadata["NPI"] for doc in documents]

        found_at = next((rank for rank, npi in enumerate(retrieved, start=1)
                         if npi in wanted), None)
        if found_at:
            hits += 1
            reciprocal_ranks.append(1 / found_at)
        else:
            reciprocal_ranks.append(0.0)
            misses.append(item["question"])

        recalled += len(wanted & set(retrieved))
        expected_total += len(wanted)

    total = len(answerable_items())
    return {
        "hit_rate": hits / total,
        "mrr": sum(reciprocal_ranks) / total,
        "record_recall": recalled / expected_total,
        "misses": misses,
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    results = {}
    header = f"{'configuration':<24}{'k':>4}{'hit@k':>9}{'MRR':>8}{'record recall':>15}"
    print(header)
    print("-" * len(header))

    for label, kwargs in ARMS:
        for k in KS:
            result = evaluate(top_k=k, **kwargs)
            results[(label, k)] = result
            print(f"{label:<24}{k:>4}{result['hit_rate']:>9.3f}{result['mrr']:>8.3f}"
                  f"{result['record_recall']:>15.3f}")
        print()

    # The two questions the baseline could never reach, at every k. If BM25 is worth having
    # here, this is where it shows -- and if it is not, this says so just as plainly.
    print("the two questions dense retrieval never reached:")
    for k in KS:
        dense = results[("dense only (shipped)", k)]["misses"]
        hybrid = results[("dense + BM25 (hybrid)", k)]["misses"]
        rescued = set(dense) - set(hybrid)
        broken = set(hybrid) - set(dense)
        print(f"  k={k:<3} dense missed {len(dense)}, hybrid missed {len(hybrid)}"
              f"{'  RESCUED: ' + '; '.join(sorted(rescued))[:80] if rescued else ''}"
              f"{'  NEWLY BROKEN: ' + '; '.join(sorted(broken))[:80] if broken else ''}")

    print("\nA component earns its place only if switching it ON makes things better -- and "
          "the thing to watch is whether it fixes the failures that motivated it, not "
          "whether the average moves.")
