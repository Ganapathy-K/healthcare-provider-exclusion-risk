"""Blocks a commit when retrieval drops below the saved numbers, to catch a worse retriever early.

1. Three numbers, not two: record recall catches a question with 3 right NPIs that found only 1.
2. Qdrant is checked first: a stopped Docker container says "Qdrant down", not "retrieval broke".
3. Pre-commit hook, not GitHub Actions: the data and Qdrant live only on this machine.
4. No LLM judge: every number is NPI matching, so it is free and takes seconds.
5. Better numbers never overwrite the saved ones: only --update does, so a slow slide still fails.

Run:      python src/eval_gate.py
Report:   python src/eval_gate.py --report [k ...]   # hit rate, MRR, record recall at each k
Update:   python src/eval_gate.py --update
Install:  python src/eval_gate.py --install-hook
"""

import json
import subprocess
import sys
from pathlib import Path

from config import RETRIEVER_K
from golden_set import answerable_items, refusal_items
from retrieve import retrieve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# One recorded-numbers file for the whole repo: this gate owns its "retrieval" block.
BASELINE_PATH = PROJECT_ROOT / "docs" / "baseline.json"
BASELINE_KEY = "retrieval"

# The same k the app uses (config.RETRIEVER_K = 10).
GATED_TOP_K = RETRIEVER_K

# Runs repeat exactly; 0.01 only absorbs a package update moving the third decimal.
TOLERANCE = 0.01

GATED_METRICS = ("hit_rate", "mrr", "record_recall")


def qdrant_is_reachable():
    """Whether the vector store is actually up, checked before anything is measured.

    Returns (reachable, detail) so the failure message can name the real problem instead of
    reporting a 100% quality regression when a container is simply stopped.
    """
    try:
        from vectorstore import get_client
        collections = get_client().get_collections().collections
        return True, f"{len(collections)} collection(s)"
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def evaluate(top_k=RETRIEVER_K):
    """Gives hit rate, MRR and record recall over the 20 answerable golden questions, one search each."""
    hits = 0
    reciprocal_ranks = []
    misses = []
    recalled, expected_total = 0, 0

    for item in answerable_items():
        wanted = set(item["expected_npis"])
        retrieved = [doc.metadata["NPI"] for doc in retrieve(item["question"], top_k=top_k)]

        found_at = next((rank for rank, npi in enumerate(retrieved, start=1)
                         if npi in wanted), None)
        if found_at:
            hits += 1
            reciprocal_ranks.append(1 / found_at)
        else:
            reciprocal_ranks.append(0.0)
            misses.append(item["question"])

        # Record recall: how many of ALL the right NPIs came back, not just the first.
        recalled += len(wanted & set(retrieved))
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


def measure():
    """Score the gated configuration, keeping only what the baseline compares."""
    result = evaluate(top_k=GATED_TOP_K)
    return {
        "k": GATED_TOP_K,
        "questions": result["questions"],
        "hit_rate": round(result["hit_rate"], 4),
        "mrr": round(result["mrr"], 4),
        "record_recall": round(result["record_recall"], 4),
        "records_expected": result["records_expected"],
    }


def compare(measured, baseline):
    """(regressions, improvements) as lists of (metric, before, now, delta)."""
    regressions, improvements = [], []
    for metric in GATED_METRICS:
        delta = measured[metric] - baseline[metric]
        row = (metric, baseline[metric], measured[metric], delta)
        if delta < -TOLERANCE:
            regressions.append(row)
        elif delta > TOLERANCE:
            improvements.append(row)
    return regressions, improvements


def format_row(row):
    metric, before, now, delta = row
    return f"  {metric:<14} {before:.4f} -> {now:.4f}  ({delta:+.4f})"


def install_hook():
    """Write .git/hooks/pre-commit so the gate runs without anyone remembering to run it."""
    git_dir = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    hook_path = (PROJECT_ROOT / git_dir / "hooks" / "pre-commit").resolve()
    hook_path.parent.mkdir(parents=True, exist_ok=True)

    # This Python, not bare `python`: a git hook runs outside the virtualenv.
    interpreter = Path(sys.executable).as_posix()

    # --no-verify stays, so a README fix can still go in while retrieval is broken.
    hook_path.write_text(
        "#!/bin/sh\n"
        "# Installed by src/eval_gate.py. Bypass once with: git commit --no-verify\n"
        'cd "$(git rev-parse --show-toplevel)" || exit 1\n'
        f'"{interpreter}" src/eval_gate.py || exit 1\n',
        encoding="utf-8",
    )
    hook_path.chmod(0o755)
    return hook_path


def report(ks):
    """Hit rate, MRR and record recall at each k, for choosing k -- not for gating."""
    header = f"{'k':>4}{'hit@k':>9}{'MRR':>8}{'record recall':>16}{'misses':>9}"
    print(header)
    print("-" * len(header))
    for k in ks:
        result = evaluate(top_k=k)
        print(f"{k:>4}{result['hit_rate']:>9.3f}{result['mrr']:>8.3f}"
              f"{result['record_recall']:>10.3f} "
              f"({result['records_found']}/{result['records_expected']})"
              f"{len(result['misses']):>7}")
    print(f"\n{len(refusal_items())} expected-refusal questions are excluded: they have no "
          "correct record to retrieve.")


def main():
    if "--install-hook" in sys.argv:
        print(f"installed: {install_hook()}")
        return 0

    reachable, detail = qdrant_is_reachable()
    if not reachable:
        print("GATE BLOCKED: Qdrant is not reachable, so retrieval cannot be measured.")
        print(f"  {detail}")
        print("\n  This is NOT a quality regression -- nothing was scored at all.")
        print("  Start it with:  docker start qdrant-healthcare")
        print("  To commit without measuring: git commit --no-verify")
        return 1

    if "--report" in sys.argv:
        report([int(argument) for argument in sys.argv[2:]] or [3, 5, 8, 10])
        return 0

    recorded = json.loads(BASELINE_PATH.read_text(encoding="utf-8")) if BASELINE_PATH.exists() else {}
    baseline = recorded.get(BASELINE_KEY)

    if "--update" in sys.argv or baseline is None:
        measured = measure()
        recorded[BASELINE_KEY] = measured
        BASELINE_PATH.write_text(json.dumps(recorded, indent=2), encoding="utf-8")
        action = "recorded" if baseline is None else "updated"
        print(f"baseline {action}: hit_rate={measured['hit_rate']:.4f} "
              f"MRR={measured['mrr']:.4f} record_recall={measured['record_recall']:.4f}  "
              f"(n={measured['questions']}, k={GATED_TOP_K})")
        print(f"  {BASELINE_PATH.name} -- commit this file with the change that earned it.")
        return 0

    measured = measure()

    if measured["questions"] != baseline["questions"]:
        print(f"GATE FAILED: the golden set changed size "
              f"({baseline['questions']} -> {measured['questions']} questions).")
        print("  Scores over different question sets are not comparable. If the new questions"
              " are intended,\n  rerun with --update to rebaseline against them.")
        return 1

    regressions, improvements = compare(measured, baseline)

    if regressions:
        print("GATE FAILED: retrieval quality regressed.")
        for row in regressions:
            print(format_row(row))
        print(f"\n  tolerance {TOLERANCE:+.2f}, baseline in {BASELINE_PATH.name}")
        print("  Fix it, or rebaseline deliberately with: python src/eval_gate.py --update")
        print("  To commit anyway just this once: git commit --no-verify")
        return 1

    if improvements:
        print("GATE PASSED, and retrieval improved:")
        for row in improvements:
            print(format_row(row))
        print("\n  The baseline does NOT move on its own. To keep this as the new floor:")
        print("    python src/eval_gate.py --update")
        return 0

    print(f"GATE PASSED: hit_rate={measured['hit_rate']:.4f} MRR={measured['mrr']:.4f} "
          f"record_recall={measured['record_recall']:.4f}  "
          f"(n={measured['questions']}, k={GATED_TOP_K})")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
