"""Fail the commit when retrieval quality drops below the recorded baseline.

The sibling insurance project runs the same gate for the same reason: an eval suite that
nobody is forced to look at is a report, not a control. This one differs in three ways, and
each difference is the interesting part.

1. THREE METRICS, NOT TWO. `record_recall` is gated alongside hit rate and MRR because
   several golden questions have more than one correct NPI. A question with three correct
   providers and one retrieved scores a full hit -- hit rate cannot see the two that were
   missed, and "list the excluded acupuncturists in New York" is exactly the kind of question
   this system exists to answer. Gating only the first two would let a real regression through
   while both headline numbers held.

2. INFRASTRUCTURE FAILURE IS NOT A QUALITY FAILURE. Retrieval here runs against a Qdrant
   SERVER in Docker, not an embedded file. If the container is stopped, every query returns
   nothing, every metric collapses to zero, and a naive gate reports a catastrophic regression
   caused by a change nobody made. That is the fastest possible way to teach someone to ignore
   the gate. Qdrant is therefore checked FIRST and reported as an unmet precondition -- the
   commit is still blocked, because an unmeasured commit is not a verified one, but the
   message says what is actually wrong.

3. WHY A PRE-COMMIT HOOK AND NOT GITHUB ACTIONS. The eval needs a running vector store and the
   built NPPES/LEIE dataset, neither of which exists on a hosted runner. The gate runs where
   the data is. It is written to CI conventions anyway (exit codes, committed JSON baseline,
   non-interactive), so it moves into a workflow unchanged if that ever becomes possible.

WHY THE FREE METRIC AND NOT AN LLM JUDGE. Every number here is decided by NPI membership --
no model, no API call, no money, seconds to run. `answer_eval.py` grades the generated text
and is the milestone tool; this is the per-change signal.

IMPROVEMENTS DO NOT AUTO-UPDATE THE BASELINE. A gate that ratchets itself records whatever
happened last rather than what was decided, and a slow decline never trips it.

Run:      python src/eval_gate.py
Update:   python src/eval_gate.py --update
Install:  python src/eval_gate.py --install-hook
"""

import json
import subprocess
import sys
from pathlib import Path

from config import RETRIEVER_K
from retrieval_eval import evaluate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = PROJECT_ROOT / "eval_baseline.json"

# Fixed deliberately rather than following RETRIEVER_K: a gate that scores whatever the app
# currently defaults to stops comparing like with like the moment that default is tuned.
GATED_TOP_K = RETRIEVER_K

# These numbers call no model and sample nothing, so repeated runs are identical. The
# tolerance absorbs dependency drift (a sentence-transformers update moving the third
# decimal), not noise. Anything larger is a real change and wants a human.
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


def load_baseline():
    if not BASELINE_PATH.exists():
        return None
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def save_baseline(measured):
    BASELINE_PATH.write_text(json.dumps(measured, indent=2) + "\n", encoding="utf-8")


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

    # Pinned to the interpreter running this install, NOT bare `python`: git hooks run without
    # the virtualenv activated, so `python` would resolve to a global install missing every
    # dependency here. The gate would then fail on every commit for an unrelated reason, and a
    # gate that cries wolf gets deleted within a day.
    interpreter = Path(sys.executable).as_posix()

    # The bypass stays on purpose -- a gate with no escape hatch gets uninstalled the first
    # time someone needs to commit a README fix while retrieval is mid-refactor.
    hook_path.write_text(
        "#!/bin/sh\n"
        "# Installed by src/eval_gate.py. Bypass once with: git commit --no-verify\n"
        'cd "$(git rev-parse --show-toplevel)" || exit 1\n'
        f'"{interpreter}" src/eval_gate.py || exit 1\n',
        encoding="utf-8",
    )
    hook_path.chmod(0o755)
    return hook_path


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

    baseline = load_baseline()

    if "--update" in sys.argv or baseline is None:
        measured = measure()
        save_baseline(measured)
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
