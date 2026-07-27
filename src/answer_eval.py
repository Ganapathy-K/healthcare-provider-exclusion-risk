"""Grade the whole pipeline, not just retrieval: did it answer, refuse, and cite correctly?

`retrieval_eval.py` asks whether the right record came back. This asks what happened next --
whether the model used it, whether it invented anything, and whether it declined when it
should have. Those are different failures and only the second kind is visible here.

Four outcomes per question, all decided WITHOUT an LLM judge:

  answered_correct   should answer; did; and cited at least one expected NPI
  answered_uncited   should answer; did; but named no expected NPI -- an answer nobody can check
  wrong_refusal      should answer; refused. The expensive one, because a refusal looks
                     identical whether it is right or wrong. This is the exact failure that
                     went undetected until the NPI-in-context bug was found by hand.
  correct_refusal    should refuse; did

  hallucination      should refuse; answered anyway. Worst outcome in the set.

THE TRAPS ARE THE POINT. Three golden questions retrieve records that look relevant and do not
support the question asked -- "how much money did the excluded pharmacies defraud Medicare
of?" surfaces the pharmacies, and the LEIE contains no monetary figures whatsoever. A model
that answers those is doing what this project's sibling did when it turned "22.5% of the
policy premium" into "Rs 22,500". Grounding is only proven where it is tempted.

Costs one Gemini call per question. Run:  python src/answer_eval.py
"""

import sys

from generate import REFUSAL_TEXT, answer_question
from golden_set import GOLDEN_SET, is_expected_refusal


def classify(item, answer, documents):
    """Turn one question's result into an outcome label."""
    refused = REFUSAL_TEXT.lower() in answer.lower()

    if is_expected_refusal(item):
        return ("correct_refusal", "") if refused else ("hallucination", answer[:160])

    if refused:
        retrieved = [doc.metadata["NPI"] for doc in documents]
        had_it = any(npi in retrieved for npi in item["expected_npis"])
        return ("wrong_refusal",
                "the correct record WAS retrieved" if had_it
                else "retrieval missed it too")

    cited = [npi for npi in item["expected_npis"] if str(npi) in answer]
    if cited:
        return "answered_correct", f"cited {len(cited)}/{len(item['expected_npis'])}"
    return "answered_uncited", answer[:160]


def run():
    results = []
    for item in GOLDEN_SET:
        try:
            answer, documents = answer_question(item["question"])
        except Exception as error:
            results.append((item, "error", f"{type(error).__name__}: {error}"))
            continue
        outcome, detail = classify(item, answer, documents)
        results.append((item, outcome, detail))
    return results


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    results = run()
    counts = {}
    for _, outcome, _ in results:
        counts[outcome] = counts.get(outcome, 0) + 1

    for item, outcome, detail in results:
        trap = " [TRAP]" if "TRAP" in item.get("note", "") else ""
        flag = {"answered_correct": "OK  ", "correct_refusal": "OK  "}.get(outcome, "FAIL")
        print(f"{flag}  {outcome:<18}{trap} {item['question'][:64]}")
        if detail:
            print(f"        {detail}")

    total = len(results)
    good = counts.get("answered_correct", 0) + counts.get("correct_refusal", 0)
    print(f"\n{good}/{total} correct")
    for outcome in sorted(counts):
        print(f"  {outcome:<20}{counts[outcome]}")

    if counts.get("hallucination"):
        print("\n⚠️  A HALLUCINATION is a question the records cannot support being answered "
              "anyway. On exclusion data naming real providers, this is the failure that "
              "matters more than every metric above it.")
    if counts.get("wrong_refusal"):
        print("\n⚠️  A WRONG REFUSAL is invisible in production: it looks exactly like the "
              "system working. Check whether the context contains everything the prompt "
              "demands the answer cite.")
