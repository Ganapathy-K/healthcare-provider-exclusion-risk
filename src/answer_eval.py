"""Grades each RAG answer as answered, refused or cited, to catch what eval_gate.py cannot see.

1. Five outcomes, not one score: a wrong refusal looks the same as a right refusal unless counted apart.
2. No LLM judge: every outcome is NPI matching on the answer text, so it is free and repeatable.
3. Trap questions: they pull records that look right but cannot answer them, so a made-up answer shows.

Outcomes: answered_correct · answered_uncited · wrong_refusal · correct_refusal · hallucination.

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
            answer, documents = answer_question(item["question"], role="investigator")
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
