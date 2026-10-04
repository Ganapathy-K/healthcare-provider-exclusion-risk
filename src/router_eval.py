"""Intent and NPI accuracy on 17 router questions, half on the boundary, to measure the router."""

import sys

from agent import classify_intent

# (question, expected_intent, expected_npi). "" means no NPI should be extracted.
CASES = [
    # --- clear risk ---
    ("What is the exclusion risk score for provider NPI 1871596098?", "risk", "1871596098"),
    ("Should we be worried about 1679576722?", "risk", "1679576722"),
    ("How risky is NPI 1063076651?", "risk", "1063076651"),
    ("Give me the exclusion probability for 1487083457", "risk", "1487083457"),

    # --- clear records ---
    ("Are there any excluded providers in Texas?", "rag", ""),
    ("How many providers were excluded for fraud?", "rag", ""),
    ("Which acupuncturists in New York were excluded?", "rag", ""),
    ("Who was excluded in 2024?", "rag", ""),

    # --- boundary: a bare identifier, no verb ---
    ("1871596098", "risk", "1871596098"),

    # --- boundary: an NPI present, but the question is about the RECORDS ---
    ("What specialty is NPI 1376524785?", "rag", "1376524785"),
    ("When was 1215272042 excluded?", "rag", "1215272042"),

    # --- boundary: risk vocabulary, but the answer is a property of the data ---
    ("Which specialties are the riskiest overall?", "rag", ""),
    ("What makes a provider high risk?", "rag", ""),

    # --- boundary: risk intent with no identifier to work from ---
    ("How risky is that pharmacy in New York?", "risk", ""),

    # --- malformed identifiers ---
    ("What is the risk score for NPI 123456789?", "risk", ""),   # 9 digits: not an NPI
    ("Risk score for provider ABC1234567?", "risk", ""),

    # --- off-domain: must not be routed to the scorer ---
    ("What is the capital of France?", "rag", ""),
]


def run():
    """One result row per case, to count intent and NPI accuracy."""
    results = []
    for question, expected_intent, expected_npi in CASES:
        try:
            decision = classify_intent(question)
        except Exception as error:
            results.append((question, expected_intent, expected_npi,
                            f"ERROR:{type(error).__name__}", "", False, False))
            continue
        intent_ok = decision["intent"] == expected_intent
        npi_ok = decision["npi"] == expected_npi
        results.append((question, expected_intent, expected_npi,
                        decision["intent"], decision["npi"], intent_ok, npi_ok))
    return results


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    results = run()
    for question, want_intent, want_npi, got_intent, got_npi, intent_ok, npi_ok in results:
        flag = "OK  " if (intent_ok and npi_ok) else "FAIL"
        print(f"{flag}  {question[:58]:<58} {got_intent:<5} {got_npi or '-'}")
        if not intent_ok:
            print(f"        intent: wanted {want_intent}, got {got_intent}")
        if not npi_ok:
            print(f"        npi:    wanted {want_npi or 'none'}, got {got_npi or 'none'}")

    intent_correct = sum(1 for r in results if r[5])
    npi_correct = sum(1 for r in results if r[6])
    total = len(results)

    print(f"\nintent accuracy : {intent_correct}/{total} ({intent_correct / total:.0%})")
    print(f"NPI accuracy    : {npi_correct}/{total} ({npi_correct / total:.0%})")
