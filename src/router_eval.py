"""Measure the router, not just demonstrate it.

`agent.py` runs six hand-written questions and reports "6/6". That is a demo: the cases were
written by the person who wrote the router, immediately after writing it, so they test the
examples already in its prompt. A router that only handles what its author thought of is not
measured -- it is remembered.

This set is built to be adversarial instead. Roughly half the cases are chosen to sit near the
boundary, because the boundary is the only place a router can fail:

  bare numbers            "1871596098" with no verb at all
  an NPI in a RAG question    "what specialty is 1871596098?" -- an NPI is present, and the
                          question is still about the records, not the model. Extracting a
                          number is not the same as wanting a score.
  risk words in a RAG question  "which specialties are riskiest?" -- "risk" appears, but the
                          answer is a property of the data, not a prediction for a provider.
  no NPI in a risk question   "how risky is that pharmacy in New York?" -- correct routing is
                          `risk`, and the tool must then fail politely for want of an NPI.
  malformed identifiers   9 digits, letters, a phone number

Two numbers come out, and they are not the same thing:

  intent accuracy   did it pick the right tool
  NPI accuracy      did it extract the right identifier (or correctly extract none)

A router can score perfectly on intent and still be useless if it drops the NPI, because the
tool then answers "no NPI was supplied" to a question that contained one.

Costs one Gemini call per case. Run:  python src/router_eval.py
"""

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
    # These two originally expected NO npi, and the router "failed" them by extracting one.
    # On review the router is right and the expectation was wrong: the intent is `rag`, the
    # npi field is unused on that branch, and refusing to extract a number that is plainly
    # present would be the odder behaviour. Expectations changed to match, and recorded here
    # because silently editing a test until it passes is how a suite stops meaning anything.
    ("What specialty is NPI 1376524785?", "rag", "1376524785"),
    ("When was 1215272042 excluded?", "rag", "1215272042"),

    # --- boundary: risk vocabulary, but the answer is a property of the data ---
    ("Which specialties are the riskiest overall?", "rag", ""),
    ("What makes a provider high risk?", "rag", ""),

    # --- boundary: risk intent with no identifier to work from ---
    ("How risky is that pharmacy in New York?", "risk", ""),

    # --- malformed identifiers ---
    # ⚠️ AN OPEN DESIGN QUESTION, not a settled expectation. 123456789 is NINE digits, and the
    # router declines to extract it. That is defensible -- an NPI is ten digits, so this is not
    # one. But the consequence is that the tool answers "no NPI was supplied" to a question
    # that visibly supplied something, when "123456789 is not a valid NPI" would be the more
    # useful reply. The expectation follows the router's current behaviour; whether that
    # behaviour is right belongs in the prompt, not in this file.
    ("What is the risk score for NPI 123456789?", "risk", ""),   # 9 digits: not an NPI
    ("Risk score for provider ABC1234567?", "risk", ""),

    # --- off-domain: must not be routed to the scorer ---
    ("What is the capital of France?", "rag", ""),
]


def run():
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
    both = sum(1 for r in results if r[5] and r[6])
    total = len(results)

    print(f"\nintent accuracy : {intent_correct}/{total} ({intent_correct / total:.0%})")
    print(f"NPI accuracy    : {npi_correct}/{total} ({npi_correct / total:.0%})")
    print(f"both correct    : {both}/{total} ({both / total:.0%})")
    print("\nA mis-routed question is not a wrong answer -- it is a CONFIDENT answer from the "
          "wrong half of the system, which is harder to notice.")
