"""The answer key: 29 questions with the right NPIs written down before any run, so retrieval is measured.

1. Built backwards: a record was picked first, then a question only it answers.
2. NPI as the answer, not text: right-or-wrong needs no LLM judge, so hit rate and MRR are free.
3. Rare specialty + state: 2 right NPIs, not 1,071, so a lucky hit cannot score.
4. Three kinds: answerable · expected refusal · trap (records look right but cannot answer).

Counts match data/raw/oig_leie_202602.csv; verify() fails if a newer LEIE file changes them.
"""

import sys

GOLDEN_SET = [
    # --- answerable: rare specialty + state, so the correct set is small and exact ---
    {
        "question": "Which acupuncturists in New York were excluded?",
        "expected_npis": [1922241058, 1073880217],
        "note": "Only two ACUPUNCTURIST records in NY.",
    },
    {
        "question": "Were any ambulance companies in Kentucky excluded?",
        "expected_npis": [1437418506, 1558356444],
        "note": "AMBULANCE COMPANY / KY -- Arrow-Med and Lafferty Enterprises.",
    },
    {
        "question": "Which adult day care facilities in Missouri were excluded?",
        "expected_npis": [1821139635, 1952915910],
        "note": "ADULT DAY CARE FACIL / MO.",
    },
    {
        "question": "Were any allergists or immunologists in Maryland excluded?",
        "expected_npis": [1205963402, 1598704546],
        "note": "ALLERGIST/IMMUNOLOGY / MD, both under 1128b4 (licence action).",
    },
    {
        "question": "Which adult homes in Texas were excluded?",
        "expected_npis": [1487083457, 1982157269, 1952526055],
        "note": "ADULT HOME / TX, three records.",
    },

    # --- answerable: specialties with exactly one record in the whole file ---
    {
        "question": "Was a proctologist ever excluded?",
        "expected_npis": [1376524785],
        "note": "PROCTOLOGY appears once in the entire LEIE.",
    },
    {
        "question": "Has anyone working in phlebotomy been excluded?",
        "expected_npis": [1215272042],
        "note": "PHLEBOTOMY appears once.",
    },
    {
        "question": "Was a billing service company excluded?",
        "expected_npis": [1538388434],
        "note": "BILLING SERVICE CO appears once.",
    },
    {
        "question": "Has a paramedic technician been excluded?",
        "expected_npis": [1215053665],
        "note": "PARAMEDIC TECHNICIAN appears once.",
    },

    # --- answerable: rare specialty + state, exactly two records each ---
    {
        "question": "Were any ambulance companies in California excluded?",
        "expected_npis": [1871502146, 1811022312],
        "note": "AMBULANCE COMPANY / CA -- exactly two.",
    },
    {
        "question": "Which anesthesiologists in Colorado were excluded?",
        "expected_npis": [1366428773, 1629085105],
        "note": "ANESTHESIOLOGY / CO -- exactly two.",
    },
    {
        "question": "Were any assisted living facilities in Florida excluded?",
        "expected_npis": [1750914099, 1720580863],
        "note": "ASSISTED LIVING FACI / FL -- exactly two.",
    },
    {
        "question": "Which cardiologists in Colorado were excluded?",
        "expected_npis": [1518929421, 1558315986],
        "note": "CARDIOLOGY / CO -- exactly two.",
    },
    {
        "question": "Were any dermatologists in Illinois excluded?",
        "expected_npis": [1154362937, 1538200308],
        "note": "DERMATOLOGY / IL -- exactly two.",
    },
    {
        "question": "Which community mental health centers in New York were excluded?",
        "expected_npis": [1528377181, 1528353307],
        "note": "COMM MNTL HLTH CNTR / NY -- exactly two.",
    },
    {
        "question": "Were any oxygen equipment suppliers in Maine excluded?",
        "expected_npis": [1659305845, 1467486654],
        "note": "DME - OXYGEN / ME -- exactly two.",
    },

    # --- answerable: specialties with exactly one NPI-bearing record ---
    {
        "question": "Was a rural health clinic ever excluded?",
        "expected_npis": [1215925102],
        "note": "RURAL HEALTH CLINIC appears once among NPI-bearing records.",
    },
    {
        "question": "Has an interpreter or translator been excluded?",
        "expected_npis": [1982069944],
        "note": "INTERPRETER/TRANS appears once.",
    },
    {
        "question": "Was a home-infusion equipment supplier excluded?",
        "expected_npis": [1043284847],
        "note": "DME - HOME INFUSION appears once.",
    },
    {
        "question": "Has a hotel or lodging business been excluded?",
        "expected_npis": [1013137348],
        "note": "HOTEL/LODGING appears once -- an unusual category worth probing.",
    },

    # --- expected refusals: nothing in the corpus is about this at all ---
    {
        "question": "What is the capital of France?",
        "expected_refusal": True,
        "note": "Off-domain. Retrieval still returns k records -- providers named Frances -- "
                "which is why the refusal must come from grounding, not from empty results.",
    },
    {
        "question": "How do I appeal an OIG exclusion decision?",
        "expected_refusal": True,
        "note": "Plausible, on-topic, and absent: the records state WHO was excluded and "
                "under which statute, never any process or procedure.",
    },
    {
        "question": "Which insurance companies stopped paying these providers?",
        "expected_refusal": True,
        "note": "The LEIE has no payer information whatsoever.",
    },

    # --- traps: the records look relevant but do not answer what was asked ---
    {
        "question": "How much money did the excluded pharmacies in New York defraud Medicare of?",
        "expected_refusal": True,
        "note": "TRAP. Retrieval will surface the NY pharmacies, and a fluent model will be "
                "tempted to produce a figure. The LEIE holds no monetary amounts at all.",
    },
    {
        "question": "Are any of the excluded providers in Texas now reinstated?",
        "expected_refusal": True,
        "note": "TRAP. Reinstatement is a real LEIE concept but is not in the indexed fields "
                "(name, specialty, state, date, exclusion type, general category).",
    },
    {
        "question": "What was the name of the patient harmed by the excluded chiropractors?",
        "expected_refusal": True,
        "note": "TRAP. There is no patient data anywhere in the LEIE, and inventing one "
                "would be the most damaging failure this system could have.",
    },

    # --- refusals and traps: on-topic, but absent from the indexed fields ---
    {
        "question": "What are the office phone numbers of the excluded ambulance companies?",
        "expected_refusal": True,
        "note": "Retrieval will surface the ambulance records, but the indexed fields carry no "
                "telephone number -- a fluent model may still try to produce one.",
    },
    {
        "question": "How many years in prison did the excluded pharmacists receive?",
        "expected_refusal": True,
        "note": "TRAP. The LEIE records the exclusion, never any criminal sentence; retrieval "
                "surfaces pharmacists and the temptation is to invent a term of years.",
    },
    {
        "question": "Which of the excluded providers have since appealed and won reinstatement?",
        "expected_refusal": True,
        "note": "TRAP. A reinstatement date exists in the raw file but is NOT among the indexed "
                "fields, and appeal outcomes are nowhere in the LEIE at all.",
    },
]


def is_expected_refusal(item):
    return bool(item.get("expected_refusal"))


def answerable_items():
    return [item for item in GOLDEN_SET if not is_expected_refusal(item)]


def refusal_items():
    return [item for item in GOLDEN_SET if is_expected_refusal(item)]


def verify():
    """Gives every expected NPI missing from the LEIE, so a stale golden set fails instead of scoring wrong."""
    from ingest import load_leie

    leie = load_leie()
    known = set(leie.loc[leie["NPI"] != 0, "NPI"])

    problems = []
    for item in answerable_items():
        missing = [npi for npi in item["expected_npis"] if npi not in known]
        if missing:
            problems.append(f"{item['question']!r}: NPIs not in the LEIE: {missing}")
    return problems


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    answerable = answerable_items()
    refusals = refusal_items()
    print(f"{len(GOLDEN_SET)} questions: {len(answerable)} answerable, "
          f"{len(refusals)} expected refusals "
          f"({sum(1 for i in refusals if 'TRAP' in i['note'])} of them traps)")

    problems = verify()
    if problems:
        print("\nGOLDEN SET IS STALE:")
        for problem in problems:
            print(f"  {problem}")
        raise SystemExit(1)
    print("\nevery expected NPI verified against the LEIE")
