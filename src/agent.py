"""Route a question to the right tool: the risk model, or retrieval over the exclusion records.

Extracted from notebook 05. Two tools, one router:

  score_provider_risk   "what is the risk score for NPI 1871596098?"  -> the XGBoost model
  query_leie_rag        "are there any excluded pharmacies in NY?"    -> retrieval + Gemini

The router is an LLM, not a keyword rule, because the two intents are separated by what the
user WANTS rather than by any word they use: "tell me about 1871596098" and "who else did what
1871596098 did" share every token that matters and need different tools.

⚠️ TWO THINGS CORRECTED DURING EXTRACTION, both worth knowing:

1. **The notebook loaded a different model.** It read the XGBoost model straight from an MLflow
   artifact path (`mlruns/0/models/m-787607a1.../artifacts`) -- which is the OLD unweighted
   model, the one measured at recall 0.177. The agent was therefore answering with a model
   nobody had validated and nobody had deployed. It now loads `serving/model.ubj`, the same
   artefact the live API serves, so the agent and the endpoint cannot disagree.

2. **The encoding logic was a third copy.** Notebook 05, `serving/app.py` and the training path
   each had their own. It now comes from `features.encode_provider_record`.

Still open, and not fixed here: the router has no measured accuracy. There is no set of
questions with known correct routes, so "6 out of 6 worked" is the entire evidence base.
"""

import json
import re
import sys
from typing import TypedDict

import pandas as pd
import xgboost as xgb
from google.genai import types
from langgraph.graph import END, StateGraph

from config import (GENERATION_MODEL_NAME, LABELLED_DATASET_PATH, MODEL_PATH,
                    RISK_THRESHOLD)
from features import FEATURE_COLUMNS, encode_provider_record, load_encoding_maps
from generate import REFUSAL_TEXT, answer_question, get_client
from retrieve import format_sources

# Only the columns the encoder needs. The labelled dataset is 331 columns wide and reading all
# of them to score one provider costs about a gigabyte of memory for no benefit.
LOOKUP_COLUMNS = [
    "NPI", "Entity Type Code", "Provider Business Mailing Address Telephone Number",
    "Provider Enumeration Date", "Last Update Date", "Provider Sex Code",
    "Healthcare Provider Primary Taxonomy Switch_1", "Is Sole Proprietor",
    "Healthcare Provider Taxonomy Code_1", "Provider Business Mailing Address State Name",
    "Provider Business Practice Location Address State Name",
    "Provider License Number State Code_1",
]

ROUTER_PROMPT = (
    "You are the router for a healthcare provider-integrity assistant. "
    "Classify the question into exactly one intent and extract any NPI (a 10-digit provider "
    "identifier).\n\n"
    'intent = "risk": the user wants the ML model\'s exclusion-RISK SCORE or probability for '
    'ONE specific provider (usually given by an NPI). Signals: "risk score", "how risky", '
    '"probability", "should we pay/worry about <NPI>", or a bare 10-digit number.\n'
    'intent = "rag": the user asks about the exclusion RECORDS themselves -- looking up, '
    'listing, counting, or explaining who is excluded, why, where, or when. Signals: '
    '"are there any", "who", "list", "how many", "what reason", "in <state>".\n\n'
    "THE DECIDING TEST, which matters more than any keyword: is the question about ONE "
    "PARTICULAR PROVIDER, or about the data in general? A question containing the words "
    '"risk", "risky" or "riskiest" is still "rag" when it asks about a group, a specialty, a '
    "state, or what tends to be true -- because the model scores one provider at a time and "
    "cannot answer a question about a population.\n\n"
    'Respond with ONLY a JSON object: {"intent": "risk" | "rag"}.\n\n'
    "Examples:\n"
    'Q: Are there any excluded providers in Texas? -> {"intent": "rag"}\n'
    'Q: How many providers were excluded for fraud? -> {"intent": "rag"}\n'
    'Q: Which specialties are the riskiest overall? -> {"intent": "rag"}   '
    "(about a population, not one provider)\n"
    'Q: What makes a provider high risk? -> {"intent": "rag"}   '
    "(asks what tends to be true, not for a score)\n"
    'Q: What is the risk score for NPI 1234567890? -> {"intent": "risk"}\n'
    'Q: Should we be worried about 1679576722? -> {"intent": "risk"}\n'
    'Q: How risky is that pharmacy in New York? -> {"intent": "risk"}   '
    "(one provider, even with no NPI given)\n\n"
    "Question: "
)

_model = None
_lookup = None


def get_model():
    """The SAME artefact the deployed API serves -- see correction 1 in the module docstring."""
    global _model
    if _model is None:
        _model = xgb.XGBClassifier()
        _model.load_model(MODEL_PATH)
    return _model


def get_lookup():
    global _lookup
    if _lookup is None:
        _lookup = pd.read_parquet(LABELLED_DATASET_PATH, columns=LOOKUP_COLUMNS)
    return _lookup


def score_provider_risk(npi):
    """Score one provider by NPI. Every failure returns a sentence, never an exception --
    this is a tool an LLM calls, and a traceback is not an answer a user can act on."""
    if npi in (None, "", "null"):
        return ("No NPI was supplied, so I can't score a specific provider. "
                "Please include a 10-digit NPI.")
    try:
        npi_value = int(str(npi).strip())
    except ValueError:
        return f"'{npi}' is not a valid NPI (expected 10 digits)."

    matches = get_lookup().query("NPI == @npi_value")
    if matches.empty:
        return f"NPI {npi_value} was not found in the provider dataset."

    features = encode_provider_record(matches.iloc[0], load_encoding_maps())
    probability = float(get_model().predict_proba(features[FEATURE_COLUMNS])[0][1])
    tier = "high" if probability >= RISK_THRESHOLD else "low"
    return (f"Risk score for NPI {npi_value}: {probability:.4f} (risk tier: {tier} — higher "
            "means more likely to be excluded). This prioritises review; it is not a finding.")


def query_leie_rag(question):
    """Grounded retrieval over the exclusion records, with sources appended.

    Sources are appended only when the question was actually ANSWERED. Listing them under a
    refusal reads as "here is what I based that on", when the truth is the opposite -- those
    records were retrieved and then found not to answer anything. Retrieval always returns k
    records, so on an off-topic question ("what is the capital of France?") the nearest
    neighbours are simply providers whose names contain "Frances", and printing them beneath
    a refusal makes a correct refusal look like a confused answer.
    """
    answer, documents = answer_question(question)
    refused = REFUSAL_TEXT.lower() in answer.lower()
    if refused or not documents:
        return answer
    return f"{answer}\n\nSources:\n{format_sources(documents)}"


class AgentState(TypedDict):
    question: str
    tool_used: str
    npi: str
    answer: str


# An NPI is exactly ten digits. The lookarounds stop a 12-digit string from yielding a
# spurious 10-digit "match" out of its middle.
NPI_PATTERN = re.compile(r"(?<!\d)\d{10}(?!\d)")


def extract_npi(question):
    """Pull the NPI out with a regex, not the LLM.

    ⚠️ WHY THIS WAS TAKEN AWAY FROM THE MODEL. The router originally did two jobs in one call:
    decide the intent, and extract the identifier. The second is a fixed-width number in a
    string -- a regex does it perfectly, for free, identically every time.

    That matters more than it sounds, because LLM output is NOT reproducible even at
    temperature 0. Measured 2026-07-27 on this exact prompt: two consecutive runs at
    temperature 0 disagreed on which questions they got right, and the extracted NPI for
    "risk score for NPI 123456789?" changed between runs. Server-side batching and routing
    mean identical inputs need not give identical outputs; "set temperature to 0 for
    determinism" is folklore.

    So the model keeps the judgement call it is actually needed for -- what the user WANTS --
    and the deterministic half stops being a source of variance.
    """
    match = NPI_PATTERN.search(question)
    return match.group(0) if match else ""


def classify_intent(question):
    """Decide which tool this question needs. The NPI comes from `extract_npi`, not from here.

    Falls back to `rag` when the JSON cannot be parsed. That default is deliberate: retrieval
    over records is the harmless branch, while a mis-routed `risk` either scores the wrong
    provider or fails for want of an NPI.
    """
    response = get_client().models.generate_content(
        model=GENERATION_MODEL_NAME, contents=ROUTER_PROMPT + question,
        config=types.GenerateContentConfig(temperature=0))
    match = re.search(r"\{.*\}", response.text or "", re.DOTALL)
    parsed = json.loads(match.group(0)) if match else {"intent": "rag"}

    return {
        "intent": "risk" if parsed.get("intent") == "risk" else "rag",
        "npi": extract_npi(question),
    }


def router_node(state: AgentState) -> AgentState:
    decision = classify_intent(state["question"])
    state["tool_used"] = ("score_provider_risk" if decision["intent"] == "risk"
                          else "query_leie_rag")
    state["npi"] = decision["npi"]
    return state


def tool_node(state: AgentState) -> AgentState:
    if state["tool_used"] == "score_provider_risk":
        state["answer"] = score_provider_risk(state["npi"])
    else:
        state["answer"] = query_leie_rag(state["question"])
    return state


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("tool", tool_node)
    graph.set_entry_point("router")
    graph.add_edge("router", "tool")
    graph.add_edge("tool", END)
    return graph.compile()


def ask(question, agent=None):
    agent = agent or build_agent()
    return agent.invoke({"question": question, "tool_used": "", "npi": "", "answer": ""})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    # Each case names the tool it SHOULD reach, so a mis-route is visible rather than merely
    # producing a plausible answer from the wrong half of the system.
    cases = [
        ("Are there any excluded providers in Texas?", "query_leie_rag"),
        ("What is the exclusion risk score for provider NPI 1871596098?",
         "score_provider_risk"),
        ("Should we be worried about 1679576722?", "score_provider_risk"),
        ("Who was excluded for patient abuse?", "query_leie_rag"),
        ("What is the capital of France?", "query_leie_rag"),
        ("Give me the risk score for NPI 9999999999", "score_provider_risk"),
    ]

    agent = build_agent()
    correct = 0
    for question, expected in cases:
        result = ask(question, agent)
        routed_right = result["tool_used"] == expected
        correct += routed_right
        npi_note = f" (npi {result['npi']})" if result["npi"] else ""
        print(f"{'OK  ' if routed_right else 'MISS'}  {question}")
        print(f"      -> {result['tool_used']}{npi_note}")
        print(f"      {result['answer'].strip()[:220]}\n")

    print(f"routing: {correct}/{len(cases)} correct")
