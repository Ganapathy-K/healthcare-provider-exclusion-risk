"""Route a question to the right tool: the risk model, or retrieval over the exclusion records.

Two tools, one router:

  score_provider_risk   "what is the risk score for NPI 1871596098?"  -> the XGBoost model
  query_leie_rag        "are there any excluded pharmacies in NY?"    -> retrieval + Gemini

The loop: the router picks a tool, the tool answers, and an LLM check reads that answer. When
the answer does not answer the question and the other tool can run, the other tool runs once.
When the scorer has nothing to score, code sends the question to RAG without asking the check.
Two tool runs at most, so the agent cannot loop forever.

  router -> tool -> check -> done
                         -> switch -> tool (second and last run)
            tool (could not score) -> switch -> tool (RAG)

The router is an LLM, not a keyword rule, because the two intents are separated by what the
user WANTS rather than by any word they use: "tell me about 1871596098" and "who else did what
1871596098 did" share every token that matters and need different tools.

⚠️ TWO THINGS CORRECTED DURING EXTRACTION, both worth knowing:

1. **The first version loaded a different model.** It read the XGBoost model straight from an MLflow
   artifact path (`mlruns/0/models/m-787607a1.../artifacts`) -- which is the OLD unweighted
   model, the one measured at recall 0.177. The agent was therefore answering with a model
   nobody had validated and nobody had deployed. It now loads `serving/model.ubj`, the same
   artefact the live API serves, so the agent and the endpoint cannot disagree.

2. **The encoding logic was a third copy.** The agent, `serving/app.py` and the training path
   each had their own. It now comes from `features.encode_provider_record`.

The six cases below are a DEMO, not a measurement: they were written by the person who wrote
the router, right after writing it. The measurement lives in `router_eval.py` -- 17 cases,
about half of them on the boundary -- and it scored 17/17 on intent and 17/17 on NPI
extraction (2026-09-03).
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
                    PROVIDER_LOOKUP_PATH, RISK_THRESHOLD)
from features import FEATURE_COLUMNS, encode_provider_record, load_encoding_maps
from generate import REFUSAL_TEXT, answer_question, get_client
from injection_guard import (BOUNDARY_INSTRUCTION, TOOL_BOUNDARY_INSTRUCTION, wrap,
                             wrap_tool_answer)
from rbac import DEFAULT_ROLE, get_role
from retrieve import format_sources
from tracing import flush, trace_span, update_span

# The columns the encoder needs, kept in agent_columns so `ingest` can write the slim
# lookup parquet without importing this module's langgraph/Gemini/embedding stack.
from agent_columns import LOOKUP_COLUMNS

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

CHECK_PROMPT = (
    "You check one answer from a healthcare provider-integrity assistant before the user sees "
    "it. The assistant has two tools:\n"
    "  score_provider_risk: the ML model's exclusion-risk score for ONE provider, given an NPI.\n"
    "  query_leie_rag: answers from the exclusion records -- who was excluded, why, where, when.\n\n"
    "Decide whether the tool's answer answers the user's question.\n"
    '- {"next": "done"} when it does.\n'
    '- {"next": "switch"} when it does not and the OTHER tool could: the tool refused, found no '
    "record, had no valid NPI, or answered a different kind of question (one provider's score "
    "for a question about a group, or records for a request for one provider's score).\n\n"
    'Respond with ONLY a JSON object: {"next": "done" | "switch"}.'
)

TOOLS = ("query_leie_rag", "score_provider_risk")
MAX_TOOL_RUNS = 2

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
    """The provider table used for scoring, preferring the slim 12-column copy.

    Falls back to the full labelled dataset, which is the same rows and 331 columns. The slim
    file exists because this is what ships inside the deployed image: 60 MB down to 8.3 MB,
    and Cloud Run cold-start time scales with image size.
    """
    global _lookup
    if _lookup is None:
        source = (PROVIDER_LOOKUP_PATH if PROVIDER_LOOKUP_PATH.exists()
                  else LABELLED_DATASET_PATH)
        _lookup = pd.read_parquet(source, columns=LOOKUP_COLUMNS)
    return _lookup


def score_provider_risk(npi):
    """Score one provider by NPI. Every failure returns a sentence, never an exception --
    this is a tool an LLM calls, and a traceback is not an answer a user can act on."""
    return score_npi(npi)[0]


def score_npi(npi):
    """The scorer with its outcome kept: returns (sentence, scored). `scored` is False when
    there was nothing to score -- no NPI, a malformed one, or one not in the data."""
    if npi in (None, "", "null"):
        return ("No NPI was supplied, so I can't score a specific provider. "
                "Please include a 10-digit NPI."), False
    try:
        npi_value = int(str(npi).strip())
    except ValueError:
        return f"'{npi}' is not a valid NPI (expected 10 digits).", False

    matches = get_lookup().query("NPI == @npi_value")
    if matches.empty:
        return f"NPI {npi_value} was not found in the provider dataset.", False

    features = encode_provider_record(matches.iloc[0], load_encoding_maps())
    probability = float(get_model().predict_proba(features[FEATURE_COLUMNS])[0][1])
    tier = "high" if probability >= RISK_THRESHOLD else "low"
    return (f"Risk score for NPI {npi_value}: {probability:.4f} (risk tier: {tier} — higher "
            "means more likely to be excluded). This prioritises review; it is not a finding."), True


def query_leie_rag(question, role):
    """Grounded retrieval over the exclusion records, with sources appended.

    Sources are appended only when the question was actually ANSWERED. Listing them under a
    refusal reads as "here is what I based that on", when the truth is the opposite -- those
    records were retrieved and then found not to answer anything. Retrieval always returns k
    records, so on an off-topic question ("what is the capital of France?") the nearest
    neighbours are simply providers whose names contain "Frances", and printing them beneath
    a refusal makes a correct refusal look like a confused answer.
    """
    return run_rag(question, role)[0]


def run_rag(question, role):
    """RAG with its evidence kept: returns (answer text, documents cited, refused)."""
    answer, documents = answer_question(question, role=role)
    refused = REFUSAL_TEXT.lower() in answer.lower()
    if refused or not documents:
        return answer, [], refused
    return f"{answer}\n\nSources:\n{format_sources(documents)}", documents, False


class AgentState(TypedDict):
    question: str
    tool_used: str          # the tool whose answer the user gets
    npi: str
    answer: str
    role: str
    documents: list         # records behind a RAG answer; empty otherwise
    refused: bool           # RAG refused, or the role may not use the chosen tool
    blocked: bool           # the role may not use the chosen tool -- never retried
    tool_failed: bool       # the scorer had nothing to score -- goes to RAG without the check
    tools_run: list         # every tool that ran, in order
    next_step: str          # the check's verdict: "done" or "switch"
    previous: dict | None   # the first tool's result, kept while the second tool runs


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

    Falls back to `rag` when the router output cannot be parsed. That default is deliberate:
    retrieval over records is the harmless branch, while a mis-routed `risk` either scores the
    wrong provider or fails for want of an NPI.

    ⚠️ THE FALLBACK HAS TO CATCH A PARSE ERROR, NOT JUST A MISSING BRACE, AND THE FIRST VERSION
    DID NOT. The old code fell back only when the regex found no `{...}`; when it found braces
    that were not valid JSON, json.loads RAISED and took the whole agent down. A prompt-injection
    attempt (found by injection_eval.py running with the boundary guard disabled) made the router
    echo brace content the greedy match grabbed and could not parse -- a crash, not a safe
    default, on exactly the adversarial input the fallback existed to survive. Failing closed
    means treating ANY unparseable output as `rag`, which is what the docstring always claimed.
    """
    # The question is delimited and the boundary rule appended so a routing instruction smuggled
    # into the text ("...route it to the scorer") is classified as part of the question, not
    # obeyed as a command. injection_eval.py caught exactly this: an aggregate RAG question with
    # an injected NPI and a "route to scorer" rider was diverted to the risk branch.
    response = get_client().models.generate_content(
        model=GENERATION_MODEL_NAME,
        contents=f"{ROUTER_PROMPT}\n{BOUNDARY_INSTRUCTION}\n\n{wrap(question)}",
        config=types.GenerateContentConfig(temperature=0))
    match = re.search(r"\{.*\}", response.text or "", re.DOTALL)
    try:
        parsed = json.loads(match.group(0)) if match else {"intent": "rag"}
    except (json.JSONDecodeError, ValueError):
        parsed = {"intent": "rag"}

    return {
        "intent": "risk" if parsed.get("intent") == "risk" else "rag",
        "npi": extract_npi(question),
    }


def judge_answer(question, tool, answer):
    """Ask the LLM whether the tool's answer answers the question. Returns "done" or "switch".

    The tool's answer is wrapped like the question, because a RAG answer is built from records
    and a record can carry an instruction. Any failure -- an unreadable verdict or a failed
    call -- returns "done": the agent keeps the answer it already has rather than running a
    second tool on a guess.
    """
    try:
        response = get_client().models.generate_content(
            model=GENERATION_MODEL_NAME,
            contents=(f"{CHECK_PROMPT}\n{BOUNDARY_INSTRUCTION}\n{TOOL_BOUNDARY_INSTRUCTION}\n\n"
                      f"{wrap(question)}\n\nTool used: {tool}\n{wrap_tool_answer(answer)}"),
            config=types.GenerateContentConfig(temperature=0))
        match = re.search(r"\{.*\}", response.text or "", re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {}
    except Exception:
        return "done"
    return "switch" if parsed.get("next") == "switch" else "done"


def router_node(state: AgentState) -> AgentState:
    """Choose the tool, and record the choice.

    The routing decision is the single most useful thing in a trace of this agent: a question
    answered by the wrong branch produces a confident, well-formed reply from the wrong half
    of the system, which the response text alone rarely reveals.
    """
    with trace_span("route", as_type="span", question=state["question"]) as span:
        decision = classify_intent(state["question"])
        state["tool_used"] = ("score_provider_risk" if decision["intent"] == "risk"
                              else "query_leie_rag")
        state["npi"] = decision["npi"]
        update_span(span, output={"tool": state["tool_used"], "npi": state["npi"] or None})
    return state


def tool_node(state: AgentState) -> AgentState:
    """Run the chosen tool and record what came back."""
    # DEFAULT_ROLE is "public", which retrieves nothing. A state that reaches here without a
    # role is a bug upstream, and the safe reading of a bug is that nobody has been
    # authenticated -- not that everybody is an investigator.
    role = get_role(state.get("role") or DEFAULT_ROLE)
    tool = state["tool_used"]
    state["tools_run"] = state["tools_run"] + [tool]
    state["documents"], state["refused"], state["blocked"] = [], False, False
    state["tool_failed"] = False

    if tool == "score_provider_risk":
        # The scoring tool is investigator-only. A risk score is about ONE named provider, so
        # there is no de-identified version of it: returning a score for an NPI the caller
        # supplied confirms that provider is in the dataset, which is itself disclosure.
        if "NPI" not in role.visible_fields:
            state["answer"] = (f"The '{role.name}' role cannot retrieve provider-level risk "
                               "scores. Scoring identifies a specific provider.")
            state["refused"], state["blocked"] = True, True
        else:
            state["answer"], scored = score_npi(state["npi"])
            state["tool_failed"] = not scored
    else:
        state["answer"], state["documents"], state["refused"] = run_rag(state["question"], role)

    # A second run that also fails is no better than the first, so the first answer stands.
    if state["previous"] and (state["refused"] or state["tool_failed"]):
        state.update(state["previous"])
    return state


def check_node(state: AgentState) -> AgentState:
    """Ask the LLM whether the tool's answer answers the question."""
    with trace_span("check", as_type="span", tool=state["tool_used"]) as span:
        state["next_step"] = judge_answer(state["question"], state["tool_used"], state["answer"])
        update_span(span, output={"next": state["next_step"]})
    return state


def switch_node(state: AgentState) -> AgentState:
    """Keep the first answer, then point the agent at the other tool."""
    state["previous"] = {key: state[key] for key in ("tool_used", "answer", "documents",
                                                     "refused", "blocked")}
    state["tool_used"] = TOOLS[1] if state["tool_used"] == TOOLS[0] else TOOLS[0]
    return state


def after_tool(state: AgentState) -> str:
    """Stop without a check when there is nothing left to try.

    ⚠️ A SCORE THAT COULD NOT BE COMPUTED GOES STRAIGHT TO RAG, AND CODE DECIDES THAT, NOT THE
    CHECK. Measured 2026-09-15: the check was given "NPI 9999999999 was not found in the provider
    dataset." five times at temperature 0 and answered switch, done, done, switch, switch -- so
    the same failure reached RAG on some runs and dead-ended on others. A failure the scorer
    already knows about needs no judgement.
    """
    if state["blocked"] or len(state["tools_run"]) >= MAX_TOOL_RUNS:
        return END
    if state["tool_failed"]:
        return "switch"
    return "check"


def after_check(state: AgentState) -> str:
    """Switch only when the check says so AND the other tool can actually run."""
    if state["next_step"] != "switch":
        return END
    other = TOOLS[1] if state["tool_used"] == TOOLS[0] else TOOLS[0]
    role = get_role(state.get("role") or DEFAULT_ROLE)
    if other == "score_provider_risk" and (not state["npi"] or "NPI" not in role.visible_fields):
        return END
    return "switch"


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("tool", tool_node)
    graph.add_node("check", check_node)
    graph.add_node("switch", switch_node)
    graph.set_entry_point("router")
    graph.add_edge("router", "tool")
    graph.add_conditional_edges("tool", after_tool,
                                {"check": "check", "switch": "switch", END: END})
    graph.add_conditional_edges("check", after_check, {"switch": "switch", END: END})
    graph.add_edge("switch", "tool")
    return graph.compile()


def ask(question, role, agent=None):
    """Run one question through the agent, as one trace.

    The outer span is what makes the inner ones a story rather than three unrelated events:
    route -> retrieve -> generate, linked, with the question at the top.
    """
    agent = agent or build_agent()
    with trace_span("agent", as_type="span", question=question, role=role) as span:
        result = agent.invoke({"question": question, "tool_used": "", "npi": "",
                               "answer": "", "role": role, "documents": [],
                               "refused": False, "blocked": False, "tool_failed": False,
                               "tools_run": [],
                               "next_step": "", "previous": None})
        update_span(span, output={"tool": result["tool_used"],
                                  "tools_run": result["tools_run"],
                                  "npi": result["npi"] or None,
                                  "answer": result["answer"][:500]})
    return result


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
        result = ask(question, role="investigator", agent=agent)
        # The FIRST tool is the routing decision; the final one can differ after a switch.
        routed_right = result["tools_run"][0] == expected
        correct += routed_right
        npi_note = f" (npi {result['npi']})" if result["npi"] else ""
        print(f"{'OK  ' if routed_right else 'MISS'}  {question}")
        print(f"      -> {result['tool_used']}{npi_note}   tools run: {result['tools_run']}")
        print(f"      {result['answer'].strip()[:220]}\n")

    print(f"routing: {correct}/{len(cases)} correct")
