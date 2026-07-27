"""Run every extracted module once and assert the things that were silently wrong tonight.

Not a unit-test suite -- there are no mocks and nothing is isolated. It is the check a person
would run before trusting the refactor: does each module import, does it do its one job, and
do the pieces that must agree still agree?

Every assertion here exists because the corresponding thing was ACTUALLY BROKEN on
2026-07-27, not because it seemed worth checking:

  wrong model deployed      serving/model.ubj had no scale_pos_weight; recall was 0.177
  wrong model in the agent  notebook 05 loaded a different artefact from MLflow entirely
  encoding maps mismatched  the maps shipped were fitted on data the model was not
  feature order duplicated  three copies of the column list, XGBoost validates none of it
  refusal on answerable Qs  the NPI was missing from the context the model was shown

Requires: the Qdrant container running (docker start qdrant-healthcare) and an API key.
Costs a handful of Gemini calls.

Run:  python src/smoke_test.py
"""

import sys
import traceback

CHECKS = []


def check(name):
    def register(function):
        CHECKS.append((name, function))
        return function
    return register


@check("config resolves every path it declares")
def _config():
    import config
    assert config.LEIE_PATH.exists(), f"missing LEIE at {config.LEIE_PATH}"
    assert config.LABELLED_DATASET_PATH.exists(), "labelled dataset missing"
    assert config.MODEL_PATH.exists(), "serving/model.ubj missing"
    assert config.GOOGLE_API_KEY, "no API key in the environment"
    return f"project root {config.PROJECT_ROOT.name}"


@check("ingest rebuilds the same labelled dataset")
def _ingest():
    import json

    from config import PROJECT_ROOT
    from ingest import build_labelled_dataset

    _, report = build_labelled_dataset()
    recorded = json.loads((PROJECT_ROOT / "docs" / "baseline.json").read_text())["dataset"]
    assert report["positives"] == recorded["positives"], (
        f"positives {report['positives']} != baseline {recorded['positives']}")
    assert report["nppes_rows"] == recorded["rows"], "row count drifted"
    return f"{report['nppes_rows']:,} rows, {report['positives']} excluded"


@check("feature order matches serving/app.py")
def _alignment():
    from features import FEATURE_COLUMNS, check_serving_alignment
    aligned, serving_columns = check_serving_alignment()
    assert aligned, (f"serving has {serving_columns}\nfeatures has {FEATURE_COLUMNS}")
    return f"{len(FEATURE_COLUMNS)} columns, identical in both places"


@check("the deployed model IS the weighted one")
def _model_config():
    import json

    import xgboost as xgb

    from config import MODEL_PATH
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    config = json.loads(model.get_booster().save_config())
    # Not under tree_train_param, where it looks like it should be: scale_pos_weight is an
    # objective parameter, because it reweights the loss rather than the tree search. And it
    # is NOT recoverable from get_params() on a loaded model -- that returns None for both the
    # weighted and unweighted files, which is precisely why this check reads the saved config.
    weight = config["learner"]["objective"]["reg_loss_param"]["scale_pos_weight"]
    assert float(weight) > 1, (
        f"scale_pos_weight is {weight} -- this is the unweighted model that catches "
        "1 excluded provider in 5. See src/model.py.")
    return f"scale_pos_weight={weight}"


@check("the model reproduces its recorded baseline")
def _baseline():
    import json

    from baseline import BASELINE_PATH, capture, differences
    recorded = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    drift = differences(recorded, capture())
    assert not drift, "\n".join(f"{where}: {was} -> {now}" for where, was, now in drift)
    return f"recall {recorded['model']['recall']:.4f}, roc_auc {recorded['model']['roc_auc']:.4f}"


@check("Qdrant is reachable and in sync with the LEIE")
def _vectorstore():
    from config import QDRANT_COLLECTION_NAME
    from vectorstore import build_documents, get_client

    client = get_client()
    assert client.collection_exists(QDRANT_COLLECTION_NAME), (
        "collection missing -- is the container running? docker start qdrant-healthcare")
    indexed = client.get_collection(QDRANT_COLLECTION_NAME).points_count
    expected = len(build_documents())
    assert indexed == expected, f"{indexed} indexed vs {expected} documents"
    return f"{indexed:,} vectors"


@check("retrieval finds the obvious record")
def _retrieve():
    from retrieve import retrieve
    documents = retrieve("Are there any excluded pharmacies in New York?")
    assert documents, "nothing retrieved"
    states = {doc.metadata["STATE"] for doc in documents}
    assert "NY" in states, f"expected a New York record, got states {states}"
    return f"{len(documents)} records, states {sorted(states)}"


@check("an answerable question is ANSWERED, with NPIs cited")
def _generate_answers():
    from generate import REFUSAL_TEXT, answer_question
    answer, documents = answer_question("Are there any excluded pharmacies in New York?")
    assert REFUSAL_TEXT.lower() not in answer.lower(), (
        "refused a question the retrieved records answer -- check that build_prompt puts the "
        "NPI in the context; demanding a citation the context cannot supply causes this.")
    cited = [str(doc.metadata["NPI"]) for doc in documents if str(doc.metadata["NPI"]) in answer]
    assert cited, f"no NPI cited in: {answer[:200]}"
    return f"{len(cited)} of {len(documents)} NPIs cited"


@check("an off-domain question is REFUSED")
def _generate_refuses():
    from generate import REFUSAL_TEXT, answer_question
    answer, _ = answer_question("What is the capital of France?")
    assert REFUSAL_TEXT.lower() in answer.lower(), f"answered off-domain: {answer[:200]}"
    return "refused"


@check("roles are enforced: names and NPIs stripped, unknown roles fail closed")
def _rbac():
    import json
    import re

    from generate import answer_question
    from rbac import get_role
    from retrieve import retrieve

    question = "Which acupuncturists in New York were excluded?"

    # An unknown role must not widen access. A typo that returns everything is the worst
    # possible default, and it is the default you get by accident.
    assert get_role("typo-role").name == "public", "unknown role did not fall back to public"
    assert get_role("").name == "public", "empty role did not fall back to public"

    answer, documents = answer_question(question, role="analyst")
    text = answer + json.dumps([doc.metadata for doc in documents])
    assert not re.search(r"GOHRING|ORLANDO", text.upper()), f"analyst saw a name: {text[:200]}"
    assert not re.search(r"(?<!\d)\d{10}(?!\d)", text), (
        "analyst saw an NPI -- one NPPES lookup turns that back into a name, so this is not "
        "de-identified")

    # The corpus must survive redaction. The first implementation mutated documents in place,
    # so one analyst query stripped the names out of the BM25 index for every later caller.
    after = retrieve(question, top_k=3)
    assert any("NAME" in doc.metadata for doc in after), (
        "redaction destroyed the shared corpus -- documents must be COPIED, not mutated")

    return "analyst de-identified, corpus intact, unknown roles closed"


@check("the agent routes both intents and scores a real NPI")
def _agent():
    from agent import ask, build_agent
    agent = build_agent()

    rag = ask("Are there any excluded providers in Texas?", agent)
    assert rag["tool_used"] == "query_leie_rag", f"routed to {rag['tool_used']}"

    risk = ask("What is the exclusion risk score for provider NPI 1871596098?", agent)
    assert risk["tool_used"] == "score_provider_risk", f"routed to {risk['tool_used']}"
    assert "1871596098" in risk["answer"], f"NPI lost: {risk['answer'][:150]}"
    assert "not found" not in risk["answer"], "known NPI reported missing"
    return "both intents routed correctly"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    passed, failed = 0, 0
    for name, function in CHECKS:
        try:
            detail = function()
            passed += 1
            print(f"PASS  {name}\n        {detail}")
        except Exception as error:
            failed += 1
            print(f"FAIL  {name}")
            print("        " + "\n        ".join(
                str(error).splitlines() or [traceback.format_exc()]))

    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
