"""Freeze what the pipeline does today, so the refactor can be proved not to have changed it.

The six notebooks are currently the only record of this project's behaviour. Moving that code
into `src/` without a reference means the only available check is "the output still looks
about right", which catches nothing subtle -- and the subtle failures are the ones that
matter here: a reordered column, an encoding map fitted differently, a dropped row.

So: run the pipeline as it stands, write the numbers to `docs/baseline.json`, and from then on
require every extraction to reproduce them. `--check` compares instead of writing, and exits
non-zero on any drift, so it can sit in front of a commit.

This is a characterization test, not a quality test. It asserts that behaviour is UNCHANGED,
including behaviour that is wrong -- notably the target leakage documented in features.py.
Fixing that is a separate, deliberate change that should move these numbers and be announced.

Run:  python src/baseline.py            # record
      python src/baseline.py --check    # compare, non-zero exit on drift
"""

import json
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

from config import (LABELLED_DATASET_PATH, MODEL_PATH, PROJECT_ROOT, RANDOM_STATE,
                    RISK_THRESHOLD, TARGET_COLUMN, TEST_SIZE)
from features import (FEATURE_COLUMNS, check_serving_alignment, fit_encoding_maps,
                      prepare_features)

BASELINE_PATH = PROJECT_ROOT / "docs" / "baseline.json"

# Metrics are rounded before comparison: XGBoost and sklearn can differ in the last bits
# across BLAS builds, and a test that fails on 1e-15 gets switched off within a week.
PRECISION_DIGITS = 6


def capture():
    """Run the current pipeline end to end and return every number worth pinning."""
    raw = pd.read_parquet(LABELLED_DATASET_PATH)

    # The split is taken on raw rows and the encodings fitted on the training side only, so
    # this mirrors exactly how the shipped model was built. Scoring it with full-data
    # encodings would hand the model numbers it was never trained on and report a drift that
    # is really a difference in the measuring stick.
    train_rows, test_rows = train_test_split(
        raw.index, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=raw[TARGET_COLUMN])
    encoding_maps = fit_encoding_maps(raw.loc[train_rows])
    features, target, _ = prepare_features(raw, encoding_maps=encoding_maps)
    features_test, target_test = features.loc[test_rows], target.loc[test_rows]

    aligned, _ = check_serving_alignment()

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    probabilities = model.predict_proba(features_test[FEATURE_COLUMNS])[:, 1]
    predictions = (probabilities >= RISK_THRESHOLD).astype(int)

    return {
        "dataset": {
            "rows": int(len(raw)),
            "raw_columns": int(raw.shape[1]),
            "feature_columns": int(features.shape[1]),
            "positives": int(target.sum()),
            "negatives": int(len(target) - target.sum()),
        },
        "feature_order": FEATURE_COLUMNS,
        "serving_alignment": aligned,
        "encoding_map_sizes": {column: len(mapping)
                               for column, mapping in encoding_maps.items()},
        "model": {
            "test_rows": int(len(target_test)),
            "test_positives": int(target_test.sum()),
            "threshold": RISK_THRESHOLD,
            "recall": round(float(recall_score(target_test, predictions)), PRECISION_DIGITS),
            "precision": round(float(precision_score(target_test, predictions)),
                               PRECISION_DIGITS),
            "f1": round(float(f1_score(target_test, predictions)), PRECISION_DIGITS),
            "roc_auc": round(float(roc_auc_score(target_test, probabilities)),
                             PRECISION_DIGITS),
        },
    }


def differences(recorded, current, path=""):
    """Every leaf where the two disagree, as (location, was, now)."""
    found = []
    if isinstance(recorded, dict) and isinstance(current, dict):
        for key in sorted(set(recorded) | set(current)):
            found.extend(differences(recorded.get(key), current.get(key),
                                     f"{path}.{key}" if path else key))
    elif recorded != current:
        found.append((path, recorded, current))
    return found


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    checking = "--check" in sys.argv

    current = capture()

    if checking:
        if not BASELINE_PATH.exists():
            raise SystemExit(f"no baseline recorded at {BASELINE_PATH}; run without --check")
        recorded = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        drift = differences(recorded, current)
        if drift:
            print(f"BASELINE DRIFT — {len(drift)} value(s) changed:\n")
            for location, was, now in drift:
                print(f"  {location}\n    was: {was}\n    now: {now}")
            raise SystemExit(1)
        print("baseline matches — behaviour unchanged")
    else:
        BASELINE_PATH.parent.mkdir(exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")
        model = current["model"]
        dataset = current["dataset"]
        print(f"dataset  : {dataset['rows']:,} rows, {dataset['positives']} excluded "
              f"({dataset['positives'] / dataset['rows']:.4%})")
        print(f"features : {dataset['feature_columns']} columns, "
              f"serving aligned: {current['serving_alignment']}")
        print(f"\ntest set : {model['test_rows']:,} rows, {model['test_positives']} positives")
        print(f"  recall    {model['recall']:.4f}")
        print(f"  precision {model['precision']:.4f}")
        print(f"  f1        {model['f1']:.4f}")
        print(f"  roc_auc   {model['roc_auc']:.4f}")
        print(f"\nrecorded -> {BASELINE_PATH}")
        print("\nRAG/agent behaviour is NOT captured here: it needs the Qdrant container on "
              f"{__import__('config').QDRANT_URL}, which was not running. Add it before "
              "extracting notebooks 04 and 05.")
