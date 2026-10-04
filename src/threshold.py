"""The threshold picked on out-of-fold train scores and checked on test rows, to replace 0.5."""

import re
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import (LABELLED_DATASET_PATH, MODEL_PATH, PROJECT_ROOT, RANDOM_STATE,
                    SCALE_POS_WEIGHT, TARGET_COLUMN, TEST_SIZE)
from features import FEATURE_COLUMNS, fit_encoding_maps, prepare_features
from model import build_model

CONFIG_PATH = PROJECT_ROOT / "src" / "config.py"
FOLDS = 5

# The threshold in use today, printed next to the derived one for comparison.
INHERITED_THRESHOLD = 0.5


def out_of_fold_probabilities(raw, train_rows):
    """One out-of-fold score per train row (5 folds), to pick the threshold on unseen scores."""
    target = raw.loc[train_rows, TARGET_COLUMN]
    probabilities = pd.Series(index=train_rows, dtype=float)

    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for number, (fit_positions, score_positions) in enumerate(
            folds.split(train_rows, target), 1):
        fit_rows = train_rows[fit_positions]
        score_rows = train_rows[score_positions]

        encoding_maps = fit_encoding_maps(raw.loc[fit_rows])
        features, labels, _ = prepare_features(raw, encoding_maps=encoding_maps)

        model = build_model()
        model.fit(features.loc[fit_rows, FEATURE_COLUMNS], labels.loc[fit_rows])
        probabilities.loc[score_rows] = model.predict_proba(
            features.loc[score_rows, FEATURE_COLUMNS])[:, 1]
        print(f"  fold {number}/{FOLDS} scored {len(score_rows):,} rows", flush=True)

    return target.to_numpy(), probabilities.to_numpy()


def cost_threshold(target, probabilities, cost_ratio):
    """The score with the lowest 0 × TP + 422 × FN + 0 × TN + 1 × FP, to minimise total cost."""
    false_positive_rate, recall, thresholds = roc_curve(target, probabilities)
    positives = int(target.sum())
    negatives = len(target) - positives
    costs = cost_ratio * positives * (1 - recall) + negatives * false_positive_rate
    # thresholds[0] is roc_curve's "flag nobody" point (infinity); a cut-off must flag someone.
    return float(thresholds[1:][int(np.argmin(costs[1:]))])


def youden_threshold(target, probabilities):
    """The score where recall − false-positive rate is biggest, to cross-check the cost pick."""
    false_positive_rate, recall, thresholds = roc_curve(target, probabilities)
    return float(thresholds[int(np.argmax(recall - false_positive_rate))])


def threshold_report(target, probabilities, threshold):
    """Caught, flagged, recall and precision at one threshold, to compare thresholds."""
    _, false_alarms, _, caught = confusion_matrix(
        target, probabilities >= threshold, labels=[0, 1]).ravel()
    caught, false_alarms = int(caught), int(false_alarms)
    flagged = caught + false_alarms
    positives = int(target.sum())
    return {
        "threshold": round(threshold, 4),
        "caught": caught,
        "positives": positives,
        "flagged": flagged,
        "recall": round(caught / positives, 4),
        "precision": round(caught / flagged, 5) if flagged else 0.0,
    }


def write_threshold(value):
    """RISK_THRESHOLD rewritten in config.py, to ship the derived threshold."""
    source = CONFIG_PATH.read_text(encoding="utf-8")
    updated, replaced = re.subn(r"^RISK_THRESHOLD = [\d.]+$", f"RISK_THRESHOLD = {value}",
                                source, count=1, flags=re.MULTILINE)
    if replaced != 1:
        raise SystemExit("RISK_THRESHOLD assignment not found in config.py -- not written.")
    CONFIG_PATH.write_text(updated, encoding="utf-8")
    print(f"config.py: RISK_THRESHOLD = {value}")


def main(write=False):
    raw = pd.read_parquet(LABELLED_DATASET_PATH)
    train_rows, test_rows = train_test_split(
        raw.index, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=raw[TARGET_COLUMN])

    print(f"deriving on {len(train_rows):,} training rows, {FOLDS}-fold out-of-fold")
    train_target, train_probabilities = out_of_fold_probabilities(raw, train_rows)

    by_cost = cost_threshold(train_target, train_probabilities, SCALE_POS_WEIGHT)
    by_youden = youden_threshold(train_target, train_probabilities)
    print(f"\ncost (ratio {SCALE_POS_WEIGHT}): {by_cost:.4f}    youden: {by_youden:.4f}")

    # Two methods, different reasons: the average is used, so neither decides alone.
    derived = round((by_cost + by_youden) / 2, 2)
    print(f"derived threshold: {derived}")

    encoding_maps = fit_encoding_maps(raw.loc[train_rows])
    features, labels, _ = prepare_features(raw, encoding_maps=encoding_maps)
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    test_target = labels.loc[test_rows].to_numpy()
    test_probabilities = model.predict_proba(features.loc[test_rows, FEATURE_COLUMNS])[:, 1]

    print("\nheld-out test split, reported once:")
    for name, threshold in (("inherited", INHERITED_THRESHOLD), ("derived", derived)):
        found = threshold_report(test_target, test_probabilities, threshold)
        print(f"  {name:<14} threshold {found['threshold']:<8} "
              f"caught {found['caught']}/{found['positives']}  "
              f"flagged {found['flagged']:,}  recall {found['recall']}")

    if write:
        print()
        write_threshold(derived)
    return derived


if __name__ == "__main__":
    main(write="--write" in sys.argv)
