"""Derive the decision threshold from the data instead of inheriting XGBoost's 0.5.

`RISK_THRESHOLD` was 0.5 -- the library default, never chosen for this problem. 0.5 is the
correct cut-off only when the two classes are equally common and the two mistakes cost the
same. Here one provider in 422 is excluded, and a missed exclusion pays out a fraudulent
claim while a false flag costs one analyst review. Neither condition holds, so the inherited
default was not a model setting: it was an unstated claim about cost.

The threshold is derived on OUT-OF-FOLD predictions over the training rows only. Deriving it
on the test split would repeat, at the decision layer, the same mistake the target encodings
made at the feature layer -- choosing a number using the answer sheet it is later scored
against. The test split is used once, at the end, to report what the derived threshold does.

Two criteria are computed, and they are meant to agree or disagree in public:

  cost      minimises `cost_ratio * false_negatives + false_positives`. The ratio used is
            SCALE_POS_WEIGHT (422), which is not a new assumption: weighting the positive
            class by 422 during training already says a missed positive costs 422 false
            alarms. Using the same figure at the decision layer keeps training and serving
            telling one story.
  youden    maximises `recall - false_positive_rate`. It uses no cost figure at all, so
            agreement between the two is evidence the answer is not an artefact of the ratio.

Run:  python src/threshold.py              # derive, print the sweep, change nothing
      python src/threshold.py --write      # also write it to config.py and stamp model.ubj
"""

import re
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import (LABELLED_DATASET_PATH, MODEL_PATH, PROJECT_ROOT, RANDOM_STATE,
                    SCALE_POS_WEIGHT, TARGET_COLUMN, TEST_SIZE)
from features import FEATURE_COLUMNS, fit_encoding_maps, prepare_features
from model import build_model

CONFIG_PATH = PROJECT_ROOT / "src" / "config.py"
FOLDS = 5

# What the project shipped before this file existed. Kept as a literal rather than read from
# RISK_THRESHOLD, which this run overwrites -- comparing the new number against itself is a
# report that can never say anything.
INHERITED_THRESHOLD = 0.5

# The ratios printed in the sweep: how many false alarms one missed exclusion is worth. They
# span the range where the answer actually moves -- below roughly 50 the model flags nobody,
# above roughly 2500 it flags everybody, and both ends are printed rather than trimmed
# because the shape of the collapse is the argument for the middle.
COST_RATIOS = [1, 10, 50, 100, 250, SCALE_POS_WEIGHT, 1000, 2500, 5000, 10000]


def out_of_fold_probabilities(raw, train_rows):
    """Score every training row with a model that was not trained on it.

    Each fold refits the encoding maps as well as the model, because a map fitted on rows the
    fold is about to score is the leak this project already found once.
    """
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


def candidate_thresholds(probabilities):
    """Every distinct predicted probability is a candidate -- the counts only change where a
    row crosses over, so a finer grid gains nothing. Capped so the sweep stays quick."""
    distinct = np.unique(np.round(probabilities, 5))
    if len(distinct) > 20000:
        distinct = np.quantile(distinct, np.linspace(0, 1, 20000))
    return distinct


def confusion(target, probabilities, threshold):
    """(caught, false alarms, missed) at one cut-off."""
    flagged = probabilities >= threshold
    caught = int((flagged & (target == 1)).sum())
    return caught, int(flagged.sum()) - caught, int(target.sum()) - caught


def cost_threshold(target, probabilities, cost_ratio):
    """The cut-off minimising `cost_ratio * misses + false alarms`."""
    grid = candidate_thresholds(probabilities)
    costs = [cost_ratio * missed + false_alarms
             for _, false_alarms, missed in
             (confusion(target, probabilities, threshold) for threshold in grid)]
    return float(grid[int(np.argmin(costs))])


def youden_threshold(target, probabilities):
    """The cut-off maximising recall minus false-positive rate."""
    false_positive_rate, recall, thresholds = roc_curve(target, probabilities)
    return float(thresholds[int(np.argmax(recall - false_positive_rate))])


def describe(target, probabilities, threshold):
    caught, false_alarms, missed = confusion(target, probabilities, threshold)
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
    """Replace the RISK_THRESHOLD literal in config.py, leaving its comment intact."""
    source = CONFIG_PATH.read_text(encoding="utf-8")
    updated, replaced = re.subn(r"^RISK_THRESHOLD = [\d.]+$", f"RISK_THRESHOLD = {value}",
                                source, count=1, flags=re.MULTILINE)
    if replaced != 1:
        raise SystemExit("RISK_THRESHOLD assignment not found in config.py -- not written.")
    CONFIG_PATH.write_text(updated, encoding="utf-8")
    print(f"config.py: RISK_THRESHOLD = {value}")


def stamp_model(value):
    """Store the threshold on the booster itself, beside the weights it applies to.

    `serving/app.py` ships as a standalone container -- its Dockerfile copies the app, the
    model and the encoding maps and nothing else -- so it cannot import config.py, and until
    this existed it compared against a hard-coded 0.5 that no amount of editing config.py
    would have changed. Writing it here means the scorer reads the cut-off out of the same
    file it scores with, which is already how that file gets its column order, and for the
    same reason: a second copy is a thing that can disagree.

    Only an attribute is written. The weights, and therefore every probability, are untouched.
    """
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    model.get_booster().set_attr(risk_threshold=str(value))
    model.save_model(MODEL_PATH)
    print(f"{MODEL_PATH.name}: risk_threshold = {value}")


def main(write=False):
    raw = pd.read_parquet(LABELLED_DATASET_PATH)
    train_rows, test_rows = train_test_split(
        raw.index, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=raw[TARGET_COLUMN])

    print(f"deriving on {len(train_rows):,} training rows, {FOLDS}-fold out-of-fold")
    train_target, train_probabilities = out_of_fold_probabilities(raw, train_rows)

    print(f"\ncost ratio sweep (out-of-fold, {int(train_target.sum())} excluded providers)")
    print(f"{'1 miss = N alarms':>18} {'threshold':>10} {'caught':>8} {'flagged':>10}")
    for ratio in COST_RATIOS:
        found = describe(train_target, train_probabilities,
                         cost_threshold(train_target, train_probabilities, ratio))
        print(f"{ratio:>18,} {found['threshold']:>10.4f} "
              f"{found['caught']:>8,} {found['flagged']:>10,}")

    by_cost = cost_threshold(train_target, train_probabilities, SCALE_POS_WEIGHT)
    by_youden = youden_threshold(train_target, train_probabilities)
    print(f"\ncost (ratio {SCALE_POS_WEIGHT}): {by_cost:.4f}    youden: {by_youden:.4f}")

    # The two are averaged rather than one being picked, because they rest on different
    # arguments and land close together. Picking one would quietly discard the other's vote.
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
        found = describe(test_target, test_probabilities, threshold)
        print(f"  {name:<14} threshold {found['threshold']:<8} "
              f"caught {found['caught']}/{found['positives']}  "
              f"flagged {found['flagged']:,}  recall {found['recall']}")

    if write:
        print()
        write_threshold(derived)
        stamp_model(derived)
    return derived


if __name__ == "__main__":
    main(write="--write" in sys.argv)
