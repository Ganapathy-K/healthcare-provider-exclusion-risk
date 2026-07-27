"""Train, evaluate and save the provider-exclusion classifier.

Extracted from notebook 03. The notebook's chosen configuration is the one below --
XGBoost with `scale_pos_weight`, which is how the model is told that excluded providers are
rare. Without it a classifier scores extremely well by answering "not excluded" to everything,
since 99.76% of the time that is correct, and quietly misses most of what it exists to find.

⚠️ WHY THIS FILE WAS WRITTEN. `baseline.py` scored the deployed `serving/model.ubj` and found
it had **no** `scale_pos_weight` and a learning rate of 0.3, not 0.1 -- it is not the
configuration notebook 03 selected or the README describes. An earlier, unweighted run was
copied out of MLflow in notebook 06 and shipped. Measured recall at threshold 0.5: **0.177**.

Only the training configuration is corrected here. The target leakage in the encoding maps
(see features.py) is a separate defect and is deliberately NOT touched in the same change --
two fixes at once means neither can be attributed.
"""

import json
import sys

import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from config import (ENCODING_MAPS_PATH, EVAL_METRIC, LABELLED_DATASET_PATH, LEARNING_RATE, MAX_DEPTH,
                    MODEL_PATH, N_ESTIMATORS, RANDOM_STATE, RISK_THRESHOLD,
                    SCALE_POS_WEIGHT, TARGET_COLUMN, TEST_SIZE)
from features import FEATURE_COLUMNS, fit_encoding_maps, prepare_features


def build_model(scale_pos_weight=SCALE_POS_WEIGHT):
    """The notebook-03 configuration. `scale_pos_weight` is the one that matters here.

    `aucpr` rather than `auc` as the eval metric for the same reason: with 1 positive per 422
    negatives, precision-recall describes the minority class the model is actually for, while
    ROC hides it behind a very large true-negative count.
    """
    return xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        random_state=RANDOM_STATE,
        eval_metric=EVAL_METRIC,
        scale_pos_weight=scale_pos_weight,
    )


def split(features, target):
    """Notebook 03's split: stratified, 20% test, seed 42. Stratify is not optional --
    an unstratified 20% of a 1:422 problem can differ in positive count by enough to move
    every metric on its own."""
    return train_test_split(features, target, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=target)


def evaluate(model, features_test, target_test, threshold=RISK_THRESHOLD):
    probabilities = model.predict_proba(features_test[FEATURE_COLUMNS])[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "recall": float(recall_score(target_test, predictions)),
        "precision": float(precision_score(target_test, predictions, zero_division=0)),
        "f1": float(f1_score(target_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target_test, probabilities)),
        "average_precision": float(average_precision_score(target_test, probabilities)),
        "flagged": int(predictions.sum()),
    }


def train(save=False, leak_free=False):
    """Train on the notebook's split. `leak_free=True` fits the target encodings on the
    training rows only, instead of on every row (see features.py, defect 1).

    The split is taken on the RAW frame first in the leak-free path, because the maps must be
    fitted before the features that use them exist. The row membership is identical either
    way -- same seed, same stratification -- so the two runs remain comparable.
    """
    raw = pd.read_parquet(LABELLED_DATASET_PATH)

    if leak_free:
        train_rows, test_rows = train_test_split(
            raw.index, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=raw[TARGET_COLUMN])
        encoding_maps = fit_encoding_maps(raw.loc[train_rows])
        features, target, _ = prepare_features(raw, encoding_maps=encoding_maps)
        features_train, target_train = features.loc[train_rows], target.loc[train_rows]
        features_test, target_test = features.loc[test_rows], target.loc[test_rows]
    else:
        features, target, encoding_maps = prepare_features(raw)
        features_train, features_test, target_train, target_test = split(features, target)

    model = build_model()
    model.fit(features_train[FEATURE_COLUMNS], target_train)
    scores = evaluate(model, features_test, target_test)

    if save:
        # Keep the superseded artefacts. They are the only proof of what was actually
        # deployed, and each recorded baseline describes one of them.
        if MODEL_PATH.exists():
            MODEL_PATH.with_name("model_superseded.ubj").write_bytes(MODEL_PATH.read_bytes())
        if ENCODING_MAPS_PATH.exists():
            ENCODING_MAPS_PATH.with_name("encoding_maps_superseded.json").write_text(
                ENCODING_MAPS_PATH.read_text(encoding="utf-8"), encoding="utf-8")

        model.save_model(MODEL_PATH)

        # The maps MUST ship with the model that was trained on them. serving/app.py looks up
        # every categorical value in this file, so a model trained on train-only means paired
        # with full-data means would be scored on numbers it never saw -- silently, since the
        # column names and order would still line up perfectly.
        ENCODING_MAPS_PATH.write_text(json.dumps(encoding_maps), encoding="utf-8")

    return model, scores, (features_test, target_test)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    saving = "--save" in sys.argv
    leaky = "--leaky" in sys.argv          # reproduce the old behaviour for comparison only

    model, scores, (features_test, target_test) = train(save=saving, leak_free=not leaky)

    positives = int(target_test.sum())
    print(f"test set: {len(target_test):,} rows, {positives} excluded providers")
    print(f"encodings fitted on: {'every row (leaky)' if leaky else 'training rows only'}\n")
    for key in ("recall", "precision", "f1", "roc_auc", "average_precision"):
        print(f"  {key:<20}{scores[key]:.4f}")
    print(f"  {'providers flagged':<20}{scores['flagged']:,}")
    print(f"\ncaught {round(scores['recall'] * positives)} of {positives} excluded providers")

    if saving:
        print(f"\nsaved -> {MODEL_PATH}")
        print(f"        -> {ENCODING_MAPS_PATH}")
        print("previous artefacts kept as model_superseded.ubj / "
              "encoding_maps_superseded.json")
    else:
        print("\nnothing written. re-run with --save to replace the serving artefacts")
