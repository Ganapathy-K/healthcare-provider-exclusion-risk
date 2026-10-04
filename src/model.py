"""Train, evaluate and save the XGBoost model that scores each provider's exclusion risk."""

import json
import sys

import pandas as pd
import xgboost as xgb
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

from config import (ENCODING_MAPS_PATH, LABELLED_DATASET_PATH, LEARNING_RATE, MAX_DEPTH,
                    MODEL_PATH, N_ESTIMATORS, RANDOM_STATE, RISK_THRESHOLD,
                    SCALE_POS_WEIGHT, TARGET_COLUMN, TEST_SIZE)
from features import FEATURE_COLUMNS, fit_encoding_maps, prepare_features


def build_model(scale_pos_weight=SCALE_POS_WEIGHT):
    """XGBoost with scale_pos_weight 422, so a missed excluded provider costs 422 false alarms."""
    return xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
    )


def evaluate(model, features_test, target_test, threshold=RISK_THRESHOLD):
    """Recall and the number of providers flagged, on the test rows."""
    probabilities = model.predict_proba(features_test[FEATURE_COLUMNS])[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "recall": float(recall_score(target_test, predictions)),
        "flagged": int(predictions.sum()),
    }


def train(save=False):
    """Stratified 80/20 split on the raw rows, encoding maps fitted on the training rows only."""
    raw = pd.read_parquet(LABELLED_DATASET_PATH)
    train_rows, test_rows = train_test_split(
        raw.index, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=raw[TARGET_COLUMN])
    encoding_maps = fit_encoding_maps(raw.loc[train_rows])
    features, target, _ = prepare_features(raw, encoding_maps=encoding_maps)
    features_train, target_train = features.loc[train_rows], target.loc[train_rows]
    features_test, target_test = features.loc[test_rows], target.loc[test_rows]

    model = build_model()
    model.fit(features_train[FEATURE_COLUMNS], target_train)
    scores = evaluate(model, features_test, target_test)

    if save:
        model.save_model(MODEL_PATH)
        # The encoding maps ship with the model trained on them; the agent reads both.
        ENCODING_MAPS_PATH.write_text(json.dumps(encoding_maps), encoding="utf-8")

    return model, scores, (features_test, target_test)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    saving = "--save" in sys.argv

    model, scores, (features_test, target_test) = train(save=saving)

    positives = int(target_test.sum())
    print(f"test rows : {len(target_test):,}, {positives} excluded providers")
    print(f"recall    : {scores['recall']:.4f}")
    print(f"flagged   : {scores['flagged']:,}")
    print(f"caught {round(scores['recall'] * positives)} of {positives} excluded providers")

    if saving:
        print(f"\nsaved -> {MODEL_PATH}")
        print(f"        -> {ENCODING_MAPS_PATH}")
    else:
        print("\nnothing written. re-run with --save to replace the model files")
