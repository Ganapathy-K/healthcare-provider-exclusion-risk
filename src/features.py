"""Turns the labelled NPPES + LEIE dataset into the 16 features the XGBoost model was trained on.

1. Same steps as the modelling notebook: the shipped model and baseline.json stay matched.
2. Target encodings fitted on train rows only: a test row's own label never reaches its feature.
3. FEATURE_COLUMNS is the one column order, checked against the model by check_serving_alignment.

Steps: drop >30% empty -> drop IDs -> years -> drop >1000 values -> encode 4 -> one-hot 3 -> median fill.
"""

import json

import pandas as pd

from config import (ENCODING_MAPS_PATH, HIGH_CARDINALITY_LIMIT, MAX_NULL_FRACTION,
                    MODEL_PATH, TARGET_COLUMN)

# The exact order the model expects. `check_serving_alignment()` verifies it against the model.
FEATURE_COLUMNS = [
    "Entity Type Code",
    "Provider Business Mailing Address State Name",
    "Provider Business Mailing Address Telephone Number",
    "Provider Business Practice Location Address State Name",
    "Healthcare Provider Taxonomy Code_1",
    "Provider License Number State Code_1",
    "Provider Enumeration Year",
    "Last Update Year",
    "Provider Sex Code_F",
    "Provider Sex Code_M",
    "Provider Sex Code_U",
    "Healthcare Provider Primary Taxonomy Switch_1_N",
    "Healthcare Provider Primary Taxonomy Switch_1_Y",
    "Is Sole Proprietor_N",
    "Is Sole Proprietor_X",
    "Is Sole Proprietor_Y",
]

# Dropped: an NPI or an address lets the model memorise one provider instead of learning risk.
IDENTIFIER_COLUMNS = [
    "NPI",
    "Provider Last Name (Legal Name)",
    "Provider First Name",
    "Provider Credential Text",
    "Provider First Line Business Mailing Address",
    "Provider First Line Business Practice Location Address",
    "Provider Business Practice Location Address Telephone Number",
]

# Over 99% of NPPES providers are US-based, so the country code is constant in practice.
NEAR_ZERO_VARIANCE_COLUMNS = [
    "Provider Business Mailing Address Country Code (If outside U.S.)",
    "Provider Business Practice Location Address Country Code (If outside U.S.)",
]

TARGET_ENCODED_COLUMNS = [
    "Healthcare Provider Taxonomy Code_1",
    "Provider Business Mailing Address State Name",
    "Provider Business Practice Location Address State Name",
    "Provider License Number State Code_1",
]

ONE_HOT_COLUMNS = [
    "Provider Sex Code",
    "Healthcare Provider Primary Taxonomy Switch_1",
    "Is Sole Proprietor",
]

DATE_COLUMNS = {
    "Provider Enumeration Date": "Provider Enumeration Year",
    "Last Update Date": "Last Update Year",
}


def prepare_features(providers_raw, encoding_maps=None):
    """Gives (features, target, encoding_maps); pass train-row encoding_maps so test labels never leak in."""
    providers = providers_raw.copy()

    null_fraction = providers.isnull().mean()
    providers.drop(columns=null_fraction[null_fraction > MAX_NULL_FRACTION].index,
                   inplace=True)
    providers.drop(columns=IDENTIFIER_COLUMNS, inplace=True, errors="ignore")

    for source_column, year_column in DATE_COLUMNS.items():
        providers[year_column] = pd.to_datetime(
            providers[source_column]).dt.year.astype("Int64")
    providers.drop(columns=list(DATE_COLUMNS), inplace=True, errors="ignore")
    providers["Entity Type Code"] = providers["Entity Type Code"].astype("category")

    cardinality = providers.select_dtypes(include="object").nunique()
    providers.drop(columns=cardinality[cardinality > HIGH_CARDINALITY_LIMIT].index,
                   inplace=True)
    providers.drop(columns=NEAR_ZERO_VARIANCE_COLUMNS, inplace=True, errors="ignore")

    # Target encoding: each category becomes its mean exclusion rate.
    if encoding_maps is None:
        encoding_maps = {}
        for column in TARGET_ENCODED_COLUMNS:
            mapping = providers.groupby(column)[TARGET_COLUMN].mean()
            encoding_maps[column] = mapping.to_dict()
            providers[column] = providers[column].map(mapping)
    else:
        for column in TARGET_ENCODED_COLUMNS:
            providers[column] = providers[column].map(encoding_maps[column])

    providers = pd.get_dummies(providers, columns=ONE_HOT_COLUMNS)

    target = providers[TARGET_COLUMN]
    features = providers.drop(columns=[TARGET_COLUMN])
    features["Entity Type Code"] = features["Entity Type Code"].astype("float")
    features = features.fillna(features.median(numeric_only=True))

    return features, target, encoding_maps


def fit_encoding_maps(providers_train_raw):
    """Gives the 4 target encodings from train rows only, so a test row's own label never reaches its feature."""
    return {
        column: providers_train_raw.groupby(column)[TARGET_COLUMN].mean().to_dict()
        for column in TARGET_ENCODED_COLUMNS
    }


def load_encoding_maps():
    """The target-encoding maps the deployed model was trained with."""
    return json.loads(ENCODING_MAPS_PATH.read_text(encoding="utf-8"))


def encode_provider_record(record, encoding_maps=None):
    """Gives the 16 model features for one provider row; an unseen category gets 0.0, meaning no risk history."""
    maps = encoding_maps if encoding_maps is not None else load_encoding_maps()

    def target_encode(column, value):
        return maps[column].get(str(value), 0.0)

    row = {
        "Entity Type Code": int(record["Entity Type Code"])
        if pd.notna(record["Entity Type Code"]) else 0,
        "Provider Business Mailing Address State Name": target_encode(
            "Provider Business Mailing Address State Name",
            record["Provider Business Mailing Address State Name"]),
        "Provider Business Mailing Address Telephone Number": float(
            record["Provider Business Mailing Address Telephone Number"])
        if pd.notna(record["Provider Business Mailing Address Telephone Number"]) else 0.0,
        "Provider Business Practice Location Address State Name": target_encode(
            "Provider Business Practice Location Address State Name",
            record["Provider Business Practice Location Address State Name"]),
        "Healthcare Provider Taxonomy Code_1": target_encode(
            "Healthcare Provider Taxonomy Code_1",
            record["Healthcare Provider Taxonomy Code_1"]),
        "Provider License Number State Code_1": target_encode(
            "Provider License Number State Code_1",
            record["Provider License Number State Code_1"]),
        "Provider Enumeration Year": pd.to_datetime(record["Provider Enumeration Date"]).year,
        "Last Update Year": pd.to_datetime(record["Last Update Date"]).year,
        "Provider Sex Code_F": int(record["Provider Sex Code"] == "F"),
        "Provider Sex Code_M": int(record["Provider Sex Code"] == "M"),
        "Provider Sex Code_U": int(record["Provider Sex Code"] == "U"),
        "Healthcare Provider Primary Taxonomy Switch_1_N": int(
            record["Healthcare Provider Primary Taxonomy Switch_1"] == "N"),
        "Healthcare Provider Primary Taxonomy Switch_1_Y": int(
            record["Healthcare Provider Primary Taxonomy Switch_1"] == "Y"),
        "Is Sole Proprietor_N": int(record["Is Sole Proprietor"] == "N"),
        "Is Sole Proprietor_X": int(record["Is Sole Proprietor"] == "X"),
        "Is Sole Proprietor_Y": int(record["Is Sole Proprietor"] == "Y"),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def check_serving_alignment():
    """Gives (matches, model columns), because XGBoost reads columns by position and never warns on a wrong order."""
    import xgboost as xgb

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    trained_columns = list(model.get_booster().feature_names)
    return trained_columns == FEATURE_COLUMNS, trained_columns


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    from config import LABELLED_DATASET_PATH

    aligned, model_columns = check_serving_alignment()
    print(f"model column order matches: {aligned}")
    if not aligned:
        print(f"  model has {len(model_columns)}: {model_columns}")
        print(f"  features has {len(FEATURE_COLUMNS)}: {FEATURE_COLUMNS}")

    raw = pd.read_parquet(LABELLED_DATASET_PATH)
    features, target, maps = prepare_features(raw)
    print(f"\nraw      : {raw.shape}")
    print(f"features : {features.shape}")
    print(f"positives: {int(target.sum())} of {len(target)}")
    print(f"columns match FEATURE_COLUMNS: {list(features.columns) == FEATURE_COLUMNS}")
    for column, mapping in maps.items():
        print(f"  {column}: {len(mapping)} categories")

    saved = json.loads(ENCODING_MAPS_PATH.read_text())
    same = all(len(saved.get(column, {})) == len(mapping) for column, mapping in maps.items())
    print(f"encoding maps match the saved serving copy: {same}")
