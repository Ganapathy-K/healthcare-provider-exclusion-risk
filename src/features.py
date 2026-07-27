"""Turn the labelled NPPES+LEIE dataset into the 16 features the model was trained on.

This is a FAITHFUL reproduction of notebook 03's preparation, not an improved one. It exists
so the refactor can be checked against the notebook's own numbers -- a characterization test
is only meaningful if it reproduces current behaviour exactly, including the parts that are
wrong. Two of those are called out below and deliberately left alone.

The order of operations matters and is the notebook's:
  drop columns >30% null -> drop identifiers and free text -> derive year columns ->
  drop high-cardinality (>1000 distinct) and near-zero-variance columns ->
  target-encode four categoricals -> one-hot three low-cardinality ones ->
  fill remaining nulls with the column median.

⚠️ KNOWN DEFECT 1 — TARGET LEAKAGE IN THE ENCODING MAPS. The four target encodings are fit
on the WHOLE dataset, before the train/test split. Each category's value is the mean of
`excluded` over every row, so a test row's own label contributes to the feature it is later
scored on. Test metrics are therefore optimistic by an unknown amount. The fix is to fit the
maps on the training split only, but doing that CHANGES the model, so it must happen after
the baseline is frozen and be reported as a new number -- not folded in silently.

⚠️ KNOWN DEFECT 2 — THE COLUMN ORDER LIVES IN TWO PLACES. `FEATURE_COLUMNS` here and in
`serving/app.py` are separate literal lists. XGBoost scores whatever it is handed in
positional order, so if the two ever disagree the deployed service predicts on misaligned
columns and raises nothing at all. `check_serving_alignment()` below exists to catch that,
and should stay until there is exactly one list.
"""

import json

import pandas as pd

from config import (ENCODING_MAPS_PATH, HIGH_CARDINALITY_LIMIT, MAX_NULL_FRACTION,
                    TARGET_COLUMN)

# The exact order the model expects. Must match serving/app.py -- see defect 2 above.
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

# Identifiers and free text: an NPI is unique per provider and a street address is nearly so,
# and a model given either can memorise individuals instead of learning risk.
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
    """Reproduce notebook 03's preparation. Returns (X, y, encoding_maps).

    Pass `encoding_maps` to APPLY maps fitted elsewhere instead of fitting new ones on this
    data. That is how the leakage in defect 1 is avoided: fit on the training split, then
    apply those maps to the test split, so a test row's own label never reaches the feature
    it is scored on. Omit it and the original leaky behaviour is reproduced exactly, which is
    what `baseline.py` needs in order to keep describing the shipped pipeline.
    """
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

    # Target encoding: each category becomes its mean exclusion rate. Fitted here only when
    # no maps are supplied -- see defect 1, and `fit_encoding_maps` for the leak-free path.
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
    """Fit the four target encodings on TRAINING ROWS ONLY -- the fix for defect 1.

    Fitted on the whole dataset, a category's value is the mean of `excluded` across every
    row, so each test row contributes its own answer to the number it is later scored on.
    Rare categories are the worst case: a taxonomy code appearing twice, once excluded, gets
    the value 0.5 largely BECAUSE of the row being predicted.

    Categories that appear only in the test split are left unmapped and become NaN, which the
    median fill downstream handles. That is the honest outcome -- at serving time a genuinely
    new taxonomy code has no history either.
    """
    return {
        column: providers_train_raw.groupby(column)[TARGET_COLUMN].mean().to_dict()
        for column in TARGET_ENCODED_COLUMNS
    }


def load_encoding_maps():
    """The target-encoding maps the deployed model was trained with."""
    return json.loads(ENCODING_MAPS_PATH.read_text(encoding="utf-8"))


def encode_provider_record(record, encoding_maps=None):
    """Encode ONE raw NPPES row into the 16 model features, for scoring a single provider.

    This is the row-at-a-time counterpart to `prepare_features`, which works on the whole
    frame. Notebook 05 and `serving/app.py` each carried their own copy of this logic -- three
    implementations of one encoding, any of which could drift from the other two without a
    single error being raised, because XGBoost scores by position and checks nothing.

    Unknown categories fall back to 0.0, matching serving. That is a real modelling decision
    worth naming: 0.0 is the exclusion rate of a category never seen in training, i.e. "no
    evidence of risk", which is the safe direction for a queue that prioritises review.
    """
    maps = encoding_maps if encoding_maps is not None else load_encoding_maps()

    def target_encode(column, value):
        return maps[column].get(str(value), 0.0)

    def as_float(value, default=0.0):
        return float(value) if pd.notna(value) else default

    row = {
        "Entity Type Code": int(record["Entity Type Code"])
        if pd.notna(record["Entity Type Code"]) else 0,
        "Provider Business Mailing Address State Name": target_encode(
            "Provider Business Mailing Address State Name",
            record["Provider Business Mailing Address State Name"]),
        "Provider Business Mailing Address Telephone Number": as_float(
            record["Provider Business Mailing Address Telephone Number"]),
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
    """Fail loudly if serving/app.py's column order has drifted from this one.

    XGBoost validates nothing about column NAMES -- it scores by position. Misaligned columns
    produce confident, silent nonsense, which is the worst failure mode a deployed scorer has.
    """
    serving_source = (ENCODING_MAPS_PATH.parent / "app.py").read_text(encoding="utf-8")
    start = serving_source.index("FEATURE_COLUMNS = [")
    end = serving_source.index("]", start)
    serving_columns = [
        line.strip().strip(',').strip('"')
        for line in serving_source[start:end].splitlines()[1:]
        if line.strip()
    ]
    return serving_columns == FEATURE_COLUMNS, serving_columns


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    from config import LABELLED_DATASET_PATH

    aligned, serving_columns = check_serving_alignment()
    print(f"serving/app.py column order matches: {aligned}")
    if not aligned:
        print(f"  serving has {len(serving_columns)}: {serving_columns}")
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
