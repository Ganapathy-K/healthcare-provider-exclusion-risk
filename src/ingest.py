"""Build the labelled dataset: NPPES providers, tagged with whether the OIG has excluded them.

Extracted from notebook 01. Two public US datasets:

  NPPES  the national registry of every healthcare provider and their NPI number
  LEIE   the OIG's List of Excluded Individuals/Entities -- providers already excluded

The label is the join: a provider is `excluded = 1` if their NPI appears in the LEIE. There
is no free lunch in that definition and it is worth stating plainly -- this predicts *who
looks like the providers already caught*, which is not the same as *who is committing fraud*.
Anyone the OIG has never investigated is labelled 0 whatever they have done.

NPPES ships as a ~1 GB zip and is read directly out of it rather than extracted: unzipping
costs several gigabytes on disk to produce a file that is read once, and pandas can stream
from the archive member. Only the first 500,000 rows are taken, which is what every number
recorded in docs/baseline.json describes.
"""

import sys

import pandas as pd

from config import (LABELLED_DATASET_PATH, LEIE_PATH, NPPES_FILENAME, NPPES_SAMPLE_ROWS,
                    NPPES_ZIP_PATH, PROCESSED_DIR, TARGET_COLUMN)


def load_leie():
    """The full exclusions list. Small enough to read whole.

    latin-1, not utf-8: the OIG file contains bytes that are not valid utf-8, and pandas
    raises rather than guessing. latin-1 maps every byte to something, so it never fails --
    which is the right trade for a file whose names occasionally carry accents.
    """
    return pd.read_csv(LEIE_PATH, encoding="latin-1", low_memory=False)


def load_nppes(rows=NPPES_SAMPLE_ROWS):
    """Read the provider registry straight out of the zip, without extracting it."""
    import zipfile

    with zipfile.ZipFile(NPPES_ZIP_PATH) as archive:
        with archive.open(NPPES_FILENAME) as member:
            return pd.read_csv(member, nrows=rows, low_memory=False)


def excluded_npis(leie):
    """The NPIs the OIG has excluded.

    NPI 0 is dropped: the LEIE uses it as a placeholder for records where no NPI was
    recorded, so keeping it would label every NPPES row whose NPI failed to parse as
    excluded. It is a missing value wearing a number's clothing.
    """
    return set(leie.loc[leie["NPI"] != 0, "NPI"])


def build_labelled_dataset(save=False):
    """Join the two sources into the labelled dataset. Returns (dataframe, report)."""
    leie = load_leie()
    nppes = load_nppes()

    excluded = excluded_npis(leie)
    nppes[TARGET_COLUMN] = nppes["NPI"].isin(excluded).astype(int)

    matched = excluded & set(nppes["NPI"])
    report = {
        "leie_rows": int(len(leie)),
        "leie_with_npi": int(len(excluded)),
        "nppes_rows": int(len(nppes)),
        "matched_npis": int(len(matched)),
        "match_rate": len(matched) / len(excluded) if excluded else 0.0,
        "positives": int(nppes[TARGET_COLUMN].sum()),
        "negatives": int((nppes[TARGET_COLUMN] == 0).sum()),
    }

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        nppes.to_parquet(LABELLED_DATASET_PATH, index=False)

    return nppes, report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    saving = "--save" in sys.argv

    dataset, report = build_labelled_dataset(save=saving)

    print(f"LEIE rows            : {report['leie_rows']:,}")
    print(f"  with a valid NPI   : {report['leie_with_npi']:,}")
    print(f"NPPES sample rows    : {report['nppes_rows']:,}")
    print(f"matched NPIs         : {report['matched_npis']:,}")
    print(f"match rate LEIE->NPPES: {report['match_rate']:.1%}")
    print(f"\nexcluded (1)         : {report['positives']:,}")
    print(f"not excluded (0)     : {report['negatives']:,}")
    print(f"class ratio          : {report['positives'] / len(dataset):.4%}")

    if saving:
        print(f"\nsaved -> {LABELLED_DATASET_PATH}")
    else:
        print(f"\nnothing written. re-run with --save to rebuild "
              f"{LABELLED_DATASET_PATH.name}")
        print("⚠️  rebuilding changes nothing only if NPPES and LEIE are the same files as "
              "before; run `python src/baseline.py --check` afterwards.")
