"""Build the labelled dataset: NPPES providers, excluded = 1 when their NPI is in the LEIE."""

import sys

import pandas as pd

from config import (LABELLED_DATASET_PATH, LEIE_PATH, LOOKUP_COLUMNS, NPPES_FILENAME,
                    NPPES_SAMPLE_ROWS, NPPES_ZIP_PATH, PROCESSED_DIR, PROVIDER_LOOKUP_PATH,
                    TARGET_COLUMN)


def load_leie():
    """Read the whole LEIE file (latin-1, because it has bytes that are not valid utf-8)."""
    return pd.read_csv(LEIE_PATH, encoding="latin-1", low_memory=False)


def load_nppes(rows=NPPES_SAMPLE_ROWS):
    """Read the first 500,000 NPPES rows straight from the zip."""
    import zipfile

    with zipfile.ZipFile(NPPES_ZIP_PATH) as archive:
        with archive.open(NPPES_FILENAME) as member:
            return pd.read_csv(member, nrows=rows, low_memory=False)


def excluded_npis(leie):
    """The set of excluded NPIs, without NPI 0 (the LEIE's placeholder for a missing NPI)."""
    return set(leie.loc[leie["NPI"] != 0, "NPI"])


def build_labelled_dataset(save=False):
    """Set excluded = 1 where the NPI is in the LEIE set. Returns (dataframe, report)."""
    leie = load_leie()
    nppes = load_nppes()

    excluded = excluded_npis(leie)
    nppes[TARGET_COLUMN] = nppes["NPI"].isin(excluded).astype(int)

    report = {
        "leie_rows": int(len(leie)),
        "leie_with_npi": int(len(excluded)),
        "nppes_rows": int(len(nppes)),
        "positives": int(nppes[TARGET_COLUMN].sum()),
    }

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        nppes.to_parquet(LABELLED_DATASET_PATH, index=False)
        # The slim copy the agent ships with: same rows, only the columns the scorer reads.
        nppes[LOOKUP_COLUMNS].to_parquet(PROVIDER_LOOKUP_PATH, index=False)

    return nppes, report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    saving = "--save" in sys.argv

    _, report = build_labelled_dataset(save=saving)

    print(f"LEIE rows          : {report['leie_rows']:,}")
    print(f"  distinct NPIs    : {report['leie_with_npi']:,}")
    print(f"NPPES rows         : {report['nppes_rows']:,}")
    print(f"excluded (1)       : {report['positives']:,}")

    if saving:
        print(f"\nsaved -> {LABELLED_DATASET_PATH}")
    else:
        print("\nnothing written. re-run with --save to rebuild the dataset")
