"""Preprocessing: loading the dataset, cleaning it and splitting into train/test sets. (Not implemented yet)"""
"""
preprocessing.py  -  Phase 2: dataset preparation

What this file does:
  1. Gets the Genomic Benchmarks dataset "demo_coding_vs_intergenomic_seqs"
     (200 bp DNA sequences, coding vs intergenomic).
  2. Cleans it (length check, A/C/G/T check, duplicates, label conflicts).
  3. Makes a simple stratified train/test split (80% / 20%).
  4. Saves everything into data/processed/.
  5. Gives easy functions to reload the saved data later.

How to run (from the DNA_Intelligence folder, with .venv active):
    python -m src.preprocessing

If the automatic download does not work, use a folder you downloaded yourself:
    python -m src.preprocessing --source-folder /path/to/demo_coding_vs_intergenomic_seqs
"""

import argparse
import json
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Settings (all in one place so they are easy to find)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATASET_NAME = "demo_coding_vs_intergenomic_seqs"
BASE_URL = (
    "https://huggingface.co/datasets/katielink/genomic-benchmarks/resolve/main/"
    + DATASET_NAME
)

CLASS_LABELS = {"coding": 1, "intergenomic": 0}   # label numbers used in this project
ORIGINAL_SPLITS = ["train", "test"]               # folders in the original dataset

SEQ_LENGTH = 200
TEST_SIZE = 0.2
RANDOM_STATE = 42

PROCESSED_CSV = PROCESSED_DIR / "dna_dataset.csv"
SPLIT_INFO_JSON = PROCESSED_DIR / "split_info.json"
DATASET_INFO_MD = PROCESSED_DIR / "DATASET_INFO.md"


# ---------------------------------------------------------------------------
# Step 1: get the raw data
# ---------------------------------------------------------------------------
def download_raw_files():
    """Downloads the 4 small .csv.gz files into data/raw/ (skips files that exist)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for split in ORIGINAL_SPLITS:
        for class_name in CLASS_LABELS:
            file_name = f"{class_name}_seqs.csv.gz"
            local_path = RAW_DIR / f"{split}_{file_name}"
            if local_path.exists():
                print(f"Already downloaded: {local_path.name}")
                continue
            url = f"{BASE_URL}/{split}/{file_name}"
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, local_path)


def _looks_like_dna(text):
    """True if the text looks like a DNA sequence (only A, C, G, T, N)."""
    text = str(text).strip().upper()
    return len(text) >= 20 and set(text) <= set("ACGTN")


def read_sequences_from_csv(path):
    """
    Reads one .csv.gz file and returns a list of sequences.
    It finds the column that contains DNA, so it works with or without a header.
    """
    table = pd.read_csv(path, header=None, dtype=str)

    sequence_column = None
    for column in table.columns:
        sample = table[column].dropna().iloc[1:21]   # skip row 0 (could be a header)
        if len(sample) > 0 and all(_looks_like_dna(s) for s in sample):
            sequence_column = column
            break
    if sequence_column is None:
        raise ValueError(f"Could not find a DNA column in {path.name}")

    values = table[sequence_column].dropna().astype(str).str.strip().str.upper()
    if not _looks_like_dna(values.iloc[0]):   # first row was a header like "seq"
        values = values.iloc[1:]
    return values.tolist()


def load_raw_dataframe():
    """Reads the 4 downloaded files into one table."""
    parts = []
    for split in ORIGINAL_SPLITS:
        for class_name, label in CLASS_LABELS.items():
            path = RAW_DIR / f"{split}_{class_name}_seqs.csv.gz"
            sequences = read_sequences_from_csv(path)
            parts.append(pd.DataFrame({
                "sequence": sequences,
                "label": label,
                "label_name": class_name,
                "original_split": split,
            }))
    return pd.concat(parts, ignore_index=True)


def load_from_txt_folders(root):
    """
    Alternative: reads the ORIGINAL dataset format, which is
        root/train/<class folder>/*.txt   and   root/test/<class folder>/*.txt
    with one sequence per .txt file (this is what the 'genomic-benchmarks'
    package and the Zenodo zip contain).
    """
    root = Path(root)
    parts = []
    for split in ORIGINAL_SPLITS:
        split_folder = root / split
        if not split_folder.exists():
            raise FileNotFoundError(f"Folder not found: {split_folder}")
        for class_folder in sorted(p for p in split_folder.iterdir() if p.is_dir()):
            name = class_folder.name.lower()
            if "intergenomic" in name:
                class_name = "intergenomic"
            elif "coding" in name:
                class_name = "coding"
            else:
                continue
            sequences = [f.read_text().strip().upper() for f in sorted(class_folder.glob("*.txt"))]
            parts.append(pd.DataFrame({
                "sequence": sequences,
                "label": CLASS_LABELS[class_name],
                "label_name": class_name,
                "original_split": split,
            }))
    if not parts:
        raise ValueError(f"No coding/intergenomic folders with .txt files found in {root}")
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 2: clean the data
# ---------------------------------------------------------------------------
def clean_dataframe(df):
    """
    Removes bad rows and returns (clean_table, report_dictionary).
    Order of cleaning: wrong length -> non A/C/G/T -> conflicting labels -> duplicates.
    """
    df = df.copy()
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    report = {"rows_loaded": int(len(df))}

    # 1. Keep only sequences of exactly 200 bp
    keep = df["sequence"].str.len() == SEQ_LENGTH
    report["removed_wrong_length"] = int((~keep).sum())
    df = df[keep]

    # 2. Keep only sequences made of A, C, G, T (removes N and other symbols)
    keep = df["sequence"].str.fullmatch(r"[ACGT]+")
    report["removed_non_ACGT"] = int((~keep).sum())
    df = df[keep]

    # 3. Remove sequences that appear with BOTH labels (we cannot know the truth)
    labels_per_sequence = df.groupby("sequence")["label"].nunique()
    conflicting = labels_per_sequence[labels_per_sequence > 1].index
    keep = ~df["sequence"].isin(conflicting)
    report["removed_conflicting_labels"] = int((~keep).sum())
    df = df[keep]

    # 4. Remove exact duplicates (keeps the first copy).
    #    This also stops the same sequence from landing in both train and test.
    before = len(df)
    df = df.drop_duplicates(subset="sequence", keep="first")
    report["removed_exact_duplicates"] = int(before - len(df))

    df = df.reset_index(drop=True)
    report["rows_after_cleaning"] = int(len(df))
    report["coding_after_cleaning"] = int((df["label_name"] == "coding").sum())
    report["intergenomic_after_cleaning"] = int((df["label_name"] == "intergenomic").sum())
    return df, report


# ---------------------------------------------------------------------------
# Step 3: train/test split
# ---------------------------------------------------------------------------
def make_split(df):
    """
    Adds a 'split' column ('train' or 'test').
    Stratified = both classes keep the same proportion in train and test.
    random_state is fixed so everyone gets the same split.
    """
    df = df.copy()
    _, test_index = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    df["split"] = "train"
    df.loc[test_index, "split"] = "test"
    return df


def _count_table(df):
    """Counts sequences per split and class."""
    counts = {}
    for split in ["train", "test"]:
        part = df[df["split"] == split]
        counts[split] = {
            "coding": int((part["label_name"] == "coding").sum()),
            "intergenomic": int((part["label_name"] == "intergenomic").sum()),
            "total": int(len(part)),
        }
    return counts


# ---------------------------------------------------------------------------
# Step 4: save everything
# ---------------------------------------------------------------------------
def save_processed(df, report, source_text):
    """Saves the CSV, the split info (JSON) and the dataset documentation (Markdown)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    columns = ["sequence", "label", "label_name", "original_split", "split"]
    df[columns].to_csv(PROCESSED_CSV, index=False)

    counts = _count_table(df)
    split_info = {
        "dataset": DATASET_NAME,
        "created_on": str(date.today()),
        "source_used": source_text,
        "split_method": "stratified train_test_split on the cleaned data",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "label_meaning": {"1": "coding", "0": "intergenomic"},
        "sequence_length_bp": SEQ_LENGTH,
        "counts": counts,
        "cleaning_report": report,
    }
    with open(SPLIT_INFO_JSON, "w") as f:
        json.dump(split_info, f, indent=2)

    write_dataset_info(report, counts, source_text)


def write_dataset_info(report, counts, source_text):
    """Writes data/processed/DATASET_INFO.md using the REAL numbers from this run."""
    lines = [
        "# Dataset information",
        "",
        "## Name and source",
        f"- **Dataset:** Genomic Benchmarks - `{DATASET_NAME}`",
        "- **Paper:** Gresova K. et al., \"Genomic benchmarks: a collection of datasets for "
        "genomic sequence classification\", BMC Genomic Data 24, 25 (2023).",
        "- **Code / package:** https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks",
        f"- **Files used in this project:** {source_text}",
        f"- **Prepared on:** {date.today()}",
        "",
        "## What the data is",
        f"- Every sequence is exactly **{SEQ_LENGTH} bp** long (fixed length).",
        "- Two classes: **coding** (label 1) and **intergenomic** (label 0).",
        "- Intergenomic is used here as the 'non-coding' class, following the dataset's naming.",
        "",
        "## How the labels were obtained",
        "- The labels come from the dataset authors, not from us. The original dataset is "
        "stored as one group of sequences per class, and the class name is the label.",
        "- The authors describe the 'demo' datasets as computationally generated for their "
        "benchmark collection (see the paper above for the exact procedure).",
        "- We did not check the labels experimentally.",
        "",
        "## Filtering done in this project",
        f"- Rows loaded: **{report['rows_loaded']}**",
        f"- Removed, length not {SEQ_LENGTH} bp: **{report['removed_wrong_length']}**",
        f"- Removed, characters other than A/C/G/T: **{report['removed_non_ACGT']}**",
        f"- Removed, same sequence with both labels: **{report['removed_conflicting_labels']}**",
        f"- Removed, exact duplicate sequences: **{report['removed_exact_duplicates']}**",
        f"- **Rows kept: {report['rows_after_cleaning']}**",
        "",
        "## Final class counts (after cleaning)",
        f"- Coding sequences: **{report['coding_after_cleaning']}**",
        f"- Intergenomic (non-coding) sequences: **{report['intergenomic_after_cleaning']}**",
        "",
        "## Train / test split",
        f"- Method: stratified split, test size {int(TEST_SIZE * 100)}%, random_state={RANDOM_STATE}.",
        "- The split is stored in the `split` column of `dna_dataset.csv`. "
        "The original benchmark split is kept in the `original_split` column.",
        "",
        "| Split | Coding | Intergenomic | Total |",
        "|---|---|---|---|",
        f"| train | {counts['train']['coding']} | {counts['train']['intergenomic']} | {counts['train']['total']} |",
        f"| test | {counts['test']['coding']} | {counts['test']['intergenomic']} | {counts['test']['total']} |",
        "",
        "## Limitations (please read)",
        "- This is a benchmark of short 200 bp fragments. A model trained on it is a "
        "learning exercise, not a gene finder.",
        "- Results should not be assumed to hold for other lengths, other species or real "
        "genome annotation.",
    ]
    with open(DATASET_INFO_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Step 5: reload the processed data (used by later phases)
# ---------------------------------------------------------------------------
def load_processed_data():
    """Returns the full processed table (sequence, label, label_name, original_split, split)."""
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError(
            "Processed data not found. Run this first:  python -m src.preprocessing"
        )
    return pd.read_csv(PROCESSED_CSV)


def load_train_test():
    """
    Returns X_train, X_test, y_train, y_test
    (X = list of DNA sequences, y = numpy array of labels: 1 = coding, 0 = intergenomic).
    """
    df = load_processed_data()
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    return (
        train["sequence"].tolist(),
        test["sequence"].tolist(),
        train["label"].to_numpy(),
        test["label"].to_numpy(),
    )


def load_split_info():
    """Returns the saved split information as a dictionary."""
    if not SPLIT_INFO_JSON.exists():
        raise FileNotFoundError(
            "Split info not found. Run this first:  python -m src.preprocessing"
        )
    with open(SPLIT_INFO_JSON) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------
def run_pipeline(source_folder=None):
    if source_folder:
        print(f"Reading .txt files from: {source_folder}")
        df = load_from_txt_folders(source_folder)
        source_text = f"local folder supplied by the user ({source_folder})"
    else:
        download_raw_files()
        df = load_raw_dataframe()
        source_text = (
            "CSV version of the dataset hosted on Hugging Face "
            "(katielink/genomic-benchmarks, an upload of the original benchmark files)"
        )

    df, report = clean_dataframe(df)
    df = make_split(df)
    save_processed(df, report, source_text)

    counts = _count_table(df)
    print("\nDone. Summary:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    print(f"  train: {counts['train']}")
    print(f"  test:  {counts['test']}")
    print(f"\nSaved to: {PROCESSED_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the DNA dataset (Phase 2).")
    parser.add_argument(
        "--source-folder",
        help="Folder with the original dataset (train/ and test/ with .txt files). "
             "If not given, the CSV files are downloaded automatically.",
    )
    args = parser.parse_args()

    try:
        run_pipeline(args.source_folder)
    except Exception as error:   # friendly message for beginners
        print("\nSomething went wrong:", error)
        print("If the download failed, see the 'Alternative' steps in the Phase 2 instructions.")
        raise SystemExit(1)