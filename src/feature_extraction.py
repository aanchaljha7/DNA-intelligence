"""
feature_extraction.py  -  Phase 4: turning DNA sequences into numbers

A machine learning model cannot read letters, so every DNA sequence is converted
into a fixed-length list of numbers (a "feature vector"). Every sequence gives
exactly the same number of features, in the same order.

The 326 features are:
  - gc_content, at_content                  (2 numbers)
  - freq_A, freq_C, freq_G, freq_T          (4 numbers)
  - k3_AAA ... k3_TTT                       (64 numbers: 3-mer frequencies)
  - k4_AAAA ... k4_TTTT                     (256 numbers: 4-mer frequencies)

All values are FREQUENCIES between 0 and 1 (not counts), so sequences of different
lengths can be compared. The k-mer "vocabulary" is fixed (all 4^k possible k-mers in
alphabetical order), so column number 10 always means the same thing.

The SAME function is used for the training data and for one sequence typed by a
user later (Phase 5), so the model always sees features built the same way.

Run from the DNA_Intelligence folder (with .venv active):
    python -m src.feature_extraction
This builds the training/test matrices from data/processed/dna_dataset.csv and
saves them to data/processed/features.npz.

Scientific note: these features only describe the composition of the sequence.
They say nothing about gene structure, so a model that uses them is a simple
pattern recognizer, not a gene finder.
"""

import json
from collections import Counter
from datetime import date
from itertools import product

import numpy as np

from src import preprocessing as pp
from src import sequence_analysis as sa

# ---------------------------------------------------------------------------
# Feature definition (fixed - do not change after a model has been trained)
# ---------------------------------------------------------------------------
BASES = "ACGT"
K_VALUES = (3, 4)
MIN_SEQUENCE_LENGTH = max(K_VALUES)     # shortest sequence that has at least one 4-mer

BASIC_FEATURE_NAMES = ["gc_content", "at_content", "freq_A", "freq_C", "freq_G", "freq_T"]

FEATURES_NPZ = pp.PROCESSED_DIR / "features.npz"
FEATURE_INFO_JSON = pp.PROCESSED_DIR / "feature_info.json"


def kmer_vocabulary(k):
    """All 4^k possible k-mers in alphabetical order (AAA, AAC, AAG, ... TTT)."""
    return ["".join(letters) for letters in product(BASES, repeat=k)]


# Build the list of feature names and remember where each k-mer block starts.
FEATURE_NAMES = list(BASIC_FEATURE_NAMES)
KMER_BLOCKS = {}      # k -> (first column of the block, {k-mer: column number})
_offset = len(BASIC_FEATURE_NAMES)
for _k in K_VALUES:
    _vocabulary = kmer_vocabulary(_k)
    KMER_BLOCKS[_k] = (_offset, {kmer: i for i, kmer in enumerate(_vocabulary)})
    FEATURE_NAMES.extend(f"k{_k}_{kmer}" for kmer in _vocabulary)
    _offset += len(_vocabulary)

N_FEATURES = len(FEATURE_NAMES)      # 6 + 64 + 256 = 326


def get_feature_names():
    """Names of all features, in column order."""
    return list(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# One sequence  ->  one feature vector
# ---------------------------------------------------------------------------
def _check_sequence(sequence):
    """Cleans a sequence (strip + uppercase) and raises ValueError if it cannot be used."""
    cleaned = str(sequence).strip().upper()

    is_valid, message = sa.validate_sequence(cleaned)
    if not is_valid:
        raise ValueError(message)
    if len(cleaned) < MIN_SEQUENCE_LENGTH:
        raise ValueError(
            f"The sequence is too short ({len(cleaned)} bp). "
            f"At least {MIN_SEQUENCE_LENGTH} bases are needed."
        )
    return cleaned


def _fill_row(row, sequence):
    """Writes the features of one (already checked) sequence into the array 'row'."""
    length = len(sequence)
    count_a = sequence.count("A")
    count_c = sequence.count("C")
    count_g = sequence.count("G")
    count_t = sequence.count("T")

    row[0] = (count_g + count_c) / length      # gc_content
    row[1] = (count_a + count_t) / length      # at_content
    row[2] = count_a / length                  # freq_A
    row[3] = count_c / length                  # freq_C
    row[4] = count_g / length                  # freq_G
    row[5] = count_t / length                  # freq_T

    for k, (offset, columns) in KMER_BLOCKS.items():
        total = length - k + 1                 # number of overlapping k-mers
        counts = Counter(sequence[i:i + k] for i in range(total))
        for kmer, count in counts.items():
            row[offset + columns[kmer]] = count / total


def extract_features(sequence):
    """
    Converts ONE DNA sequence into a feature vector.
    Returns a numpy array of shape (326,), dtype float32.
    Raises ValueError (with a clear message) for empty, invalid (N, U, ...) or too short input.
    """
    cleaned = _check_sequence(sequence)
    row = np.zeros(N_FEATURES, dtype=np.float32)
    _fill_row(row, cleaned)
    return row


# ---------------------------------------------------------------------------
# Many sequences  ->  feature matrix
# ---------------------------------------------------------------------------
def extract_features_batch(sequences):
    """
    Converts many sequences (a list or a pandas column) into a feature matrix.
    Returns a numpy array of shape (number of sequences, 326), dtype float32.
    Row i of the matrix is exactly extract_features(sequences[i]).
    """
    sequences = list(sequences)
    matrix = np.zeros((len(sequences), N_FEATURES), dtype=np.float32)
    for i, sequence in enumerate(sequences):
        try:
            cleaned = _check_sequence(sequence)
        except ValueError as error:
            raise ValueError(f"Sequence number {i + 1}: {error}") from None
        _fill_row(matrix[i], cleaned)
    return matrix


# ---------------------------------------------------------------------------
# Training / test matrices from the processed dataset
# ---------------------------------------------------------------------------
def build_feature_matrices():
    """
    Reads data/processed/dna_dataset.csv (made in Phase 2) and returns
    X_train, X_test, y_train, y_test.
    X = feature matrix, y = labels (1 = coding, 0 = intergenomic).
    The train/test split is the one saved in Phase 2.
    """
    table = pp.load_processed_data()
    train = table[table["split"] == "train"]
    test = table[table["split"] == "test"]

    print(f"Extracting features for {len(train):,} training sequences...")
    X_train = extract_features_batch(train["sequence"])
    print(f"Extracting features for {len(test):,} test sequences...")
    X_test = extract_features_batch(test["sequence"])

    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()

    # Safety check: no missing or infinite numbers
    if not (np.isfinite(X_train).all() and np.isfinite(X_test).all()):
        raise ValueError("The feature matrices contain NaN or infinite values.")

    return X_train, X_test, y_train, y_test


def save_features(X_train, X_test, y_train, y_test):
    """Saves the matrices (features.npz) and a description of the features (feature_info.json)."""
    pp.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FEATURES_NPZ,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        feature_names=np.array(FEATURE_NAMES),
    )

    groups = [{
        "name": "basic composition",
        "columns": f"0-{len(BASIC_FEATURE_NAMES) - 1}",
        "features": BASIC_FEATURE_NAMES,
    }]
    for k, (offset, columns) in KMER_BLOCKS.items():
        groups.append({
            "name": f"{k}-mer frequencies",
            "columns": f"{offset}-{offset + len(columns) - 1}",
            "features": (
                f"k{k}_{'A' * k} ... k{k}_{'T' * k} "
                f"(alphabetical order, {len(columns)} features)"
            ),
        })

    info = {
        "created_on": str(date.today()),
        "source": "data/processed/dna_dataset.csv (split column from Phase 2)",
        "n_features": N_FEATURES,
        "k_values": list(K_VALUES),
        "feature_groups": groups,
        "value_meaning": (
            "All features are frequencies between 0 and 1. gc_content = (G+C)/length, "
            "at_content = (A+T)/length, k-mer frequency = count of that overlapping k-mer "
            "divided by the number of k-mers in the sequence (length - k + 1)."
        ),
        "label_meaning": {"1": "coding", "0": "intergenomic"},
        "shapes": {
            "X_train": list(X_train.shape), "y_train": list(y_train.shape),
            "X_test": list(X_test.shape), "y_test": list(y_test.shape),
        },
        "class_counts": {
            "train": {"coding": int((y_train == 1).sum()), "intergenomic": int((y_train == 0).sum())},
            "test": {"coding": int((y_test == 1).sum()), "intergenomic": int((y_test == 0).sum())},
        },
    }
    with open(FEATURE_INFO_JSON, "w") as f:
        json.dump(info, f, indent=2)


def load_features():
    """
    Loads the saved matrices. Returns X_train, X_test, y_train, y_test.
    Raises an error if the file is missing or was made with a different feature definition.
    """
    if not FEATURES_NPZ.exists():
        raise FileNotFoundError(
            "Feature file not found. Run this first:  python -m src.feature_extraction"
        )
    data = np.load(FEATURES_NPZ)
    if list(data["feature_names"]) != FEATURE_NAMES:
        raise ValueError(
            "The saved features do not match the current feature definition. "
            "Run again:  python -m src.feature_extraction"
        )
    return data["X_train"], data["X_test"], data["y_train"], data["y_test"]


def get_feature_matrices():
    """Loads the saved matrices if they exist, otherwise builds (and saves) them."""
    try:
        return load_features()
    except FileNotFoundError:
        matrices = build_feature_matrices()
        save_features(*matrices)
        return matrices


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test = build_feature_matrices()
        save_features(X_train, X_test, y_train, y_test)
    except Exception as error:   # friendly message for beginners
        print("\nSomething went wrong:", error)
        raise SystemExit(1)

    print("\nDone. Summary:")
    print(f"  Number of features: {N_FEATURES}")
    print(f"  X_train shape: {X_train.shape}   y_train shape: {y_train.shape}")
    print(f"  X_test shape:  {X_test.shape}   y_test shape:  {y_test.shape}")
    print(f"  Train labels: coding={int((y_train == 1).sum())}, intergenomic={int((y_train == 0).sum())}")
    print(f"  Test labels:  coding={int((y_test == 1).sum())}, intergenomic={int((y_test == 0).sum())}")
    print(f"\nSaved to: {FEATURES_NPZ}")
    print(f"Description: {FEATURE_INFO_JSON}")