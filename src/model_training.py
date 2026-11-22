"""
model_training.py  -  Phase 5: training and evaluating the classifiers

What this file does:
  1. Loads the feature matrices made in Phase 4 (X = numbers, y = labels).
  2. Trains two models: a Random Forest and an SVM (support vector machine).
  3. Tests both on the held-out test set (sequences the models never saw).
  4. Prints the real results and saves them.
  5. Saves the trained models and the feature information into models/.

How to run (from the DNA_Intelligence folder, with .venv active):
    python -m src.model_training

Files created in models/:
    random_forest_model.pkl   the trained Random Forest
    svm_model.pkl             the trained SVM (a scaler + the SVM, saved together)
    feature_info.json         the feature names and k-mer vocabulary used for training
    metrics.json              all test-set results

Labels: 1 = coding, 0 = intergenomic. "Precision / recall / F1" in the summary
are for the coding class (the positive class); both classes are printed in full.

Scientific note: the models learn composition patterns in 200 bp benchmark
fragments. A good score here does not make them gene finders.
"""

import json
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import feature_extraction as fe
from src import preprocessing as pp

# ---------------------------------------------------------------------------
# Settings (all in one place)
# ---------------------------------------------------------------------------
MODELS_DIR = pp.PROJECT_ROOT / "models"
RF_MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"
SVM_MODEL_PATH = MODELS_DIR / "svm_model.pkl"
FEATURE_INFO_PATH = MODELS_DIR / "feature_info.json"
METRICS_PATH = MODELS_DIR / "metrics.json"

RANDOM_STATE = 42
CLASS_NAMES = {0: "intergenomic", 1: "coding"}

# Random Forest settings. min_samples_leaf stops the trees from memorising single
# sequences and keeps the saved file small enough for GitHub.
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,            # None = trees may grow as deep as needed
    "min_samples_leaf": 5,
    "n_jobs": -1,                 # use all CPU cores
    "random_state": RANDOM_STATE,
}

# SVM settings. An RBF-kernel SVM becomes very slow (and its file very big) when
# trained on all ~80,000 sequences, so it is trained on a random subset.
# Training time grows very fast with the number of sequences. Set SVM_TRAIN_SIZE = None
# to use every training sequence (much slower, bigger file).
# The SVM gives a label and a score (model.decision_function): above 0 means "coding".
# It does not give probabilities (SVC(probability=True) is deprecated in scikit-learn).
SVM_TRAIN_SIZE = 20000
SVM_PARAMS = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "cache_size": 1000,           # MB of memory the SVM may use to go faster
    "random_state": RANDOM_STATE,
}

SIZE_WARNING_MB = 90              # GitHub refuses files bigger than 100 MB


# ---------------------------------------------------------------------------
# Building the models
# ---------------------------------------------------------------------------
def build_random_forest(**changes):
    """A Random Forest: many decision trees that vote. Pass keyword arguments to change settings."""
    params = dict(RF_PARAMS)
    params.update(changes)
    return RandomForestClassifier(**params)


def build_svm(**changes):
    """
    An SVM. SVMs need all features on a similar scale, so StandardScaler is
    placed in front of it and the two are saved together as one 'pipeline'.
    """
    params = dict(SVM_PARAMS)
    params.update(changes)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(**params)),
    ])


def take_training_subset(X, y, size):
    """Random subset of 'size' rows with the same class balance (None = use everything)."""
    if size is None or size >= len(y):
        return X, y
    X_subset, _, y_subset, _ = train_test_split(
        X, y, train_size=size, stratify=y, random_state=RANDOM_STATE
    )
    return X_subset, y_subset


def train_model(model, X, y, name):
    """Trains the model and returns the training time in seconds."""
    print(f"\nTraining {name} on {len(y):,} sequences...")
    start = time.time()
    model.fit(X, y)
    seconds = time.time() - start
    print(f"  finished in {seconds:.1f} seconds")
    return seconds


# ---------------------------------------------------------------------------
# Evaluating the models
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    """
    Tests the model on data it has never seen. Returns a dictionary with:
      accuracy, precision, recall, f1   (for the coding class = label 1)
      per_class                         (precision / recall / f1 / support for both classes)
      confusion_matrix                  [[TN, FP], [FN, TP]]; rows = true, columns = predicted,
                                        order = [intergenomic, coding]
    """
    predicted = model.predict(X_test)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predicted, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(y_test, predicted, labels=[0, 1])

    per_class = {}
    for label, class_name in CLASS_NAMES.items():
        per_class[class_name] = {
            "precision": float(precision[label]),
            "recall": float(recall[label]),
            "f1": float(f1[label]),
            "support": int(support[label]),
        }

    return {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision[1]),
        "recall": float(recall[1]),
        "f1": float(f1[1]),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "test_samples": int(len(y_test)),
    }


def print_evaluation(name, metrics):
    """Prints the results of evaluate_model() in a readable way."""
    print(f"\n=== {name}: results on the test set ({metrics['test_samples']:,} sequences) ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"{'Class':<15}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
    for class_name, values in metrics["per_class"].items():
        print(f"{class_name:<15}{values['precision']:>10.4f}{values['recall']:>10.4f}"
              f"{values['f1']:>10.4f}{values['support']:>10,}")

    (tn, fp), (fn, tp) = metrics["confusion_matrix"]
    print("Confusion matrix (rows = true class, columns = predicted class):")
    print(f"{'':<20}{'pred. intergenomic':>20}{'pred. coding':>15}")
    print(f"{'true intergenomic':<20}{tn:>20,}{fp:>15,}")
    print(f"{'true coding':<20}{fn:>20,}{tp:>15,}")


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------
def save_model(model, path):
    """Saves a model with Joblib (compressed). Returns the file size in MB."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    return path.stat().st_size / (1024 * 1024)


def load_model(path):
    """Loads a model saved with save_model()."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path.name}. Train the models first:  "
            "python -m src.model_training"
        )
    return joblib.load(path)


def load_random_forest():
    return load_model(RF_MODEL_PATH)


def load_svm():
    return load_model(SVM_MODEL_PATH)


def save_feature_info(path=FEATURE_INFO_PATH):
    """
    Saves the feature definition used for training, so that prediction later can
    build exactly the same features (and check that nothing has changed).
    """
    info = {
        "created_on": str(date.today()),
        "n_features": fe.N_FEATURES,
        "feature_names": fe.get_feature_names(),
        "k_values": list(fe.K_VALUES),
        "kmer_vocabulary": {str(k): fe.kmer_vocabulary(k) for k in fe.K_VALUES},
        "value_meaning": (
            "All features are frequencies between 0 and 1: gc_content, at_content, "
            "freq_A/C/G/T, and the frequency of every overlapping 3-mer and 4-mer."
        ),
        "min_sequence_length": fe.MIN_SEQUENCE_LENGTH,
        "training_sequence_length": pp.SEQ_LENGTH,
        "label_meaning": {"1": "coding", "0": "intergenomic"},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=2)


def load_feature_info(path=FEATURE_INFO_PATH):
    """Loads the saved feature definition."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "Feature information not found. Train the models first:  python -m src.model_training"
        )
    with open(path) as f:
        return json.load(f)


def check_feature_compatibility(info=None):
    """
    Raises ValueError if the saved models were trained with a different feature
    definition than the one in src/feature_extraction.py. Returns True otherwise.
    """
    if info is None:
        info = load_feature_info()
    if info["feature_names"] != fe.get_feature_names():
        raise ValueError(
            "The saved models were trained with a different feature definition. "
            "Train them again:  python -m src.model_training"
        )
    return True


def save_metrics(results, path=METRICS_PATH):
    """Saves the results of all models (and the library version) as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_on": str(date.today()),
        "scikit_learn_version": sklearn.__version__,
        "label_meaning": {"1": "coding", "0": "intergenomic"},
        "note": "precision, recall and f1 are for the coding class (label 1). "
                "confusion_matrix = [[TN, FP], [FN, TP]] with order [intergenomic, coding].",
        "models": results,
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def load_metrics(path=METRICS_PATH):
    """Loads the saved results."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "Metrics not found. Train the models first:  python -m src.model_training"
        )
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------
def _top_forest_features(forest, how_many=10):
    """The features the Random Forest used most (name, importance)."""
    names = fe.get_feature_names()
    order = np.argsort(forest.feature_importances_)[::-1][:how_many]
    return [{"feature": names[i], "importance": float(forest.feature_importances_[i])} for i in order]


def _train_evaluate_save(name, model, X_train, y_train, X_test, y_test, path, params):
    """Trains one model, tests it, saves it. Returns its results dictionary."""
    seconds = train_model(model, X_train, y_train, name)
    metrics = evaluate_model(model, X_test, y_test)
    print_evaluation(name, metrics)

    size_mb = save_model(model, path)
    print(f"\nSaved {Path(path).name} ({size_mb:.1f} MB)")
    if size_mb > SIZE_WARNING_MB:
        print(f"WARNING: this file is bigger than {SIZE_WARNING_MB} MB. GitHub does not accept "
              "files over 100 MB. Use fewer trees or a smaller SVM_TRAIN_SIZE.")

    metrics.update({
        "file": f"models/{Path(path).name}",
        "file_size_mb": round(size_mb, 2),
        "train_samples": int(len(y_train)),
        "training_seconds": round(seconds, 1),
        "settings": params,
    })
    return metrics


def run_training():
    X_train, X_test, y_train, y_test = fe.get_feature_matrices()
    print(f"Training data: {X_train.shape}   Test data: {X_test.shape}")

    results = {}

    forest = build_random_forest()
    results["random_forest"] = _train_evaluate_save(
        "Random Forest", forest,
        X_train, y_train, X_test, y_test, RF_MODEL_PATH, dict(RF_PARAMS),
    )
    results["random_forest"]["top_features"] = _top_forest_features(forest)

    X_svm, y_svm = take_training_subset(X_train, y_train, SVM_TRAIN_SIZE)
    svm = build_svm()
    results["svm"] = _train_evaluate_save(
        "SVM (RBF kernel)", svm,
        X_svm, y_svm, X_test, y_test, SVM_MODEL_PATH, dict(SVM_PARAMS),
    )

    save_feature_info()
    save_metrics(results)
    return results


def print_summary(results):
    """Side-by-side table of the two models (real numbers from this run)."""
    print("\n" + "=" * 66)
    print("SUMMARY (test set; precision, recall and F1 are for the coding class)")
    print("=" * 66)
    print(f"{'Model':<20}{'Trained on':>12}{'Accuracy':>10}{'Precision':>10}{'Recall':>8}{'F1':>8}")
    for key, name in [("random_forest", "Random Forest"), ("svm", "SVM (RBF)")]:
        r = results[key]
        print(f"{name:<20}{r['train_samples']:>12,}{r['accuracy']:>10.4f}"
              f"{r['precision']:>10.4f}{r['recall']:>8.4f}{r['f1']:>8.4f}")
    print(f"\nSaved in: {MODELS_DIR}")
    print("Files: random_forest_model.pkl, svm_model.pkl, feature_info.json, metrics.json")


if __name__ == "__main__":
    try:
        all_results = run_training()
    except Exception as error:   # friendly message for beginners
        print("\nSomething went wrong:", error)
        raise SystemExit(1)
    print_summary(all_results)