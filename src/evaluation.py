"""
evaluation.py  -  Phase 6: reading and checking the results of the trained models

Everything here uses the SAVED models and the results saved in models/:
  - metrics.json   (made by src.model_training)
  - roc_data.json  (made by this file)
  - the trained models themselves (for feature importance)

How to run (after training, from the DNA_Intelligence folder, with .venv active):
    python -m src.evaluation
This re-tests the two saved models on the test set (no re-training), checks that the
results match metrics.json, and saves the ROC curve data to models/roc_data.json.

The dashboard page in app.py only displays what these functions return.
No number is typed in by hand: every value comes from the trained models.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve

from src import feature_extraction as fe
from src import model_training as mt

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
ROC_PATH = mt.MODELS_DIR / "roc_data.json"
MAX_ROC_POINTS = 300          # the saved ROC curve is thinned to this many points

MODEL_TITLES = {"random_forest": "Random Forest", "svm": "SVM (RBF)"}
CLASS_TITLES = {"intergenomic": "Non-coding (intergenomic)", "coding": "Coding"}


# ---------------------------------------------------------------------------
# Feature descriptions
# ---------------------------------------------------------------------------
def describe_feature(name):
    """Plain-English meaning of a feature name such as 'k3_ATG' or 'gc_content'."""
    if name == "gc_content":
        return "Fraction of bases that are G or C"
    if name == "at_content":
        return "Fraction of bases that are A or T"
    if name.startswith("freq_"):
        return f"Fraction of bases that are {name[-1]}"
    if name.startswith("k") and "_" in name:
        k, kmer = name[1:].split("_", 1)
        return f"Frequency of the {k}-mer {kmer}"
    return name


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------
def get_scores(model, X):
    """
    One score per row: the higher the score, the more 'coding' the sequence looks.
    Uses the probability of coding if the model has one, otherwise the SVM decision score.
    Returns (scores, description).
    """
    if hasattr(model, "predict_proba"):
        column = list(model.classes_).index(1)
        return model.predict_proba(X)[:, column], "probability of coding"
    return model.decision_function(X), "SVM decision score (above 0 = coding)"


def roc_curve_data(model, X_test, y_test, max_points=MAX_ROC_POINTS):
    """
    ROC curve of a trained model on the test set.
    The AUC is calculated from all points; only the saved curve is thinned out.
    """
    scores, score_type = get_scores(model, X_test)
    fpr, tpr, _ = roc_curve(y_test, scores)
    auc = float(roc_auc_score(y_test, scores))

    if len(fpr) > max_points:
        keep = np.linspace(0, len(fpr) - 1, max_points).astype(int)
        fpr, tpr = fpr[keep], tpr[keep]

    return {
        "auc": auc,
        "score_type": score_type,
        "fpr": [round(float(v), 5) for v in fpr],
        "tpr": [round(float(v), 5) for v in tpr],
    }


def save_roc_data(roc_by_model, test_samples, path=ROC_PATH):
    """Saves the ROC data of all models as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_on": str(date.today()),
        "scikit_learn_version": sklearn.__version__,
        "test_samples": int(test_samples),
        "note": "Curves are thinned to at most " + str(MAX_ROC_POINTS) + " points for storage; "
                "each AUC is calculated from the full curve.",
        "models": roc_by_model,
    }
    with open(path, "w") as f:
        json.dump(report, f)


def load_roc_data(path=ROC_PATH):
    """Loads the saved ROC data (a dictionary with a 'models' entry)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "ROC data not found. Create it (no re-training needed) with:  python -m src.evaluation"
        )
    with open(path) as f:
        return json.load(f)


def load_results(path=mt.METRICS_PATH):
    """Loads metrics.json (made by src.model_training)."""
    return mt.load_metrics(path)


def evaluate_saved_models():
    """
    Tests the saved models again on the test set, checks the results against
    metrics.json, and saves the ROC data. Does NOT train anything.
    """
    mt.check_feature_compatibility()
    saved = mt.load_metrics()["models"]
    _, X_test, _, y_test = fe.get_feature_matrices()
    models = {"random_forest": mt.load_random_forest(), "svm": mt.load_svm()}

    roc = {}
    for key, model in models.items():
        print(f"\nTesting {MODEL_TITLES[key]} on {len(y_test):,} test sequences...")
        metrics = mt.evaluate_model(model, X_test, y_test)
        same = metrics["confusion_matrix"] == saved[key]["confusion_matrix"]
        print(f"  accuracy {metrics['accuracy']:.4f}  - matches metrics.json: "
              f"{'yes' if same else 'NO (train the models again)'}")
        roc[key] = roc_curve_data(model, X_test, y_test)
        print(f"  ROC AUC {roc[key]['auc']:.4f}")

    save_roc_data(roc, len(y_test))
    print(f"\nSaved: {ROC_PATH}")
    return roc


# ---------------------------------------------------------------------------
# Tables for the dashboard (all values come from the saved results)
# ---------------------------------------------------------------------------
def comparison_table(models_metrics, roc_models=None):
    """Side-by-side table: one row per measure, one column per model (text, ready to show)."""
    columns = {}
    for key, title in MODEL_TITLES.items():
        if key not in models_metrics:
            continue
        m = models_metrics[key]
        column = {
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision (coding)": f"{m['precision']:.4f}",
            "Recall (coding)": f"{m['recall']:.4f}",
            "F1 (coding)": f"{m['f1']:.4f}",
        }
        if roc_models and key in roc_models:
            column["ROC AUC"] = f"{roc_models[key]['auc']:.4f}"
        column["Trained on (sequences)"] = f"{m['train_samples']:,}"
        column["Training time (seconds)"] = f"{m['training_seconds']:.1f}"
        column["Model file (MB)"] = f"{m['file_size_mb']:.1f}"
        columns[title] = column

    table = pd.DataFrame(columns)
    first_column = next(iter(columns.values()))
    return table.reindex(list(first_column.keys()))


def classification_report_table(model_metrics):
    """A classification report (like scikit-learn's) built from the saved per-class results."""
    per_class = model_metrics["per_class"]
    total = sum(v["support"] for v in per_class.values())

    rows = []
    for class_name in ("intergenomic", "coding"):
        v = per_class[class_name]
        rows.append([CLASS_TITLES[class_name], f"{v['precision']:.4f}", f"{v['recall']:.4f}",
                     f"{v['f1']:.4f}", f"{v['support']:,}"])

    rows.append(["accuracy", "", "", f"{model_metrics['accuracy']:.4f}", f"{total:,}"])

    for label, weights in (("macro avg", None), ("weighted avg", "support")):
        values = {}
        for measure in ("precision", "recall", "f1"):
            numbers = [per_class[c][measure] for c in per_class]
            if weights:
                support = [per_class[c]["support"] for c in per_class]
                values[measure] = sum(n * s for n, s in zip(numbers, support)) / total
            else:
                values[measure] = sum(numbers) / len(numbers)
        rows.append([label, f"{values['precision']:.4f}", f"{values['recall']:.4f}",
                     f"{values['f1']:.4f}", f"{total:,}"])

    return pd.DataFrame(rows, columns=["Class", "Precision", "Recall", "F1-score", "Support"])


def confusion_matrix_table(model_metrics):
    """The confusion matrix as a small table with readable labels."""
    (tn, fp), (fn, tp) = model_metrics["confusion_matrix"]
    return pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["True non-coding", "True coding"],
        columns=["Predicted non-coding", "Predicted coding"],
    )


# ---------------------------------------------------------------------------
# Random Forest feature importance (read from the trained forest itself)
# ---------------------------------------------------------------------------
def _importances(forest):
    names = fe.get_feature_names()
    importances = forest.feature_importances_
    if len(importances) != len(names):
        raise ValueError("The forest and the feature definition do not match. "
                         "Train the models again:  python -m src.model_training")
    return names, importances


def importance_table(forest, top_n=None):
    """Features sorted by importance: Feature, Meaning, Importance."""
    names, importances = _importances(forest)
    order = np.argsort(importances)[::-1]
    if top_n:
        order = order[:top_n]
    return pd.DataFrame({
        "Feature": [names[i] for i in order],
        "Meaning": [describe_feature(names[i]) for i in order],
        "Importance": [round(float(importances[i]), 5) for i in order],
    })


def importance_by_group(forest):
    """Adds up the importance of each kind of feature (basic composition, 3-mers, 4-mers)."""
    names, importances = _importances(forest)
    groups = {
        "Basic composition (GC, AT, A, C, G, T)": [0, 0.0],
        "3-mer frequencies": [0, 0.0],
        "4-mer frequencies": [0, 0.0],
    }
    for name, value in zip(names, importances):
        if name.startswith("k3_"):
            key = "3-mer frequencies"
        elif name.startswith("k4_"):
            key = "4-mer frequencies"
        else:
            key = "Basic composition (GC, AT, A, C, G, T)"
        groups[key][0] += 1
        groups[key][1] += float(value)

    return pd.DataFrame({
        "Group": list(groups.keys()),
        "Number of features": [g[0] for g in groups.values()],
        "Share of total importance (%)": [round(g[1] * 100, 1) for g in groups.values()],
    })


if __name__ == "__main__":
    try:
        evaluate_saved_models()
    except Exception as error:   # friendly message for beginners
        print("\nSomething went wrong:", error)
        raise SystemExit(1)