"""
Tests for src/evaluation.py

Fast tests with small made-up data (GC-rich vs AT-rich sequences). No real dataset
and no saved models are needed.

Run from the DNA_Intelligence folder (with .venv active):
    python -m pytest -q
"""

import tempfile
from pathlib import Path

import numpy as np

from src import evaluation as ev
from src import feature_extraction as fe
from src import model_training as mt


def make_data(n_per_class=150, seed=0):
    """Made-up data: GC-rich sequences get label 1, AT-rich sequences get label 0."""
    rng = np.random.default_rng(seed)
    sequences, labels = [], []
    for label, gc in ((1, 0.6), (0, 0.4)):
        probabilities = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
        for _ in range(n_per_class):
            sequences.append("".join(rng.choice(list("ACGT"), size=200, p=probabilities)))
            labels.append(label)
    return fe.extract_features_batch(sequences), np.array(labels)


def trained_models_and_results():
    X, y = make_data(200, seed=1)
    X_test, y_test = make_data(100, seed=2)
    forest = mt.build_random_forest(n_estimators=20).fit(X, y)
    svm = mt.build_svm().fit(X, y)
    models_metrics = {}
    for key, model in (("random_forest", forest), ("svm", svm)):
        metrics = mt.evaluate_model(model, X_test, y_test)
        metrics.update({"train_samples": len(y), "training_seconds": 1.5, "file_size_mb": 2.25})
        models_metrics[key] = metrics
    return forest, svm, X_test, y_test, models_metrics


def raises(error_type, function, *args):
    try:
        function(*args)
    except error_type:
        return True
    return False


# ---------- feature descriptions ----------
def test_describe_feature():
    assert "G or C" in ev.describe_feature("gc_content")
    assert "A or T" in ev.describe_feature("at_content")
    assert ev.describe_feature("freq_G") == "Fraction of bases that are G"
    assert ev.describe_feature("k3_ATG") == "Frequency of the 3-mer ATG"
    assert ev.describe_feature("k4_AAAA") == "Frequency of the 4-mer AAAA"
    for name in fe.get_feature_names():
        assert ev.describe_feature(name) != name      # every real feature has a description


# ---------- ROC ----------
def test_roc_curves_are_valid():
    forest, svm, X_test, y_test, _ = trained_models_and_results()
    for model, score_word in ((forest, "probability"), (svm, "decision score")):
        roc = ev.roc_curve_data(model, X_test, y_test)
        assert score_word in roc["score_type"]
        assert 0.9 < roc["auc"] <= 1.0
        assert len(roc["fpr"]) == len(roc["tpr"])
        assert roc["fpr"][0] == 0 and roc["tpr"][0] == 0
        assert roc["fpr"][-1] == 1 and roc["tpr"][-1] == 1
        assert all(b >= a for a, b in zip(roc["fpr"], roc["fpr"][1:]))
        assert all(b >= a for a, b in zip(roc["tpr"], roc["tpr"][1:]))


def test_roc_curve_is_thinned_but_auc_is_unchanged():
    forest, _, X_test, y_test, _ = trained_models_and_results()
    full = ev.roc_curve_data(forest, X_test, y_test, max_points=10 ** 6)
    thin = ev.roc_curve_data(forest, X_test, y_test, max_points=15)
    assert len(thin["fpr"]) == 15 < len(full["fpr"])
    assert thin["auc"] == full["auc"]


def test_roc_save_and_load():
    forest, _, X_test, y_test, _ = trained_models_and_results()
    roc = {"random_forest": ev.roc_curve_data(forest, X_test, y_test)}
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "roc.json"
        ev.save_roc_data(roc, len(y_test), path)
        saved = ev.load_roc_data(path)
        assert saved["test_samples"] == len(y_test)
        assert saved["models"]["random_forest"]["auc"] == roc["random_forest"]["auc"]
    assert raises(FileNotFoundError, ev.load_roc_data, Path(tempfile.gettempdir()) / "missing_roc.json")


# ---------- tables ----------
def test_comparison_table():
    _, _, _, _, models_metrics = trained_models_and_results()
    table = ev.comparison_table(models_metrics)
    assert list(table.columns) == ["Random Forest", "SVM (RBF)"]
    assert "ROC AUC" not in table.index
    assert table.loc["Accuracy", "SVM (RBF)"] == f"{models_metrics['svm']['accuracy']:.4f}"
    assert table.loc["Trained on (sequences)", "Random Forest"] == "400"

    roc = {"random_forest": {"auc": 0.97}, "svm": {"auc": 0.98}}
    with_roc = ev.comparison_table(models_metrics, roc)
    assert with_roc.loc["ROC AUC", "SVM (RBF)"] == "0.9800"
    assert list(with_roc.index)[:4] == ["Accuracy", "Precision (coding)", "Recall (coding)", "F1 (coding)"]


def test_classification_report_table():
    _, _, _, y_test, models_metrics = trained_models_and_results()
    metrics = models_metrics["random_forest"]
    table = ev.classification_report_table(metrics)
    assert list(table["Class"]) == ["Non-coding (intergenomic)", "Coding", "accuracy", "macro avg", "weighted avg"]
    assert list(table.columns) == ["Class", "Precision", "Recall", "F1-score", "Support"]
    assert table.loc[2, "F1-score"] == f"{metrics['accuracy']:.4f}"
    assert table.loc[2, "Support"] == f"{len(y_test):,}"
    coding = metrics["per_class"]["coding"]
    assert table.loc[1, "Recall"] == f"{coding['recall']:.4f}"
    # balanced classes: the macro and weighted averages must be equal
    assert table.loc[3, "F1-score"] == table.loc[4, "F1-score"]


def test_confusion_matrix_table():
    _, _, _, y_test, models_metrics = trained_models_and_results()
    metrics = models_metrics["svm"]
    table = ev.confusion_matrix_table(metrics)
    assert table.shape == (2, 2)
    assert int(table.to_numpy().sum()) == len(y_test)
    assert table.loc["True coding", "Predicted coding"] == metrics["confusion_matrix"][1][1]


# ---------- feature importance ----------
def test_importance_table_and_groups():
    forest, _, _, _, _ = trained_models_and_results()
    table = ev.importance_table(forest)
    assert len(table) == fe.N_FEATURES
    assert list(table["Importance"]) == sorted(table["Importance"], reverse=True)
    assert abs(table["Importance"].sum() - 1.0) < 0.01
    assert len(ev.importance_table(forest, top_n=7)) == 7

    groups = ev.importance_by_group(forest)
    assert list(groups["Number of features"]) == [6, 64, 256]
    assert abs(groups["Share of total importance (%)"].sum() - 100.0) < 0.5


def test_importance_rejects_a_forest_with_the_wrong_number_of_features():
    from sklearn.ensemble import RandomForestClassifier
    X = np.random.default_rng(0).random((60, 5))
    y = np.array([0, 1] * 30)
    odd_forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    assert raises(ValueError, ev.importance_table, odd_forest)