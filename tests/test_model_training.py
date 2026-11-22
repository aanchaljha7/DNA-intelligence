"""
Tests for src/model_training.py

These tests are fast: they use small made-up data (GC-rich vs AT-rich sequences),
so the real dataset is not needed.

Run from the DNA_Intelligence folder (with .venv active):
    python -m pytest -q
"""

import tempfile
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src import feature_extraction as fe
from src import model_training as mt


def make_sequences(n_per_class=300, seed=0):
    """Made-up sequences: GC-rich ones get label 1, AT-rich ones get label 0 (200 bp each)."""
    rng = np.random.default_rng(seed)
    sequences, labels = [], []
    for label, gc in ((1, 0.65), (0, 0.35)):
        probabilities = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]   # A, C, G, T
        for _ in range(n_per_class):
            sequences.append("".join(rng.choice(list("ACGT"), size=200, p=probabilities)))
            labels.append(label)
    return sequences, np.array(labels)


def make_data(n_per_class=300, seed=0):
    """The same made-up data as a feature matrix and labels."""
    sequences, labels = make_sequences(n_per_class, seed)
    return fe.extract_features_batch(sequences), labels


def small_models():
    return mt.build_random_forest(n_estimators=20), mt.build_svm()


# ---------- building ----------
def test_builders_and_overrides():
    forest, svm = small_models()
    assert isinstance(forest, RandomForestClassifier)
    assert mt.build_random_forest(n_estimators=7).n_estimators == 7
    assert [name for name, _ in svm.steps] == ["scaler", "svm"]
    assert mt.build_svm(C=5.0).named_steps["svm"].C == 5.0


def test_training_subset_keeps_class_balance():
    X, y = make_data()
    X_small, y_small = mt.take_training_subset(X, y, 100)
    assert len(y_small) == 100 and X_small.shape == (100, fe.N_FEATURES)
    assert int((y_small == 1).sum()) == 50
    assert len(mt.take_training_subset(X, y, None)[1]) == len(y)
    assert len(mt.take_training_subset(X, y, 10 ** 6)[1]) == len(y)


# ---------- training and evaluating ----------
def test_both_models_learn_and_metrics_are_consistent():
    X, y = make_data(300, seed=1)
    X_test, y_test = make_data(100, seed=2)
    for model in small_models():
        mt.train_model(model, X, y, "test model")
        metrics = mt.evaluate_model(model, X_test, y_test)
        assert metrics["accuracy"] > 0.9
        (tn, fp), (fn, tp) = metrics["confusion_matrix"]
        assert tn + fp + fn + tp == len(y_test) == metrics["test_samples"]
        assert np.isclose(metrics["accuracy"], (tn + tp) / len(y_test))
        assert np.isclose(metrics["recall"], tp / (tp + fn))
        assert np.isclose(metrics["precision"], tp / (tp + fp))
        assert set(metrics["per_class"]) == {"intergenomic", "coding"}
        assert metrics["per_class"]["coding"]["support"] == 100


# ---------- saving and loading ----------
def test_save_and_load_give_the_same_predictions():
    X, y = make_data(150)
    with tempfile.TemporaryDirectory() as folder:
        for model in small_models():
            model.fit(X, y)
            path = Path(folder) / "model.pkl"
            size_mb = mt.save_model(model, path)
            assert size_mb > 0
            loaded = mt.load_model(path)
            assert np.array_equal(model.predict(X), loaded.predict(X))


def test_loading_a_missing_model_gives_a_clear_error():
    try:
        mt.load_model(Path(tempfile.gettempdir()) / "no_such_model_file.pkl")
    except FileNotFoundError as error:
        assert "Train the models first" in str(error)
    else:
        assert False, "a FileNotFoundError was expected"


def test_single_user_sequences_can_be_predicted():
    """The one-sequence path (extract_features) must work with the trained models."""
    X, y = make_data(150)
    new_sequences, new_labels = make_sequences(10, seed=5)
    for model in small_models():
        model.fit(X, y)
        correct = 0
        for sequence, label in zip(new_sequences, new_labels):
            vector = fe.extract_features(sequence).reshape(1, -1)
            assert vector.shape == (1, fe.N_FEATURES)
            correct += int(model.predict(vector)[0]) == label
        assert correct >= 18      # 20 sequences; these made-up classes are easy to separate


# ---------- feature information and metrics ----------
def test_feature_info_roundtrip_and_compatibility_check():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "feature_info.json"
        mt.save_feature_info(path)
        info = mt.load_feature_info(path)
        assert info["n_features"] == fe.N_FEATURES == len(info["feature_names"])
        assert len(info["kmer_vocabulary"]["3"]) == 64
        assert len(info["kmer_vocabulary"]["4"]) == 256
        assert mt.check_feature_compatibility(info) is True

        info["feature_names"][0] = "something_else"
        try:
            mt.check_feature_compatibility(info)
        except ValueError:
            pass
        else:
            assert False, "a ValueError was expected"


def test_metrics_roundtrip():
    X, y = make_data(100)
    forest = mt.build_random_forest(n_estimators=10).fit(X, y)
    results = {"random_forest": mt.evaluate_model(forest, X, y)}
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "metrics.json"
        mt.save_metrics(results, path)
        saved = mt.load_metrics(path)
        assert saved["models"]["random_forest"]["accuracy"] == results["random_forest"]["accuracy"]
        assert "scikit_learn_version" in saved