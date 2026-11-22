"""
Tests for src/prediction.py

Most tests use small models trained on made-up data (GC-rich vs AT-rich sequences),
so they are fast and need no real dataset. One test checks the real saved models
and is skipped quietly if they have not been trained yet.

Run from the DNA_Intelligence folder (with .venv active):
    python -m pytest -q
"""

import numpy as np

from src import feature_extraction as fe
from src import model_training as mt
from src import prediction as pr


def make_sequences(n_per_class=150, seed=0):
    """Made-up sequences: GC-rich ones get label 1, AT-rich ones get label 0 (200 bp each)."""
    rng = np.random.default_rng(seed)
    sequences, labels = [], []
    for label, gc in ((1, 0.65), (0, 0.35)):
        probabilities = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]   # A, C, G, T
        for _ in range(n_per_class):
            sequences.append("".join(rng.choice(list("ACGT"), size=200, p=probabilities)))
            labels.append(label)
    return sequences, np.array(labels)


def small_trained_models():
    sequences, labels = make_sequences()
    X = fe.extract_features_batch(sequences)
    forest = mt.build_random_forest(n_estimators=20).fit(X, labels)
    svm = mt.build_svm().fit(X, labels)
    return {"random_forest": forest, "svm": svm}


def raises_value_error(function, *args):
    try:
        function(*args)
    except ValueError:
        return True
    return False


# ---------- predicting ----------
def test_predictions_have_the_right_shape_and_content():
    models = small_trained_models()
    new_sequences, _ = make_sequences(3, seed=7)
    sequence = new_sequences[0]

    forest_result = pr.predict_with_model(models["random_forest"], "random_forest", sequence)
    assert forest_result["label"] in (0, 1)
    assert forest_result["label_text"] == pr.LABEL_TEXT[forest_result["label"]]
    total = forest_result["probability_coding"] + forest_result["probability_non_coding"]
    assert np.isclose(total, 1.0)
    assert forest_result["score"] is None            # the forest has no decision score
    assert forest_result["features"].shape == (fe.N_FEATURES,)
    assert forest_result["length"] == 200

    svm_result = pr.predict_with_model(models["svm"], "svm", sequence)
    assert svm_result["probability_coding"] is None  # the SVM has no probability
    assert svm_result["probability_non_coding"] is None
    assert isinstance(svm_result["score"], float)


def test_svm_score_sign_matches_the_label():
    models = small_trained_models()
    sequences, _ = make_sequences(10, seed=8)
    for sequence in sequences:
        result = pr.predict_with_model(models["svm"], "svm", sequence)
        assert (result["score"] > 0) == (result["label"] == 1)


def test_forest_label_matches_its_probability():
    models = small_trained_models()
    sequences, _ = make_sequences(10, seed=9)
    for sequence in sequences:
        result = pr.predict_with_model(models["random_forest"], "random_forest", sequence)
        assert (result["probability_coding"] > 0.5) == (result["label"] == 1)


def test_made_up_gc_rich_and_at_rich_sequences_are_told_apart():
    models = small_trained_models()
    sequences, labels = make_sequences(10, seed=5)
    for key, model in models.items():
        correct = sum(
            pr.predict_with_model(model, key, s)["label"] == label
            for s, label in zip(sequences, labels)
        )
        assert correct >= 18      # 20 sequences, easy classes


def test_lowercase_and_spaces_around_the_sequence_are_fine():
    models = small_trained_models()
    sequence = make_sequences(1, seed=3)[0][0]
    plain = pr.predict_with_model(models["svm"], "svm", sequence)
    messy = pr.predict_with_model(models["svm"], "svm", "  " + sequence.lower() + "\n")
    assert plain["label"] == messy["label"] and np.isclose(plain["score"], messy["score"])


def test_bad_sequences_give_clear_errors_not_crashes():
    forest = small_trained_models()["random_forest"]
    for bad in ("", "ATG", "ATGNNCATG", "AUGCAUGCAUGC", "ATG-CATGCA"):
        assert raises_value_error(pr.predict_with_model, forest, "random_forest", bad)


# ---------- explaining the input ----------
def test_length_warning_levels():
    assert pr.length_warning(200) is None
    assert pr.length_warning(180)[0] == "info"
    assert pr.length_warning(250)[0] == "info"
    assert pr.length_warning(100)[0] == "warning"
    assert pr.length_warning(5000)[0] == "warning"
    assert "200" in pr.length_warning(100)[1]


def test_key_features_table():
    forest = small_trained_models()["random_forest"]
    sequence = make_sequences(1, seed=4)[0][0]
    features = fe.extract_features(sequence)
    table = pr.key_features_table(features, forest, top_n=5)
    assert len(table) == 5
    assert list(table.columns) == ["Feature", "Meaning", "Value in your sequence", "Forest importance"]
    assert list(table["Forest importance"]) == sorted(table["Forest importance"], reverse=True)
    names = fe.get_feature_names()
    for _, row in table.iterrows():
        assert np.isclose(row["Value in your sequence"], features[names.index(row["Feature"])], atol=1e-4)


def test_disclaimer_and_labels():
    assert "not a gene finder" in pr.DISCLAIMER
    assert pr.LABEL_TEXT == {1: "Coding", 0: "Non-coding"}
    assert set(pr.MODEL_CHOICES.values()) == {"random_forest", "svm"}


# ---------- loading ----------
def test_unknown_model_name_is_rejected():
    assert raises_value_error(pr.get_model, "neural_network")


def test_models_available_reports_both_models():
    assert set(pr.models_available()) == {"random_forest", "svm"}


def test_real_saved_models_work_if_they_exist():
    if not all(pr.models_available().values()):
        return          # models not trained yet - nothing to check
    sequence = make_sequences(1, seed=11)[0][0]
    for key in ("random_forest", "svm"):
        result = pr.predict_sequence(sequence, key)
        assert result["label"] in (0, 1)
        assert result["features"].shape == (fe.N_FEATURES,)
    assert mt.check_feature_compatibility() is True