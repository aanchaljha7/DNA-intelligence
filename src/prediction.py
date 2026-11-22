"""
prediction.py  -  Phase 6: predicting with the trained models

Takes ONE DNA sequence, turns it into the same 326 features that were used for
training (src/feature_extraction.py), and asks a saved model for its opinion.

  - Random Forest: gives a label AND a probability.
  - SVM:           gives a label and a score (the signed distance from the decision
                   boundary: above 0 means coding). It has no probability.

Nothing in this file uses Streamlit, so it can be tested on its own.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from src import evaluation as ev
from src import feature_extraction as fe
from src import model_training as mt
from src import preprocessing as pp

# Names shown in the app  ->  names used in the code
MODEL_CHOICES = {"Random Forest": "random_forest", "SVM": "svm"}
MODEL_FILES = {"random_forest": mt.RF_MODEL_PATH, "svm": mt.SVM_MODEL_PATH}

LABEL_TEXT = {1: "Coding", 0: "Non-coding"}     # label 0 is the 'intergenomic' class
TRAINING_LENGTH = pp.SEQ_LENGTH                  # the models were trained on 200 bp sequences

DISCLAIMER = (
    "**Scientific disclaimer.** This prediction comes from a simple machine learning model "
    "trained on one benchmark dataset of 200 bp DNA fragments (Genomic Benchmarks "
    "'demo_coding_vs_intergenomic_seqs'). It only looks at composition patterns (GC content "
    "and 3-mer / 4-mer frequencies). It is meant for learning and demonstration. It is not a "
    "gene finder, it is not evidence that a sequence is or is not functional, and it must "
    "not be used for medical, clinical or research decisions. The training labels were "
    "generated computationally, not confirmed in the laboratory, and 'Non-coding' here only "
    "means 'looks like the intergenomic examples'. The test-set results do not carry over "
    "to other organisms, other sequence lengths or real genome annotation."
)


# ---------------------------------------------------------------------------
# Loading the saved models
# ---------------------------------------------------------------------------
def models_available():
    """{'random_forest': True/False, 'svm': True/False} - do the model files exist?"""
    return {key: path.exists() for key, path in MODEL_FILES.items()}


@lru_cache(maxsize=None)
def get_model(model_key):
    """
    Loads a saved model (only the first time; after that it comes from memory).
    Checks that the model was trained with the current feature definition.
    """
    if model_key not in MODEL_FILES:
        raise ValueError(f"Unknown model '{model_key}'. Choose one of: {', '.join(MODEL_FILES)}")
    mt.check_feature_compatibility()
    return mt.load_model(MODEL_FILES[model_key])


def clear_model_cache():
    """Forget the loaded models (needed only if you train new ones while the app is running)."""
    get_model.cache_clear()


# ---------------------------------------------------------------------------
# Predicting
# ---------------------------------------------------------------------------
def predict_with_model(model, model_key, sequence):
    """
    Predicts for one sequence with a model that is already loaded.
    Raises ValueError (clear message) for empty, invalid or too short sequences.
    Returns a dictionary:
      label, label_text          1 / 'Coding' or 0 / 'Non-coding'
      probability_coding         number between 0 and 1, or None if the model has none
      probability_non_coding     the same for non-coding, or None
      score                      the SVM decision score, or None
      features                   the 326 numbers given to the model
      length                     length of the sequence in bp
    """
    cleaned = str(sequence).strip().upper()
    features = fe.extract_features(cleaned)          # also validates the sequence
    x = features.reshape(1, -1)

    label = int(model.predict(x)[0])
    result = {
        "model": model_key,
        "label": label,
        "label_text": LABEL_TEXT[label],
        "probability_coding": None,
        "probability_non_coding": None,
        "score": None,
        "features": features,
        "length": len(cleaned),
    }

    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        probabilities = model.predict_proba(x)[0]
        result["probability_coding"] = float(probabilities[classes.index(1)])
        result["probability_non_coding"] = float(probabilities[classes.index(0)])

    if hasattr(model, "decision_function"):
        result["score"] = float(np.ravel(model.decision_function(x))[0])

    return result


def predict_sequence(sequence, model_key):
    """Predicts with a saved model ('random_forest' or 'svm')."""
    return predict_with_model(get_model(model_key), model_key, sequence)


# ---------------------------------------------------------------------------
# Explaining the input
# ---------------------------------------------------------------------------
def length_warning(length):
    """
    None if the length is the training length (200 bp), otherwise (level, message)
    where level is 'info' or 'warning'.
    """
    if length == TRAINING_LENGTH:
        return None
    if 150 <= length <= 250:
        return ("info", f"Your sequence is {length} bp. The models were trained on "
                        f"{TRAINING_LENGTH} bp sequences, so this is slightly outside what "
                        "they have seen.")
    return ("warning", f"Your sequence is {length} bp, but the models were trained only on "
                       f"{TRAINING_LENGTH} bp sequences. Predictions for such a different "
                       "length are much less reliable and should not be trusted.")


def key_features_table(features, forest, top_n=10):
    """
    The features the Random Forest relies on most, with their values in YOUR sequence.
    Columns: Feature, Meaning, Value in your sequence, Forest importance.
    """
    names = fe.get_feature_names()
    importances = forest.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    return pd.DataFrame({
        "Feature": [names[i] for i in order],
        "Meaning": [ev.describe_feature(names[i]) for i in order],
        "Value in your sequence": [round(float(features[i]), 4) for i in order],
        "Forest importance": [round(float(importances[i]), 5) for i in order],
    })