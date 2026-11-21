"""
Tests for src/feature_extraction.py

Run from the DNA_Intelligence folder (with .venv active):
    python -m pytest -q
"""

import numpy as np

from src import feature_extraction as fe

K3_START, K4_START = fe.KMER_BLOCKS[3][0], fe.KMER_BLOCKS[4][0]


def raises_value_error(function, *args):
    try:
        function(*args)
    except ValueError:
        return True
    return False


# ---------- feature definition ----------
def test_feature_count_and_names():
    names = fe.get_feature_names()
    assert fe.N_FEATURES == 326 == len(names)
    assert len(set(names)) == len(names)            # no duplicates
    assert names[:6] == ["gc_content", "at_content", "freq_A", "freq_C", "freq_G", "freq_T"]
    assert names[K3_START] == "k3_AAA" and names[K4_START - 1] == "k3_TTT"
    assert names[K4_START] == "k4_AAAA" and names[-1] == "k4_TTTT"


def test_kmer_vocabulary_is_fixed():
    vocab3 = fe.kmer_vocabulary(3)
    assert len(vocab3) == 64 and vocab3[0] == "AAA" and vocab3[1] == "AAC" and vocab3[-1] == "TTT"
    assert len(fe.kmer_vocabulary(4)) == 256


# ---------- one sequence ----------
def test_simple_sequence_values():
    vector = fe.extract_features("AACCGGTT")
    assert vector.shape == (326,) and vector.dtype == np.float32
    assert np.isclose(vector[0], 0.5) and np.isclose(vector[1], 0.5)      # GC and AT
    assert np.allclose(vector[2:6], 0.25)                                 # A, C, G, T
    # 6 overlapping 3-mers: AAC ACC CCG CGG GGT GTT, each 1/6
    index_aac = fe.get_feature_names().index("k3_AAC")
    assert np.isclose(vector[index_aac], 1 / 6)


def test_frequency_blocks_add_up_to_one():
    vector = fe.extract_features("ATGCGTACGTTAGCCGATAGCTAGGCTAATCGGATC")
    assert np.isclose(vector[2:6].sum(), 1.0)
    assert np.isclose(vector[K3_START:K4_START].sum(), 1.0, atol=1e-5)
    assert np.isclose(vector[K4_START:].sum(), 1.0, atol=1e-5)
    assert np.isclose(vector[0] + vector[1], 1.0)
    assert vector.min() >= 0.0 and vector.max() <= 1.0


def test_lowercase_and_spaces_around_are_accepted():
    assert np.array_equal(fe.extract_features("  atgcatgc \n"), fe.extract_features("ATGCATGC"))


def test_minimum_length_is_four():
    assert fe.extract_features("ATGC").shape == (326,)
    assert raises_value_error(fe.extract_features, "ATG")


def test_bad_input_raises_clear_errors():
    assert raises_value_error(fe.extract_features, "")
    assert raises_value_error(fe.extract_features, "ATGNNCATG")      # N
    assert raises_value_error(fe.extract_features, "AUGCAUGC")       # RNA
    assert raises_value_error(fe.extract_features, "ATG-CATG")       # gap


# ---------- many sequences ----------
def test_batch_matches_single_sequences():
    sequences = ["ATGCATGCAA", "GGGGCCCCGG", "ATATATATAT", "acgtacgtac"]
    matrix = fe.extract_features_batch(sequences)
    assert matrix.shape == (4, 326) and matrix.dtype == np.float32
    for i, sequence in enumerate(sequences):
        assert np.array_equal(matrix[i], fe.extract_features(sequence))


def test_batch_reports_which_sequence_is_bad():
    try:
        fe.extract_features_batch(["ATGCATGC", "ATGNATGC"])
    except ValueError as error:
        assert "Sequence number 2" in str(error)
    else:
        assert False, "a ValueError was expected"


def test_batch_of_nothing():
    assert fe.extract_features_batch([]).shape == (0, 326)


def test_length_does_not_change_the_feature_count():
    short = fe.extract_features("ATGCATGCAT")
    long = fe.extract_features("ATGC" * 500)
    assert short.shape == long.shape == (326,)