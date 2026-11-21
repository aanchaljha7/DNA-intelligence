"""
Tests for src/sequence_analysis.py

Run from the DNA_Intelligence folder (with .venv active):
    pip install pytest
    python -m pytest -q
"""

from src import sequence_analysis as sa


# ---------- reading and validating input ----------
def test_parse_plain_text():
    name, sequence, notes = sa.parse_input(" atg c\ncc ")
    assert name == "Pasted sequence"
    assert sequence == "ATGCCC"
    assert notes == []


def test_parse_fasta():
    name, sequence, notes = sa.parse_input(">seq1 demo\nATGC\nAAA\n")
    assert name == "seq1 demo"
    assert sequence == "ATGCAAA"
    assert notes == []


def test_parse_fasta_uses_first_record_only():
    name, sequence, notes = sa.parse_input(">a\nAAAA\n>b\nCCCC")
    assert (name, sequence) == ("a", "AAAA")
    assert len(notes) == 1


def test_parse_removes_numbers_but_keeps_other_symbols():
    _, sequence, notes = sa.parse_input("ATG 10 NN-")
    assert sequence == "ATGNN-"
    assert len(notes) == 1


def test_validate_good_sequence():
    assert sa.validate_sequence("ATGCATGC") == (True, "")


def test_validate_empty():
    ok, message = sa.validate_sequence("")
    assert not ok and "No sequence" in message


def test_validate_n_gives_clear_message():
    ok, message = sa.validate_sequence("ATGNNC")
    assert not ok
    assert "N" in message and "position 4" in message


def test_validate_ambiguous_rna_and_invalid():
    ok, message = sa.validate_sequence("ATGRY")
    assert not ok and "R, Y" in message
    ok, message = sa.validate_sequence("AUGC")
    assert not ok and "RNA" in message
    ok, message = sa.validate_sequence("AT-GC*")
    assert not ok and "Invalid characters" in message


# ---------- basic analysis ----------
def test_counts_percentages_gc_at():
    sequence = "AACCGGTT"
    assert sa.nucleotide_counts(sequence) == {"A": 2, "C": 2, "G": 2, "T": 2}
    assert sa.nucleotide_percentages(sequence)["A"] == 25.0
    assert sa.gc_content(sequence) == 50.0
    assert sa.at_content(sequence) == 50.0
    assert sa.gc_content("GGGG") == 100.0
    assert sa.at_content("GGGG") == 0.0


def test_empty_sequence_does_not_crash():
    assert sa.gc_content("") == 0.0
    assert sa.nucleotide_percentages("") == {"A": 0.0, "C": 0.0, "G": 0.0, "T": 0.0}


def test_reverse_complement():
    assert sa.reverse_complement("ATGC") == "GCAT"
    assert sa.reverse_complement("AAACCC") == "GGGTTT"
    assert sa.reverse_complement(sa.reverse_complement("ATGGCCTTA")) == "ATGGCCTTA"


# ---------- k-mers ----------
def test_kmer_counts_overlap():
    table = sa.kmer_table("AAAA", 2)
    assert list(table["K-mer"]) == ["AA"]
    assert list(table["Count"]) == [3]


def test_kmer_table_and_summary():
    table = sa.kmer_table("ATGATG", 3)
    assert table.loc[0, "K-mer"] == "ATG"
    assert table.loc[0, "Count"] == 2
    assert table.loc[0, "Percent"] == 50.0
    summary = sa.kmer_summary("ATGATG", 3)
    assert summary == {"total_kmers": 4, "distinct_kmers": 3, "possible_kmers": 64}


def test_kmer_sequence_shorter_than_k():
    assert sa.kmer_table("AT", 3).empty
    assert sa.kmer_summary("AT", 3)["total_kmers"] == 0


# ---------- codons ----------
def test_genetic_code():
    assert len(sa.CODON_TABLE) == 64
    assert sa.translate_codon("ATG") == "M"
    assert sa.translate_codon("TGG") == "W"
    assert sorted(c for c, a in sa.CODON_TABLE.items() if a == "*") == ["TAA", "TAG", "TGA"]


def test_genetic_code_matches_biopython_if_installed():
    try:
        from Bio.Seq import Seq
    except ImportError:
        return
    for codon, letter in sa.CODON_TABLE.items():
        assert str(Seq(codon).translate()) == letter


def test_split_codons_frames():
    sequence = "ATGAAATAGC"
    assert sa.split_codons(sequence, 1) == ["ATG", "AAA", "TAG"]
    assert sa.split_codons(sequence, 2) == ["TGA", "AAT", "AGC"]
    assert sa.split_codons(sequence, 3) == ["GAA", "ATA"]
    assert sa.split_codons("AT", 1) == []


def test_codon_summary():
    summary = sa.codon_summary("ATGAAATAGC", 1)
    assert summary["codon_count"] == 3
    assert summary["start_count"] == 1
    assert summary["stop_count"] == 1
    assert summary["stop_breakdown"]["TAG"] == 1
    assert summary["ignored_bases"] == 1


def test_codon_table_types():
    table = sa.codon_table(["ATG", "AAA", "TAG", "AAA"])
    assert table.loc[0, "Codon"] == "AAA" and table.loc[0, "Count"] == 2
    types = dict(zip(table["Codon"], table["Type"]))
    assert types == {"AAA": "Other", "ATG": "Start", "TAG": "Stop"}
    assert sa.codon_table([]).empty


def test_highlight_html():
    html, truncated = sa.highlight_codons_html(["ATG", "AAA", "TAA"])
    assert "#81c784" in html and "#e57373" in html and not truncated
    _, truncated = sa.highlight_codons_html(["AAA"] * 10, max_codons=5)
    assert truncated


# ---------- ORFs ----------
def test_find_simple_orf():
    sequence = "CC" + "ATGAAACCCGGGTAA" + "GG"
    orfs = sa.find_orfs(sequence, min_aa=3)
    assert len(orfs) == 1
    orf = orfs[0]
    assert (orf["frame"], orf["start"], orf["end"]) == (3, 3, 17)
    assert orf["length_bp"] == 15 and orf["protein_length"] == 4
    assert orf["protein"] == "MKPG" and orf["stop_codon"] == "TAA" and orf["complete"]
    assert sa.find_orfs(sequence, min_aa=5) == []


def test_incomplete_orf_only_when_requested():
    assert sa.find_orfs("ATGAAACCC", min_aa=1) == []
    orfs = sa.find_orfs("ATGAAACCC", min_aa=1, include_incomplete=True)
    assert len(orfs) == 1
    assert orfs[0]["complete"] is False and orfs[0]["protein"] == "MKP"
    assert orfs[0]["stop_codon"] == "none"


def test_nested_atg_is_not_reported_separately():
    orfs = sa.find_orfs("ATGATGAAATAA", min_aa=1)
    assert len(orfs) == 1
    assert orfs[0]["start"] == 1 and orfs[0]["protein"] == "MMK"


def test_orfs_in_different_frames_and_table():
    sequence = "ATGAAATAA" + "CATGAAATAA"   # frame 1 and frame 2 ORFs
    orfs = sa.find_orfs(sequence, min_aa=1)
    assert sa.orf_count_by_frame(orfs)[1] == 1
    assert sum(sa.orf_count_by_frame(orfs).values()) == len(orfs)
    table = sa.orf_dataframe(orfs)
    assert list(table.columns)[0] == "Frame" and len(table) == len(orfs)


def test_orf_on_very_short_or_empty_sequences():
    assert sa.find_orfs("", 1) == []
    assert sa.find_orfs("AT", 1) == []
    assert sa.orf_dataframe([]).empty


def test_translate_dna_stops_at_stop_codon():
    assert sa.translate_dna("ATGAAATAA") == "MK"
    assert sa.translate_dna("ATGAAA") == "MK"