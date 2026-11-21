"""
sequence_analysis.py  -  Phase 3: DNA sequence analysis logic

All the "thinking" lives here, so the Streamlit pages (app.py) stay short.
Nothing in this file uses Streamlit, so every function can be tested on its own.

Contents:
  1. Reading and validating input (plain text or FASTA)
  2. Basic analysis: counts, percentages, GC/AT content, reverse complement
  3. K-mer analysis
  4. Codon analysis
  5. ORF (open reading frame) detection in the three forward frames

IMPORTANT (scientific note): an ORF is only a stretch of DNA that starts with ATG
and has no stop codon in the same reading frame. It is NOT proof of a real gene.
"""

from collections import Counter

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_BASES = "ACGT"
START_CODON = "ATG"
STOP_CODONS = ("TAA", "TAG", "TGA")

K_MIN = 2
K_MAX = 6
K_DEFAULT = 3

MAX_LENGTH = 1_000_000          # protects the app from huge inputs
IUPAC_AMBIGUOUS = set("RYKMSWBDHV")   # ambiguity codes (N is handled separately)

COMPLEMENT = str.maketrans("ACGT", "TGCA")

# The standard genetic code (NCBI translation table 1).
# Built from the usual 64-letter string; codons are ordered T, C, A, G.
_AMINO_ACIDS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
_BASE_ORDER = "TCAG"
CODON_TABLE = {}
_index = 0
for _b1 in _BASE_ORDER:
    for _b2 in _BASE_ORDER:
        for _b3 in _BASE_ORDER:
            CODON_TABLE[_b1 + _b2 + _b3] = _AMINO_ACIDS[_index]
            _index += 1

AMINO_ACID_NAMES = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "E": "Glu", "Q": "Gln", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
}


# ---------------------------------------------------------------------------
# 1. Reading and validating input
# ---------------------------------------------------------------------------
def parse_input(raw_text):
    """
    Accepts plain DNA text OR FASTA text.
    Returns (name, sequence, notes):
      name     - FASTA header or "Pasted sequence"
      sequence - UPPERCASE, with spaces/newlines/digits removed
      notes    - list of friendly messages (e.g. "only the first FASTA record is used")
    Other characters (N, -, *, ...) are kept on purpose, so validation can report them.
    """
    text = raw_text.strip()
    notes = []

    if text.startswith(">"):
        records = []   # each record = [name, list of sequence lines]
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                records.append([line[1:].strip() or "unnamed", []])
            else:
                records[-1][1].append(line)
        name, lines = records[0]
        sequence_raw = "".join(lines)
        if len(records) > 1:
            notes.append(
                f"The input has {len(records)} FASTA records. "
                f"Only the first one ('{name}') is analysed."
            )
    else:
        name = "Pasted sequence"
        sequence_raw = text

    if any(ch.isdigit() for ch in sequence_raw):
        notes.append("Numbers were removed from the input (for example GenBank position numbers).")

    sequence = "".join(ch for ch in sequence_raw if not ch.isspace() and not ch.isdigit())
    return name, sequence.upper(), notes


def validate_sequence(sequence):
    """
    Checks a cleaned sequence. Returns (is_valid, error_message).
    Only A, C, G and T are accepted. The message explains exactly what is wrong.
    """
    if len(sequence) == 0:
        return False, "No sequence found. Please type, paste or upload a DNA sequence."

    if len(sequence) > MAX_LENGTH:
        return False, (
            f"The sequence is too long ({len(sequence):,} bp). "
            f"The limit is {MAX_LENGTH:,} bp."
        )

    bad = set(sequence) - set(VALID_BASES)
    if not bad:
        return True, ""

    problems = []
    if "N" in bad:
        problems.append(
            f"**N** (unknown base) appears {sequence.count('N')} time(s), "
            f"first at position {sequence.index('N') + 1}."
        )

    ambiguous = sorted(bad & IUPAC_AMBIGUOUS)
    if ambiguous:
        first = min(sequence.index(c) for c in ambiguous) + 1
        problems.append(
            "Ambiguous base codes found: **" + ", ".join(ambiguous) + "** "
            f"(first at position {first}). These mean 'one of several bases'."
        )

    if "U" in bad:
        problems.append("**U** found: this looks like RNA. Please use T instead of U.")

    other = sorted(bad - IUPAC_AMBIGUOUS - {"N", "U"})
    if other:
        shown = ", ".join(repr(c) for c in other)
        problems.append(f"Invalid characters: {shown}.")

    message = (
        "This sequence cannot be analysed:\n\n- "
        + "\n- ".join(problems)
        + "\n\nOnly **A, C, G and T** are allowed."
    )
    return False, message


# ---------------------------------------------------------------------------
# 2. Basic analysis
# ---------------------------------------------------------------------------
def nucleotide_counts(sequence):
    """Returns {'A': n, 'C': n, 'G': n, 'T': n}."""
    return {base: sequence.count(base) for base in VALID_BASES}


def nucleotide_percentages(sequence):
    """Returns the percentage of each base (rounded to 2 decimals)."""
    length = len(sequence)
    if length == 0:
        return {base: 0.0 for base in VALID_BASES}
    return {base: round(sequence.count(base) / length * 100, 2) for base in VALID_BASES}


def gc_content(sequence):
    """GC content as a percentage."""
    if len(sequence) == 0:
        return 0.0
    return round((sequence.count("G") + sequence.count("C")) / len(sequence) * 100, 2)


def at_content(sequence):
    """AT content as a percentage."""
    if len(sequence) == 0:
        return 0.0
    return round((sequence.count("A") + sequence.count("T")) / len(sequence) * 100, 2)


def reverse_complement(sequence):
    """The opposite DNA strand, read 5' to 3'."""
    return sequence.translate(COMPLEMENT)[::-1]


def basic_stats(sequence):
    """Everything the Sequence Analyzer page needs, in one dictionary."""
    return {
        "length": len(sequence),
        "counts": nucleotide_counts(sequence),
        "percentages": nucleotide_percentages(sequence),
        "gc_content": gc_content(sequence),
        "at_content": at_content(sequence),
    }


# ---------------------------------------------------------------------------
# 3. K-mer analysis
# ---------------------------------------------------------------------------
def kmer_table(sequence, k):
    """
    Counts every overlapping k-mer (a DNA 'word' of length k).
    Returns a table with columns: K-mer, Count, Percent (sorted, most common first).
    If the sequence is shorter than k, the table is empty.
    """
    columns = ["K-mer", "Count", "Percent"]
    total = len(sequence) - k + 1
    if k < 1 or total <= 0:
        return pd.DataFrame(columns=columns)

    counts = Counter(sequence[i:i + k] for i in range(total))
    table = pd.DataFrame(list(counts.items()), columns=["K-mer", "Count"])
    table["Percent"] = (table["Count"] / total * 100).round(2)
    table = table.sort_values(["Count", "K-mer"], ascending=[False, True])
    return table.reset_index(drop=True)


def kmer_summary(sequence, k):
    """Small summary: total k-mers, distinct k-mers seen, and how many are possible."""
    table = kmer_table(sequence, k)
    return {
        "total_kmers": max(len(sequence) - k + 1, 0),
        "distinct_kmers": int(len(table)),
        "possible_kmers": 4 ** k,
    }


# ---------------------------------------------------------------------------
# 4. Codon analysis
# ---------------------------------------------------------------------------
def translate_codon(codon):
    """Returns the one-letter amino acid for a codon ('*' means stop)."""
    return CODON_TABLE.get(codon, "?")


def split_codons(sequence, frame=1):
    """
    Cuts the sequence into codons (3 bases) starting at frame 1, 2 or 3.
    Bases left over at the end (fewer than 3) are ignored.
    """
    start = frame - 1
    return [sequence[i:i + 3] for i in range(start, len(sequence) - 2, 3)]


def codon_type(codon):
    """'Start' for ATG, 'Stop' for TAA/TAG/TGA, otherwise 'Other'."""
    if codon == START_CODON:
        return "Start"
    if codon in STOP_CODONS:
        return "Stop"
    return "Other"


def amino_acid_label(codon):
    """Nice text such as 'M (Met)' or 'Stop'."""
    letter = translate_codon(codon)
    if letter == "*":
        return "Stop"
    return f"{letter} ({AMINO_ACID_NAMES.get(letter, '?')})"


def codon_table(codons):
    """
    Counts each codon. Columns: Codon, Amino acid, Type, Count, Percent.
    """
    columns = ["Codon", "Amino acid", "Type", "Count", "Percent"]
    if not codons:
        return pd.DataFrame(columns=columns)

    counts = Counter(codons)
    total = len(codons)
    rows = []
    for codon, count in counts.items():
        rows.append({
            "Codon": codon,
            "Amino acid": amino_acid_label(codon),
            "Type": codon_type(codon),
            "Count": count,
            "Percent": round(count / total * 100, 2),
        })
    table = pd.DataFrame(rows, columns=columns)
    table = table.sort_values(["Count", "Codon"], ascending=[False, True])
    return table.reset_index(drop=True)


def codon_summary(sequence, frame=1):
    """Numbers shown above the codon table."""
    codons = split_codons(sequence, frame)
    leftover = max(len(sequence) - (frame - 1), 0) % 3
    return {
        "codon_count": len(codons),
        "start_count": codons.count(START_CODON),
        "stop_count": sum(1 for c in codons if c in STOP_CODONS),
        "stop_breakdown": {stop: codons.count(stop) for stop in STOP_CODONS},
        "ignored_bases": leftover,
    }


def highlight_codons_html(codons, max_codons=600):
    """
    Returns (html, was_truncated). Start codons (ATG) are green, stop codons are red.
    Only the first max_codons codons are shown so the page stays fast.
    """
    shown = codons[:max_codons]
    parts = []
    for codon in shown:
        if codon == START_CODON:
            parts.append(
                '<span style="background:#81c784;color:#000;padding:1px 4px;'
                f'border-radius:3px;font-weight:bold">{codon}</span>'
            )
        elif codon in STOP_CODONS:
            parts.append(
                '<span style="background:#e57373;color:#000;padding:1px 4px;'
                f'border-radius:3px;font-weight:bold">{codon}</span>'
            )
        else:
            parts.append(codon)

    html = (
        '<div style="font-family:monospace;line-height:2.1;word-wrap:break-word">'
        + " ".join(parts)
        + "</div>"
    )
    return html, len(codons) > max_codons


# ---------------------------------------------------------------------------
# 5. ORF detection (three forward frames only)
# ---------------------------------------------------------------------------
def translate_dna(dna):
    """Translates DNA to a protein string, stopping at the first stop codon."""
    protein = []
    for i in range(0, len(dna) - 2, 3):
        letter = translate_codon(dna[i:i + 3])
        if letter == "*":
            break
        protein.append(letter)
    return "".join(protein)


def find_orfs(sequence, min_aa=20, include_incomplete=False):
    """
    Finds ORFs in the three FORWARD reading frames.

    Rules used (kept simple on purpose):
      - An ORF starts at an ATG.
      - It ends at the first stop codon (TAA, TAG, TGA) in the same frame.
      - After a start codon is found, any later ATG inside the same ORF is ignored,
        so we report the longest ORF for that stop codon.
      - ORFs shorter than min_aa amino acids are dropped.
      - If include_incomplete is True, an ORF that runs off the end of the sequence
        (no stop codon found) is also reported and marked as not complete.

    Returns a list of dictionaries. Positions are 1-based and inclusive.
    """
    orfs = []

    for frame in (1, 2, 3):
        start_pos = None            # index (0-based) of the current ATG, if any
        i = frame - 1
        while i + 3 <= len(sequence):
            codon = sequence[i:i + 3]
            if start_pos is None:
                if codon == START_CODON:
                    start_pos = i
            elif codon in STOP_CODONS:
                _add_orf(orfs, sequence, frame, start_pos, i + 3, True, min_aa)
                start_pos = None
            i += 3

        if start_pos is not None and include_incomplete:
            end = start_pos + ((len(sequence) - start_pos) // 3) * 3
            _add_orf(orfs, sequence, frame, start_pos, end, False, min_aa)

    orfs.sort(key=lambda orf: (-orf["protein_length"], orf["start"]))
    return orfs


def _add_orf(orfs, sequence, frame, start_index, end_index, complete, min_aa):
    """Helper for find_orfs: builds one ORF record and adds it if long enough."""
    dna = sequence[start_index:end_index]
    protein = translate_dna(dna)
    if len(protein) < min_aa:
        return
    orfs.append({
        "frame": frame,
        "start": start_index + 1,
        "end": end_index,
        "length_bp": len(dna),
        "protein_length": len(protein),
        "complete": complete,
        "stop_codon": dna[-3:] if complete else "none",
        "dna": dna,
        "protein": protein,
    })


def orf_dataframe(orfs, protein_preview=30):
    """Turns the ORF list into a table for display (long proteins are shortened)."""
    columns = ["Frame", "Start", "End", "Length (bp)", "Protein length (aa)",
               "Stop codon", "Complete", "Protein (preview)"]
    rows = []
    for orf in orfs:
        preview = orf["protein"]
        if len(preview) > protein_preview:
            preview = preview[:protein_preview] + "..."
        rows.append([
            orf["frame"], orf["start"], orf["end"], orf["length_bp"],
            orf["protein_length"], orf["stop_codon"],
            "Yes" if orf["complete"] else "No", preview,
        ])
    return pd.DataFrame(rows, columns=columns)


def orf_count_by_frame(orfs):
    """Returns {1: n, 2: n, 3: n}."""
    return {frame: sum(1 for orf in orfs if orf["frame"] == frame) for frame in (1, 2, 3)}