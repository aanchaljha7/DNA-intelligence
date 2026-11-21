"""
DNA Intelligence - AI-Powered DNA Sequence Analysis and Classification System
Phase 3: Module 2 (DNA Sequence Analyzer) and Module 3 (Biological Pattern Analysis).

Run with:  streamlit run app.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import sequence_analysis as sa
from src import visualization as viz

st.set_page_config(page_title="DNA Intelligence", page_icon="🧬", layout="wide")

PAGES = [
    "1. Home / Dashboard",
    "2. DNA Sequence Analyzer",
    "3. Biological Pattern Analysis",
    "4. AI Classification",
    "5. Prediction System",
    "6. Model Performance",
    "7. Biology Learning",
]

# A made-up 200 bp example that contains one clear ORF (frame 3, positions 21-146).
EXAMPLE_SEQUENCE = (
    "CGTCAGCACGAAACTTGTTGATGCCGTACCGTAGCTTGTCTGATACATCACAAAACTTGGGAACTCTTTTATCCCGACGC"
    "TCTCTATCCACGCGATTGGATAATTCGCTCAGTAGTAACTTGAATAACCGTTTGCTCTTAACGTAAGCCCAGTGTGAATCG"
    "CTTAAGGGTTAAGTAAGTGTGATGCATACGCCTTTACTT"
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def load_example(key):
    """Button callback: puts the example sequence into the text box."""
    st.session_state[f"{key}_text"] = EXAMPLE_SEQUENCE


def get_valid_sequence(key):
    """
    Shows the input widgets (type/paste or upload a file), then reads and validates
    the sequence. Returns (name, sequence), or (None, None) if there is nothing
    valid to analyse yet. Error and info messages are shown here.
    """
    method = st.radio(
        "Input method",
        ["Type or paste", "Upload FASTA file"],
        horizontal=True,
        key=f"{key}_method",
    )

    raw_text = ""
    if method == "Type or paste":
        # Start with the text from the other page, so you do not need to paste twice.
        if f"{key}_text" not in st.session_state:
            st.session_state[f"{key}_text"] = st.session_state.get("shared_text", "")
        st.text_area(
            "DNA sequence (plain text or FASTA)",
            height=150,
            key=f"{key}_text",
            placeholder="Example: ATGGCC...  or a FASTA record starting with >name",
        )
        st.button("Load example sequence", on_click=load_example, args=(key,), key=f"{key}_example")
        raw_text = st.session_state[f"{key}_text"]
    else:
        uploaded = st.file_uploader(
            "FASTA or text file", type=["fasta", "fa", "fna", "txt"], key=f"{key}_file"
        )
        if uploaded is not None:
            raw_text = uploaded.getvalue().decode("utf-8", errors="ignore")

    st.session_state["shared_text"] = raw_text

    if not raw_text.strip():
        st.info("Enter, paste or upload a DNA sequence above to begin.")
        return None, None

    name, sequence, notes = sa.parse_input(raw_text)
    for note in notes:
        st.info(note)

    is_valid, message = sa.validate_sequence(sequence)
    if not is_valid:
        st.error(message)
        return None, None

    st.success(f"Valid DNA sequence: **{name}** ({len(sequence):,} bp)")
    return name, sequence


def show_figure(fig):
    """Displays a Matplotlib figure in Streamlit and frees its memory."""
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Module 1: Home
# ---------------------------------------------------------------------------
def show_home():
    st.title("🧬 DNA Intelligence")
    st.subheader("AI-Powered DNA Sequence Analysis and Classification System")
    st.write(
        "Explore DNA sequences with the **DNA Sequence Analyzer** and the "
        "**Biological Pattern Analysis** pages. Later phases add a machine learning "
        "classifier for coding vs intergenomic sequences."
    )
    st.info("Pick a module in the sidebar. Use **Load example sequence** to try things quickly.")


# ---------------------------------------------------------------------------
# Module 2: DNA Sequence Analyzer
# ---------------------------------------------------------------------------
def show_sequence_analyzer():
    st.title("DNA Sequence Analyzer")
    st.write("Check a DNA sequence and see its basic properties.")

    name, sequence = get_valid_sequence("analyzer")
    if sequence is None:
        return

    if len(sequence) < 20:
        st.warning("This is a very short sequence, so the percentages are not very meaningful.")

    stats = sa.basic_stats(sequence)

    col1, col2, col3 = st.columns(3)
    col1.metric("Length (bp)", f"{stats['length']:,}")
    col2.metric("GC content", f"{stats['gc_content']}%")
    col3.metric("AT content", f"{stats['at_content']}%")

    st.subheader("Nucleotide counts and percentages")
    left, right = st.columns(2)
    with left:
        table = pd.DataFrame({
            "Base": list(stats["counts"].keys()),
            "Count": list(stats["counts"].values()),
            "Percent": [stats["percentages"][b] for b in stats["counts"]],
        })
        st.dataframe(table, hide_index=True)
    with right:
        show_figure(viz.plot_nucleotide_distribution(stats["counts"], stats["percentages"]))

    st.subheader("Reverse complement")
    st.caption("The opposite DNA strand, written 5' to 3'.")
    reverse = sa.reverse_complement(sequence)
    limit = 2000
    st.code(reverse[:limit], language=None)
    if len(reverse) > limit:
        st.caption(f"Showing the first {limit:,} bases. Download the file to get all of it.")
    st.download_button(
        "Download reverse complement (FASTA)",
        data=f">{name}_reverse_complement\n{reverse}\n",
        file_name="reverse_complement.fasta",
    )


# ---------------------------------------------------------------------------
# Module 3: Biological Pattern Analysis
# ---------------------------------------------------------------------------
def show_kmer_tab(sequence):
    st.write(
        "A **k-mer** is a short DNA 'word' of length k. Counting k-mers shows "
        "which short patterns are common in the sequence."
    )
    k = st.slider("k (length of each k-mer)", sa.K_MIN, sa.K_MAX, sa.K_DEFAULT, key="kmer_k")

    if len(sequence) < k:
        st.warning(f"The sequence has {len(sequence)} bases, which is shorter than k = {k}. "
                   "Choose a smaller k or use a longer sequence.")
        return

    summary = sa.kmer_summary(sequence, k)
    c1, c2, c3 = st.columns(3)
    c1.metric("K-mers counted", f"{summary['total_kmers']:,}")
    c2.metric("Different k-mers seen", f"{summary['distinct_kmers']:,}")
    c3.metric("Possible k-mers (4^k)", f"{summary['possible_kmers']:,}")

    table = sa.kmer_table(sequence, k)
    top_n = st.slider("How many k-mers to show in the chart", 5, 30, 15, key="kmer_top")
    show_figure(viz.plot_kmer_top(table, top_n))

    st.write("Full k-mer table:")
    st.dataframe(table, hide_index=True)


def show_codon_tab(sequence):
    st.write(
        "A **codon** is 3 DNA bases in a row. Codons are read in a **reading frame**: "
        "frame 1 starts at the first base, frame 2 at the second, frame 3 at the third."
    )
    frame = st.radio("Reading frame", [1, 2, 3], horizontal=True, key="codon_frame")

    codons = sa.split_codons(sequence, frame)
    if not codons:
        st.warning("The sequence is too short for codon analysis (at least 3 bases are needed "
                   "in this frame).")
        return

    summary = sa.codon_summary(sequence, frame)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Codons", f"{summary['codon_count']:,}")
    c2.metric("Start codons (ATG)", summary["start_count"])
    c3.metric("Stop codons", summary["stop_count"])
    c4.metric("Ignored end bases", summary["ignored_bases"])
    breakdown = ", ".join(f"{s}: {n}" for s, n in summary["stop_breakdown"].items())
    st.caption(f"Stop codons by type: {breakdown}")

    st.subheader("Codons in this frame")
    st.markdown(
        '<span style="background:#81c784;color:#000;padding:1px 6px;border-radius:3px">ATG</span>'
        " = start codon &nbsp;&nbsp; "
        '<span style="background:#e57373;color:#000;padding:1px 6px;border-radius:3px">TAA / TAG / TGA</span>'
        " = stop codon",
        unsafe_allow_html=True,
    )
    html, truncated = sa.highlight_codons_html(codons)
    st.markdown(html, unsafe_allow_html=True)
    if truncated:
        st.caption("Only the first 600 codons are shown here.")
    st.caption("ATG codes for methionine everywhere. It only works as a start codon at the "
               "beginning of a gene.")

    st.subheader("Codon counts")
    table = sa.codon_table(codons)
    only_special = st.checkbox("Show only start and stop codons", key="codon_special")
    if only_special:
        table = table[table["Type"] != "Other"]
    st.dataframe(table, hide_index=True)


def show_orf_tab(sequence):
    st.warning(
        "**An ORF is not a gene.** An ORF is only a stretch of DNA that starts with ATG and "
        "has no stop codon in the same reading frame. Random DNA contains many ORFs by "
        "chance, especially short ones. Deciding whether an ORF is a real, functional gene "
        "needs more evidence (for example length, conservation, RNA or protein data, and "
        "laboratory tests). This tool searches the three forward frames only, so genes on "
        "the reverse strand are not shown."
    )

    c1, c2 = st.columns(2)
    with c1:
        min_aa = st.number_input(
            "Minimum ORF length (amino acids)", min_value=1, max_value=1000, value=20, step=1,
            key="orf_min",
        )
    with c2:
        include_incomplete = st.checkbox(
            "Also show ORFs with no stop codon (they run off the end of the sequence)",
            key="orf_incomplete",
        )

    if len(sequence) < 6:
        st.info("The sequence is too short for an ORF (at least 6 bases are needed: ATG + a stop codon).")
        return

    orfs = sa.find_orfs(sequence, int(min_aa), include_incomplete)
    if not orfs:
        st.info("No ORFs found with these settings. Try a smaller minimum length.")
        return

    per_frame = sa.orf_count_by_frame(orfs)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ORFs found", len(orfs))
    m2.metric("Longest ORF (aa)", orfs[0]["protein_length"])
    m3.metric("Frame 1 / 2 / 3", f"{per_frame[1]} / {per_frame[2]} / {per_frame[3]}")
    m4.metric("Complete ORFs", sum(1 for o in orfs if o["complete"]))

    show_figure(viz.plot_orf_map(orfs, len(sequence)))
    st.caption("Green = ORF ends with a stop codon. Orange = no stop codon found.")

    st.dataframe(sa.orf_dataframe(orfs), hide_index=True)

    st.subheader("Look at one ORF")
    options = list(range(len(orfs)))
    chosen = st.selectbox(
        "Choose an ORF",
        options,
        format_func=lambda i: (
            f"Frame {orfs[i]['frame']}: {orfs[i]['start']}-{orfs[i]['end']} "
            f"({orfs[i]['protein_length']} aa)"
        ),
        key="orf_choice",
    )
    orf = orfs[chosen]
    st.write("DNA sequence:")
    st.code(orf["dna"], language=None)
    st.write("Protein sequence (one-letter amino acid codes):")
    st.code(orf["protein"], language=None)


def show_pattern_analysis():
    st.title("Biological Pattern Analysis")
    st.write("Look for patterns in a DNA sequence: k-mers, codons and open reading frames.")

    name, sequence = get_valid_sequence("patterns")
    if sequence is None:
        return

    tab_kmer, tab_codon, tab_orf = st.tabs(["K-mer analysis", "Codon analysis", "ORF finder"])
    with tab_kmer:
        show_kmer_tab(sequence)
    with tab_codon:
        show_codon_tab(sequence)
    with tab_orf:
        show_orf_tab(sequence)


# ---------------------------------------------------------------------------
# Modules still to come (placeholders)
# ---------------------------------------------------------------------------
def show_ai_classification():
    st.title("AI Classification")
    st.info("Placeholder: information about how the coding vs intergenomic classifier works.")


def show_prediction():
    st.title("Prediction System")
    st.info("Placeholder: the trained model's predictions will appear here.")


def show_model_performance():
    st.title("Model Performance")
    st.info("Placeholder: accuracy, confusion matrix and other metrics will appear here.")


def show_biology_learning():
    st.title("Biology Learning")
    st.info("Placeholder: short beginner-friendly explanations of DNA concepts.")


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
def main():
    st.sidebar.title("🧬 DNA Intelligence")
    page = st.sidebar.radio("Navigation", PAGES)
    st.sidebar.caption("Phase 3: sequence analysis")
    st.sidebar.warning("For learning and demonstration only.")

    if page == PAGES[0]:
        show_home()
    elif page == PAGES[1]:
        show_sequence_analyzer()
    elif page == PAGES[2]:
        show_pattern_analysis()
    elif page == PAGES[3]:
        show_ai_classification()
    elif page == PAGES[4]:
        show_prediction()
    elif page == PAGES[5]:
        show_model_performance()
    else:
        show_biology_learning()


main()