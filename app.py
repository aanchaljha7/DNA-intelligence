"""
DNA Intelligence - AI-Powered DNA Sequence Analysis and Classification System
Phase 3: Module 2 (DNA Sequence Analyzer) and Module 3 (Biological Pattern Analysis).

Run with:  streamlit run app.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import evaluation as ev
from src import model_training as mt
from src import prediction as pr
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
    st.write("Analyse DNA sequences, explore their patterns, and use machine learning to tell "
             "**coding** from **non-coding (intergenomic)** 200 bp fragments.")

    # Quick numbers, read from the trained models' saved results
    try:
        models = ev.load_results()["models"]
    except FileNotFoundError:
        models = None
    if models:
        c1, c2, c3 = st.columns(3)
        c1.metric("Random Forest accuracy", f"{models['random_forest']['accuracy'] * 100:.1f}%")
        c2.metric("SVM accuracy", f"{models['svm']['accuracy'] * 100:.1f}%")
        c3.metric("Test sequences", f"{models['svm']['test_samples']:,}")

    needed = [("Random Forest model", mt.RF_MODEL_PATH), ("SVM model", mt.SVM_MODEL_PATH),
              ("metrics.json", mt.METRICS_PATH), ("roc_data.json", ev.ROC_PATH)]
    missing = [name for name, path in needed if not path.exists()]
    if missing:
        st.warning("Missing files: " + ", ".join(missing) + ". Create them with "
                   "`python -m src.model_training` and `python -m src.evaluation`.")

    st.subheader("What you can do")
    left, right = st.columns(2)
    left.markdown(
        "**🔬 DNA Sequence Analyzer**  \nBase counts, GC content, reverse complement.\n\n"
        "**🧩 Biological Pattern Analysis**  \nK-mers, codons and ORFs.\n\n"
        "**🤖 AI Classification**  \nHow the classifier works."
    )
    right.markdown(
        "**🎯 Prediction System**  \nCoding or non-coding? Pick a model.\n\n"
        "**📊 Model Performance**  \nMetrics, confusion matrices, ROC curves.\n\n"
        "**📚 Biology Learning**  \nShort explanations of the key ideas."
    )

    st.info("**Quick start:** open *DNA Sequence Analyzer* or *Prediction System* in the sidebar "
            "and click **Load example sequence**.")
    st.caption("For learning and demonstration only. Not for medical, clinical or research decisions.")


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
    st.write("**The task:** decide whether a 200 bp DNA fragment looks like a **coding** "
             "sequence or a **non-coding (intergenomic)** one.")
    st.markdown(
        "**How it works**\n"
        "1. The sequence becomes 326 numbers (features): GC and AT content, the frequency of "
        "A, C, G and T, and the frequency of every 3-mer and 4-mer.\n"
        "2. A trained model turns those numbers into a prediction.\n"
        "3. Two models are available:\n"
        "   - **Random Forest**: many decision trees vote. It gives a probability.\n"
        "   - **SVM**: finds a boundary between the two classes. It gives a score."
    )
    try:
        models = ev.load_results()["models"]
        st.write(f"The Random Forest learned from {models['random_forest']['train_samples']:,} "
                 f"training sequences and the SVM from {models['svm']['train_samples']:,}. Both "
                 f"were tested on {models['svm']['test_samples']:,} sequences they had never seen. "
                 "See *Model Performance* for the results.")
    except FileNotFoundError:
        st.info("Train the models to see their training details here.")
    st.warning("The models only recognise composition patterns in this benchmark dataset. They "
               "are not gene finders and should not be trusted for other lengths or organisms.")


def show_prediction():
    st.title("DNA Prediction System")
    st.warning(pr.DISCLAIMER)

    if not all(pr.models_available().values()):
        st.error("The trained model files were not found in the models/ folder. "
                 "Train them first with:  python -m src.model_training")
        return

    model_name = st.radio("Choose a model", list(pr.MODEL_CHOICES.keys()), horizontal=True,
                          key="predict_model")
    model_key = pr.MODEL_CHOICES[model_name]

    name, sequence = get_valid_sequence("predict")
    if sequence is None:
        return

    try:
        result = pr.predict_sequence(sequence, model_key)
    except (ValueError, FileNotFoundError) as error:
        st.error(str(error))
        return

    warning = pr.length_warning(result["length"])
    if warning:
        level, text = warning
        if level == "warning":
            st.warning(text)
        else:
            st.info(text)

    st.subheader(f"Prediction ({model_name})")
    col1, col2 = st.columns(2)
    col1.metric("Predicted class", result["label_text"])
    if result["probability_coding"] is not None:
        col2.metric("Probability of coding", f"{result['probability_coding'] * 100:.1f}%")
        show_figure(viz.plot_probability_bar(result["probability_coding"]))
        st.caption("This probability is the forest's own confidence. It is not the chance "
                   "that the sequence is really a coding sequence in a genome.")
    else:
        col2.metric("SVM score", f"{result['score']:+.3f}")
        st.caption("The SVM gives no probability. Its score is the distance from the decision "
                   "boundary: above 0 means coding, below 0 means non-coding, and the further "
                   "from 0, the more clearly the sequence falls on that side. It is not a "
                   "percentage.")
    st.caption("For learning and demonstration only. This is not gene prediction.")

    st.subheader("Your sequence")
    stats = sa.basic_stats(sequence)
    m1, m2, m3 = st.columns(3)
    m1.metric("Length (bp)", f"{stats['length']:,}")
    m2.metric("GC content", f"{stats['gc_content']}%")
    m3.metric("AT content", f"{stats['at_content']}%")
    st.dataframe(pd.DataFrame({
        "Base": list(stats["counts"].keys()),
        "Count": list(stats["counts"].values()),
        "Frequency (%)": [stats["percentages"][b] for b in stats["counts"]],
    }), hide_index=True)

    st.subheader("Key features")
    st.write("These are the features the Random Forest relies on most (read from the trained "
             "forest). The table shows their values in your sequence. It is shown for either model.")
    forest = pr.get_model("random_forest")
    st.dataframe(pr.key_features_table(result["features"], forest, top_n=10), hide_index=True)
    st.caption("All 326 features are frequencies between 0 and 1: GC and AT content, the four "
               "bases, and every 3-mer and 4-mer. Importance shows what the forest used, "
               "not a biological reason.")


def show_model_performance():
    st.title("Model Performance Dashboard")

    try:
        report = ev.load_results()
    except FileNotFoundError as error:
        st.error(str(error))
        return

    models = report["models"]
    titles = {key: ev.MODEL_TITLES[key] for key in ev.MODEL_TITLES if key in models}
    if not titles:
        st.error("No model results were found in models/metrics.json.")
        return

    try:
        roc_models = ev.load_roc_data()["models"]
    except FileNotFoundError:
        roc_models = None

    test_samples = next(iter(models.values()))["test_samples"]
    st.caption(
        f"Models trained on {report['created_on']} with scikit-learn "
        f"{report['scikit_learn_version']}. Every result below was measured on the same "
        f"{test_samples:,} held-out test sequences, which the models never saw during training."
    )

    st.subheader("Side-by-side comparison")
    st.dataframe(ev.comparison_table(models, roc_models))
    show_figure(viz.plot_metric_comparison({titles[k]: models[k] for k in titles}))
    trained = ", ".join(f"{titles[k]} on {models[k]['train_samples']:,} sequences" for k in titles)
    st.caption(f"Training sizes: {trained}. Precision, recall and F1 are for the coding class "
               "(the positive class).")

    st.subheader("Confusion matrices")
    for column, key in zip(st.columns(len(titles)), titles):
        with column:
            show_figure(viz.plot_confusion_matrix(models[key]["confusion_matrix"], titles[key]))
    st.caption("Rows are the true class, columns are what the model predicted. "
               "The percentages are shares of each true class.")

    st.subheader("Classification reports")
    for column, key in zip(st.columns(len(titles)), titles):
        with column:
            st.write(f"**{titles[key]}**")
            st.dataframe(ev.classification_report_table(models[key]), hide_index=True)

    st.subheader("ROC curves")
    if roc_models:
        show_figure(viz.plot_roc_curves({titles[k]: roc_models[k] for k in titles if k in roc_models}))
        st.caption("A ROC curve shows how well a model's score separates the two classes at every "
                   "possible cut-off. The closer to the top-left corner (AUC near 1), the better. "
                   "The dashed diagonal (AUC 0.5) is random guessing. The forest uses its "
                   "probability, the SVM its decision score.")
    else:
        st.info("ROC curves are not available yet. Create the data (no re-training needed) with:  "
                "python -m src.evaluation")

    st.subheader("Random Forest feature importance")
    try:
        forest = pr.get_model("random_forest")
    except (ValueError, FileNotFoundError) as error:
        st.error(str(error))
        return
    top_n = st.slider("Number of features to show", 5, 30, 15, key="importance_top")
    table = ev.importance_table(forest)
    show_figure(viz.plot_feature_importance(table, top_n))
    st.dataframe(table.head(top_n), hide_index=True)
    st.write("Importance by kind of feature:")
    st.dataframe(ev.importance_by_group(forest), hide_index=True)
    st.caption("Importance shows which features the forest used most when splitting. It does not "
               "prove that a feature is biologically responsible for a sequence being coding.")

    st.warning("These scores describe 200 bp benchmark fragments from one dataset, measured on "
               "one train/test split. They do not show that the models can find genes in real "
               "genomes.")


LEARNING_TOPICS = [
    ("DNA and its four bases",
     "DNA is a long chain of four bases: A (adenine), C (cytosine), G (guanine) and T (thymine). "
     "A pairs with T and C pairs with G. The order of the bases carries information."),
    ("GC content",
     "GC content is the percentage of bases that are G or C. It differs between organisms and "
     "between regions of a genome, so it is a simple but useful feature."),
    ("Reverse complement",
     "DNA has two strands that run in opposite directions. The reverse complement is the other "
     "strand read 5' to 3': swap A with T and C with G, then reverse the order."),
    ("Codons and the genetic code",
     "A codon is three bases in a row. Most codons stand for an amino acid, and amino acids join "
     "to make a protein. ATG usually marks a start; TAA, TAG and TGA are stop codons."),
    ("Reading frames and ORFs",
     "A sequence can be read in three frames. An open reading frame (ORF) starts at ATG and runs "
     "to the next stop codon. An ORF is only a candidate: random DNA contains many ORFs, so it "
     "is not proof of a gene."),
    ("Coding and non-coding DNA",
     "Coding DNA holds the instructions for proteins; non-coding DNA does not. In this project's "
     "dataset the non-coding class is called 'intergenomic', and the models learn to tell the two "
     "apart from patterns in the sequence."),
    ("K-mers",
     "A k-mer is a short 'word' of k bases, such as the 3-mer ATG. Counting k-mers captures "
     "local patterns, which is how the models 'see' a sequence."),
    ("Reading the model results",
     "Accuracy is the share of correct predictions. Precision is how many sequences called coding "
     "really are. Recall is how many coding sequences were found. F1 combines precision and "
     "recall. The ROC curve and its AUC show how well the score separates the classes "
     "(1.0 is perfect, 0.5 is guessing)."),
]


def show_biology_learning():
    st.title("Biology Learning")
    st.write("Short explanations of the ideas used in this app.")
    for title, text in LEARNING_TOPICS:
        with st.expander(title):
            st.write(text)    


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
def main():
    st.sidebar.title("🧬 DNA Intelligence")
    page = st.sidebar.radio("Navigation", PAGES)
    st.sidebar.caption("AI-powered DNA sequence analysis")
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