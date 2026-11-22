"""
visualization.py  -  Matplotlib / Seaborn charts used by the Streamlit app.

Every function returns a Matplotlib figure. The Streamlit page shows it with
st.pyplot(fig) and then closes it with plt.close(fig).
"""

import matplotlib.pyplot as plt
import seaborn as sns

BASE_COLORS = {"A": "#4caf50", "C": "#2196f3", "G": "#ff9800", "T": "#f44336"}


def plot_nucleotide_distribution(counts, percentages):
    """Bar chart of A, C, G, T counts, with the percentage written above each bar."""
    bases = list(counts.keys())
    values = [counts[base] for base in bases]
    labels = [f"{percentages[base]}%" for base in bases]
    colors = [BASE_COLORS[base] for base in bases]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = ax.bar(bases, values, color=colors)
    ax.bar_label(bars, labels=labels, padding=3)
    ax.set_xlabel("Nucleotide")
    ax.set_ylabel("Count")
    ax.set_title("Nucleotide distribution")
    ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_kmer_top(kmer_table, top_n=15):
    """Horizontal bar chart of the most common k-mers."""
    top = kmer_table.head(top_n)
    height = max(2.5, 0.32 * len(top) + 1)

    fig, ax = plt.subplots(figsize=(6, height))
    sns.barplot(data=top, x="Count", y="K-mer", color="#2196f3", ax=ax)
    ax.set_title(f"Top {len(top)} most common k-mers")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_orf_map(orfs, sequence_length, max_orfs=300):
    """
    Simple map of the three forward reading frames.
    Green bars = ORFs with a stop codon, orange bars = ORFs that run off the end.
    """
    fig, ax = plt.subplots(figsize=(7, 2.6))

    for orf in orfs[:max_orfs]:
        color = "#4caf50" if orf["complete"] else "#ff9800"
        ax.barh(
            y=orf["frame"],
            width=orf["end"] - orf["start"] + 1,
            left=orf["start"] - 1,
            height=0.55,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlim(0, max(sequence_length, 1))
    ax.set_ylim(3.6, 0.4)   # frame 1 at the top
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["Frame 1", "Frame 2", "Frame 3"])
    ax.set_xlabel("Position in sequence (bp)")
    ax.set_title("ORF map (forward strand)")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig



# ---------------------------------------------------------------------------
# Phase 6: charts for the prediction page and the performance dashboard
# ---------------------------------------------------------------------------
def plot_probability_bar(probability_coding):
    """One horizontal bar split into 'non-coding' and 'coding' parts (a probability from 0 to 1)."""
    probability_non_coding = 1 - probability_coding

    fig, ax = plt.subplots(figsize=(6, 1.3))
    ax.barh([0], [probability_non_coding], color="#ff9800")
    ax.barh([0], [probability_coding], left=[probability_non_coding], color="#4caf50")
    for start, width, text in ((0, probability_non_coding, "Non-coding"),
                               (probability_non_coding, probability_coding, "Coding")):
        if width > 0.12:      # only write inside a bar if it is wide enough
            ax.text(start + width / 2, 0, f"{text} {width * 100:.0f}%",
                    ha="center", va="center", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Probability")
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(matrix, title):
    """
    Heatmap of a 2x2 confusion matrix [[TN, FP], [FN, TP]].
    Each box shows the count and the percentage of its true class (its row).
    """
    labels = ["Non-coding", "Coding"]
    notes = []
    for row in matrix:
        total = sum(row)
        notes.append([f"{value:,}\n({value / total * 100:.1f}%)" if total else f"{value:,}"
                      for value in row])

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(matrix, annot=notes, fmt="", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_metric_comparison(results_by_model):
    """
    Grouped bars: accuracy, precision, recall and F1 for each model.
    results_by_model = {"Random Forest": {"accuracy": ..., "precision": ..., ...}, ...}
    """
    measures = [("accuracy", "Accuracy"), ("precision", "Precision"),
                ("recall", "Recall"), ("f1", "F1")]
    colors = ["#2196f3", "#ff9800", "#4caf50", "#9c27b0"]
    n_models = len(results_by_model)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, (title, results) in enumerate(results_by_model.items()):
        positions = [j + i * width - 0.4 + width / 2 for j in range(len(measures))]
        values = [results[key] for key, _ in measures]
        bars = ax.bar(positions, values, width=width * 0.95, label=title, color=colors[i % len(colors)])
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=2, fontsize=8)

    ax.set_xticks(range(len(measures)))
    ax.set_xticklabels([label for _, label in measures])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score (test set)")
    ax.set_title("Model comparison (precision, recall and F1 are for the coding class)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=n_models, frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_roc_curves(roc_by_title):
    """
    ROC curves of all models in one plot.
    roc_by_title = {"Random Forest": {"fpr": [...], "tpr": [...], "auc": 0.95}, ...}
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for title, roc in roc_by_title.items():
        ax.plot(roc["fpr"], roc["tpr"], linewidth=2, label=f"{title} (AUC = {roc['auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guessing")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate (non-coding called coding)")
    ax.set_ylabel("True positive rate (coding found)")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_feature_importance(importance_table, top_n=15):
    """Horizontal bars of the most important features of the Random Forest."""
    top = importance_table.head(top_n)
    height = max(2.5, 0.32 * len(top) + 1)

    fig, ax = plt.subplots(figsize=(6, height))
    sns.barplot(data=top, x="Importance", y="Feature", color="#4caf50", ax=ax)
    ax.set_title(f"Top {len(top)} features used by the Random Forest")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig