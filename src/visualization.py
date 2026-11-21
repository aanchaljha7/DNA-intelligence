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