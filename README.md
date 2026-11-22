# DNA Intelligence
AI-Powered DNA Sequence Analysis and Classification System.

Status: under construction (Phase 5 - model training).
## Setup
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Features so far
- **DNA Sequence Analyzer:** type, paste or upload a FASTA file; validation (only A, C, G, T);
  length, base counts and percentages, GC/AT content, bar chart, reverse complement.
- **Biological Pattern Analysis:** k-mer analysis (k = 2 to 6), codon analysis with
  start/stop highlighting, ORF finder for the three forward reading frames.
- Note: an ORF is not proof of a real gene.

## Dataset (Phase 2)
- **Dataset:** Genomic Benchmarks `demo_coding_vs_intergenomic_seqs`
  (Gresova et al., BMC Genomic Data, 2023): fixed 200 bp sequences, coding vs intergenomic.
- **Prepare the data:**
      pip install genomic-benchmarks        # one-time download helper, not needed by the app
      python -c "from genomic_benchmarks.loc2seq import download_dataset; download_dataset('demo_coding_vs_intergenomic_seqs')"
      python -m src.preprocessing --source-folder ~/.genomic_benchmarks/demo_coding_vs_intergenomic_seqs
- **Output (in `data/processed/`):**
  - `dna_dataset.csv`: cleaned sequences with label and train/test split
  - `split_info.json`: split settings and counts
  - `DATASET_INFO.md`: source, labels, filtering and exact counts
- **Reload in Python:**
  `from src.preprocessing import load_train_test`
  `X_train, X_test, y_train, y_test = load_train_test()`
- Labels: 1 = coding, 0 = intergenomic.

## Features for machine learning (Phase 4)
- Each DNA sequence becomes a vector of 326 numbers: GC and AT content, A/C/G/T
  frequencies, and all 64 3-mer and 256 4-mer frequencies (fixed alphabetical vocabulary).
- **Build the matrices:**  `python -m src.feature_extraction`
  (creates `data/processed/features.npz` and `feature_info.json`)
- **Use in Python:**
  `from src.feature_extraction import extract_features, get_feature_matrices`
  `vector = extract_features("ATGC...")`  (one sequence)
  `X_train, X_test, y_train, y_test = get_feature_matrices()`  (whole dataset)
## Run the app
    streamlit run app.py

## Run the tests
    pip install pytest
    python -m pytest -q

## Disclaimer
For learning and demonstration only. Not for clinical or research decisions.
Model predictions on 200 bp benchmark fragments are not gene annotation.