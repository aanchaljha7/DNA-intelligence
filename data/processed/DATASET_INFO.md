# Dataset information

## Name and source
- **Dataset:** Genomic Benchmarks - `demo_coding_vs_intergenomic_seqs`
- **Paper:** Gresova K. et al., "Genomic benchmarks: a collection of datasets for genomic sequence classification", BMC Genomic Data 24, 25 (2023).
- **Code / package:** https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks
- **Files used in this project:** local folder supplied by the user (/home/proxxy/.genomic_benchmarks/demo_coding_vs_intergenomic_seqs)
- **Prepared on:** 2026-11-21

## What the data is
- Every sequence is exactly **200 bp** long (fixed length).
- Two classes: **coding** (label 1) and **intergenomic** (label 0).
- Intergenomic is used here as the 'non-coding' class, following the dataset's naming.

## How the labels were obtained
- The labels come from the dataset authors, not from us. The original dataset is stored as one group of sequences per class, and the class name is the label.
- The authors describe the 'demo' datasets as computationally generated for their benchmark collection (see the paper above for the exact procedure).
- We did not check the labels experimentally.

## Filtering done in this project
- Rows loaded: **100000**
- Removed, length not 200 bp: **0**
- Removed, characters other than A/C/G/T: **0**
- Removed, same sequence with both labels: **0**
- Removed, exact duplicate sequences: **81**
- **Rows kept: 99919**

## Final class counts (after cleaning)
- Coding sequences: **49927**
- Intergenomic (non-coding) sequences: **49992**

## Train / test split
- Method: stratified split, test size 20%, random_state=42.
- The split is stored in the `split` column of `dna_dataset.csv`. The original benchmark split is kept in the `original_split` column.

| Split | Coding | Intergenomic | Total |
|---|---|---|---|
| train | 39942 | 39993 | 79935 |
| test | 9985 | 9999 | 19984 |

## Limitations (please read)
- This is a benchmark of short 200 bp fragments. A model trained on it is a learning exercise, not a gene finder.
- Results should not be assumed to hold for other lengths, other species or real genome annotation.
