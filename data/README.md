# Data

This directory contains the benchmark split files used by CrossPPI. Each file stores protein pairs and binary interaction labels.

## Format

Most files use three columns:

```text
protein_a_id    protein_b_id    label
```

where `label` is `1` for interacting protein pairs and `0` for non-interacting protein pairs.

## Directory Structure

```text
data/
├── yeast/
│   ├── train1.txt ... train5.txt
│   └── valid1.txt ... valid5.txt
├── multi_species/
│   ├── train_10_final.tsv, test_10_final.tsv
│   ├── train_25_final.tsv, test_25_final.tsv
│   ├── train_any_final.tsv, test_any_final.tsv
│   ├── train_cold_final.tsv
│   ├── test_s1_final.tsv
│   └── test_s2_final.tsv
└── human_virus/
    ├── VF1/
    ├── VF2/
    ├── VF3/
    ├── VF4/
    └── VF5/
```

## Notes

Only text split files are included in this release. Raw PDB files, processed `.pt` files, extracted ESM-2/ProteinMPNN feature files, and trained checkpoints are not included because they are large generated artifacts.

To run the full pipeline, prepare the corresponding PDB files locally and pass their directory to `preprocess_universal.py` using `--pdb-dir`.
