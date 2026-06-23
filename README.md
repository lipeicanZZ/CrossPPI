# CrossPPI

CrossPPI is a sequence-structure dual-modal framework for protein-protein interaction prediction. It combines residue-level sequence representations from a frozen ESM-2 encoder with structure-aware residue representations from a frozen ProteinMPNN encoder. The two modalities are projected into a shared latent space, fused within each protein, and passed through bidirectional cross-protein attention for final PPI prediction.

This release provides a compact, dataset-agnostic implementation suitable for reproducing the core workflow.

## Files

```text
CrossPPI_release/
├── preprocess_universal.py   # Extract protein sequences and N/CA/C/O backbone coordinates from PDB files
├── features.py               # Extract frozen ESM-2 and ProteinMPNN residue-level features
├── model_crossppi.py         # Train and evaluate the CrossPPI classifier
├── requirements.txt          # Python package requirements
├── environment.yml           # Optional conda environment file
├── LICENSE                   # Open-source license
├── data/                     # Benchmark split files
│   ├── yeast/
│   ├── multi_species/
│   └── human_virus/
└── README.md
```

## Environment

```bash
conda create -n crossppi python=3.9
conda activate crossppi
pip install -r requirements.txt
```

ProteinMPNN is required for structure feature extraction. Install it separately and pass its code directory to `features.py`.

```text
https://github.com/dauparas/ProteinMPNN
```

ProteinMPNN is not redistributed in this repository.

## Input Format

This release includes benchmark split files under `data/`:

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

Each pair file follows a three-column format:

```text
protein_a_id    protein_b_id    label
P12345          Q99999          1
P11111          Q22222          0
```

Each protein ID should correspond to a PDB file in `--pdb-dir`, for example `P12345.pdb`.

## Preprocessing

The preprocessing script extracts the amino acid sequence and the N, CA, C, and O backbone atom coordinates for each residue. Residues without CA are skipped; missing N, C, or O atoms are filled with the CA coordinate.

```bash
python preprocess_universal.py \
  --pairs data/multi_species/train_any_final.tsv \
  --pdb-dir data/pdbs \
  --output data/processed/train_any.pt
```

The output items contain:

```text
protein_a_id, protein_b_id, a_seq, b_seq, a_coords, b_coords, label
```

## Feature Extraction

```bash
python features.py \
  --input data/processed/train_any.pt \
  --output data/features/train_any_features.pt \
  --proteinmpnn-path /path/to/ProteinMPNN
```

The feature file contains:

```text
a_seq_feat, b_seq_feat, a_str_feat, b_str_feat, a_ca_coords, b_ca_coords, label
```

## Training

Train on one prepared split:

```bash
python model_crossppi.py \
  --train data/features/train_features.pt \
  --valid data/features/valid_features.pt \
  --checkpoint-dir checkpoints
```

Train five folds when files follow the pattern `train1_features.pt`, `valid1_features.pt`, ..., `train5_features.pt`, `valid5_features.pt`:

```bash
python model_crossppi.py \
  --feature-dir data/features \
  --folds 5 \
  --checkpoint-dir checkpoints
```

## Notes

Large generated files should usually not be committed directly to GitHub:

```text
*.pt
*.pth
*.npy
*.npz
__pycache__/
checkpoints/
data/processed/
data/features/
```

For large datasets, extracted features, and trained checkpoints, use an external archive such as Zenodo, Figshare, Google Drive, or institutional storage, then link it from this README.

## Citation

If you use CrossPPI in your work, please cite the corresponding paper.

```bibtex
@article{crossppi,
  title   = {CrossPPI: Sequence-Structure Dual-Modal Cross-Protein Attention for Protein-Protein Interaction Prediction},
  author  = {Your Name and Co-authors},
  journal = {To appear},
  year    = {2026}
}
```

## License

This project is released under the MIT License.
