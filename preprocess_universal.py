import argparse
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

BACKBONE_ATOMS = ("N", "CA", "C", "O")


def parse_pdb_backbone(pdb_path: str) -> Tuple[Optional[str], Optional[List[List[List[float]]]]]:
    """Extract sequence and N/CA/C/O backbone coordinates from a PDB file.

    Residues without CA are skipped. Missing N, C, or O coordinates are filled
    with the residue CA coordinate, matching the preprocessing used by CrossPPI.
    """
    residues: Dict[Tuple[str, str, str], Dict[str, object]] = {}

    try:
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.startswith("ATOM"):
                    continue

                atom_name = line[12:16].strip()
                if atom_name not in BACKBONE_ATOMS:
                    continue

                alt_loc = line[16].strip()
                if alt_loc not in ("", "A"):
                    continue

                res_name = line[17:20].strip()
                chain_id = line[21].strip()
                res_seq = line[22:26].strip()
                insertion_code = line[26].strip()
                res_key = (chain_id, res_seq, insertion_code)

                residues.setdefault(
                    res_key,
                    {"res_name": res_name, **{atom: None for atom in BACKBONE_ATOMS}},
                )
                residues[res_key][atom_name] = [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ]
    except (OSError, ValueError):
        return None, None

    sequence = []
    coords = []
    for residue in residues.values():
        ca_coord = residue["CA"]
        if ca_coord is None:
            continue

        sequence.append(THREE_TO_ONE.get(str(residue["res_name"]), "X"))
        coords.append([residue[atom] if residue[atom] is not None else ca_coord for atom in BACKBONE_ATOMS])

    if not sequence:
        return None, None
    return "".join(sequence), coords


def read_pairs(args: argparse.Namespace) -> pd.DataFrame:
    header = 0 if args.header else None
    return pd.read_csv(args.pairs, sep=args.sep, header=header)


def get_value(row: pd.Series, column: str):
    if column.isdigit():
        return row.iloc[int(column)]
    return row[column]


def preprocess_pairs(args: argparse.Namespace) -> None:
    pairs = read_pairs(args)
    processed = []
    missing_count = 0
    parse_error_count = 0

    for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Preprocessing pairs"):
        protein_a_id = str(get_value(row, args.protein_a_col)).strip()
        protein_b_id = str(get_value(row, args.protein_b_col)).strip()
        label = int(get_value(row, args.label_col))

        protein_a_pdb = os.path.join(args.pdb_dir, f"{protein_a_id}{args.pdb_suffix}")
        protein_b_pdb = os.path.join(args.pdb_dir, f"{protein_b_id}{args.pdb_suffix}")

        if not os.path.exists(protein_a_pdb) or not os.path.exists(protein_b_pdb):
            missing_count += 1
            continue

        a_seq, a_coords = parse_pdb_backbone(protein_a_pdb)
        b_seq, b_coords = parse_pdb_backbone(protein_b_pdb)
        if a_seq is None or b_seq is None:
            parse_error_count += 1
            continue

        processed.append(
            {
                "protein_a_id": protein_a_id,
                "protein_b_id": protein_b_id,
                "a_seq": a_seq,
                "b_seq": b_seq,
                "a_coords": a_coords,
                "b_coords": b_coords,
                "label": label,
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(processed, args.output)

    print(f"Saved {len(processed)} pairs to {args.output}")
    print(f"Skipped pairs: missing_pdb={missing_count}, parse_error={parse_error_count}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess protein-pair data for CrossPPI.")
    parser.add_argument("--pairs", required=True, help="Input pair table, e.g. TSV/CSV.")
    parser.add_argument("--pdb-dir", required=True, help="Directory containing protein PDB files.")
    parser.add_argument("--output", required=True, help="Output .pt file.")
    parser.add_argument("--sep", default="\t", help="Input table delimiter. Default: tab.")
    parser.add_argument("--header", action="store_true", help="Set when the input table has a header row.")
    parser.add_argument("--protein-a-col", default="0", help="Column name or index for protein A ID.")
    parser.add_argument("--protein-b-col", default="1", help="Column name or index for protein B ID.")
    parser.add_argument("--label-col", default="2", help="Column name or index for binary label.")
    parser.add_argument("--pdb-suffix", default=".pdb", help="PDB filename suffix. Default: .pdb")
    return parser


if __name__ == "__main__":
    preprocess_pairs(build_arg_parser().parse_args())
