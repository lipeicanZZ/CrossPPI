import argparse
import os
import sys
from typing import Any, Dict, List

import esm
import numpy as np
import torch
from scipy.spatial.distance import cdist
from tqdm import tqdm


MPNN_ALPHABET = "ARNDCQEGHILKMFPSTWYVX"
MPNN_DICT = {aa: idx for idx, aa in enumerate(MPNN_ALPHABET)}


class FrozenESM2Extractor:
    def __init__(self, device: torch.device):
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model.eval().to(device)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.device = device

    @torch.no_grad()
    def extract(self, sequence: str) -> torch.Tensor:
        _, _, tokens = self.batch_converter([("protein", sequence)])
        tokens = tokens.to(self.device)
        output = self.model(tokens, repr_layers=[33], return_contacts=False)
        return output["representations"][33][0, 1 : len(sequence) + 1].cpu()


class FrozenProteinMPNNExtractor:
    def __init__(self, proteinmpnn_path: str, device: torch.device):
        if proteinmpnn_path not in sys.path:
            sys.path.append(proteinmpnn_path)

        from protein_mpnn_utils import ProteinMPNN, gather_nodes

        self.gather_nodes = gather_nodes
        self.model = ProteinMPNN(
            num_letters=21,
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=48,
            augment_eps=0.0,
        ).eval().to(device)
        self.device = device

    @torch.no_grad()
    def extract(self, sequence: str, coords: List[List[List[float]]]) -> torch.Tensor:
        length = len(sequence)
        seq_tensor = torch.tensor(
            [MPNN_DICT.get(aa, 20) for aa in sequence],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        coord_tensor = torch.tensor(coords, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.ones((1, length), dtype=torch.float32, device=self.device)
        residue_idx = torch.arange(length, device=self.device).unsqueeze(0)
        chain_encoding = torch.ones((1, length), dtype=torch.long, device=self.device)

        edge_feat, edge_idx = self.model.features(coord_tensor, mask, residue_idx, chain_encoding)
        node_feat = torch.zeros((edge_feat.shape[0], edge_feat.shape[1], edge_feat.shape[-1]), device=self.device)
        for layer in self.model.encoder_layers:
            node_feat, edge_feat = layer(node_feat, edge_feat, edge_idx, mask)

        seq_embed = self.model.W_s(seq_tensor)
        seq_neighbors = self.gather_nodes(seq_embed, edge_idx)
        seq_self = seq_embed.unsqueeze(2).expand(-1, -1, edge_feat.shape[2], -1)
        decoder_edge_feat = torch.cat([edge_feat, seq_neighbors, seq_self], dim=-1)
        node_feat = node_feat + seq_embed

        mask_attend = self.gather_nodes(mask.unsqueeze(-1), edge_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.model.decoder_layers:
            node_feat = layer(node_feat, decoder_edge_feat, mask, mask_attend)

        return node_feat.squeeze(0).cpu()


def get_surface_mask(coords: List[List[List[float]]], radius: float = 10.0, max_neighbors: int = 20) -> torch.Tensor:
    ca_coords = np.asarray(coords, dtype=np.float32)[:, 1, :]
    distance_matrix = cdist(ca_coords, ca_coords)
    neighbor_counts = (distance_matrix <= radius).sum(axis=1)
    return torch.tensor(neighbor_counts <= max_neighbors, dtype=torch.bool)


def extract_features(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data: List[Dict[str, Any]] = torch.load(args.input, weights_only=False)

    esm_extractor = FrozenESM2Extractor(device)
    mpnn_extractor = FrozenProteinMPNNExtractor(args.proteinmpnn_path, device)

    features = []
    for item in tqdm(data, desc="Extracting features"):
        a_seq_feat = esm_extractor.extract(item["a_seq"]).half()
        b_seq_feat = esm_extractor.extract(item["b_seq"]).half()
        a_str_feat = mpnn_extractor.extract(item["a_seq"], item["a_coords"]).half()
        b_str_feat = mpnn_extractor.extract(item["b_seq"], item["b_coords"]).half()

        features.append(
            {
                "protein_a_id": item.get("protein_a_id", ""),
                "protein_b_id": item.get("protein_b_id", ""),
                "label": item["label"],
                "a_seq_feat": a_seq_feat,
                "b_seq_feat": b_seq_feat,
                "a_str_feat": a_str_feat,
                "b_str_feat": b_str_feat,
                "a_surface_mask": get_surface_mask(item["a_coords"]),
                "b_surface_mask": get_surface_mask(item["b_coords"]),
                "a_ca_coords": np.asarray(item["a_coords"], dtype=np.float32)[:, 1, :],
                "b_ca_coords": np.asarray(item["b_coords"], dtype=np.float32)[:, 1, :],
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(features, args.output)
    print(f"Saved {len(features)} feature items to {args.output}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract ESM-2 and ProteinMPNN features for CrossPPI.")
    parser.add_argument("--input", required=True, help="Preprocessed .pt file from preprocess_universal.py.")
    parser.add_argument("--output", required=True, help="Output feature .pt file.")
    parser.add_argument("--proteinmpnn-path", required=True, help="Directory containing protein_mpnn_utils.py.")
    parser.add_argument("--device", default="cuda:0", help="Preferred CUDA device. Default: cuda:0")
    return parser


if __name__ == "__main__":
    extract_features(build_arg_parser().parse_args())
