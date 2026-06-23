import argparse
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class Config:
    seq_dim = 1280
    str_dim = 128
    model_dim = 256
    num_heads = 4
    pooling_temperature = 0.5


class PPIDataset(Dataset):
    def __init__(self, feature_path: str):
        self.data = torch.load(feature_path, weights_only=False)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        return self.data[index]


def pad_features(features: Iterable[torch.Tensor], dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    feature_list = [torch.as_tensor(feature) for feature in features]
    batch_size = len(feature_list)
    max_len = max(feature.shape[0] for feature in feature_list)
    padded = torch.zeros(batch_size, max_len, dim, dtype=torch.float32)
    pad_mask = torch.ones(batch_size, max_len, dtype=torch.bool)

    for index, feature in enumerate(feature_list):
        length = feature.shape[0]
        padded[index, :length] = feature.float()
        pad_mask[index, :length] = False
    return padded, pad_mask


def pad_bool_masks(masks: Iterable[torch.Tensor]) -> torch.Tensor:
    mask_list = [torch.as_tensor(mask, dtype=torch.bool) for mask in masks]
    batch_size = len(mask_list)
    max_len = max(mask.shape[0] for mask in mask_list)
    padded = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for index, mask in enumerate(mask_list):
        padded[index, : mask.shape[0]] = mask
    return padded


def crossppi_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)

    a_seq, a_seq_mask = pad_features([item["a_seq_feat"] for item in batch], Config.seq_dim)
    b_seq, b_seq_mask = pad_features([item["b_seq_feat"] for item in batch], Config.seq_dim)
    a_str, a_str_mask = pad_features([item["a_str_feat"] for item in batch], Config.str_dim)
    b_str, b_str_mask = pad_features([item["b_str_feat"] for item in batch], Config.str_dim)
    a_coords, _ = pad_features([item["a_ca_coords"] for item in batch], 3)
    b_coords, _ = pad_features([item["b_ca_coords"] for item in batch], 3)

    default_a_surface = [torch.ones(item["a_seq_feat"].shape[0], dtype=torch.bool) for item in batch]
    default_b_surface = [torch.ones(item["b_seq_feat"].shape[0], dtype=torch.bool) for item in batch]

    return {
        "a_seq": a_seq,
        "b_seq": b_seq,
        "a_str": a_str,
        "b_str": b_str,
        "a_coords": a_coords,
        "b_coords": b_coords,
        "a_seq_mask": a_seq_mask,
        "b_seq_mask": b_seq_mask,
        "a_str_mask": a_str_mask,
        "b_str_mask": b_str_mask,
        "a_surface_mask": pad_bool_masks([item.get("a_surface_mask", default) for item, default in zip(batch, default_a_surface)]),
        "b_surface_mask": pad_bool_masks([item.get("b_surface_mask", default) for item, default in zip(batch, default_b_surface)]),
        "label": labels,
    }


class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - pt) ** self.gamma * bce_loss).mean()


def symmetric_contrastive_loss(feat_a: torch.Tensor, feat_b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = torch.matmul(feat_a, feat_b.t()) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def internal_attention_penalty(attn_weights: torch.Tensor, target_surface_mask: torch.Tensor, target_pad_mask: torch.Tensor) -> torch.Tensor:
    internal_mask = (~target_surface_mask) & (~target_pad_mask)
    return (attn_weights * internal_mask.unsqueeze(1).float()).sum(dim=-1).mean(dim=-1)


def spatial_smoothness_loss(attn_weights: torch.Tensor, coords: torch.Tensor, pad_mask: torch.Tensor, radius: float = 10.0) -> torch.Tensor:
    distance_matrix = torch.cdist(coords, coords)
    neighbor_mask = (distance_matrix <= radius).float()
    valid_mask = (~pad_mask).float()
    neighbor_mask = neighbor_mask * valid_mask.unsqueeze(1) * valid_mask.unsqueeze(2)
    neighbor_mask = neighbor_mask * (1.0 - torch.eye(neighbor_mask.size(1), device=coords.device).unsqueeze(0))

    importance = attn_weights.sum(dim=1)
    diff = (importance.unsqueeze(2) - importance.unsqueeze(1)) ** 2
    return (diff * neighbor_mask).sum(dim=(1, 2)) / (neighbor_mask.sum(dim=(1, 2)) + 1e-9)


class IntraProteinFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, seq_feat: torch.Tensor, str_feat: torch.Tensor) -> torch.Tensor:
        seq_len, str_len = seq_feat.size(1), str_feat.size(1)
        if seq_len > str_len:
            str_feat = F.pad(str_feat, (0, 0, 0, seq_len - str_len))
        elif str_len > seq_len:
            str_feat = str_feat[:, :seq_len, :]
        return self.norm(seq_feat + str_feat)


class CrossPPIModel(nn.Module):
    def __init__(self, config: Config = Config()):
        super().__init__()
        self.config = config
        dim = config.model_dim
        self.seq_proj = nn.Sequential(nn.Linear(config.seq_dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.str_proj = nn.Sequential(nn.Linear(config.str_dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.fusion = IntraProteinFusion(dim)
        self.cross_attention = nn.MultiheadAttention(dim, config.num_heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(dim * 4 + 1, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    @staticmethod
    def masked_avg_pool(features: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        valid = (~pad_mask).float().unsqueeze(-1)
        return (features * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-9)

    def _pool_with_attention(self, context: torch.Tensor, importance: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(importance.masked_fill(pad_mask, -1e9) / self.config.pooling_temperature, dim=1)
        return torch.sum(context * weights.unsqueeze(-1), dim=1)

    def forward(self, batch: Dict[str, torch.Tensor], use_seq: bool = True, use_str: bool = True) -> Dict[str, torch.Tensor]:
        a_seq = self.seq_proj(batch["a_seq"])
        b_seq = self.seq_proj(batch["b_seq"])
        a_str = self.str_proj(batch["a_str"])
        b_str = self.str_proj(batch["b_str"])

        if not use_seq:
            a_seq, b_seq = torch.zeros_like(a_seq), torch.zeros_like(b_seq)
        if not use_str:
            a_str, b_str = torch.zeros_like(a_str), torch.zeros_like(b_str)

        a_seq_global = F.normalize(self.masked_avg_pool(a_seq, batch["a_seq_mask"]), p=2, dim=-1)
        a_str_global = F.normalize(self.masked_avg_pool(a_str, batch["a_str_mask"]), p=2, dim=-1)
        b_seq_global = F.normalize(self.masked_avg_pool(b_seq, batch["b_seq_mask"]), p=2, dim=-1)
        b_str_global = F.normalize(self.masked_avg_pool(b_str, batch["b_str_mask"]), p=2, dim=-1)

        a_feat = self.fusion(a_seq, a_str)
        b_feat = self.fusion(b_seq, b_str)

        a_context, a_to_b_attn = self.cross_attention(
            query=a_feat,
            key=b_feat,
            value=b_feat,
            key_padding_mask=batch["b_seq_mask"],
            need_weights=True,
            average_attn_weights=False,
        )
        b_context, b_to_a_attn = self.cross_attention(
            query=b_feat,
            key=a_feat,
            value=a_feat,
            key_padding_mask=batch["a_seq_mask"],
            need_weights=True,
            average_attn_weights=False,
        )

        a_to_b_attn = a_to_b_attn.mean(dim=1)
        b_to_a_attn = b_to_a_attn.mean(dim=1)
        a_importance = b_to_a_attn.sum(dim=1)
        b_importance = a_to_b_attn.sum(dim=1)

        a_pool = self._pool_with_attention(a_context, a_importance, batch["a_seq_mask"])
        b_pool = self._pool_with_attention(b_context, b_importance, batch["b_seq_mask"])
        pair_repr = torch.cat(
            [
                a_pool,
                b_pool,
                a_pool * b_pool,
                torch.abs(a_pool - b_pool),
                F.cosine_similarity(a_pool, b_pool, dim=-1).unsqueeze(-1),
            ],
            dim=-1,
        )

        return {
            "logits": self.classifier(pair_repr).squeeze(-1),
            "a_seq_norm": a_seq_global,
            "a_str_norm": a_str_global,
            "b_seq_norm": b_seq_global,
            "b_str_norm": b_str_global,
            "a_to_b_attn": a_to_b_attn,
            "b_to_a_attn": b_to_a_attn,
        }


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def compute_metrics(labels: List[float], probs: List[float], threshold: float = 0.5) -> Dict[str, float]:
    labels_np = np.asarray(labels)
    probs_np = np.asarray(probs)
    preds = (probs_np >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels_np, preds, labels=[0, 1]).ravel()
    return {
        "ACC": accuracy_score(labels_np, preds),
        "Precision": precision_score(labels_np, preds, zero_division=0),
        "Recall": recall_score(labels_np, preds, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "F1": f1_score(labels_np, preds, zero_division=0),
        "AUC": roc_auc_score(labels_np, probs_np),
        "MCC": matthews_corrcoef(labels_np, preds),
    }


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    all_labels: List[float] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            output = model(batch)
            loss = criterion(output["logits"], batch["label"])
            total_loss += loss.item()
            all_probs.extend(torch.sigmoid(output["logits"]).cpu().numpy().tolist())
            all_labels.extend(batch["label"].cpu().numpy().tolist())

    return total_loss / max(len(loader), 1), compute_metrics(all_labels, all_probs)


def train_one_split(args: argparse.Namespace, train_path: str, valid_path: str, fold_name: str = "split") -> Dict[str, float]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(PPIDataset(train_path), batch_size=args.batch_size, shuffle=True, collate_fn=crossppi_collate_fn)
    valid_loader = DataLoader(PPIDataset(valid_path), batch_size=args.batch_size, shuffle=False, collate_fn=crossppi_collate_fn)

    model = CrossPPIModel().to(device)
    criterion = BinaryFocalLossWithLogits(alpha=args.focal_alpha, gamma=args.focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_auc = -1.0
    best_metrics: Dict[str, float] = {}
    patience = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"{fold_name} epoch {epoch}/{args.epochs}"):
            batch = move_to_device(batch, device)
            optimizer.zero_grad()
            output = model(batch, use_seq=not args.no_sequence, use_str=not args.no_structure)
            cls_loss = criterion(output["logits"], batch["label"])
            align_loss = 0.5 * (
                symmetric_contrastive_loss(output["a_seq_norm"], output["a_str_norm"], args.temperature)
                + symmetric_contrastive_loss(output["b_seq_norm"], output["b_str_norm"], args.temperature)
            )
            loss = cls_loss + args.align_weight * align_loss

            if args.surface_weight > 0:
                pos_mask = (batch["label"] == 1.0).float()
                valid_pos = pos_mask.sum() + 1e-9
                surface_loss = (
                    internal_attention_penalty(output["a_to_b_attn"], batch["b_surface_mask"], batch["b_seq_mask"])
                    + internal_attention_penalty(output["b_to_a_attn"], batch["a_surface_mask"], batch["a_seq_mask"])
                )
                loss = loss + args.surface_weight * (surface_loss * pos_mask).sum() / valid_pos

            if args.smooth_weight > 0:
                pos_mask = (batch["label"] == 1.0).float()
                valid_pos = pos_mask.sum() + 1e-9
                smooth_loss = (
                    spatial_smoothness_loss(output["a_to_b_attn"], batch["b_coords"], batch["b_seq_mask"])
                    + spatial_smoothness_loss(output["b_to_a_attn"], batch["a_coords"], batch["a_seq_mask"])
                )
                loss = loss + args.smooth_weight * (smooth_loss * pos_mask).sum() / valid_pos

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        valid_loss, metrics = evaluate(model, valid_loader, criterion, device)
        scheduler.step()
        print(
            f"{fold_name} epoch {epoch}: train_loss={train_loss / max(len(train_loader), 1):.4f}, "
            f"valid_loss={valid_loss:.4f}, AUC={metrics['AUC']:.4f}, ACC={metrics['ACC']:.4f}, F1={metrics['F1']:.4f}"
        )

        if metrics["AUC"] > best_auc:
            best_auc = metrics["AUC"]
            best_metrics = metrics
            patience = 0
            checkpoint_path = os.path.join(args.checkpoint_dir, f"crossppi_{fold_name}.pth")
            torch.save({"model_state_dict": model.state_dict(), "metrics": metrics, "args": vars(args)}, checkpoint_path)
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping on {fold_name}.")
                break

    return best_metrics


def run_training(args: argparse.Namespace) -> None:
    if args.folds > 1:
        fold_metrics = []
        for fold in range(1, args.folds + 1):
            train_path = os.path.join(args.feature_dir, f"train{fold}_features.pt")
            valid_path = os.path.join(args.feature_dir, f"valid{fold}_features.pt")
            fold_metrics.append(train_one_split(args, train_path, valid_path, fold_name=f"fold{fold}"))

        print("\nCross-validation summary")
        for metric in fold_metrics[0].keys():
            values = np.asarray([item[metric] for item in fold_metrics], dtype=np.float32)
            print(f"{metric}: {values.mean() * 100:.2f} ± {values.std() * 100:.2f}")
    else:
        metrics = train_one_split(args, args.train, args.valid)
        print("\nBest validation metrics")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CrossPPI on extracted feature files.")
    parser.add_argument("--train", help="Training feature .pt file.")
    parser.add_argument("--valid", help="Validation feature .pt file.")
    parser.add_argument("--feature-dir", default="data/features", help="Directory for fold-based feature files.")
    parser.add_argument("--folds", type=int, default=1, help="Use N-fold files train{i}_features.pt and valid{i}_features.pt.")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--focal-alpha", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--align-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--surface-weight", type=float, default=0.0)
    parser.add_argument("--smooth-weight", type=float, default=0.0)
    parser.add_argument("--no-sequence", action="store_true", help="Disable sequence features for ablation.")
    parser.add_argument("--no-structure", action="store_true", help="Disable structure features for ablation.")
    return parser


if __name__ == "__main__":
    parsed_args = build_arg_parser().parse_args()
    if parsed_args.folds <= 1 and (not parsed_args.train or not parsed_args.valid):
        raise ValueError("Provide --train and --valid for single-split training, or set --folds > 1.")
    run_training(parsed_args)
