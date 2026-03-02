#!/usr/bin/env python3
#Copyright 2026 LAVSEN DAHAL
"""
Train a linear probe on pre-extracted Pillar-0 features.

Loads the saved [N, 1152] feature files from extract_pillar_features.py and
trains a Linear(1152, num_diseases) head with BCE loss.
Runs in minutes on CPU.

Usage:
    python train_linear_probe.py \\
        --features_dir /scratch/railabs/ld258/output/ct_triage/oracle-ct/pillar_features \\
        --output_dir   /scratch/railabs/ld258/output/ct_triage/oracle-ct/pillar_linear_probe
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_aucs(labels_np, logits_np, disease_names):
    """Compute per-disease AUC and macro AUC. Skip diseases with only one class."""
    aucs = {}
    for i, name in enumerate(disease_names):
        y_true = labels_np[:, i]
        y_score = logits_np[:, i]
        if len(np.unique(y_true[~np.isnan(y_true)])) < 2:
            continue
        try:
            # Mask uncertain labels (NaN or -1)
            mask = ~np.isnan(y_true) & (y_true >= 0)
            if mask.sum() > 0 and len(np.unique(y_true[mask])) == 2:
                aucs[name] = roc_auc_score(y_true[mask], y_score[mask])
        except Exception:
            pass
    macro = float(np.mean(list(aucs.values()))) if aucs else 0.0
    return aucs, macro


def train_linear_probe(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # Load pre-extracted features
    # ------------------------------------------------------------------
    print("Loading features...")
    train_data = torch.load(Path(args.features_dir) / "train_features.pt", map_location="cpu")
    val_data   = torch.load(Path(args.features_dir) / "val_features.pt",   map_location="cpu")
    test_data  = torch.load(Path(args.features_dir) / "test_features.pt",  map_location="cpu")

    disease_names = train_data["disease_names"]
    num_diseases  = len(disease_names)
    feat_dim      = train_data["features"].shape[1]

    print(f"Feature dim: {feat_dim}  |  Diseases: {num_diseases}")
    print(f"Train: {train_data['features'].shape[0]}  Val: {val_data['features'].shape[0]}  Test: {test_data['features'].shape[0]}")

    # Sanity check — NaN features mean extraction used fp16 instead of bf16
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        n_nan = torch.isnan(split_data["features"]).sum().item()
        if n_nan > 0:
            raise ValueError(
                f"{split_name} features contain {n_nan} NaN values. "
                "Re-run extract_pillar_features.py (bf16 fix already applied)."
            )

    def make_loader(data, shuffle):
        feat   = data["features"].float()
        labels = data["labels"].float()
        # Replace NaN/-1 (uncertain) with 0 for BCE (BCE loss handles masking separately)
        labels = torch.nan_to_num(labels, nan=0.0)
        labels = labels.clamp(0.0, 1.0)
        ds = TensorDataset(feat, labels)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_data, shuffle=True)
    val_loader   = make_loader(val_data,   shuffle=False)
    test_loader  = make_loader(test_data,  shuffle=False)

    # ------------------------------------------------------------------
    # Model: single shared linear layer [feat_dim → num_diseases]
    # ------------------------------------------------------------------
    model = nn.Linear(feat_dim, num_diseases).to(device)
    nn.init.xavier_uniform_(model.weight)
    nn.init.zeros_(model.bias)

    # Class-weighted BCE
    pos_counts = (train_data["labels"].nan_to_num(0).clamp(0, 1) == 1).sum(0).float()
    neg_counts = (train_data["labels"].nan_to_num(0).clamp(0, 1) == 0).sum(0).float()
    pos_weight = (neg_counts / pos_counts.clamp(min=1)).clamp(max=10.0).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_auc = -1.0
    best_ckpt    = output_dir / "best_linear_probe.pt"
    history      = []

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for feat, labels in train_loader:
            feat, labels = feat.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(feat)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        # Validate
        model.eval()
        val_logits_list, val_labels_list = [], []
        with torch.no_grad():
            for feat, labels in val_loader:
                val_logits_list.append(model(feat.to(device)).cpu())
                val_labels_list.append(labels)
        val_logits = torch.cat(val_logits_list).numpy()
        val_labels = torch.cat(val_labels_list).numpy()
        # Use sigmoid scores for AUC
        val_scores = torch.sigmoid(torch.tensor(val_logits)).numpy()

        _, val_macro = compute_aucs(val_labels, val_scores, disease_names)

        print(f"Epoch {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  val_macro_auc={val_macro:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_auc": val_macro})

        if val_macro > best_val_auc:
            best_val_auc = val_macro
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_macro_auc": val_macro, "disease_names": disease_names}, best_ckpt)
            print(f"  ↑ New best saved ({val_macro:.4f})")

    # ------------------------------------------------------------------
    # Test evaluation with best checkpoint
    # ------------------------------------------------------------------
    print(f"\nBest val macro AUC: {best_val_auc:.4f}")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    test_logits_list, test_labels_list = [], []
    with torch.no_grad():
        for feat, labels in test_loader:
            test_logits_list.append(model(feat.to(device)).cpu())
            test_labels_list.append(labels)
    test_logits = torch.cat(test_logits_list).numpy()
    test_labels = torch.cat(test_labels_list).numpy()
    test_scores = torch.sigmoid(torch.tensor(test_logits)).numpy()

    per_disease_aucs, test_macro = compute_aucs(test_labels, test_scores, disease_names)

    print(f"\nTest macro AUC: {test_macro:.4f}")
    print("\nPer-disease AUCs:")
    for name, auc in sorted(per_disease_aucs.items(), key=lambda x: -x[1]):
        print(f"  {name:<45s}  {auc:.4f}")

    # Save results
    results = {
        "test_macro_auc": test_macro,
        "best_val_macro_auc": best_val_auc,
        "per_disease_aucs": per_disease_aucs,
        "history": history,
        "config": vars(args),
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train linear probe on Pillar features")
    parser.add_argument("--features_dir", required=True,
                        help="Directory with {train,val,test}_features.pt from extract_pillar_features.py")
    parser.add_argument("--output_dir",   required=True,
                        help="Output directory for checkpoint and results")
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=256)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train_linear_probe(args)


if __name__ == "__main__":
    main()
