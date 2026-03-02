#!/usr/bin/env python3
#Copyright 2026 LAVSEN DAHAL
"""
Extract Pillar-0 backbone features for linear probing.

Runs the frozen Pillar backbone over train/val/test splits ONCE and saves
[N, 1152] pooled feature vectors + labels to disk.

Why: backbone is frozen during linear probing → features are identical every
epoch. Pre-extracting them reduces 210 hours of training to ~minutes.

Output per split:
    {output_dir}/{split}_features.pt  →  {
        "features":  FloatTensor [N, 1152],
        "labels":    FloatTensor [N, num_diseases],
        "case_ids":  List[str],
        "disease_names": List[str],
    }

Usage:
    python extract_pillar_features.py \\
        --pillar_pack_root /cachedata/ld258/janus/merlin/pillar_packs_384 \\
        --mask_pack_root   /cachedata/ld258/janus/merlin/packs \\
        --labels_csv       /scratch/railabs/ld258/dataset/merlin/merlinabdominalctdataset/zero_shot_findings_disease_cls.csv \\
        --train_ids        /home/ld258/ipredict/janus/splits/train_ids.txt \\
        --val_ids          /home/ld258/ipredict/janus/splits/val_ids.txt \\
        --test_ids         /home/ld258/ipredict/janus/splits/test_ids.txt \\
        --output_dir       /scratch/railabs/ld258/output/ct_triage/oracle-ct/pillar_features \\
        --model_repo_id    YalaLab/Pillar0-AbdomenCT
"""

import argparse
import sys
import types
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]          # oracle-ct/
_ORACLE_CT  = _REPO_ROOT / "oracle_ct"
_RAVE       = _REPO_ROOT / "rave"
_FINETUNE   = _REPO_ROOT / "pillar-finetune"

for _p in [str(_REPO_ROOT), str(_RAVE), str(_FINETUNE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Stub out pillar.datasets → lifelines → datetime.UTC (Python 3.11+ only)
for _stub in [
    "pillar.datasets", "pillar.datasets.nlst", "pillar.datasets.image_loaders",
    "pillar.datasets.abstract_loader", "pillar.datasets.nlst_utils",
    "pillar.engines", "pillar.losses", "pillar.metrics", "pillar.augmentations",
]:
    sys.modules.setdefault(_stub, types.ModuleType(_stub))

from pillar.models.backbones.mmatlas import MultimodalAtlas          # noqa: E402
from oracle_ct.datamodules.pillar_dataset import PillarDataset       # noqa: E402
from oracle_ct.configs.disease_config import get_all_diseases        # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_ids(path: str):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def build_backbone(model_repo_id: str, model_revision, device: torch.device):
    from easydict import EasyDict
    backbone = MultimodalAtlas(
        args=EasyDict({}),
        device="cpu",
        model_repo_id=model_repo_id,
        model_revision=model_revision,
        pretrained=True,
    )
    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


def collate_fn(batch):
    """Stack images + labels; keep case_ids as list."""
    images    = torch.stack([b["image"] for b in batch])
    labels    = torch.stack([b["labels"] for b in batch])
    case_ids  = [b["case_id"] for b in batch]
    return {"image": images, "labels": labels, "case_ids": case_ids}


@torch.no_grad()
def extract_split(backbone, dataset, device, batch_size, num_workers, modality):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    all_features  = []
    all_labels    = []
    all_case_ids  = []

    for batch in tqdm(loader, desc="Extracting"):
        image = batch["image"].to(device)         # [B, 11, 384, 384, 384]

        # Use bfloat16 — matches pillar-finetune (bf16-mixed).
        # fp16 causes NaN from L2-norm underflow inside the Pillar backbone.
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = backbone(image, batch={"anatomy": [modality] * image.shape[0]})

        pooled = output["pooled"].cpu().float()   # [B, 1152]

        all_features.append(pooled)
        all_labels.append(batch["labels"])
        all_case_ids.extend(batch["case_ids"])

    return {
        "features":     torch.cat(all_features, dim=0),    # [N, 1152]
        "labels":       torch.cat(all_labels,   dim=0),    # [N, num_diseases]
        "case_ids":     all_case_ids,
        "disease_names": dataset.disease_names,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract Pillar-0 features for linear probing")
    parser.add_argument("--pillar_pack_root", required=True)
    parser.add_argument("--mask_pack_root",   required=True)
    parser.add_argument("--labels_csv",       required=True)
    parser.add_argument("--train_ids",        required=True)
    parser.add_argument("--val_ids",          required=True)
    parser.add_argument("--test_ids",         required=True)
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--model_repo_id",    default="YalaLab/Pillar0-AbdomenCT")
    parser.add_argument("--model_revision",   default=None)
    parser.add_argument("--modality",         default="abdomen_ct")
    parser.add_argument("--num_diseases",     type=int, default=30)
    parser.add_argument("--batch_size",       type=int, default=1)
    parser.add_argument("--num_workers",      type=int, default=2)
    parser.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    disease_names = get_all_diseases()[:args.num_diseases]

    print(f"Loading Pillar backbone: {args.model_repo_id}")
    backbone = build_backbone(args.model_repo_id, args.model_revision, device)
    print(f"Backbone loaded. Hidden dim: {backbone.hidden_dim}")

    splits = {
        "train": load_ids(args.train_ids),
        "val":   load_ids(args.val_ids),
        "test":  load_ids(args.test_ids),
    }

    for split_name, case_ids in splits.items():
        out_path = output_dir / f"{split_name}_features.pt"
        if out_path.exists():
            print(f"\n[{split_name}] Already exists at {out_path} — skipping.")
            continue

        print(f"\n[{split_name}] Building dataset ({len(case_ids)} case IDs)...")
        dataset = PillarDataset(
            pillar_pack_root=args.pillar_pack_root,
            mask_pack_root=args.mask_pack_root,
            labels_csv=args.labels_csv,
            case_ids=case_ids,
            disease_names=disease_names,
        )
        print(f"[{split_name}] {len(dataset)} valid cases")

        print(f"[{split_name}] Extracting features...")
        result = extract_split(
            backbone, dataset, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            modality=args.modality,
        )

        torch.save(result, out_path)
        print(f"[{split_name}] Saved → {out_path}")
        print(f"  features: {result['features'].shape}  labels: {result['labels'].shape}")

    print("\nDone. All splits extracted.")


if __name__ == "__main__":
    main()
