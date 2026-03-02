#!/usr/bin/env python3
#Copyright 2026 LAVSEN DAHAL
"""
Extract Pillar-0 backbone features for linear probing — multi-GPU version.

Each GPU processes its own slice of case IDs (rank::world_size), saves a shard,
then rank 0 merges all shards into the final {split}_features.pt file.
No NCCL communication during forward pass — pure data parallelism.

Single-GPU usage:
    python extract_pillar_features.py --pillar_pack_root ... --output_dir ...

Multi-GPU usage (via torchrun):
    torchrun --nproc_per_node=8 extract_pillar_features.py --pillar_pack_root ... --output_dir ...

Output per split:
    {output_dir}/{split}_features.pt  →  {
        "features":      FloatTensor [N, 1152],
        "labels":        FloatTensor [N, num_diseases],
        "case_ids":      List[str],
        "disease_names": List[str],
    }
"""

import argparse
import os
import sys
import types
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]          # oracle-ct/
_RAVE      = _REPO_ROOT / "rave"
_FINETUNE  = _REPO_ROOT / "pillar-finetune"

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


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def init_dist():
    """Initialize process group if launched with torchrun, else single-GPU."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def is_main(rank):
    return rank == 0


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
    images   = torch.stack([b["image"]  for b in batch])
    labels   = torch.stack([b["labels"] for b in batch])
    case_ids = [b["case_id"] for b in batch]
    return {"image": images, "labels": labels, "case_ids": case_ids}


@torch.no_grad()
def extract_split(backbone, dataset, device, batch_size, num_workers, modality, rank):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    all_features = []
    all_labels   = []
    all_case_ids = []

    pbar = tqdm(loader, desc="Extracting", disable=(rank != 0))
    for batch in pbar:
        image = batch["image"].to(device)

        # bfloat16 — matches pillar-finetune (bf16-mixed).
        # fp16 causes NaN from L2-norm underflow inside the Pillar backbone.
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = backbone(image, batch={"anatomy": [modality] * image.shape[0]})

        pooled = output["pooled"].cpu().float()   # [B, 1152]
        all_features.append(pooled)
        all_labels.append(batch["labels"])
        all_case_ids.extend(batch["case_ids"])

    return {
        "features":      torch.cat(all_features, dim=0),
        "labels":        torch.cat(all_labels,   dim=0),
        "case_ids":      all_case_ids,
        "disease_names": dataset.disease_names,
    }


def merge_shards(output_dir: Path, split_name: str, world_size: int):
    """Concatenate per-rank shard files into one final file, then delete shards."""
    all_features, all_labels, all_case_ids = [], [], []
    disease_names = None

    for rank in range(world_size):
        shard_path = output_dir / f"{split_name}_features_rank{rank}.pt"
        shard = torch.load(shard_path, map_location="cpu")
        all_features.append(shard["features"])
        all_labels.append(shard["labels"])
        all_case_ids.extend(shard["case_ids"])
        disease_names = shard["disease_names"]
        shard_path.unlink()

    result = {
        "features":      torch.cat(all_features, dim=0),
        "labels":        torch.cat(all_labels,   dim=0),
        "case_ids":      all_case_ids,
        "disease_names": disease_names,
    }
    out_path = output_dir / f"{split_name}_features.pt"
    torch.save(result, out_path)
    return result, out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract Pillar-0 features (multi-GPU)")
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
    parser.add_argument("--num_diseases",     type=int, default=0,
                        help="Limit to first N diseases from CSV (0 = all)")
    parser.add_argument("--batch_size",       type=int, default=1)
    parser.add_argument("--num_workers",      type=int, default=2)
    parser.add_argument("--max_cases",        type=int, default=None,
                        help="Limit each split to first N cases (sanity check only)")
    args = parser.parse_args()

    rank, world_size, local_rank = init_dist()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read disease names directly from the labels CSV.
    # get_all_diseases() requires load_config_globally() to be called first
    # and would return [] without it — so read from the CSV instead.
    _labels_df = pd.read_csv(args.labels_csv).set_index("study id")
    disease_names = list(_labels_df.columns)
    if args.num_diseases > 0:
        disease_names = disease_names[:args.num_diseases]

    if is_main(rank):
        print(f"Pillar feature extraction: {world_size} GPU(s)")
        print(f"Diseases: {len(disease_names)}  →  {disease_names[:5]}{'...' if len(disease_names) > 5 else ''}")
        print(f"Loading Pillar backbone: {args.model_repo_id}")

    backbone = build_backbone(args.model_repo_id, args.model_revision, device)

    if is_main(rank):
        print(f"Backbone loaded on {world_size} GPU(s). Hidden dim: {backbone.hidden_dim}")

    splits = {
        "train": load_ids(args.train_ids),
        "val":   load_ids(args.val_ids),
        "test":  load_ids(args.test_ids),
    }

    for split_name, case_ids in splits.items():
        if args.max_cases is not None:
            case_ids = case_ids[:args.max_cases]

        final_path = output_dir / f"{split_name}_features.pt"
        if final_path.exists():
            if is_main(rank):
                print(f"\n[{split_name}] Already exists — skipping.")
            continue

        # Each rank processes its own slice: rank, rank+world_size, rank+2*world_size ...
        my_case_ids = case_ids[rank::world_size]

        if is_main(rank):
            print(f"\n[{split_name}] {len(case_ids)} total cases → "
                  f"{len(my_case_ids)} per GPU across {world_size} GPU(s)")

        dataset = PillarDataset(
            pillar_pack_root=args.pillar_pack_root,
            mask_pack_root=args.mask_pack_root,
            labels_csv=args.labels_csv,
            case_ids=my_case_ids,
            disease_names=disease_names,
        )

        print(f"  [rank {rank}] {len(dataset)} valid cases")

        shard_path = output_dir / f"{split_name}_features_rank{rank}.pt"
        result = extract_split(
            backbone, dataset, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            modality=args.modality,
            rank=rank,
        )
        torch.save(result, shard_path)
        print(f"  [rank {rank}] Shard saved → {shard_path}  "
              f"features: {result['features'].shape}")

        # Wait for all ranks to finish this split before merging
        if dist.is_initialized():
            dist.barrier()

        # Rank 0 merges all shards into the final file
        if is_main(rank):
            merged, out_path = merge_shards(output_dir, split_name, world_size)
            n_nan = torch.isnan(merged["features"]).sum().item()
            print(f"\n[{split_name}] Merged → {out_path}")
            print(f"  Total cases: {merged['features'].shape[0]}  "
                  f"Feature dim: {merged['features'].shape[1]}  NaN: {n_nan}")
            if n_nan > 0:
                raise ValueError(
                    f"{split_name} features contain {n_nan} NaN values — "
                    "check bfloat16 AMP is being used."
                )

        if dist.is_initialized():
            dist.barrier()

    if dist.is_initialized():
        dist.destroy_process_group()

    if is_main(rank):
        print("\nDone. All splits extracted.")


if __name__ == "__main__":
    main()
