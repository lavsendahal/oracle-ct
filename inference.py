#Copyright 2026 LAVSEN DAHAL

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#!/usr/bin/env python3
"""
Janus Inference Script (Hydra)

Runs inference on a split (default: test) and saves per-case probabilities to CSV.

Example:
  python janus/inference.py \
    experiment=dinov3_scalar_fusion \
    paths.checkpoint=outputs/OracleCT_DINOv3_MaskedUnaryAttnScalar/.../checkpoints/best.pt \
    logging.use_wandb=false \
    training.use_ddp=true
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Add parent to path (repo root)
sys.path.insert(0, str(Path(__file__).parent.parent))

from janus.train import build_model, ddp_setup, init_distributed, ddp_cleanup, is_main_process
from janus.datamodules.dataset import JanusDataset, janus_collate_fn
from janus.configs.disease_config import load_config_globally, get_all_diseases
from janus.train import compute_metrics


def load_ids_from_file(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    if not has_module_prefix:
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def load_checkpoint_state_dict(ckpt_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    elif isinstance(ckpt, dict):
        state_dict = ckpt
        meta = {}
    else:
        raise ValueError(f"Unsupported checkpoint format type={type(ckpt)} at {ckpt_path}")

    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint model_state_dict is not a dict at {ckpt_path}")

    return _strip_module_prefix(state_dict), meta


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Load disease config from LR pipeline output
    disease_config_path = cfg.paths.get("disease_config")
    if disease_config_path:
        load_config_globally(disease_config_path)
        print(f"\n✓ Loaded disease config from: {disease_config_path}")
    else:
        raise ValueError("paths.disease_config is required for inference")

    use_ddp = cfg.training.get("use_ddp", False)

    # Initialize DDP if requested
    if use_ddp:
        is_distributed = init_distributed(backend=cfg.training.get("backend", "nccl"))
        if not is_distributed:
            use_ddp = False
            if is_main_process():
                print("Warning: DDP requested but not available, using single GPU")

    rank, world_size, local_rank = ddp_setup()

    if use_ddp:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        print("=" * 80)
        print("Inference configuration:")
        print(OmegaConf.to_yaml(cfg))
        print("=" * 80)
        print(f"Device: {device}")
        if use_ddp:
            print(f"Using DDP with {world_size} GPUs")

    ckpt_path = cfg.paths.get("checkpoint")
    if not ckpt_path:
        raise ValueError("Missing required config: paths.checkpoint=/path/to/checkpoint.pt")
    ckpt_path = str(ckpt_path)

    # Split IDs (default: test)
    split_name = cfg.get("inference", {}).get("split", "test") if isinstance(cfg.get("inference", None), DictConfig) else "test"

    if split_name == "test":
        ids_path = cfg.paths.test_ids
    elif split_name == "val":
        ids_path = cfg.paths.val_ids
    elif split_name == "train":
        ids_path = cfg.paths.train_ids
    else:
        raise ValueError(f"Unknown inference.split='{split_name}' (expected train/val/test)")

    case_ids = load_ids_from_file(ids_path)
    if is_main_process():
        print(f"\nSplit: {split_name} ({len(case_ids)} cases from {ids_path})")

    # Ensure disease ordering matches model outputs
    all_diseases = get_all_diseases()
    disease_names_cfg = cfg.model.get("disease_names", None)
    if disease_names_cfg is None:
        disease_names_cfg = all_diseases[: cfg.model.num_diseases]

    # Only load features for scalar fusion models
    features_parquet = None
    feature_columns = None
    if cfg.model.name in [
        "OracleCT_DINOv3_MaskedUnaryAttnScalar",
        "OracleCT_ResNet3D_MaskedUnaryAttnScalar",
    ]:
        features_parquet = cfg.paths.get("features_parquet")
        feature_columns = cfg.model.get("feature_columns")

    dataset = JanusDataset(
        pack_root=cfg.paths.pack_root,
        labels_csv=cfg.paths.labels_csv,
        case_ids=case_ids,
        features_parquet=features_parquet,
        feature_columns=feature_columns,
        disease_names=disease_names_cfg,
        cache_packs=False,
        use_augmentation=False,
    )

    sampler = None
    if use_ddp and dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    loader = DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        persistent_workers=cfg.dataset.persistent_workers,
        prefetch_factor=cfg.dataset.prefetch_factor,
        collate_fn=janus_collate_fn,
    )

    # Build + load model
    model = build_model(cfg).to(device)
    state_dict, ckpt_meta = load_checkpoint_state_dict(ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if is_main_process():
        print(f"\nLoaded checkpoint: {ckpt_path}")
        if ckpt_meta.get("epoch") is not None:
            print(f"  Checkpoint epoch: {ckpt_meta.get('epoch')}")
        if missing:
            print(f"  Missing keys: {len(missing)} (ok if you changed model options like visual_pooling)")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)} (ok if you changed model options like visual_pooling)")

    model.eval()

    all_case_ids: List[str] = []
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    use_amp = cfg.training.get("use_amp", True)
    pbar = tqdm(loader, desc=f"Inference [{split_name}]", disable=not is_main_process())
    with torch.no_grad():
        for batch in pbar:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(batch)

            logits = output["logits"] if isinstance(output, dict) else output
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            batch_case_ids = batch["case_id"]
            labels = batch.get("labels")

            all_case_ids.extend(batch_case_ids)
            all_probs.append(probs)
            if labels is not None:
                all_labels.append(labels.detach().cpu().numpy())

    probs_np = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, len(disease_names_cfg)), dtype=np.float32)
    labels_np = np.concatenate(all_labels, axis=0) if all_labels else None

    # Gather across ranks
    if use_ddp and dist.is_initialized():
        gathered_case_ids: List[List[str]] = [None for _ in range(world_size)]  # type: ignore[list-item]
        gathered_probs: List[np.ndarray] = [None for _ in range(world_size)]  # type: ignore[list-item]
        gathered_labels: List[np.ndarray] = [None for _ in range(world_size)]  # type: ignore[list-item]

        dist.all_gather_object(gathered_case_ids, all_case_ids)
        dist.all_gather_object(gathered_probs, probs_np)
        if labels_np is not None:
            dist.all_gather_object(gathered_labels, labels_np)

        if is_main_process():
            all_case_ids = [cid for part in gathered_case_ids for cid in part]
            probs_np = np.concatenate(gathered_probs, axis=0)
            labels_np = np.concatenate([x for x in gathered_labels if x is not None], axis=0) if labels_np is not None else None

            # BUGFIX: DistributedSampler with drop_last=False pads the dataset,
            # causing some samples to be duplicated. Deduplicate here.
            seen = set()
            unique_indices = []
            for i, cid in enumerate(all_case_ids):
                if cid not in seen:
                    seen.add(cid)
                    unique_indices.append(i)

            if len(unique_indices) < len(all_case_ids):
                n_dups = len(all_case_ids) - len(unique_indices)
                print(f"Note: Removed {n_dups} duplicate samples from DDP padding")
                all_case_ids = [all_case_ids[i] for i in unique_indices]
                probs_np = probs_np[unique_indices]
                if labels_np is not None:
                    labels_np = labels_np[unique_indices]

    # Save outputs (main process only)
    if is_main_process():
        # By default, write outputs next to the training run that produced the checkpoint:
        #   .../<run_dir>/checkpoints/<name>.pt  ->  .../<run_dir>/<dataset>/<split>_predictions.csv
        ckpt_path_p = Path(ckpt_path).expanduser().resolve()
        if ckpt_path_p.parent.name == "checkpoints":
            run_dir = ckpt_path_p.parent.parent
        else:
            # Fallback to Hydra's run dir if checkpoint path doesn't follow the usual structure.
            from hydra.core.hydra_config import HydraConfig
            hydra_cfg = HydraConfig.get()
            run_dir = Path(hydra_cfg.runtime.output_dir)

        # Get dataset name from Hydra config (e.g., "merlin" or "duke")
        # This comes from the dataset config file that was loaded
        from hydra.core.hydra_config import HydraConfig
        hydra_cfg = HydraConfig.get()
        # Extract dataset name from the overrides or defaults
        dataset_name = "unknown"
        for override in hydra_cfg.overrides.task:
            if override.startswith("dataset="):
                dataset_name = override.split("=")[1]
                break
        else:
            # Check the defaults if not in overrides
            for default in hydra_cfg.runtime.choices.values():
                if default in ["merlin", "duke"]:
                    dataset_name = default
                    break

        # Create dataset-specific output directory
        output_dir = run_dir / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(probs_np, columns=list(disease_names_cfg))
        df.insert(0, "case_id", all_case_ids)

        preds_path = output_dir / f"{split_name}_predictions.csv"
        df.to_csv(preds_path, index=False)
        print(f"\nSaved predictions: {preds_path}")

        # Optional metrics (requires labels)
        if labels_np is not None:
            metrics = compute_metrics(probs_np, labels_np, disease_names=list(disease_names_cfg))
            metrics_path = output_dir / f"{split_name}_metrics.json"
            import json
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved metrics: {metrics_path}")

    if use_ddp and dist.is_initialized():
        ddp_cleanup()


if __name__ == "__main__":
    main()
