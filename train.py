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
Janus Training Script

Usage:
    python train.py experiment=baseline_gap paths.labels_csv=/path/to/labels.csv
    python train.py experiment=masked_attn
    python train.py experiment=scalar_fusion
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add oracle-ct directory to path for self-contained model imports
sys.path.insert(0, str(Path(__file__).parent))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

from janus.losses import build_loss_from_config, BCEUncertainLoss
from models.dinov3_oracle_ct import (
    JanusGAP, JanusMaskedAttn, JanusScalarFusion)
from models.resnet3d_oracle_ct import (
    JanusResNet3DGAP, JanusResNet3DMaskedAttn,
    JanusResNet3DScalarFusion)

from janus.datamodules.dataset import JanusDataset, janus_collate_fn
from janus.configs.disease_config import load_config_globally, get_all_diseases


# =============================================================================
# Distributed Training Utilities
# =============================================================================

def ddp_setup():
    """Initialize DDP from environment variables set by torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    return rank, world_size, local_rank


def init_distributed(backend: str = "nccl"):
    """Initialize distributed process group."""
    if not dist.is_available():
        return False

    if "RANK" not in os.environ:
        # Not running with torchrun
        return False

    dist.init_process_group(backend=backend)
    return True


def ddp_cleanup():
    """Clean up distributed process group."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank():
    """Get current process rank."""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """Get world size (number of processes)."""
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def concat_all_gather(tensor):
    """
    Gather tensors from all processes and concatenate them.

    Args:
        tensor: Tensor to gather [batch_size, ...]

    Returns:
        Concatenated tensor from all ranks [world_size * batch_size, ...]
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    world_size = get_world_size()
    if world_size == 1:
        return tensor

    # Gather tensors from all GPUs
    tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_ids_from_file(path: str):
    """Load case IDs from text file (one ID per line)."""
    with open(path, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids


def build_model(cfg: DictConfig) -> nn.Module:
    """Build model from config."""
    model_name = cfg.model.name

    # Get feature_stats_path: prefer model.feature_stats_path, fallback to paths.feature_stats
    feature_stats_path = cfg.model.get("feature_stats_path", None)
    if feature_stats_path is None:
        feature_stats_path = cfg.paths.get("feature_stats", None)

    if model_name == "JanusGAP":
        model = JanusGAP(
            num_diseases=cfg.model.num_diseases,
            variant=cfg.model.variant,
            image_size=cfg.model.image_size,
            tri_stride=cfg.model.tri_stride,
            freeze_backbone=cfg.model.freeze_backbone,
            use_gradient_checkpointing=cfg.model.get("use_gradient_checkpointing", False),
        )
    elif model_name == "JanusMaskedAttn":
        model = JanusMaskedAttn(
            num_diseases=cfg.model.num_diseases,
            disease_names=cfg.model.get("disease_names", None),
            variant=cfg.model.variant,
            image_size=cfg.model.image_size,
            tri_stride=cfg.model.tri_stride,
            freeze_backbone=cfg.model.freeze_backbone,
            learn_tau=cfg.training.get("learn_tau", True),
            init_tau=cfg.training.get("init_tau", 0.7),
            fixed_tau=cfg.training.get("fixed_tau", 1.0),
            use_mask_bias=cfg.training.get("use_mask_bias", True),
            init_inside=cfg.training.get("init_inside", 0.8),
            init_outside=cfg.training.get("init_outside", 0.2),
            use_gradient_checkpointing=cfg.model.get("use_gradient_checkpointing", False),
            allow_comparative=cfg.model.get("allow_comparative", False),
        )
    elif model_name == "JanusScalarFusion":
        model = JanusScalarFusion(
            num_diseases=cfg.model.num_diseases,
            disease_names=cfg.model.get("disease_names", None),
            variant=cfg.model.variant,
            image_size=cfg.model.image_size,
            tri_stride=cfg.model.tri_stride,
            freeze_backbone=cfg.model.freeze_backbone,
            learn_tau=cfg.training.get("learn_tau", True),
            init_tau=cfg.training.get("init_tau", 0.7),
            fixed_tau=cfg.training.get("fixed_tau", 1.0),
            use_mask_bias=cfg.training.get("use_mask_bias", True),
            init_inside=cfg.training.get("init_inside", 0.8),
            init_outside=cfg.training.get("init_outside", 0.2),
            fusion_hidden=cfg.model.fusion_hidden,
            feature_stats_path=feature_stats_path,
            use_gradient_checkpointing=cfg.model.get("use_gradient_checkpointing", False),
        )
    elif model_name == "JanusScalarFusionVolume":
        model = JanusScalarFusionVolume(
            num_diseases=cfg.model.num_diseases,
            disease_names=cfg.model.get("disease_names", None),
            variant=cfg.model.variant,
            image_size=cfg.model.image_size,
            tri_stride=cfg.model.tri_stride,
            freeze_backbone=cfg.model.freeze_backbone,
            learn_tau=cfg.training.get("learn_tau", True),
            init_tau=cfg.training.get("init_tau", 0.7),
            fixed_tau=cfg.training.get("fixed_tau", 1.0),
            use_mask_bias=cfg.training.get("use_mask_bias", True),
            init_inside=cfg.training.get("init_inside", 0.8),
            init_outside=cfg.training.get("init_outside", 0.2),
            fusion_hidden=cfg.model.get("fusion_hidden", 256),
            feature_stats_path=feature_stats_path,
            use_gradient_checkpointing=cfg.model.get("use_gradient_checkpointing", False),
            debug=cfg.model.get("debug", False),
        )
    elif model_name == "JanusMaskedAttnOracle":
        model = JanusMaskedAttnOracle(
            num_diseases=cfg.model.num_diseases,
            num_organ_groups=cfg.model.get("num_organ_groups", 14),
            disease_names=cfg.model.get("disease_names", None),
            variant=cfg.model.variant,
            image_size=cfg.model.image_size,
            tri_stride=cfg.model.tri_stride,
            freeze_backbone=cfg.model.freeze_backbone,
            use_gradient_checkpointing=cfg.model.get("use_gradient_checkpointing", False),
            learn_tau=cfg.model.get("learn_tau", True),
            use_mask_bias=cfg.model.get("use_mask_bias", True),
        )
    elif model_name == "JanusScalarFusionOracle":
        model = JanusScalarFusionOracle(
            num_diseases=cfg.model.num_diseases,
            num_organ_groups=cfg.model.get("num_organ_groups", 14),
            disease_names=cfg.model.get("disease_names", None),
            variant=cfg.model.variant,
            image_size=cfg.model.image_size,
            tri_stride=cfg.model.tri_stride,
            freeze_backbone=cfg.model.freeze_backbone,
            use_gradient_checkpointing=cfg.model.get("use_gradient_checkpointing", False),
            learn_tau=cfg.model.get("learn_tau", True),
            use_mask_bias=cfg.model.get("use_mask_bias", True),
        )
    elif model_name == "JanusI3D_GAP":
        model = JanusI3D_GAP(
            num_diseases=cfg.model.num_diseases,
            disease_names=cfg.model.get("disease_names", None),
            resnet_name=cfg.model.get("resnet_name", "resnet50"),
            pretrained=cfg.model.get("pretrained", True),
            use_checkpoint=cfg.model.get("use_gradient_checkpointing", True),
        )
    elif model_name == "JanusI3D_MaskedAttn":
        model = JanusI3D_MaskedAttn(
            num_diseases=cfg.model.num_diseases,
            disease_names=cfg.model.get("disease_names", None),
            resnet_name=cfg.model.get("resnet_name", "resnet50"),
            pretrained=cfg.model.get("pretrained", True),
            use_checkpoint=cfg.model.get("use_gradient_checkpointing", True),
            feature_stats_path=feature_stats_path,
        )
    elif model_name == "JanusI3D_ScalarFusion":
        model = JanusI3D_ScalarFusion(
            num_diseases=cfg.model.num_diseases,
            disease_names=cfg.model.get("disease_names", None),
            resnet_name=cfg.model.get("resnet_name", "resnet50"),
            pretrained=cfg.model.get("pretrained", True),
            use_checkpoint=cfg.model.get("use_gradient_checkpointing", True),
            feature_stats_path=feature_stats_path,
        )
    elif model_name == "JanusResNet3DGAP":
        model = JanusResNet3DGAP(
            num_diseases=cfg.model.num_diseases,
            backbone=cfg.model.get("backbone", "resnet50"),
            pretrained=cfg.model.get("pretrained", True),
            use_checkpoint=cfg.model.get("use_checkpoint", True),
            freeze_backbone=cfg.model.get("freeze_backbone", False),
        )
    elif model_name == "JanusResNet3DMaskedAttn":
        model = JanusResNet3DMaskedAttn(
            num_diseases=cfg.model.num_diseases,
            disease_names=cfg.model.get("disease_names", None),
            backbone=cfg.model.get("backbone", "resnet50"),
            pretrained=cfg.model.get("pretrained", True),
            use_checkpoint=cfg.model.get("use_checkpoint", True),
            freeze_backbone=cfg.model.get("freeze_backbone", False),
            learn_tau=cfg.model.get("learn_tau", True),
            init_tau=cfg.model.get("init_tau", 0.7),
            fixed_tau=cfg.model.get("fixed_tau", 1.0),
            use_mask_bias=cfg.model.get("use_mask_bias", True),
            init_inside=cfg.model.get("init_inside", 0.8),
            init_outside=cfg.model.get("init_outside", 0.2),
        )
    elif model_name == "JanusResNet3DScalarFusion":
        model = JanusResNet3DScalarFusion(
            num_diseases=cfg.model.num_diseases,
            disease_names=cfg.model.get("disease_names", None),
            backbone=cfg.model.get("backbone", "resnet50"),
            pretrained=cfg.model.get("pretrained", True),
            use_checkpoint=cfg.model.get("use_checkpoint", True),
            freeze_backbone=cfg.model.get("freeze_backbone", False),
            learn_tau=cfg.model.get("learn_tau", True),
            init_tau=cfg.model.get("init_tau", 0.7),
            fixed_tau=cfg.model.get("fixed_tau", 1.0),
            use_mask_bias=cfg.model.get("use_mask_bias", True),
            init_inside=cfg.model.get("init_inside", 0.8),
            init_outside=cfg.model.get("init_outside", 0.2),
            visual_proj_dim=cfg.model.get("visual_proj_dim", 256),
            scalar_proj_dim=cfg.model.get("scalar_proj_dim", 256),
            fusion_hidden=cfg.model.get("fusion_hidden", 256),
            feature_stats_path=feature_stats_path,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def count_parameters(model: nn.Module) -> dict:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig, steps_per_epoch: int):
    """
    Build LR scheduler from config.

    Note: When rebuilding optimizers mid-training (e.g., unfreezing scalar heads),
    we also rebuild the scheduler with the same config.
    """
    # Scheduler with warmup and min_lr
    num_training_steps = steps_per_epoch * cfg.training.max_epochs
    warmup_epochs = cfg.training.get("warmup_epochs", 0)
    warmup_steps = steps_per_epoch * warmup_epochs
    min_lr_scale = cfg.training.get("min_lr_scale", 0.1)

    if cfg.training.scheduler == "cosine":
        from torch.optim.lr_scheduler import LambdaLR
        import math

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_scale + (1.0 - min_lr_scale) * cosine_decay

        return LambdaLR(optimizer, lr_lambda)

    return None


def compute_class_weights(dataset: JanusDataset, device: torch.device) -> torch.Tensor:
    """
    Compute per-disease class weights for weighted BCE loss.

    Computes pos_weight = num_negatives / num_positives for each disease.
    Ignores -1 labels.

    Args:
        dataset: JanusDataset
        device: torch device

    Returns:
        pos_weight tensor [num_diseases]
    """
    print("\nComputing class weights from labels CSV (fast)...")

    # Read directly from the labels dataframe - much faster than loading pack files!
    labels_df = dataset.labels_df
    disease_names = dataset.disease_names
    case_ids = dataset.case_ids

    # Filter to only training case_ids
    train_labels = labels_df.loc[case_ids]

    num_diseases = len(disease_names)
    pos_counts = np.zeros(num_diseases)
    neg_counts = np.zeros(num_diseases)

    # Count positives and negatives per disease
    for i, disease in enumerate(disease_names):
        labels_col = train_labels[disease]
        pos_counts[i] = (labels_col == 1).sum()
        neg_counts[i] = (labels_col == 0).sum()
        # -1 labels are ignored (not counted)

    # Compute pos_weight = neg / pos (avoid division by zero)
    pos_weights = []
    for i in range(num_diseases):
        if pos_counts[i] > 0:
            weight = neg_counts[i] / pos_counts[i]
        else:
            weight = 1.0  # Default if no positives
        pos_weights.append(weight)

    pos_weights = torch.tensor(pos_weights, dtype=torch.float32, device=device)

    # Print statistics
    print("\nClass distribution per disease:")
    print(f"{'Disease':<12} {'Positives':<12} {'Negatives':<12} {'Pos Weight':<12}")
    print("-" * 50)
    for i in range(min(10, num_diseases)):  # Show first 10
        print(f"{i:<12} {int(pos_counts[i]):<12} {int(neg_counts[i]):<12} {pos_weights[i].item():<12.3f}")
    if num_diseases > 10:
        print(f"... ({num_diseases - 10} more diseases)")
    print()

    return pos_weights


# =============================================================================
# Training & Validation
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, device, epoch, cfg, scheduler=None):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    num_batches = 0
    scheduler_step_mode = cfg.training.get("scheduler_step", "epoch")
    accumulation_steps = cfg.training.get("gradient_accumulation_steps", 1)
    use_amp = cfg.training.get("use_amp", True)  # Enable AMP by default
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer.zero_grad()

    # Only show progress bar on main process
    if is_main_process():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    else:
        pbar = loader

    accum_has_grad = False
    for batch_idx, batch in enumerate(pbar):
        if batch_idx % accumulation_steps == 0:
            accum_has_grad = False

        # Move to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        # Forward with automatic mixed precision
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(batch)
            labels = batch["labels"]

            # Compute loss (handles -1 labels internally based on loss type)
            if isinstance(criterion, BCEUncertainLoss):
                # BCEUncertainLoss - pass epoch
                loss = criterion(logits, labels, epoch=epoch, for_eval=False)
            else:
                # BCEWithLogitsLoss - manual masking for -1 labels
                # Note: Use reduction="none" to get per-element loss for proper masking
                mask = (labels != -1).float()
                loss_unreduced = F.binary_cross_entropy_with_logits(
                    logits, labels.clamp(0, 1), reduction="none"
                )
                loss = (loss_unreduced * mask).sum() / mask.sum().clamp(min=1.0)

            # Scale loss by accumulation steps (so gradients average correctly)
            loss = loss / accumulation_steps

        # Backward with gradient scaling
        # NOTE: Some debug/eval configurations can intentionally produce a loss that does not
        # require grad (e.g., scalar-only debug + all relevant params frozen). In that case,
        # skip backward/optimizer steps instead of crashing.
        if loss.requires_grad:
            scaler.scale(loss).backward()
            accum_has_grad = True
        else:
            if is_main_process() and batch_idx == 0:
                print(
                    "\n⚠️  Warning: loss.requires_grad=False; skipping backward/optimizer steps. "
                    "This is expected if all contributing parameters are frozen or a debug mode "
                    "bypasses trainable paths.\n"
                )

        # Only step optimizer every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0:
            if not accum_has_grad:
                optimizer.zero_grad()
                continue

            # Gradient clipping (unscale first for accurate clipping)
            if cfg.training.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.training.gradient_clip
                )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Step scheduler per optimizer update if configured
            if scheduler is not None and scheduler_step_mode == "step":
                scheduler.step()

        # Logging (log unscaled loss for readability)
        total_loss += loss.item() * accumulation_steps
        num_batches += 1
        if is_main_process() and hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({"loss": total_loss / num_batches})

    # Handle remaining gradients if last batch didn't trigger optimizer step
    if num_batches > 0 and (batch_idx + 1) % accumulation_steps != 0 and accum_has_grad:
        if cfg.training.gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.gradient_clip
            )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss / num_batches


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, split="Val", use_ddp=False, disease_names=None, use_amp=True):
    """Validate model."""
    model.eval()

    all_logits = []
    all_labels = []
    total_loss = 0
    num_batches = 0

    # Only show progress bar on main process
    if is_main_process():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [{split}]")
    else:
        pbar = loader

    for batch in pbar:
        # Move to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        # Forward with AMP (inference only, no backward pass needed)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(batch)
            labels = batch["labels"]

            # Loss (always use eval mode for validation - ignores uncertain labels)
            if isinstance(criterion, BCEUncertainLoss):
                # BCEUncertainLoss - pass epoch and for_eval=True
                loss = criterion(logits, labels, epoch=epoch, for_eval=True)
            else:
                # BCEWithLogitsLoss - manual masking for -1 labels
                # Note: This assumes criterion uses reduction="none" for per-element loss
                mask = (labels != -1).float()
                loss_unreduced = F.binary_cross_entropy_with_logits(
                    logits, labels.clamp(0, 1), reduction="none"
                )
                loss = (loss_unreduced * mask).sum() / mask.sum().clamp(min=1.0)

        total_loss += loss.item()
        num_batches += 1

        # Collect predictions (keep on device for DDP gathering)
        all_logits.append(logits.detach())
        all_labels.append(labels.detach())

        if is_main_process() and hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({"loss": total_loss / num_batches})

    # Concatenate all predictions from this rank
    all_logits = torch.cat(all_logits, dim=0)  # [N_local, num_diseases]
    all_labels = torch.cat(all_labels, dim=0)  # [N_local, num_diseases]

    # Gather predictions from all ranks if using DDP
    if use_ddp and dist.is_initialized():
        all_logits = concat_all_gather(all_logits)
        all_labels = concat_all_gather(all_labels)

    # Move to CPU after gathering
    all_logits = all_logits.cpu()
    all_labels = all_labels.cpu()

    # Compute average loss
    avg_loss = total_loss / num_batches

    # Only compute metrics on main process
    if is_main_process():
        # Compute metrics
        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        metrics = compute_metrics(probs, labels_np, disease_names=disease_names)
        metrics["loss"] = avg_loss
        return avg_loss, metrics
    else:
        # Non-main processes still need to return consistent values
        return avg_loss, {}


def compute_metrics(probs: np.ndarray, labels: np.ndarray, disease_names=None):
    """
    Compute evaluation metrics, ignoring -1 labels.

    Args:
        probs: [N, num_diseases] predicted probabilities
        labels: [N, num_diseases] ground truth (0, 1, or -1 for missing)
        disease_names: Optional list of disease names for readable keys

    Returns:
        dict with macro_auc, macro_ap, per_disease_auc, per_disease_ap
    """
    num_diseases = probs.shape[1]

    # Per-disease AUC and AP (ignoring -1)
    aucs = []
    aps = []
    per_disease_auc = {}
    per_disease_ap = {}

    for i in range(num_diseases):
        # Mask out -1 labels
        mask = labels[:, i] != -1

        if mask.sum() > 0 and len(np.unique(labels[mask, i])) > 1:
            try:
                auc = roc_auc_score(labels[mask, i], probs[mask, i])
                ap = average_precision_score(labels[mask, i], probs[mask, i])
                aucs.append(auc)
                aps.append(ap)

                # Use disease name if available, otherwise use index
                key = disease_names[i] if disease_names and i < len(disease_names) else f"disease_{i}"
                per_disease_auc[key] = auc
                per_disease_ap[key] = ap
            except Exception as e:
                # Handle edge cases (e.g., all same label)
                disease_name = disease_names[i] if disease_names and i < len(disease_names) else f"disease_{i}"
                print(f"Warning: Could not compute metrics for {disease_name}: {e}")
                continue

    # Macro averages
    macro_auc = np.mean(aucs) if aucs else 0.0
    macro_ap = np.mean(aps) if aps else 0.0

    return {
        "macro_auc": macro_auc,
        "macro_ap": macro_ap,
        "num_valid_diseases": len(aucs),
        "per_disease_auc": per_disease_auc,
        "per_disease_ap": per_disease_ap,
    }


# =============================================================================
# Main Training Loop
# =============================================================================

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""

    # ==========================================================================
    # LOAD DISEASE CONFIG FROM LR PIPELINE OUTPUT
    # ==========================================================================
    # The disease config (with pruned features) comes from the logistic regression
    # pipeline. This ensures we use the same features that were selected by the LR.
    disease_config_path = cfg.paths.get("disease_config")
    if disease_config_path:
        load_config_globally(disease_config_path)
        print(f"\n✓ Loaded disease config from LR pipeline: {disease_config_path}")
    else:
        raise ValueError(
            "paths.disease_config is required. "
            "Set it to the disease_config_final.py from your LR run, e.g.:\n"
            "  paths.disease_config=/path/to/lr_run/disease_config_final.py"
        )

    # Initialize DDP if requested
    use_ddp = cfg.training.get("use_ddp", False)
    use_amp = cfg.training.get("use_amp", True)  # Enable AMP by default
    rank, world_size, local_rank = 0, 1, 0

    if use_ddp:
        is_distributed = init_distributed(backend=cfg.training.get("backend", "nccl"))
        if is_distributed:
            rank, world_size, local_rank = ddp_setup()
            if is_main_process():
                print(f"Initialized DDP with {world_size} processes")
        else:
            use_ddp = False
            if is_main_process():
                print("Warning: DDP requested but not available, using single GPU")

    # Print config (main process only)
    if is_main_process():
        print("=" * 80)
        print("Configuration:")
        print(OmegaConf.to_yaml(cfg))
        print("=" * 80)

    # Initialize wandb (only on main process)
    use_wandb = cfg.logging.get("use_wandb", False) and WANDB_AVAILABLE and is_main_process()
    if use_wandb:
        wandb.init(
            project=cfg.logging.get("wandb_project", "janus"),
            entity=cfg.logging.get("wandb_entity"),
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{cfg.model.name}_{cfg.experiment if hasattr(cfg, 'experiment') else 'run'}",
            tags=[cfg.model.name],
        )
        print("\nWandB initialized!")
        print(f"  Project: {cfg.logging.get('wandb_project', 'janus')}")
        print(f"  Run URL: {wandb.run.get_url()}")
    elif cfg.logging.get("use_wandb", False) and not WANDB_AVAILABLE:
        print("\nWarning: wandb logging requested but wandb not installed!")
        print("Install with: pip install wandb")

    # Set seed (with rank offset for different data augmentation per GPU)
    set_seed(cfg.seed + rank)

    # Device
    if use_ddp:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        print(f"\nDevice: {device}")
        if use_ddp:
            print(f"Using DDP with {world_size} GPUs")

    # Load train/val/test IDs
    train_ids = load_ids_from_file(cfg.paths.train_ids)
    val_ids = load_ids_from_file(cfg.paths.val_ids)
    test_ids = load_ids_from_file(cfg.paths.test_ids) if cfg.paths.get("test_ids") else None

    if is_main_process():
        print(f"\nDataset splits:")
        print(f"  Train: {len(train_ids)} cases")
        print(f"  Val:   {len(val_ids)} cases")
        if test_ids:
            print(f"  Test:  {len(test_ids)} cases")

    # Build datasets
    if is_main_process():
        print("\nBuilding datasets...")

    # Only load features for ScalarFusion models
    features_parquet = None
    feature_columns = None
    if cfg.model.name in ["JanusScalarFusion", "JanusScalarFusionVolume", "JanusI3D_ScalarFusion", "JanusScalarFusionOracle"]:
        features_parquet = cfg.paths.get("features_parquet")
        feature_columns = cfg.model.get("feature_columns")  # Optional: specific columns to use

    # IMPORTANT: Ensure label/metric disease order matches model output order.
    # Get disease names from dynamically loaded config (from LR pipeline).
    all_diseases = get_all_diseases()
    disease_names_cfg = cfg.model.get("disease_names", None)
    if disease_names_cfg is None:
        disease_names_cfg = all_diseases[: cfg.model.num_diseases]
    if is_main_process():
        print("\nDisease ordering (must match model outputs):")
        print(f"  Num diseases: {len(disease_names_cfg)}")
        print(f"  First 5: {disease_names_cfg[:5]}")

    # Get augmentation settings
    use_augmentation = cfg.dataset.get("use_augmentation", False)
    aug_preset = cfg.dataset.get("aug_preset", "anatomy_safe_v2")
    aug_params_all = cfg.dataset.get("aug_params", {})
    aug_params = aug_params_all.get(aug_preset, {})

    if is_main_process() and use_augmentation:
        print(f"\n✓ Augmentation enabled:")
        print(f"  Preset: {aug_preset}")
        print(f"  Params: {aug_params}")

    train_dataset = JanusDataset(
        pack_root=cfg.paths.pack_root,
        labels_csv=cfg.paths.labels_csv,
        case_ids=train_ids,
        features_parquet=features_parquet,
        feature_columns=feature_columns,
        disease_names=disease_names_cfg,
        cache_packs=False,  # Dataset too large for full caching
        use_augmentation=use_augmentation,  # Enable augmentation for training
        aug_preset=aug_preset,
        aug_params=aug_params,
    )

    val_dataset = JanusDataset(
        pack_root=cfg.paths.pack_root,
        labels_csv=cfg.paths.labels_csv,
        case_ids=val_ids,
        features_parquet=features_parquet,
        feature_columns=feature_columns,
        disease_names=disease_names_cfg,
        cache_packs=False,  # Dataset too large for full caching
        use_augmentation=False,  # No augmentation for validation
    )

    # Get disease names for metrics
    disease_names = train_dataset.disease_names

    # Build samplers for DDP
    train_sampler = None
    val_sampler = None

    if use_ddp and dist.is_initialized():
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=True,  # CRITICAL: Avoid duplicating samples for metric computation
        )

    # Build loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Only shuffle if not using sampler
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        persistent_workers=cfg.dataset.persistent_workers,
        prefetch_factor=cfg.dataset.prefetch_factor,
        collate_fn=janus_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        persistent_workers=cfg.dataset.persistent_workers,
        prefetch_factor=cfg.dataset.prefetch_factor,
        collate_fn=janus_collate_fn,
    )

    if is_main_process():
        print(f"  Train loader: {len(train_loader)} batches")
        print(f"  Val loader:   {len(val_loader)} batches")

        # Print effective batch size with gradient accumulation
        accumulation_steps = cfg.training.get("gradient_accumulation_steps", 1)
        effective_batch = cfg.dataset.batch_size * world_size * accumulation_steps
        if accumulation_steps > 1:
            print(f"\n✓ Gradient accumulation enabled:")
            print(f"  Accumulation steps: {accumulation_steps}")
            print(f"  Batch per GPU: {cfg.dataset.batch_size}")
            print(f"  Number of GPUs: {world_size}")
            print(f"  Effective batch size: {effective_batch} ({cfg.dataset.batch_size} × {world_size} × {accumulation_steps})")
        else:
            print(f"\n✓ Effective batch size: {effective_batch} ({cfg.dataset.batch_size} × {world_size} GPUs)")

    # Build model
    if is_main_process():
        print("\nBuilding model...")

    model = build_model(cfg)
    model = model.to(device)

    # Wrap model in DDP
    if use_ddp and dist.is_initialized():
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=cfg.training.get("find_unused_parameters", False),
        )

        # Fix gradient checkpointing + DDP conflict
        # When using gradient checkpointing with DDP, we need to tell DDP that the graph is static
        # This prevents "Expected to mark a variable ready only once" errors
        if cfg.model.get("use_gradient_checkpointing", False) and not cfg.model.get("freeze_backbone", True):
            model._set_static_graph()
            if is_main_process():
                print("✓ Set static graph for DDP (gradient checkpointing + DDP compatibility)")

        if is_main_process():
            print("Model wrapped in DDP")

    if is_main_process():
        params = count_parameters(model)
        print(f"  Total parameters:     {params['total']:,}")
        print(f"  Trainable parameters: {params['trainable']:,}")
        print(f"  Frozen parameters:    {params['frozen']:,}")

    # Loss function with class weighting
    pos_weight = None
    if cfg.training.get("pos_weight") is not None:
        # Manual override: same weight for all diseases
        pos_weight = torch.tensor([cfg.training.pos_weight] * cfg.model.num_diseases).to(device)
        if is_main_process():
            print(f"\nUsing manual pos_weight: {cfg.training.pos_weight}")
    elif cfg.training.get("use_class_weights", True):
        # Compute per-disease weights from training data (only on main process)
        if is_main_process():
            pos_weight = compute_class_weights(train_dataset, device)

            # Clip pos_weight to prevent extreme values
            pos_weight_clip = cfg.training.get("pos_weight_clip", None)
            if pos_weight_clip is not None and pos_weight_clip > 0:
                max_before = pos_weight.max().item()
                pos_weight = torch.clamp(pos_weight, max=pos_weight_clip)
                max_after = pos_weight.max().item()
                if max_before > pos_weight_clip:
                    print(f"\n✓ Clipped pos_weight: max {max_before:.1f} → {max_after:.1f} (clip={pos_weight_clip})")

            print(f"\nUsing computed class weights (mean: {pos_weight.mean().item():.3f})")
        else:
            # Other ranks use placeholder, will be synced via broadcast
            pos_weight = torch.zeros(cfg.model.num_diseases, device=device)

        # Broadcast weights to all ranks
        if use_ddp and dist.is_initialized():
            dist.broadcast(pos_weight, src=0)
    else:
        if is_main_process():
            print("\nNo class weighting (balanced loss)")

    # Build loss function
    loss_config = {
        "loss_fn": cfg.training.loss_fn,
        "pos_weight": pos_weight,
        "uncertain_ignore_epochs": cfg.training.get("uncertain_ignore_epochs", 10),
        "uncertain_ramp_epochs": cfg.training.get("uncertain_ramp_epochs", 10),
        "uncertain_final_weight": cfg.training.get("uncertain_final_weight", 0.3),
        "uncertain_target": cfg.training.get("uncertain_target", 0.0),
    }
    criterion = build_loss_from_config(loss_config).to(device)

    if is_main_process():
        loss_type = cfg.training.loss_fn
        print(f"\nLoss function: {loss_type}")
        if loss_type == "bce_uncertain":
            print(f"  Uncertain label handling:")
            print(f"    Ignore epochs: {loss_config['uncertain_ignore_epochs']}")
            print(f"    Ramp epochs: {loss_config['uncertain_ramp_epochs']}")
            print(f"    Final weight: {loss_config['uncertain_final_weight']}")
            print(f"    Target value: {loss_config['uncertain_target']}")

    # Optimizer with grouped learning rates
    base_lr = cfg.training.learning_rate
    weight_decay = cfg.training.weight_decay
    use_group_lrs = cfg.training.get("use_group_lrs", False)

    if use_group_lrs:
        # Separate parameter groups for backbone, heads, and alpha params
        head_lr_scale = cfg.training.get("head_lr_scale", 3.0)
        alpha_lr_scale = cfg.training.get("alpha_lr_scale", 0.3)

        # Get model without DDP wrapper
        unwrapped_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

        # Name-based parameter grouping (more robust than attribute-based)
        # This ensures ALL parameters are captured, including:
        # - fusion_heads, visual_projectors, scalar_projectors (JanusScalarFusion)
        # - score_mlps (JanusMaskedAttn, JanusScalarFusion)
        # - heads (all models)
        backbone_params = []
        alpha_params = []
        head_params = []

        for name, param in unwrapped_model.named_parameters():
            if not param.requires_grad:
                continue

            # Backbone parameters
            if name.startswith("backbone."):
                backbone_params.append(param)

            # Alpha parameters (temperature, bias, gating logits, mixture weights)
            elif any(key in name for key in [
                "temp_logit", "inside_logit", "outside_logit",
                "border_gate_logit", "border_gate",
                "visual_gate_logit", "scalar_gate_logit",
                "alpha_scalar"  # Mixture weights for dual-head fusion
            ]):
                alpha_params.append(param)

            # Everything else goes to heads (heads, score_mlps, fusion_heads, projectors, etc.)
            else:
                head_params.append(param)

        # Build parameter groups
        param_groups = []
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": base_lr,
                "name": "backbone"
            })
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": base_lr * head_lr_scale,
                "name": "heads"
            })
        if alpha_params:
            param_groups.append({
                "params": alpha_params,
                "lr": base_lr * alpha_lr_scale,
                "name": "alpha"
            })

        # Fallback: if no groups identified, use all parameters
        if not param_groups:
            param_groups = [{"params": model.parameters(), "lr": base_lr}]

        if is_main_process():
            print("\n✓ Grouped learning rates:")
            for pg in param_groups:
                n_params = sum(p.numel() for p in pg["params"])
                print(f"  {pg.get('name', 'default'):12s}: LR={pg['lr']:.6f}  ({n_params:,} params)")

        if cfg.training.optimizer == "adamw":
            optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {cfg.training.optimizer}")

    else:
        # Standard optimizer (all params same LR)
        if cfg.training.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=base_lr,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.training.optimizer}")

        if is_main_process():
            print(f"\n✓ Uniform learning rate: {base_lr}")

    # Scheduler with warmup and min_lr
    num_training_steps = len(train_loader) * cfg.training.max_epochs
    warmup_epochs = cfg.training.get("warmup_epochs", 0)
    warmup_steps = len(train_loader) * warmup_epochs
    min_lr_scale = cfg.training.get("min_lr_scale", 0.1)
    scheduler_step_mode = cfg.training.get("scheduler_step", "epoch")

    if cfg.training.scheduler == "cosine":
        from torch.optim.lr_scheduler import LambdaLR
        import math

        def lr_lambda(current_step):
            # Warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # Cosine decay phase
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_scale + (1.0 - min_lr_scale) * cosine_decay

        scheduler = LambdaLR(optimizer, lr_lambda)

        if is_main_process():
            print(f"✓ Cosine scheduler: warmup={warmup_epochs} epochs, min_lr_scale={min_lr_scale}, step_mode={scheduler_step_mode}")
    else:
        scheduler = None

    # =========================================================================
    # Training loop
    # =========================================================================
    if is_main_process():
        print("\n" + "=" * 80)
        print("Starting training...")
        print("=" * 80 + "\n")

    best_metric = -float("inf") if cfg.training.monitor_mode == "max" else float("inf")
    best_epoch = 0

    for epoch in range(1, cfg.training.max_epochs + 1):
        # Set epoch for DistributedSampler (ensures different shuffle each epoch)
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, cfg, scheduler=scheduler)

        # Validate
        if epoch % cfg.training.val_every_n_epochs == 0:
            val_loss, val_metrics = validate(model, val_loader, criterion, device, epoch, split="Val", use_ddp=use_ddp, disease_names=disease_names, use_amp=use_amp)

            # Only log and save on main process
            if is_main_process():
                print(f"\nEpoch {epoch}/{cfg.training.max_epochs}:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss:   {val_loss:.4f}")
                print(f"  Val Macro AUC: {val_metrics['macro_auc']:.4f}")
                print(f"  Val Macro AP:  {val_metrics['macro_ap']:.4f}")

                # Print per-class metrics
                print("\n  Per-Disease Metrics:")
                per_disease_auc = val_metrics.get('per_disease_auc', {})
                per_disease_ap = val_metrics.get('per_disease_ap', {})
                for disease in disease_names:
                    if disease in per_disease_auc and disease in per_disease_ap:
                        print(f"    {disease:30s} | AUC: {per_disease_auc[disease]:.4f} | AUPRC: {per_disease_ap[disease]:.4f}")

                # Save per-disease metrics to JSONL file
                from hydra.core.hydra_config import HydraConfig
                import json
                hydra_cfg = HydraConfig.get()
                output_dir = Path(hydra_cfg.runtime.output_dir)
                metrics_file = output_dir / "per_disease_metrics.jsonl"

                with open(metrics_file, 'a') as f:
                    for disease in disease_names:
                        if disease in per_disease_auc and disease in per_disease_ap:
                            metric_entry = {
                                "epoch": epoch,
                                "disease": disease,
                                "auroc": float(per_disease_auc[disease]),
                                "auprc": float(per_disease_ap[disease])
                            }
                            f.write(json.dumps(metric_entry) + '\n')

                # Log to wandb
                if use_wandb:
                    wandb_dict = {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "train/lr": optimizer.param_groups[0]['lr'],
                        "val/loss": val_metrics['loss'],
                        "val/macro_auc": val_metrics['macro_auc'],
                        "val/macro_ap": val_metrics['macro_ap'],
                    }

                    # Add per-disease metrics to wandb
                    for disease in disease_names:
                        if disease in per_disease_auc:
                            wandb_dict[f"val/auc/{disease}"] = per_disease_auc[disease]
                        if disease in per_disease_ap:
                            wandb_dict[f"val/auprc/{disease}"] = per_disease_ap[disease]

                    wandb.log(wandb_dict, step=epoch)

                # Check if best
                monitor_value = val_metrics[cfg.training.monitor_metric.split("/")[-1]]
                is_best = False

                if cfg.training.monitor_mode == "max":
                    if monitor_value > best_metric:
                        best_metric = monitor_value
                        best_epoch = epoch
                        is_best = True
                else:
                    if monitor_value < best_metric:
                        best_metric = monitor_value
                        best_epoch = epoch
                        is_best = True

                # Save checkpoint with improved naming
                # Use Hydra's output directory to avoid overwriting between experiments
                from hydra.core.hydra_config import HydraConfig
                hydra_cfg = HydraConfig.get()
                output_dir = Path(hydra_cfg.runtime.output_dir)
                checkpoint_dir = output_dir / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Get model state dict (unwrap DDP if needed)
                model_to_save = model.module if use_ddp else model

                # Model name mapping for shorter filenames
                model_name_map = {
                    "JanusGAP": "gap",
                    "JanusMaskedAttn": "masked_attn",
                    "JanusScalarFusion": "scalar_fusion",
                }
                model_short = model_name_map.get(cfg.model.name, cfg.model.name.lower())

                # Metric name for filename (e.g., "auc" from "val/macro_auc")
                metric_name = cfg.training.monitor_metric.split("/")[-1]

                # Prepare checkpoint data
                checkpoint_data = {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "val_metrics": val_metrics,
                    "best_metric": best_metric if is_best else None,
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                }

                # Save best checkpoint
                if is_best:
                    # Remove old best checkpoint
                    for old_best in checkpoint_dir.glob(f"{model_short}_best_*.pt"):
                        old_best.unlink()

                    best_ckpt_name = f"{model_short}_best_{metric_name}{monitor_value:.4f}.pt"
                    torch.save(checkpoint_data, checkpoint_dir / best_ckpt_name)
                    print(f"  *** New best model saved: {best_ckpt_name} ***")

                # Save periodic checkpoints (if not save_best_only)
                if not cfg.training.save_best_only:
                    if epoch % cfg.training.save_every_n_epochs == 0:
                        epoch_ckpt_name = f"{model_short}_epoch{epoch:03d}_{metric_name}{monitor_value:.4f}.pt"
                        torch.save(checkpoint_data, checkpoint_dir / epoch_ckpt_name)
                        print(f"  Checkpoint saved: {epoch_ckpt_name}")

                        # Keep only top-K checkpoints
                        keep_top_k = cfg.training.get("keep_top_k", -1)
                        if keep_top_k > 0:
                            # Get all epoch checkpoints sorted by metric
                            epoch_ckpts = list(checkpoint_dir.glob(f"{model_short}_epoch*_{metric_name}*.pt"))
                            if len(epoch_ckpts) > keep_top_k:
                                # Extract metric values from filenames
                                ckpt_metrics = []
                                for ckpt in epoch_ckpts:
                                    try:
                                        # Extract metric value from filename
                                        metric_str = ckpt.stem.split(f"_{metric_name}")[-1]
                                        metric_val = float(metric_str.replace(".pt", ""))
                                        ckpt_metrics.append((ckpt, metric_val))
                                    except:
                                        continue

                                # Sort by metric (ascending or descending based on monitor_mode)
                                reverse = (cfg.training.monitor_mode == "max")
                                ckpt_metrics.sort(key=lambda x: x[1], reverse=reverse)

                                # Remove worst checkpoints
                                for ckpt, _ in ckpt_metrics[keep_top_k:]:
                                    ckpt.unlink()
                                    print(f"  Removed old checkpoint: {ckpt.name}")

                # Always save last checkpoint (for resuming)
                if cfg.training.get("save_last", True):
                    last_ckpt_name = f"{model_short}_last.pt"
                    torch.save(checkpoint_data, checkpoint_dir / last_ckpt_name)

        # Step scheduler per epoch if configured
        scheduler_step_mode = cfg.training.get("scheduler_step", "epoch")
        if scheduler is not None and scheduler_step_mode == "epoch":
            scheduler.step()

    if is_main_process():
        print("\n" + "=" * 80)
        print("Training complete!")
        print(f"Best {cfg.training.monitor_metric}: {best_metric:.4f} (epoch {best_epoch})")
        print("=" * 80)

        # Save run summary
        summary = {
            "model": cfg.model.name,
            "variant": cfg.model.variant,
            "seed": cfg.seed,
            "best_val_metric": float(best_metric),
            "best_epoch": best_epoch,
            "monitor_metric": cfg.training.monitor_metric,
            "total_epochs": cfg.training.max_epochs,
            "batch_size": cfg.dataset.batch_size,
            "learning_rate": cfg.training.learning_rate,
            "num_diseases": cfg.model.num_diseases,
        }

        # Save summary to Hydra output directory
        from hydra.core.hydra_config import HydraConfig
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)
        summary_path = output_dir / "run_summary.json"
        with open(summary_path, 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        print(f"\nRun summary saved to: {summary_path}")

        # Log final best metrics to wandb
        if use_wandb:
            wandb.log({
                f"best/{cfg.training.monitor_metric}": best_metric,
                "best/epoch": best_epoch,
            })
            wandb.finish()
            print("WandB run finished!")

    # Cleanup DDP
    if use_ddp:
        ddp_cleanup()


if __name__ == "__main__":
    main()
