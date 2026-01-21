# radioprior_v2/losses.py
"""
Loss functions for RadioPrior training.

Supports:
1. Standard BCE with logits (with optional class weights)
2. BCE with uncertain label handling (-1 labels with ramped weighting)
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BCEUncertainCfg:
    """Configuration for uncertain label handling."""
    ignore_epochs: int = 10
    ramp_epochs: int = 10
    final_weight: float = 0.3     # weight applied to -1 labels after ramp
    target: float = 0.0           # target value for -1 labels


class BCEWithLogitsLoss(nn.Module):
    """
    Standard BCE with logits loss.

    Args:
        pos_weight: Per-class positive weights [num_diseases] or None
        reduction: 'mean' or 'sum'
    """
    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        epoch: Optional[int] = None,  # For compatibility with BCEUncertainLoss
        for_eval: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, num_diseases]
            targets: [B, num_diseases] in {0, 1}
            epoch: Ignored (for interface compatibility)
            for_eval: Ignored (for interface compatibility)

        Returns:
            loss: scalar
        """
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )


class BCEUncertainLoss(nn.Module):
    """
    Multi-label BCE with ramped weighting for uncertain labels (-1).

    Uncertain labels (-1) are:
    - Ignored for first `ignore_epochs` epochs
    - Gradually ramped in over `ramp_epochs` epochs
    - Treated as `target` (default 0.0) with weight `final_weight` (default 0.3)

    Args:
        pos_weight: Per-class positive weights [num_diseases] or None
        ignore_epochs: Number of epochs to completely ignore -1 labels
        ramp_epochs: Number of epochs to ramp up -1 label weight
        final_weight: Final weight for -1 labels (0.0 to 1.0)
        target: Target value for -1 labels (typically 0.0)

    Example:
        epoch=0-9:   -1 labels have weight 0.0 (ignored)
        epoch=10-19: -1 labels ramp from 0.0 to 0.3
        epoch=20+:   -1 labels have weight 0.3, target 0.0
    """
    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        ignore_epochs: int = 10,
        ramp_epochs: int = 10,
        final_weight: float = 0.3,
        target: float = 0.0,
    ):
        super().__init__()
        self.cfg = BCEUncertainCfg(
            ignore_epochs=ignore_epochs,
            ramp_epochs=ramp_epochs,
            final_weight=final_weight,
            target=target,
        )
        # Register pos_weight as buffer so it moves with .to(device)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.register_buffer("pos_weight", None, persistent=False)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        epoch: int,
        for_eval: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, num_diseases]
            targets: [B, num_diseases] in {0, 1, -1}
            epoch: Current training epoch (0-indexed)
            for_eval: If True, ignore uncertain labels completely

        Returns:
            loss: scalar
        """
        # Clone targets and create masks
        y_eff = targets.clone()
        labeled = (y_eff >= 0)      # {0, 1} labels
        uncertain = ~labeled         # {-1} labels

        # Replace -1 with target value (default 0.0)
        y_eff = torch.where(uncertain, torch.full_like(y_eff, self.cfg.target), y_eff)

        # Compute weight for uncertain labels
        if for_eval or epoch < self.cfg.ignore_epochs:
            w_unc = 0.0
        else:
            # Linear ramp from 0 to final_weight over ramp_epochs
            t = (epoch - self.cfg.ignore_epochs) / max(1, self.cfg.ramp_epochs)
            w_unc = min(self.cfg.final_weight, self.cfg.final_weight * t)

        # Create per-sample weights: 1.0 for labeled, w_unc for uncertain
        sample_weights = torch.where(
            labeled,
            torch.ones_like(y_eff),
            torch.full_like(y_eff, w_unc)
        )

        # Compute BCE with logits
        loss_per_element = F.binary_cross_entropy_with_logits(
            logits,
            y_eff,
            weight=sample_weights,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        # Normalize by total weight (avoid division by zero)
        total_weight = sample_weights.sum().clamp_min(1.0)
        loss = loss_per_element.sum() / total_weight

        return loss


def build_loss_fn(
    loss_type: str,
    pos_weight: Optional[torch.Tensor] = None,
    **kwargs
) -> nn.Module:
    """
    Build loss function from config.

    Args:
        loss_type: "bce_with_logits" or "bce_uncertain"
        pos_weight: Per-class positive weights [num_diseases] or None
        **kwargs: Additional loss-specific arguments
            For bce_with_logits:
                - reduction: 'mean' or 'sum' (default: 'mean')
            For bce_uncertain:
                - ignore_epochs: int (default: 10)
                - ramp_epochs: int (default: 10)
                - final_weight: float (default: 0.3)
                - target: float (default: 0.0)

    Returns:
        Loss function (nn.Module)
    """
    loss_type = loss_type.lower()

    if loss_type == "bce_with_logits":
        reduction = kwargs.get("reduction", "mean")
        return BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction=reduction,
        )

    elif loss_type == "bce_uncertain":
        return BCEUncertainLoss(
            pos_weight=pos_weight,
            ignore_epochs=kwargs.get("ignore_epochs", 10),
            ramp_epochs=kwargs.get("ramp_epochs", 10),
            final_weight=kwargs.get("final_weight", 0.3),
            target=kwargs.get("target", 0.0),
        )

    else:
        raise ValueError(
            f"Unknown loss_type: '{loss_type}'. "
            f"Must be 'bce_with_logits' or 'bce_uncertain'."
        )


def build_loss_from_config(cfg: Dict[str, Any]) -> nn.Module:
    """
    Build loss from training config dict.

    Expected config structure:
        training:
            loss_fn: "bce_with_logits" or "bce_uncertain"
            use_class_weights: bool
            pos_weight: Optional[Tensor] (computed elsewhere)

            # Optional: bce_uncertain specific
            uncertain_ignore_epochs: 10
            uncertain_ramp_epochs: 10
            uncertain_final_weight: 0.3
            uncertain_target: 0.0

    Args:
        cfg: Config dictionary

    Returns:
        Loss function (nn.Module)
    """
    loss_type = cfg.get("loss_fn", "bce_with_logits")
    pos_weight = cfg.get("pos_weight", None)

    # Build loss-specific kwargs
    kwargs = {}

    if loss_type == "bce_uncertain":
        kwargs = {
            "ignore_epochs": cfg.get("uncertain_ignore_epochs", 10),
            "ramp_epochs": cfg.get("uncertain_ramp_epochs", 10),
            "final_weight": cfg.get("uncertain_final_weight", 0.3),
            "target": cfg.get("uncertain_target", 0.0),
        }

    return build_loss_fn(loss_type, pos_weight=pos_weight, **kwargs)


if __name__ == "__main__":
    print("RadioPrior Loss Functions")
    print("=" * 60)
    print("""
Available loss functions:

1. BCEWithLogitsLoss (standard)
   - Standard binary cross-entropy with logits
   - Supports per-class positive weights
   - Use for clean labels in {0, 1}

2. BCEUncertainLoss (uncertain label handling)
   - Handles uncertain labels (-1) with ramped weighting
   - Gradually introduces uncertain labels during training
   - Use when labels contain {0, 1, -1}

Usage:
    from radioprior_v2.losses import build_loss_fn

    # Standard BCE
    loss_fn = build_loss_fn("bce_with_logits", pos_weight=weights)
    loss = loss_fn(logits, targets)

    # Uncertain BCE
    loss_fn = build_loss_fn("bce_uncertain", pos_weight=weights,
                           ignore_epochs=10, ramp_epochs=10)
    loss = loss_fn(logits, targets, epoch=current_epoch)
    """)
