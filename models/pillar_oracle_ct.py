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

"""
Pillar-0 foundation model variants for OracleCT.

Two model variants:
1. OracleCT_Pillar_GAP:       Pillar-0 + Global Average Pooling (baseline)
2. OracleCT_Pillar_MaskedAttn: Pillar-0 + 3D Organ-Masked Attention

Pillar-0 is a 3D vision-language foundation model pretrained on abdominal CT.
Input: [B, 11, 384, 384, 384] — 11 HU windows (10 anatomical + minmax)
Outputs:
  pooled: [B, 1152] — L2-normalised global features (3 scales maxpool-cat)
  activ:  [B, 1152, 64, 64, 64] — spatial feature map (3 scales interp-cat)

Option B: apply 11 windows at load time, keep pretrained weights unchanged.
"""

import sys
import math
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Add pillar-finetune to sys.path so MultimodalAtlas can be imported
# (both live under oracle-ct/, two sibling directories)
# ---------------------------------------------------------------------------
_PILLAR_FINETUNE = Path(__file__).resolve().parent.parent.parent / "pillar-finetune"
if _PILLAR_FINETUNE.exists() and str(_PILLAR_FINETUNE) not in sys.path:
    sys.path.insert(0, str(_PILLAR_FINETUNE))

from pillar.models.backbones.mmatlas import MultimodalAtlas  # noqa: E402

from ..configs.disease_config import (
    ORGAN_TO_CHANNEL,
    get_all_diseases,
    get_all_disease_configs,
)
from .dinov3_oracle_ct import (
    to_logit,
    inv_sigmoid_temp,
    get_attention_mask_for_disease,
)

# Pillar-0 AbdomenCT dimensions (from model config JSON)
PILLAR_HIDDEN_DIM = 1152   # 3 scales × 384-dim (embed_dim in CLIP config)
PILLAR_FEATURE_MAP = 64    # 384 / 6 (patch_size) = 64 spatial tokens per dim

# Default HuggingFace repo for the pretrained model
DEFAULT_REPO_ID = "YalaLab/Pillar0-AbdomenCT"


# =============================================================================
# 3D MASKED ATTENTION POOLING
# =============================================================================

def masked_attention_pool_3d(
    feat_map: torch.Tensor,          # [B, C, D', H', W']
    mask_3d: torch.Tensor,           # [B, 1, D, H, W] at any resolution
    score_conv: nn.Module,           # Conv3d(C, 1, kernel_size=1)
    tau: Union[float, torch.Tensor] = 1.0,
    bias_in: Optional[torch.Tensor] = None,   # learnable prior for inside mask
    bias_out: Optional[torch.Tensor] = None,  # learnable prior for outside mask
) -> torch.Tensor:                   # [B, C]
    """
    Content-aware attention restricted to organ mask with learnable priors.

    Mirrors masked_attention_pool from dinov3_oracle_ct.py but operates on
    3D spatial feature maps [B, C, D', H', W'] instead of 2D token sequences.

    The organ mask at original resolution is resized to match the feature map
    spatial dims using nearest-neighbour interpolation.
    """
    B, C, Df, Hf, Wf = feat_map.shape
    N = Df * Hf * Wf

    # Resize mask to feature map spatial dims
    mask = F.interpolate(
        mask_3d.float(), size=(Df, Hf, Wf), mode="nearest"
    )  # [B, 1, Df, Hf, Wf]
    mask_flat = (mask > 0.5).float().view(B, N)  # [B, N]

    # Compute attention logits via per-disease score conv
    logits = score_conv(feat_map).view(B, N)  # [B, N]

    # Learnable priors: push attention inside/outside the organ mask
    if bias_in is not None and bias_out is not None:
        logits = (
            logits
            + bias_in.view(1, 1) * mask_flat
            + bias_out.view(1, 1) * (1.0 - mask_flat)
        )

    # Temperature scaling
    if isinstance(tau, torch.Tensor):
        logits = logits / tau.view(1, 1)
    else:
        logits = logits / tau

    # Masked softmax: suppress outside-mask logits
    very_neg = torch.finfo(logits.dtype).min / 4
    masked_logits = torch.where(
        mask_flat > 0, logits, torch.full_like(logits, very_neg)
    )

    # Numerically stable softmax over valid positions
    lmax = masked_logits.max(dim=-1, keepdim=True).values
    exp_w = torch.exp((masked_logits - lmax).clamp(-60, 0)) * mask_flat
    weights = exp_w / exp_w.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # Fall back to uniform if mask is empty (disease not present in scan)
    empty = mask_flat.sum(dim=-1, keepdim=True) < 0.5
    if empty.any():
        uniform = torch.ones_like(weights) / N
        weights = torch.where(empty, uniform, weights)

    # NaN safety
    if torch.isnan(weights).any():
        uniform = torch.ones_like(weights) / N
        weights = torch.where(torch.isnan(weights), uniform, weights)

    # Weighted sum over spatial dims → [B, C]
    feat_flat = feat_map.view(B, C, N)  # [B, C, N]
    pooled = (feat_flat * weights.unsqueeze(1)).sum(dim=-1)  # [B, C]

    if torch.isnan(pooled).any():
        pooled = torch.where(torch.isnan(pooled), torch.zeros_like(pooled), pooled)

    return pooled


# =============================================================================
# PILLAR BACKBONE HELPER
# =============================================================================

def _build_pillar_backbone(
    model_repo_id: str,
    model_revision: Optional[str],
) -> MultimodalAtlas:
    """
    Load the Pillar-0 backbone via MultimodalAtlas wrapper.

    MultimodalAtlas wraps CLIPMultimodalAtlas (HuggingFace) and exposes:
      .model        — CLIPMultimodalAtlas
      .visual       — MultiModalAtlas vision encoder
      .hidden_dim   — 1152
      .forward(x, batch) → {"activ": [B,1152,64,64,64], "pooled": [B,1152]}
    """
    try:
        from easydict import EasyDict
    except ImportError:
        raise ImportError(
            "easydict is required for Pillar backbone. Install with: pip install easydict"
        )

    args = EasyDict({})
    backbone = MultimodalAtlas(
        args=args,
        device="cpu",           # train.py will move to GPU via model.to(device)
        model_repo_id=model_repo_id,
        model_revision=model_revision,
        pretrained=True,
    )
    return backbone


# =============================================================================
# MODEL 1: PILLAR GAP BASELINE
# =============================================================================

class OracleCT_Pillar_GAP(nn.Module):
    """
    Pillar-0 + Global Average Pooling (baseline, Option B).

    Uses Pillar's pre-normalised pooled [B, 1152] directly → per-disease Linear.
    Requires 11-channel windowed input [B, 11, 384, 384, 384].
    No organ-specific attention, no scalar features.
    """

    def __init__(
        self,
        num_diseases: int = 30,
        disease_names: Optional[List[str]] = None,
        model_repo_id: str = DEFAULT_REPO_ID,
        model_revision: Optional[str] = None,
        freeze_backbone: bool = False,
        modality: str = "abdomen_ct",
    ):
        super().__init__()

        self.num_diseases = num_diseases
        self.disease_names = disease_names or get_all_diseases()[:num_diseases]
        self.modality = modality
        self.freeze_backbone = freeze_backbone

        # Load Pillar backbone
        print(f"Loading Pillar backbone from: {model_repo_id}")
        self.backbone = _build_pillar_backbone(model_repo_id, model_revision)
        self.hidden_dim = self.backbone.hidden_dim  # 1152

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        else:
            self.backbone.train()

        # Per-disease linear heads
        self.heads = nn.ModuleDict()
        for disease in self.disease_names:
            self.heads[disease] = nn.Linear(self.hidden_dim, 1)
            nn.init.normal_(self.heads[disease].weight, 0.0, 0.01)
            nn.init.constant_(self.heads[disease].bias, 0.0)

        print(
            f"OracleCT_Pillar_GAP: {model_repo_id} → {self.hidden_dim}d pooled → "
            f"{num_diseases} diseases"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        image = batch["image"]   # [B, 11, 384, 384, 384]
        B = image.size(0)

        # Pillar forward
        pillar_batch = {"anatomy": [self.modality]}
        output = self.backbone(image, batch=pillar_batch)
        pooled = output["pooled"]  # [B, 1152]

        # Per-disease heads
        logits_list = []
        for disease in self.disease_names:
            logit = self.heads[disease](pooled)  # [B, 1]
            logits_list.append(logit)

        return torch.cat(logits_list, dim=1)  # [B, num_diseases]


# =============================================================================
# MODEL 2: PILLAR 3D MASKED ATTENTION
# =============================================================================

class OracleCT_Pillar_MaskedAttn(nn.Module):
    """
    Pillar-0 + 3D Organ-Masked Attention (Option B).

    Uses Pillar's spatial activ [B, 1152, 64, 64, 64] with per-disease
    Conv3d(1152, 1, 1) attention scorer. Organ masks from existing .pt packs
    are resized online from original resolution to 64³ for attention pooling.

    Requires 11-channel windowed input [B, 11, 384, 384, 384].
    """

    def __init__(
        self,
        num_diseases: int = 30,
        disease_names: Optional[List[str]] = None,
        model_repo_id: str = DEFAULT_REPO_ID,
        model_revision: Optional[str] = None,
        freeze_backbone: bool = False,
        modality: str = "abdomen_ct",
        learn_tau: bool = True,
        init_tau: float = 0.7,
        fixed_tau: float = 1.0,
        use_mask_bias: bool = True,
        init_inside: float = 0.8,
        init_outside: float = 0.2,
    ):
        super().__init__()

        self.num_diseases = num_diseases
        self.disease_names = disease_names or get_all_diseases()[:num_diseases]
        self.modality = modality
        self.freeze_backbone = freeze_backbone
        self.learn_tau = learn_tau
        self.fixed_tau = fixed_tau
        self.use_mask_bias = use_mask_bias

        # Load Pillar backbone
        print(f"Loading Pillar backbone from: {model_repo_id}")
        self.backbone = _build_pillar_backbone(model_repo_id, model_revision)
        self.hidden_dim = self.backbone.hidden_dim  # 1152

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        else:
            self.backbone.train()

        # Per-disease modules
        self.score_convs = nn.ModuleDict()
        self.heads = nn.ModuleDict()

        if use_mask_bias:
            self.inside_logit = nn.ParameterDict()
            self.outside_logit = nn.ParameterDict()
        if learn_tau:
            self.temp_logit = nn.ParameterDict()

        for disease in self.disease_names:
            # Attention scorer: 1×1×1 conv, very lightweight
            self.score_convs[disease] = nn.Conv3d(self.hidden_dim, 1, kernel_size=1)
            # Classification head
            self.heads[disease] = nn.Linear(self.hidden_dim, 1)
            nn.init.normal_(self.heads[disease].weight, 0.0, 0.01)
            nn.init.constant_(self.heads[disease].bias, 0.0)

            if use_mask_bias:
                self.inside_logit[disease] = nn.Parameter(
                    torch.tensor(to_logit(init_inside))
                )
                self.outside_logit[disease] = nn.Parameter(
                    torch.tensor(to_logit(init_outside))
                )
            if learn_tau:
                self.temp_logit[disease] = nn.Parameter(
                    torch.tensor(inv_sigmoid_temp(init_tau))
                )

        print(
            f"OracleCT_Pillar_MaskedAttn: {model_repo_id} → {self.hidden_dim}d activ "
            f"[64³] → 3D masked attn → {num_diseases} diseases"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        image = batch["image"]   # [B, 11, 384, 384, 384]
        masks = batch["masks"]   # [B, 20, D_mask, H_mask, W_mask] (original resolution)
        disease_rois = batch.get("disease_rois", [{}] * image.size(0))
        meta = batch.get("meta", [{}] * image.size(0))

        B = image.size(0)
        device = image.device

        # Pillar forward → spatial features
        pillar_batch = {"anatomy": [self.modality]}
        output = self.backbone(image, batch=pillar_batch)
        activ = output["activ"]  # [B, 1152, 64, 64, 64]

        # Per-disease masked attention
        logits_list = []
        for disease in self.disease_names:
            # Get organ mask (handles all strategies: single/union/roi/global)
            attn_mask = get_attention_mask_for_disease(
                disease, masks, disease_rois, meta, device,
                allow_comparative=False,
            )
            # attn_mask: [B, 1, D_mask, H_mask, W_mask]

            bias_in = self.inside_logit[disease] if self.use_mask_bias else None
            bias_out = self.outside_logit[disease] if self.use_mask_bias else None
            tau = (
                0.2 + 1.8 * torch.sigmoid(self.temp_logit[disease])
                if self.learn_tau
                else self.fixed_tau
            )

            # Pool spatial features using organ mask
            pooled = masked_attention_pool_3d(
                activ, attn_mask, self.score_convs[disease],
                tau=tau, bias_in=bias_in, bias_out=bias_out,
            )  # [B, 1152]

            logit = self.heads[disease](pooled)  # [B, 1]
            if torch.isnan(logit).any():
                logit = torch.where(
                    torch.isnan(logit), torch.zeros_like(logit), logit
                )
            logits_list.append(logit)

        return torch.cat(logits_list, dim=1)  # [B, num_diseases]
