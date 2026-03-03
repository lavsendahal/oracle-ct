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

Four model variants:
1. OracleCT_Pillar_GAP:              Pillar-0 + Global Average Pooling (baseline)
2. OracleCT_Pillar_UnaryAttnPool:    Pillar-0 + Learned Full-Volume Attention (no organ mask)
3. OracleCT_Pillar_MaskedAttn:       Pillar-0 + 3D Organ-Masked Attention
4. OracleCT_Pillar_MaskedAttnScalar: Pillar-0 + 3D Masked Attention + Scalar Fusion

Pillar-0 is a 3D vision-language foundation model pretrained on abdominal CT.
Input: [B, 11, 384, 384, 384] — 11 HU windows (10 anatomical + minmax)
Outputs:
  pooled: [B, 1152] — L2-normalised global features (3 scales maxpool-cat)
  activ:  [B, 1152, 64, 64, 64] — spatial feature map (3 scales interp-cat)

Option B: apply 11 windows at load time, keep pretrained weights unchanged.
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as grad_ckpt

# ---------------------------------------------------------------------------
# Add pillar-finetune to sys.path so MultimodalAtlas can be imported
# (both live under oracle-ct/, two sibling directories)
# ---------------------------------------------------------------------------
import types

_PILLAR_FINETUNE = Path(__file__).resolve().parent.parent.parent / "pillar-finetune"
if _PILLAR_FINETUNE.exists() and str(_PILLAR_FINETUNE) not in sys.path:
    sys.path.insert(0, str(_PILLAR_FINETUNE))

# pillar/__init__.py eagerly imports pillar.datasets → lifelines → datetime.UTC
# which only exists in Python 3.11+. Stub it out before the import runs.
for _stub in [
    "pillar.datasets", "pillar.datasets.nlst", "pillar.datasets.image_loaders",
    "pillar.datasets.abstract_loader", "pillar.datasets.nlst_utils",
    "pillar.engines", "pillar.losses", "pillar.metrics", "pillar.augmentations",
]:
    sys.modules.setdefault(_stub, types.ModuleType(_stub))

from pillar.models.backbones.mmatlas import MultimodalAtlas  # noqa: E402

from ..configs.disease_config import (
    ORGAN_TO_CHANNEL,
    get_all_diseases,
    get_all_disease_configs,
)
from ..configs.disease_config_oracle_ct import DISEASE_CONFIGS as ORACLE_SCALAR_CONFIGS
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
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.num_diseases = num_diseases
        self.disease_names = disease_names or get_all_diseases()[:num_diseases]
        self.modality = modality
        self.freeze_backbone = freeze_backbone
        self.use_gradient_checkpointing = use_gradient_checkpointing and not freeze_backbone

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
            + (" [gradient checkpointing ON]" if self.use_gradient_checkpointing else "")
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        image = batch["image"]   # [B, 11, 384, 384, 384]

        # Pillar forward (with optional gradient checkpointing to save activation memory)
        pillar_batch = {"anatomy": [self.modality]}
        if self.use_gradient_checkpointing and self.training:
            def _backbone_fwd(x):
                return self.backbone(x, batch=pillar_batch)["pooled"]
            pooled = grad_ckpt.checkpoint(_backbone_fwd, image, use_reentrant=False)
        else:
            pooled = self.backbone(image, batch=pillar_batch)["pooled"]
        # pooled: [B, 1152]

        # Per-disease heads
        logits_list = []
        for disease in self.disease_names:
            logit = self.heads[disease](pooled)  # [B, 1]
            logits_list.append(logit)

        return torch.cat(logits_list, dim=1)  # [B, num_diseases]


# =============================================================================
# MODEL 2: PILLAR UNARY ATTENTION POOLING (full-volume, no organ mask)
# =============================================================================

class OracleCT_Pillar_UnaryAttnPool(nn.Module):
    """
    Pillar-0 + Learned Full-Volume 3D Attention Pooling (no organ masking).

    Intermediate between GAP (uniform pooling) and MaskedAttn (organ-masked).
    Each disease has its own learned Conv3d(1152, 1, 1) scorer attending over
    the full 3D spatial feature map [B, 1152, 64, 64, 64].

    No organ masks or parquet features required — simpler data pipeline than MaskedAttn.
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
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.num_diseases = num_diseases
        self.disease_names = disease_names or get_all_diseases()[:num_diseases]
        self.modality = modality
        self.freeze_backbone = freeze_backbone
        self.use_gradient_checkpointing = use_gradient_checkpointing and not freeze_backbone
        self.learn_tau = learn_tau
        self.fixed_tau = fixed_tau

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
        if learn_tau:
            self.temp_logit = nn.ParameterDict()

        for disease in self.disease_names:
            self.score_convs[disease] = nn.Conv3d(self.hidden_dim, 1, kernel_size=1)
            self.heads[disease] = nn.Linear(self.hidden_dim, 1)
            nn.init.normal_(self.heads[disease].weight, 0.0, 0.01)
            nn.init.constant_(self.heads[disease].bias, 0.0)
            if learn_tau:
                self.temp_logit[disease] = nn.Parameter(
                    torch.tensor(inv_sigmoid_temp(init_tau))
                )

        print(
            f"OracleCT_Pillar_UnaryAttnPool: {model_repo_id} → {self.hidden_dim}d activ "
            f"[64³] → full-volume attn → {num_diseases} diseases"
            + (" [gradient checkpointing ON]" if self.use_gradient_checkpointing else "")
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        image = batch["image"]   # [B, 11, 384, 384, 384]
        B = image.size(0)
        device = image.device

        # Pillar forward → spatial features
        pillar_batch = {"anatomy": [self.modality]}
        if self.use_gradient_checkpointing and self.training:
            def _backbone_fwd(x):
                return self.backbone(x, batch=pillar_batch)["activ"]
            activ = grad_ckpt.checkpoint(_backbone_fwd, image, use_reentrant=False)
        else:
            activ = self.backbone(image, batch=pillar_batch)["activ"]
        # activ: [B, 1152, 64, 64, 64]

        # Full-volume ones mask — no organ guidance
        ones_mask = torch.ones(B, 1, *activ.shape[2:], device=device)

        logits_list = []
        for disease in self.disease_names:
            tau = (
                0.2 + 1.8 * torch.sigmoid(self.temp_logit[disease])
                if self.learn_tau else self.fixed_tau
            )
            pooled = masked_attention_pool_3d(
                activ, ones_mask, self.score_convs[disease],
                tau=tau, bias_in=None, bias_out=None,
            )  # [B, 1152]
            logit = self.heads[disease](pooled)  # [B, 1]
            if torch.isnan(logit).any():
                logit = torch.where(torch.isnan(logit), torch.zeros_like(logit), logit)
            logits_list.append(logit)

        return torch.cat(logits_list, dim=1)  # [B, num_diseases]


# =============================================================================
# MODEL 3: PILLAR 3D MASKED ATTENTION
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
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.num_diseases = num_diseases
        self.disease_names = disease_names or get_all_diseases()[:num_diseases]
        self.modality = modality
        self.freeze_backbone = freeze_backbone
        self.use_gradient_checkpointing = use_gradient_checkpointing and not freeze_backbone
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
            + (" [gradient checkpointing ON]" if self.use_gradient_checkpointing else "")
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

        # Pillar forward → spatial features (with optional gradient checkpointing)
        pillar_batch = {"anatomy": [self.modality]}
        if self.use_gradient_checkpointing and self.training:
            def _backbone_fwd(x):
                return self.backbone(x, batch=pillar_batch)["activ"]
            activ = grad_ckpt.checkpoint(_backbone_fwd, image, use_reentrant=False)
        else:
            activ = self.backbone(image, batch=pillar_batch)["activ"]
        # activ: [B, 1152, 64, 64, 64]

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


# =============================================================================
# MODEL 4: PILLAR 3D MASKED ATTENTION + SCALAR FUSION
# =============================================================================

class OracleCT_Pillar_MaskedAttnScalar(nn.Module):
    """
    Pillar-0 + 3D Organ-Masked Attention + Scalar Feature Fusion.

    Extends OracleCT_Pillar_MaskedAttn with per-disease scalar fusion:
      visual [B, 1152] + scalars [B, S]
        → cat [B, 1152+S] → LayerNorm → Linear(→ scalar_hidden) → ReLU → Linear(→ 1)

    Scalars from the oracle-ct minimal parquet (mean_hu, to_body_ratio, touches_border
    per disease's primary organ). S=0 for global-strategy diseases → visual-only head.
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
        scalar_hidden: int = 256,
        feature_stats_path: Optional[str] = None,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.num_diseases = num_diseases
        self.disease_names = disease_names or get_all_diseases()[:num_diseases]
        self.modality = modality
        self.freeze_backbone = freeze_backbone
        self.use_gradient_checkpointing = use_gradient_checkpointing and not freeze_backbone
        self.learn_tau = learn_tau
        self.fixed_tau = fixed_tau
        self.use_mask_bias = use_mask_bias

        # Feature stats for z-score normalisation
        if not feature_stats_path:
            raise ValueError("feature_stats_path is required for OracleCT_Pillar_MaskedAttnScalar")
        if not Path(feature_stats_path).exists():
            raise FileNotFoundError(f"feature_stats not found: {feature_stats_path}")
        with open(feature_stats_path) as f:
            self._feature_stats: Dict[str, Dict[str, float]] = json.load(f)

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
        self.score_convs  = nn.ModuleDict()
        self.heads        = nn.ModuleDict()

        if use_mask_bias:
            self.inside_logit  = nn.ParameterDict()
            self.outside_logit = nn.ParameterDict()
        if learn_tau:
            self.temp_logit = nn.ParameterDict()

        for disease in self.disease_names:
            # 3D attention scorer (1×1×1 conv, same as MaskedAttn)
            self.score_convs[disease] = nn.Conv3d(self.hidden_dim, 1, kernel_size=1)

            if use_mask_bias:
                self.inside_logit[disease]  = nn.Parameter(torch.tensor(to_logit(init_inside)))
                self.outside_logit[disease] = nn.Parameter(torch.tensor(to_logit(init_outside)))
            if learn_tau:
                self.temp_logit[disease] = nn.Parameter(torch.tensor(inv_sigmoid_temp(init_tau)))

            # Fusion head: visual [1152] + scalars [S] → LayerNorm → MLP → 1
            config = ORACLE_SCALAR_CONFIGS.get(disease)
            num_scalars = len(config.scalar_features) if config else 0
            fusion_dim = self.hidden_dim + num_scalars
            self.heads[disease] = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, scalar_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(scalar_hidden, 1),
            )

        print(
            f"OracleCT_Pillar_MaskedAttnScalar: {model_repo_id} → {self.hidden_dim}d activ "
            f"[64³] → 3D masked attn + scalars → {scalar_hidden}d → {num_diseases} diseases"
            + (" [gradient checkpointing ON]" if self.use_gradient_checkpointing else "")
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _get_scalars(self, disease: str, features_row, device: torch.device) -> torch.Tensor:
        """Look up parquet features for this disease and z-score normalize."""
        config = ORACLE_SCALAR_CONFIGS.get(disease)
        feature_names = config.scalar_features if config else []
        if not feature_names:
            return torch.empty(0, device=device)  # global disease — no scalars
        if features_row is None:
            raise RuntimeError(
                f"OracleCT_Pillar_MaskedAttnScalar requires features_parquet but "
                f"features_row is None for disease '{disease}'."
            )
        values = []
        for name in feature_names:
            try:
                val = float(features_row.get(name, float("nan")))
            except (TypeError, ValueError):
                val = float("nan")
            if math.isnan(val):
                val = 0.0  # mean-impute (mean = 0 in z-score space)
            else:
                s = self._feature_stats.get(name, {})
                std = s.get("std", 1.0) or 1.0
                val = (val - s.get("mean", 0.0)) / std
            values.append(val)
        return torch.tensor(values, dtype=torch.float32, device=device)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        image         = batch["image"]           # [B, 11, 384, 384, 384]
        masks         = batch["masks"]           # [B, 20, D_mask, H_mask, W_mask]
        features_rows = batch.get("features_row", [None] * image.size(0))
        disease_rois  = batch.get("disease_rois", [{}] * image.size(0))
        meta          = batch.get("meta", [{}] * image.size(0))

        B      = image.size(0)
        device = image.device

        # Pillar forward → spatial features (with optional gradient checkpointing)
        pillar_batch = {"anatomy": [self.modality]}
        if self.use_gradient_checkpointing and self.training:
            def _backbone_fwd(x):
                return self.backbone(x, batch=pillar_batch)["activ"]
            activ = grad_ckpt.checkpoint(_backbone_fwd, image, use_reentrant=False)
        else:
            activ = self.backbone(image, batch=pillar_batch)["activ"]
        # activ: [B, 1152, 64, 64, 64]

        logits_list = []
        for disease in self.disease_names:
            attn_mask = get_attention_mask_for_disease(
                disease, masks, disease_rois, meta, device,
                allow_comparative=False,
            )

            bias_in  = self.inside_logit[disease]  if self.use_mask_bias else None
            bias_out = self.outside_logit[disease] if self.use_mask_bias else None
            tau = (
                0.2 + 1.8 * torch.sigmoid(self.temp_logit[disease])
                if self.learn_tau else self.fixed_tau
            )

            # 3D masked attention pooling → [B, 1152]
            visual = masked_attention_pool_3d(
                activ, attn_mask, self.score_convs[disease],
                tau=tau, bias_in=bias_in, bias_out=bias_out,
            )

            # Scalar features → [B, S] (S=0 for global diseases)
            scalars_list = [
                self._get_scalars(disease, features_rows[b], device)
                for b in range(B)
            ]
            if scalars_list[0].numel() > 0:
                scalars = torch.stack(scalars_list, dim=0)  # [B, S]
                fused   = torch.cat([visual, scalars], dim=1)  # [B, 1152+S]
            else:
                fused = visual  # [B, 1152] — no scalars for this disease

            logit = self.heads[disease](fused)  # [B, 1]
            if torch.isnan(logit).any():
                logit = torch.where(torch.isnan(logit), torch.zeros_like(logit), logit)
            logits_list.append(logit)

        return torch.cat(logits_list, dim=1)  # [B, num_diseases]
