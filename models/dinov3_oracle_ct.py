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

# janus/models/janus_model.py
"""
Janus Models for Neuro-Symbolic CT Disease Classification

Four model variants:
1. JanusGAP: DINOv3 + Global Average Pooling (baseline)
2. JanusMaskedAttn: DINOv3 + Organ-Masked Attention
3. JanusScalarFusion: DINOv3 + Masked Attention + Scalar Feature Fusion

Key improvements over previous implementation:
- Precise appendix ROI (from disease_rois) instead of colon mask
- Comparative attention for hepatic steatosis (liver + spleen)
- Body-size normalized volume features
- Liver-spleen HU difference for steatosis
"""

import math
from typing import Dict, List, Optional, Any, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from ..configs.disease_config import (
    ORGAN_TO_CHANNEL,
    get_all_disease_configs,
    get_all_diseases,
)
from ..datamodules.feature_bank import FeatureBank


# ImageNet normalization
IMN_MEAN = (0.485, 0.456, 0.406)
IMN_STD = (0.229, 0.224, 0.225)

# DINOv3 model IDs
DINOV3_HF_IDS = {
    "S": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "B": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "L": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


# =============================================================================
# TRI-SLICE HELPERS (convert 3D volume to 2.5D slices for 2D ViT)
# =============================================================================

def make_trislices(x_b1dhw: torch.Tensor, stride: int = 1) -> torch.Tensor:
    """
    Convert 3D volume to TRI-slices (3 adjacent axial slices as RGB).

    Args:
        x_b1dhw: [B, 1, D, H, W] volume
        stride: Sample every N slices

    Returns:
        [B, T, 3, H, W] TRI-slices
    """
    B, _, D, H, W = x_b1dhw.shape

    # Get slice centers with given stride (no adaptive stride modification)
    centers = list(range(1, max(2, D - 1), stride))
    if not centers:
        centers = [D // 2]

    T = len(centers)

    out = torch.empty(B, T, 3, H, W, device=x_b1dhw.device, dtype=x_b1dhw.dtype)

    for t, c in enumerate(centers):
        z0 = max(0, c - 1)
        z1 = c
        z2 = min(D - 1, c + 1)
        out[:, t, 0] = x_b1dhw[:, 0, z0]
        out[:, t, 1] = x_b1dhw[:, 0, z1]
        out[:, t, 2] = x_b1dhw[:, 0, z2]

    return out


def masks_3d_to_tri(mask_b1dhw: torch.Tensor, stride: int = 1) -> torch.Tensor:
    """
    Convert 3D mask to TRI-slice format.

    Uses same sampling as make_trislices to ensure alignment.

    Args:
        mask_b1dhw: [B, 1, D, H, W] mask
        stride: Sample every N slices

    Returns:
        [B, T, H, W] mask at TRI-slice positions
    """
    B, _, D, H, W = mask_b1dhw.shape

    # Get slice centers with given stride (no adaptive stride modification)
    centers = list(range(1, max(2, D - 1), stride))
    if not centers:
        centers = [D // 2]

    T = len(centers)

    mask = (mask_b1dhw > 0.5).float()
    out = torch.empty(B, T, H, W, device=mask.device, dtype=mask.dtype)

    for t, c in enumerate(centers):
        out[:, t] = mask[:, 0, c]

    return out


# =============================================================================
# ATTENTION MECHANISMS
# =============================================================================

def to_logit(p: float) -> float:
    """Convert probability to logit (inverse sigmoid)."""
    p = max(1e-4, min(1 - 1e-4, float(p)))
    return math.log(p / (1.0 - p))


def inv_sigmoid_temp(target_tau: float, min_tau: float = 0.2, max_tau: float = 2.0) -> float:
    """
    Inverse of the temperature mapping: tau = min_tau + (max_tau - min_tau) * sigmoid(x)

    Solves for x given target_tau:
        target_tau = min_tau + (max_tau - min_tau) * sigmoid(x)
        sigmoid(x) = (target_tau - min_tau) / (max_tau - min_tau)
        x = logit(sigmoid(x))

    Args:
        target_tau: Desired temperature value (e.g., 0.7)
        min_tau: Minimum temperature after sigmoid (default 0.2)
        max_tau: Maximum temperature after sigmoid (default 2.0)

    Returns:
        The logit value that will produce target_tau after the transform
    """
    # Normalize to [0, 1]
    normalized = (target_tau - min_tau) / (max_tau - min_tau)
    # Clamp to valid sigmoid range
    normalized = max(1e-4, min(1 - 1e-4, normalized))
    # Apply inverse sigmoid
    return math.log(normalized / (1.0 - normalized))


def masked_attention_pool(
    tokens: torch.Tensor,       # [B, T, N, D]
    mask_bt_hw: torch.Tensor,   # [B, T, H, W]
    score_mlp: nn.Module,       # Linear(D -> 1)
    tau: Union[float, torch.Tensor] = 1.0,
    bias_in: Optional[torch.Tensor] = None,   # Learnable prior for inside mask
    bias_out: Optional[torch.Tensor] = None,  # Learnable prior for outside mask
) -> torch.Tensor:
    """
    Content-aware attention restricted to mask with learnable priors.

    Args:
        tokens: [B, T, N, D] ViT tokens
        mask_bt_hw: [B, T, H, W] binary mask
        score_mlp: MLP to compute attention scores from tokens
        tau: Temperature (scalar or tensor)
        bias_in: Learnable logit bias for inside mask (additive prior)
        bias_out: Learnable logit bias for outside mask (additive prior)

    Returns:
        [B, D] pooled features
    """
    B, T, N, D = tokens.shape
    gh = int(math.sqrt(N))

    # Resize mask to token grid
    mask = mask_bt_hw
    if mask.shape[-2:] != (gh, gh):
        mask = F.interpolate(
            mask.view(B * T, 1, *mask.shape[-2:]),
            size=(gh, gh), mode="nearest"
        ).view(B, T, gh, gh)

    # Flatten mask - use actual size after interpolation
    mask_flat = (mask > 0.5).float().view(B, T, -1)  # [B, T, gh*gh]

    # Handle case where N != gh*gh (e.g., N=200 but gh=14 gives 196)
    if mask_flat.shape[-1] != N:
        # Pad or interpolate to match token count
        if mask_flat.shape[-1] < N:
            # Pad with zeros
            padding = N - mask_flat.shape[-1]
            mask_flat = F.pad(mask_flat, (0, padding), value=0.0)
        else:
            # Truncate
            mask_flat = mask_flat[..., :N]

    # Compute attention scores
    tok_flat = tokens.reshape(B * T, N, D)
    logits = score_mlp(tok_flat).view(B, T, N)

    # LEARNABLE PRIORS: Add bias to inside vs outside regions
    if bias_in is not None and bias_out is not None:
        # bias_in and bias_out are learnable parameters (logits)
        # Apply to inside (mask=1) and outside (mask=0) regions
        logits = logits + bias_in.view(1, 1, 1) * mask_flat + bias_out.view(1, 1, 1) * (1.0 - mask_flat)

    # Temperature scaling
    if isinstance(tau, torch.Tensor):
        logits = logits / tau.view(1, 1, 1)
    else:
        logits = logits / tau

    # Masked softmax
    very_neg = torch.finfo(logits.dtype).min / 4
    masked_logits = torch.where(mask_flat > 0, logits, torch.full_like(logits, very_neg))

    # Stable softmax
    lmax = masked_logits.max(dim=-1, keepdim=True).values
    exp_w = torch.exp((masked_logits - lmax).clamp(-60, 0)) * mask_flat
    weights = exp_w / exp_w.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # Handle empty masks (improved: check before and after normalization)
    empty = mask_flat.sum(dim=-1, keepdim=True) < 0.5
    if empty.any():
        uniform = torch.ones_like(weights) / N
        weights = torch.where(empty, uniform, weights)

    # Additional safety: replace any NaN weights with uniform distribution
    if torch.isnan(weights).any():
        uniform = torch.ones_like(weights) / N
        weights = torch.where(torch.isnan(weights), uniform, weights)

    # Pool: weighted sum over tokens, mean over TRI-slices
    pooled = (tokens * weights.unsqueeze(-1)).sum(dim=2).mean(dim=1)  # [B, D]

    # Final safety: replace NaN pooled features with zeros (should never happen, but defensive)
    if torch.isnan(pooled).any():
        pooled = torch.where(torch.isnan(pooled), torch.zeros_like(pooled), pooled)

    return pooled


def create_roi_mask(
    roi_info: Dict[str, Any],
    shape: Tuple[int, int, int],
    spacing_mm: Tuple[float, float, float],
    device: torch.device,
) -> torch.Tensor:
    """
    Create a 3D box mask from ROI info.

    IMPORTANT: After dataset permutation, coordinates are in [Z, Y, X] order!

    Args:
        roi_info: {"center_vox": [z, y, x], "box_mm": [sz, sy, sx]}  (permuted from original!)
        shape: (D, H, W) tensor shape
        spacing_mm: Voxel spacing [sz, sy, sx] (permuted to match tensor dims)

    Returns:
        [1, 1, D, H, W] binary mask
    """
    D, H, W = shape
    center = roi_info["center_vox"]  # [z, y, x] after permutation
    box_mm = roi_info["box_mm"]      # [sz, sy, sx] after permutation

    # Convert box from mm to voxels
    box_vox = [
        box_mm[0] / spacing_mm[0],  # box_z / spacing_z
        box_mm[1] / spacing_mm[1],  # box_y / spacing_y
        box_mm[2] / spacing_mm[2],  # box_x / spacing_x
    ]

    # Compute bounds - center is [z, y, x], shape is [D, H, W]
    # center[0] = z → D dimension
    # center[1] = y → H dimension
    # center[2] = x → W dimension
    z0 = int(max(0, center[0] - box_vox[0] / 2))
    z1 = int(min(D, center[0] + box_vox[0] / 2))
    y0 = int(max(0, center[1] - box_vox[1] / 2))
    y1 = int(min(H, center[1] + box_vox[1] / 2))
    x0 = int(max(0, center[2] - box_vox[2] / 2))
    x1 = int(min(W, center[2] + box_vox[2] / 2))

    mask = torch.zeros(1, 1, D, H, W, device=device)
    mask[0, 0, z0:z1, y0:y1, x0:x1] = 1.0

    return mask


def dilate_mask_adaptive(
    mask: torch.Tensor,
    dilation_mm: float,
    spacing_mm: Tuple[float, float, float],
) -> torch.Tensor:
    """
    Resolution-adaptive morphological dilation using max pooling.

    Converts physical dilation (mm) to voxel kernel size based on image resolution.
    This ensures consistent physical dilation across different resolutions.

    Args:
        mask: [B, 1, D, H, W] binary mask
        dilation_mm: Physical dilation in millimeters (e.g., 3.5mm)
        spacing_mm: Voxel spacing in mm (e.g., [1.5, 1.5, 3.0])
                    Order matches tensor dimensions: spacing[i] for dim i

    Returns:
        [B, 1, D, H, W] dilated mask

    Examples:
        - 1.5mm spacing, 3.0mm dilation → kernel_size = 5 (2 voxels each side)
        - 1.0mm spacing, 3.0mm dilation → kernel_size = 7 (3 voxels each side)
        - 2.0mm spacing, 3.0mm dilation → kernel_size = 3 (1 voxel each side)
    """
    if dilation_mm == 0:
        return mask

    # Convert to float if needed (max_pool3d doesn't support uint8/Byte)
    original_dtype = mask.dtype
    if mask.dtype != torch.float32:
        mask = mask.float()

    # Compute kernel size per axis (anisotropic if spacing is anisotropic)
    kernel_sizes = []
    paddings = []

    for axis_spacing in spacing_mm:
        # Voxels needed for desired physical dilation
        voxels = round(dilation_mm / axis_spacing)
        # Kernel size must be odd for centered dilation
        kernel_size = 2 * voxels + 1
        kernel_sizes.append(kernel_size)
        paddings.append(kernel_size // 2)

    # 3D max pooling for morphological dilation
    # Max pooling propagates "1s" to neighbors → morphological dilation
    dilated = F.max_pool3d(
        mask,
        kernel_size=tuple(kernel_sizes),
        stride=1,
        padding=tuple(paddings)
    )

    # Convert back to original dtype if needed
    if original_dtype != torch.float32:
        dilated = dilated.to(original_dtype)

    return dilated


def get_attention_mask_for_disease(
    disease: str,
    masks: torch.Tensor,                 # [B, C, D, H, W]
    disease_rois: List[Dict],
    meta: List[Dict],
    device: torch.device,
    *,
    allow_comparative: bool = True,      # MaskedAttn=True, ScalarFusion/Gated=False
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Unified mask builder for all models.

    Returns:
      - [B,1,D,H,W] for single/union/roi/global
      - list([B,1,D,H,W], ...) for comparative (if allow_comparative=True)
    """
    B = masks.size(0)
    disease_configs = get_all_disease_configs()
    config = disease_configs.get(disease)

    if config is None:
        return torch.ones(B, 1, *masks.shape[2:], device=device)

    # NEW: Full CT / global attention
    if config.attention_strategy in ("global", "full"):
        return torch.ones(B, 1, *masks.shape[2:], device=device)

    # ROI-based
    if config.attention_strategy == "roi" and config.roi_key:
        roi_masks = []
        for b in range(B):
            if config.roi_key in disease_rois[b]:
                roi_info = disease_rois[b][config.roi_key]
                spacing = meta[b].get("spacing_final_mm", [1.5, 1.5, 1.5])
                roi_mask = create_roi_mask(roi_info, masks.shape[2:], tuple(spacing), device)
                roi_masks.append(roi_mask)
            else:
                channels = [ORGAN_TO_CHANNEL[o] for o in config.attention_organs if o in ORGAN_TO_CHANNEL]
                if channels:
                    fallback = masks[b:b+1, channels].sum(dim=1, keepdim=True).clamp(0, 1)
                else:
                    fallback = torch.ones(1, 1, *masks.shape[2:], device=device)
                roi_masks.append(fallback)

        combined = torch.cat(roi_masks, dim=0)
        if config.dilation_mm > 0:
            spacing = meta[0].get("spacing_final_mm", [1.5, 1.5, 1.5])
            combined = dilate_mask_adaptive(combined, config.dilation_mm, tuple(spacing))
        return combined

    # Organ channels
    channels = [ORGAN_TO_CHANNEL[o] for o in config.attention_organs if o in ORGAN_TO_CHANNEL]
    if not channels:
        return torch.ones(B, 1, *masks.shape[2:], device=device)

    # Comparative (only if supported by the model)
    if config.attention_strategy == "comparative":
        if not allow_comparative:
            # fallback: union
            combined = masks[:, channels].float().sum(dim=1, keepdim=True).clamp(0, 1)
            if config.dilation_mm > 0:
                spacing = meta[0].get("spacing_final_mm", [1.5, 1.5, 1.5])
                combined = dilate_mask_adaptive(combined, config.dilation_mm, tuple(spacing))
            return combined

        spacing = meta[0].get("spacing_final_mm", [1.5, 1.5, 1.5])
        organ_masks = []
        for ch in channels:
            om = masks[:, ch:ch+1]
            if config.dilation_mm > 0:
                om = dilate_mask_adaptive(om, config.dilation_mm, tuple(spacing))
            organ_masks.append(om)
        return organ_masks

    # Single/Union
    combined = masks[:, channels].float().sum(dim=1, keepdim=True).clamp(0, 1)
    if config.dilation_mm > 0:
        spacing = meta[0].get("spacing_final_mm", [1.5, 1.5, 1.5])
        combined = dilate_mask_adaptive(combined, config.dilation_mm, tuple(spacing))
    return combined




# =============================================================================
# MODEL 1: GLOBAL AVERAGE POOLING (BASELINE)
# =============================================================================

class JanusGAP(nn.Module):
    """
    Baseline: DINOv3 + Global Average Pooling

    No organ-specific attention, no scalar features.
    Tests pure neural performance.
    """

    def __init__(
        self,
        num_diseases: int = 30,
        variant: str = "S",  # "S", "B", or "L"
        image_size: int = 224,
        tri_stride: int = 1,
        freeze_backbone: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.num_diseases = num_diseases
        self.image_size = image_size
        self.tri_stride = tri_stride
        self.variant = variant.upper()

        # Get DINOv3 model ID from variant
        if self.variant not in DINOV3_HF_IDS:
            raise ValueError(f"Invalid variant '{variant}'. Must be one of: {list(DINOV3_HF_IDS.keys())}")

        backbone_id = DINOV3_HF_IDS[self.variant]

        # Load backbone
        self.backbone = AutoModel.from_pretrained(backbone_id, trust_remote_code=True)
        self.hidden_dim = self.backbone.config.hidden_size

        # Store freeze setting
        self.freeze_backbone = freeze_backbone

        # Freeze backbone if requested
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        else:
            for p in self.backbone.parameters():
                p.requires_grad = True
            self.backbone.train()

            # Enable gradient checkpointing to save memory (only when training backbone)
            if use_gradient_checkpointing:
                if hasattr(self.backbone, 'gradient_checkpointing_enable'):
                    self.backbone.gradient_checkpointing_enable()
                    print(f"✓ Gradient checkpointing enabled for {self.variant} backbone")

        # Classification head
        self.head = nn.Linear(self.hidden_dim, num_diseases)

        # Initialize head with small weights for fast convergence
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.head.bias, 0.0)

        # Normalization buffers
        self.register_buffer("_mean", torch.tensor(IMN_MEAN).view(1, 1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(IMN_STD).view(1, 1, 3, 1, 1))
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / self._std

    def train(self, mode: bool = True):
        """Override train to keep backbone frozen if requested."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        image = batch["image"]  # [B, 1, D, H, W]
        B, _, D, H, W = image.shape
        
        # Convert to TRI-slices
        tri = make_trislices(image, self.tri_stride)  # [B, T, 3, H, W]
        T = tri.size(1)
        
        # Resize and normalize
        tri = F.interpolate(
            tri.view(B * T, 3, H, W),
            size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False
        ).view(B, T, 3, self.image_size, self.image_size)
        tri = self._normalize(tri)
        
        # Forward through backbone
        tri_flat = tri.view(B * T, 3, self.image_size, self.image_size)
        out = self.backbone(pixel_values=tri_flat)
        tokens = out.last_hidden_state[:, 1:, :]  # Remove CLS token

        # Remove register tokens (DINOv3 has 4 register tokens at the end)
        num_register_tokens = getattr(self.backbone.config, "num_register_tokens", 0)
        if num_register_tokens > 0 and tokens.size(1) > num_register_tokens:
            tokens = tokens[:, :-num_register_tokens, :]  # [B*T, N, D]

        # Global average pooling
        pooled = tokens.mean(dim=1)  # [B*T, D]
        pooled = pooled.view(B, T, -1).mean(dim=1)  # [B, D]
        
        # Classify
        logits = self.head(pooled)
        
        return logits


# =============================================================================
# MODEL 2: MASKED ATTENTION
# =============================================================================

class JanusMaskedAttn(nn.Module):
    """
    DINOv3 + Organ-Masked Attention

    Per-disease attention masks:
    - Single organ: Attend to one organ (e.g., liver for hepatomegaly)
    - Union: Attend to multiple organs (e.g., liver+spleen+kidneys for ascites)
    - Comparative: Separate attention streams (e.g., liver vs spleen for steatosis)
    - ROI: Precise localization box (e.g., appendix ROI for appendicitis)
    """

    def __init__(
        self,
        num_diseases: int = 30,
        disease_names: List[str] = None,
        variant: str = "S",  # "S", "B", or "L"
        image_size: int = 224,
        tri_stride: int = 1,
        freeze_backbone: bool = True,
        learn_tau: bool = True,
        init_tau: float = 0.7,
        fixed_tau: float = 1.0,
        use_mask_bias: bool = True,
        init_inside: float = 0.8,
        init_outside: float = 0.2,
        use_gradient_checkpointing: bool = False,
        allow_comparative: bool = False,
    ):
        super().__init__()

        self.num_diseases = num_diseases
        all_diseases = get_all_diseases()
        self.disease_names = disease_names or all_diseases[:num_diseases]
        self.image_size = image_size
        self.tri_stride = tri_stride
        self.learn_tau = learn_tau
        self.fixed_tau = fixed_tau
        self.use_mask_bias = use_mask_bias
        self.variant = variant.upper()
        self.allow_comparative = allow_comparative

        # Get DINOv3 model ID from variant
        if self.variant not in DINOV3_HF_IDS:
            raise ValueError(f"Invalid variant '{variant}'. Must be one of: {list(DINOV3_HF_IDS.keys())}")

        backbone_id = DINOV3_HF_IDS[self.variant]

        # Load backbone
        self.backbone = AutoModel.from_pretrained(backbone_id, trust_remote_code=True)
        self.hidden_dim = self.backbone.config.hidden_size

        # Store freeze setting
        self.freeze_backbone = freeze_backbone

        # Freeze backbone if requested
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        else:
            for p in self.backbone.parameters():
                p.requires_grad = True
            self.backbone.train()

            # Enable gradient checkpointing to save memory (only when training backbone)
            if use_gradient_checkpointing:
                if hasattr(self.backbone, 'gradient_checkpointing_enable'):
                    self.backbone.gradient_checkpointing_enable()
                    print(f"✓ Gradient checkpointing enabled for {self.variant} backbone")

        # Per-disease attention score MLPs, heads, and learnable priors
        self.score_mlps = nn.ModuleDict()
        self.heads = nn.ModuleDict()

        # Learnable priors (bias and temperature) - CRITICAL for masking performance!
        if self.use_mask_bias:
            self.inside_logit = nn.ParameterDict()   # Bias for inside mask regions
            self.outside_logit = nn.ParameterDict()  # Bias for outside mask regions

        if self.learn_tau:
            self.temp_logit = nn.ParameterDict()  # Learnable temperature per disease

        for disease in self.disease_names:
            # Attention score MLP: simple Linear for interpretable attention weights
            self.score_mlps[disease] = nn.Linear(self.hidden_dim, 1)

            # Learnable priors for masked attention
            if self.use_mask_bias:
                # Initialize inside/outside logits from config probabilities
                self.inside_logit[disease] = nn.Parameter(torch.tensor(to_logit(init_inside)))
                self.outside_logit[disease] = nn.Parameter(torch.tensor(to_logit(init_outside)))

            if self.learn_tau:
                # BUGFIX: Use inverse sigmoid to correctly initialize temperature
                # Old: to_logit(0.7) produces tau≈1.46 (WRONG!)
                # New: inv_sigmoid_temp(0.7) produces tau=0.7 (CORRECT!)
                self.temp_logit[disease] = nn.Parameter(torch.tensor(inv_sigmoid_temp(init_tau)))

            # Check if comparative strategy (concatenates multiple organ features)
            # Only use comparative if allow_comparative=True
            disease_configs = get_all_disease_configs()
            config = disease_configs.get(disease)
            if self.allow_comparative and config and config.attention_strategy == "comparative":
                # Comparative: concatenate features from N organs → N * hidden_dim input
                num_organs = len(config.attention_organs)
                head_input_dim = self.hidden_dim * num_organs
            else:
                # Single/Union/ROI: single pooled feature → hidden_dim input
                head_input_dim = self.hidden_dim

            # Classification head: simple Linear (parameter efficient, matches TriageNet)
            # Reduces params from 9M → ~23K (30 diseases × 768 params)
            self.heads[disease] = nn.Linear(head_input_dim, 1)

        # Normalization buffers
        self.register_buffer("_mean", torch.tensor(IMN_MEAN).view(1, 1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(IMN_STD).view(1, 1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / self._std

    def train(self, mode: bool = True):
        """Override train to keep backbone frozen if requested."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        image = batch["image"]  # [B, 1, X, Y, Z] in RAS orientation
        masks = batch["masks"]  # [B, 20, X, Y, Z] in RAS orientation
        disease_rois = batch.get("disease_rois", [{}] * image.size(0))
        meta = batch.get("meta", [{}] * image.size(0))
        
        B, _, D, H, W = image.shape
        device = image.device
        
        # Convert to TRI-slices
        tri = make_trislices(image, self.tri_stride)
        T = tri.size(1)
        
        # Resize and normalize
        tri = F.interpolate(
            tri.view(B * T, 3, H, W),
            size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False
        ).view(B, T, 3, self.image_size, self.image_size)
        tri = self._normalize(tri)

        # Forward through backbone
        tri_flat = tri.view(B * T, 3, self.image_size, self.image_size)
        out = self.backbone(pixel_values=tri_flat)
        tokens = out.last_hidden_state[:, 1:, :]  # Remove CLS token

        # Remove register tokens (DINOv3 has 4 register tokens at the end)
        num_register_tokens = getattr(self.backbone.config, "num_register_tokens", 0)
        if num_register_tokens > 0 and tokens.size(1) > num_register_tokens:
            tokens = tokens[:, :-num_register_tokens, :]  # [B*T, N, D]

        N, D_tok = tokens.size(1), tokens.size(2)
        tokens = tokens.view(B, T, N, D_tok)

        # Per-disease attention
        logits_list = []

        for disease in self.disease_names:
            attn_mask = get_attention_mask_for_disease(
                disease, masks, disease_rois, meta, device, allow_comparative=self.allow_comparative
            )
            # Get learnable priors for this disease
            bias_in = self.inside_logit[disease] if self.use_mask_bias else None
            bias_out = self.outside_logit[disease] if self.use_mask_bias else None

            # Get temperature for this disease
            if self.learn_tau:
                # Map temp_logit to [0.2, 2.0] range via sigmoid: tau = 0.2 + 1.8 * sigmoid(logit)
                tau = 0.2 + 1.8 * torch.sigmoid(self.temp_logit[disease])
            else:
                tau = self.fixed_tau

            # Check if comparative (list of masks) or single mask
            if isinstance(attn_mask, list):
                # COMPARATIVE: Pool each organ separately and concatenate
                pooled_organs = []
                for organ_mask in attn_mask:
                    # Convert to TRI-slice and resize
                    mask_tri = masks_3d_to_tri(organ_mask, self.tri_stride)
                    mask_tri = F.interpolate(
                        mask_tri.view(B * T, 1, H, W),
                        size=(self.image_size, self.image_size), mode="nearest"
                    ).view(B, T, self.image_size, self.image_size)

                    # Masked attention pooling for this organ WITH learnable priors
                    organ_pooled = masked_attention_pool(
                        tokens, mask_tri, self.score_mlps[disease],
                        tau=tau, bias_in=bias_in, bias_out=bias_out
                    )
                    pooled_organs.append(organ_pooled)

                # Concatenate organ features: [liver_vec, spleen_vec] → [B, 2*hidden_dim]
                pooled = torch.cat(pooled_organs, dim=1)
            else:
                # SINGLE/UNION/ROI: Pool once with merged mask
                mask_tri = masks_3d_to_tri(attn_mask, self.tri_stride)
                mask_tri = F.interpolate(
                    mask_tri.view(B * T, 1, H, W),
                    size=(self.image_size, self.image_size), mode="nearest"
                ).view(B, T, self.image_size, self.image_size)

                # Masked attention pooling WITH learnable priors
                pooled = masked_attention_pool(
                    tokens, mask_tri, self.score_mlps[disease],
                    tau=tau, bias_in=bias_in, bias_out=bias_out
                )

            # Classification head (handles both single and concatenated features)
            logit = self.heads[disease](pooled)  # [B, 1]

            # Safety: replace NaN logits with zeros (should never happen after pooling fix)
            if torch.isnan(logit).any():
                logit = torch.where(torch.isnan(logit), torch.zeros_like(logit), logit)

            logits_list.append(logit)

        logits = torch.cat(logits_list, dim=1)  # [B, num_diseases]

        return logits


# =============================================================================
# MODEL 3: MASKED ATTENTION + SCALAR FUSION
# =============================================================================

class JanusScalarFusion(nn.Module):
    """
    DINOv3 + Masked Attention + Scalar Feature Fusion

    Fuses visual features with scalar radiomics features:
    - Volume ratios (body-size normalized)
    - HU comparisons (liver vs spleen)
    - Diameter measurements
    - Derived features (SBO ratio, etc.)

    Architecture: Separate Visual and Scalar Projectors
    - Visual features (768-3072 dim) → Visual Projector → visual_proj_dim
    - Scalar features (10-20 dim) → Scalar Projector → scalar_proj_dim
    - Concatenate balanced projections → Fusion MLP → Logit
    - This ensures equal gradient flow and representational power for both modalities
    """

    def __init__(
        self,
        num_diseases: int = 30,
        disease_names: List[str] = None,
        variant: str = "S",  # "S", "B", or "L"
        image_size: int = 224,
        tri_stride: int = 1,
        freeze_backbone: bool = True,
        learn_tau: bool = True,
        init_tau: float = 0.7,
        fixed_tau: float = 1.0,
        use_mask_bias: bool = True,
        init_inside: float = 0.8,
        init_outside: float = 0.2,
        fusion_hidden: int = 256,
        visual_proj_dim: int = 256,  # Dimension for visual projection
        scalar_proj_dim: int = 256,  # Dimension for scalar projection
        feature_stats_path: Optional[str] = None,
        use_gradient_checkpointing: bool = False,
        use_modality_gating: bool = False,  # Deprecated - use separate projectors instead
    ):
        super().__init__()

        self.num_diseases = num_diseases
        all_diseases = get_all_diseases()
        self.disease_names = disease_names or all_diseases[:num_diseases]
        self.image_size = image_size
        self.tri_stride = tri_stride
        self.learn_tau = learn_tau
        self.fixed_tau = fixed_tau
        self.use_mask_bias = use_mask_bias
        self.variant = variant.upper()
        self.use_modality_gating = use_modality_gating

        # Feature bank for scalar features
        self.feature_bank = FeatureBank(
            stats_path=feature_stats_path,
            normalize="zscore",
        )

        # Get DINOv3 model ID from variant
        if self.variant not in DINOV3_HF_IDS:
            raise ValueError(f"Invalid variant '{variant}'. Must be one of: {list(DINOV3_HF_IDS.keys())}")

        backbone_id = DINOV3_HF_IDS[self.variant]

        # Load backbone
        self.backbone = AutoModel.from_pretrained(backbone_id, trust_remote_code=True)
        self.hidden_dim = self.backbone.config.hidden_size

        # Store freeze setting
        self.freeze_backbone = freeze_backbone

        # Freeze backbone if requested
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        else:
            for p in self.backbone.parameters():
                p.requires_grad = True
            self.backbone.train()

            # Enable gradient checkpointing to save memory (only when training backbone)
            if use_gradient_checkpointing:
                if hasattr(self.backbone, 'gradient_checkpointing_enable'):
                    self.backbone.gradient_checkpointing_enable()
                    print(f"✓ Gradient checkpointing enabled for {self.variant} backbone")

        # Per-disease components: Attention MLPs + Separate Projectors + Fusion
        self.score_mlps = nn.ModuleDict()
        self.visual_projectors = nn.ModuleDict()
        self.scalar_projectors = nn.ModuleDict()
        self.fusion_heads = nn.ModuleDict()

        # Learnable priors (bias and temperature) - CRITICAL for masking performance!
        if self.use_mask_bias:
            self.inside_logit = nn.ParameterDict()   # Bias for inside mask regions
            self.outside_logit = nn.ParameterDict()  # Bias for outside mask regions

        if self.learn_tau:
            self.temp_logit = nn.ParameterDict()  # Learnable temperature per disease

        for disease in self.disease_names:
            # Attention score MLP (unchanged)
            self.score_mlps[disease] = nn.Linear(self.hidden_dim, 1)

            # Learnable priors for masked attention
            if self.use_mask_bias:
                # Initialize inside/outside logits from config probabilities
                self.inside_logit[disease] = nn.Parameter(torch.tensor(to_logit(init_inside)))
                self.outside_logit[disease] = nn.Parameter(torch.tensor(to_logit(init_outside)))

            if self.learn_tau:
                # BUGFIX: Use inverse sigmoid to correctly initialize temperature
                # Old: to_logit(0.7) produces tau≈1.46 (WRONG!)
                # New: inv_sigmoid_temp(0.7) produces tau=0.7 (CORRECT!)
                self.temp_logit[disease] = nn.Parameter(torch.tensor(inv_sigmoid_temp(init_tau)))

            disease_configs = get_all_disease_configs()
            config = disease_configs.get(disease)

            # Count scalar features: Use EXACT same features as LR baseline (no presence indicators)
            num_base_features = len(config.scalar_features) + len(config.derived_features) if config else 0
            num_scalars = num_base_features  # IDENTICAL to LR: no presence indicators

            # Visual feature dimension (always hidden_dim now - no more comparative)
            visual_dim = self.hidden_dim

            # Visual Projector: visual_dim -> visual_proj_dim
            self.visual_projectors[disease] = nn.Sequential(
                nn.Linear(visual_dim, visual_proj_dim),
                nn.ReLU(inplace=True),
            )

            # BUGFIX: Only register scalar_projector if scalars exist (ModuleDict cannot store None)
            # Old: self.scalar_projectors[disease] = None (CRASHES!)
            # New: Only register when num_scalars > 0 (CORRECT!)
            if num_scalars > 0:
                # Two-layer projector for better expressiveness
                scalar_hidden = max(64, num_scalars * 4)  # Adaptive hidden size
                self.scalar_projectors[disease] = nn.Sequential(
                    nn.Linear(num_scalars, scalar_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(scalar_hidden, scalar_proj_dim),
                    nn.ReLU(inplace=True),
                )
            # else: Do nothing - don't register at all (check with "disease in self.scalar_projectors")

            # Fusion Head: (visual_proj_dim + scalar_proj_dim) -> 1
            # Balanced input: 256 + 256 = 512 (equal representation!)
            fusion_input_dim = visual_proj_dim + (scalar_proj_dim if num_scalars > 0 else 0)

            self.fusion_heads[disease] = nn.Sequential(
                nn.LayerNorm(fusion_input_dim),
                nn.Linear(fusion_input_dim, fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(fusion_hidden, 1),
            )

        print(f"✓ Separate projectors: Visual ({visual_proj_dim}d) + Scalar ({scalar_proj_dim}d) → Fusion ({fusion_hidden}d → 1)")

        # Deprecated: Old modality gating (kept for backward compatibility)
        if self.use_modality_gating:
            self.visual_gate_logit = nn.Parameter(torch.zeros(num_diseases))
            self.scalar_gate_logit = nn.Parameter(torch.zeros(num_diseases))
            print(f"⚠ WARNING: use_modality_gating=True is deprecated. Separate projectors are now used instead.")

        # Normalization buffers
        self.register_buffer("_mean", torch.tensor(IMN_MEAN).view(1, 1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(IMN_STD).view(1, 1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / self._std

    def train(self, mode: bool = True):
        """Override train to keep backbone frozen if requested."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    
    def forward(self, batch: Dict[str, Any], **kwargs) -> torch.Tensor:
        image = batch["image"]
        masks = batch["masks"]
        features_rows = batch.get("features_row", [None] * image.size(0))  # List[pd.Series] from parquet
        disease_rois = batch.get("disease_rois", [{}] * image.size(0))
        meta = batch.get("meta", [{}] * image.size(0))
        
        B, _, D, H, W = image.shape
        device = image.device
        
        # Convert to TRI-slices
        tri = make_trislices(image, self.tri_stride)
        T = tri.size(1)
        
        tri = F.interpolate(
            tri.view(B * T, 3, H, W),
            size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False
        ).view(B, T, 3, self.image_size, self.image_size)
        tri = self._normalize(tri)

        # Forward through backbone
        tri_flat = tri.view(B * T, 3, self.image_size, self.image_size)
        out = self.backbone(pixel_values=tri_flat)
        tokens = out.last_hidden_state[:, 1:, :]  # Remove CLS token

        # Remove register tokens (DINOv3 has 4 register tokens at the end)
        num_register_tokens = getattr(self.backbone.config, "num_register_tokens", 0)
        if num_register_tokens > 0 and tokens.size(1) > num_register_tokens:
            tokens = tokens[:, :-num_register_tokens, :]  # [B*T, N, D]

        N, D_tok = tokens.size(1), tokens.size(2)
        tokens = tokens.view(B, T, N, D_tok)

        # Pre-compute derived features ONCE per batch (not per disease!)
        derived_features_batch = []
        for b in range(B):
            derived = self.feature_bank.compute_derived_features(meta[b], features_row=features_rows[b])
            derived_features_batch.append(derived)

        # Per-disease attention + scalar fusion
        logits_list = []

        for disease in self.disease_names:
            # ScalarFusion pools a single visual stream per disease; comparative masks (list of masks)
            # are only supported by JanusMaskedAttn. For comparative-config diseases, fall back
            # to a union mask here.
            attn_mask = get_attention_mask_for_disease(
                disease, masks, disease_rois, meta, device, allow_comparative=False
            )
            # Get learnable priors for this disease
            bias_in = self.inside_logit[disease] if self.use_mask_bias else None
            bias_out = self.outside_logit[disease] if self.use_mask_bias else None

            # Get temperature for this disease
            if self.learn_tau:
                # Map temp_logit to [0.2, 2.0] range via sigmoid: tau = 0.2 + 1.8 * sigmoid(logit)
                tau = 0.2 + 1.8 * torch.sigmoid(self.temp_logit[disease])
            else:
                tau = self.fixed_tau

            # Visual pooling (union mask)
            mask_tri = masks_3d_to_tri(attn_mask, self.tri_stride)
            mask_tri = F.interpolate(
                mask_tri.view(B * T, 1, H, W),
                size=(self.image_size, self.image_size), mode="nearest"
            ).view(B, T, self.image_size, self.image_size)

            # Masked attention pooling WITH learnable priors
            visual_features = masked_attention_pool(
                tokens, mask_tri, self.score_mlps[disease],
                tau=tau, bias_in=bias_in, bias_out=bias_out
            )

            # Get scalar features for this disease (using cached derived features)
            scalar_features_list = []
            for b in range(B):
                scalars, _ = self.feature_bank.get_features_for_disease(
                    disease, meta[b], features_row=features_rows[b], normalize=True,
                    cached_derived=derived_features_batch[b]  # Pass cached!
                )
                scalar_features_list.append(scalars.to(device))

            # PROJECT VISUAL FEATURES to balanced dimension
            visual_projected = self.visual_projectors[disease](visual_features)  # [B, visual_proj_dim]

            # PROJECT SCALAR FEATURES to balanced dimension (if they exist)
            # BUGFIX: Check if disease has scalar_projector registered (not if scalars are non-empty)
            if disease in self.scalar_projectors and scalar_features_list[0].numel() > 0:
                scalar_features = torch.stack(scalar_features_list, dim=0)  # [B, num_scalars]
                # Note: ALL features now have binary presence indicators (value, is_present)
                # Missing values → value=0.0, present=0.0 | Measured values → value=normalized, present=1.0

                scalar_projected = self.scalar_projectors[disease](scalar_features)  # [B, scalar_proj_dim]

                # BALANCED FUSION: Concatenate equal-dimensional projections
                fused = torch.cat([visual_projected, scalar_projected], dim=1)  # [B, visual_proj_dim + scalar_proj_dim]
            else:
                # No scalars for this disease - use only visual
                fused = visual_projected  # [B, visual_proj_dim]

            # FUSION HEAD: Balanced features → logit
            logit = self.fusion_heads[disease](fused)  # [B, 1]
            logits_list.append(logit)

        logits = torch.cat(logits_list, dim=1)

        return logits



# =============================================================================
# MODEL BUILDER
# =============================================================================

def build_model_from_config(config: Dict[str, Any]) -> nn.Module:
    """
    Build model from config dict.
    
    Config options:
        model_type: "gap", "masked_attn", "scalar_fusion"
        num_diseases: int
        disease_names: List[str]
        backbone: str
        image_size: int
        tri_stride: int
        freeze_backbone: bool
        tau: float
        fusion_hidden: int
        feature_stats_path: str
    """
    model_type = config.get("model_type", "gap")
    
    common_args = {
        "num_diseases": config.get("num_diseases", 30),
        "backbone": config.get("backbone", "facebook/dinov2-small"),
        "image_size": config.get("image_size", 224),
        "tri_stride": config.get("tri_stride", 1),
        "freeze_backbone": config.get("freeze_backbone", True),
    }
    
    if model_type == "gap":
        return JanusGAP(**common_args)
    
    elif model_type == "masked_attn":
        return JanusMaskedAttn(
            **common_args,
            disease_names=config.get("disease_names"),
            tau=config.get("tau", 1.0),
        )
    
    elif model_type == "scalar_fusion":
        return JanusScalarFusion(
            **common_args,
            disease_names=config.get("disease_names"),
            tau=config.get("tau", 1.0),
            fusion_hidden=config.get("fusion_hidden", 256),
            feature_stats_path=config.get("feature_stats_path"),
        )
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == "__main__":
    print("Janus Models")
    print("=" * 60)
    print("""
Three model variants:

1. JanusGAP (baseline)
   - DINOv3 + Global Average Pooling
   - No anatomical guidance
   
2. JanusMaskedAttn
   - Organ-masked attention per disease
   - ROI attention for appendicitis (precise box)
   - Comparative attention for steatosis (liver vs spleen)
   
3. JanusScalarFusion
   - Masked attention + scalar feature fusion
   - Body-size normalized volumes
   - Liver-spleen HU difference
   - SBO diameter ratio
   
Usage:
    model = build_model_from_config({
        "model_type": "scalar_fusion",
        "num_diseases": 30,
        "backbone": "facebook/dinov2-small",
        "feature_stats_path": "feature_stats.json",
    })
    
    logits = model(batch)  # [B, 30]
    """)
