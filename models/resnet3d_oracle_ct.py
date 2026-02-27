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

# janus/models/janus_resnet3d_model.py
"""
Janus Models — Native 3D ResNet Backbone Variants

Four model variants (mirrors janus_model.py for DINOv3):
1. JanusResNet3DGAP:          Inflated ResNet50 + Global Average Pooling (baseline)
2. JanusResNet3DMaskedAttn:   Inflated ResNet50 + 3D Organ-Masked Attention
3. JanusResNet3DScalarFusion: Inflated ResNet50 + 3D Masked Attention + Scalar Fusion
4. JanusResNet3DGatedFusion:  Inflated ResNet50 + 3D Masked Attention + Anatomical Gating

Key difference from DINOv3 JANUS (janus_model.py):
- No tri-slice conversion — processes full 3D CT volume [B,1,D,H,W] natively
- I3D inflation: pretrained 2D ImageNet ResNet50 weights inflated to 3D
- Pyramid features: f3(1024) + f4(2048) = 3072-dim per disease (models 2–4)
- Conv3d(C, 1, 1) attention scorer instead of Linear(D, 1) for 3D spatial features
- Return interface is identical — inference.py and train.py work unchanged

Usage:
    python janus/train.py experiment=resnet3d_gated_fusion dataset=merlin
    python janus/train.py experiment=resnet3d_scalar_fusion dataset=merlin
    python janus/train.py experiment=resnet3d_masked_attn dataset=merlin
    python janus/train.py experiment=resnet3d_baseline_gap dataset=merlin
"""

import math
from typing import Dict, List, Optional, Any, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import Parameter

from ..configs.disease_config import (
    ORGAN_TO_CHANNEL,
    get_all_disease_configs,
    get_all_diseases,
)
from ..datamodules.feature_bank import FeatureBank
from .dinov3_oracle_ct import (
    AnatomicallyGuidedGate,
    get_attention_mask_for_disease,
    dilate_mask_adaptive,
    create_roi_mask,
    to_logit,
    inv_sigmoid_temp,
)


# ImageNet normalization (same constants as janus_model.py)
IMN_MEAN = (0.485, 0.456, 0.406)
IMN_STD  = (0.229, 0.224, 0.225)

RESNET_VARIANTS = ("resnet50", "r50", "resnet101", "r101", "resnet152", "r152")


# =============================================================================
# INFLATED RESNET HELPERS
# Copied verbatim from ct_triage/TriageNet/models/oracle/oracle_ct.py lines 42-192
# =============================================================================

def inflate_conv(
    conv2d: nn.Conv2d,
    time_dim: int = 3,
    time_padding: int = 0,
    time_stride: int = 1,
    time_dilation: int = 1,
    center: bool = False,
) -> nn.Conv3d:
    if conv2d.kernel_size[0] == 7:  # ResNet stem
        kernel_dim = (3, 7, 7)
        padding    = (1, 3, 3)
        stride     = (1, 2, 2)
        dilation   = (1, 1, 1)
        conv3d = nn.Conv3d(
            conv2d.in_channels, conv2d.out_channels,
            kernel_dim, padding=padding, dilation=dilation, stride=stride,
            bias=(conv2d.bias is not None),
        )
        w2 = conv2d.weight.data
        if center:
            w3 = torch.zeros_like(w2).unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            mid = time_dim // 2
            w3[:, :, mid, :, :] = w2
        else:
            w3 = w2.unsqueeze(2).repeat(1, 1, time_dim, 1, 1) / float(time_dim)
        conv3d.weight = Parameter(w3)
        if conv2d.bias is not None:
            conv3d.bias = Parameter(conv2d.bias.data.clone())
        return conv3d

    kernel_dim = (time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
    padding    = (time_padding, conv2d.padding[0], conv2d.padding[1])
    stride     = (time_stride,  conv2d.stride[0],  conv2d.stride[1])
    dilation   = (time_dilation, conv2d.dilation[0], conv2d.dilation[1])
    conv3d = nn.Conv3d(
        conv2d.in_channels, conv2d.out_channels,
        kernel_dim, padding=padding, dilation=dilation, stride=stride,
        bias=(conv2d.bias is not None),
    )
    w2 = conv2d.weight.data
    if center:
        w3 = torch.zeros_like(w2).unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        mid = time_dim // 2
        w3[:, :, mid, :, :] = w2
    else:
        w3 = w2.unsqueeze(2).repeat(1, 1, time_dim, 1, 1) / float(time_dim)
    conv3d.weight = Parameter(w3)
    if conv2d.bias is not None:
        conv3d.bias = Parameter(conv2d.bias.data.clone())
    return conv3d


def inflate_batch_norm(batch2d: nn.BatchNorm2d) -> nn.BatchNorm3d:
    bn3d = nn.BatchNorm3d(batch2d.num_features)
    batch2d._check_input_dim = bn3d._check_input_dim  # type: ignore[attr-defined]
    return batch2d  # reuse running stats/affine params


def inflate_pool2d_to3d(
    pool2d: nn.Module,
    t_k: int = 3,
    t_pad: int = 1,
    t_stride: Optional[int] = None,
) -> nn.Module:
    if isinstance(pool2d, nn.AdaptiveAvgPool2d):
        return nn.AdaptiveAvgPool3d((1, 1, 1))
    if t_stride is None:
        t_stride = t_k
    if isinstance(pool2d, nn.MaxPool2d):
        return nn.MaxPool3d(
            kernel_size=(t_k, pool2d.kernel_size, pool2d.kernel_size),
            stride=(t_stride, pool2d.stride, pool2d.stride),
            padding=(t_pad, pool2d.padding, pool2d.padding),
            dilation=(pool2d.dilation, pool2d.dilation, pool2d.dilation),
            ceil_mode=pool2d.ceil_mode,
        )
    if isinstance(pool2d, nn.AvgPool2d):
        return nn.AvgPool3d(
            kernel_size=(t_k, pool2d.kernel_size, pool2d.kernel_size),
            stride=(t_stride, pool2d.stride, pool2d.stride),
        )
    raise ValueError(type(pool2d))


class Bottleneck3d(nn.Module):
    def __init__(self, bottleneck2d: nn.Module):
        super().__init__()
        if not hasattr(bottleneck2d, "conv3"):
            raise NotImplementedError("Only torchvision ResNet bottlenecks supported.")
        spatial_stride = bottleneck2d.conv2.stride[0]

        self.conv1 = inflate_conv(bottleneck2d.conv1, time_dim=1, center=True)
        self.bn1   = inflate_batch_norm(bottleneck2d.bn1)
        self.conv2 = inflate_conv(
            bottleneck2d.conv2, time_dim=3, time_padding=1,
            time_stride=spatial_stride, center=True,
        )
        self.bn2   = inflate_batch_norm(bottleneck2d.bn2)
        self.conv3 = inflate_conv(bottleneck2d.conv3, time_dim=1, center=True)
        self.bn3   = inflate_batch_norm(bottleneck2d.bn3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if bottleneck2d.downsample is not None:
            self.downsample = nn.Sequential(
                inflate_conv(
                    bottleneck2d.downsample[0], time_dim=1,
                    time_stride=spatial_stride, center=True,
                ),
                inflate_batch_norm(bottleneck2d.downsample[1]),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def core(inp: torch.Tensor) -> torch.Tensor:
            y = self.conv1(inp); y = self.bn1(y); y = self.relu(y)
            y = self.conv2(y);   y = self.bn2(y); y = self.relu(y)
            y = self.conv3(y);   y = self.bn3(y)
            return y
        identity = x if self.downsample is None else self.downsample(x)
        y = checkpoint.checkpoint(core, x, use_reentrant=False) if x.requires_grad else core(x)
        return self.relu(y + identity)


def inflate_reslayer(reslayer2d: nn.Sequential) -> nn.Sequential:
    return nn.Sequential(*[Bottleneck3d(b) for b in reslayer2d])


class I3ResNetTrunk(nn.Module):
    """
    Inflated 3D ResNet trunk — processes [B, 3, D, H, W] volumes.

    Returns pyramid features:
      f3: [B, 1024, D/8,  H/16, W/16]   (stage 3)
      f4: [B, 2048, D/16, H/32, W/32]   (stage 4)
    """

    def __init__(self, resnet2d: nn.Module, use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)

        self.conv1   = inflate_conv(resnet2d.conv1, time_dim=3, time_padding=1, center=True)
        self.bn1     = inflate_batch_norm(resnet2d.bn1)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = inflate_pool2d_to3d(resnet2d.maxpool, t_k=3, t_pad=1, t_stride=2)

        self.layer1 = inflate_reslayer(resnet2d.layer1)
        self.layer2 = inflate_reslayer(resnet2d.layer2)
        self.layer3 = inflate_reslayer(resnet2d.layer3)
        self.layer4 = inflate_reslayer(resnet2d.layer4)

    def forward(self, x3: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x3: [B, 3, D, H, W]
        def _run(inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            y  = self.conv1(inp); y = self.bn1(y); y = self.relu(y)
            y  = self.maxpool(y)
            y  = self.layer1(y)
            y  = self.layer2(y)
            f3 = self.layer3(y)
            f4 = self.layer4(f3)
            return f3, f4

        if self.use_checkpoint and self.training:
            x3 = x3.detach()
            x3.requires_grad_(True)
            return checkpoint.checkpoint(_run, x3, use_reentrant=False)
        else:
            return _run(x3)


# =============================================================================
# SHARED BACKBONE BUILDER
# =============================================================================

def _build_trunk(backbone: str, pretrained: bool, use_checkpoint: bool):
    """Load 2D ResNet, extract channel dims, inflate to 3D trunk."""
    from torchvision import models as tvm
    bk = backbone.lower()
    if bk in ("resnet50", "r50"):
        r2d = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    elif bk in ("resnet101", "r101"):
        r2d = tvm.resnet101(weights=tvm.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
    elif bk in ("resnet152", "r152"):
        r2d = tvm.resnet152(weights=tvm.ResNet152_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(f"backbone must be one of {RESNET_VARIANTS}, got '{backbone}'")
    ch3 = int(r2d.layer3[-1].conv3.out_channels)   # 1024
    ch4 = int(r2d.layer4[-1].conv3.out_channels)   # 2048
    trunk = I3ResNetTrunk(r2d, use_checkpoint=use_checkpoint)
    return trunk, ch3, ch4


# =============================================================================
# 3D MASKED ATTENTION POOLING
# Replaces the 2D tri-slice masked_attention_pool for 3D feature maps
# =============================================================================

def masked_attention_pool_3d(
    feat_map:   torch.Tensor,       # [B, C, D', H', W']
    mask_3d:    torch.Tensor,       # [B, 1, D,  H,  W ] original resolution
    score_conv: nn.Module,          # Conv3d(C, 1, kernel_size=1)
    tau:    Union[float, torch.Tensor] = 1.0,
    bias_in:  Optional[torch.Tensor] = None,
    bias_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:                  # [B, C]
    """
    Content-aware attention pooling over a 3D feature map restricted to an organ mask.

    Mirrors masked_attention_pool() from janus_model.py but operates on 3D spatial
    feature maps [B,C,D',H',W'] instead of ViT token sequences [B,T,N,D].
    """
    B, C, Dp, Hp, Wp = feat_map.shape
    N = Dp * Hp * Wp

    # Resize mask to feature map resolution
    mask      = F.interpolate(mask_3d.float(), size=(Dp, Hp, Wp), mode="nearest")
    mask_flat = (mask > 0.5).float().view(B, N)   # [B, N]

    # Attention scores via 1×1×1 conv
    logits = score_conv(feat_map).view(B, N)      # [B, N]

    # Learnable priors
    if bias_in is not None and bias_out is not None:
        logits = (logits
                  + bias_in.view(1, 1)  * mask_flat
                  + bias_out.view(1, 1) * (1.0 - mask_flat))

    # Temperature scaling
    if isinstance(tau, torch.Tensor):
        logits = logits / tau.view(1, 1)
    else:
        logits = logits / tau

    # Masked softmax
    very_neg     = torch.finfo(logits.dtype).min / 4
    masked_logits = torch.where(mask_flat > 0, logits, torch.full_like(logits, very_neg))
    lmax          = masked_logits.max(dim=-1, keepdim=True).values
    exp_w         = torch.exp((masked_logits - lmax).clamp(-60, 0)) * mask_flat
    weights       = exp_w / exp_w.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # Empty mask → uniform distribution
    empty = mask_flat.sum(dim=-1, keepdim=True) < 0.5
    if empty.any():
        weights = torch.where(empty, torch.ones_like(weights) / N, weights)

    # NaN safety
    if torch.isnan(weights).any():
        weights = torch.where(torch.isnan(weights), torch.ones_like(weights) / N, weights)

    # Weighted sum over spatial positions
    feat_flat = feat_map.view(B, C, N)
    pooled    = (feat_flat * weights.unsqueeze(1)).sum(dim=-1)   # [B, C]

    if torch.isnan(pooled).any():
        pooled = torch.where(torch.isnan(pooled), torch.zeros_like(pooled), pooled)

    return pooled


# =============================================================================
# MODEL 1: GLOBAL AVERAGE POOLING (BASELINE)
# =============================================================================

class JanusResNet3DGAP(nn.Module):
    """
    Baseline: Inflated ResNet50 + Global Average Pooling

    No organ-specific attention, no scalar features.
    Uses only f4 (2048-dim) with 3D GAP — clean neural baseline.
    """

    def __init__(
        self,
        num_diseases:    int  = 30,
        backbone:        str  = "resnet50",
        pretrained:      bool = True,
        use_checkpoint:  bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.num_diseases    = num_diseases
        self.freeze_backbone = freeze_backbone

        self.trunk, _ch3, self._ch4 = _build_trunk(backbone, pretrained, use_checkpoint)
        self.hidden_dim = self._ch4   # 2048 — GAP uses only f4

        if freeze_backbone:
            for p in self.trunk.parameters():
                p.requires_grad = False
            self.trunk.eval()

        self.head = nn.Linear(self.hidden_dim, num_diseases)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.head.bias, 0.0)

        self.register_buffer("_mean", torch.tensor(IMN_MEAN).view(1, 3, 1, 1, 1))
        self.register_buffer("_std",  torch.tensor(IMN_STD).view(1, 3, 1, 1, 1))

        print(f"✓ JanusResNet3DGAP: {backbone} → f4({self.hidden_dim}d) GAP → head")

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / self._std

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.trunk.eval()
        return self

    def forward(self, batch: Dict[str, Any], **kwargs) -> torch.Tensor:
        image = batch["image"]   # [B, 1, D, H, W]
        B = image.size(0)

        image_3ch = image.expand(-1, 3, -1, -1, -1).contiguous()
        image_3ch = self._normalize(image_3ch)

        _f3, f4 = self.trunk(image_3ch)

        # 3D global average pooling → [B, 2048]
        pooled = F.adaptive_avg_pool3d(f4, 1).flatten(1)

        return self.head(pooled)


# =============================================================================
# MODEL 2: 3D MASKED ATTENTION
# =============================================================================

class JanusResNet3DMaskedAttn(nn.Module):
    """
    Inflated ResNet50 + 3D Organ-Masked Attention

    Per-disease 3D attention masks restrict pooling to the relevant organ region.
    Uses pyramid f3+f4 features (3072-dim), pooled separately per disease.
    No scalar features — pure visual with anatomical guidance.
    """

    def __init__(
        self,
        num_diseases:    int   = 30,
        disease_names:   List[str] = None,
        backbone:        str   = "resnet50",
        pretrained:      bool  = True,
        use_checkpoint:  bool  = True,
        freeze_backbone: bool  = False,
        learn_tau:       bool  = True,
        init_tau:        float = 0.7,
        fixed_tau:       float = 1.0,
        use_mask_bias:   bool  = True,
        init_inside:     float = 0.8,
        init_outside:    float = 0.2,
    ):
        super().__init__()
        self.num_diseases    = num_diseases
        self.disease_names   = disease_names or get_all_diseases()[:num_diseases]
        self.learn_tau       = learn_tau
        self.fixed_tau       = fixed_tau
        self.use_mask_bias   = use_mask_bias
        self.freeze_backbone = freeze_backbone

        self.trunk, self._ch3, self._ch4 = _build_trunk(backbone, pretrained, use_checkpoint)
        self.hidden_dim = self._ch3 + self._ch4   # 3072

        if freeze_backbone:
            for p in self.trunk.parameters():
                p.requires_grad = False
            self.trunk.eval()

        # Per-disease modules
        self.score_convs_f3 = nn.ModuleDict()
        self.score_convs_f4 = nn.ModuleDict()
        self.heads          = nn.ModuleDict()

        if use_mask_bias:
            self.inside_logit  = nn.ParameterDict()
            self.outside_logit = nn.ParameterDict()
        if learn_tau:
            self.temp_logit = nn.ParameterDict()

        for disease in self.disease_names:
            self.score_convs_f3[disease] = nn.Conv3d(self._ch3, 1, kernel_size=1, bias=True)
            self.score_convs_f4[disease] = nn.Conv3d(self._ch4, 1, kernel_size=1, bias=True)
            self.heads[disease]          = nn.Linear(self.hidden_dim, 1)

            if use_mask_bias:
                self.inside_logit[disease]  = nn.Parameter(torch.tensor(to_logit(init_inside)))
                self.outside_logit[disease] = nn.Parameter(torch.tensor(to_logit(init_outside)))
            if learn_tau:
                self.temp_logit[disease] = nn.Parameter(torch.tensor(inv_sigmoid_temp(init_tau)))

        self.register_buffer("_mean", torch.tensor(IMN_MEAN).view(1, 3, 1, 1, 1))
        self.register_buffer("_std",  torch.tensor(IMN_STD).view(1, 3, 1, 1, 1))

        print(f"✓ JanusResNet3DMaskedAttn: {backbone} → "
              f"f3({self._ch3}) + f4({self._ch4}) = {self.hidden_dim}d masked attention")

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / self._std

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.trunk.eval()
        return self

    def forward(self, batch: Dict[str, Any], **kwargs) -> torch.Tensor:
        image        = batch["image"]
        masks        = batch["masks"]
        disease_rois = batch.get("disease_rois", [{}] * image.size(0))
        meta         = batch.get("meta",         [{}] * image.size(0))

        B = image.size(0)
        device = image.device

        image_3ch = image.expand(-1, 3, -1, -1, -1).contiguous()
        image_3ch = self._normalize(image_3ch)
        f3, f4    = self.trunk(image_3ch)

        logits_list = []
        for disease in self.disease_names:
            tau     = (0.2 + 1.8 * torch.sigmoid(self.temp_logit[disease])
                       if self.learn_tau else self.fixed_tau)
            bias_in  = self.inside_logit[disease]  if self.use_mask_bias else None
            bias_out = self.outside_logit[disease] if self.use_mask_bias else None

            attn_mask = get_attention_mask_for_disease(
                disease, masks, disease_rois, meta, device, allow_comparative=False,
            )

            p3 = masked_attention_pool_3d(f3, attn_mask, self.score_convs_f3[disease],
                                          tau=tau, bias_in=bias_in, bias_out=bias_out)
            p4 = masked_attention_pool_3d(f4, attn_mask, self.score_convs_f4[disease],
                                          tau=tau, bias_in=bias_in, bias_out=bias_out)
            visual_features = torch.cat([p3, p4], dim=1)   # [B, 3072]

            logit = self.heads[disease](visual_features)
            if torch.isnan(logit).any():
                logit = torch.where(torch.isnan(logit), torch.zeros_like(logit), logit)
            logits_list.append(logit)

        return torch.cat(logits_list, dim=1)


# =============================================================================
# MODEL 3: 3D SCALAR FUSION
# =============================================================================

class JanusResNet3DScalarFusion(nn.Module):
    """
    Inflated ResNet50 + 3D Masked Attention + Scalar Feature Fusion

    Fuses visual features (3072-dim pyramid) with scalar radiomics features
    via separate projectors → balanced concatenation → fusion MLP.
    Same fusion strategy as JanusScalarFusion (DINOv3).
    """

    def __init__(
        self,
        num_diseases:    int   = 30,
        disease_names:   List[str] = None,
        backbone:        str   = "resnet50",
        pretrained:      bool  = True,
        use_checkpoint:  bool  = True,
        freeze_backbone: bool  = False,
        learn_tau:       bool  = True,
        init_tau:        float = 0.7,
        fixed_tau:       float = 1.0,
        use_mask_bias:   bool  = True,
        init_inside:     float = 0.8,
        init_outside:    float = 0.2,
        visual_proj_dim: int   = 256,
        scalar_proj_dim: int   = 256,
        fusion_hidden:   int   = 256,
        feature_stats_path: Optional[str] = None,
    ):
        super().__init__()
        self.num_diseases    = num_diseases
        self.disease_names   = disease_names or get_all_diseases()[:num_diseases]
        self.learn_tau       = learn_tau
        self.fixed_tau       = fixed_tau
        self.use_mask_bias   = use_mask_bias
        self.freeze_backbone = freeze_backbone

        self.trunk, self._ch3, self._ch4 = _build_trunk(backbone, pretrained, use_checkpoint)
        self.hidden_dim = self._ch3 + self._ch4   # 3072

        if freeze_backbone:
            for p in self.trunk.parameters():
                p.requires_grad = False
            self.trunk.eval()

        self.feature_bank = FeatureBank(stats_path=feature_stats_path, normalize="zscore")

        # Per-disease modules
        self.score_convs_f3  = nn.ModuleDict()
        self.score_convs_f4  = nn.ModuleDict()
        self.visual_projectors = nn.ModuleDict()
        self.scalar_projectors = nn.ModuleDict()
        self.fusion_heads      = nn.ModuleDict()

        if use_mask_bias:
            self.inside_logit  = nn.ParameterDict()
            self.outside_logit = nn.ParameterDict()
        if learn_tau:
            self.temp_logit = nn.ParameterDict()

        disease_configs = get_all_disease_configs()

        for disease in self.disease_names:
            self.score_convs_f3[disease] = nn.Conv3d(self._ch3, 1, kernel_size=1, bias=True)
            self.score_convs_f4[disease] = nn.Conv3d(self._ch4, 1, kernel_size=1, bias=True)

            if use_mask_bias:
                self.inside_logit[disease]  = nn.Parameter(torch.tensor(to_logit(init_inside)))
                self.outside_logit[disease] = nn.Parameter(torch.tensor(to_logit(init_outside)))
            if learn_tau:
                self.temp_logit[disease] = nn.Parameter(torch.tensor(inv_sigmoid_temp(init_tau)))

            # Visual projector: 3072 → visual_proj_dim
            self.visual_projectors[disease] = nn.Sequential(
                nn.Linear(self.hidden_dim, visual_proj_dim),
                nn.ReLU(inplace=True),
            )

            config = disease_configs.get(disease)
            num_scalars = (
                len(config.scalar_features) + len(config.derived_features)
            ) if config else 0

            if num_scalars > 0:
                scalar_hidden = max(64, num_scalars * 4)
                self.scalar_projectors[disease] = nn.Sequential(
                    nn.Linear(num_scalars, scalar_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(scalar_hidden, scalar_proj_dim),
                    nn.ReLU(inplace=True),
                )

            fusion_input_dim = visual_proj_dim + (scalar_proj_dim if num_scalars > 0 else 0)
            self.fusion_heads[disease] = nn.Sequential(
                nn.LayerNorm(fusion_input_dim),
                nn.Linear(fusion_input_dim, fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(fusion_hidden, 1),
            )

        self.register_buffer("_mean", torch.tensor(IMN_MEAN).view(1, 3, 1, 1, 1))
        self.register_buffer("_std",  torch.tensor(IMN_STD).view(1, 3, 1, 1, 1))

        print(f"✓ JanusResNet3DScalarFusion: {backbone} → "
              f"f3({self._ch3})+f4({self._ch4})={self.hidden_dim}d → "
              f"Visual({visual_proj_dim}d) + Scalar({scalar_proj_dim}d) → Fusion({fusion_hidden}d)")

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / self._std

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.trunk.eval()
        return self

    def forward(self, batch: Dict[str, Any], **kwargs) -> torch.Tensor:
        image         = batch["image"]
        masks         = batch["masks"]
        features_rows = batch.get("features_row", [None] * image.size(0))
        disease_rois  = batch.get("disease_rois", [{}] * image.size(0))
        meta          = batch.get("meta",         [{}] * image.size(0))

        B = image.size(0)
        device = image.device

        image_3ch = image.expand(-1, 3, -1, -1, -1).contiguous()
        image_3ch = self._normalize(image_3ch)
        f3, f4    = self.trunk(image_3ch)

        derived_features_batch = [
            self.feature_bank.compute_derived_features(meta[b], features_row=features_rows[b])
            for b in range(B)
        ]

        logits_list = []
        for disease in self.disease_names:
            tau     = (0.2 + 1.8 * torch.sigmoid(self.temp_logit[disease])
                       if self.learn_tau else self.fixed_tau)
            bias_in  = self.inside_logit[disease]  if self.use_mask_bias else None
            bias_out = self.outside_logit[disease] if self.use_mask_bias else None

            attn_mask = get_attention_mask_for_disease(
                disease, masks, disease_rois, meta, device, allow_comparative=False,
            )

            p3 = masked_attention_pool_3d(f3, attn_mask, self.score_convs_f3[disease],
                                          tau=tau, bias_in=bias_in, bias_out=bias_out)
            p4 = masked_attention_pool_3d(f4, attn_mask, self.score_convs_f4[disease],
                                          tau=tau, bias_in=bias_in, bias_out=bias_out)
            visual_features   = torch.cat([p3, p4], dim=1)              # [B, 3072]
            visual_projected  = self.visual_projectors[disease](visual_features)  # [B, visual_proj_dim]

            if disease in self.scalar_projectors:
                scalar_list = [
                    self.feature_bank.get_features_for_disease(
                        disease, meta[b], features_row=features_rows[b], normalize=True,
                        cached_derived=derived_features_batch[b],
                    )[0].to(device)
                    for b in range(B)
                ]
                scalar_features  = torch.stack(scalar_list, dim=0)
                scalar_projected = self.scalar_projectors[disease](scalar_features)
                fused = torch.cat([visual_projected, scalar_projected], dim=1)
            else:
                fused = visual_projected

            logits_list.append(self.fusion_heads[disease](fused))

        return torch.cat(logits_list, dim=1)


# =============================================================================
# MODEL 4: 3D GATED FUSION
# =============================================================================

class JanusResNet3DGatedFusion(nn.Module):
    """
    Inflated ResNet50 + 3D Masked Attention + Anatomically Guided Gating

    Scalar priors gate (element-wise modulate) the 3072-dim pyramid visual features.
    Same gating mechanism as JanusGatedFusion (DINOv3).
    Return interface is identical — inference.py works unchanged.
    """

    def __init__(
        self,
        num_diseases:    int   = 30,
        disease_names:   List[str] = None,
        backbone:        str   = "resnet50",
        pretrained:      bool  = True,
        use_checkpoint:  bool  = True,
        freeze_backbone: bool  = False,
        learn_tau:       bool  = True,
        init_tau:        float = 0.7,
        fixed_tau:       float = 1.0,
        use_mask_bias:   bool  = True,
        init_inside:     float = 0.8,
        init_outside:    float = 0.2,
        feature_stats_path: Optional[str] = None,
    ):
        super().__init__()
        self.num_diseases    = num_diseases
        self.disease_names   = disease_names or get_all_diseases()[:num_diseases]
        self.learn_tau       = learn_tau
        self.fixed_tau       = fixed_tau
        self.use_mask_bias   = use_mask_bias
        self.freeze_backbone = freeze_backbone

        self.trunk, self._ch3, self._ch4 = _build_trunk(backbone, pretrained, use_checkpoint)
        self.hidden_dim = self._ch3 + self._ch4   # 3072

        if freeze_backbone:
            for p in self.trunk.parameters():
                p.requires_grad = False
            self.trunk.eval()

        self.feature_bank = FeatureBank(stats_path=feature_stats_path, normalize="zscore")

        # Per-disease modules
        self.score_convs_f3 = nn.ModuleDict()
        self.score_convs_f4 = nn.ModuleDict()
        self.gating_modules = nn.ModuleDict()
        self.heads_visual   = nn.ModuleDict()

        if use_mask_bias:
            self.inside_logit  = nn.ParameterDict()
            self.outside_logit = nn.ParameterDict()
        if learn_tau:
            self.temp_logit = nn.ParameterDict()

        disease_configs = get_all_disease_configs()

        for disease in self.disease_names:
            self.score_convs_f3[disease] = nn.Conv3d(self._ch3, 1, kernel_size=1, bias=True)
            self.score_convs_f4[disease] = nn.Conv3d(self._ch4, 1, kernel_size=1, bias=True)

            if use_mask_bias:
                self.inside_logit[disease]  = nn.Parameter(torch.tensor(to_logit(init_inside)))
                self.outside_logit[disease] = nn.Parameter(torch.tensor(to_logit(init_outside)))
            if learn_tau:
                self.temp_logit[disease] = nn.Parameter(torch.tensor(inv_sigmoid_temp(init_tau)))

            config = disease_configs.get(disease)
            num_scalars = (
                len(config.scalar_features) + len(config.derived_features)
            ) if config else 0

            if num_scalars > 0:
                self.gating_modules[disease] = AnatomicallyGuidedGate(
                    visual_dim=self.hidden_dim,
                    scalar_dim=num_scalars,
                )

            self.heads_visual[disease] = nn.Linear(self.hidden_dim, 1)

        self.register_buffer("_mean", torch.tensor(IMN_MEAN).view(1, 3, 1, 1, 1))
        self.register_buffer("_std",  torch.tensor(IMN_STD).view(1, 3, 1, 1, 1))

        print(f"✓ JanusResNet3DGatedFusion: {backbone} → "
              f"f3({self._ch3})+f4({self._ch4})={self.hidden_dim}d → Anatomical Gating")

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / self._std

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.trunk.eval()
        return self

    def forward(
        self,
        batch: Dict[str, Any],
        return_ungated: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass — return interface identical to JanusGatedFusion.
        return_ungated=True returns {"logits": ..., "logits_ungated": ...} for veto analysis.
        """
        image         = batch["image"]
        masks         = batch["masks"]
        features_rows = batch.get("features_row", [None] * image.size(0))
        disease_rois  = batch.get("disease_rois", [{}] * image.size(0))
        meta          = batch.get("meta",         [{}] * image.size(0))

        B = image.size(0)
        device = image.device

        image_3ch = image.expand(-1, 3, -1, -1, -1).contiguous()
        image_3ch = self._normalize(image_3ch)
        f3, f4    = self.trunk(image_3ch)

        derived_features_batch = [
            self.feature_bank.compute_derived_features(meta[b], features_row=features_rows[b])
            for b in range(B)
        ]

        logits_list         = []
        logits_ungated_list = [] if return_ungated else None

        for disease in self.disease_names:
            tau     = (0.2 + 1.8 * torch.sigmoid(self.temp_logit[disease])
                       if self.learn_tau else self.fixed_tau)
            bias_in  = self.inside_logit[disease]  if self.use_mask_bias else None
            bias_out = self.outside_logit[disease] if self.use_mask_bias else None

            attn_mask = get_attention_mask_for_disease(
                disease, masks, disease_rois, meta, device, allow_comparative=False,
            )

            p3 = masked_attention_pool_3d(f3, attn_mask, self.score_convs_f3[disease],
                                          tau=tau, bias_in=bias_in, bias_out=bias_out)
            p4 = masked_attention_pool_3d(f4, attn_mask, self.score_convs_f4[disease],
                                          tau=tau, bias_in=bias_in, bias_out=bias_out)
            visual_features = torch.cat([p3, p4], dim=1)   # [B, 3072]

            if disease in self.gating_modules:
                scalar_list = [
                    self.feature_bank.get_features_for_disease(
                        disease, meta[b], features_row=features_rows[b], normalize=True,
                        cached_derived=derived_features_batch[b],
                    )[0].to(device)
                    for b in range(B)
                ]
                scalar_features = torch.stack(scalar_list, dim=0)

                gated_visual = self.gating_modules[disease](visual_features, scalar_features)

                if return_ungated:
                    ungated_visual = visual_features

                logit = self.heads_visual[disease](gated_visual)
                if return_ungated:
                    logit_ungated = self.heads_visual[disease](ungated_visual)
            else:
                logit = self.heads_visual[disease](visual_features)
                if return_ungated:
                    logit_ungated = logit

            logits_list.append(logit)
            if return_ungated:
                logits_ungated_list.append(logit_ungated)

        logits = torch.cat(logits_list, dim=1)

        if return_ungated:
            return {
                "logits"         : logits,
                "logits_ungated" : torch.cat(logits_ungated_list, dim=1),
            }

        return logits
