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
# radioprior_v2/datamodules/augmentation.py
"""
Data Augmentation for RadioPrior

Supports two presets:
1. legacy_v1: Simple flips and 90-degree rotations
2. anatomy_safe_v2: Realistic augmentation (small rotations, scaling, gamma, noise)
"""

from __future__ import annotations
import math
import random
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------
# Legacy (Simple Augmentation)
# ---------------------------
def _sample_aug_ops_legacy(
    allow_rot90: bool = True,
    allow_flip_w: bool = True,
    allow_flip_h: bool = True,
    allow_flip_d: bool = True,
) -> Dict[str, object]:
    """Sample augmentation operations for legacy mode."""
    return {
        "rot_k": (random.randint(1, 3) if (allow_rot90 and random.random() < 0.5) else 0),
        "flip_w": (allow_flip_w and random.random() < 0.5),
        "flip_h": (allow_flip_h and random.random() < 0.5),
        "flip_d": (allow_flip_d and random.random() < 0.5),
    }


def _apply_ops_3d(x: torch.Tensor, ops: Dict[str, object]) -> torch.Tensor:
    """Apply geometric transforms to 3D tensor."""
    if x is None:
        return None

    # x can be [1,D,H,W] or [O,D,H,W]
    if ops.get("rot_k", 0):
        x = torch.rot90(x, int(ops["rot_k"]), dims=(-1, -2))  # rot over H/W
    if ops.get("flip_w", False):
        x = torch.flip(x, dims=(-1,))
    if ops.get("flip_h", False):
        x = torch.flip(x, dims=(-2,))
    if ops.get("flip_d", False):
        x = torch.flip(x, dims=(-3,))
    return x.contiguous()


def legacy_aug(
    image_1dhw: torch.Tensor,
    mask_odhw: Optional[torch.Tensor],
    *,
    allow_rot90: bool = True,
    allow_flip_w: bool = True,
    allow_flip_h: bool = True,
    allow_flip_d: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Legacy augmentation: 90-degree rotations and axis flips.

    Args:
        image_1dhw: Image tensor [1, D, H, W]
        mask_odhw: Mask tensor [O, D, H, W] or None
        allow_rot90: Allow 90-degree rotations
        allow_flip_w: Allow width (left-right) flips
        allow_flip_h: Allow height (anterior-posterior) flips
        allow_flip_d: Allow depth (superior-inferior) flips

    Returns:
        Augmented image and mask
    """
    ops = _sample_aug_ops_legacy(allow_rot90, allow_flip_w, allow_flip_h, allow_flip_d)

    img = _apply_ops_3d(image_1dhw, ops)
    msk = _apply_ops_3d(mask_odhw, ops) if (mask_odhw is not None) else None

    return img, msk


# ---------------------------------------
# Anatomy-Safe Strong Augmentation
# ---------------------------------------
def _affine_grid_3d(theta_3x4: torch.Tensor, size_1cDHW: torch.Size) -> torch.Tensor:
    """Create affine grid for 3D transformation."""
    return F.affine_grid(theta_3x4, size_1cDHW, align_corners=True)


def _rand_uniform(a: float, b: float) -> float:
    """Sample from uniform distribution [a, b]."""
    return a + (b - a) * random.random()


def anatomy_safe_strong_aug(
    image_1dhw: torch.Tensor,
    mask_odhw: Optional[torch.Tensor],
    *,
    p_affine: float = 0.35,
    rot_deg: float = 10.0,             # in-plane (axial) rotation ±rot_deg
    translate_xy: float = 5.0,         # pixels
    scale_min: float = 0.95,
    scale_max: float = 1.05,
    allow_flip_w: bool = False,        # left-right flip OFF by default
    allow_flip_h: bool = False,        # AP flip often unrealistic; default OFF
    allow_flip_d: bool = False,        # cranio-caudal flip OFF by default
    p_gamma: float = 0.30,
    gamma_min: float = 0.90,
    gamma_max: float = 1.10,
    p_noise: float = 0.30,
    noise_std: float = 0.01,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Anatomy-safe realistic augmentation.

    Applies:
    1. Small affine transforms (rotation, translation, scaling)
    2. Optional flips (disabled by default for anatomical realism)
    3. Gamma correction (brightness/contrast)
    4. Additive Gaussian noise

    Args:
        image_1dhw: Image tensor [1, D, H, W]
        mask_odhw: Mask tensor [O, D, H, W] or None
        p_affine: Probability of applying affine transform
        rot_deg: Maximum rotation in degrees (±rot_deg)
        translate_xy: Maximum translation in pixels
        scale_min: Minimum scaling factor
        scale_max: Maximum scaling factor
        allow_flip_w: Allow left-right flips
        allow_flip_h: Allow anterior-posterior flips
        allow_flip_d: Allow superior-inferior flips
        p_gamma: Probability of gamma correction
        gamma_min: Minimum gamma value
        gamma_max: Maximum gamma value
        p_noise: Probability of adding noise
        noise_std: Standard deviation of Gaussian noise

    Returns:
        Augmented image and mask
    """
    img = image_1dhw
    msk = mask_odhw

    # 1) Optional small realistic 3D affine
    if random.random() < p_affine:
        assert img.dim() == 4 and img.size(0) == 1, f"image must be [1,D,H,W], got {tuple(img.shape)}"
        _, D, H, W = img.shape

        # Sample affine parameters
        theta = _rand_uniform(-rot_deg, rot_deg) * math.pi / 180.0
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        s = _rand_uniform(scale_min, scale_max)

        # Translate (pixels -> normalized [-1,1])
        tx_pix = _rand_uniform(-translate_xy, translate_xy)
        ty_pix = _rand_uniform(-translate_xy, translate_xy)
        tx = (2.0 * tx_pix) / max(W - 1, 1)
        ty = (2.0 * ty_pix) / max(H - 1, 1)

        # 3x4 affine matrix (rotate/scale in x/y plane; z unchanged)
        # Use float32 for grid_sample compatibility (fp16 not supported on CPU)
        A = torch.tensor(
            [[s * cos_t, -s * sin_t, 0.0, tx],
             [s * sin_t,  s * cos_t, 0.0, ty],
             [0.0,        0.0,       1.0, 0.0]],
            dtype=torch.float32, device=img.device
        ).unsqueeze(0)  # [1, 3, 4]

        # Apply affine transform to image
        # Convert to float32 for grid_sample (fp16 not supported on CPU)
        img_ = img.unsqueeze(0).float()  # [1, 1, D, H, W]
        grid = _affine_grid_3d(A, img_.size())
        img_ = F.grid_sample(img_, grid, mode="bilinear", padding_mode="border", align_corners=True)
        img = img_.squeeze(0).to(img.dtype)  # Convert back to original dtype

        # Apply affine transform to mask
        if msk is not None:
            m_ = msk.unsqueeze(0)  # [1, O, D, H, W]
            m_ = F.grid_sample(m_.float(), grid, mode="nearest", padding_mode="border", align_corners=True)
            msk = (m_ > 0.5).to(msk.dtype).squeeze(0)

    # 2) Optional simple flips (disabled by default)
    ops = {
        "rot_k": 0,
        "flip_w": (allow_flip_w and random.random() < 0.5),
        "flip_h": (allow_flip_h and random.random() < 0.5),
        "flip_d": (allow_flip_d and random.random() < 0.5),
    }
    img = _apply_ops_3d(img, ops)
    if msk is not None:
        msk = _apply_ops_3d(msk, ops)

    # 3) Optional gamma jitter (brightness/contrast)
    if random.random() < p_gamma:
        imin = float(img.min())
        img_shift = img - imin
        g = _rand_uniform(gamma_min, gamma_max)
        img = torch.clamp(img_shift, min=0.0) ** g + imin

    # 4) Optional additive Gaussian noise
    if random.random() < p_noise:
        noise = torch.randn_like(img) * noise_std
        img = img + noise

    return img.contiguous(), (msk.contiguous() if msk is not None else None)


# ---------------------------
# Public API
# ---------------------------
def apply_augmentation(
    image_1dhw: torch.Tensor,
    mask_odhw: Optional[torch.Tensor],
    *,
    preset: str,
    params: Dict[str, object],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Apply augmentation to image and mask.

    Args:
        image_1dhw: Image tensor [1, D, H, W]
        mask_odhw: Mask tensor [O, D, H, W] or None
        preset: Augmentation preset ("legacy_v1" or "anatomy_safe_v2")
        params: Parameters for the chosen preset

    Returns:
        Augmented image and mask

    Example:
        >>> img_aug, mask_aug = apply_augmentation(
        ...     image, mask,
        ...     preset="anatomy_safe_v2",
        ...     params={"p_affine": 0.35, "rot_deg": 10.0}
        ... )
    """
    preset = str(preset).lower()

    if preset == "legacy_v1":
        return legacy_aug(
            image_1dhw, mask_odhw,
            allow_rot90=bool(params.get("allow_rot90", True)),
            allow_flip_w=bool(params.get("allow_flip_w", True)),
            allow_flip_h=bool(params.get("allow_flip_h", True)),
            allow_flip_d=bool(params.get("allow_flip_d", True)),
        )

    elif preset == "anatomy_safe_v2":
        return anatomy_safe_strong_aug(
            image_1dhw, mask_odhw,
            p_affine=float(params.get("p_affine", 0.35)),
            rot_deg=float(params.get("rot_deg", 10.0)),
            translate_xy=float(params.get("translate_xy", 5.0)),
            scale_min=float(params.get("scale_min", 0.95)),
            scale_max=float(params.get("scale_max", 1.05)),
            allow_flip_w=bool(params.get("allow_flip_w", False)),
            allow_flip_h=bool(params.get("allow_flip_h", False)),
            allow_flip_d=bool(params.get("allow_flip_d", False)),
            p_gamma=float(params.get("p_gamma", 0.30)),
            gamma_min=float(params.get("gamma_min", 0.90)),
            gamma_max=float(params.get("gamma_max", 1.10)),
            p_noise=float(params.get("p_noise", 0.30)),
            noise_std=float(params.get("noise_std", 0.01)),
        )

    else:
        # Unknown preset => no augmentation
        return image_1dhw, mask_odhw
