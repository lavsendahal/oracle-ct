# radioprior_v2/datamodules/pleural_space.py
"""
Pleural Space Generation

Creates a pleural space mask by dilating lung masks.
The pleural space is where pleural effusion accumulates.

Anatomically, the pleural space is the thin cavity between:
- Visceral pleura (lung surface)
- Parietal pleura (chest wall)

We approximate this by:
1. Dilating lung masks by a few mm (e.g., 3-5mm)
2. Subtracting original lung mask
3. Result = thin shell around lungs = pleural space
"""

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from typing import Tuple


def create_pleural_space_mask(
    lung_mask: np.ndarray,
    spacing_mm: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    dilation_mm: float = 4.0,
) -> np.ndarray:
    """
    Create pleural space mask by dilating lungs.
    
    Args:
        lung_mask: Binary lung mask [D, H, W] (both lungs combined)
        spacing_mm: Voxel spacing in mm (x, y, z)
        dilation_mm: How many mm to dilate (typically 3-5mm for pleural space)
    
    Returns:
        pleural_space_mask: Binary mask [D, H, W] of pleural space
    """
    if lung_mask.sum() == 0:
        # No lungs - return empty mask
        return np.zeros_like(lung_mask, dtype=np.uint8)
    
    # Calculate dilation radius in voxels for each dimension
    dilation_voxels = [
        max(1, int(np.round(dilation_mm / spacing_mm[0]))),  # x
        max(1, int(np.round(dilation_mm / spacing_mm[1]))),  # y
        max(1, int(np.round(dilation_mm / spacing_mm[2]))),  # z
    ]
    
    # Create anisotropic structuring element (accounts for spacing)
    struct = np.zeros((
        2 * dilation_voxels[2] + 1,  # z
        2 * dilation_voxels[1] + 1,  # y
        2 * dilation_voxels[0] + 1,  # x
    ), dtype=bool)
    
    # Fill ellipsoid
    center = np.array([dilation_voxels[2], dilation_voxels[1], dilation_voxels[0]])
    for z in range(struct.shape[0]):
        for y in range(struct.shape[1]):
            for x in range(struct.shape[2]):
                pos = np.array([z, y, x])
                dist_normalized = np.sum(((pos - center) / dilation_voxels[::-1]) ** 2)
                if dist_normalized <= 1.0:
                    struct[z, y, x] = True
    
    # Dilate lung mask
    dilated_lung = binary_dilation(lung_mask > 0, structure=struct).astype(np.uint8)
    
    # Pleural space = dilated region - original lung
    pleural_space = dilated_lung - (lung_mask > 0).astype(np.uint8)
    
    return pleural_space


def create_pleural_space_from_bilateral_lungs(
    left_lung_mask: np.ndarray,
    right_lung_mask: np.ndarray,
    spacing_mm: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    dilation_mm: float = 4.0,
) -> np.ndarray:
    """
    Create pleural space from separate left and right lung masks.
    
    Args:
        left_lung_mask: Binary left lung mask [D, H, W]
        right_lung_mask: Binary right lung mask [D, H, W]
        spacing_mm: Voxel spacing in mm
        dilation_mm: Dilation distance in mm
    
    Returns:
        pleural_space_mask: Binary mask [D, H, W]
    """
    # Combine lungs
    combined_lungs = ((left_lung_mask > 0) | (right_lung_mask > 0)).astype(np.uint8)
    
    # Create pleural space
    return create_pleural_space_mask(combined_lungs, spacing_mm, dilation_mm)


def create_separate_pleural_spaces(
    left_lung_mask: np.ndarray,
    right_lung_mask: np.ndarray,
    spacing_mm: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    dilation_mm: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create separate left and right pleural space masks.
    
    Useful for detecting unilateral pleural effusion.
    
    Args:
        left_lung_mask: Binary left lung mask [D, H, W]
        right_lung_mask: Binary right lung mask [D, H, W]
        spacing_mm: Voxel spacing in mm
        dilation_mm: Dilation distance in mm
    
    Returns:
        (left_pleural_space, right_pleural_space): Tuple of binary masks
    """
    left_pleural = create_pleural_space_mask(left_lung_mask, spacing_mm, dilation_mm)
    right_pleural = create_pleural_space_mask(right_lung_mask, spacing_mm, dilation_mm)
    
    return left_pleural, right_pleural


if __name__ == "__main__":
    print("Pleural Space Generation")
    print("=" * 60)
    print("""
Pleural space is created by:
1. Dilating lung masks by 3-5mm (typical pleural thickness)
2. Subtracting original lung mask
3. Result = thin shell around lungs

This is where pleural effusion (fluid) accumulates!

Usage:
    from datamodules.pleural_space import create_pleural_space_mask
    
    # From combined lungs
    pleural_space = create_pleural_space_mask(
        lung_mask, 
        spacing_mm=(1.5, 1.5, 1.5),
        dilation_mm=4.0
    )
    
    # From left/right lungs separately
    pleural_space = create_pleural_space_from_bilateral_lungs(
        left_lung_mask,
        right_lung_mask,
        spacing_mm=(1.5, 1.5, 1.5),
        dilation_mm=4.0
    )
    
Typical dilation distances:
- 3mm: Minimal pleural space (normal)
- 4mm: Standard (captures most effusions)
- 5mm: Larger search area (may include some chest wall)
    """)
