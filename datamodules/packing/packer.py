# radioprior_v2/datamodules/packing/packer.py
"""
CT Pack Builder for RadioPrior Deep Learning Pipeline

This module creates standardized .pt pack files from raw NIFTI data:
- CT image (resampled to isotropic resolution)
- Merged organ masks (14 channels for radioprior_v1)
- Anatomical landmarks (computed from segmentation)
- Disease ROI coordinates (from pre-extracted features)
- Metadata (spacing, volumes, border flags, body volume, organ mean HU)

PACK STRUCTURE:
    {
        "image": Tensor[1, D, H, W],         # float16, normalized [0,1]
        "masks": Tensor[14, D, H, W],        # uint8, one-hot per organ
        "landmarks": Dict[str, Any],         # Computed from segmentation
        "disease_rois": Dict[str, Dict],     # From features CSV
        "meta": {
            "case_id": str,
            "organs": List[str],             # Channel order
            "spacing_orig_mm": Tuple[float, 3],
            "spacing_final_mm": Tuple[float, 3],
            "shape_orig": Tuple[int, 3],
            "shape_final": Tuple[int, 3],
            "zoom_factors": List[float],
            "hu_range": Tuple[float, 2],
            "has_kidney_cysts": bool,
            "body_volume_ml": float,         # For body-size normalization
            "organ_volume_ml": Dict[str, float],
            "organ_mean_hu": Dict[str, float],  # For HU-based disease detection
            "organ_touches_border": Dict[str, bool],
        }
    }

USAGE:
    from datamodules.packing.packer import PackBuilder, PackConfig

    config = PackConfig(
        target_spacing=(1.5, 1.5, 1.5),
        target_shape=(224, 224, 224),
        merge_name="radioprior_v1",
        hu_min=-1000.0,
        hu_max=1000.0,
    )

    builder = PackBuilder(
        config=config,
        features_df=pd.read_parquet("features.parquet"),
    )

    pack = builder.build(
        case_id="AC123abc",
        image_path="/path/to/ct.nii.gz",
        seg_path="/path/to/seg.nii.gz",
        body_mask_path="/path/to/body.nii.gz",  # Optional
        kidney_cyst_path="/path/to/cysts.nii.gz",  # Optional
    )

    torch.save(pack, "AC123abc.pt")

BACKWARDS COMPATIBILITY:
    # Legacy RadioPriorPacker API is also available
    from datamodules.packing.packer import RadioPriorPacker

    packer = RadioPriorPacker(
        target_spacing=(1.5, 1.5, 1.5),
        target_shape=(224, 224, 224),
        hu_range=(-1000, 1000),
    )
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import torch

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    from scipy.ndimage import zoom
except ImportError:
    zoom = None

from ..class_map import (
    get_organ_id_map,
    merge_kidney_cysts_to_masks,
    compute_pleural_space_mask,
    compute_all_dilated_spaces,
    RADIOPRIOR_V1_CHANNEL_LIST,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PackConfig:
    """Configuration for pack building."""

    # Resampling
    target_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5)
    target_shape: Optional[Tuple[int, int, int]] = (224, 224, 224)

    # HU clipping and normalization
    hu_min: float = -1000.0
    hu_max: float = 1000.0

    # Organ merging
    merge_name: str = "radioprior_v1"
    organs: List[str] = field(default_factory=lambda: RADIOPRIOR_V1_CHANNEL_LIST.copy())

    # Output precision
    image_dtype: str = "float16"  # float16 or float32
    mask_dtype: str = "uint8"

    # Body masking
    apply_body_mask: bool = True
    body_mask_hu_threshold: float = -500.0  # For auto-generating body mask

    # Validation
    min_organ_volume_ml: float = 1.0  # Warn if organ smaller than this


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_nifti_as_ras(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load NIFTI and convert to RAS orientation."""
    if nib is None:
        raise ImportError("nibabel required: pip install nibabel")

    nii = nib.load(str(path))
    nii_ras = nib.as_closest_canonical(nii)
    image = nii_ras.get_fdata()
    affine = nii_ras.affine
    spacing = np.abs(np.diag(affine)[:3])
    return image, affine, spacing


def resample_volume(
    volume: np.ndarray,
    source_spacing: Tuple[float, float, float],
    target_spacing: Optional[Tuple[float, float, float]] = None,
    target_shape: Optional[Tuple[int, int, int]] = None,
    order: int = 1,
) -> np.ndarray:
    """
    Resample 3D volume.

    Args:
        volume: Input volume
        source_spacing: Original spacing in mm
        target_spacing: Target spacing in mm (mutually exclusive with target_shape)
        target_shape: Target shape in voxels (mutually exclusive with target_spacing)
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)

    Returns:
        Resampled volume
    """
    if zoom is None:
        raise ImportError("scipy required: pip install scipy")

    if target_shape is not None:
        # Resize to fixed shape
        zoom_factors = np.array(target_shape) / np.array(volume.shape)
    elif target_spacing is not None:
        # Resample to target spacing
        zoom_factors = np.array(source_spacing) / np.array(target_spacing)
    else:
        raise ValueError("Must specify either target_spacing or target_shape")

    if np.allclose(zoom_factors, 1.0, atol=0.01):
        return volume

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return zoom(volume, zoom_factors, order=order, mode="nearest")


def compute_zoom_factors(
    original_shape: Tuple[int, int, int],
    target_shape: Tuple[int, int, int],
) -> np.ndarray:
    """Compute zoom factors from original to target shape."""
    return np.array(target_shape) / np.array(original_shape)


def clip_and_normalize_hu(
    image: np.ndarray,
    hu_min: float = -1000.0,
    hu_max: float = 1000.0,
) -> np.ndarray:
    """Clip HU values and normalize to [0, 1]."""
    clipped = np.clip(image, hu_min, hu_max)
    return ((clipped - hu_min) / (hu_max - hu_min)).astype(np.float32)


def apply_body_mask(
    image: np.ndarray,
    body_mask: np.ndarray,
    background_value: float = 0.0,
) -> np.ndarray:
    """Apply body mask to image (set outside to background value)."""
    masked = image.copy()
    masked[body_mask == 0] = background_value
    return masked


def compute_body_volume_ml(
    body_mask: np.ndarray,
    spacing_mm: Tuple[float, float, float],
) -> float:
    """Compute body volume from body mask in ml."""
    voxel_volume_ml = (spacing_mm[0] * spacing_mm[1] * spacing_mm[2]) / 1000.0
    return float((body_mask > 0).sum() * voxel_volume_ml)


def compute_organ_mean_hu(
    image_normalized: np.ndarray,
    masks: np.ndarray,
    organ_names: List[str],
    hu_min: float,
    hu_max: float,
) -> Dict[str, Optional[float]]:
    """
    Compute mean HU for each organ (for liver-spleen comparison, etc.).

    Args:
        image_normalized: Normalized image [0, 1]
        masks: Organ masks [num_organs, D, H, W]
        organ_names: List of organ names corresponding to mask channels
        hu_min: Minimum HU value used for normalization
        hu_max: Maximum HU value used for normalization

    Returns:
        Dictionary mapping organ name to mean HU value (None if organ missing)
    """
    organ_mean_hu = {}

    for idx, organ_name in enumerate(organ_names):
        mask = masks[idx] > 0

        if mask.sum() == 0:
            organ_mean_hu[organ_name] = None
            continue

        mean_normalized = image_normalized[mask].mean()
        mean_hu = mean_normalized * (hu_max - hu_min) + hu_min
        organ_mean_hu[organ_name] = float(mean_hu)

    return organ_mean_hu


def load_disease_rois_from_features(
    case_id: str,
    features_df: Optional["pd.DataFrame"],
    original_shape: Tuple[int, int, int],
    original_spacing: Tuple[float, float, float],
    target_shape: Tuple[int, int, int],
) -> Dict[str, Dict[str, Any]]:
    """
    Load disease ROI coordinates with proper coordinate conversion.

    Currently handles:
    - Appendicitis: center coordinates from appendix feature extractor
    - Aortic Aneurysm (AAA): z-location of max diameter

    Coordinates are converted from original mm space to resampled voxel space.
    """
    disease_rois = {}

    if features_df is None:
        return disease_rois

    # Find case in features
    case_row = features_df[features_df["case_id"] == case_id]
    if len(case_row) == 0:
        return disease_rois

    case_row = case_row.iloc[0]
    zoom_factors = compute_zoom_factors(original_shape, target_shape)

    # =========================================================================
    # APPENDICITIS ROI
    # =========================================================================
    appendix_cols = ["appendix_center_x_mm", "appendix_center_y_mm", "appendix_center_z_mm"]
    if all(col in case_row.index for col in appendix_cols):
        x_mm = case_row.get("appendix_center_x_mm", np.nan)
        y_mm = case_row.get("appendix_center_y_mm", np.nan)
        z_mm = case_row.get("appendix_center_z_mm", np.nan)

        if not (np.isnan(x_mm) or np.isnan(y_mm) or np.isnan(z_mm)):
            center_vox = [
                (x_mm / original_spacing[0]) * zoom_factors[0],
                (y_mm / original_spacing[1]) * zoom_factors[1],
                (z_mm / original_spacing[2]) * zoom_factors[2],
            ]

            disease_rois["appendicitis"] = {
                "center_vox": center_vox,
                "center_mm_original": [float(x_mm), float(y_mm), float(z_mm)],
                "box_mm": [
                    float(case_row.get("appendix_box_size_x_mm", 70.0)),
                    float(case_row.get("appendix_box_size_y_mm", 70.0)),
                    float(case_row.get("appendix_box_size_z_mm", 60.0)),
                ],
                "confidence": float(case_row.get("appendix_loc_confidence", 0.5)),
            }

    # =========================================================================
    # AORTIC ANEURYSM (AAA) ROI
    # =========================================================================
    if "max_diam_location_z_mm" in case_row.index:
        z_mm = case_row.get("max_diam_location_z_mm", np.nan)
        max_diam = case_row.get("aorta_max_diameter_mm", np.nan)

        if not np.isnan(z_mm) and not np.isnan(max_diam):
            z_vox = (z_mm / original_spacing[2]) * zoom_factors[2]
            disease_rois["aortic_aneurysm"] = {
                "z_mm_original": float(z_mm),
                "z_vox": float(z_vox),
                "max_diameter_mm": float(max_diam),
            }

    return disease_rois


# =============================================================================
# PACK BUILDER
# =============================================================================

class PackBuilder:
    """
    Build standardized CT packs for deep learning.

    The builder handles:
    1. Loading NIFTI files (image, segmentation, optional body mask, optional kidney cysts)
    2. Resampling to isotropic resolution or fixed shape
    3. HU clipping and normalization
    4. Merging raw labels into organ channels
    5. Computing landmarks from segmentation
    6. Computing body volume and organ mean HU values
    7. Loading disease ROI coordinates from features
    8. Saving as .pt file
    """

    def __init__(
        self,
        config: Optional[PackConfig] = None,
        features_df: Optional["pd.DataFrame"] = None,
    ):
        """
        Initialize the pack builder.

        Args:
            config: Pack configuration (uses defaults if None)
            features_df: DataFrame with pre-extracted features (for disease ROIs)
                        Must have 'case_id' column and ROI coordinate columns
        """
        if nib is None:
            raise ImportError("nibabel required: pip install nibabel")
        if zoom is None:
            raise ImportError("scipy required: pip install scipy")

        self.config = config or PackConfig()
        self.features_df = features_df

        # Build organ ID mapping
        self.organ_id_map = get_organ_id_map(
            scheme="totalseg",
            merge_name=self.config.merge_name,
            organs=self.config.organs,
        )

        # Index features by case_id for fast lookup
        if features_df is not None:
            self._features_index = features_df.set_index("case_id")
        else:
            self._features_index = None

    def build(
        self,
        case_id: str,
        image_path: Union[str, Path],
        seg_path: Union[str, Path],
        body_mask_path: Optional[Union[str, Path]] = None,
        kidney_cyst_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Build a pack for a single case.

        Args:
            case_id: Unique identifier for this case
            image_path: Path to CT image NIFTI
            seg_path: Path to TotalSegmentator segmentation NIFTI
            body_mask_path: Optional path to body mask NIFTI
            kidney_cyst_path: Optional path to kidney cyst segmentation NIFTI

        Returns:
            Dictionary containing image, masks, landmarks, disease_rois, meta
        """
        # Load NIFTI files
        image, _, orig_spacing = load_nifti_as_ras(image_path)
        seg, _, seg_spacing = load_nifti_as_ras(seg_path)
        original_shape = image.shape

        # Load optional body mask
        body_mask = None
        if body_mask_path is not None and Path(body_mask_path).exists():
            body_mask, _, _ = load_nifti_as_ras(body_mask_path)
        elif self.config.apply_body_mask:
            # Auto-generate from HU threshold
            body_mask = image > self.config.body_mask_hu_threshold

        # Load optional kidney cyst segmentation
        kidney_cyst_seg = None
        kidney_cyst_spacing = None
        if kidney_cyst_path is not None and Path(kidney_cyst_path).exists():
            kidney_cyst_seg, _, kidney_cyst_spacing = load_nifti_as_ras(kidney_cyst_path)

        # Step 1: Resample to target spacing (shape varies per case based on FOV)
        image_after_spacing = resample_volume(
            image,
            orig_spacing,
            target_spacing=self.config.target_spacing,
            target_shape=None,  # Only resample to target spacing
            order=1,  # Linear interpolation
        )
        shape_after_resampling = image_after_spacing.shape  # Capture intermediate shape

        # Step 2: Resize to target shape (crop/pad to fixed size)
        if self.config.target_shape is not None:
            image_resampled = resample_volume(
                image_after_spacing,
                self.config.target_spacing,
                target_shape=self.config.target_shape,
                order=1,
            )
        else:
            image_resampled = image_after_spacing

        # Resample segmentation (same two-step process)
        seg_after_spacing = resample_volume(
            seg,
            seg_spacing,
            target_spacing=self.config.target_spacing,
            target_shape=None,
            order=0,  # Nearest neighbor for labels
        )
        if self.config.target_shape is not None:
            seg_resampled = resample_volume(
                seg_after_spacing,
                self.config.target_spacing,
                target_shape=self.config.target_shape,
                order=0,
            ).astype(np.int16)
        else:
            seg_resampled = seg_after_spacing.astype(np.int16)

        # Resample body mask
        body_mask_resampled = None
        if body_mask is not None:
            body_after_spacing = resample_volume(
                body_mask,
                orig_spacing,
                target_spacing=self.config.target_spacing,
                target_shape=None,
                order=0,
            )
            if self.config.target_shape is not None:
                body_mask_resampled = resample_volume(
                    body_after_spacing,
                    self.config.target_spacing,
                    target_shape=self.config.target_shape,
                    order=0,
                )
            else:
                body_mask_resampled = body_after_spacing

        # Resample kidney cyst segmentation
        kidney_cyst_resampled = None
        if kidney_cyst_seg is not None:
            cyst_after_spacing = resample_volume(
                kidney_cyst_seg,
                kidney_cyst_spacing,
                target_spacing=self.config.target_spacing,
                target_shape=None,
                order=0,
            )
            if self.config.target_shape is not None:
                kidney_cyst_resampled = resample_volume(
                    cyst_after_spacing,
                    self.config.target_spacing,
                    target_shape=self.config.target_shape,
                    order=0,
                ).astype(np.int16)
            else:
                kidney_cyst_resampled = cyst_after_spacing.astype(np.int16)

        # Normalize image
        image_normalized = clip_and_normalize_hu(
            image_resampled,
            self.config.hu_min,
            self.config.hu_max,
        )

        # Apply body mask to normalized image
        if body_mask_resampled is not None:
            image_normalized = apply_body_mask(image_normalized, body_mask_resampled, 0.0)

        # Merge segmentation into organ channels
        masks = self._merge_segmentation(seg_resampled)

        # Merge kidney cysts into masks if available
        if kidney_cyst_resampled is not None:
            masks = merge_kidney_cysts_to_masks(masks, kidney_cyst_resampled)

        # Compute pleural space mask from lungs
        masks = compute_pleural_space_mask(
            masks,
            spacing_mm=self.config.target_spacing,
            dilation_mm=4.0
        )

        # Compute all dilated spaces (periportal, perivascular, pericardial, subcutaneous)
        masks = compute_all_dilated_spaces(
            masks,
            spacing_mm=self.config.target_spacing,
            body_mask=body_mask_resampled
        )

        # Compute zoom factors (needed for coordinate transformations)
        zoom_factors = compute_zoom_factors(original_shape, self.config.target_shape or image_resampled.shape)

        # NOTE: We no longer compute landmarks, organ_volumes, organ_mean_hu, disease_rois here
        # All features are now in parquet file - no redundancy!

        # Convert to tensors
        image_dtype = torch.float16 if self.config.image_dtype == "float16" else torch.float32
        mask_dtype = torch.uint8 if self.config.mask_dtype == "uint8" else torch.int16

        image_tensor = torch.from_numpy(image_normalized).unsqueeze(0).to(image_dtype)
        masks_tensor = torch.from_numpy(masks).to(mask_dtype)

        # Build minimal metadata (only what's needed for image processing)
        meta = {
            "case_id": case_id,
            "organs": self.config.organs,
            "spacing_orig_mm": list(orig_spacing),
            "spacing_final_mm": list(self.config.target_spacing),
            "shape_orig": list(original_shape),
            "shape_after_resampling": list(shape_after_resampling),  # Shape after spacing resample, before resize
            "shape_final": list(image_tensor.shape[1:]),
            "zoom_factors": list(zoom_factors),
            "hu_range": [self.config.hu_min, self.config.hu_max],
        }

        return {
            "image": image_tensor,
            "masks": masks_tensor,
            "meta": meta,
            # NOTE: landmarks, disease_rois, organ_volumes, organ_mean_hu are all in parquet!
        }

    def _merge_segmentation(self, seg: np.ndarray) -> np.ndarray:
        """
        Merge raw segmentation labels into organ channels.

        Args:
            seg: Raw segmentation with TotalSegmentator label IDs

        Returns:
            Array of shape (num_organs, D, H, W) with binary masks
        """
        num_organs = len(self.config.organs)
        masks = np.zeros((num_organs, *seg.shape), dtype=np.uint8)

        # Build reverse mapping: organ_name -> channel_index
        organ_to_channel = {name: i for i, name in enumerate(self.config.organs)}

        # For each raw label, add to the appropriate channel
        for label_id, organ_name in self.organ_id_map.items():
            channel_idx = organ_to_channel.get(organ_name)
            if channel_idx is not None:
                masks[channel_idx] |= (seg == label_id).astype(np.uint8)

        return masks

    def validate_pack(self, pack: Dict[str, Any]) -> List[str]:
        """
        Validate a built pack and return list of warnings.
        """
        warnings = []

        # Check image shape
        image_shape = pack["image"].shape
        if image_shape[0] != 1:
            warnings.append(f"Image has {image_shape[0]} channels, expected 1")

        # Check masks shape
        masks_shape = pack["masks"].shape
        expected_channels = len(self.config.organs)
        if masks_shape[0] != expected_channels:
            warnings.append(f"Masks have {masks_shape[0]} channels, expected {expected_channels}")

        # Check spatial dimensions match
        if image_shape[1:] != masks_shape[1:]:
            warnings.append(f"Image shape {image_shape[1:]} != masks shape {masks_shape[1:]}")

        # Check for empty masks
        for i, organ in enumerate(self.config.organs):
            if pack["masks"][i].sum() == 0:
                warnings.append(f"Empty mask for organ: {organ}")

        # Note: organ volume validation removed - get from parquet instead

        return warnings

    def save_pack(self, pack: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """Save pack to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pack, output_path)


# =============================================================================
# BACKWARDS COMPATIBILITY - Legacy RadioPriorPacker API
# =============================================================================

class RadioPriorPacker:
    """
    Legacy pack builder with simplified API.

    This class provides backwards compatibility with the old RadioPriorPacker API.
    Internally, it uses the new PackBuilder implementation.

    DEPRECATED: Use PackBuilder with PackConfig instead for new code.
    """

    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
        target_shape: Tuple[int, int, int] = (224, 224, 224),
        hu_range: Tuple[float, float] = (-1000, 1000),
        image_dtype: str = "float16",
        mask_dtype: str = "uint8",
    ):
        """
        Initialize legacy packer.

        Args:
            target_spacing: Target voxel spacing in mm
            target_shape: Target volume shape
            hu_range: (min, max) HU values for clipping
            image_dtype: Image tensor dtype (float16 or float32)
            mask_dtype: Mask tensor dtype (uint8)
        """
        config = PackConfig(
            target_spacing=target_spacing,
            target_shape=target_shape,
            hu_min=hu_range[0],
            hu_max=hu_range[1],
            image_dtype=image_dtype,
            mask_dtype=mask_dtype,
        )
        self._builder = PackBuilder(config=config)

    def build_pack(
        self,
        image_path: Union[str, Path],
        seg_path: Union[str, Path],
        case_id: str,
        features_df: Optional["pd.DataFrame"] = None,
        body_mask_path: Optional[Union[str, Path]] = None,
        kidney_cyst_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Build pack with legacy API."""
        # Update features_df if provided
        if features_df is not None:
            self._builder.features_df = features_df
            if features_df is not None:
                self._builder._features_index = features_df.set_index("case_id")

        return self._builder.build(
            case_id=case_id,
            image_path=image_path,
            seg_path=seg_path,
            body_mask_path=body_mask_path,
            kidney_cyst_path=kidney_cyst_path,
        )

    def save_pack(self, pack: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """Save pack to disk."""
        self._builder.save_pack(pack, output_path)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def build_pack(
    case_id: str,
    image_path: Union[str, Path],
    seg_path: Union[str, Path],
    output_path: Union[str, Path],
    body_mask_path: Optional[Union[str, Path]] = None,
    kidney_cyst_path: Optional[Union[str, Path]] = None,
    features_df: Optional["pd.DataFrame"] = None,
    config: Optional[PackConfig] = None,
) -> Dict[str, Any]:
    """
    Convenience function to build and save a pack in one call.

    Returns the pack dict for inspection.
    """
    builder = PackBuilder(config=config, features_df=features_df)
    pack = builder.build(
        case_id=case_id,
        image_path=image_path,
        seg_path=seg_path,
        body_mask_path=body_mask_path,
        kidney_cyst_path=kidney_cyst_path,
    )

    # Validate
    warnings_list = builder.validate_pack(pack)
    if warnings_list:
        print(f"Warnings for {case_id}:")
        for w in warnings_list:
            print(f"  - {w}")

    # Save
    torch.save(pack, str(output_path))

    return pack


def load_pack(pack_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a pack from disk."""
    return torch.load(pack_path, weights_only=False)


def estimate_pack_size_mb(
    shape: Tuple[int, int, int],
    num_organs: int = 14,
) -> float:
    """
    Estimate pack file size in MB.

    Args:
        shape: Volume shape (D, H, W)
        num_organs: Number of organ channels

    Returns:
        Estimated size in MB
    """
    voxels = shape[0] * shape[1] * shape[2]

    # Image: float16 = 2 bytes per voxel
    image_bytes = voxels * 2

    # Masks: uint8 = 1 byte per voxel per channel
    mask_bytes = voxels * num_organs

    # Metadata, landmarks, etc. ~10KB
    overhead = 10 * 1024

    total_bytes = image_bytes + mask_bytes + overhead
    return total_bytes / (1024 * 1024)


if __name__ == "__main__":
    # Test configuration
    config = PackConfig()
    print("Default PackConfig:")
    print(f"  Target spacing: {config.target_spacing}")
    print(f"  Target shape: {config.target_shape}")
    print(f"  HU range: ({config.hu_min}, {config.hu_max})")
    print(f"  Merge profile: {config.merge_name}")
    print(f"  Organs ({len(config.organs)}): {config.organs[:5]}...")

    # Estimate pack size for typical volume
    shape = (224, 224, 224)
    size = estimate_pack_size_mb(shape)
    print(f"\nEstimated pack size for {shape}: {size:.1f} MB")
