#!/usr/bin/env python3
"""
ROI Utilities for Disease-Specific Region of Interest Handling

Handles coordinate transformation from ORIGINAL image space to RESAMPLED pack space.
Used by both:
1. Dataset (during training/inference) - returns dict for model
2. inspect_pack.py (for visualization) - returns binary mask

IMPORTANT: After nibabel.as_closest_canonical(), arrays are in [X, Y, Z] format
where X=LR (Left→Right), Y=PA (Posterior→Anterior), Z=IS (Inferior→Superior)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List


def transform_appendix_roi(
    case_id: str,
    features_df: pd.DataFrame,
    shape_orig: Tuple[int, int, int],
    shape_final: Tuple[int, int, int],
    spacing_final_mm: Tuple[float, float, float],
) -> Optional[Dict[str, Any]]:
    """
    Transform appendix ROI coordinates from ORIGINAL to RESAMPLED space.

    This function handles the coordinate transformation for appendix bounding boxes,
    converting from the original image resolution (where features were extracted)
    to the resampled resolution (used in packs for training).

    Args:
        case_id: Case identifier
        features_df: DataFrame with appendix features including voxel coordinates
        shape_orig: Original image shape [X, Y, Z] from pack metadata
        shape_final: Resampled image shape [X, Y, Z]
        spacing_final_mm: Final voxel spacing [sx, sy, sz] in mm

    Returns:
        Dictionary with ROI info for model:
        {
            "center_vox": [x, y, z],  # Center in RESAMPLED voxel space
            "box_mm": [sx, sy, sz]     # Box size in mm
        }
        Returns None if appendix coordinates not available for this case
    """
    # Find case in features
    case_row = features_df[features_df["case_id"] == case_id]
    if len(case_row) == 0:
        return None

    case_row = case_row.iloc[0]

    # Check if appendix voxel coordinates exist
    vox_cols = ["appendix_center_x_vox", "appendix_center_y_vox", "appendix_center_z_vox"]
    if not all(col in case_row.index for col in vox_cols):
        return None

    # Get VOXEL coordinates from parquet (in ORIGINAL RAS image voxel space)
    # RAS: X=Left→Right, Y=Posterior→Anterior, Z=Inferior→Superior
    center_x_vox_orig = case_row.get("appendix_center_x_vox", np.nan)
    center_y_vox_orig = case_row.get("appendix_center_y_vox", np.nan)
    center_z_vox_orig = case_row.get("appendix_center_z_vox", np.nan)

    if np.isnan(center_x_vox_orig) or np.isnan(center_y_vox_orig) or np.isnan(center_z_vox_orig):
        return None

    # Get box sizes in mm
    box_x_mm = case_row.get("appendix_box_size_x_mm", 70.0)
    box_y_mm = case_row.get("appendix_box_size_y_mm", 70.0)
    box_z_mm = case_row.get("appendix_box_size_z_mm", 60.0)

    # CRITICAL: After as_canonical(), nibabel returns arrays in [X, Y, Z] = [LR, PA, IS] format
    # shape_orig and shape_final are both in [X, Y, Z] format
    # This matches the coordinate system used in AppendixFeatureExtractor
    zoom_x = shape_final[0] / shape_orig[0]  # X axis
    zoom_y = shape_final[1] / shape_orig[1]  # Y axis
    zoom_z = shape_final[2] / shape_orig[2]  # Z axis

    # Transform coordinates from ORIGINAL voxel space to RESAMPLED voxel space
    center_x_vox_final = float(center_x_vox_orig * zoom_x)
    center_y_vox_final = float(center_y_vox_orig * zoom_y)
    center_z_vox_final = float(center_z_vox_orig * zoom_z)

    # CRITICAL: After dataset.py permutes tensors from [X,Y,Z] to [Z,Y,X],
    # we need to permute ROI coordinates to match!
    # Original ROI: [x, y, z] in pack file coordinates
    # After permute: need [z, y, x] to match tensor order [D, H, W]
    return {
        "center_vox": [center_z_vox_final, center_y_vox_final, center_x_vox_final],  # [z, y, x]
        "box_mm": [box_z_mm, box_y_mm, box_x_mm],  # [sz, sy, sx]
    }


def create_disease_rois_batch(
    case_ids: List[str],
    features_df: pd.DataFrame,
    metas: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Create disease_rois for a batch of cases.

    Currently only handles appendicitis ROI, but can be extended for other diseases.

    Args:
        case_ids: List of case IDs in the batch
        features_df: DataFrame with all features including ROI coordinates
        metas: List of metadata dicts from packs (contains shape_orig, shape_final, spacing_final_mm)

    Returns:
        List of disease_rois dicts, one per case:
        [
            {"appendicitis": {"center_vox": [x,y,z], "box_mm": [sx,sy,sz]}},
            {"appendicitis": {...}},
            ...
        ]
        Returns empty dict {} for cases without ROI data
    """
    disease_rois_batch = []

    for case_id, meta in zip(case_ids, metas):
        disease_rois = {}

        # Get shapes and spacing from pack metadata
        shape_orig = tuple(meta.get("shape_orig", [224, 224, 224]))
        shape_final = tuple(meta.get("shape_final", [224, 224, 224]))
        spacing_final_mm = tuple(meta.get("spacing_final_mm", [1.5, 1.5, 1.5]))

        # Transform appendix ROI (if available)
        appendix_roi = transform_appendix_roi(
            case_id=case_id,
            features_df=features_df,
            shape_orig=shape_orig,
            shape_final=shape_final,
            spacing_final_mm=spacing_final_mm,
        )

        if appendix_roi is not None:
            disease_rois["appendicitis"] = appendix_roi

        # TODO: Add other disease-specific ROIs here as needed
        # e.g., disease_rois["cholecystitis"] = transform_gallbladder_roi(...)

        disease_rois_batch.append(disease_rois)

    return disease_rois_batch


def create_appendix_bbox_mask(
    case_id: str,
    features_df: pd.DataFrame,
    shape_orig: Tuple[int, int, int],
    shape_final: Tuple[int, int, int],
    spacing_final_mm: Tuple[float, float, float],
) -> Optional[np.ndarray]:
    """
    Create appendix bounding box mask for visualization.

    This is a wrapper around transform_appendix_roi() that creates a binary mask
    instead of returning coordinate dict. Used by inspect_pack.py.

    Args:
        case_id: Case identifier
        features_df: DataFrame with appendix features
        shape_orig: Original image shape [X, Y, Z]
        shape_final: Resampled image shape [X, Y, Z]
        spacing_final_mm: Final voxel spacing [sx, sy, sz]

    Returns:
        [X, Y, Z] binary mask with appendix ROI box, or None if not available
    """
    # Get transformed ROI coordinates
    roi_info = transform_appendix_roi(
        case_id=case_id,
        features_df=features_df,
        shape_orig=shape_orig,
        shape_final=shape_final,
        spacing_final_mm=spacing_final_mm,
    )

    if roi_info is None:
        return None

    # Extract center and box size
    center_vox = roi_info["center_vox"]
    box_mm = roi_info["box_mm"]

    # Convert box size from mm to voxels in FINAL space
    box_x_vox = box_mm[0] / spacing_final_mm[0]
    box_y_vox = box_mm[1] / spacing_final_mm[1]
    box_z_vox = box_mm[2] / spacing_final_mm[2]

    # Compute bounds in FINAL space
    X_max, Y_max, Z_max = shape_final

    x0 = int(max(0, center_vox[0] - box_x_vox / 2))
    x1 = int(min(X_max, center_vox[0] + box_x_vox / 2))
    y0 = int(max(0, center_vox[1] - box_y_vox / 2))
    y1 = int(min(Y_max, center_vox[1] + box_y_vox / 2))
    z0 = int(max(0, center_vox[2] - box_z_vox / 2))
    z1 = int(min(Z_max, center_vox[2] + box_z_vox / 2))

    # Create binary mask in NumPy array format [X, Y, Z] = [LR, PA, IS]
    mask = np.zeros(shape_final, dtype=np.uint8)
    mask[x0:x1, y0:y1, z0:z1] = 1

    return mask
