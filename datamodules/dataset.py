# radioprior_v2/datamodules/dataset.py
"""
RadioPrior Dataset

Loads:
1. Pre-packed .pt files (image, masks, landmarks, disease_rois, meta)
2. Features from parquet (scalar radiomics features)  
3. Labels from CSV (binary disease labels)

Returns samples ready for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

from .augmentation import apply_augmentation
from .roi_utils import create_disease_rois_batch


class RadioPriorDataset(Dataset):
    """
    Dataset for RadioPrior neuro-symbolic model.

    Args:
        pack_root: Directory containing .pt pack files
        labels_csv: Path to labels CSV (columns: case_id, disease1, disease2, ...)
        case_ids: List of case IDs to use (required for train/val/test splits)
        features_parquet: Optional path to features parquet file (only for ScalarFusion model)
        feature_columns: Optional list of specific feature columns to use (filters parquet)
        disease_names: List of disease column names to use as labels
        transform: Optional transform for images
        cache_packs: Whether to cache loaded packs in memory
    """

    def __init__(
        self,
        pack_root: str,
        labels_csv: str,
        case_ids: List[str],
        features_parquet: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        disease_names: Optional[List[str]] = None,
        transform=None,
        cache_packs: bool = False,
        use_augmentation: bool = False,
        aug_preset: str = "anatomy_safe_v2",
        aug_params: Optional[Dict] = None,
    ):
        self.pack_root = Path(pack_root)
        self.transform = transform
        self.cache_packs = cache_packs
        self._cache = {}

        # Augmentation settings
        self.use_augmentation = use_augmentation
        self.aug_preset = aug_preset
        self.aug_params = aug_params or {}

        # Load features (optional - only for ScalarFusion model)
        self.use_features = features_parquet is not None
        if self.use_features:
            self.features_df = pd.read_parquet(features_parquet)

            # If specific feature columns are provided, use only those
            if feature_columns is not None:
                # Ensure case_id is included for indexing
                cols_to_keep = ["case_id"] + feature_columns
                available_cols = [c for c in cols_to_keep if c in self.features_df.columns]
                self.features_df = self.features_df[available_cols]
            else:
                # Remove non-numeric columns like 'success', 'error', etc.
                non_numeric_cols = []
                for col in self.features_df.columns:
                    if col == "case_id":
                        continue
                    try:
                        pd.to_numeric(self.features_df[col], errors='raise')
                    except (ValueError, TypeError):
                        non_numeric_cols.append(col)

                if non_numeric_cols:
                    print(f"Removing non-numeric columns: {non_numeric_cols}")
                    self.features_df = self.features_df.drop(columns=non_numeric_cols)

            self.features_df = self.features_df.set_index("case_id")
            self.num_features = len(self.features_df.columns)
            print(f"Loaded {self.num_features} numeric features")
        else:
            self.features_df = None
            self.num_features = 0
            print("No features loaded (GAP/MaskedAttn model)")

        # Load labels
        self.labels_df = pd.read_csv(labels_csv)
        self.labels_df = self.labels_df.set_index("case_id")

        # Determine disease names
        if disease_names is None:
            # Use all columns except case_id as disease labels
            disease_names = [c for c in self.labels_df.columns if c != "case_id"]
        self.disease_names = disease_names
        self.num_diseases = len(disease_names)

        # Use provided case_ids and verify they exist
        self.case_ids = []
        for case_id in case_ids:
            pack_path = self.pack_root / f"{case_id}.pt"
            has_pack = pack_path.exists()
            has_labels = case_id in self.labels_df.index
            has_features = (not self.use_features) or (case_id in self.features_df.index)

            if has_pack and has_labels and has_features:
                self.case_ids.append(case_id)

        print(f"RadioPriorDataset: {len(self.case_ids)} cases with packs and labels" +
              (f" and features" if self.use_features else ""))
    
    def __len__(self):
        return len(self.case_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case_id = self.case_ids[idx]

        # Load pack
        if self.cache_packs and case_id in self._cache:
            pack = self._cache[case_id]
        else:
            pack_path = self.pack_root / f"{case_id}.pt"
            # Use weights_only=True if possible, or use mmap for faster loading
            try:
                pack = torch.load(pack_path, weights_only=False, map_location='cpu', mmap=True)
            except:
                pack = torch.load(pack_path, weights_only=False, map_location='cpu')
            if self.cache_packs:
                self._cache[case_id] = pack
        
        # Get image and masks
        # CRITICAL FIX: Pack stores data as [C, X, Y, Z] from nibabel
        # But we need [C, Z, Y, X] for medical imaging (axial slices in depth dim)
        # Permute: [C, dim0, dim1, dim2] -> [C, dim2, dim1, dim0]
        image = pack["image"].permute(0, 3, 2, 1).contiguous()  # [1, X, Y, Z] -> [1, Z, Y, X]
        masks = pack["masks"].permute(0, 3, 2, 1).contiguous()  # [C, X, Y, Z] -> [C, Z, Y, X]
        # Now: image/masks are [C, D=160, H=224, W=224] where D is axial slices

        # Apply augmentation (only during training)
        if self.use_augmentation:
            image, masks = apply_augmentation(
                image,
                masks,
                preset=self.aug_preset,
                params=self.aug_params,
            )

        # Apply transform
        if self.transform is not None:
            image = self.transform(image)

        # Get features from parquet (REQUIRED - all features are here now!)
        if self.use_features:
            features_row = self.features_df.loc[case_id]
        else:
            # Even if not using features for model, load them for compatibility
            # (landmarks, disease_rois data is in parquet now)
            features_row = None

        # Get labels
        labels_row = self.labels_df.loc[case_id]
        labels = torch.tensor(
            [float(labels_row.get(d, 0)) for d in self.disease_names],
            dtype=torch.float32
        )

        # Permute metadata to match permuted tensors
        meta = pack["meta"].copy()
        spacing_orig = meta.get("spacing_final_mm", [1.5, 1.5, 3.0])
        # Original pack has spacing as [sx, sy, sz] matching [X, Y, Z]
        # After permute to [Z, Y, X], spacing should be [sz, sy, sx]
        meta["spacing_final_mm"] = [spacing_orig[2], spacing_orig[1], spacing_orig[0]]

        return {
            "image": image,
            "masks": masks,
            "features_row": features_row,  # Pass full parquet row (pandas Series)
            "labels": labels,
            "meta": meta,  # Updated metadata with permuted spacing
            "case_id": case_id,
            # NOTE: landmarks and disease_rois removed from .pt files
            # They're now accessed from features_row (parquet) if needed
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature column names."""
        if self.use_features:
            return list(self.features_df.columns)
        else:
            return []
    
    def get_disease_names(self) -> List[str]:
        """Get list of disease names."""
        return self.disease_names


def radioprior_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for RadioPrior dataset.

    Handles variable-size metadata while stacking tensors.
    Also creates disease_rois from features_row + metadata.
    """
    # Stack tensors
    images = torch.stack([b["image"] for b in batch])
    masks = torch.stack([b["masks"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])

    # Keep features_row as list (pandas Series - can't stack)
    features_rows = [b["features_row"] for b in batch]

    # Keep lists for non-tensor data
    meta = [b["meta"] for b in batch]
    case_ids = [b["case_id"] for b in batch]

    # Populate organ_touches_border in meta from features_row
    for i, (m, fr) in enumerate(zip(meta, features_rows)):
        if fr is not None and "organs" in m:
            organ_touches_border = {}
            for organ in m["organs"]:
                border_key = f"{organ}_touches_border"
                if border_key in fr.index:
                    organ_touches_border[organ] = bool(fr[border_key])
                else:
                    organ_touches_border[organ] = False  # Default: not touching
            m["organ_touches_border"] = organ_touches_border
        else:
            # No features available - default all to False
            m["organ_touches_border"] = {organ: False for organ in m.get("organs", [])}

    # Create disease_rois from features_row if available
    # This transforms coordinates from ORIGINAL to RESAMPLED space
    disease_rois = []
    if any(fr is not None for fr in features_rows):
        # Create a DataFrame from features_rows for batch processing
        # Filter out None values and create DataFrame
        valid_features = []
        valid_indices = []
        for i, fr in enumerate(features_rows):
            if fr is not None:
                valid_features.append(fr)
                valid_indices.append(i)

        if valid_features:
            features_df = pd.DataFrame(valid_features)
            # Make sure case_id is in the dataframe
            if "case_id" not in features_df.columns:
                features_df["case_id"] = [case_ids[i] for i in valid_indices]

            # Create disease_rois for all cases in batch
            disease_rois = create_disease_rois_batch(
                case_ids=case_ids,
                features_df=features_df,
                metas=meta,
            )
        else:
            # No valid features, return empty dicts
            disease_rois = [{}] * len(batch)
    else:
        # No features available, return empty dicts
        disease_rois = [{}] * len(batch)

    return {
        "image": images,            # [B, 1, X, Y, Z] in RAS orientation
        "masks": masks,             # [B, 20, X, Y, Z] in RAS orientation
        "features_row": features_rows,  # List[pd.Series] - parquet rows
        "labels": labels,           # [B, num_diseases]
        "meta": meta,               # List[dict] - minimal metadata
        "case_id": case_ids,        # List[str]
        "disease_rois": disease_rois,  # List[dict] - disease-specific ROIs
    }


def create_dataloader(
    pack_root: str,
    labels_csv: str,
    case_ids: List[str],
    features_parquet: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    disease_names: Optional[List[str]] = None,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a dataloader for RadioPrior dataset."""

    dataset = RadioPriorDataset(
        pack_root=pack_root,
        labels_csv=labels_csv,
        case_ids=case_ids,
        features_parquet=features_parquet,
        feature_columns=feature_columns,
        disease_names=disease_names,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=radioprior_collate_fn,
    )


if __name__ == "__main__":
    print("RadioPrior Dataset")
    print("=" * 50)
    print("""
    Usage:
        dataset = RadioPriorDataset(
            pack_root="/path/to/packs",
            features_parquet="/path/to/features.parquet",
            labels_csv="/path/to/labels.csv",
        )
        
        sample = dataset[0]
        # sample["image"]: [1, 224, 224, 224]
        # sample["masks"]: [14, 224, 224, 224]
        # sample["features"]: [~600]
        # sample["labels"]: [30]
        # sample["meta"]["organ_volume_ml"]: dict
        # sample["meta"]["organ_mean_hu"]: dict
        # sample["meta"]["body_volume_ml"]: float
    """)
