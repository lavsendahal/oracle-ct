# janus/datamodules/feature_bank.py
"""
Feature Bank for Janus

Handles:
1. Selection of scalar features per disease
2. Computation of derived features (ratios, differences)
3. Normalization (z-score, min-max)
4. Body-size normalization for volume features

Key derived features:
- liver_spleen_hu_diff: For hepatic steatosis (negative = fatty)
- liver_spleen_hu_ratio: For hepatic steatosis (< 1.0 = fatty)
- liver_volume_ratio: liver_volume / body_volume (body-size normalized)
- sbo_diameter_ratio: small_bowel_diam / colon_diam (>1.5 = SBO)
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from ..configs.disease_config import get_all_disease_configs, get_all_diseases


class FeatureBank:
    """
    Manages scalar features for neuro-symbolic fusion.
    
    Features come from two sources:
    1. Pre-computed: From features parquet (e.g., aorta_max_diameter_mm)
    2. Pack metadata: From .pt pack meta (e.g., organ_volume_ml, organ_mean_hu)
    
    Derived features are computed at runtime:
    - Volume ratios (organ_volume / body_volume)
    - HU differences (liver_hu - spleen_hu)
    - Diameter ratios (small_bowel_diam / colon_diam)
    """
    
    def __init__(
        self,
        stats_path: Optional[str] = None,
        normalize: str = "zscore",  # "zscore", "minmax", "none"
    ):
        """
        Args:
            stats_path: Path to JSON with feature statistics (mean, std, min, max)
            normalize: Normalization method
        """
        self.normalize_method = normalize
        self.stats: Dict[str, Dict[str, float]] = {}

        if normalize in ("zscore", "minmax"):
            if not stats_path:
                raise ValueError(
                    f"FeatureBank: normalize='{normalize}' requires a stats_path, but none was provided.\n"
                    f"Run scripts/compute_feature_stats.py on your training split and pass the output "
                    f"via paths.feature_stats in config.yaml."
                )
            if not Path(stats_path).exists():
                raise FileNotFoundError(
                    f"FeatureBank: stats file not found: {stats_path}\n"
                    f"Run scripts/compute_feature_stats.py on your training split to generate it."
                )
            with open(stats_path, "r") as f:
                self.stats = json.load(f)
    
    def compute_derived_features(
        self,
        meta: Dict[str, Any],
        features_row: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """
        Get features from parquet (preferred) or compute as fallback.

        Args:
            meta: Pack metadata (minimal - just spacing, shape, case_id)
            features_row: Row from features parquet (PRIMARY SOURCE)

        Returns:
            dict of feature values (mostly from parquet, some computed)
        """
        derived = {}

        # If no features_row, return empty (all features should be in parquet)
        if features_row is None:
            return derived

        # =====================================================================
        # DERIVED FEATURES - Only compute what's NOT in parquet
        # =====================================================================

        # NOTE: Most ratios/diffs are already in parquet! Only add what's missing.
        # Parquet already has:
        # - liver_to_body_ratio, spleen_to_body_ratio, etc. (volume ratios)
        # - liver_spleen_hu_diff (HU difference)
        # - sb_to_colon_diam_ratio (SBO ratio)

        # Cardiothoracic ratio - compute only if not in parquet
        if "cardiothoracic_ratio" not in features_row.index:
            # Compute from parquet volumes if available
            if "heart_vol_cc" in features_row.index and "lungs_vol_cc" in features_row.index:
                heart_vol = features_row.get("heart_vol_cc", np.nan)
                lungs_vol = features_row.get("lungs_vol_cc", np.nan)
                if not np.isnan(heart_vol) and not np.isnan(lungs_vol) and (heart_vol + lungs_vol) > 0:
                    derived["cardiothoracic_ratio"] = heart_vol / (heart_vol + lungs_vol)
                else:
                    derived["cardiothoracic_ratio"] = np.nan
            else:
                derived["cardiothoracic_ratio"] = np.nan
        else:
            derived["cardiothoracic_ratio"] = features_row["cardiothoracic_ratio"]
        
        # =====================================================================
        # HU DIFFERENCES AND RATIOS - Use parquet (already computed!)
        # =====================================================================

        # Liver-Spleen HU difference is already in parquet!
        if "liver_spleen_hu_diff" in features_row.index:
            derived["liver_spleen_hu_diff"] = features_row["liver_spleen_hu_diff"]
        else:
            derived["liver_spleen_hu_diff"] = np.nan

        # Liver-Spleen HU ratio - compute only if not in parquet
        if "liver_spleen_hu_ratio" not in features_row.index:
            # Compute from parquet HU values
            if "liver_mean_hu" in features_row.index and "spleen_mean_hu" in features_row.index:
                liver_hu = features_row.get("liver_mean_hu", np.nan)
                spleen_hu = features_row.get("spleen_mean_hu", np.nan)
                if not np.isnan(liver_hu) and not np.isnan(spleen_hu) and spleen_hu != 0:
                    derived["liver_spleen_hu_ratio"] = liver_hu / spleen_hu
                else:
                    derived["liver_spleen_hu_ratio"] = np.nan
            else:
                derived["liver_spleen_hu_ratio"] = np.nan
        else:
            derived["liver_spleen_hu_ratio"] = features_row["liver_spleen_hu_ratio"]
        
        # =====================================================================
        # BILATERAL FEATURES - Max of left/right
        # =====================================================================

        # Max renal pelvis diameter (bilateral max)
        left_rp = features_row.get("left_renal_pelvis_diameter_mm", np.nan)
        right_rp = features_row.get("right_renal_pelvis_diameter_mm", np.nan)

        if not np.isnan(left_rp) or not np.isnan(right_rp):
            derived["max_renal_pelvis_diameter_mm"] = np.nanmax([left_rp, right_rp])
        else:
            derived["max_renal_pelvis_diameter_mm"] = np.nan

        # =====================================================================
        # BINARY FLAGS - Compute only if not in parquet
        # =====================================================================

        # Gallbladder absent flag - use parquet volume
        if "gallbladder_volume_ml" in features_row.index or "gb_volume_cc" in features_row.index:
            gb_vol = features_row.get("gallbladder_volume_ml", features_row.get("gb_volume_cc", 0.0))
            derived["gallbladder_absent_flag"] = 1.0 if gb_vol < 1.0 else 0.0
        else:
            derived["gallbladder_absent_flag"] = np.nan
        
        return derived
    
    def get_features_for_disease(
        self,
        disease: str,
        meta: Dict[str, Any],
        features_row: Optional[pd.Series] = None,
        normalize: bool = True,
        cached_derived: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Get scalar features for a specific disease.

        Matches the scalar-only baseline used in `janus/train_ml_baseline.py` (Option A):
        - Z-score normalize using `feature_stats.json` mean/std (when available)
        - Missing values remain NaN through normalization, then are set to 0.0
          (mean-imputation in standardized space)

        Args:
            disease: Disease name
            meta: Pack metadata
            features_row: Row from features parquet
            normalize: Whether to apply normalization
            cached_derived: Pre-computed derived features (for efficiency)

        Returns:
            (features_tensor, feature_names)
        """
        disease_configs = get_all_disease_configs()
        if disease not in disease_configs:
            return torch.tensor([]), []

        config = disease_configs[disease]
        feature_names = config.scalar_features + config.derived_features

        if not feature_names:
            return torch.tensor([]), []

        # Use cached derived features if available (30x faster!)
        if cached_derived is not None:
            derived = cached_derived
        else:
            # Compute derived features
            derived = self.compute_derived_features(meta, features_row)

        # Collect feature values (VALUE only; no presence indicators)
        values = []
        valid_names = []

        for name in feature_names:
            value = None
            raw_value = None  # Keep raw value for presence detection

            # Priority 1: Check parquet features (primary source)
            if features_row is not None and name in features_row.index:
                raw_value = features_row[name]
                if pd.isna(raw_value):
                    value = None
                else:
                    value = raw_value

            # Priority 2: Check derived/computed features
            elif name in derived:
                raw_value = derived[name]
                value = raw_value

            # Priority 3: Fallback to None (feature missing)
            else:
                value = None
                raw_value = None

            # Missing values: keep NaN through normalization, then set to 0.0
            if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
                value = np.nan
            else:
                value = float(raw_value)

            # Normalize (z-score). If value is NaN, keep it NaN here.
            if normalize and self.normalize_method != "none" and name in self.stats:
                value = self._normalize_value(value, name)

            # Mean-impute in standardized space (NaN -> 0.0)
            if isinstance(value, float) and np.isnan(value):
                value = 0.0

            values.append(float(value))
            valid_names.append(name)

        return torch.tensor(values, dtype=torch.float32), valid_names
    
    def _normalize_value(self, value: float, feature_name: str) -> float:
        """Normalize a single feature value."""
        if feature_name not in self.stats:
            return value
        
        stats = self.stats[feature_name]
        
        if self.normalize_method == "zscore":
            mean = stats.get("mean", 0.0)
            std = stats.get("std", 1.0)
            if std == 0:
                std = 1.0
            return (value - mean) / std
        
        elif self.normalize_method == "minmax":
            min_val = stats.get("min", 0.0)
            max_val = stats.get("max", 1.0)
            if max_val == min_val:
                return 0.0
            return (value - min_val) / (max_val - min_val)
        
        return value
    
    def get_all_features(
        self,
        meta: Dict[str, Any],
        features_row: Optional[pd.Series] = None,
        normalize: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Get features for all diseases.
        
        Returns:
            dict: {disease: features_tensor}
        """
        all_features = {}
        all_diseases = get_all_diseases()

        for disease in all_diseases:
            features, _ = self.get_features_for_disease(
                disease, meta, features_row, normalize
            )
            all_features[disease] = features
        
        return all_features


def compute_feature_statistics(
    features_parquet: str,
    output_path: str,
    train_ids_file: Optional[str] = None,
    case_ids: Optional[List[str]] = None,
):
    """
    Compute feature statistics (mean, std, min, max) directly from parquet.

    NO PACK LOADING NEEDED - Everything is in the parquet!

    Args:
        features_parquet: Path to features parquet
        output_path: Path to save statistics JSON
        train_ids_file: Path to train_ids.txt (only use training data)
        case_ids: Alternative to train_ids_file - provide list of case IDs directly
    """
    from tqdm import tqdm

    print("=" * 80)
    print("Computing Feature Statistics from Parquet (No Packs Needed)")
    print("=" * 80)

    # Load case IDs
    if train_ids_file is not None:
        with open(train_ids_file, 'r') as f:
            case_ids = [line.strip() for line in f if line.strip()]
        print(f"\n✓ Loaded {len(case_ids)} case IDs from {train_ids_file}")
    elif case_ids is not None:
        print(f"\n✓ Using {len(case_ids)} provided case IDs")
    else:
        case_ids = None  # Use all cases
        print(f"\n✓ Using all cases in parquet")

    # Load features parquet
    features_df = pd.read_parquet(features_parquet)

    if 'case_id' in features_df.columns:
        features_df = features_df.set_index("case_id")

    print(f"✓ Loaded parquet: {features_df.shape[0]} cases, {features_df.shape[1]} features")

    # Filter to specified case IDs
    if case_ids is not None:
        case_ids_in_parquet = [cid for cid in case_ids if cid in features_df.index]
        features_df = features_df.loc[case_ids_in_parquet]
        print(f"✓ Filtered to {len(case_ids_in_parquet)} cases")

    # Compute statistics for all numeric columns
    # IMPORTANT: Match sklearn pipeline used in LR baseline:
    #   1) median imputation per feature
    #   2) StandardScaler fit on imputed values (mean/std with ddof=0)
    print(f"\n✓ Computing statistics (median-impute then zscore stats)...")

    stats = {}

    for col in tqdm(features_df.columns, desc="Processing features"):
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(features_df[col]):
            continue

        series = features_df[col]

        # Skip columns with no observed values at all
        non_nan = series.dropna().values
        if len(non_nan) == 0:
            continue

        median = float(np.median(non_nan))
        imputed = series.fillna(median).values

        stats[col] = {
            "median": median,
            "mean": float(np.mean(imputed)),
            "std": float(np.std(imputed)),
            "min": float(np.min(imputed)),
            "max": float(np.max(imputed)),
            "count": int(len(imputed)),
            "count_non_nan": int(len(non_nan)),
        }

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Saved statistics for {len(stats)} features to {output_path}")
    print("=" * 80)

    return stats


if __name__ == "__main__":
    print("Feature Bank for Janus")
    print("=" * 60)
    print("""
Key Derived Features:
---------------------
1. Volume Ratios (body-size normalized):
   - liver_volume_ratio = liver_volume / body_volume
   - spleen_volume_ratio = spleen_volume / body_volume
   - cardiothoracic_ratio = heart / (heart + lungs)

2. HU Comparisons (for hepatic steatosis):
   - liver_spleen_hu_diff = liver_hu - spleen_hu (negative = fatty)
   - liver_spleen_hu_ratio = liver_hu / spleen_hu (< 1.0 = fatty)

3. Diameter Ratios (for bowel obstruction):
   - sbo_diameter_ratio = small_bowel_diam / colon_diam (>1.5 = SBO)

Usage:
------
    feature_bank = FeatureBank(stats_path="feature_stats.json")
    
    # Get features for a specific disease
    features, names = feature_bank.get_features_for_disease(
        "hepatic_steatosis", meta, features_row
    )
    # features: tensor([liver_mean_hu_zscore, spleen_mean_hu_zscore, 
    #                   liver_spleen_hu_diff_zscore, liver_spleen_hu_ratio_zscore])
    """)
