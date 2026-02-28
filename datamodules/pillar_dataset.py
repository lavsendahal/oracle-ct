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
PillarDataset — Dataset for training Pillar-0 backbone variants.

Loads:
  1. RAVE LZ4 packs at 384³ 1.5mm isotropic (raw int16 HU) → applies 11 HU windows
  2. Existing .pt mask packs at original 224×224×160 resolution (kept for attention)
  3. Labels from CSV
  4. Scalar features from parquet (optional, for MaskedAttnScalar variant)

Returns:
  image:        [11, 384, 384, 384] float32  (10 CT anatomical windows + minmax, in [0,1])
  masks:        [20, D, H, W] float32        (original resolution, used for organ attention)
  labels:       [num_diseases] float32
  meta:         dict
  case_id:      str
  features_row: pandas.Series or None

The 11 input channels match the pretrained Pillar-0 AbdomenCT first conv:
  Conv3d(11, 64, kernel_size=3) — weights are kept fully intact (Option B).

Window order (matches Pillar-0 training):
  [0] lung          center=-600, width=1500
  [1] mediastinum   center=50,   width=400
  [2] abdomen       center=40,   width=400
  [3] liver         center=80,   width=150
  [4] bone          center=400,  width=1800
  [5] brain         center=40,   width=80
  [6] subdural      center=75,   width=215
  [7] stroke        center=40,   width=40
  [8] temporal_bone center=600,  width=2800
  [9] soft_tissue   center=50,   width=350
 [10] minmax        global min-max normalisation
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Add RAVE to sys.path so rve can be imported
# ---------------------------------------------------------------------------
_RAVE_PATH = Path(__file__).resolve().parent.parent.parent / "rave"
if _RAVE_PATH.exists() and str(_RAVE_PATH) not in sys.path:
    sys.path.insert(0, str(_RAVE_PATH))

import rve  # noqa: E402 — load_sample, batch_apply_windowing_vectorized, ANATOMICAL_WINDOWS

# CT window names in the order Pillar-0 was trained with
CT_WINDOW_NAMES: List[str] = list(rve.ANATOMICAL_WINDOWS["CT"].keys())  # 10 windows


# =============================================================================
# WINDOWING HELPER
# =============================================================================

def apply_pillar_windows(volume_int16: torch.Tensor) -> torch.Tensor:
    """
    Apply 11 HU windows to a raw int16 CT volume.

    Args:
        volume_int16: [D, H, W] int16 tensor with raw HU values

    Returns:
        [11, D, H, W] float32 tensor in [0, 1] range
        Channels: 10 anatomical windows (lung/mediastinum/…) + minmax
    """
    D, H, W = volume_int16.shape

    # batch_apply_windowing_vectorized expects at least (B, C, *spatial)
    # We add batch and channel dims: [1, 1, D, H, W]
    vol_5d = volume_int16.float().unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]

    # Apply 10 anatomical windows → output [1, 10, D, H, W]
    windowed = rve.batch_apply_windowing_vectorized(
        vol_5d,
        windows="all",
        modality="CT",
        torch_operating_dtype=torch.float32,
        compute_stats_per_sample=True,
    )  # [1, 10, D, H, W] in [0, 1]

    windowed = windowed.squeeze(0)  # [10, D, H, W]

    # 11th channel: global min-max normalisation
    vol_float = volume_int16.float()
    vmin = vol_float.min()
    vmax = vol_float.max()
    minmax = (vol_float - vmin) / (vmax - vmin + 1e-8)  # [D, H, W] in [0, 1]

    result = torch.cat([windowed, minmax.unsqueeze(0)], dim=0)  # [11, D, H, W]
    return result.contiguous()


# =============================================================================
# DATASET
# =============================================================================

class PillarDataset(Dataset):
    """
    Dataset for Pillar-0 backbone variants (OracleCT_Pillar_*).

    Two pack sources:
    - pillar_pack_root: directory containing {case_id}.tar.lz4 files
                        (384³ 1.5mm isotropic, raw int16 HU, RAVE LZ4 format)
    - mask_pack_root:   directory containing {case_id}.pt files
                        (existing pack files; only masks channel is used)

    Args:
        pillar_pack_root: Path to 384³ RAVE LZ4 pack directory
        mask_pack_root:   Path to original .pt mask pack directory
        labels_csv:       Path to labels CSV (index: study id)
        case_ids:         List of case IDs for this split
        features_parquet: Optional path to radiomics parquet (scalar features)
        feature_columns:  Optional list of specific feature columns
        disease_names:    List of disease column names
        cache_packs:      Cache loaded pillar volumes in memory (expensive — 11×384³ per sample)
    """

    def __init__(
        self,
        pillar_pack_root: str,
        mask_pack_root: str,
        labels_csv: str,
        case_ids: List[str],
        features_parquet: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        disease_names: Optional[List[str]] = None,
        cache_packs: bool = False,
    ):
        self.pillar_pack_root = Path(pillar_pack_root)
        self.mask_pack_root = Path(mask_pack_root)
        self.cache_packs = cache_packs
        self._cache: Dict[str, torch.Tensor] = {}  # cache windowed volumes only

        # Optional scalar features
        self.use_features = features_parquet is not None
        if self.use_features:
            self.features_df = pd.read_parquet(features_parquet)
            if feature_columns is not None:
                cols_to_keep = ["case_id"] + feature_columns
                available_cols = [c for c in cols_to_keep if c in self.features_df.columns]
                self.features_df = self.features_df[available_cols]
            else:
                # Drop non-numeric columns
                non_numeric = []
                for col in self.features_df.columns:
                    if col == "case_id":
                        continue
                    try:
                        pd.to_numeric(self.features_df[col], errors="raise")
                    except (ValueError, TypeError):
                        non_numeric.append(col)
                if non_numeric:
                    print(f"Removing non-numeric columns: {non_numeric}")
                    self.features_df = self.features_df.drop(columns=non_numeric)
            self.features_df = self.features_df.set_index("case_id")
            self.num_features = len(self.features_df.columns)
            print(f"Loaded {self.num_features} numeric features")
        else:
            self.features_df = None
            self.num_features = 0

        # Labels
        self.labels_df = pd.read_csv(labels_csv)
        self.labels_df = self.labels_df.set_index("study id")

        if disease_names is None:
            disease_names = list(self.labels_df.columns)
        self.disease_names = disease_names
        self.num_diseases = len(disease_names)

        # Filter to valid case IDs
        self.case_ids = []
        skipped = 0
        for case_id in case_ids:
            pillar_pack = self.pillar_pack_root / f"{case_id}.tar.lz4"
            mask_pack = self.mask_pack_root / f"{case_id}.pt"
            has_pillar = pillar_pack.exists()
            has_mask = mask_pack.exists()
            has_labels = case_id in self.labels_df.index
            has_features = (not self.use_features) or (case_id in self.features_df.index)

            if has_pillar and has_mask and has_labels and has_features:
                self.case_ids.append(case_id)
            else:
                skipped += 1

        print(
            f"PillarDataset: {len(self.case_ids)} valid cases "
            f"(skipped {skipped} missing pillar/mask/labels)"
            + (" + features" if self.use_features else "")
        )

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case_id = self.case_ids[idx]

        # ------------------------------------------------------------------
        # 1. Load windowed Pillar image [11, 384, 384, 384]
        # ------------------------------------------------------------------
        if self.cache_packs and case_id in self._cache:
            image = self._cache[case_id]
        else:
            pillar_path = str(self.pillar_pack_root / f"{case_id}.tar.lz4")
            # load_sample returns [D, H, W] int16 (or int16-like) tensor
            raw_volume = rve.load_sample(pillar_path, device="cpu")
            # raw_volume: [D, H, W] with raw HU values as int16

            if raw_volume.dtype != torch.int16:
                raw_volume = raw_volume.to(torch.int16)

            image = apply_pillar_windows(raw_volume)  # [11, 384, 384, 384] float32

            if self.cache_packs:
                self._cache[case_id] = image

        # ------------------------------------------------------------------
        # 2. Load masks from existing .pt pack [20, D, H, W]
        # ------------------------------------------------------------------
        mask_path = self.mask_pack_root / f"{case_id}.pt"
        try:
            pack = torch.load(mask_path, weights_only=False, map_location="cpu", mmap=True)
        except Exception:
            pack = torch.load(mask_path, weights_only=False, map_location="cpu")

        # Permute from nibabel [C, X, Y, Z] → [C, Z, Y, X] (axial depth = first dim)
        masks = pack["masks"].permute(0, 3, 2, 1).contiguous()  # [20, D, H, W]
        meta = pack["meta"].copy()

        # Update spacing to match permuted orientation [sz, sy, sx]
        spacing_orig = meta.get("spacing_final_mm", [1.5, 1.5, 3.0])
        meta["spacing_final_mm"] = [spacing_orig[2], spacing_orig[1], spacing_orig[0]]

        # ------------------------------------------------------------------
        # 3. Labels
        # ------------------------------------------------------------------
        labels_row = self.labels_df.loc[case_id]
        labels = torch.tensor(
            [float(labels_row.get(d, 0)) for d in self.disease_names],
            dtype=torch.float32,
        )

        # ------------------------------------------------------------------
        # 4. Scalar features (optional)
        # ------------------------------------------------------------------
        features_row = self.features_df.loc[case_id] if self.use_features else None

        return {
            "image": image,          # [11, 384, 384, 384] float32
            "masks": masks,          # [20, D, H, W] float32
            "features_row": features_row,
            "labels": labels,
            "meta": meta,
            "case_id": case_id,
        }

    def get_disease_names(self) -> List[str]:
        return self.disease_names


# =============================================================================
# COLLATE FUNCTION (same structure as janus_collate_fn)
# =============================================================================

def pillar_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for PillarDataset.

    Mirrors janus_collate_fn: creates disease_rois from features_row + meta.
    """
    from .roi_utils import create_disease_rois_batch  # local import to avoid circular

    images = torch.stack([b["image"] for b in batch])    # [B, 11, 384, 384, 384]
    masks = torch.stack([b["masks"] for b in batch])     # [B, 20, D, H, W]
    labels = torch.stack([b["labels"] for b in batch])   # [B, num_diseases]

    features_rows = [b["features_row"] for b in batch]
    meta = [b["meta"] for b in batch]
    case_ids = [b["case_id"] for b in batch]

    # Populate organ_touches_border in meta from features_row
    for m, fr in zip(meta, features_rows):
        if fr is not None and "organs" in m:
            organ_touches_border = {}
            for organ in m["organs"]:
                border_key = f"{organ}_touches_border"
                if border_key in fr.index:
                    organ_touches_border[organ] = bool(fr[border_key])
                else:
                    organ_touches_border[organ] = False
            m["organ_touches_border"] = organ_touches_border
        else:
            m["organ_touches_border"] = {o: False for o in m.get("organs", [])}

    # Build disease_rois from parquet features
    disease_rois: List[Dict] = []
    valid_features = [fr for fr in features_rows if fr is not None]
    valid_indices = [i for i, fr in enumerate(features_rows) if fr is not None]

    if valid_features:
        features_df = pd.DataFrame(valid_features)
        if "case_id" not in features_df.columns:
            features_df["case_id"] = [case_ids[i] for i in valid_indices]
        disease_rois = create_disease_rois_batch(
            case_ids=case_ids,
            features_df=features_df,
            metas=meta,
        )
    else:
        disease_rois = [{}] * len(batch)

    return {
        "image": images,
        "masks": masks,
        "features_row": features_rows,
        "labels": labels,
        "meta": meta,
        "case_id": case_ids,
        "disease_rois": disease_rois,
    }
