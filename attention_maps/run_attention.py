#!/usr/bin/env python3
"""
Attention map visualization for oracle-ct models.

Supports:
  - OracleCT_DINOv3_MaskedUnaryAttn
  - OracleCT_Pillar_MaskedAttn

Output per case:
  OUT_DIR/<study_id>/raw/<study_id>__z<ZZZ>.png        ← CT only
  OUT_DIR/<study_id>/overlay/<study_id>__z<ZZZ>.png   ← CT + mask contour + attention heatmap
  OUT_DIR/<study_id>/slices.txt                        ← saved slice indices + disease score

All configuration lives in config.py — edit that file, not this one.
"""

from __future__ import annotations
import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Load user config from config.py in the same directory
# ---------------------------------------------------------------------------
_CFG_PATH = Path(__file__).parent / "config.py"
_spec = importlib.util.spec_from_file_location("attn_config", _CFG_PATH)
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)

# ---------------------------------------------------------------------------
# oracle-ct on sys.path
# ---------------------------------------------------------------------------
_ORACLE_CT_ROOT = Path(__file__).resolve().parent.parent.parent  # oracle-ct/
if str(_ORACLE_CT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ORACLE_CT_ROOT))

# ---------------------------------------------------------------------------
# CT display helpers
# ---------------------------------------------------------------------------
HU_CLIP = (-1000.0, 1000.0)


def x01_to_hu(x01: np.ndarray) -> np.ndarray:
    lo, hi = HU_CLIP
    return np.asarray(x01, dtype=np.float32) * (hi - lo) + lo


def wlww_to_vmin_vmax(wl: float, ww: float) -> Tuple[float, float]:
    return wl - ww / 2.0, wl + ww / 2.0


def view_transform(arr2d: np.ndarray) -> np.ndarray:
    """Radiological orientation: rot90 CCW + horizontal flip."""
    return np.fliplr(np.rot90(arr2d, k=1))


# ---------------------------------------------------------------------------
# Slice selection
# ---------------------------------------------------------------------------
def pick_slices(attn_dhw: np.ndarray, z_hint: int, frac: float = 0.30) -> List[int]:
    """Return slice indices where attention >= frac * global_max. Falls back to z_hint."""
    A = np.abs(np.asarray(attn_dhw, dtype=np.float32))
    per_slice = A.reshape(A.shape[0], -1).max(axis=1)
    gmax = float(per_slice.max()) if per_slice.size else 0.0
    if gmax <= 0.0:
        return [int(z_hint)]
    idxs = np.where(per_slice >= frac * gmax)[0].tolist()
    return sorted([int(i) for i in idxs]) if idxs else [int(z_hint)]


# ---------------------------------------------------------------------------
# PNG rendering
# ---------------------------------------------------------------------------
def save_raw(hu_slice: np.ndarray, wl: float, ww: float, path: Path) -> None:
    vmin, vmax = wlww_to_vmin_vmax(wl, ww)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(hu_slice, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_overlay(
    hu_slice: np.ndarray,
    mask_slice: Optional[np.ndarray],
    attn_slice: Optional[np.ndarray],
    wl: float,
    ww: float,
    path: Path,
) -> None:
    vmin, vmax = wlww_to_vmin_vmax(wl, ww)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))

    # CT background
    ax.imshow(hu_slice, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")

    # Organ mask — cyan contour outline
    if mask_slice is not None and mask_slice.max() > 0:
        ax.contour(
            mask_slice.astype(float),
            levels=[0.5],
            colors=["cyan"],
            linewidths=1.5,
        )

    # Attention heatmap (magma, normalized per-slice)
    if attn_slice is not None:
        A = attn_slice.astype(np.float32)
        A -= A.min()
        if A.max() > 0:
            A /= A.max()
        ax.imshow(A, cmap="magma", alpha=0.40, origin="upper")

    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Checkpoint loading (handles DDP "module." prefix)
# ---------------------------------------------------------------------------
def load_state_dict(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"]
    # Strip DDP "module." prefix if checkpoint was saved with DDP
    if all(k.startswith("module.") for k in sd.keys()):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    print(f"  Loaded checkpoint: {Path(ckpt_path).name}  (epoch {ckpt.get('epoch', '?')})")


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def build_dinov3(disease_names: List[str], device: torch.device) -> torch.nn.Module:
    from oracle_ct.models.dinov3_oracle_ct import OracleCT_DINOv3_MaskedUnaryAttn
    model = OracleCT_DINOv3_MaskedUnaryAttn(
        num_diseases=len(disease_names),
        disease_names=disease_names,
        variant=cfg.DINOV3_VARIANT,
        image_size=cfg.DINOV3_IMAGE_SIZE,
        tri_stride=cfg.DINOV3_TRI_STRIDE,
        freeze_backbone=True,
        return_attn=True,
    )
    load_state_dict(model, cfg.CKPT)
    model.eval().to(device)
    return model


def build_pillar(disease_names: List[str], device: torch.device) -> torch.nn.Module:
    from oracle_ct.models.pillar_oracle_ct import OracleCT_Pillar_MaskedAttn
    model = OracleCT_Pillar_MaskedAttn(
        num_diseases=len(disease_names),
        disease_names=disease_names,
        model_repo_id=cfg.PILLAR_MODEL_REPO,
        freeze_backbone=True,
        return_attn=True,
    )
    load_state_dict(model, cfg.CKPT)
    model.eval().to(device)
    return model


# ---------------------------------------------------------------------------
# Data loading — one case at a time, no augmentation
# ---------------------------------------------------------------------------
def load_batch_dinov3(case_id: str, disease_names: List[str]) -> Dict:
    from oracle_ct.datamodules.dataset import JanusDataset, janus_collate_fn
    from torch.utils.data import DataLoader
    ds = JanusDataset(
        pack_root=cfg.PACK_ROOT,
        labels_csv=cfg.LABELS_CSV,
        case_ids=[case_id],
        disease_names=disease_names,
        use_augmentation=False,
        cache_packs=False,
    )
    loader = DataLoader(ds, batch_size=1, collate_fn=janus_collate_fn)
    return next(iter(loader))


def load_batch_pillar(case_id: str, disease_names: List[str]) -> Dict:
    from oracle_ct.datamodules.pillar_dataset import PillarDataset, pillar_collate_fn
    from torch.utils.data import DataLoader
    ds = PillarDataset(
        pillar_pack_root=cfg.PILLAR_PACK_ROOT,
        mask_pack_root=cfg.PILLAR_MASK_ROOT,
        labels_csv=cfg.LABELS_CSV,
        case_ids=[case_id],
        disease_names=disease_names,
        cache_packs=False,
    )
    loader = DataLoader(ds, batch_size=1, collate_fn=pillar_collate_fn)
    return next(iter(loader))


# ---------------------------------------------------------------------------
# Display image — raw HU from the Janus .pt pack
# (used for CT background regardless of model type)
# ---------------------------------------------------------------------------
def load_display_hu(case_id: str, pack_root: str) -> np.ndarray:
    """Load [D, H, W] HU volume from Janus-format .pt pack."""
    packs = sorted(Path(pack_root).glob(f"{case_id}*.pt"))
    if not packs:
        raise FileNotFoundError(f"No pack found for {case_id} in {pack_root}")
    pack = torch.load(str(packs[0]), map_location="cpu")
    for key in ["image_x01", "image", "x01"]:
        if key in pack:
            img = pack[key]
            if torch.is_tensor(img):
                img = img.detach().cpu().numpy()
            img = np.asarray(img, dtype=np.float32)
            if img.ndim == 4 and img.shape[0] == 1:
                img = img[0]
            return x01_to_hu(img)
    raise KeyError(f"No image key in pack for {case_id} (tried: image_x01, image, x01)")


# ---------------------------------------------------------------------------
# Organ mask extraction from batch
# ---------------------------------------------------------------------------
def get_organ_mask(batch: Dict, organ: str) -> Optional[np.ndarray]:
    """Return [D, H, W] binary mask for the requested organ, or None."""
    masks = batch["masks"]       # [1, O, D, H, W]
    meta  = batch["meta"][0]     # dict
    organs = meta.get("organs", [])
    if organ not in organs:
        print(f"  [warn] organ '{organ}' not in meta organs: {organs}")
        return None
    oi = organs.index(organ)
    m = masks[0, oi].cpu().numpy()
    return (m > 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Disease config — must be loaded before get_all_diseases()
    from oracle_ct.configs.disease_config import get_all_diseases, load_config_globally
    disease_cfg_path = Path(__file__).resolve().parent.parent / "configs" / "disease_config_oracle_ct.py"
    load_config_globally(str(disease_cfg_path))
    disease_names = get_all_diseases()[:cfg.NUM_DISEASES]
    print(f"Diseases: {len(disease_names)}  ({disease_names[:3]} ...)")

    # Validate all requested diseases exist
    for sid, spec in cfg.CASES.items():
        d = spec["disease"]
        if d not in disease_names:
            raise ValueError(
                f"Disease '{d}' for case '{sid}' not found in disease list. "
                f"Check config.py CASES and NUM_DISEASES."
            )

    # Build model
    print(f"\nBuilding model: {cfg.MODEL.upper()}  checkpoint: {Path(cfg.CKPT).name}")
    if cfg.MODEL == "dinov3":
        model = build_dinov3(disease_names, device)
        display_pack_root = cfg.PACK_ROOT
    elif cfg.MODEL == "pillar":
        model = build_pillar(disease_names, device)
        display_pack_root = cfg.PILLAR_MASK_ROOT  # Janus-format packs for CT display
    else:
        raise ValueError(f"Unknown MODEL '{cfg.MODEL}' in config.py. Use 'dinov3' or 'pillar'.")

    out_root = Path(cfg.OUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output → {out_root}\n")

    for sid, spec in cfg.CASES.items():
        organ   = spec["organ"]
        disease = spec["disease"]
        z_hint  = int(spec["z_hint"])
        wl      = float(spec["wl"])
        ww      = float(spec["ww"])

        print(f"[*] {sid}  organ={organ}  disease={disease}  WL/WW={wl}/{ww}")

        # --- Load batch for model ---
        try:
            if cfg.MODEL == "dinov3":
                batch = load_batch_dinov3(sid, disease_names)
            else:
                batch = load_batch_pillar(sid, disease_names)
        except Exception as e:
            print(f"  [error] Could not load data for {sid}: {e}")
            continue

        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        # --- Inference ---
        with torch.inference_mode():
            out = model(batch)

        logits   = out["logits"][0]               # [num_diseases]
        attn_map = out["attn"].get(disease)        # [1, 1, D, H, W] or None

        if attn_map is None:
            print(f"  [warn] No attention map for '{disease}' — skipping.")
            continue

        A = attn_map[0, 0].cpu().float().numpy()  # [D, H, W]

        # Disease prediction score
        d_idx = disease_names.index(disease)
        score = torch.sigmoid(logits[d_idx]).item()
        print(f"  Sigmoid score for '{disease}': {score:.4f}")

        # --- Display image (raw CT) ---
        try:
            hu = load_display_hu(sid, display_pack_root)
        except Exception as e:
            print(f"  [error] Could not load display image for {sid}: {e}")
            continue

        # --- Organ mask for contour ---
        mask3d = get_organ_mask(batch, organ)  # [D, H, W] or None

        # --- Slice selection ---
        z_idxs = pick_slices(A, z_hint, frac=cfg.ATTN_FRAC_OF_GLOBAL_MAX)
        print(f"  Slices: {len(z_idxs)} selected  {z_idxs[:10]}{'...' if len(z_idxs) > 10 else ''}")

        # --- Render ---
        case_dir = out_root / sid
        raw_dir  = case_dir / "raw"
        ovl_dir  = case_dir / "overlay"
        raw_dir.mkdir(parents=True, exist_ok=True)
        ovl_dir.mkdir(parents=True, exist_ok=True)

        saved = []
        for z in z_idxs:
            if z < 0 or z >= hu.shape[0]:
                continue

            hu_sl   = view_transform(hu[z])
            attn_sl = view_transform(A[z])
            mask_sl = view_transform(mask3d[z]) if mask3d is not None else None

            fname = f"{sid}__z{z:03d}.png"
            save_raw(hu_sl, wl, ww, raw_dir / fname)
            save_overlay(hu_sl, mask_sl, attn_sl, wl, ww, ovl_dir / fname)
            saved.append(z)

        (case_dir / "slices.txt").write_text(
            f"model:   {cfg.MODEL}\n"
            f"disease: {disease}  score: {score:.4f}\n"
            f"organ:   {organ}\n"
            f"slices:  {', '.join(str(i) for i in saved)}\n",
            encoding="utf-8",
        )
        print(f"  [saved] {len(saved)} slices → {case_dir}")

    print("\n[done] All cases rendered.")


if __name__ == "__main__":
    main()
