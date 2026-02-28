#!/usr/bin/env python3
#Copyright 2026 LAVSEN DAHAL
"""
repack_pillar384.py — Convert NIfTI CT scans → 384³ 1.5mm isotropic RAVE LZ4 packs.

Creates tar.lz4 files compatible with PillarDataset / rve.load_sample().

Usage:
    python repack_pillar384.py \\
        --nii_csv    /path/to/nii_paths.csv \\
        --output_dir /path/to/pillar_packs_384 \\
        --workers    8

Input CSV format (must have columns: case_id, nii_path):
    case_id,nii_path
    case_001,/data/merlin/case_001.nii.gz
    case_002,/data/merlin/case_002.nii.gz
    ...

Output:
    output_dir/{case_id}.tar.lz4   — RAVE LZ4 pack (volume.npy + metadata.json)
    output_dir/mapping.csv         — case_id → tar.lz4 path mapping
    output_dir/failed.csv          — cases that failed processing

Notes:
- Resamples to 1.5mm isotropic using scipy.ndimage.zoom (trilinear)
- Crops/pads to 384³ after resampling
- Saves raw int16 HU values (no windowing applied here — PillarDataset does that)
- LZ4 compression typically achieves 6-8× ratio on int16 CT data
- Estimated output: ~18 MB per case (384³ × 2 bytes ÷ ~7× LZ4 ratio)
- ~18 GB total for 1000 cases
"""

import argparse
import io
import json
import lz4.frame
import logging
import tarfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Target volume parameters
TARGET_SIZE = (384, 384, 384)
TARGET_SPACING_MM = (1.5, 1.5, 1.5)
HU_CLIP_MIN = -1024
HU_CLIP_MAX = 3071


# =============================================================================
# RESAMPLING + RESIZING
# =============================================================================

def resample_to_isotropic(
    volume: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float] = TARGET_SPACING_MM,
) -> np.ndarray:
    """
    Resample volume from original spacing to target isotropic spacing.

    Args:
        volume: [X, Y, Z] int16 numpy array (nibabel convention)
        spacing_xyz: original voxel spacing (sx, sy, sz) in mm
        target_spacing: target spacing (all equal for isotropic)

    Returns:
        Resampled [X', Y', Z'] float32 array
    """
    from scipy.ndimage import zoom

    sx, sy, sz = spacing_xyz
    tx, ty, tz = target_spacing
    zoom_factors = (sx / tx, sy / ty, sz / tz)

    # Trilinear interpolation (order=1) is fast and smooth for CT
    resampled = zoom(volume.astype(np.float32), zoom_factors, order=1)
    return resampled


def crop_or_pad_to_target(
    volume: np.ndarray,
    target: Tuple[int, int, int] = TARGET_SIZE,
    pad_value: float = float(HU_CLIP_MIN),
) -> np.ndarray:
    """
    Crop (centre-crop) or pad (symmetric) volume to target shape.

    Args:
        volume: [X, Y, Z] float32 array
        target: target shape (Xt, Yt, Zt)
        pad_value: fill value for padding (default: -1024 HU = air)

    Returns:
        [Xt, Yt, Zt] float32 array
    """
    result = np.full(target, pad_value, dtype=np.float32)
    src_shape = volume.shape

    slices_src = []
    slices_dst = []

    for src_dim, tgt_dim in zip(src_shape, target):
        if src_dim >= tgt_dim:
            # Crop: take centre tgt_dim voxels
            start = (src_dim - tgt_dim) // 2
            slices_src.append(slice(start, start + tgt_dim))
            slices_dst.append(slice(0, tgt_dim))
        else:
            # Pad: centre the source in target
            offset = (tgt_dim - src_dim) // 2
            slices_src.append(slice(0, src_dim))
            slices_dst.append(slice(offset, offset + src_dim))

    result[tuple(slices_dst)] = volume[tuple(slices_src)]
    return result


# =============================================================================
# RAVE LZ4 PACK WRITER
# =============================================================================

def write_tar_lz4(
    volume_int16: np.ndarray,  # [X, Y, Z] int16
    output_path: Path,
    case_id: str,
    original_spacing: Tuple[float, float, float],
    original_shape: Tuple[int, int, int],
) -> None:
    """
    Write a RAVE-compatible tar.lz4 file.

    Format: tar archive containing:
      volume.npy   — int16 numpy array, shape [D, H, W] where D=Z (axial)
      metadata.json — series metadata

    Note: RAVE's load_vision_sample returns shape (D, H, W) where D=Z.
    We permute [X, Y, Z] → [Z, Y, X] = [D, H, W] here to match RAVE convention.
    """
    # Permute from nibabel [X, Y, Z] → [Z, Y, X] = [D, H, W] (RAVE/axial convention)
    # This matches what load_vision_sample returns and what PillarDataset expects
    volume_dhw = np.transpose(volume_int16, (2, 1, 0))  # [X,Y,Z] → [Z,Y,X]

    # Build metadata (minimal, matching RAVE's metadata.json structure)
    metadata = {
        "series_info": {
            "accession": case_id,
            "series_number": "0",
            "modality": "CT",
        },
        "export_info": {
            "format": "lz4",
            "dtype": "int16",
            "shape": list(volume_dhw.shape),  # [D, H, W]
            "target_spacing_mm": list(TARGET_SPACING_MM),
            "target_size": list(TARGET_SIZE),
            "original_spacing_mm": list(original_spacing),
            "original_shape": list(original_shape),
        },
        "case_id": case_id,
    }

    # Create in-memory tar containing volume.npy + metadata.json
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        # volume.npy
        npy_buffer = io.BytesIO()
        np.save(npy_buffer, volume_dhw)
        npy_bytes = npy_buffer.getvalue()

        npy_info = tarfile.TarInfo(name="volume.npy")
        npy_info.size = len(npy_bytes)
        tar.addfile(npy_info, io.BytesIO(npy_bytes))

        # metadata.json
        meta_bytes = json.dumps(metadata, indent=2).encode("utf-8")
        meta_info = tarfile.TarInfo(name="metadata.json")
        meta_info.size = len(meta_bytes)
        tar.addfile(meta_info, io.BytesIO(meta_bytes))

    # LZ4 compress
    tar_bytes = tar_buffer.getvalue()
    compressed = lz4.frame.compress(tar_bytes, compression_level=1)  # fast compress

    output_path.write_bytes(compressed)
    logger.debug(
        f"  {case_id}: {len(tar_bytes) / 1e6:.1f} MB → {len(compressed) / 1e6:.1f} MB LZ4"
    )


# =============================================================================
# PER-CASE PROCESSING
# =============================================================================

def process_case(
    case_id: str,
    nii_path: str,
    output_dir: Path,
) -> Tuple[str, bool, str]:
    """
    Process a single case: load NIfTI → resample → crop/pad → save tar.lz4.

    Returns:
        (case_id, success, message)
    """
    try:
        import nibabel as nib

        output_path = output_dir / f"{case_id}.tar.lz4"
        if output_path.exists():
            return case_id, True, "already_exists"

        # Load NIfTI
        nii = nib.load(nii_path)
        nii = nib.as_closest_canonical(nii)  # RAS+ orientation
        volume = nii.get_fdata(dtype=np.float32)  # [X, Y, Z] in RAS+ mm space
        header = nii.header

        # Get voxel spacing (in mm)
        zooms = header.get_zooms()[:3]  # (sx, sy, sz) in mm
        spacing_xyz = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
        original_shape = volume.shape[:3]

        # Clip HU to valid range
        volume = np.clip(volume, HU_CLIP_MIN, HU_CLIP_MAX)

        # Resample to 1.5mm isotropic
        resampled = resample_to_isotropic(volume, spacing_xyz, TARGET_SPACING_MM)

        # Crop/pad to 384³
        cropped = crop_or_pad_to_target(resampled, TARGET_SIZE, pad_value=float(HU_CLIP_MIN))

        # Convert to int16
        volume_int16 = np.clip(cropped, HU_CLIP_MIN, HU_CLIP_MAX).astype(np.int16)

        # Write tar.lz4
        write_tar_lz4(
            volume_int16, output_path, case_id,
            original_spacing=spacing_xyz,
            original_shape=original_shape,
        )

        size_mb = output_path.stat().st_size / 1e6
        return case_id, True, f"{size_mb:.1f} MB"

    except Exception as e:
        return case_id, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Repack NIfTI CT scans to 384³ 1.5mm RAVE LZ4 packs for Pillar-0."
    )
    parser.add_argument(
        "--nii_csv", required=True,
        help="CSV with columns: case_id, nii_path"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for .tar.lz4 packs"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--case_ids_filter", default=None,
        help="Optional .txt file with case IDs to process (one per line); "
             "if omitted, processes all rows in nii_csv"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input CSV
    df = pd.read_csv(args.nii_csv)
    assert "case_id" in df.columns and "nii_path" in df.columns, (
        "nii_csv must have columns: case_id, nii_path"
    )

    # Optional filter
    if args.case_ids_filter:
        with open(args.case_ids_filter) as f:
            filter_ids = set(line.strip() for line in f if line.strip())
        df = df[df["case_id"].isin(filter_ids)].reset_index(drop=True)
        logger.info(f"Filtered to {len(df)} cases from {args.case_ids_filter}")

    logger.info(f"Processing {len(df)} cases → {output_dir}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Target: {TARGET_SIZE} @ {TARGET_SPACING_MM} mm")

    results = []

    if args.workers <= 1:
        # Single-threaded for debugging
        for _, row in df.iterrows():
            case_id, success, msg = process_case(
                str(row["case_id"]), str(row["nii_path"]), output_dir
            )
            status = "OK" if success else "FAIL"
            logger.info(f"  [{status}] {case_id}: {msg}")
            results.append({"case_id": case_id, "success": success, "message": msg})
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_case, str(row["case_id"]), str(row["nii_path"]), output_dir
                ): row["case_id"]
                for _, row in df.iterrows()
            }

            done = 0
            for future in as_completed(futures):
                case_id, success, msg = future.result()
                done += 1
                status = "OK" if success else "FAIL"
                logger.info(f"  [{status}] [{done}/{len(df)}] {case_id}: {msg}")
                results.append({"case_id": case_id, "success": success, "message": msg})

    # Save results
    results_df = pd.DataFrame(results)

    # Mapping CSV (successful cases only)
    success_df = results_df[results_df["success"]].copy()
    success_df["tar_lz4_path"] = success_df["case_id"].apply(
        lambda cid: str(output_dir / f"{cid}.tar.lz4")
    )
    mapping_path = output_dir / "mapping.csv"
    success_df[["case_id", "tar_lz4_path"]].to_csv(mapping_path, index=False)
    logger.info(f"\nMapping CSV saved: {mapping_path} ({len(success_df)} cases)")

    # Failed cases
    failed_df = results_df[~results_df["success"]]
    if len(failed_df) > 0:
        failed_path = output_dir / "failed.csv"
        failed_df.to_csv(failed_path, index=False)
        logger.warning(f"Failed cases saved: {failed_path} ({len(failed_df)} cases)")

    # Summary
    n_ok = results_df["success"].sum()
    n_fail = len(results_df) - n_ok
    total_mb = sum(
        (output_dir / f"{cid}.tar.lz4").stat().st_size / 1e6
        for cid in success_df["case_id"]
        if (output_dir / f"{cid}.tar.lz4").exists()
    )
    logger.info(
        f"\nDone: {n_ok} OK / {n_fail} failed | "
        f"Total size: {total_mb:.1f} MB ({total_mb / 1024:.2f} GB)"
    )


if __name__ == "__main__":
    main()
