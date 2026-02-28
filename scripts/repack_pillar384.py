#!/usr/bin/env python3
#Copyright 2026 LAVSEN DAHAL
"""
repack_pillar384.py — Convert NIfTI CT scans → 384³ 1.5mm isotropic RAVE LZ4 packs.

Uses SimpleITK for resampling (same as RAVE's CTProcessor) and ThreadPoolExecutor
(same as RAVE's LZ4Exporter) to avoid process-pool hangs.

Creates tar.lz4 files compatible with PillarDataset / rve.load_sample().

Usage:
    python repack_pillar384.py \\
        --nii_csv    /path/to/nii_paths.csv \\
        --output_dir /path/to/pillar_packs_384 \\
        --workers    48

Input CSV format (must have columns: case_id, nii_path):
    case_id,nii_path
    case_001,/data/merlin/case_001.nii.gz

Output:
    output_dir/{case_id}.tar.lz4   — RAVE LZ4 pack (volume.npy + metadata.json)
    output_dir/mapping.csv         — case_id → tar.lz4 path mapping
    output_dir/failed.csv          — cases that failed processing
"""

import argparse
import io
import json
import lz4.frame
import logging
import tarfile
import traceback
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Target volume parameters
TARGET_SIZE = (384, 384, 384)          # (D, H, W)
TARGET_SPACING_MM = (1.5, 1.5, 1.5)   # (z, y, x) in SimpleITK convention
HU_CLIP_MIN = -1024
HU_CLIP_MAX = 3071


# =============================================================================
# RESAMPLING — mirrors RAVE CTProcessor._resample_volume()
# =============================================================================

def resample_to_isotropic(sitk_image: sitk.Image,
                           target_spacing: Tuple[float, float, float]) -> sitk.Image:
    """
    Resample a SimpleITK image to target isotropic spacing.
    Mirrors RAVE's CTProcessor._resample_volume() exactly.
    Uses sitkLinear interpolation (trilinear).
    """
    original_spacing = sitk_image.GetSpacing()   # (x, y, z) in SimpleITK
    original_size    = sitk_image.GetSize()       # (x, y, z)

    # target_spacing is stored as (z, y, x) to match numpy convention;
    # SimpleITK needs (x, y, z)
    ts_xyz = (target_spacing[2], target_spacing[1], target_spacing[0])

    new_size = [
        int(round(original_size[i] * original_spacing[i] / ts_xyz[i]))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(ts_xyz)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(HU_CLIP_MIN)
    resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(sitk_image)


# =============================================================================
# CROP / PAD — mirrors RAVE CTProcessor._crop_pad_volume()
# =============================================================================

def crop_or_pad_to_target(volume: np.ndarray,
                           target: Tuple[int, int, int],
                           pad_value: int = HU_CLIP_MIN) -> np.ndarray:
    """
    Centre-crop or symmetrically pad a (D, H, W) volume to target shape.
    Builds slices for all dims first, then does a single assignment.
    """
    result = np.full(target, pad_value, dtype=volume.dtype)
    slices_src = []
    slices_dst = []
    for dim in range(3):
        src = volume.shape[dim]
        tgt = target[dim]
        if src >= tgt:
            start = (src - tgt) // 2
            slices_src.append(slice(start, start + tgt))
            slices_dst.append(slice(0, tgt))
        else:
            offset = (tgt - src) // 2
            slices_src.append(slice(0, src))
            slices_dst.append(slice(offset, offset + src))
    result[tuple(slices_dst)] = volume[tuple(slices_src)]
    return result


# =============================================================================
# RAVE LZ4 PACK WRITER — mirrors RAVE LZ4Exporter._export_single_series()
# =============================================================================

def write_tar_lz4(volume_int16: np.ndarray,   # (D, H, W) int16
                  output_path: Path,
                  case_id: str,
                  original_spacing: Tuple,
                  original_shape: Tuple) -> None:
    """
    Write a RAVE-compatible tar.lz4 file.
    Volume shape must already be (D, H, W) — SimpleITK GetArrayFromImage() gives this.
    LZ4 parameters match RAVE's LZ4Exporter exactly.
    """
    metadata = {
        "series_info": {
            "accession": case_id,
            "series_number": "0",
            "modality": "CT",
        },
        "export_info": {
            "format": "lz4",
            "dtype": "int16",
            "shape": list(volume_int16.shape),          # [D, H, W]
            "target_spacing_mm": list(TARGET_SPACING_MM),
            "target_size": list(TARGET_SIZE),
            "original_spacing_mm": list(original_spacing),
            "original_shape": list(original_shape),
        },
        "case_id": case_id,
    }

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        # volume.npy
        npy_buf = io.BytesIO()
        np.save(npy_buf, volume_int16)
        npy_bytes = npy_buf.getvalue()
        info = tarfile.TarInfo(name="volume.npy")
        info.size = len(npy_bytes)
        tar.addfile(info, io.BytesIO(npy_bytes))

        # metadata.json
        meta_bytes = json.dumps(metadata, indent=2).encode("utf-8")
        info = tarfile.TarInfo(name="metadata.json")
        info.size = len(meta_bytes)
        tar.addfile(info, io.BytesIO(meta_bytes))

    # LZ4 compress — same parameters as RAVE's LZ4Exporter
    compressed = lz4.frame.compress(
        tar_buffer.getvalue(),
        compression_level=1,
        block_size=lz4.frame.BLOCKSIZE_MAX4MB,
        block_linked=True,
        content_checksum=True,
        return_bytearray=False,
    )
    output_path.write_bytes(compressed)


# =============================================================================
# PER-CASE PROCESSING
# =============================================================================

def process_case(case_id: str,
                 nii_path: str,
                 output_dir: Path) -> Tuple[str, bool, str]:
    """
    Load NIfTI → resample (SimpleITK) → crop/pad → save tar.lz4.
    Returns (case_id, success, message).
    """
    try:
        output_path = output_dir / f"{case_id}.tar.lz4"
        if output_path.exists():
            return case_id, True, "already_exists"

        t0 = time.time()

        # Load with SimpleITK (handles .nii.gz natively)
        sitk_image = sitk.ReadImage(str(nii_path))

        # Original metadata
        orig_spacing_xyz = sitk_image.GetSpacing()   # (x, y, z)
        orig_size_xyz    = sitk_image.GetSize()       # (x, y, z)
        # Store as (z, y, x) to match numpy (D, H, W) convention
        original_spacing = (orig_spacing_xyz[2], orig_spacing_xyz[1], orig_spacing_xyz[0])
        original_shape   = (orig_size_xyz[2],    orig_size_xyz[1],    orig_size_xyz[0])

        # Resample to 1.5mm isotropic (same as RAVE CTProcessor)
        resampled = resample_to_isotropic(sitk_image, TARGET_SPACING_MM)

        # Convert to numpy (D, H, W) = (Z, Y, X) — GetArrayFromImage gives this order
        volume = sitk.GetArrayFromImage(resampled).astype(np.int16)  # (D, H, W)

        # Clip HU
        volume = np.clip(volume, HU_CLIP_MIN, HU_CLIP_MAX).astype(np.int16)

        # Crop/pad to 384³
        volume = crop_or_pad_to_target(volume, TARGET_SIZE, pad_value=HU_CLIP_MIN)

        # Write RAVE LZ4 pack
        write_tar_lz4(volume, output_path, case_id, original_spacing, original_shape)

        elapsed = time.time() - t0
        size_mb = output_path.stat().st_size / 1e6
        return case_id, True, f"{size_mb:.1f} MB  {elapsed:.1f}s"

    except Exception as e:
        return case_id, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Repack NIfTI CT scans to 384³ 1.5mm RAVE LZ4 packs for Pillar-0."
    )
    parser.add_argument("--nii_csv",     required=True,
                        help="CSV with columns: case_id, nii_path")
    parser.add_argument("--output_dir",  required=True,
                        help="Output directory for .tar.lz4 packs")
    parser.add_argument("--workers",     type=int, default=48,
                        help="Number of parallel threads (default: 48)")
    parser.add_argument("--case_ids_filter", default=None,
                        help="Optional .txt file with case IDs to process (one per line)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.nii_csv)
    assert "case_id" in df.columns and "nii_path" in df.columns, \
        "nii_csv must have columns: case_id, nii_path"

    if args.case_ids_filter:
        with open(args.case_ids_filter) as f:
            filter_ids = {l.strip() for l in f if l.strip()}
        df = df[df["case_id"].isin(filter_ids)].reset_index(drop=True)
        logger.info(f"Filtered to {len(df)} cases")

    logger.info(f"Processing {len(df)} cases → {output_dir}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Target: {TARGET_SIZE} @ {TARGET_SPACING_MM} mm")

    results = []

    # ThreadPoolExecutor — same as RAVE's LZ4Exporter.export_tarballs()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_id = {
            executor.submit(process_case,
                            str(row["case_id"]),
                            str(row["nii_path"]),
                            output_dir): row["case_id"]
            for _, row in df.iterrows()
        }

        done = 0
        for future in as_completed(future_to_id):
            cid = future_to_id[future]
            try:
                case_id, success, msg = future.result()
            except Exception as e:
                case_id, success, msg = cid, False, f"Unexpected: {e}"
            done += 1
            status = "OK" if success else "FAIL"
            logger.info(f"  [{status}] [{done}/{len(df)}] {case_id}: {msg}")
            results.append({"case_id": case_id, "success": success, "message": msg})

    # Save results
    results_df = pd.DataFrame(results)

    success_df = results_df[results_df["success"]].copy()
    success_df["tar_lz4_path"] = success_df["case_id"].apply(
        lambda cid: str(output_dir / f"{cid}.tar.lz4")
    )
    mapping_path = output_dir / "mapping.csv"
    success_df[["case_id", "tar_lz4_path"]].to_csv(mapping_path, index=False)
    logger.info(f"Mapping CSV saved: {mapping_path} ({len(success_df)} cases)")

    failed_df = results_df[~results_df["success"]]
    if len(failed_df) > 0:
        failed_path = output_dir / "failed.csv"
        failed_df.to_csv(failed_path, index=False)
        logger.warning(f"Failed cases: {failed_path} ({len(failed_df)} cases)")

    n_ok   = results_df["success"].sum()
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
