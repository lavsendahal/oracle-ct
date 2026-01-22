#!/usr/bin/env python3
"""
Janus Pack Inspector with Appendix ROI Visualization

Inspect and visualize .pt pack files:
- Export image as NIFTI
- Export all 20 organ masks as NIFTI
- Export appendix bounding box from parquet coordinates
- Print summary

Usage:
    python inspect_pack.py --pack_path case.pt --parquet /path/to/features.parquet --output_dir /output
    python inspect_pack.py --pack_dir /packs --parquet /path/to/features.parquet --output_dir /output --limit 5
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import nibabel as nib
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from janus.datamodules.class_map import ORGAN_NAMES
from janus.datamodules.roi_utils import create_appendix_bbox_mask


def print_pack_summary(pack: Dict[str, Any], case_id: str):
    """Print detailed summary of a pack."""
    meta = pack["meta"]

    print(f"\n{'='*70}")
    print(f"PACK: {case_id}")
    print(f"{'='*70}")

    # Image info
    image = pack["image"]
    print(f"\nIMAGE:")
    print(f"  Shape: {list(image.shape)}")
    print(f"  Dtype: {image.dtype}")
    print(f"  Range: [{image.min():.4f}, {image.max():.4f}]")

    # Masks info
    masks = pack["masks"]
    num_channels = masks.shape[0]
    print(f"\nMASKS:")
    print(f"  Shape: {list(masks.shape)}")
    print(f"  Channels: {num_channels}")
    print(f"  Expected: 20 (15 organs + 5 computed spaces)")

    # Print organ list
    print(f"\nORGAN CHANNELS:")
    for i, organ_name in enumerate(ORGAN_NAMES):
        if i < num_channels:
            mask_vol = (masks[i] > 0).sum().item()
            has_mask = "✓" if mask_vol > 0 else "✗"
            print(f"  [{i:2d}] {organ_name:25s} {has_mask}")

    # Metadata
    print(f"\nMETADATA:")
    print(f"  Original shape: {meta.get('shape_orig')}")
    print(f"  Final shape: {meta.get('shape_final')}")
    print(f"  Original spacing: {meta.get('spacing_orig_mm')}")
    print(f"  Final spacing: {meta.get('spacing_final_mm')}")
    print(f"  HU range: {meta.get('hu_range')}")
    print(f"  Has kidney cysts: {meta.get('has_kidney_cysts', False)}")


def save_pack_as_nifti(
    pack: Dict[str, Any],
    output_dir: Path,
    case_id: str,
    features_df: Optional[pd.DataFrame] = None,
):
    """Save pack contents as NIFTI files for visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = pack["meta"]

    # Use ORIGINAL spacing for NIFTI export (to match original CT resolution)
    spacing_orig = meta.get("spacing_orig_mm", [1.5, 1.5, 1.5])
    spacing_final = meta.get("spacing_final_mm", [1.5, 1.5, 1.5])
    hu_range = meta.get("hu_range", [-1000, 1000])

    print(f"\nExporting {case_id}...")
    print(f"  Original spacing: {spacing_orig}")
    print(f"  Final spacing: {spacing_final}")

    # Create affine using FINAL spacing (pack is in final resolution)
    affine = np.diag([spacing_final[0], spacing_final[1], spacing_final[2], 1.0])

    # Save image (convert back to HU)
    image = pack["image"].numpy()  # [1, X, Y, Z]
    image_hu = image[0] * (hu_range[1] - hu_range[0]) + hu_range[0]
    image_nii = nib.Nifti1Image(image_hu.astype(np.float32), affine)
    nib.save(image_nii, output_dir / f"{case_id}_image.nii.gz")
    print(f"  ✓ Saved image: {case_id}_image.nii.gz")

    # Save all 20 organ masks
    masks = pack["masks"].numpy()  # [20, X, Y, Z]
    num_channels = masks.shape[0]

    print(f"  ✓ Saving {num_channels} organ masks...")
    for i in range(num_channels):
        organ_name = ORGAN_NAMES[i] if i < len(ORGAN_NAMES) else f"channel_{i}"
        mask_nii = nib.Nifti1Image(masks[i].astype(np.uint8), affine)
        nib.save(mask_nii, output_dir / f"{case_id}_mask_{i:02d}_{organ_name}.nii.gz")

    # Save combined masks (all 20 channels)
    combined = np.zeros(masks.shape[1:], dtype=np.uint8)
    for i in range(num_channels):
        combined[masks[i] > 0] = i + 1

    combined_nii = nib.Nifti1Image(combined, affine)
    nib.save(combined_nii, output_dir / f"{case_id}_masks_combined.nii.gz")
    print(f"  ✓ Saved combined masks: {case_id}_masks_combined.nii.gz")

    # Save appendix bounding box if available
    if features_df is not None:
        shape_orig = tuple(meta.get("shape_orig", masks.shape[1:]))
        shape_final = masks.shape[1:]

        appendix_bbox = create_appendix_bbox_mask(
            case_id=case_id,
            features_df=features_df,
            shape_orig=shape_orig,
            shape_final=shape_final,
            spacing_final_mm=spacing_final,
        )

        if appendix_bbox is not None:
            appendix_nii = nib.Nifti1Image(appendix_bbox.astype(np.uint8), affine)
            nib.save(appendix_nii, output_dir / f"{case_id}_appendix_bbox.nii.gz")
            print(f"  ✓ Saved appendix ROI: {case_id}_appendix_bbox.nii.gz")

            # Also save appendix + colon for comparison
            colon_mask = masks[9] if num_channels > 9 else np.zeros_like(appendix_bbox)  # Channel 9 = colon
            appendix_plus_colon = np.zeros_like(appendix_bbox, dtype=np.uint8)
            appendix_plus_colon[colon_mask > 0] = 1  # Colon = 1
            appendix_plus_colon[appendix_bbox > 0] = 2  # Appendix ROI = 2

            comparison_nii = nib.Nifti1Image(appendix_plus_colon, affine)
            nib.save(comparison_nii, output_dir / f"{case_id}_colon_vs_appendix.nii.gz")
            print(f"  ✓ Saved comparison: {case_id}_colon_vs_appendix.nii.gz (1=colon, 2=appendix_roi)")
        else:
            print(f"  ✗ No appendix coordinates in parquet for {case_id}")

    print(f"  ✓ Done!")


def create_label_description(output_dir: Path):
    """Create label description file for ITK-SNAP with all 20 organs."""
    lines = [
        "# ITK-SNAP Label Description File for Janus",
        "# Format: IDX  R G B A VIS MSH LABEL",
        "0     0   0   0   0  0  0    \"Background\"",
    ]

    # Colors for 20 organs
    colors = [
        (255, 0, 0),      # 0: liver - red
        (0, 255, 0),      # 1: gallbladder - green
        (255, 255, 0),    # 2: pancreas - yellow
        (128, 0, 128),    # 3: spleen - purple
        (255, 165, 0),    # 4: kidneys - orange
        (255, 192, 203),  # 5: kidney_cysts - light pink
        (0, 128, 128),    # 6: prostate - teal
        (255, 192, 203),  # 7: stomach_esophagus - pink
        (139, 69, 19),    # 8: small_bowel - brown
        (165, 42, 42),    # 9: colon - dark brown
        (0, 0, 255),      # 10: lungs - blue
        (255, 0, 255),    # 11: heart - magenta
        (255, 0, 0),      # 12: aorta - bright red
        (0, 0, 139),      # 13: veins - dark blue
        (255, 255, 255),  # 14: bones - white
        (173, 216, 230),  # 15: pleural_space - light blue
        (255, 218, 185),  # 16: periportal_space - peach
        (255, 182, 193),  # 17: perivascular_space - light pink
        (255, 160, 122),  # 18: pericardial_space - light salmon
        (240, 230, 140),  # 19: subcutaneous_space - khaki
    ]

    for i, organ_name in enumerate(ORGAN_NAMES):
        r, g, b = colors[i] if i < len(colors) else (128, 128, 128)
        lines.append(f'{i+1}    {r:3d} {g:3d} {b:3d} 255  1  1    "{organ_name}"')

    # Add appendix bbox label
    lines.append(f'{21}    {255:3d} {0:3d} {0:3d} 255  1  1    "appendix_bbox"')

    with open(output_dir / "LABEL_DESCRIPTION.txt", "w") as f:
        f.write("\n".join(lines))

    print(f"\n✓ Created LABEL_DESCRIPTION.txt for ITK-SNAP")


def main():
    parser = argparse.ArgumentParser(description="Inspect and export Janus pack files")
    parser.add_argument("--pack_path", type=str, help="Path to single pack file")
    parser.add_argument("--pack_dir", type=str, help="Directory of pack files")
    parser.add_argument("--parquet", type=str, required=True, help="Path to features parquet file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for NIFTI files")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of packs to process")
    parser.add_argument("--summary_only", action="store_true", help="Only print summary, don't export NIFTI")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load parquet
    print(f"Loading features parquet: {args.parquet}")
    features_df = pd.read_parquet(args.parquet)
    print(f"  Loaded {len(features_df)} cases")
    print(f"  Columns: {len(features_df.columns)}")

    # Check for appendix columns
    appendix_cols = ["appendix_center_x_mm", "appendix_center_y_mm", "appendix_center_z_mm"]
    has_appendix = all(col in features_df.columns for col in appendix_cols)
    if has_appendix:
        n_with_appendix = features_df["appendix_center_x_mm"].notna().sum()
        print(f"  ✓ Appendix coordinates available for {n_with_appendix} cases")
    else:
        print(f"  ✗ Warning: Appendix coordinates not found in parquet!")

    # Get pack files
    if args.pack_path:
        pack_files = [Path(args.pack_path)]
    elif args.pack_dir:
        pack_files = sorted(Path(args.pack_dir).glob("*.pt"))
        if args.limit:
            pack_files = pack_files[:args.limit]
    else:
        print("Error: Provide --pack_path or --pack_dir")
        return

    print(f"\nProcessing {len(pack_files)} pack(s)...\n")

    for i, pack_file in enumerate(pack_files):
        case_id = pack_file.stem
        print(f"[{i+1}/{len(pack_files)}] Loading {case_id}...")

        pack = torch.load(pack_file, weights_only=False)

        # Print summary
        print_pack_summary(pack, case_id)

        # Export NIFTI
        if not args.summary_only:
            case_output = output_dir / case_id
            save_pack_as_nifti(pack, case_output, case_id, features_df)

    # Create label description (once)
    if not args.summary_only:
        create_label_description(output_dir)

    print(f"\n{'='*70}")
    print(f"Done! Output in {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()




# # Test a single pack
# python janus/scripts/inspect_pack.py \
#     --pack_path /cachedata/ld258/janus/packs/case001.pt \
#     --parquet /scratch/railabs/ld258/output/ct_triage/second_project/janus/janus_features_v1.parquet \
#     --output_dir /tmp/test_pack_output

# # Test multiple packs (limit to 5)
# python /home/ld258/ipredict/neuro_symbolic/janus/scripts/inspect_pack.py \
#     --pack_dir /cachedata/ld258/janus/packs \
#     --parquet /scratch/railabs/ld258/output/ct_triage/second_project/janus/janus_features_v2.parquet \
#     --output_dir /scratch/railabs/ld258/output/ct_triage/second_project/janus/nifti_vis_v2 \
#     --limit 5

# # Just print summary without exporting NIFTI
# python janus/scripts/inspect_pack.py \
#     --pack_path /cachedata/ld258/janus/packs/case001.pt \
#     --parquet /scratch/railabs/ld258/output/ct_triage/second_project/janus/janus_features_v1.parquet \
#     --output_dir /tmp/test_pack_output \
#     --summary_only