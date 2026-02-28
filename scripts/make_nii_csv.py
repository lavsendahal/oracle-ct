#!/usr/bin/env python3
"""
make_nii_csv.py — Build the nii_csv required by repack_pillar384.py.

Scans the NIfTI directory for *.nii.gz files, optionally filters to only
case IDs that exist in the labels CSV, then writes:
    output_csv: case_id, nii_path

Usage:
    python make_nii_csv.py \\
        --nii_dir  /scratch/railabs/ld258/dataset/merlin/merlinabdominalctdataset/merlin_data \\
        --output   /scratch/railabs/ld258/dataset/merlin/nii_paths.csv

Optional filter (recommended — only process cases you have labels for):
    --labels_csv  /scratch/railabs/ld258/dataset/merlin/merlinabdominalctdataset/zero_shot_findings_disease_cls.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Generate nii_paths.csv for repack_pillar384.py"
    )
    parser.add_argument(
        "--nii_dir", required=True,
        help="Directory containing *.nii.gz CT files (case_id.nii.gz)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output CSV path (columns: case_id, nii_path)"
    )
    parser.add_argument(
        "--labels_csv", default=None,
        help="Optional: only include cases present in this labels CSV (column: study id)"
    )
    args = parser.parse_args()

    nii_dir = Path(args.nii_dir)
    assert nii_dir.exists(), f"nii_dir not found: {nii_dir}"

    # Collect all NIfTI files
    nii_files = sorted(nii_dir.glob("*.nii.gz"))
    if not nii_files:
        # Also try sub-directories (some datasets organise per case)
        nii_files = sorted(nii_dir.rglob("*.nii.gz"))

    records = [{"case_id": p.name.replace(".nii.gz", ""), "nii_path": str(p)}
               for p in nii_files]
    df = pd.DataFrame(records)
    print(f"Found {len(df)} NIfTI files in {nii_dir}")

    # Optional filter: keep only cases in labels CSV
    if args.labels_csv:
        labels_df = pd.read_csv(args.labels_csv)
        label_ids = set(labels_df["study id"].astype(str))
        before = len(df)
        df = df[df["case_id"].isin(label_ids)].reset_index(drop=True)
        print(f"After filtering to labels CSV: {len(df)} cases (dropped {before - len(df)})")

    # Write output CSV
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"\nWrote {len(df)} rows → {output}")
    print(f"Preview:\n{df.head(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
