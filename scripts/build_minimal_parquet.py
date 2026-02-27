#!/usr/bin/env python3
"""
Build OracleCT Minimal Feature Parquet

Filters the full macro-radiomics parquet down to only the columns needed
by OracleCT's MaskedUnaryAttnScalar model:
  - case_id
  - *_mean_hu
  - *_to_body_ratio
  - *_touches_border
  - *_volume_cc
  - total_body_volume_cc
  - lumbar_hu_mean, vertebrae_*_mean_hu   (bones — no standard mean_hu naming)
  - aorta_hu_mean, aorta_calc_fraction    (aorta — different naming convention)
  - liver_spleen_hu_diff                  (hepatic steatosis comparative signal)

Usage:
    python scripts/build_minimal_parquet.py \
        --input  /scratch/railabs/ld258/output/ct_triage/macroradiomics/merlin/features/features_combined.parquet \
        --output /scratch/railabs/ld258/output/ct_triage/oracle-ct/data/merlin/features_minimal.parquet
"""

import argparse
from pathlib import Path

import pandas as pd


def build_minimal_parquet(input_path: str, output_path: str) -> None:
    print(f"Reading: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"  Full parquet: {df.shape[0]} cases × {df.shape[1]} columns")

    cols = list(df.columns)

    keep = set()

    # Always keep case_id if present as a column (may be index)
    if "case_id" in cols:
        keep.add("case_id")

    # Standard patterns
    for c in cols:
        if (
            c.endswith("_mean_hu")        or
            c.endswith("_to_body_ratio")  or
            c.endswith("_touches_border") or
            c.endswith("_volume_cc")
        ):
            keep.add(c)

    # Body volume normaliser
    for c in ["total_body_volume_cc", "lung_total_volume_cc"]:
        if c in cols:
            keep.add(c)

    # Bones: lumbar aggregate + vertebral HU (no standard _mean_hu naming)
    for c in cols:
        if c.startswith("lumbar_hu") or (c.startswith("vertebrae_") and c.endswith("_mean_hu")):
            keep.add(c)

    # Aorta: different naming convention
    for c in ["aorta_hu_mean", "aorta_calc_fraction"]:
        if c in cols:
            keep.add(c)

    # Hepatic steatosis comparative signal
    if "liver_spleen_hu_diff" in cols:
        keep.add("liver_spleen_hu_diff")

    # Filter and preserve original column order
    ordered = [c for c in cols if c in keep]
    df_min = df[ordered]

    print(f"  Minimal parquet: {df_min.shape[0]} cases × {df_min.shape[1]} columns")

    # Save parquet
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_min.to_parquet(out, index=True)
    print(f"  Saved parquet → {out}")

    # Save CSV alongside parquet for inspection
    csv_out = out.with_suffix(".csv")
    df_min.to_csv(csv_out, index=True)
    print(f"  Saved CSV    → {csv_out}")

    # Summary
    patterns = {
        "_mean_hu":        sum(1 for c in ordered if c.endswith("_mean_hu")),
        "_to_body_ratio":  sum(1 for c in ordered if c.endswith("_to_body_ratio")),
        "_touches_border": sum(1 for c in ordered if c.endswith("_touches_border")),
        "_volume_cc":      sum(1 for c in ordered if c.endswith("_volume_cc")),
        "other":           sum(1 for c in ordered if not any(
            c.endswith(p) for p in ("_mean_hu", "_to_body_ratio", "_touches_border", "_volume_cc")
        )),
    }
    print("\n  Column breakdown:")
    for pat, n in patterns.items():
        print(f"    {pat:20s}: {n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build OracleCT minimal feature parquet")
    parser.add_argument(
        "--input",
        default="/scratch/railabs/ld258/output/ct_triage/macroradiomics/merlin/features/features_combined.parquet",
        help="Full macro-radiomics features parquet",
    )
    parser.add_argument(
        "--output",
        default="/scratch/railabs/ld258/output/ct_triage/oracle-ct/data/merlin/features_minimal.parquet",
        help="Output path for minimal parquet",
    )
    args = parser.parse_args()
    build_minimal_parquet(args.input, args.output)
