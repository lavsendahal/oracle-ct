#!/usr/bin/env python3
"""
Compute Feature Statistics for OracleCT

Computes mean, std, min, max for all features directly from parquet.
Required for z-score normalization in MaskedUnaryAttnScalar model.

Run on the TRAINING split of the minimal parquet:
    python scripts/compute_feature_stats.py \
        --features_parquet /scratch/railabs/ld258/output/ct_triage/oracle-ct/data/merlin/features_minimal.parquet \
        --train_ids /home/ld258/ipredict/oracle-ct/oracle-ct/splits/train_ids.txt \
        --output /scratch/railabs/ld258/output/ct_triage/oracle-ct/data/merlin/feature_stats.json
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


def compute_feature_statistics(
    features_parquet: str,
    output_path: str,
    train_ids_file: str | None = None,
) -> dict:
    """
    Compute feature statistics (mean, std, min, max) directly from parquet.

    Matches the sklearn pipeline used in the LR baseline:
      1) median imputation per feature
      2) StandardScaler fit on imputed values (mean/std with ddof=0)

    Args:
        features_parquet: Path to features parquet file
        output_path: Path to save statistics JSON
        train_ids_file: Path to train_ids.txt — only use training cases

    Returns:
        dict of {feature_name: {mean, std, min, max, median, count, count_non_nan}}
    """
    print("=" * 80)
    print("Computing Feature Statistics from Parquet")
    print("=" * 80)

    # Load parquet
    df = pd.read_parquet(features_parquet)
    if "case_id" in df.columns:
        df = df.set_index("case_id")
    print(f"\n✓ Loaded parquet: {df.shape[0]} cases × {df.shape[1]} features")

    # Filter to training split if provided
    if train_ids_file is not None:
        with open(train_ids_file, "r") as f:
            train_ids = [line.strip() for line in f if line.strip()]
        in_parquet = [cid for cid in train_ids if cid in df.index]
        df = df.loc[in_parquet]
        print(f"✓ Filtered to {len(in_parquet)} training cases (from {len(train_ids)} in file)")
    else:
        print("✓ Using all cases (no train_ids_file provided)")

    # Compute statistics for all numeric columns
    stats: dict = {}
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        series = df[col]
        non_nan = series.dropna().values
        if len(non_nan) == 0:
            continue
        median = float(np.median(non_nan))
        imputed = series.fillna(median).values
        stats[col] = {
            "median":        median,
            "mean":          float(np.mean(imputed)),
            "std":           float(np.std(imputed)),
            "min":           float(np.min(imputed)),
            "max":           float(np.max(imputed)),
            "count":         int(len(imputed)),
            "count_non_nan": int(len(non_nan)),
        }

    # Save JSON
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ Saved statistics for {len(stats)} features → {out}")
    print("=" * 80)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute feature statistics from parquet (no packs needed)"
    )
    parser.add_argument(
        "--features_parquet",
        required=True,
        help="Path to features parquet (use the minimal oracle-ct parquet)",
    )
    parser.add_argument(
        "--train_ids",
        default=None,
        help="Path to train_ids.txt — only compute stats on training cases",
    )
    parser.add_argument(
        "--output",
        default="feature_stats.json",
        help="Output path for statistics JSON",
    )
    args = parser.parse_args()

    stats = compute_feature_statistics(
        features_parquet=args.features_parquet,
        output_path=args.output,
        train_ids_file=args.train_ids,
    )

    # Print sample stats for key oracle-ct features
    key_features = [
        "liver_mean_hu",
        "liver_to_body_ratio",
        "spleen_mean_hu",
        "liver_spleen_hu_diff",
        "aorta_hu_mean",
        "lumbar_hu_mean",
    ]
    print("\nSample Feature Statistics:")
    print("=" * 80)
    for feat in key_features:
        if feat in stats:
            s = stats[feat]
            print(f"\n  {feat}:")
            print(f"    mean={s['mean']:.3f}  std={s['std']:.3f}")
            print(f"    min={s['min']:.3f}  max={s['max']:.3f}")
            print(f"    count={s['count']}  (non-nan={s['count_non_nan']})")


if __name__ == "__main__":
    main()
