#!/usr/bin/env python3
"""
Compute Feature Statistics (No Packs Needed!)

Computes mean, std, min, max for all features directly from parquet.
Required for z-score normalization in scalar fusion model.

Usage:
    python compute_feature_stats.py \
        --features_parquet /data/features.parquet \
        --train_ids /data/splits/train_ids.txt \
        --output feature_stats.json
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from janus.datamodules.feature_bank import compute_feature_statistics


def main():
    parser = argparse.ArgumentParser(
        description="Compute feature statistics from parquet (no packs needed!)"
    )
    parser.add_argument(
        "--features_parquet",
        type=str,
        required=True,
        help="Path to features parquet file"
    )
    parser.add_argument(
        "--train_ids",
        type=str,
        default=None,
        help="Path to train_ids.txt (optional - only compute stats on training data)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="feature_stats.json",
        help="Output path for statistics JSON"
    )
    args = parser.parse_args()

    stats = compute_feature_statistics(
        features_parquet=args.features_parquet,
        output_path=args.output,
        train_ids_file=args.train_ids,
    )

    # Print some key stats
    key_features = [
        "liver_vol_cc",
        "liver_to_body_ratio",
        "liver_mean_hu",
        "spleen_mean_hu",
        "liver_spleen_hu_diff",
    ]

    print("\nSample Feature Statistics:")
    print("=" * 80)
    for feat in key_features:
        if feat in stats:
            s = stats[feat]
            print(f"\n{feat}:")
            print(f"  mean={s['mean']:.3f}, std={s['std']:.3f}")
            print(f"  min={s['min']:.3f}, max={s['max']:.3f}")
            print(f"  count={s['count']}")


if __name__ == "__main__":
    main()
