#!/usr/bin/env python3
"""
Prepare train/val/test splits from your existing split files.

Usage:
    python prepare_splits.py \
        --train_csv /path/to/train_labels.csv \
        --val_csv /path/to/val_labels.csv \
        --test_csv /path/to/test_labels.csv \
        --output_dir ./splits
"""

import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--id_col", type=str, default="study id",
                        help="Column name containing case IDs")
    parser.add_argument("--output_dir", type=str, default="./splits")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read CSVs and extract IDs
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    train_ids = train_df[args.id_col].tolist()
    val_ids = val_df[args.id_col].tolist()

    # Write train IDs
    with open(output_dir / "train_ids.txt", "w") as f:
        for case_id in train_ids:
            f.write(f"{case_id}\n")
    print(f"Wrote {len(train_ids)} train IDs to {output_dir / 'train_ids.txt'}")

    # Write val IDs
    with open(output_dir / "val_ids.txt", "w") as f:
        for case_id in val_ids:
            f.write(f"{case_id}\n")
    print(f"Wrote {len(val_ids)} val IDs to {output_dir / 'val_ids.txt'}")

    # Write test IDs if provided
    if args.test_csv:
        test_df = pd.read_csv(args.test_csv)
        test_ids = test_df[args.id_col].tolist()
        with open(output_dir / "test_ids.txt", "w") as f:
            for case_id in test_ids:
                f.write(f"{case_id}\n")
        print(f"Wrote {len(test_ids)} test IDs to {output_dir / 'test_ids.txt'}")


if __name__ == "__main__":
    main()
