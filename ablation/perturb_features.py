#!/usr/bin/env python3
"""
ablation/perturb_features.py

Creates perturbed versions of the scalar features parquet by multiplying
each numeric measurement by a random factor drawn from [1-pct, 1+pct].
This simulates segmentation failures where organ measurements are inaccurate.

Binary/flag columns (only contain {0, 1, NaN}) are left unchanged.
NaN values are left unchanged.

Usage:
    python janus/ablation/perturb_features.py \
        --parquet /path/to/features_combined.parquet \
        --out_dir /path/to/ablation/perturbed \
        --pct 0.20 \
        --seeds 1 2 3 4 5
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


def is_binary_column(series: pd.Series) -> bool:
    """True if column only contains values in {0, 1, NaN} — i.e. a flag column."""
    unique_vals = set(series.dropna().unique())
    return unique_vals.issubset({0, 1, 0.0, 1.0})


def perturb_parquet(
    parquet_path: str,
    out_dir: str,
    pct: float = 0.20,
    seeds: list = None,
):
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)

    # Standardise index
    if "case_id" in df.columns:
        df = df.set_index("case_id")

    # Identify columns to perturb: numeric but NOT binary flags
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    binary_cols  = [c for c in numeric_cols if is_binary_column(df[c])]
    perturb_cols = [c for c in numeric_cols if c not in binary_cols]

    print(f"Parquet: {parquet_path}")
    print(f"  Cases            : {len(df)}")
    print(f"  Numeric columns  : {len(numeric_cols)}")
    print(f"  Binary (skipped) : {len(binary_cols)}")
    print(f"  To perturb       : {len(perturb_cols)}")
    print(f"  Perturbation     : ±{int(pct * 100)}%  |  seeds: {seeds}\n")

    for seed in seeds:
        rng = np.random.default_rng(seed)
        df_perturbed = df.copy()

        for col in perturb_cols:
            col_data  = df_perturbed[col].values.copy().astype(float)
            nan_mask  = np.isnan(col_data)
            factors   = rng.uniform(1.0 - pct, 1.0 + pct, size=len(col_data))
            col_data  = col_data * factors
            col_data[nan_mask] = np.nan          # preserve original NaNs
            df_perturbed[col] = col_data

        out_path = out_dir / f"features_perturbed_pct{int(pct * 100):02d}_seed{seed}.parquet"
        df_perturbed.reset_index().to_parquet(out_path, index=False)
        print(f"  Saved seed {seed}: {out_path}")

    # Save metadata alongside the parquets for reproducibility
    meta = {
        "original_parquet"     : str(parquet_path),
        "pct"                  : pct,
        "seeds"                : seeds,
        "n_cases"              : len(df),
        "n_perturbed_cols"     : len(perturb_cols),
        "n_binary_cols_skipped": len(binary_cols),
        "binary_cols"          : binary_cols,
        "perturbed_cols"       : perturb_cols,
    }
    meta_path = out_dir / f"perturbation_meta_pct{int(pct * 100):02d}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Metadata: {meta_path}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True,
                        help="Path to features_combined.parquet")
    parser.add_argument("--out_dir", required=True,
                        help="Directory to write perturbed parquets")
    parser.add_argument("--pct",   type=float, default=0.20,
                        help="Perturbation fraction (default 0.20 = ±20%%)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="Random seeds (default: 1 2 3 4 5)")
    args = parser.parse_args()

    perturb_parquet(
        parquet_path=args.parquet,
        out_dir=args.out_dir,
        pct=args.pct,
        seeds=args.seeds,
    )
