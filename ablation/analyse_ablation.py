#!/usr/bin/env python3
"""
ablation/analyse_ablation.py

Compares JANUS predictions under baseline (unperturbed) vs perturbed scalar
features. Reports per-disease and macro AUC degradation across seeds and
produces a publication-quality bar chart.

Usage:
    python janus/ablation/analyse_ablation.py \
        --baseline_csv  /path/to/ablation/results/merlin/baseline/test_predictions.csv \
        --ablation_dir  /path/to/ablation/results/merlin \
        --labels_csv    /path/to/zero_shot_findings_disease_cls.csv \
        --out_dir       /path/to/ablation/results/merlin/analysis

Outputs:
    ablation_per_disease.csv   — per-disease baseline vs perturbed AUC
    ablation_summary.json      — macro AUC summary
    ablation_delta_auc.png/pdf/svg
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score


# ── Publication style (matches physiological_veto figure) ─────────────────────
plt.rcParams.update({
    'font.family'      : 'sans-serif',
    'font.sans-serif'  : ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size'        : 16,
    'axes.labelsize'   : 18,
    'axes.titlesize'   : 19,
    'legend.fontsize'  : 12,
    'xtick.labelsize'  : 14,
    'ytick.labelsize'  : 14,
    'figure.dpi'       : 300,
    'savefig.dpi'      : 300,
    'axes.linewidth'   : 1.2,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})

COLOR_NEG = '#D32F2F'   # red  — degraded
COLOR_POS = '#1976D2'   # blue — improved / neutral


def load_predictions(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, index_col="case_id")


def load_labels(labels_csv: str, case_ids: list, diseases: list) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    id_col = "case_id" if "case_id" in df.columns else "study id"
    df = df.set_index(id_col).reindex(case_ids)
    # Keep only columns that exist
    available = [d for d in diseases if d in df.columns]
    return df[available]


def compute_aucs(probs: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Compute per-disease AUC, skipping diseases with <10 samples or one class."""
    aucs = {}
    for disease in probs.columns:
        if disease not in labels.columns:
            continue
        mask = labels[disease].notna() & (labels[disease] != -1)
        if mask.sum() < 10:
            continue
        y_true  = labels[disease][mask].values.astype(int)
        y_score = probs[disease][mask].values
        if len(np.unique(y_true)) < 2:
            continue
        try:
            aucs[disease] = roc_auc_score(y_true, y_score)
        except Exception:
            pass
    return aucs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_csv", required=True,
                        help="Baseline test_predictions.csv (unperturbed)")
    parser.add_argument("--ablation_dir", required=True,
                        help="Directory containing seed*/test_predictions.csv")
    parser.add_argument("--labels_csv",   required=True,
                        help="Ground-truth labels CSV")
    parser.add_argument("--out_dir",      required=True,
                        help="Where to write analysis outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ablation_dir = Path(args.ablation_dir)

    # ── Load baseline ──────────────────────────────────────────────────────────
    baseline_probs = load_predictions(args.baseline_csv)
    case_ids = baseline_probs.index.tolist()
    diseases = baseline_probs.columns.tolist()

    labels = load_labels(args.labels_csv, case_ids, diseases)

    baseline_aucs   = compute_aucs(baseline_probs, labels)
    macro_baseline  = np.mean(list(baseline_aucs.values()))
    print(f"Baseline macro AUC : {macro_baseline:.4f}  ({len(baseline_aucs)} diseases)")

    # ── Find all seed CSVs ─────────────────────────────────────────────────────
    # Layout: seed{N}/<dataset>/test_predictions.csv  (set by hydra.run.dir)
    seed_csvs = sorted(ablation_dir.glob("seed*/*/test_predictions.csv"))
    if not seed_csvs:
        raise FileNotFoundError(
            f"No seed*/<dataset>/test_predictions.csv found in {ablation_dir}\n"
            "Run run_ablation.sh first."
        )
    print(f"Ablation seeds found: {len(seed_csvs)}")

    # ── Per-seed AUCs ──────────────────────────────────────────────────────────
    all_seed_aucs = []
    for csv_path in seed_csvs:
        probs = load_predictions(str(csv_path)).reindex(case_ids)
        all_seed_aucs.append(compute_aucs(probs, labels))

    # ── Per-disease stats ──────────────────────────────────────────────────────
    rows = []
    for disease in sorted(baseline_aucs.keys()):
        seed_vals = [s[disease] for s in all_seed_aucs if disease in s]
        if not seed_vals:
            continue
        mean_p = np.mean(seed_vals)
        std_p  = np.std(seed_vals)
        delta  = mean_p - baseline_aucs[disease]
        rows.append({
            "disease"             : disease,
            "baseline_auc"        : round(baseline_aucs[disease], 4),
            "perturbed_auc_mean"  : round(mean_p, 4),
            "perturbed_auc_std"   : round(std_p,  4),
            "delta_auc"           : round(delta,  4),
        })

    results_df = pd.DataFrame(rows).sort_values("delta_auc")
    csv_out = out_dir / "ablation_per_disease.csv"
    results_df.to_csv(csv_out, index=False)

    print(f"\nPer-disease results:\n{results_df.to_string(index=False)}")
    print(f"\nSaved: {csv_out}")

    # ── Macro summary ──────────────────────────────────────────────────────────
    macro_per_seed = [np.mean(list(s.values())) for s in all_seed_aucs]
    macro_mean     = np.mean(macro_per_seed)
    macro_std      = np.std(macro_per_seed)
    macro_delta    = macro_mean - macro_baseline

    summary = {
        "baseline_macro_auc"       : round(macro_baseline, 4),
        "perturbed_macro_auc_mean" : round(macro_mean,     4),
        "perturbed_macro_auc_std"  : round(macro_std,      4),
        "macro_delta_auc"          : round(macro_delta,    4),
        "n_seeds"                  : len(seed_csvs),
        "n_diseases"               : len(baseline_aucs),
    }
    with open(out_dir / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMacro AUC: {macro_baseline:.4f} → {macro_mean:.4f} ± {macro_std:.4f}  "
          f"(Δ = {macro_delta:+.4f})")
    print(f"Saved: {out_dir / 'ablation_summary.json'}")

    # ── Figure ─────────────────────────────────────────────────────────────────
    df_plot = results_df.sort_values("delta_auc")
    n       = len(df_plot)
    colors  = [COLOR_NEG if d < 0 else COLOR_POS for d in df_plot["delta_auc"]]

    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.38)), facecolor="white")

    ax.barh(df_plot["disease"], df_plot["delta_auc"],
            color=colors, alpha=0.80, zorder=2)
    ax.errorbar(df_plot["delta_auc"], df_plot["disease"],
                xerr=df_plot["perturbed_auc_std"],
                fmt="none", color="black", capsize=3, linewidth=1, zorder=3)
    ax.axvline(0, color="black", linewidth=0.9, linestyle="--", zorder=1)

    ax.set_xlabel("ΔAUC (Perturbed − Baseline)", fontweight="bold")
    ax.set_title(
        f"Scalar Perturbation Robustness (±20%)\n"
        f"Macro ΔAUC = {macro_delta:+.4f} ± {macro_std:.4f}  "
        f"({len(seed_csvs)} seeds)",
        fontweight="bold",
    )
    ax.grid(True, axis="x", alpha=0.3, linestyle="--", linewidth=0.6)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_NEG, alpha=0.8, label="AUC degraded"),
        Patch(facecolor=COLOR_POS, alpha=0.8, label="AUC preserved / improved"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9,
              edgecolor="#CCCCCC", fancybox=True)

    fig.tight_layout(pad=2.5)

    for fmt in ["png", "pdf", "svg"]:
        p = out_dir / f"ablation_delta_auc.{fmt}"
        fig.savefig(p, format=fmt, bbox_inches="tight", facecolor="white",
                    **({"dpi": 300} if fmt == "png" else {}))
        print(f"Saved: {p}")

    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
