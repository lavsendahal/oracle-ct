#!/usr/bin/env python3
"""
ablation/plot_perturbation_robustness.py

Two-panel robustness figure (one per dataset) showing how macro-AUROC
degrades under increasing scalar perturbation for three models:
  - JANUS GatedFusion
  - ScalarFusion (additive baseline)
  - ViT-Baseline (no scalars, flat reference line)

Reads directly from ablation_summary_merlin/duke.csv produced by
collect_ablation_results.py.

Usage:
    python janus/ablation/plot_perturbation_robustness.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score

# ── Style — consistent with plot_gate_aaa.py ──────────────────────────────────
plt.rcParams.update({
    'font.family'      : 'sans-serif',
    'font.sans-serif'  : ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size'        : 16,
    'axes.labelsize'   : 18,
    'axes.titlesize'   : 19,
    'legend.fontsize'  : 12,
    'xtick.labelsize'  : 16,
    'ytick.labelsize'  : 16,
    'figure.dpi'       : 300,
    'savefig.dpi'      : 300,
    'axes.linewidth'   : 1.2,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})

COLOR_GATED  = '#1976D2'   # blue  — GatedFusion (JANUS)
COLOR_SCALAR = '#D32F2F'   # red   — ScalarFusion
COLOR_VIT    = '#F57C00'   # amber — ViT-Baseline (no scalars)

# ── Paths ─────────────────────────────────────────────────────────────────────
ABLATION_ROOT = Path("/scratch/railabs/ld258/output/ct_triage/janus/ablation")
OUTPUT_DIR    = Path("/scratch/railabs/ld258/output/ct_triage/janus/miccai2026/figures")

GAP_PREDICTIONS = {
    "merlin": "/scratch/railabs/ld258/output/ct_triage/janus/runs/JanusGAP/2026-02-10_16-42-53_seed25/merlin/test_predictions.csv",
    "duke"  : "/scratch/railabs/ld258/output/ct_triage/janus/runs/JanusGAP/2026-02-10_16-42-53_seed25/duke/test_predictions.csv",
}
LABELS = {
    "merlin": "/scratch/railabs/ld258/dataset/merlin/merlinabdominalctdataset/zero_shot_findings_disease_cls.csv",
    "duke"  : "/scratch/railabs/ld258/output/ct_triage/macroradiomics/duke/duke_disease_labels_consensus.csv",
}
PANEL_TITLES = {
    "merlin": "Internal Cohort (MERLIN)",
    "duke"  : "External Dataset (Private)",
}


def compute_macro_auroc(preds_csv: str, labels_csv: str) -> float:
    probs  = pd.read_csv(preds_csv, index_col="case_id")
    df_lbl = pd.read_csv(labels_csv)
    id_col = "case_id" if "case_id" in df_lbl.columns else "study id"
    df_lbl = df_lbl.set_index(id_col).reindex(probs.index)

    aurocs = []
    for disease in probs.columns:
        if disease not in df_lbl.columns:
            continue
        mask = df_lbl[disease].notna() & (df_lbl[disease] != -1)
        if mask.sum() < 10:
            continue
        y_true  = df_lbl[disease][mask].values.astype(int)
        y_score = probs[disease][mask].values
        if len(np.unique(y_true)) < 2:
            continue
        try:
            aurocs.append(roc_auc_score(y_true, y_score))
        except Exception:
            pass
    return float(np.mean(aurocs)) if aurocs else np.nan


def plot_panel(ax, df: pd.DataFrame, vit_auroc: float, title: str):
    pcts = sorted(df["pct"].unique())

    for model, color, label, ls, marker in [
        ("GatedFusion",  COLOR_GATED,  "JANUS (Gated Fusion)",    "-",  "o"),
        ("ScalarFusion", COLOR_SCALAR, "ORACLE-CT + OSF",         "--", "s"),
    ]:
        sub = df[df["model"] == model].sort_values("pct")
        means = sub["macro_auroc_mean"].values
        stds  = sub["macro_auroc_std"].values
        xs    = sub["pct"].values

        ax.plot(xs, means, color=color, linewidth=2.5, linestyle=ls,
                marker=marker, markersize=8, zorder=4, label=label)

        # Shaded std band (visible once >1 seed)
        if stds.max() > 0:
            ax.fill_between(xs, means - stds, means + stds,
                            color=color, alpha=0.15, zorder=3)

        # Annotate 50% value
        ax.annotate(f"{means[-1]:.2f}",
                    xy=(xs[-1], means[-1]),
                    xytext=(5, 4 if model == "GatedFusion" else -14),
                    textcoords="offset points",
                    fontsize=11, color=color, fontweight="bold")

    # ViT-Baseline flat reference
    ax.axhline(vit_auroc, color=COLOR_VIT, linewidth=2.0,
               linestyle=":", zorder=2,
               label=f"ViT-Baseline ({vit_auroc:.2f})")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Scalar Perturbation (%)", fontweight="bold")
    ax.set_ylabel("AUROC", fontweight="bold")
    ax.set_xticks(pcts)
    ax.set_xlim(-3, 55)

    all_vals = df["macro_auroc_mean"].values
    ymin = min(all_vals.min(), vit_auroc) - 0.015
    ymax = max(all_vals.max(), vit_auroc) + 0.015
    ax.set_ylim(ymin, ymax)

    ax.legend(loc="lower left", framealpha=0.95,
              edgecolor="#CCCCCC", fancybox=True)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.6)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")

    for ax, dataset in zip(axes, ["merlin", "duke"]):
        csv_path = ABLATION_ROOT / f"ablation_summary_{dataset}.csv"
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found — skipping {dataset}")
            continue

        df = pd.read_csv(csv_path)

        # Compute ViT-Baseline AUROC from GAP predictions
        vit_auroc = compute_macro_auroc(GAP_PREDICTIONS[dataset], LABELS[dataset])
        print(f"{dataset} ViT-Baseline AUROC: {vit_auroc:.2f}")

        plot_panel(ax, df, vit_auroc, PANEL_TITLES[dataset])

    plt.tight_layout(pad=2.5)

    for fmt in ["pdf", "png", "svg"]:
        p = OUTPUT_DIR / f"perturbation_robustness.{fmt}"
        fig.savefig(p, format=fmt, bbox_inches="tight", facecolor="white",
                    **({"dpi": 300} if fmt == "png" else {}))
        print(f"Saved: {p}")

    plt.close(fig)


if __name__ == "__main__":
    main()
