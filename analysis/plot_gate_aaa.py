#!/usr/bin/env python3
"""
plot_gate_aaa.py — Deep-dive gate analysis for Abdominal Aortic Aneurysm

Three-panel figure:
  Panel A: Violin — gate_mean by label (neg vs pos)
  Panel B: Scatter — gate_mean vs aorta_vessel_diam_max, coloured by label,
            with within-group OLS regression lines
  Panel C: OLS summary — gate ~ diameter + label (β coefficients + 95% CI)

Statistical outputs printed and saved to gate_aaa_stats.csv

Usage:
    python analysis/plot_gate_aaa.py
    python analysis/plot_gate_aaa.py --dataset duke
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

# ── Paths ─────────────────────────────────────────────────────────────────────
RUNS_ROOT   = Path("/scratch/railabs/ld258/output/ct_triage/janus/runs")
OUTPUT_DIR  = Path("/scratch/railabs/ld258/output/ct_triage/janus/miccai2026/gate_analysis")

MERLIN_LABELS   = Path("/scratch/railabs/ld258/dataset/merlin/merlinabdominalctdataset/zero_shot_findings_disease_cls.csv")
MERLIN_FEATURES = Path("/scratch/railabs/ld258/output/ct_triage/macroradiomics/merlin/features/features_combined.parquet")

DUKE_LABELS   = Path("/scratch/railabs/ld258/output/ct_triage/macroradiomics/duke/duke_disease_labels_consensus.csv")
DUKE_FEATURES = Path("/scratch/railabs/ld258/output/ct_triage/macroradiomics/duke/features/features_combined.parquet")

GATED_RUN   = RUNS_ROOT / "JanusGatedFusion/2026-02-10_11-58-42_seed25"
DISEASE     = "abdominal_aortic_aneurysm"
SCALAR      = "aorta_vessel_diam_max"

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200,
})

COLOR_NEG = "#4878CF"
COLOR_POS = "#D65F5F"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(dataset: str):
    gates_csv     = GATED_RUN / dataset / "test_gate_means.csv"
    labels_path   = MERLIN_LABELS   if dataset == "merlin" else DUKE_LABELS
    features_path = MERLIN_FEATURES if dataset == "merlin" else DUKE_FEATURES

    if not gates_csv.exists():
        raise FileNotFoundError(f"Gate means not found: {gates_csv}\n"
                                f"Run inference with +inference.return_gates=true first.")

    df_gates = pd.read_csv(gates_csv, index_col="case_id")
    case_ids = df_gates.index.tolist()

    gate_col = f"gate_mean_{DISEASE}"
    if gate_col not in df_gates.columns:
        raise KeyError(f"Column '{gate_col}' not in gate means CSV.")
    gate = df_gates[gate_col].values.astype(float)

    # Labels
    df_lbl = pd.read_csv(labels_path)
    id_col = "case_id" if "case_id" in df_lbl.columns else "study id"
    df_lbl = df_lbl.set_index(id_col).reindex(case_ids)
    labels = df_lbl[DISEASE].replace(-1, np.nan).values.astype(float) if DISEASE in df_lbl.columns \
             else np.full(len(case_ids), np.nan)

    # Scalar features
    df_feat = pd.read_parquet(features_path)
    id_col2 = "case_id" if "case_id" in df_feat.columns else df_feat.index.name or df_feat.columns[0]
    if id_col2 in df_feat.columns:
        df_feat = df_feat.set_index(id_col2)
    df_feat = df_feat.reindex(case_ids)
    if SCALAR not in df_feat.columns:
        raise KeyError(f"Scalar '{SCALAR}' not found. Available: {[c for c in df_feat.columns if 'aorta' in c]}")
    scalar = df_feat[SCALAR].values.astype(float)

    return gate, labels, scalar, case_ids


def ols_within_group(gate, scalar, labels, group_val, label_name):
    """OLS: gate ~ scalar within one label group."""
    mask = (labels == group_val) & ~np.isnan(gate) & ~np.isnan(scalar)
    g, s = gate[mask], scalar[mask]
    if len(g) < 5:
        return None
    X = sm.add_constant(s)
    model = sm.OLS(g, X).fit()
    r, p = stats.spearmanr(s, g)
    return {"n": len(g), "beta": model.params[1], "beta_se": model.bse[1],
            "pval": model.pvalues[1], "spearman_r": r, "spearman_p": p,
            "model": model, "g": g, "s": s, "label": label_name}


def ols_full(gate, scalar, labels):
    """OLS: gate ~ scalar + label (controls for disease status)."""
    mask = ~np.isnan(gate) & ~np.isnan(scalar) & ~np.isnan(labels)
    g, s, lbl = gate[mask], scalar[mask], labels[mask]
    X = sm.add_constant(np.column_stack([s, lbl]))
    model = sm.OLS(g, X).fit()
    return model, g, s, lbl


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_figure(gate, labels, scalar, dataset, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_valid = ~np.isnan(labels)
    gate_neg = gate[(labels == 0)]
    gate_pos = gate[(labels == 1)]
    scalar_neg = scalar[(labels == 0) & ~np.isnan(scalar)]
    scalar_pos = scalar[(labels == 1) & ~np.isnan(scalar)]

    n_neg, n_pos = len(gate_neg), len(gate_pos)
    mean_neg, mean_pos = np.nanmean(gate_neg), np.nanmean(gate_pos)

    # AUC
    mask_auc = ~np.isnan(labels) & ~np.isnan(gate)
    auc_raw = roc_auc_score(labels[mask_auc], gate[mask_auc])
    sep_auc = max(auc_raw, 1 - auc_raw)
    direction = "suppressed" if mean_pos < mean_neg else "activated"

    # Spearman gate vs label
    r_lbl, p_lbl = stats.spearmanr(gate[mask_auc], labels[mask_auc])

    # Within-group OLS
    res_neg = ols_within_group(gate, scalar, labels, 0, "Negative")
    res_pos = ols_within_group(gate, scalar, labels, 1, "Positive")

    # Full OLS: gate ~ diameter + label
    full_model, g_full, s_full, lbl_full = ols_full(gate, scalar, labels)

    # ── Print stats ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"AAA Gate Analysis  |  dataset={dataset}")
    print(f"{'='*65}")
    print(f"  N negative: {n_neg}    N positive: {n_pos}")
    print(f"  Mean gate (neg): {mean_neg:.4f}    Mean gate (pos): {mean_pos:.4f}")
    print(f"  Gate direction: {direction}")
    print(f"  AUC (signed): {auc_raw:.4f}    sep_AUC: {sep_auc:.4f}")
    print(f"  Spearman(gate, label): ρ={r_lbl:.3f}  p={p_lbl:.4f} {sig_stars(p_lbl)}")
    print()
    if res_neg:
        print(f"  Within-group OLS (Negative, n={res_neg['n']}):")
        print(f"    β_diameter = {res_neg['beta']:.5f}  SE={res_neg['beta_se']:.5f}  "
              f"p={res_neg['pval']:.4f} {sig_stars(res_neg['pval'])}")
        print(f"    Spearman ρ = {res_neg['spearman_r']:.3f}  p={res_neg['spearman_p']:.4f}")
    if res_pos:
        print(f"  Within-group OLS (Positive, n={res_pos['n']}):")
        print(f"    β_diameter = {res_pos['beta']:.5f}  SE={res_pos['beta_se']:.5f}  "
              f"p={res_pos['pval']:.4f} {sig_stars(res_pos['pval'])}")
        print(f"    Spearman ρ = {res_pos['spearman_r']:.3f}  p={res_pos['spearman_p']:.4f}")
    print()
    print("  Full OLS: gate ~ diameter + label")
    print(f"    β_intercept = {full_model.params[0]:.4f}")
    print(f"    β_diameter  = {full_model.params[1]:.5f}  "
          f"p={full_model.pvalues[1]:.4f} {sig_stars(full_model.pvalues[1])}")
    print(f"    β_label     = {full_model.params[2]:.5f}  "
          f"p={full_model.pvalues[2]:.4f} {sig_stars(full_model.pvalues[2])}")
    print(f"    R²  = {full_model.rsquared:.4f}")

    # ── Save stats CSV ────────────────────────────────────────────────────────
    stats_rows = [
        {"stat": "n_negative",             "value": n_neg},
        {"stat": "n_positive",             "value": n_pos},
        {"stat": "mean_gate_neg",          "value": round(mean_neg, 4)},
        {"stat": "mean_gate_pos",          "value": round(mean_pos, 4)},
        {"stat": "gate_direction",         "value": direction},
        {"stat": "auc_signed",             "value": round(auc_raw, 4)},
        {"stat": "sep_auc",                "value": round(sep_auc, 4)},
        {"stat": "spearman_gate_label_r",  "value": round(r_lbl, 4)},
        {"stat": "spearman_gate_label_p",  "value": round(p_lbl, 4)},
    ]
    if res_neg:
        stats_rows += [
            {"stat": "neg_beta_diameter",  "value": round(res_neg["beta"], 6)},
            {"stat": "neg_beta_se",        "value": round(res_neg["beta_se"], 6)},
            {"stat": "neg_beta_p",         "value": round(res_neg["pval"], 4)},
            {"stat": "neg_spearman_r",     "value": round(res_neg["spearman_r"], 4)},
        ]
    if res_pos:
        stats_rows += [
            {"stat": "pos_beta_diameter",  "value": round(res_pos["beta"], 6)},
            {"stat": "pos_beta_se",        "value": round(res_pos["beta_se"], 6)},
            {"stat": "pos_beta_p",         "value": round(res_pos["pval"], 4)},
            {"stat": "pos_spearman_r",     "value": round(res_pos["spearman_r"], 4)},
        ]
    stats_rows += [
        {"stat": "full_ols_beta_intercept", "value": round(full_model.params[0], 4)},
        {"stat": "full_ols_beta_diameter",  "value": round(full_model.params[1], 6)},
        {"stat": "full_ols_beta_diameter_p","value": round(full_model.pvalues[1], 4)},
        {"stat": "full_ols_beta_label",     "value": round(full_model.params[2], 4)},
        {"stat": "full_ols_beta_label_p",   "value": round(full_model.pvalues[2], 4)},
        {"stat": "full_ols_r2",             "value": round(full_model.rsquared, 4)},
    ]
    pd.DataFrame(stats_rows).to_csv(out_dir / f"gate_aaa_stats_{dataset}.csv", index=False)

    # ── Figure ────────────────────────────────────────────────────────────────
    scalar_display = "Aorta Vessel Maximum Diameter"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor="white")

    # ── Panel A: Violin ───────────────────────────────────────────────────────
    parts = ax1.violinplot(
        [gate_neg[~np.isnan(gate_neg)], gate_pos[~np.isnan(gate_pos)]],
        positions=[0, 1], showmedians=True, showextrema=True,
    )
    for pc, color in zip(parts["bodies"], [COLOR_NEG, COLOR_POS]):
        pc.set_facecolor(color); pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cbars"].set_color("gray")
    parts["cmaxes"].set_color("gray")
    parts["cmins"].set_color("gray")

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([f"Negative\n(n={n_neg})", f"Positive\n(n={n_pos})"], fontsize=12)
    ax1.set_ylabel("Mean Gate Activation", fontsize=12)
    ax1.set_ylim(0, 1)
    dir_arrow = "↓" if direction == "suppressed" else "↑"
    ax1.set_title(f"Gate by Label\nGate Separability AUC={sep_auc:.3f} ({dir_arrow}{direction})",
                  fontsize=12, fontweight="bold")
    ax1.axhline(0.5, ls="--", color="gray", lw=0.8, alpha=0.5)
    ax1.set_facecolor("#f8f8f8")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel B: Scatter + within-group regression ────────────────────────────
    for lbl_val, color, res, lbl_name in [
        (0, COLOR_NEG, res_neg, "Negative"),
        (1, COLOR_POS, res_pos, "Positive"),
    ]:
        m = (labels == lbl_val) & ~np.isnan(scalar) & ~np.isnan(gate)
        ax2.scatter(scalar[m], gate[m], c=color, alpha=0.4, s=18,
                    edgecolors="none", label=lbl_name)
        if res:
            x_line = np.linspace(scalar[m].min(), scalar[m].max(), 100)
            y_line = res["model"].params[0] + res["model"].params[1] * x_line
            p_str = "p<0.001" if res["pval"] < 0.001 else f"p={res['pval']:.3f}"
            ax2.plot(x_line, y_line, color=color, linewidth=2.2,
                     label=f"{lbl_name}  (ρ={res['spearman_r']:.2f}, {p_str})")

    ax2.set_xlabel(scalar_display, fontsize=12)
    ax2.set_ylabel("Mean Gate Activation", fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.set_title(f"Gate vs {scalar_display}\nWithin-group OLS regression",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=12, framealpha=0.9)
    ax2.set_facecolor("#f8f8f8")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(pad=2.5)

    png_path = out_dir / f"gate_aaa_detailed_{dataset}.png"
    pdf_path = out_dir / f"gate_aaa_detailed_{dataset}.pdf"
    svg_path = out_dir / f"gate_aaa_detailed_{dataset}.svg"
    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {svg_path}")
    return png_path


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="merlin", choices=["merlin", "duke"])
    args = parser.parse_args()

    out_dir = OUTPUT_DIR / args.dataset / "test"
    gate, labels, scalar, case_ids = load_data(args.dataset)
    make_figure(gate, labels, scalar, args.dataset, out_dir)
