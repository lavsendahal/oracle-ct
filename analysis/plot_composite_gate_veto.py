#!/usr/bin/env python3
"""
plot_composite_gate_veto.py — Composite 3-panel figure for MICCAI 2026

Combines (for the Duke / external cohort only):
  (a) AAA gate by label — violin
  (b) AAA gate vs Aorta Vessel Maximum Diameter — scatter + within-group OLS
  (c) Physiological veto — overconfident FP suppression

Usage:
    python analysis/plot_composite_gate_veto.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

# ── Paths ─────────────────────────────────────────────────────────────────────
RUNS_ROOT   = Path("/scratch/railabs/ld258/output/ct_triage/janus/runs")
OUTPUT_DIR  = Path("/scratch/railabs/ld258/output/ct_triage/janus/miccai2026/figures")

DUKE_LABELS   = Path("/scratch/railabs/ld258/output/ct_triage/macroradiomics/duke/duke_disease_labels_consensus.csv")
DUKE_FEATURES = Path("/scratch/railabs/ld258/output/ct_triage/macroradiomics/duke/features/features_combined.parquet")

GAP_RUN    = RUNS_ROOT / "JanusGAP/2026-02-10_16-42-53_seed25"
GATED_RUN  = RUNS_ROOT / "JanusGatedFusion/2026-02-10_11-58-42_seed25"

DISEASE = "abdominal_aortic_aneurysm"
SCALAR  = "aorta_vessel_diam_max"
SCALAR_DISPLAY = "Aorta Vessel Maximum Diameter"

# ── Colors ────────────────────────────────────────────────────────────────────
COLOR_NEG   = "#4878CF"   # blue  — negative / ViT-Baseline
COLOR_POS   = "#D65F5F"   # red   — positive
COLOR_GAP   = "#D32F2F"   # deep red for baseline KDE
COLOR_GATED = "#1976D2"   # deep blue for JANUS KDE

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     13,
    "axes.linewidth":     1.2,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "legend.fontsize":    9,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "figure.dpi":         200,
})


# ── AAA helpers ───────────────────────────────────────────────────────────────

def load_aaa_data():
    gates_csv = GATED_RUN / "duke" / "test_gate_means.csv"
    if not gates_csv.exists():
        raise FileNotFoundError(f"Gate means not found: {gates_csv}")

    df_gates = pd.read_csv(gates_csv, index_col="case_id")
    case_ids = df_gates.index.tolist()

    gate_col = f"gate_mean_{DISEASE}"
    if gate_col not in df_gates.columns:
        raise KeyError(f"Column '{gate_col}' not in gate means CSV.")
    gate = df_gates[gate_col].values.astype(float)

    # Labels
    df_lbl = pd.read_csv(DUKE_LABELS).set_index("case_id").reindex(case_ids)
    labels = df_lbl[DISEASE].replace(-1, np.nan).values.astype(float) \
             if DISEASE in df_lbl.columns else np.full(len(case_ids), np.nan)

    # Scalar
    df_feat = pd.read_parquet(DUKE_FEATURES)
    if "case_id" in df_feat.columns:
        df_feat = df_feat.set_index("case_id")
    df_feat = df_feat.reindex(case_ids)
    if SCALAR not in df_feat.columns:
        raise KeyError(f"Scalar '{SCALAR}' not found.")
    scalar = df_feat[SCALAR].values.astype(float)

    return gate, labels, scalar


def ols_within_group(gate, scalar, labels, group_val, label_name):
    mask = (labels == group_val) & ~np.isnan(gate) & ~np.isnan(scalar)
    g, s = gate[mask], scalar[mask]
    if len(g) < 5:
        return None
    X = sm.add_constant(s)
    model = sm.OLS(g, X).fit()
    r, p = stats.spearmanr(s, g)
    return {"n": len(g), "model": model, "spearman_r": r, "pval": p}


# ── Veto helpers ──────────────────────────────────────────────────────────────

def load_labels_veto(labels_path, case_ids, diseases):
    df = pd.read_csv(labels_path)
    id_col = "case_id" if "case_id" in df.columns else "study id"
    df = df.set_index(id_col).reindex(case_ids)
    labels = np.full((len(case_ids), len(diseases)), np.nan)
    for i, d in enumerate(diseases):
        if d in df.columns:
            labels[:, i] = df[d].values.astype(float)
    labels[labels == -1] = np.nan
    return labels


def load_veto_data():
    gap_df = pd.read_csv(GAP_RUN / "duke" / "test_predictions.csv")
    gap_df = gap_df.drop_duplicates(subset=["case_id"], keep="first").set_index("case_id")

    gated_df = pd.read_csv(GATED_RUN / "duke" / "test_predictions.csv")
    gated_df = gated_df.drop_duplicates(subset=["case_id"], keep="first").set_index("case_id")

    labels_df = pd.read_csv(DUKE_LABELS).set_index("case_id")
    common = gap_df.index.intersection(gated_df.index).intersection(labels_df.index).tolist()

    gap_df   = gap_df.loc[common]
    gated_df = gated_df.loc[common]
    diseases = list(gap_df.columns)
    labels   = load_labels_veto(DUKE_LABELS, common, diseases)

    all_gap, all_gated, all_lbl = [], [], []
    for d_idx, d in enumerate(diseases):
        y = labels[:, d_idx]
        vm = ~np.isnan(y)
        if vm.sum() < 10:
            continue
        all_gap.extend(gap_df.loc[common, d].values[vm])
        all_gated.extend(gated_df.loc[common, d].values[vm])
        all_lbl.extend(y[vm].astype(int))

    return np.array(all_lbl), np.array(all_gap), np.array(all_gated)


def compute_veto_stats(all_labels, all_gap_probs, all_gated_probs):
    fp_mask = (all_labels == 0) & (all_gap_probs >= 0.8)
    fp_gap   = all_gap_probs[fp_mask]
    fp_gated = all_gated_probs[fp_mask]
    fp_vetoed     = (fp_gated < 0.5).sum()
    fp_veto_rate  = fp_vetoed / len(fp_gated) * 100 if len(fp_gated) > 0 else 0

    tp_mask = (all_labels == 1) & (all_gap_probs >= 0.8)
    tp_gap   = all_gap_probs[tp_mask]
    tp_gated = all_gated_probs[tp_mask]
    tp_suppressed    = (tp_gated < 0.5).sum()
    tp_suppress_rate = tp_suppressed / len(tp_gated) * 100 if len(tp_gated) > 0 else 0

    selectivity = fp_veto_rate / max(tp_suppress_rate, 0.01)
    return dict(
        fp_gap=fp_gap, fp_gated=fp_gated,
        fp_count=len(fp_gap), fp_vetoed=fp_vetoed, fp_veto_rate=fp_veto_rate,
        tp_count=len(tp_gated), tp_suppressed=tp_suppressed,
        tp_suppress_rate=tp_suppress_rate, selectivity=selectivity,
    )


# ── Panel drawers ─────────────────────────────────────────────────────────────

def draw_violin(ax, gate, labels):
    gate_neg = gate[labels == 0]
    gate_pos = gate[labels == 1]
    n_neg, n_pos = len(gate_neg), len(gate_pos)
    mean_neg, mean_pos = np.nanmean(gate_neg), np.nanmean(gate_pos)

    mask_auc = ~np.isnan(labels) & ~np.isnan(gate)
    auc_raw  = roc_auc_score(labels[mask_auc], gate[mask_auc])
    sep_auc  = max(auc_raw, 1 - auc_raw)
    direction = "suppressed" if mean_pos < mean_neg else "activated"

    parts = ax.violinplot(
        [gate_neg[~np.isnan(gate_neg)], gate_pos[~np.isnan(gate_pos)]],
        positions=[0, 1], showmedians=True, showextrema=True,
    )
    for pc, color in zip(parts["bodies"], [COLOR_NEG, COLOR_POS]):
        pc.set_facecolor(color); pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cbars"].set_color("gray")
    parts["cmaxes"].set_color("gray")
    parts["cmins"].set_color("gray")

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Negative\n(n={n_neg})", f"Positive\n(n={n_pos})"], fontsize=10)
    ax.set_ylabel("Mean Gate Activation", fontsize=11)
    ax.set_ylim(0, 1)
    dir_arrow = "↓" if direction == "suppressed" else "↑"
    ax.set_title(
        f"Gate by Label\nGate Separability AUC={sep_auc:.3f} ({dir_arrow}{direction})",
        fontsize=12, fontweight="bold",
    )
    ax.axhline(0.5, ls="--", color="gray", lw=0.8, alpha=0.5)
    ax.set_facecolor("#f8f8f8")


def draw_scatter(ax, gate, labels, scalar):
    res_neg = ols_within_group(gate, scalar, labels, 0, "Negative")
    res_pos = ols_within_group(gate, scalar, labels, 1, "Positive")

    for lbl_val, color, res, lbl_name in [
        (0, COLOR_NEG, res_neg, "Negative"),
        (1, COLOR_POS, res_pos, "Positive"),
    ]:
        m = (labels == lbl_val) & ~np.isnan(scalar) & ~np.isnan(gate)
        ax.scatter(scalar[m], gate[m], c=color, alpha=0.4, s=18,
                   edgecolors="none", label=lbl_name)
        if res:
            x_line = np.linspace(scalar[m].min(), scalar[m].max(), 100)
            y_line = res["model"].params[0] + res["model"].params[1] * x_line
            p_str = "p<0.001" if res["pval"] < 0.001 else f"p={res['pval']:.3f}"
            ax.plot(x_line, y_line, color=color, linewidth=2.2,
                    label=f"{lbl_name}  (ρ={res['spearman_r']:.2f}, {p_str})")

    ax.set_xlabel(SCALAR_DISPLAY, fontsize=11)
    ax.set_ylabel("Mean Gate Activation", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Gate vs {SCALAR_DISPLAY}\nWithin-group OLS regression",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_facecolor("#f8f8f8")


def draw_veto(ax, veto_stats):
    fp_gap        = veto_stats["fp_gap"]
    fp_gated      = veto_stats["fp_gated"]
    fp_veto_rate  = veto_stats["fp_veto_rate"]
    tp_suppress   = veto_stats["tp_suppress_rate"]
    selectivity   = veto_stats["selectivity"]

    ax.axvspan(0,   0.5, alpha=0.12, color="#4CAF50", zorder=0)
    ax.axvspan(0.8, 1.0, alpha=0.15, color="#F44336", zorder=0)

    bins = np.linspace(0, 1, 30)
    ax.hist(fp_gap,   bins=bins, alpha=0.55, color=COLOR_GAP,
            density=True, edgecolor="white", linewidth=0.5, zorder=2)
    ax.hist(fp_gated, bins=bins, alpha=0.55, color=COLOR_GATED,
            density=True, edgecolor="white", linewidth=0.5, zorder=2)

    if len(fp_gap) > 5:
        kde_gap   = stats.gaussian_kde(fp_gap,   bw_method=0.12)
        kde_gated = stats.gaussian_kde(fp_gated, bw_method=0.12)
        x_kde = np.linspace(0, 1, 200)
        ax.plot(x_kde, kde_gap(x_kde),   color=COLOR_GAP,   linewidth=3,
                label=f"ViT-Baseline (n={len(fp_gap):,})", zorder=3)
        ax.plot(x_kde, kde_gated(x_kde), color=COLOR_GATED, linewidth=3,
                label="JANUS (same cases)", zorder=3)

    ymax = ax.get_ylim()[1]

    ax.annotate("", xy=(0.22, ymax * 0.72), xytext=(0.88, ymax * 0.72),
                arrowprops=dict(arrowstyle="->", color=COLOR_GATED,
                                lw=4, mutation_scale=25), zorder=5)
    ax.text(0.55, ymax * 0.80, "Physiological Veto", fontsize=13,
            fontweight="bold", color=COLOR_GATED, ha="center", zorder=5)

    ax.text(0.25, ymax * 0.95, "Safe Zone\n(p < 0.5)", fontsize=9,
            color="#2E7D32", ha="center", fontweight="bold", alpha=0.9)
    ax.text(0.90, ymax * 0.95, "Danger\nZone", fontsize=9,
            color="#C62828", ha="center", fontweight="bold", alpha=0.9)

    props = dict(boxstyle="round,pad=0.4", facecolor="white",
                 edgecolor=COLOR_GATED, linewidth=2)
    ax.text(0.19, ymax * 0.58,
            f"{fp_veto_rate:.1f}% of FPs\nSuppressed\n\n"
            f"Only {tp_suppress:.1f}% of TPs\nAffected\n\n"
            f"Selectivity: {selectivity:.1f}\u00d7",
            fontsize=10, fontweight="bold", color=COLOR_GATED, ha="center",
            bbox=props, zorder=5)

    ax.set_xlabel("Predicted Probability", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="#CCCCCC",
              fancybox=True, bbox_to_anchor=(0.0, 0.27))
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, axis="y")
    ax.set_title("FP Suppression — External Cohort\n(Overconfident Baseline FPs, p ≥ 0.8)",
                 fontsize=12, fontweight="bold")
    ax.set_facecolor("#f8f8f8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading AAA gate data (Duke)...")
    gate, labels, scalar = load_aaa_data()
    mask_valid = ~np.isnan(labels)
    gate   = gate[mask_valid]
    labels = labels[mask_valid]
    scalar = scalar[mask_valid]

    print("Loading veto data (Duke)...")
    veto_labels, veto_gap, veto_gated = load_veto_data()
    veto_stats = compute_veto_stats(veto_labels, veto_gap, veto_gated)
    print(f"  FP veto rate:   {veto_stats['fp_veto_rate']:.1f}%")
    print(f"  TP suppress:    {veto_stats['tp_suppress_rate']:.1f}%")
    print(f"  Selectivity:    {veto_stats['selectivity']:.1f}×")

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 5),
        facecolor="white",
        gridspec_kw={"width_ratios": [1, 1.4, 1.3]},
    )
    ax_violin, ax_scatter, ax_veto = axes

    draw_violin(ax_violin,   gate, labels)
    draw_scatter(ax_scatter, gate, labels, scalar)
    draw_veto(ax_veto, veto_stats)

    # Subfigure labels (a), (b), (c)
    label_kw = dict(fontsize=14, fontweight="bold", transform=None,
                    ha="left", va="top")
    for ax, letter in zip(axes, ["(a)", "(b)", "(c)"]):
        ax.text(-0.12, 1.06, letter, transform=ax.transAxes, **label_kw)

    plt.tight_layout(pad=2.5)

    pdf_path = OUTPUT_DIR / "composite_gate_veto_duke.pdf"
    png_path = OUTPUT_DIR / "composite_gate_veto_duke.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
