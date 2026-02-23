#!/usr/bin/env python3
"""
JANUS Physiological Veto Figure

Two-panel (1x2) figure showing the Physiological Veto effect:
- Panel A: External Cohort (Duke) — overconfident FP suppression
- Panel B: Internal Cohort (MERLIN) — overconfident FP suppression

Each panel shows:
- Distribution of overconfident FPs (p>=0.8, y=0) under baseline
- What JANUS does to those same cases
- FP veto rate vs TP suppress rate (selectivity)

Uses CROSS-MODEL comparison (ViT-Baseline vs JANUS).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 9,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Paths
RUNS_ROOT = Path("/scratch/railabs/ld258/output/ct_triage/janus/runs")
OUTPUT_DIR = Path("/scratch/railabs/ld258/output/ct_triage/janus/miccai2026/figures")
DUKE_LABELS = Path("/scratch/railabs/ld258/output/ct_triage/macroradiomics/duke/duke_disease_labels_consensus.csv")
MERLIN_LABELS = Path("/scratch/railabs/ld258/dataset/merlin/merlinabdominalctdataset/zero_shot_findings_disease_cls.csv")

GAP_RUN = RUNS_ROOT / "JanusGAP/2026-02-10_16-42-53_seed25"
GATED_RUN = RUNS_ROOT / "JanusGatedFusion/2026-02-10_11-58-42_seed25"

# Colors
COLOR_GAP = '#D32F2F'      # Red for baseline
COLOR_GATED = '#1976D2'    # Blue for JANUS


def load_labels(labels_path: Path, case_ids: list, diseases: list) -> np.ndarray:
    """Load ground truth labels."""
    df = pd.read_csv(labels_path)
    id_col = "case_id" if "case_id" in df.columns else "study id"
    df = df.set_index(id_col)
    df = df.loc[df.index.intersection(case_ids)]
    df = df.reindex(case_ids)

    labels = np.full((len(case_ids), len(diseases)), np.nan)
    for i, disease in enumerate(diseases):
        if disease in df.columns:
            labels[:, i] = df[disease].values.astype(float)
    labels[labels == -1] = np.nan
    return labels


def compute_veto_stats(all_labels, all_gap_probs, all_gated_probs):
    """Compute FP veto and TP suppression statistics."""
    # False positives: overconfident FPs from baseline
    fp_mask = (all_labels == 0) & (all_gap_probs >= 0.8)
    fp_gap = all_gap_probs[fp_mask]
    fp_gated = all_gated_probs[fp_mask]
    fp_vetoed = (fp_gated < 0.5).sum()
    fp_veto_rate = fp_vetoed / len(fp_gated) * 100 if len(fp_gated) > 0 else 0

    # True positives: high-confidence TPs from baseline
    tp_mask = (all_labels == 1) & (all_gap_probs >= 0.8)
    tp_gap = all_gap_probs[tp_mask]
    tp_gated = all_gated_probs[tp_mask]
    tp_suppressed = (tp_gated < 0.5).sum()
    tp_suppress_rate = tp_suppressed / len(tp_gated) * 100 if len(tp_gated) > 0 else 0

    selectivity = fp_veto_rate / max(tp_suppress_rate, 0.01)

    return {
        "fp_gap": fp_gap,
        "fp_gated": fp_gated,
        "fp_count": len(fp_gap),
        "fp_vetoed": fp_vetoed,
        "fp_veto_rate": fp_veto_rate,
        "tp_gap": tp_gap,
        "tp_gated": tp_gated,
        "tp_count": len(tp_gap),
        "tp_suppressed": tp_suppressed,
        "tp_suppress_rate": tp_suppress_rate,
        "selectivity": selectivity,
    }


def load_dataset(dataset: str):
    """Load predictions and labels for a dataset."""
    labels_path = DUKE_LABELS if dataset == "duke" else MERLIN_LABELS

    # Load and deduplicate predictions (DDP padding can cause duplicates)
    gap_df = pd.read_csv(GAP_RUN / dataset / "test_predictions.csv")
    gap_df = gap_df.drop_duplicates(subset=["case_id"], keep="first")
    gap_df = gap_df.set_index("case_id")

    gated_df = pd.read_csv(GATED_RUN / dataset / "test_predictions.csv")
    gated_df = gated_df.drop_duplicates(subset=["case_id"], keep="first")
    gated_df = gated_df.set_index("case_id")

    # Load labels and find common cases
    labels_df = pd.read_csv(labels_path)
    id_col = "case_id" if "case_id" in labels_df.columns else "study id"
    labels_df = labels_df.set_index(id_col)

    common_cases = gap_df.index.intersection(gated_df.index).intersection(labels_df.index).tolist()

    gap_df = gap_df.loc[common_cases]
    gated_df = gated_df.loc[common_cases]

    diseases = list(gap_df.columns)
    labels = load_labels(labels_path, common_cases, diseases)

    # Flatten all valid predictions
    all_gap_probs = []
    all_gated_probs = []
    all_labels = []

    for d_idx, disease in enumerate(diseases):
        y_true = labels[:, d_idx]
        valid_mask = ~np.isnan(y_true)

        if valid_mask.sum() < 10:
            continue

        all_gap_probs.extend(gap_df.loc[common_cases, disease].values[valid_mask])
        all_gated_probs.extend(gated_df.loc[common_cases, disease].values[valid_mask])
        all_labels.extend(y_true[valid_mask].astype(int))

    return (
        np.array(all_labels),
        np.array(all_gap_probs),
        np.array(all_gated_probs),
    )


def plot_veto_panel(ax, veto_stats):
    """Plot a single Physiological Veto panel."""
    fp_gap = veto_stats["fp_gap"]
    fp_gated = veto_stats["fp_gated"]
    fp_veto_rate = veto_stats["fp_veto_rate"]
    tp_suppress_rate = veto_stats["tp_suppress_rate"]
    selectivity = veto_stats["selectivity"]

    # Shaded zones
    ax.axvspan(0, 0.5, alpha=0.12, color='#4CAF50', zorder=0)
    ax.axvspan(0.8, 1.0, alpha=0.15, color='#F44336', zorder=0)

    # Density histograms
    bins = np.linspace(0, 1, 30)

    ax.hist(fp_gap, bins=bins, alpha=0.55, color=COLOR_GAP,
            density=True, edgecolor='white', linewidth=0.5, zorder=2)

    ax.hist(fp_gated, bins=bins, alpha=0.55, color=COLOR_GATED,
            density=True, edgecolor='white', linewidth=0.5, zorder=2)

    # KDE curves
    if len(fp_gap) > 5:
        kde_gap = stats.gaussian_kde(fp_gap, bw_method=0.12)
        kde_gated = stats.gaussian_kde(fp_gated, bw_method=0.12)
        x_kde = np.linspace(0, 1, 200)
        ax.plot(x_kde, kde_gap(x_kde), color=COLOR_GAP, linewidth=3,
                label=f'ViT-Baseline (n={len(fp_gap):,})', zorder=3)
        ax.plot(x_kde, kde_gated(x_kde), color=COLOR_GATED, linewidth=3,
                label='JANUS (same cases)', zorder=3)

    ymax = ax.get_ylim()[1]

    # Veto arrow
    ax.annotate('', xy=(0.22, ymax * 0.72), xytext=(0.88, ymax * 0.72),
                arrowprops=dict(arrowstyle='->', color=COLOR_GATED,
                                lw=4, mutation_scale=25), zorder=5)
    ax.text(0.55, ymax * 0.80, 'Physiological Veto', fontsize=13,
            fontweight='bold', color=COLOR_GATED, ha='center', zorder=5)

    # Zone labels
    ax.text(0.25, ymax * 0.95, 'Safe Zone\n(p < 0.5)', fontsize=9,
            color='#2E7D32', ha='center', fontweight='bold', alpha=0.9)
    ax.text(0.90, ymax * 0.95, 'Danger\nZone', fontsize=9,
            color='#C62828', ha='center', fontweight='bold', alpha=0.9)

    # Selectivity annotation box — shifted left so it doesn't overlap the arrow text
    props = dict(boxstyle='round,pad=0.4', facecolor='white',
                 edgecolor=COLOR_GATED, linewidth=2)
    ax.text(0.19, ymax * 0.58,
            f'{fp_veto_rate:.1f}% of FPs\nSuppressed\n\n'
            f'Only {tp_suppress_rate:.1f}% of TPs\nAffected\n\n'
            f'Selectivity: {selectivity:.1f}\u00d7',
            fontsize=10, fontweight='bold', color=COLOR_GATED, ha='center',
            bbox=props, zorder=5)

    # Formatting
    ax.set_xlabel('Predicted Probability', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='#CCCCCC', fancybox=True,
              bbox_to_anchor=(0.0, 0.27))
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("JANUS Physiological Veto Figure")
    print("=" * 60)

    # =========================================================================
    # Load data for both datasets
    # =========================================================================
    print("\nLoading External (Duke)...")
    duke_labels, duke_gap, duke_gated = load_dataset("duke")
    duke_stats = compute_veto_stats(duke_labels, duke_gap, duke_gated)

    print(f"  Total samples: {len(duke_labels)}")
    print(f"  Positives: {duke_labels.sum()} ({100*duke_labels.mean():.1f}%)")
    print(f"  FP Veto: {duke_stats['fp_vetoed']}/{duke_stats['fp_count']} ({duke_stats['fp_veto_rate']:.1f}%)")
    print(f"  TP Suppress: {duke_stats['tp_suppressed']}/{duke_stats['tp_count']} ({duke_stats['tp_suppress_rate']:.1f}%)")
    print(f"  Selectivity: {duke_stats['selectivity']:.1f}x")

    print("\nLoading Internal (MERLIN)...")
    merlin_labels, merlin_gap, merlin_gated = load_dataset("merlin")
    merlin_stats = compute_veto_stats(merlin_labels, merlin_gap, merlin_gated)

    print(f"  Total samples: {len(merlin_labels)}")
    print(f"  Positives: {merlin_labels.sum()} ({100*merlin_labels.mean():.1f}%)")
    print(f"  FP Veto: {merlin_stats['fp_vetoed']}/{merlin_stats['fp_count']} ({merlin_stats['fp_veto_rate']:.1f}%)")
    print(f"  TP Suppress: {merlin_stats['tp_suppressed']}/{merlin_stats['tp_count']} ({merlin_stats['tp_suppress_rate']:.1f}%)")
    print(f"  Selectivity: {merlin_stats['selectivity']:.1f}x")

    # =========================================================================
    # Save one figure per dataset
    # =========================================================================
    for tag, ds_stats in [
        ("external_cohort",        duke_stats),
        ("internal_cohort_merlin", merlin_stats),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
        plot_veto_panel(ax, ds_stats)
        plt.tight_layout(pad=2.5)

        pdf_path = OUTPUT_DIR / f"physiological_veto_{tag}.pdf"
        png_path = OUTPUT_DIR / f"physiological_veto_{tag}.png"
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
        fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300, facecolor='white')
        plt.close(fig)
        print(f"\nSaved: {pdf_path}")
        print(f"Saved: {png_path}")

    # =========================================================================
    # Save summary CSV
    # =========================================================================
    summary = {
        "metric": [
            # Duke
            "duke_total_pairs",
            "duke_fp_count",
            "duke_fp_vetoed",
            "duke_fp_veto_rate_pct",
            "duke_tp_count",
            "duke_tp_suppressed",
            "duke_tp_suppress_rate_pct",
            "duke_selectivity",
            "duke_mean_baseline_fp_prob",
            "duke_mean_janus_fp_prob",
            # Merlin
            "merlin_total_pairs",
            "merlin_fp_count",
            "merlin_fp_vetoed",
            "merlin_fp_veto_rate_pct",
            "merlin_tp_count",
            "merlin_tp_suppressed",
            "merlin_tp_suppress_rate_pct",
            "merlin_selectivity",
            "merlin_mean_baseline_fp_prob",
            "merlin_mean_janus_fp_prob",
        ],
        "value": [
            # Duke
            len(duke_labels),
            duke_stats["fp_count"],
            duke_stats["fp_vetoed"],
            round(duke_stats["fp_veto_rate"], 2),
            duke_stats["tp_count"],
            duke_stats["tp_suppressed"],
            round(duke_stats["tp_suppress_rate"], 2),
            round(duke_stats["selectivity"], 2),
            round(duke_stats["fp_gap"].mean(), 4) if len(duke_stats["fp_gap"]) > 0 else 0,
            round(duke_stats["fp_gated"].mean(), 4) if len(duke_stats["fp_gated"]) > 0 else 0,
            # Merlin
            len(merlin_labels),
            merlin_stats["fp_count"],
            merlin_stats["fp_vetoed"],
            round(merlin_stats["fp_veto_rate"], 2),
            merlin_stats["tp_count"],
            merlin_stats["tp_suppressed"],
            round(merlin_stats["tp_suppress_rate"], 2),
            round(merlin_stats["selectivity"], 2),
            round(merlin_stats["fp_gap"].mean(), 4) if len(merlin_stats["fp_gap"]) > 0 else 0,
            round(merlin_stats["fp_gated"].mean(), 4) if len(merlin_stats["fp_gated"]) > 0 else 0,
        ],
    }

    summary_df = pd.DataFrame(summary)
    summary_csv_path = OUTPUT_DIR / "physiological_veto_stats.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved: {summary_csv_path}")

    # =========================================================================
    # Per-disease breakdown for both datasets
    # =========================================================================
    for dataset_name, dataset_str in [("duke", "duke"), ("merlin", "merlin")]:
        labels_path = DUKE_LABELS if dataset_str == "duke" else MERLIN_LABELS

        # Load and deduplicate predictions
        gap_df = pd.read_csv(GAP_RUN / dataset_str / "test_predictions.csv")
        gap_df = gap_df.drop_duplicates(subset=["case_id"], keep="first")
        gap_df = gap_df.set_index("case_id")

        gated_df = pd.read_csv(GATED_RUN / dataset_str / "test_predictions.csv")
        gated_df = gated_df.drop_duplicates(subset=["case_id"], keep="first")
        gated_df = gated_df.set_index("case_id")

        # Load labels and find common cases
        labels_df = pd.read_csv(labels_path)
        id_col = "case_id" if "case_id" in labels_df.columns else "study id"
        labels_df = labels_df.set_index(id_col)

        common_cases = gap_df.index.intersection(gated_df.index).intersection(labels_df.index).tolist()

        gap_df = gap_df.loc[common_cases]
        gated_df = gated_df.loc[common_cases]

        diseases = list(gap_df.columns)
        labels = load_labels(labels_path, common_cases, diseases)

        disease_stats = []
        for d_idx, disease in enumerate(diseases):
            y_true = labels[:, d_idx]
            valid_mask = ~np.isnan(y_true)

            if valid_mask.sum() < 10:
                continue

            y_valid = y_true[valid_mask].astype(int)
            gap_d = gap_df.loc[common_cases, disease].values[valid_mask]
            gated_d = gated_df.loc[common_cases, disease].values[valid_mask]

            # FP veto
            fp_mask = (y_valid == 0) & (gap_d >= 0.8)
            n_fp = fp_mask.sum()
            fp_vetoed = (gated_d[fp_mask] < 0.5).sum() if n_fp > 0 else 0

            # TP suppression
            tp_mask = (y_valid == 1) & (gap_d >= 0.8)
            n_tp = tp_mask.sum()
            tp_suppressed = (gated_d[tp_mask] < 0.5).sum() if n_tp > 0 else 0

            fp_rate = 100 * fp_vetoed / n_fp if n_fp > 0 else 0
            tp_rate = 100 * tp_suppressed / n_tp if n_tp > 0 else 0

            disease_stats.append({
                "disease": disease,
                "n_samples": valid_mask.sum(),
                "n_positives": int(y_valid.sum()),
                "n_fp_overconfident": n_fp,
                "n_fp_vetoed": fp_vetoed,
                "fp_veto_rate_pct": round(fp_rate, 2),
                "n_tp_high_conf": n_tp,
                "n_tp_suppressed": tp_suppressed,
                "tp_suppress_rate_pct": round(tp_rate, 2),
                "selectivity": round(fp_rate / max(tp_rate, 0.01), 2),
            })

        disease_df = pd.DataFrame(disease_stats)
        disease_csv = OUTPUT_DIR / f"veto_by_disease_{dataset_name}.csv"
        disease_df.to_csv(disease_csv, index=False)
        print(f"Saved: {disease_csv}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()