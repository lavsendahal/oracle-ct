#!/usr/bin/env python3
"""
Gate Value Analysis - Anatomical Gating Mechanism Verification

Loads test_gate_means.csv (produced by inference with return_gates=True)
and shows whether the gate is learning something physically meaningful.

NOTE on directionality:
  JanusGatedFusion (use_residual=false) uses the gate as a damping/focusing
  mechanism — scalars route information through visual channels. Positive cases
  may exhibit LOWER mean gate (gate narrows to key visual dimensions) while
  negative cases have HIGHER mean gate (visual pathway fully open).
  A raw AUC < 0.5 therefore means strong separation in the SUPPRESSED direction,
  not a failure. Use sep_auc = max(auc, 1−auc) + gate_direction for correct
  interpretation.

Two complementary plots per disease:
  1. Violin plot  — gate_mean distribution by label (neg vs pos)
                    answers: does the gate separate positive from negative cases?
  2. Scatter plot — gate_mean vs the most diagnostically relevant scalar
                    answers: does the gate track the underlying physiology?

Summary table saved to gate_analysis_summary.csv with:
  - sep_auc        : max(auc, 1−auc) — separability regardless of direction
  - gate_direction : "suppressed" if gate_pos < gate_neg, else "activated"
  - Spearman correlation (gate_mean vs label, signed)
  - Mean gate (neg) vs mean gate (pos)

Usage:
    python gate_analysis.py                        # merlin test set (default)
    python gate_analysis.py --dataset duke
    python gate_analysis.py --dataset merlin --split val
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

# ── Paths ──────────────────────────────────────────────────────────────────────
RUNS_ROOT   = Path("/scratch/railabs/ld258/output/ct_triage/janus/runs")
OUTPUT_DIR  = Path("/scratch/railabs/ld258/output/ct_triage/janus/miccai2026/gate_analysis")

MERLIN_LABELS   = Path("/scratch/railabs/ld258/dataset/merlin/merlinabdominalctdataset/zero_shot_findings_disease_cls.csv")
MERLIN_FEATURES = Path("/scratch/railabs/ld258/output/ct_triage/macroradiomics/merlin/features/features_combined.parquet")

DUKE_LABELS   = Path("/scratch/railabs/ld258/output/ct_triage/macroradiomics/duke/duke_disease_labels_consensus.csv")
DUKE_FEATURES = Path("/scratch/railabs/ld258/output/ct_triage/macroradiomics/duke/features/features_combined.parquet")

# Checkpoint to use (update to your best gated fusion run)
GATED_RUN = RUNS_ROOT / "JanusGatedFusion/2026-02-10_11-58-42_seed25"

# ── Primary scalar per disease ─────────────────────────────────────────────────
# The single most physically interpretable scalar for each disease.
# Used for the scatter plot (gate_mean vs this scalar).
DISEASE_KEY_SCALAR = {
    "splenomegaly":               "spleen_volume_cc",
    "hepatomegaly":               "liver_volume_cc",
    "hepatic_steatosis":          "liver_mean_hu",              # low HU → steatosis (expect neg ρ)
    "abdominal_aortic_aneurysm":  "aorta_vessel_diam_max",     # was: aorta_ascending_max_diam_mm (not in parquet)
    "atherosclerosis":            "aorta_calc_fraction",        # was: aorta_descending_max_diam_mm (not in parquet)
    "cardiomegaly":               "heart_volume_cc",
    "pleural_effusion":           "pleural_effusion_volume_cc",
    "ascites":                    "ascites_volume_cc",
    "bowel_obstruction":          "small_bowel_max_diameter_mm", # was: sb_max_diam_mm (not in parquet)
    "biliary_ductal_dilation":    "cbd_diam_max_mm",
    "pancreatic_atrophy":         "pancreas_volume_cc",
    "hydronephrosis":             "kidney_fluid_fraction_asymmetry",  # was: kidney_volume_cc (not in parquet)
    "gallstones":                 "gallbladder_volume_cc",
    "osteopenia":                 "vertebrae_L1_mean_hu",       # was: vertebral_body_mean_hu (not in parquet)
    "renal_cyst":                 "kidney_right_cyst_burden",   # was: kidney_volume_cc (not in parquet)
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_labels(path: Path, case_ids: list, diseases: list) -> pd.DataFrame:
    df = pd.read_csv(path)
    id_col = "case_id" if "case_id" in df.columns else "study id"
    df = df.set_index(id_col)
    out = pd.DataFrame(index=case_ids)
    for d in diseases:
        if d in df.columns:
            col = df[d].reindex(case_ids)
            col = col.replace(-1, np.nan)
            out[d] = col.values
        else:
            print(f"  Warning: label column '{d}' missing from {path.name} — disease will be skipped")
    return out


def load_features(path: Path, case_ids: list) -> pd.DataFrame:
    if not path.exists():
        print(f"  Warning: features parquet not found at {path}")
        return pd.DataFrame(index=case_ids)
    df = pd.read_parquet(path)
    id_col = "case_id" if "case_id" in df.columns else df.index.name or df.columns[0]
    if id_col in df.columns:
        df = df.set_index(id_col)
    return df.reindex(case_ids)


def safe_auc(gate: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return (signed_auc, sep_auc) with NaN handling.

    signed_auc : roc_auc_score(labels, gate)  — < 0.5 means gate is LOWER for positives
    sep_auc    : max(signed_auc, 1-signed_auc) — separability regardless of direction
    """
    mask = ~np.isnan(labels) & ~np.isnan(gate)
    g, l = gate[mask], labels[mask]
    if len(np.unique(l)) < 2 or len(g) < 10:
        return float("nan"), float("nan")
    auc = roc_auc_score(l, g)
    return float(auc), float(max(auc, 1.0 - auc))


def spearman(gate: np.ndarray, scalar: np.ndarray) -> tuple[float, float]:
    mask = ~np.isnan(gate) & ~np.isnan(scalar)
    if mask.sum() < 10:
        return float("nan"), float("nan")
    r, p = stats.spearmanr(gate[mask], scalar[mask])
    return float(r), float(p)


# ── Main analysis ──────────────────────────────────────────────────────────────

def analyse(dataset: str, split: str, run_dir: Path,
            labels_override: Path | None = None) -> None:
    print(f"\n{'='*60}")
    print(f"Gate Analysis  |  dataset={dataset}  split={split}")
    print(f"Run: {run_dir}")
    print("=" * 60)

    # Paths
    gates_csv     = run_dir / dataset / f"{split}_gate_means.csv"
    labels_path   = labels_override if labels_override else (MERLIN_LABELS if dataset == "merlin" else DUKE_LABELS)
    features_path = MERLIN_FEATURES if dataset == "merlin" else DUKE_FEATURES

    if labels_override:
        print(f"  Labels override: {labels_path}")

    if not gates_csv.exists():
        print(f"ERROR: gate means CSV not found: {gates_csv}")
        print("  → Run inference with +inference.return_gates=true first")
        return

    # Load gate means
    df_gates = pd.read_csv(gates_csv, index_col="case_id")
    case_ids = df_gates.index.tolist()
    print(f"  Cases: {len(case_ids)}")

    # Detect diseases from column names  (gate_mean_{disease})
    diseases = [c.replace("gate_mean_", "") for c in df_gates.columns]
    print(f"  Gated diseases: {diseases}")

    # Load labels and scalar features
    df_labels   = load_labels(labels_path, case_ids, diseases)
    df_features = load_features(features_path, case_ids)

    # Use a distinct subfolder when non-default labels are used (e.g. consensus)
    label_tag  = labels_path.stem.replace("duke_disease_labels_", "").replace("zero_shot_findings_disease_cls", "default")
    subfolder  = dataset if label_tag in ("default", "merlin") else f"{dataset}_{label_tag}"
    output_dir = OUTPUT_DIR / subfolder / split
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for disease in diseases:
        gate_col = f"gate_mean_{disease}"
        gate = df_gates[gate_col].values.astype(float)
        labels = df_labels[disease].values.astype(float) if disease in df_labels.columns else np.full(len(gate), np.nan)

        n_pos = int(np.nansum(labels == 1))
        n_neg = int(np.nansum(labels == 0))

        # Skip if not enough data
        if n_pos < 5 or n_neg < 5:
            print(f"  Skipping {disease}: too few cases (pos={n_pos}, neg={n_neg})")
            continue

        gate_neg = gate[labels == 0]
        gate_pos = gate[labels == 1]

        mean_neg = float(np.nanmean(gate_neg))
        mean_pos = float(np.nanmean(gate_pos))

        # Gate direction: does gate go UP or DOWN for positives?
        # In a damping/veto architecture gate may be LOWER for positives —
        # that is still strong separation, just in the suppressed direction.
        gate_direction = "suppressed" if mean_pos < mean_neg else "activated"

        auc, sep_auc = safe_auc(gate, labels)
        corr_lbl, p_lbl = spearman(gate, labels)

        # Scalar correlation
        key_scalar = DISEASE_KEY_SCALAR.get(disease)
        scalar_vals = None
        corr_scalar, p_scalar = float("nan"), float("nan")
        if key_scalar:
            if key_scalar not in df_features.columns:
                raise KeyError(
                    f"DISEASE_KEY_SCALAR mismatch: '{key_scalar}' for disease '{disease}' "
                    f"does not exist in features parquet.\n"
                    f"  Available columns matching '{key_scalar.split('_')[0]}': "
                    f"{[c for c in df_features.columns if key_scalar.split('_')[0] in c]}"
                )
            scalar_vals = df_features[key_scalar].values.astype(float)
            corr_scalar, p_scalar = spearman(gate, scalar_vals)

        summary_rows.append({
            "disease":            disease,
            "n_pos":              n_pos,
            "n_neg":              n_neg,
            "gate_mean_neg":      mean_neg,
            "gate_mean_pos":      mean_pos,
            "gate_direction":     gate_direction,
            "auc_signed":         auc,
            "sep_auc":            sep_auc,
            "spearman_vs_label":  corr_lbl,
            "p_vs_label":         p_lbl,
            "key_scalar":         key_scalar or "—",
            "spearman_vs_scalar": corr_scalar,
            "p_vs_scalar":        p_scalar,
        })

        # ── Figure: 2-panel (violin + scatter) ────────────────────────────────
        has_scalar = scalar_vals is not None and not np.all(np.isnan(scalar_vals))
        ncols = 2 if has_scalar else 1
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), facecolor="white")
        if ncols == 1:
            axes = [axes]

        # Panel 1: Violin by label
        ax = axes[0]
        parts = ax.violinplot(
            [gate_neg[~np.isnan(gate_neg)], gate_pos[~np.isnan(gate_pos)]],
            positions=[0, 1], showmedians=True, showextrema=True,
        )
        colors = ["#4878CF", "#D65F5F"]
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        parts["cbars"].set_color("gray")
        parts["cmaxes"].set_color("gray")
        parts["cmins"].set_color("gray")

        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Negative\n(n={n_neg})", f"Positive\n(n={n_pos})"], fontsize=11)
        ax.set_ylabel("Mean Gate Activation", fontsize=11)
        ax.set_ylim(0, 1)
        dir_arrow = "↓" if gate_direction == "suppressed" else "↑"
        ax.set_title(
            f"{disease.replace('_', ' ').title()}\n"
            f"Gate by Label  (sep_AUC={sep_auc:.3f}, {dir_arrow}{gate_direction})",
            fontsize=12, fontweight="bold",
        )
        ax.axhline(0.5, ls="--", color="gray", lw=0.8, alpha=0.5)
        ax.set_facecolor("#f8f8f8")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Panel 2: Scatter gate vs scalar
        if has_scalar:
            ax2 = axes[1]
            mask = ~np.isnan(gate) & ~np.isnan(scalar_vals) & ~np.isnan(labels)
            c_map = {0: "#4878CF", 1: "#D65F5F"}
            for lbl_val, lbl_name in [(0, "Negative"), (1, "Positive")]:
                m = mask & (labels == lbl_val)
                ax2.scatter(scalar_vals[m], gate[m], c=c_map[lbl_val], label=lbl_name,
                            alpha=0.5, s=20, edgecolors="none")

            ax2.set_xlabel(key_scalar.replace("_", " "), fontsize=11)
            ax2.set_ylabel("Mean Gate Activation", fontsize=11)
            sig_str = "***" if p_scalar < 0.001 else ("**" if p_scalar < 0.01 else ("*" if p_scalar < 0.05 else "ns"))
            ax2.set_title(f"Gate vs {key_scalar.replace('_',' ')}\nSpearman ρ={corr_scalar:.3f}  {sig_str}", fontsize=12, fontweight="bold")
            ax2.set_ylim(0, 1)
            ax2.legend(fontsize=10)
            ax2.set_facecolor("#f8f8f8")
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)

        plt.tight_layout()
        fig_path = output_dir / f"gate_{disease}.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  [{disease}]  sep_AUC={sep_auc:.3f} ({gate_direction})  ρ_label={corr_lbl:.3f}  ρ_scalar={corr_scalar:.3f}  → {fig_path.name}")

    # ── Summary table ──────────────────────────────────────────────────────────
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows).sort_values("sep_auc", ascending=False)
        summary_path = output_dir / "gate_analysis_summary.csv"
        df_summary.to_csv(summary_path, index=False, float_format="%.4f")
        print(f"\nSummary saved: {summary_path}")
        print("\nNOTE: sep_auc = max(auc, 1-auc)  — measures gate separability regardless of direction")
        print("      gate_direction='suppressed' means gate is LOWER for positive cases (damping/focusing)")
        print()
        print(df_summary[["disease", "n_pos", "n_neg", "gate_mean_neg", "gate_mean_pos",
                           "gate_direction", "sep_auc", "auc_signed", "spearman_vs_label",
                           "key_scalar", "spearman_vs_scalar"]].to_string(index=False))

    print(f"\nAll figures saved to: {output_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gate value analysis")
    parser.add_argument("--dataset",  default="merlin", choices=["merlin", "duke"])
    parser.add_argument("--split",    default="test",   choices=["test", "val", "train"])
    parser.add_argument("--run_dir",  default=str(GATED_RUN),
                        help="Path to the JanusGatedFusion run directory")
    parser.add_argument("--labels",   default=None,
                        help="Override labels CSV (e.g. duke_disease_labels_consensus.csv)")
    args = parser.parse_args()

    labels_override = Path(args.labels) if args.labels else None
    analyse(args.dataset, args.split, Path(args.run_dir), labels_override=labels_override)
