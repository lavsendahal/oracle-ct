#!/usr/bin/env python3
"""
analysis/in_model_pvr.py

Within-model Physiological Veto Rate (PVR) analysis using ungated predictions.

Computes the veto rate using the correct methodology:
- p_ungated: predictions with gate=1 (visual stream fully trusted)
- p_final: predictions with anatomical gating applied

Veto@0.8 = Pr(p_final < 0.8 | p_ungated >= 0.8, y=0)

This directly answers: "How often does the anatomical gate prevent the model
from being high-confidence wrong?"

Usage:
    python janus/analysis/in_model_pvr.py

Output:
- in_model_pvr_duke.csv: Per-disease veto statistics for Duke
- in_model_pvr_merlin.csv: Per-disease veto statistics for Merlin
- in_model_pvr_summary.csv: Overall summary statistics
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
RUNS_ROOT  = Path("/scratch/railabs/ld258/output/ct_triage/janus/runs")
OUTPUT_DIR = Path("/scratch/railabs/ld258/output/ct_triage/janus/miccai2026/figures")
MERLIN_LABELS = Path("/scratch/railabs/ld258/dataset/merlin/merlinabdominalctdataset/zero_shot_findings_disease_cls.csv")
DUKE_LABELS   = Path("/scratch/railabs/ld258/output/ct_triage/macroradiomics/duke/duke_disease_labels_consensus.csv")

GATED_RUN = RUNS_ROOT / "JanusGatedFusion/2026-02-10_11-58-42_seed25"
# ──────────────────────────────────────────────────────────────────────────────


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


def compute_veto_analysis_for_dataset(dataset: str) -> dict:
    """
    Compute within-model PVR for a dataset using ungated predictions.

    Args:
        dataset: "duke" or "merlin"

    Returns:
        Dictionary with summary statistics and per-disease breakdown
    """
    print(f"\n{'='*60}")
    print(f"In-Model PVR for {dataset.upper()}")
    print(f"{'='*60}")

    labels_path = DUKE_LABELS if dataset == "duke" else MERLIN_LABELS
    final_path   = GATED_RUN / dataset / "test_predictions.csv"
    ungated_path = GATED_RUN / dataset / "test_predictions_ungated.csv"

    if not ungated_path.exists():
        print(f"ERROR: Ungated predictions not found: {ungated_path}")
        print("Run inference with +inference.return_ungated=true")
        return None

    # Load predictions (may have duplicates from DDP gathering — deduplicate)
    final_df = pd.read_csv(final_path)
    final_df = final_df.drop_duplicates(subset=["case_id"], keep="first")
    final_df = final_df.set_index("case_id")

    ungated_df = pd.read_csv(ungated_path)
    ungated_df = ungated_df.drop_duplicates(subset=["case_id"], keep="first")
    ungated_df = ungated_df.set_index("case_id")

    labels_df = pd.read_csv(labels_path)
    id_col = "case_id" if "case_id" in labels_df.columns else "study id"
    labels_df = labels_df.set_index(id_col)

    # Ensure same cases across all three sources
    common_cases = final_df.index.intersection(ungated_df.index).intersection(labels_df.index).tolist()
    final_df   = final_df.loc[common_cases]
    ungated_df = ungated_df.loc[common_cases]

    diseases = list(final_df.columns)
    labels   = load_labels(labels_path, common_cases, diseases)

    print(f"Cases: {len(common_cases)}")
    print(f"Diseases: {len(diseases)}")

    # Flatten all predictions and labels
    all_final_probs   = []
    all_ungated_probs = []
    all_labels        = []
    all_diseases      = []

    for d_idx, disease in enumerate(diseases):
        y_true     = labels[:, d_idx]
        valid_mask = ~np.isnan(y_true)

        if valid_mask.sum() < 10:
            continue

        final_probs_d   = final_df.loc[common_cases, disease].values
        ungated_probs_d = ungated_df.loc[common_cases, disease].values

        all_final_probs.extend(final_probs_d[valid_mask])
        all_ungated_probs.extend(ungated_probs_d[valid_mask])
        all_labels.extend(y_true[valid_mask].astype(int))
        all_diseases.extend([disease] * valid_mask.sum())

    all_final_probs   = np.array(all_final_probs)
    all_ungated_probs = np.array(all_ungated_probs)
    all_labels        = np.array(all_labels)

    print(f"\nTotal (case, disease) pairs: {len(all_labels)}")
    print(f"Positives: {all_labels.sum()} ({100*all_labels.mean():.1f}%)")

    # ── FALSE POSITIVE VETO ────────────────────────────────────────────────────
    fp_mask         = (all_labels == 0) & (all_ungated_probs >= 0.8)
    fp_ungated_probs = all_ungated_probs[fp_mask]
    fp_final_probs   = all_final_probs[fp_mask]

    vetoed_08    = (fp_final_probs < 0.8).sum()
    veto_rate_08 = 100 * vetoed_08 / len(fp_final_probs) if len(fp_final_probs) > 0 else 0

    vetoed_05    = (fp_final_probs < 0.5).sum()
    veto_rate_05 = 100 * vetoed_05 / len(fp_final_probs) if len(fp_final_probs) > 0 else 0

    print(f"\n--- FALSE POSITIVE VETO (y=0, p_ungated >= 0.8) ---")
    print(f"High-confidence FPs (ungated): {len(fp_ungated_probs)}")
    print(f"Veto@0.8 (p_final < 0.8): {vetoed_08}/{len(fp_ungated_probs)} ({veto_rate_08:.1f}%)")
    print(f"Veto@0.5 (p_final < 0.5): {vetoed_05}/{len(fp_ungated_probs)} ({veto_rate_05:.1f}%)")

    # ── TRUE POSITIVE PRESERVATION ─────────────────────────────────────────────
    tp_mask          = (all_labels == 1) & (all_ungated_probs >= 0.8)
    tp_ungated_probs = all_ungated_probs[tp_mask]
    tp_final_probs   = all_final_probs[tp_mask]

    suppressed_08    = (tp_final_probs < 0.8).sum()
    suppress_rate_08 = 100 * suppressed_08 / len(tp_final_probs) if len(tp_final_probs) > 0 else 0

    suppressed_05    = (tp_final_probs < 0.5).sum()
    suppress_rate_05 = 100 * suppressed_05 / len(tp_final_probs) if len(tp_final_probs) > 0 else 0

    print(f"\n--- TRUE POSITIVE PRESERVATION (y=1, p_ungated >= 0.8) ---")
    print(f"High-confidence TPs (ungated): {len(tp_ungated_probs)}")
    print(f"Suppressed@0.8 (p_final < 0.8): {suppressed_08}/{len(tp_ungated_probs)} ({suppress_rate_08:.1f}%)")
    print(f"Suppressed@0.5 (p_final < 0.5): {suppressed_05}/{len(tp_ungated_probs)} ({suppress_rate_05:.1f}%)")

    # ── SELECTIVITY ────────────────────────────────────────────────────────────
    selectivity_08 = veto_rate_08 / max(suppress_rate_08, 0.1)
    selectivity_05 = veto_rate_05 / max(suppress_rate_05, 0.1)

    print(f"\n--- SELECTIVITY ---")
    print(f"Selectivity@0.8 (FP veto / TP suppress): {selectivity_08:.2f}x")
    print(f"Selectivity@0.5 (FP veto / TP suppress): {selectivity_05:.2f}x")

    # ── Per-disease breakdown ──────────────────────────────────────────────────
    disease_stats = []
    for d_idx, disease in enumerate(diseases):
        y_true     = labels[:, d_idx]
        valid_mask = ~np.isnan(y_true)

        if valid_mask.sum() < 10:
            continue

        y_valid   = y_true[valid_mask].astype(int)
        p_final   = final_df.loc[common_cases, disease].values[valid_mask]
        p_ungated = ungated_df.loc[common_cases, disease].values[valid_mask]

        fp_mask_d   = (y_valid == 0) & (p_ungated >= 0.8)
        n_fp        = fp_mask_d.sum()
        fp_vetoed_08 = (p_final[fp_mask_d] < 0.8).sum() if n_fp > 0 else 0
        fp_vetoed_05 = (p_final[fp_mask_d] < 0.5).sum() if n_fp > 0 else 0

        tp_mask_d       = (y_valid == 1) & (p_ungated >= 0.8)
        n_tp            = tp_mask_d.sum()
        tp_suppressed_08 = (p_final[tp_mask_d] < 0.8).sum() if n_tp > 0 else 0
        tp_suppressed_05 = (p_final[tp_mask_d] < 0.5).sum() if n_tp > 0 else 0

        disease_stats.append({
            "disease"                  : disease,
            "n_samples"                : valid_mask.sum(),
            "n_positives"              : int(y_valid.sum()),
            "n_negatives"              : int((y_valid == 0).sum()),
            "n_high_conf_fps_ungated"  : n_fp,
            "n_fps_vetoed_08"          : fp_vetoed_08,
            "n_fps_vetoed_05"          : fp_vetoed_05,
            "fp_veto_rate_08_pct"      : round(100 * fp_vetoed_08 / n_fp, 2) if n_fp > 0 else 0,
            "fp_veto_rate_05_pct"      : round(100 * fp_vetoed_05 / n_fp, 2) if n_fp > 0 else 0,
            "n_high_conf_tps_ungated"  : n_tp,
            "n_tps_suppressed_08"      : tp_suppressed_08,
            "n_tps_suppressed_05"      : tp_suppressed_05,
            "tp_suppress_rate_08_pct"  : round(100 * tp_suppressed_08 / n_tp, 2) if n_tp > 0 else 0,
            "tp_suppress_rate_05_pct"  : round(100 * tp_suppressed_05 / n_tp, 2) if n_tp > 0 else 0,
            "mean_ungated_prob_fps"    : round(p_ungated[fp_mask_d].mean(), 4) if n_fp > 0 else 0,
            "mean_final_prob_fps"      : round(p_final[fp_mask_d].mean(), 4)   if n_fp > 0 else 0,
            "mean_ungated_prob_tps"    : round(p_ungated[tp_mask_d].mean(), 4) if n_tp > 0 else 0,
            "mean_final_prob_tps"      : round(p_final[tp_mask_d].mean(), 4)   if n_tp > 0 else 0,
        })

    summary = {
        "dataset"                  : dataset,
        "n_cases"                  : len(common_cases),
        "n_diseases"               : len(diseases),
        "n_case_disease_pairs"     : len(all_labels),
        "n_positives"              : int(all_labels.sum()),
        "n_negatives"              : int((all_labels == 0).sum()),
        "positive_rate_pct"        : round(100 * all_labels.mean(), 2),
        "n_high_conf_fps_ungated"  : len(fp_ungated_probs),
        "n_fps_vetoed_08"          : vetoed_08,
        "n_fps_vetoed_05"          : vetoed_05,
        "fp_veto_rate_08_pct"      : round(veto_rate_08, 2),
        "fp_veto_rate_05_pct"      : round(veto_rate_05, 2),
        "n_high_conf_tps_ungated"  : len(tp_ungated_probs),
        "n_tps_suppressed_08"      : suppressed_08,
        "n_tps_suppressed_05"      : suppressed_05,
        "tp_suppress_rate_08_pct"  : round(suppress_rate_08, 2),
        "tp_suppress_rate_05_pct"  : round(suppress_rate_05, 2),
        "selectivity_08"           : round(selectivity_08, 2),
        "selectivity_05"           : round(selectivity_05, 2),
        "mean_prob_reduction_fps"  : round((fp_ungated_probs - fp_final_probs).mean(), 4) if len(fp_ungated_probs) > 0 else 0,
        "mean_prob_reduction_tps"  : round((tp_ungated_probs - tp_final_probs).mean(), 4) if len(tp_ungated_probs) > 0 else 0,
    }

    return {
        "summary"    : summary,
        "per_disease": pd.DataFrame(disease_stats),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("IN-MODEL PHYSIOLOGICAL VETO RATE (PVR)")
    print("Ungated predictions (gate=1) vs final gated predictions")
    print("=" * 60)
    print("\nVeto@0.8 = Pr(p_final < 0.8 | p_ungated >= 0.8, y=0)")
    print("Measures: How often does the gate prevent high-confidence FPs?")

    summaries = []

    for dataset in ["merlin", "duke"]:
        result = compute_veto_analysis_for_dataset(dataset)

        if result is None:
            continue

        disease_path = OUTPUT_DIR / f"in_model_pvr_{dataset}.csv"
        result["per_disease"].to_csv(disease_path, index=False)
        print(f"\nSaved: {disease_path}")

        summaries.append(result["summary"])

    if summaries:
        summary_df   = pd.DataFrame(summaries)
        summary_path = OUTPUT_DIR / "in_model_pvr_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved: {summary_path}")

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("\n                          MERLIN      DUKE")
        print("-" * 50)
        rows = {r["dataset"]: r for _, r in summary_df.iterrows()}
        merlin, duke = rows.get("merlin"), rows.get("duke")
        if merlin is not None and duke is not None:
            print(f"High-conf FPs (ungated): {merlin['n_high_conf_fps_ungated']:>6}    {duke['n_high_conf_fps_ungated']:>6}")
            print(f"FP Veto@0.8:             {merlin['fp_veto_rate_08_pct']:>5.2f}%    {duke['fp_veto_rate_08_pct']:>5.2f}%")
            print(f"FP Veto@0.5:             {merlin['fp_veto_rate_05_pct']:>5.2f}%    {duke['fp_veto_rate_05_pct']:>5.2f}%")
            print("-" * 50)
            print(f"High-conf TPs (ungated): {merlin['n_high_conf_tps_ungated']:>6}    {duke['n_high_conf_tps_ungated']:>6}")
            print(f"TP Suppress@0.8:         {merlin['tp_suppress_rate_08_pct']:>5.2f}%    {duke['tp_suppress_rate_08_pct']:>5.2f}%")
            print(f"TP Suppress@0.5:         {merlin['tp_suppress_rate_05_pct']:>5.2f}%    {duke['tp_suppress_rate_05_pct']:>5.2f}%")
            print("-" * 50)
            print(f"Selectivity@0.8:         {merlin['selectivity_08']:>5.2f}x    {duke['selectivity_08']:>5.2f}x")
            print(f"Selectivity@0.5:         {merlin['selectivity_05']:>5.2f}x    {duke['selectivity_05']:>5.2f}x")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
