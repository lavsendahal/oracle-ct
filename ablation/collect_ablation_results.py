#!/usr/bin/env python3
"""
ablation/collect_ablation_results.py

Collects macro AUROC and AUPRC across all perturbation levels (pct) and models.
Produces two summary CSVs — one per dataset (merlin, duke).

Works with however many seeds are available (even just 1).

Usage:
    python janus/ablation/collect_ablation_results.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

# ── Paths ─────────────────────────────────────────────────────────────────────
ABLATION_ROOT = Path("/scratch/railabs/ld258/output/ct_triage/janus/ablation")

BASELINES = {
    "GatedFusion": {
        "merlin": "/scratch/railabs/ld258/output/ct_triage/janus/runs/JanusGatedFusion/2026-02-10_11-58-42_seed25/merlin/test_predictions.csv",
        "duke"  : "/scratch/railabs/ld258/output/ct_triage/janus/runs/JanusGatedFusion/2026-02-10_11-58-42_seed25/duke/test_predictions.csv",
    },
    "ScalarFusion": {
        "merlin": "/scratch/railabs/ld258/output/ct_triage/janus/runs/JanusScalarFusion/2026-02-11_11-02-50_seed25/merlin/test_predictions.csv",
        "duke"  : "/scratch/railabs/ld258/output/ct_triage/janus/runs/JanusScalarFusion/2026-02-11_11-02-50_seed25/duke/test_predictions.csv",
    },
}

ABLATION_DIRS = {
    "GatedFusion"  : ABLATION_ROOT / "results",
    "ScalarFusion" : ABLATION_ROOT / "results_scalar_fusion",
}

LABELS = {
    "merlin": "/scratch/railabs/ld258/dataset/merlin/merlinabdominalctdataset/zero_shot_findings_disease_cls.csv",
    "duke"  : "/scratch/railabs/ld258/output/ct_triage/macroradiomics/duke/duke_disease_labels_consensus.csv",
}

PCTS     = [10, 20, 50]
DATASETS = ["merlin", "duke"]
# ──────────────────────────────────────────────────────────────────────────────


def load_labels(labels_csv: str, case_ids: list, diseases: list) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    id_col = "case_id" if "case_id" in df.columns else "study id"
    df = df.set_index(id_col).reindex(case_ids)
    available = [d for d in diseases if d in df.columns]
    return df[available].replace(-1, np.nan)


def compute_macro_metrics(probs: pd.DataFrame, labels: pd.DataFrame):
    """Compute macro AUROC and AUPRC across all valid diseases."""
    aurocs, auprcs = [], []
    for disease in probs.columns:
        if disease not in labels.columns:
            continue
        mask = labels[disease].notna()
        if mask.sum() < 10:
            continue
        y_true  = labels[disease][mask].values.astype(int)
        y_score = probs[disease][mask].values
        if len(np.unique(y_true)) < 2:
            continue
        try:
            aurocs.append(roc_auc_score(y_true, y_score))
            auprcs.append(average_precision_score(y_true, y_score))
        except Exception:
            pass
    return np.mean(aurocs) if aurocs else np.nan, np.mean(auprcs) if auprcs else np.nan


def get_seed_csvs(ablation_dir: Path, dataset: str, pct: int) -> list:
    """Find all completed seed CSVs for a given dataset/pct."""
    pattern = f"{dataset}/pct{pct}/seed*/{dataset}/test_predictions.csv"
    return sorted(ablation_dir.glob(pattern))


def main():
    for dataset in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")

        labels_csv = LABELS[dataset]
        rows = []

        for model, ablation_dir in ABLATION_DIRS.items():
            baseline_csv = BASELINES[model][dataset]

            # ── Baseline (pct=0) ──────────────────────────────────────────────
            if not Path(baseline_csv).exists():
                print(f"  WARNING: Baseline not found for {model}/{dataset}: {baseline_csv}")
                baseline_auroc, baseline_auprc = np.nan, np.nan
            else:
                probs  = pd.read_csv(baseline_csv, index_col="case_id")
                labels = load_labels(labels_csv, probs.index.tolist(), probs.columns.tolist())
                baseline_auroc, baseline_auprc = compute_macro_metrics(probs, labels)

            rows.append({
                "model"             : model,
                "pct"               : 0,
                "n_seeds"           : 1,
                "macro_auroc_mean"  : round(baseline_auroc, 4),
                "macro_auroc_std"   : 0.0,
                "macro_auprc_mean"  : round(baseline_auprc, 4),
                "macro_auprc_std"   : 0.0,
            })
            print(f"  {model} pct=0  AUROC={baseline_auroc:.4f}  AUPRC={baseline_auprc:.4f}  (baseline)")

            # ── Perturbed (pct = 10, 20, 50) ──────────────────────────────────
            for pct in PCTS:
                seed_csvs = get_seed_csvs(ablation_dir, dataset, pct)

                if not seed_csvs:
                    print(f"  {model} pct={pct}  — no results found, skipping")
                    continue

                seed_aurocs, seed_auprcs = [], []
                for csv_path in seed_csvs:
                    probs  = pd.read_csv(csv_path, index_col="case_id")
                    labels = load_labels(labels_csv, probs.index.tolist(), probs.columns.tolist())
                    auroc, auprc = compute_macro_metrics(probs, labels)
                    seed_aurocs.append(auroc)
                    seed_auprcs.append(auprc)

                rows.append({
                    "model"             : model,
                    "pct"               : pct,
                    "n_seeds"           : len(seed_csvs),
                    "macro_auroc_mean"  : round(np.mean(seed_aurocs), 4),
                    "macro_auroc_std"   : round(np.std(seed_aurocs),  4),
                    "macro_auprc_mean"  : round(np.mean(seed_auprcs), 4),
                    "macro_auprc_std"   : round(np.std(seed_auprcs),  4),
                })
                print(f"  {model} pct={pct:>2}  AUROC={np.mean(seed_aurocs):.4f}±{np.std(seed_aurocs):.4f}"
                      f"  AUPRC={np.mean(seed_auprcs):.4f}±{np.std(seed_auprcs):.4f}"
                      f"  ({len(seed_csvs)} seed{'s' if len(seed_csvs) > 1 else ''})")

        # ── Save CSV ──────────────────────────────────────────────────────────
        out_path = ABLATION_ROOT / f"ablation_summary_{dataset}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
