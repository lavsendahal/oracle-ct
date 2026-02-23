#!/usr/bin/env python3
"""
compute_final_metrics.py — MICCAI 2026 Final Evaluation
========================================================

Computes AUROC, AUPRC, ECE, Brier score for all Janus model variants
on both the Merlin test set and the Duke external cohort (consensus labels).

Outputs (all under OUT_ROOT):
  macro_bootstrap_ci.csv     — macro-averaged metrics + 95% CI (bootstrap n=1000)
  per_disease_auroc.csv      — AUROC per disease × model, both datasets
  per_disease_auprc.csv      — AUPRC per disease × model, both datasets

Usage:
    python analysis/compute_final_metrics.py
    python analysis/compute_final_metrics.py --n_boot 2000 --seed 0
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

# ── Output ─────────────────────────────────────────────────────────────────────
OUT_ROOT = Path("/scratch/railabs/ld258/output/ct_triage/janus/miccai2026/metrics")

# ── Model runs ─────────────────────────────────────────────────────────────────
RUNS_ROOT = Path("/scratch/railabs/ld258/output/ct_triage/janus/runs")

MODELS = {
    "JanusGAP":         RUNS_ROOT / "JanusGAP/2026-02-10_16-42-53_seed25",
    "JanusMaskedAttn":  RUNS_ROOT / "JanusMaskedAttn/2026-02-10_12-43-43_seed25",
    "JanusScalarFusion":RUNS_ROOT / "JanusScalarFusion/2026-02-11_11-02-50_seed25",
    "JanusGatedFusion": RUNS_ROOT / "JanusGatedFusion/2026-02-10_11-58-42_seed25",
}

# ── Dataset configs ────────────────────────────────────────────────────────────
MERLIN_LABELS  = Path("/scratch/railabs/ld258/dataset/merlin/merlinabdominalctdataset/zero_shot_findings_disease_cls.csv")
MERLIN_IDS     = Path("/home/ld258/ipredict/janus/splits/test_ids.txt")

DUKE_LABELS    = Path("/scratch/railabs/ld258/output/ct_triage/macroradiomics/duke/duke_disease_labels_consensus.csv")

# ── Metrics ────────────────────────────────────────────────────────────────────

def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & (probs < hi)
        if m.sum() == 0:
            continue
        total += m.sum() * abs(labels[m].mean() - probs[m].mean())
    return total / len(probs)


def per_disease_metrics(probs_mat: np.ndarray,
                        labels_mat: np.ndarray,
                        disease_names: list[str],
                        indices: np.ndarray | None = None,
                        ) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return (aurocs, auprcs, eces, briers) across diseases.
    NaN for any disease that lacks both classes or < 10 samples."""
    if indices is not None:
        probs_mat  = probs_mat[indices]
        labels_mat = labels_mat[indices]

    aurocs, auprcs, eces, briers = [], [], [], []
    for j in range(probs_mat.shape[1]):
        p = probs_mat[:, j]
        y = labels_mat[:, j]
        mask = ~np.isnan(y)
        p, y = p[mask], y[mask]
        if len(y) < 10 or len(np.unique(y)) < 2:
            aurocs.append(np.nan); auprcs.append(np.nan)
            eces.append(np.nan);   briers.append(np.nan)
            continue
        aurocs.append(roc_auc_score(y, p))
        auprcs.append(average_precision_score(y, p))
        eces.append(ece(p, y))
        briers.append(float(np.mean((p - y) ** 2)))
    return aurocs, auprcs, eces, briers


def macro(values: list[float]) -> float:
    v = [x for x in values if not np.isnan(x)]
    return float(np.mean(v)) if v else np.nan


# ── Data loading ───────────────────────────────────────────────────────────────

def load_merlin_labels(disease_names: list[str]) -> tuple[pd.Index, np.ndarray]:
    """Load Merlin test labels aligned to test_ids order."""
    test_ids = Path(MERLIN_IDS).read_text().splitlines()
    test_ids = [t.strip() for t in test_ids if t.strip()]

    df = pd.read_csv(MERLIN_LABELS)
    id_col = "case_id" if "case_id" in df.columns else "study id"
    df = df.set_index(id_col).reindex(test_ids)

    labels = np.full((len(test_ids), len(disease_names)), np.nan)
    for j, d in enumerate(disease_names):
        if d in df.columns:
            col = df[d].replace(-1, np.nan).values.astype(float)
            labels[:, j] = col
    return pd.Index(test_ids), labels


def load_duke_labels(case_ids: pd.Index, disease_names: list[str]) -> np.ndarray:
    """Load Duke consensus labels aligned to case_ids."""
    df = pd.read_csv(DUKE_LABELS).set_index("case_id").reindex(case_ids)
    labels = np.full((len(case_ids), len(disease_names)), np.nan)
    for j, d in enumerate(disease_names):
        if d in df.columns:
            labels[:, j] = df[d].values.astype(float)
    return labels


def load_predictions(run_dir: Path, dataset: str,
                     case_ids: pd.Index, disease_names: list[str]) -> np.ndarray:
    path = run_dir / dataset / "test_predictions.csv"
    df = pd.read_csv(path)
    # Remove DDP-padding duplicates (keep first occurrence)
    df = df.drop_duplicates(subset="case_id", keep="first").set_index("case_id").reindex(case_ids)
    probs = np.zeros((len(case_ids), len(disease_names)), dtype=np.float32)
    for j, d in enumerate(disease_names):
        if d in df.columns:
            probs[:, j] = df[d].values.astype(np.float32)
    return probs


# ── Bootstrap CI ───────────────────────────────────────────────────────────────

def bootstrap_macro_ci(probs_mat: np.ndarray,
                       labels_mat: np.ndarray,
                       disease_names: list[str],
                       n_boot: int = 1000,
                       seed: int = 42,
                       ) -> dict[str, tuple[float, float, float]]:
    """Return {metric: (point_estimate, ci_lo, ci_hi)} macro-averaged."""
    rng = np.random.default_rng(seed)
    n   = probs_mat.shape[0]

    boot = {k: [] for k in ("auroc", "auprc", "ece", "brier")}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a, ap, e, b = per_disease_metrics(probs_mat, labels_mat, disease_names, idx)
        boot["auroc"].append(macro(a))
        boot["auprc"].append(macro(ap))
        boot["ece"].append(macro(e))
        boot["brier"].append(macro(b))

    # Point estimates (no resampling)
    a, ap, e, b = per_disease_metrics(probs_mat, labels_mat, disease_names)
    points = dict(auroc=macro(a), auprc=macro(ap), ece=macro(e), brier=macro(b))

    result = {}
    for k in boot:
        arr = [x for x in boot[k] if not np.isnan(x)]
        lo, hi = np.percentile(arr, [2.5, 97.5])
        result[k] = (round(points[k], 4), round(lo, 4), round(hi, 4))
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def run(n_boot: int, seed: int) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Disease names: union of gated fusion prediction columns (canonical order)
    gated_pred_merlin = MODELS["JanusGatedFusion"] / "merlin" / "test_predictions.csv"
    disease_names = [c for c in pd.read_csv(gated_pred_merlin).columns if c != "case_id"]
    print(f"Diseases: {len(disease_names)}")

    # Case ID sets
    merlin_ids, merlin_labels = load_merlin_labels(disease_names)
    duke_ids   = pd.read_csv(MODELS["JanusGatedFusion"] / "duke" / "test_predictions.csv")["case_id"].tolist()
    duke_ids   = pd.Index(duke_ids)
    duke_labels = load_duke_labels(duke_ids, disease_names)

    print(f"Merlin test: {len(merlin_ids)} cases")
    print(f"Duke:        {len(duke_ids)} cases")

    # Determine which diseases are valid in Duke (both classes present, ≥10 samples)
    # Used to define the "common" disease subset for fair apples-to-apples comparison.
    common_idx = []
    common_diseases = []
    for j, d in enumerate(disease_names):
        y = duke_labels[:, j]
        y = y[~np.isnan(y)]
        if len(y) >= 10 and len(np.unique(y)) == 2:
            common_idx.append(j)
            common_diseases.append(d)
    print(f"Common diseases (valid in Duke): {len(common_diseases)}/{len(disease_names)}")
    print(f"  {common_diseases}")

    ci_rows       = []
    per_d_auroc   = {"disease": disease_names}
    per_d_auprc   = {"disease": disease_names}

    # Three evaluation settings:
    #   merlin       — all 30 diseases on Merlin test set
    #   duke         — diseases valid in Duke, on Duke cases
    #   merlin_common— same disease subset as Duke, on Merlin cases (fair comparison)
    datasets = {
        "merlin":        (merlin_ids,  merlin_labels,                    disease_names),
        "duke":          (duke_ids,    duke_labels,                      disease_names),
        "merlin_common": (merlin_ids,  merlin_labels[:, common_idx],     common_diseases),
    }

    for model_name, run_dir in MODELS.items():
        for dataset, (case_ids, labels_mat, dnames) in datasets.items():
            print(f"\n── {model_name}  ×  {dataset} ──")

            probs_mat = load_predictions(run_dir,
                                         dataset.replace("_common", ""),
                                         case_ids, disease_names)
            # For merlin_common: slice probs to common disease columns only
            if dataset == "merlin_common":
                probs_mat = probs_mat[:, common_idx]

            # Per-disease point estimates
            aurocs, auprcs, eces, briers = per_disease_metrics(
                probs_mat, labels_mat, dnames)

            # Store per-disease columns only for merlin and duke (not merlin_common)
            if dataset in ("merlin", "duke"):
                col = f"{model_name}_{dataset}"
                per_d_auroc[col] = [round(v, 4) if not np.isnan(v) else np.nan for v in aurocs]
                per_d_auprc[col] = [round(v, 4) if not np.isnan(v) else np.nan for v in auprcs]

            n_valid = sum(1 for v in aurocs if not np.isnan(v))
            print(f"  Valid diseases: {n_valid}/{len(dnames)}")
            print(f"  Macro AUROC={macro(aurocs):.4f}  AUPRC={macro(auprcs):.4f}  "
                  f"ECE={macro(eces):.4f}  Brier={macro(briers):.4f}")

            # Bootstrap CI
            print(f"  Bootstrap CI (n={n_boot})...", flush=True)
            ci = bootstrap_macro_ci(probs_mat, labels_mat, dnames, n_boot, seed)

            row = {"model": model_name, "dataset": dataset}
            for metric, (pt, lo, hi) in ci.items():
                row[metric]             = pt
                row[f"{metric}_lo"]     = lo
                row[f"{metric}_hi"]     = hi
                row[f"{metric}_ci"]     = f"{pt:.3f} ({lo:.3f}–{hi:.3f})"
                print(f"  {metric:<6} {row[f'{metric}_ci']}")
            ci_rows.append(row)

    # ── Save ──────────────────────────────────────────────────────────────────
    df_ci = pd.DataFrame(ci_rows)
    ci_path = OUT_ROOT / "macro_bootstrap_ci.csv"
    df_ci.to_csv(ci_path, index=False, float_format="%.4f")
    print(f"\nSaved: {ci_path}")

    df_auroc = pd.DataFrame(per_d_auroc)
    df_auroc.to_csv(OUT_ROOT / "per_disease_auroc.csv", index=False, float_format="%.4f")

    df_auprc = pd.DataFrame(per_d_auprc)
    df_auprc.to_csv(OUT_ROOT / "per_disease_auprc.csv", index=False, float_format="%.4f")

    print(f"Saved: {OUT_ROOT / 'per_disease_auroc.csv'}")
    print(f"Saved: {OUT_ROOT / 'per_disease_auprc.csv'}")

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("MACRO METRICS WITH 95% CI")
    print("="*80)
    for _, r in df_ci.iterrows():
        print(f"\n{r['model']}  [{r['dataset']}]")
        for m in ("auroc", "auprc", "ece", "brier"):
            print(f"  {m:<6} {r[f'{m}_ci']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed",   type=int, default=25)
    args = parser.parse_args()
    run(args.n_boot, args.seed)
