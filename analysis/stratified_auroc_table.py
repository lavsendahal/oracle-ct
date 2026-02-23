#!/usr/bin/env python3
"""
stratified_auroc_table.py — Stratified per-disease AUROC by pathology group

For each of the 5 pathology groups, reports:
  - Group mean AUROC (Merlin + Duke) for all 4 models
  - Top 1 and worst 1 disease by absolute JANUS vs ViT gain on Duke

Output:
  stratified_auroc_table.csv    — formatted display table
  stratified_auroc_raw.csv      — raw numeric values

Usage:
    python analysis/stratified_auroc_table.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from collections import OrderedDict
from pathlib import Path
from sklearn.metrics import roc_auc_score

# ── Paths ─────────────────────────────────────────────────────────────────────
RUNS_ROOT     = Path("/scratch/railabs/ld258/output/ct_triage/janus/runs")
OUT_ROOT      = Path("/scratch/railabs/ld258/output/ct_triage/janus/miccai2026/metrics")
MERLIN_LABELS = Path("/scratch/railabs/ld258/dataset/merlin/merlinabdominalctdataset/zero_shot_findings_disease_cls.csv")
MERLIN_IDS    = Path("/home/ld258/ipredict/janus/splits/test_ids.txt")
DUKE_LABELS   = Path("/scratch/railabs/ld258/output/ct_triage/macroradiomics/duke/duke_disease_labels_consensus.csv")

# ── Models (display name → run dir) ───────────────────────────────────────────
MODELS = OrderedDict([
    ("ViT",    RUNS_ROOT / "JanusGAP/2026-02-10_16-42-53_seed25"),
    ("ORACLE", RUNS_ROOT / "JanusMaskedAttn/2026-02-10_12-43-43_seed25"),
    ("+OSF",   RUNS_ROOT / "JanusScalarFusion/2026-02-11_11-02-50_seed25"),
    ("JANUS",  RUNS_ROOT / "JanusGatedFusion/2026-02-10_11-58-42_seed25"),
])
MODEL_COLS = list(MODELS.keys())

# ── Pathology groups ───────────────────────────────────────────────────────────
GROUPS = OrderedDict([
    ("A) Geometric", [
        "hepatomegaly", "splenomegaly", "cardiomegaly", "prostatomegaly",
        "abdominal_aortic_aneurysm", 
        "hydronephrosis", "bowel_obstruction",
    ]),
    ("B) Densitometric", [
        "hepatic_steatosis", "osteopenia", "gallstones",
        "coronary_calcification", "atherosclerosis",
    ]),
    ("C) Fluid / Global", [
        "pleural_effusion", "ascites", "anasarca",
    ]),
    ("D) Chronic / Structural", [
        "atelectasis", "pancreatic_atrophy", "renal_cyst",
        "surgically_absent_gallbladder", "hiatal_hernia","biliary_ductal_dilation"
    ]),
    ("E) Focal (Control)", [
        "thrombosis", "appendicitis", "free_air", "fracture",
        "metastatic_disease", "lymphadenopathy",
    ]),
])

# ── Representative diseases shown per group (2 per group, manually curated) ───
REPRESENTATIVES = {
    "A) Geometric":            ["prostatomegaly",    "abdominal_aortic_aneurysm"],
    "B) Densitometric":        ["hepatic_steatosis", "gallstones"],
    "C) Fluid / Global":       ["pleural_effusion",  "ascites"],
    "D) Chronic / Structural": ["renal_cyst",        "pancreatic_atrophy"],
    "E) Focal (Control)":      ["appendicitis",      "fracture"],
}

# ── Display names ──────────────────────────────────────────────────────────────
DISPLAY = {
    "hepatomegaly":                  "Hepatomegaly",
    "splenomegaly":                  "Splenomegaly",
    "cardiomegaly":                  "Cardiomegaly",
    "prostatomegaly":                "Prostatomegaly",
    "abdominal_aortic_aneurysm":     "Abd. Aortic Aneurysm",
    "biliary_ductal_dilation":       "Biliary Ductal Dilation",
    "hydronephrosis":                "Hydronephrosis",
    "bowel_obstruction":             "Bowel Obstruction",
    "hepatic_steatosis":             "Hepatic Steatosis",
    "osteopenia":                    "Osteopenia",
    "gallstones":                    "Gallstones",
    "coronary_calcification":        "Coronary Calcification",
    "atherosclerosis":               "Atherosclerosis",
    "pleural_effusion":              "Pleural Effusion",
    "ascites":                       "Ascites",
    "anasarca":                      "Anasarca",
    "atelectasis":                   "Atelectasis",
    "pancreatic_atrophy":            "Pancreatic Atrophy",
    "renal_cyst":                    "Renal Cyst",
    "surgically_absent_gallbladder": "Surgically Absent GB",
    "lymphadenopathy":               "Lymphadenopathy",
    "hiatal_hernia":                 "Hiatal Hernia",
    "thrombosis":                    "Thrombosis",
    "appendicitis":                  "Appendicitis",
    "free_air":                      "Free Air",
    "fracture":                      "Fracture",
    "metastatic_disease":            "Metastatic Disease",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_score)
    y, p = y_true[mask], y_score[mask]
    if len(np.unique(y)) < 2 or len(y) < 10:
        return np.nan
    return float(roc_auc_score(y, p))


def group_mean(auroc_dict: dict, diseases: list[str]) -> float:
    vals = [auroc_dict[d] for d in diseases if d in auroc_dict and not np.isnan(auroc_dict[d])]
    return float(np.mean(vals)) if vals else np.nan


def load_merlin_labels(disease_names: list[str]) -> tuple[list, dict]:
    test_ids = [t.strip() for t in Path(MERLIN_IDS).read_text().splitlines() if t.strip()]
    df = pd.read_csv(MERLIN_LABELS)
    id_col = "case_id" if "case_id" in df.columns else "study id"
    df = df.set_index(id_col).reindex(test_ids)
    labels = {}
    for d in disease_names:
        if d in df.columns:
            labels[d] = df[d].replace(-1, np.nan).values.astype(float)
        else:
            labels[d] = np.full(len(test_ids), np.nan)
    return test_ids, labels


def load_duke_labels(case_ids: list, disease_names: list[str]) -> dict:
    df = pd.read_csv(DUKE_LABELS).set_index("case_id").reindex(case_ids)
    labels = {}
    for d in disease_names:
        if d in df.columns:
            labels[d] = df[d].values.astype(float)
        else:
            labels[d] = np.full(len(case_ids), np.nan)
    return labels


def load_predictions(run_dir: Path, dataset: str,
                     case_ids: list, disease_names: list[str]) -> dict:
    path = run_dir / dataset / "test_predictions.csv"
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset="case_id", keep="first").set_index("case_id").reindex(case_ids)
    preds = {}
    for d in disease_names:
        if d in df.columns:
            preds[d] = df[d].values.astype(float)
        else:
            preds[d] = np.full(len(case_ids), np.nan)
    return preds


def fmt(v: float) -> str:
    return f"{v:.2f}" if not np.isnan(v) else "—"


def fmt_gain(v: float) -> str:
    if np.isnan(v):
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.1f}%"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Disease names from gated fusion predictions (canonical order)
    gated_run = MODELS["JANUS"]
    all_diseases = [c for c in pd.read_csv(gated_run / "merlin" / "test_predictions.csv").columns
                    if c != "case_id"]
    print(f"Total diseases: {len(all_diseases)}")

    # Load labels
    merlin_ids, merlin_labels = load_merlin_labels(all_diseases)
    duke_ids_raw = pd.read_csv(gated_run / "duke" / "test_predictions.csv")["case_id"].drop_duplicates().tolist()
    duke_labels  = load_duke_labels(duke_ids_raw, all_diseases)
    print(f"Merlin test: {len(merlin_ids)} cases")
    print(f"Duke:        {len(duke_ids_raw)} cases")

    # Compute per-disease AUROC for all models × datasets
    auroc: dict[str, dict[str, dict[str, float]]] = {
        m: {"merlin": {}, "duke": {}} for m in MODEL_COLS
    }
    for model_name, run_dir in MODELS.items():
        merlin_preds = load_predictions(run_dir, "merlin", merlin_ids,   all_diseases)
        duke_preds   = load_predictions(run_dir, "duke",   duke_ids_raw, all_diseases)
        for d in all_diseases:
            auroc[model_name]["merlin"][d] = safe_auroc(merlin_labels[d], merlin_preds[d])
            auroc[model_name]["duke"][d]   = safe_auroc(duke_labels[d],   duke_preds[d])
        print(f"  Computed: {model_name}")

    # Valid diseases in Duke (both classes, ≥10 samples)
    duke_valid = set()
    for d in all_diseases:
        y = duke_labels[d]
        y_clean = y[~np.isnan(y)]
        if len(y_clean) >= 10 and len(np.unique(y_clean)) == 2:
            duke_valid.add(d)
    print(f"Valid in Duke: {len(duke_valid)}/{len(all_diseases)}")

    # ── Build table ───────────────────────────────────────────────────────────
    rows      = []   # for display CSV
    raw_rows  = []   # for numeric CSV

    for group_name, group_diseases in GROUPS.items():
        valid = [d for d in group_diseases if d in duke_valid]
        n_valid = len(valid)

        # Group mean row
        g_merlin = {m: group_mean(auroc[m]["merlin"], valid) for m in MODEL_COLS}
        g_duke   = {m: group_mean(auroc[m]["duke"],   valid) for m in MODEL_COLS}
        gain_grp = round(g_duke["JANUS"], 2) - round(g_duke["ViT"], 2)

        rows.append({
            "Pathology":       f"{group_name} (n={n_valid})",
            "row_type":        "group",
            **{f"Merlin_{m}":  fmt(g_merlin[m]) for m in MODEL_COLS},
            **{f"Duke_{m}":    fmt(g_duke[m])   for m in MODEL_COLS},
            "Gain_Ext":        fmt_gain(gain_grp),
        })
        raw_rows.append({
            "Pathology":       f"{group_name} (n={n_valid})",
            "row_type":        "group",
            **{f"Merlin_{m}":  g_merlin[m] for m in MODEL_COLS},
            **{f"Duke_{m}":    g_duke[m]   for m in MODEL_COLS},
            "Gain_Ext":        gain_grp,
        })

        selected = REPRESENTATIVES.get(group_name, [])

        for d in selected:
            duke_gain_d = round(auroc["JANUS"]["duke"][d], 2) - round(auroc["ViT"]["duke"][d], 2)
            rows.append({
                "Pathology":       f"  {DISPLAY.get(d, d)}",
                "row_type":        "disease",
                **{f"Merlin_{m}":  fmt(auroc[m]["merlin"][d]) for m in MODEL_COLS},
                **{f"Duke_{m}":    fmt(auroc[m]["duke"][d])   for m in MODEL_COLS},
                "Gain_Ext":        fmt_gain(duke_gain_d),
            })
            raw_rows.append({
                "Pathology":       f"  {DISPLAY.get(d, d)}",
                "row_type":        "disease",
                **{f"Merlin_{m}":  auroc[m]["merlin"][d] for m in MODEL_COLS},
                **{f"Duke_{m}":    auroc[m]["duke"][d]   for m in MODEL_COLS},
                "Gain_Ext":        duke_gain_d,
            })

    # ── Save ──────────────────────────────────────────────────────────────────
    display_cols = (["Pathology"] +
                    [f"Merlin_{m}" for m in MODEL_COLS] +
                    [f"Duke_{m}"   for m in MODEL_COLS] +
                    ["Gain_Ext"])

    df_display = pd.DataFrame(rows)[display_cols]
    df_raw     = pd.DataFrame(raw_rows)

    out_display = OUT_ROOT / "stratified_auroc_table.csv"
    out_raw     = OUT_ROOT / "stratified_auroc_raw.csv"
    df_display.to_csv(out_display, index=False)
    df_raw.to_csv(out_raw, index=False, float_format="%.4f")
    print(f"\nSaved: {out_display}")
    print(f"Saved: {out_raw}")

    # ── Print ─────────────────────────────────────────────────────────────────
    merlin_hdr = "  ".join(f"{m:>7}" for m in MODEL_COLS)
    duke_hdr   = "  ".join(f"{m:>7}" for m in MODEL_COLS)
    print(f"\n{'Pathology':<38}  {'── MERLIN ──':^35}  {'── DUKE ──':^35}  {'Gain':>7}")
    print(f"{'':38}  {merlin_hdr}  {duke_hdr}  {'(Ext.)':>7}")
    print("─" * 115)

    for row in rows:
        name = row["Pathology"]
        m_vals = "  ".join(f"{row[f'Merlin_{m}']:>7}" for m in MODEL_COLS)
        d_vals = "  ".join(f"{row[f'Duke_{m}']:>7}" for m in MODEL_COLS)
        gain   = row["Gain_Ext"]
        if row["row_type"] == "group":
            print(f"\n{name:<38}  {m_vals}  {d_vals}  {gain:>7}")
        else:
            print(f"{name:<38}  {m_vals}  {d_vals}  {gain:>7}")

    print(f"\n{'─'*115}")
    print(f"N Merlin = {len(merlin_ids):,}    N Duke = {len(duke_ids_raw):,}")


if __name__ == "__main__":
    main()
