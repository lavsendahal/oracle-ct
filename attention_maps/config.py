# =============================================================================
# Attention Map Visualization Config
# Edit this file to select model, checkpoint, cases, and diseases.
# Then run:  python run_attention.py
# =============================================================================

# ---- Model selection --------------------------------------------------------
# "dinov3"  →  OracleCT_DINOv3_MaskedUnaryAttn
# "pillar"  →  OracleCT_Pillar_MaskedAttn
MODEL = "dinov3"

# Checkpoint (.pt file saved by train.py — key: "model_state_dict")
CKPT = "/scratch/railabs/ld258/output/ct_triage/oracle-ct/runs/OracleCT_DINOv3_MaskedUnaryAttn/YYYY-MM-DD_HH-MM-SS_seed25/checkpoints/dinov3_masked_unary_attn_best_val_macro_auc0.0000.pt"

# Number of diseases (must match training config)
NUM_DISEASES = 30

# Output directory for PNG images
OUT_DIR = "/scratch/railabs/ld258/output/ct_triage/oracle-ct/attention_maps"

# Fraction of global max attention to use for slice selection
# (slices whose max attention >= ATTN_FRAC_OF_GLOBAL_MAX * global_max are saved)
ATTN_FRAC_OF_GLOBAL_MAX = 0.30

# =============================================================================
# Cases to visualize
# Each entry: study_id → {organ, disease, z_hint, wl, ww}
#   organ:   organ name as stored in meta["organs"] (used for mask contour)
#   disease: disease name (must be in disease_names list — this selects the attention map)
#   z_hint:  fallback slice index if attention-based selection finds nothing
#   wl/ww:   CT window level / window width for display
# =============================================================================
CASES = {
    "AC42136a1": {"organ": "lungs",   "disease": "pleural_effusion",  "z_hint": 145, "wl": -600, "ww": 1500},
    "AC42136a3": {"organ": "lungs",   "disease": "atelectasis",        "z_hint": 118, "wl": -600, "ww": 1500},
    "AC421390e": {"organ": "kidneys", "disease": "renal_cyst",         "z_hint":  91, "wl":   50, "ww":  400},
    "AC42136cb": {"organ": "kidneys", "disease": "hydronephrosis",     "z_hint": 115, "wl":   50, "ww":  400},
}

# =============================================================================
# DINOv3-specific settings (used when MODEL = "dinov3")
# =============================================================================
DINOV3_VARIANT    = "B"    # "S", "B", or "L"
DINOV3_IMAGE_SIZE = 224
DINOV3_TRI_STRIDE = 1

# Janus .pt pack directory (image + masks + meta)
PACK_ROOT  = "/cachedata/ld258/janus/merlin/packs"
LABELS_CSV = "/scratch/railabs/ld258/dataset/merlin/merlinabdominalctdataset/zero_shot_findings_disease_cls.csv"

# =============================================================================
# Pillar-specific settings (used when MODEL = "pillar")
# =============================================================================
PILLAR_MODEL_REPO  = "/scratch/railabs/ld258/output/ct_triage/oracle-ct/pretrained_models/Pillar0-AbdomenCT"
PILLAR_PACK_ROOT   = "/cachedata/ld258/janus/merlin/pillar_packs_384"   # 384³ RAVE LZ4 packs
PILLAR_MASK_ROOT   = "/cachedata/ld258/janus/merlin/packs"              # original .pt packs (masks + display image)
# LABELS_CSV is shared with DINOv3 above
