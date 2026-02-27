# janus/configs/disease_config.py
"""
Dynamic Disease Config Loader for Janus

This module loads disease configurations from the logistic regression pipeline output.
The LR pipeline produces pruned disease configs with selected features.

Usage:
    # Load from LR run directory
    from configs.disease_config import load_disease_config

    DISEASE_CONFIGS, ALL_DISEASES = load_disease_config(
        "/path/to/lr_run/disease_config_final.py"
    )

    # Or use default (for backward compatibility)
    from configs.disease_config import DISEASE_CONFIGS, ALL_DISEASES
"""

import importlib.util
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DiseaseConfig:
    """Configuration for a single disease's features."""
    scalar_features: List[str] = field(default_factory=list)
    derived_features: List[str] = field(default_factory=list)

    # Attention strategy for visual models (from original Janus)
    attention_organs: List[str] = field(default_factory=list)
    attention_strategy: str = "single"  # "single", "union", "comparative", "roi", "global"
    roi_key: Optional[str] = None
    dilation_mm: float = 0.0


# Organ channel mapping for attention masks
ORGAN_TO_CHANNEL: Dict[str, int] = {
    "liver": 0,
    "gallbladder": 1,
    "pancreas": 2,
    "spleen": 3,
    "kidneys": 4,
    "kidney_cysts": 5,
    "prostate": 6,
    "stomach_esophagus": 7,
    "small_bowel": 8,
    "colon": 9,
    "lungs": 10,
    "heart": 11,
    "aorta": 12,
    "veins": 13,
    "bones": 14,
    "pleural_space": 15,
    "periportal_space": 16,
    "perivascular_space": 17,
    "pericardial_space": 18,
    "subcutaneous_space": 19,
}

CHANNEL_TO_ORGAN = {v: k for k, v in ORGAN_TO_CHANNEL.items()}


# Default attention configuration for each disease (visual model)
DEFAULT_ATTENTION_CONFIG = {
    "hepatomegaly": {"organs": ["liver"], "strategy": "single", "dilation": 3.0},
    "splenomegaly": {"organs": ["spleen"], "strategy": "single", "dilation": 3.0},
    "cardiomegaly": {"organs": ["heart", "lungs"], "strategy": "union", "dilation": 4.0},
    "prostatomegaly": {"organs": ["prostate"], "strategy": "single", "dilation": 3.0},
    "hepatic_steatosis": {"organs": ["liver", "spleen"], "strategy": "comparative", "dilation": 3.0},
    "osteopenia": {"organs": ["bones"], "strategy": "single", "dilation": 0.0},
    "gallstones": {"organs": ["gallbladder"], "strategy": "single", "dilation": 3.0},
    "abdominal_aortic_aneurysm": {"organs": ["aorta"], "strategy": "single", "dilation": 3.0},
    "aortic_valve_calcification": {"organs": ["heart", "aorta"], "strategy": "union", "dilation": 3.0},
    "coronary_calcification": {"organs": ["heart"], "strategy": "single", "dilation": 3.0},
    "atherosclerosis": {"organs": ["aorta"], "strategy": "single", "dilation": 3.0},
    "thrombosis": {"organs": [], "strategy": "global", "dilation": 0.0},
    "bowel_obstruction": {"organs": ["small_bowel", "colon"], "strategy": "union", "dilation": 3.0},
    "appendicitis": {"organs": ["colon"], "strategy": "roi", "roi_key": "appendix_roi", "dilation": 3.0},
    "hiatal_hernia": {"organs": ["stomach_esophagus", "lungs"], "strategy": "union", "dilation": 3.0},
    "submucosal_edema": {"organs": ["small_bowel", "colon"], "strategy": "union", "dilation": 3.0},
    "free_air": {"organs": [], "strategy": "global", "dilation": 0.0},
    "biliary_ductal_dilation": {"organs": ["liver", "gallbladder", "pancreas"], "strategy": "union", "dilation": 3.0},
    "surgically_absent_gallbladder": {"organs": ["gallbladder", "liver"], "strategy": "union", "dilation": 3.0},
    "pancreatic_atrophy": {"organs": ["pancreas"], "strategy": "single", "dilation": 3.0},
    "hydronephrosis": {"organs": ["kidneys"], "strategy": "single", "dilation": 3.0},
    "renal_cyst": {"organs": ["kidneys", "kidney_cysts"], "strategy": "union", "dilation": 3.0},
    "renal_hypodensities": {"organs": ["kidneys"], "strategy": "single", "dilation": 3.0},
    "pleural_effusion": {"organs": ["lungs", "pleural_space"], "strategy": "union", "dilation": 3.0},
    "atelectasis": {"organs": ["lungs"], "strategy": "single", "dilation": 3.0},
    "ascites": {"organs": [], "strategy": "global", "dilation": 0.0},
    "anasarca": {"organs": [], "strategy": "global", "dilation": 0.0},
    "metastatic_disease": {"organs": [], "strategy": "global", "dilation": 0.0},
    "lymphadenopathy": {"organs": [], "strategy": "global", "dilation": 0.0},
    "fracture": {"organs": ["bones"], "strategy": "single", "dilation": 0.0},
}


def load_disease_config(config_path: str) -> Tuple[Dict[str, DiseaseConfig], List[str]]:
    """
    Load disease configuration from a Python file (output of LR pipeline).

    Args:
        config_path: Path to disease_config_final.py or similar

    Returns:
        (DISEASE_CONFIGS dict, ALL_DISEASES list)
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Disease config not found: {config_path}")

    # Load the module dynamically
    # IMPORTANT: Register module in sys.modules BEFORE exec_module
    # This is required for dataclass decorator to work properly
    module_name = "disease_config_loaded"
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # Register before exec to fix dataclass issue
    spec.loader.exec_module(module)

    # Get the configs from the loaded module
    loaded_configs = getattr(module, "DISEASE_CONFIGS", {})
    loaded_diseases = getattr(module, "ALL_DISEASES", list(loaded_configs.keys()))

    # Convert to our DiseaseConfig format and add attention info
    disease_configs = {}
    for disease_name, cfg in loaded_configs.items():
        # Get attention config for this disease
        attn_cfg = DEFAULT_ATTENTION_CONFIG.get(disease_name, {
            "organs": [],
            "strategy": "global",
            "dilation": 0.0
        })

        # Extract scalar features from loaded config
        if hasattr(cfg, "scalar_features"):
            scalar_features = list(cfg.scalar_features)
        else:
            scalar_features = []

        if hasattr(cfg, "derived_features"):
            derived_features = list(cfg.derived_features)
        else:
            derived_features = []

        disease_configs[disease_name] = DiseaseConfig(
            scalar_features=scalar_features,
            derived_features=derived_features,
            attention_organs=attn_cfg.get("organs", []),
            attention_strategy=attn_cfg.get("strategy", "single"),
            roi_key=attn_cfg.get("roi_key"),
            dilation_mm=attn_cfg.get("dilation", 0.0),
        )

    return disease_configs, loaded_diseases


# Global state - will be set by load_config_globally() or remain as defaults
_DISEASE_CONFIGS: Dict[str, DiseaseConfig] = {}
_ALL_DISEASES: List[str] = []
_CONFIG_LOADED = False


def load_config_globally(config_path: str) -> None:
    """
    Load disease config and set as global module state.

    Call this once at the start of training to configure the module.
    """
    global _DISEASE_CONFIGS, _ALL_DISEASES, _CONFIG_LOADED

    _DISEASE_CONFIGS, _ALL_DISEASES = load_disease_config(config_path)
    _CONFIG_LOADED = True

    print(f"✓ Loaded disease config from: {config_path}")
    print(f"  Diseases: {len(_ALL_DISEASES)}")
    print(f"  First 5: {_ALL_DISEASES[:5]}")


def get_disease_config(disease: str) -> Optional[DiseaseConfig]:
    """Get config for a specific disease."""
    return _DISEASE_CONFIGS.get(disease)


def get_organs_for_disease(disease: str) -> List[str]:
    """Get attention organs for a disease."""
    cfg = _DISEASE_CONFIGS.get(disease)
    if cfg:
        return cfg.attention_organs
    return []


# Module-level exports (these will be empty until load_config_globally is called)
# Use property-like access via functions or access after loading
@property
def DISEASE_CONFIGS() -> Dict[str, DiseaseConfig]:
    """Get disease configs (must call load_config_globally first)."""
    if not _CONFIG_LOADED:
        raise RuntimeError(
            "Disease config not loaded. Call load_config_globally(path) first, "
            "or use load_disease_config(path) directly."
        )
    return _DISEASE_CONFIGS


@property
def ALL_DISEASES() -> List[str]:
    """Get all disease names (must call load_config_globally first)."""
    if not _CONFIG_LOADED:
        raise RuntimeError(
            "Disease config not loaded. Call load_config_globally(path) first."
        )
    return _ALL_DISEASES


# For backward compatibility - these will work after load_config_globally is called
# Or they can be imported after calling the load function
def get_all_disease_configs():
    """Get all disease configs."""
    return _DISEASE_CONFIGS


def get_all_diseases():
    """Get list of all disease names."""
    return _ALL_DISEASES


# Check if config is loaded
def is_config_loaded() -> bool:
    """Check if disease config has been loaded."""
    return _CONFIG_LOADED
