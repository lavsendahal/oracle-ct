"""
OracleCT Minimal Scalar Feature Config

One small, fixed feature set per disease:
  - {organ}_mean_hu          — tissue density
  - {organ}_to_body_ratio    — body-size-normalised volume
  - {organ}_touches_border   — FOV truncation flag (0/1)

Special cases where the standard triple doesn't apply:
  - hepatic_steatosis  : liver vs spleen HU comparison (liver_spleen_hu_diff)
  - osteopenia         : bone density via lumbar aggregate
  - fracture           : lumbar + vertebral HU
  - aorta diseases     : aorta uses aorta_hu_mean (different naming), aorta_calc_fraction
  - bilateral kidneys  : left + right columns
  - global diseases    : empty — no organ anchor, scalars not used

These features are all present in the oracle-ct minimal parquet:
  /scratch/railabs/ld258/output/ct_triage/oracle-ct/data/merlin/features_minimal.parquet
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class DiseaseConfig:
    scalar_features: List[str] = field(default_factory=list)
    derived_features: List[str] = field(default_factory=list)


DISEASE_CONFIGS: Dict[str, DiseaseConfig] = {

    # -------------------------------------------------------------------------
    # ORGAN SIZE / VOLUME
    # -------------------------------------------------------------------------
    "hepatomegaly": DiseaseConfig(
        scalar_features=["liver_mean_hu", "liver_to_body_ratio", "liver_touches_border"],
    ),
    "splenomegaly": DiseaseConfig(
        scalar_features=["spleen_mean_hu", "spleen_to_body_ratio", "spleen_touches_border"],
    ),
    "cardiomegaly": DiseaseConfig(
        scalar_features=[
            "heart_mean_hu", "heart_to_body_ratio", "heart_touches_border",
            "lung_total_to_body_ratio",
        ],
    ),
    "prostatomegaly": DiseaseConfig(
        scalar_features=["prostate_mean_hu", "prostate_to_body_ratio", "prostate_touches_border"],
    ),
    "pancreatic_atrophy": DiseaseConfig(
        scalar_features=["pancreas_mean_hu", "pancreas_to_body_ratio", "pancreas_touches_border"],
    ),

    # -------------------------------------------------------------------------
    # LIVER / BILIARY
    # -------------------------------------------------------------------------
    "hepatic_steatosis": DiseaseConfig(
        # Comparative signal: liver vs spleen HU — the core steatosis marker
        scalar_features=["liver_mean_hu", "spleen_mean_hu", "liver_spleen_hu_diff"],
    ),
    "gallstones": DiseaseConfig(
        scalar_features=[
            "gallbladder_mean_hu", "gallbladder_to_body_ratio", "gallbladder_touches_border",
        ],
    ),
    "biliary_ductal_dilation": DiseaseConfig(
        scalar_features=["liver_mean_hu", "gallbladder_mean_hu", "pancreas_mean_hu"],
    ),
    "surgically_absent_gallbladder": DiseaseConfig(
        scalar_features=[
            "gallbladder_mean_hu", "gallbladder_to_body_ratio", "gallbladder_touches_border",
        ],
    ),

    # -------------------------------------------------------------------------
    # AORTA / CARDIOVASCULAR
    # -------------------------------------------------------------------------
    "abdominal_aortic_aneurysm": DiseaseConfig(
        # aorta uses aorta_hu_mean (not aorta_mean_hu) — different naming in parquet
        scalar_features=["aorta_hu_mean", "aorta_calc_fraction"],
    ),
    "aortic_valve_calcification": DiseaseConfig(
        scalar_features=["aorta_hu_mean", "heart_mean_hu", "heart_to_body_ratio"],
    ),
    "coronary_calcification": DiseaseConfig(
        scalar_features=["heart_mean_hu", "heart_to_body_ratio", "heart_touches_border"],
    ),
    "atherosclerosis": DiseaseConfig(
        scalar_features=["aorta_hu_mean", "aorta_calc_fraction"],
    ),

    # -------------------------------------------------------------------------
    # BONES
    # -------------------------------------------------------------------------
    "osteopenia": DiseaseConfig(
        # No to_body_ratio or touches_border for bones — use lumbar HU aggregate
        scalar_features=["lumbar_hu_mean", "vertebrae_L1_mean_hu", "vertebrae_L2_mean_hu"],
    ),
    "fracture": DiseaseConfig(
        scalar_features=["lumbar_hu_mean", "vertebrae_L1_mean_hu", "vertebrae_L2_mean_hu"],
    ),

    # -------------------------------------------------------------------------
    # BOWEL / GI
    # -------------------------------------------------------------------------
    "bowel_obstruction": DiseaseConfig(
        scalar_features=[
            "small_bowel_mean_hu", "small_bowel_to_body_ratio",
            "colon_mean_hu", "colon_to_body_ratio",
        ],
    ),
    "appendicitis": DiseaseConfig(
        scalar_features=[
            "colon_mean_hu", "colon_touches_border", "small_bowel_mean_hu",
        ],
    ),
    "hiatal_hernia": DiseaseConfig(
        scalar_features=[
            "stomach_mean_hu", "stomach_to_body_ratio", "stomach_touches_border",
        ],
    ),
    "submucosal_edema": DiseaseConfig(
        scalar_features=[
            "small_bowel_mean_hu", "small_bowel_to_body_ratio", "colon_mean_hu",
        ],
    ),

    # -------------------------------------------------------------------------
    # KIDNEYS (bilateral)
    # -------------------------------------------------------------------------
    "hydronephrosis": DiseaseConfig(
        scalar_features=[
            "kidney_left_mean_hu", "kidney_right_mean_hu",
            "kidney_left_to_body_ratio", "kidney_right_to_body_ratio",
        ],
    ),
    "renal_cyst": DiseaseConfig(
        scalar_features=[
            "kidney_left_mean_hu", "kidney_right_mean_hu",
            "kidney_left_to_body_ratio", "kidney_right_to_body_ratio",
        ],
    ),
    "renal_hypodensities": DiseaseConfig(
        scalar_features=["kidney_left_mean_hu", "kidney_right_mean_hu"],
    ),

    # -------------------------------------------------------------------------
    # LUNGS / PLEURAL
    # -------------------------------------------------------------------------
    "pleural_effusion": DiseaseConfig(
        scalar_features=[
            "lung_lower_lobe_left_mean_hu", "lung_lower_lobe_right_mean_hu",
            "lung_total_to_body_ratio",
        ],
    ),
    "atelectasis": DiseaseConfig(
        scalar_features=[
            "lung_lower_lobe_left_mean_hu", "lung_lower_lobe_right_mean_hu",
            "lung_total_to_body_ratio",
        ],
    ),

    # -------------------------------------------------------------------------
    # GLOBAL STRATEGY — no organ anchor, no scalars
    # -------------------------------------------------------------------------
    "thrombosis":         DiseaseConfig(scalar_features=[]),
    "free_air":           DiseaseConfig(scalar_features=[]),
    "ascites":            DiseaseConfig(scalar_features=[]),
    "anasarca":           DiseaseConfig(scalar_features=[]),
    "metastatic_disease": DiseaseConfig(scalar_features=[]),
    "lymphadenopathy":    DiseaseConfig(scalar_features=[]),
}

ALL_DISEASES = list(DISEASE_CONFIGS.keys())
