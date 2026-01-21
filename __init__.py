# radioprior_v2/__init__.py
"""
RadioPrior v2: Neuro-Symbolic CT Disease Classification

Three progressive model variants:
1. GAP Baseline: DINOv3 + Global Average Pooling
2. Masked Attention: + Organ-specific attention (comparative for steatosis)
3. Scalar Fusion: + Body-normalized volumes, HU comparisons, diameter ratios

Key Features:
- 30 disease classification from CT scans
- 14-channel organ segmentation masks
- Body-size normalized volume features
- Liver-spleen HU comparison for steatosis detection
- Precise ROI attention for appendicitis
- SBO diameter ratio (small_bowel / colon)

Usage:
    from radioprior_v2.models import build_model_from_config
    from radioprior_v2.datamodules import RadioPriorDataset, FeatureBank
    
    model = build_model_from_config({
        "model_type": "scalar_fusion",
        "num_diseases": 30,
    })
"""

__version__ = "1.0.0"
