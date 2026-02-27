# janus/__init__.py
"""
OracleCT: Neuro-Symbolic CT Disease Classification

Four progressive model variants (DINOv3 and ResNet3D backbones):
1. GAP Baseline:           + Global Average Pooling (uniform)
2. UnaryAttnPool:          + Learned full-volume attention (no organ mask)
3. MaskedUnaryAttn:        + Organ-guided attention (organ mask prior)
4. MaskedUnaryAttnScalar:  + Scalar features (HU, volume, diameters)

Key Features:
- 30 disease classification from CT scans
- 14-channel organ segmentation masks
- Body-size normalized volume features
- Liver-spleen HU comparison for steatosis detection
- Precise ROI attention for appendicitis
- SBO diameter ratio (small_bowel / colon)

Usage:
    from janus.models import build_model_from_config
    from janus.datamodules import JanusDataset, FeatureBank

    model = build_model_from_config({
        "model_type": "masked_unary_attn_scalar",
        "num_diseases": 30,
    })
"""

__version__ = "1.0.0"
