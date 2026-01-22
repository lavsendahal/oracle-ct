# Janus: Neuro-Symbolic CT Disease Classification

Janus is a neuro-symbolic deep learning framework for multi-label disease classification from 3D abdominal CT scans. It combines visual deep learning (DINOv3 Vision Transformer) with anatomical priors and scalar radiomics features to classify 30 diseases with interpretability.

## Key Features

- **Neuro-Symbolic Fusion**: Combines visual features with domain-knowledge scalar features
- **Anatomical Attention**: Disease-specific organ-masked attention pooling
- **Multiple Architectures**: From simple baselines to advanced gated fusion
- **Distributed Training**: Multi-GPU support via PyTorch DDP
- **Flexible Configuration**: Hydra-based configuration with experiment overrides

## Supported Diseases (30)

hepatomegaly, splenomegaly, cardiomegaly, prostatomegaly, hepatic_steatosis, osteopenia, gallstones, abdominal_aortic_aneurysm, aortic_valve_calcification, coronary_calcification, atherosclerosis, thrombosis, bowel_obstruction, appendicitis, hiatal_hernia, submucosal_edema, free_air, biliary_ductal_dilation, surgically_absent_gallbladder, pancreatic_atrophy, hydronephrosis, renal_cyst, renal_hypodensities, pleural_effusion, atelectasis, ascites, anasarca, metastatic_disease, lymphadenopathy, fracture

## Model Architectures

### 1. RadioPriorGAP (Baseline)
Pure neural baseline without anatomical priors.
- DINOv3 backbone → Global Average Pooling → Classification heads
- Config: `experiment=dinov3_baseline_gap`

### 2. RadioPriorMaskedAttn (Anatomical Attention)
Disease-specific organ-masked attention pooling.
- Attention strategies: single, union, comparative, roi, global
- Learnable temperature (tau) and mask bias parameters
- Config: `experiment=dinov3_masked_attn`

### 3. RadioPriorScalarFusion (Visual + Scalar Concatenation)
Fuses visual and scalar features via concatenation.
- Visual projector (768→256d) + Scalar projector (N→256d)
- Fusion MLP for final classification
- Config: `experiment=dinov3_scalar_fusion`

### 4. RadioPriorGatedFusion (Advanced Gating)
Scalar features gate visual features (element-wise multiplication).
- Visual pooling: `masked_attn` or `gap`
- Optional dual-head mode with learnable mixture weights
- Can warm-start from pretrained logistic regression weights
- Config: `experiment=dinov3_gated_fusion`

## Installation

```bash
# Clone repository
cd /path/to/ipredict/neuro_symbolic/janus

# Install dependencies
pip install torch torchvision transformers hydra-core omegaconf
pip install pandas numpy scikit-learn tqdm wandb

# Set environment variables
export SCRATCH=/scratch/your_username
export IPREDICT_ROOT=/path/to/ipredict
export NETID=your_netid
export PYTHONPATH=$IPREDICT_ROOT/neuro_symbolic:$PYTHONPATH
```

## Data Requirements

### Directory Structure
```
$SCRATCH/
├── output/ct_triage/.../macroradiomics/merlin/
│   └── v1_new_features/
│       ├── disease_config_final.py   # Per-disease feature/attention configs
│       ├── feature_stats.json        # Mean/std for z-score normalization
│       └── model_weights.json        # Pretrained LR weights (optional)
│
/cachedata/$NETID/merlin/packs/
├── case_001.pt                       # Preprocessed CT + masks
├── case_002.pt
└── ...

$IPREDICT_ROOT/neuro_symbolic/
├── splits/
│   ├── train_ids.txt                 # One case_id per line
│   ├── val_ids.txt
│   └── test_ids.txt
└── janus/
    └── zero_shot_findings_disease_cls.csv  # Labels
```

### Pack File Format (.pt)
```python
{
    "image": torch.Tensor([1, D, H, W]),     # CT volume (160×224×224)
    "masks": torch.Tensor([20, D, H, W]),    # Organ segmentation masks
    "meta": {
        "case_id": "case_001",
        "spacing_final_mm": [1.5, 1.5, 3.0]
    }
}
```

### Labels CSV Format
```csv
case_id,hepatomegaly,splenomegaly,...,fracture
case_001,1,0,-1,...,0
case_002,0,1,0,...,1
```
- `0` = negative, `1` = positive, `-1` = uncertain/missing

### Features Parquet
Scalar radiomics features indexed by `case_id`:
- Base features: liver_volume_ml, spleen_mean_hu, etc.
- Derived features: liver_spleen_hu_diff, cardiothoracic_ratio, etc.

## Configuration

### Environment Variables
The config uses OmegaConf's `${oc.env:VAR}` syntax:
```yaml
paths:
  lr_run_dir: ${oc.env:SCRATCH}/output/.../v1_new_features
  pack_root: /cachedata/${oc.env:NETID}/merlin/packs
```

### Key Configuration Options

#### Model Settings (`configs/experiment/*.yaml`)
```yaml
model:
  name: RadioPriorGatedFusion
  variant: B                          # S (small), B (base), L (large)
  visual_pooling: masked_attn         # masked_attn | gap
  use_residual: false                 # Dual-head mode
  load_lr_weights: true               # Warm-start scalar heads
  freeze_backbone: false
  use_gradient_checkpointing: true    # Saves ~50% GPU memory
```

#### Training Settings (`configs/config.yaml`)
```yaml
training:
  max_epochs: 20
  learning_rate: 3e-4
  optimizer: adamw
  scheduler: cosine
  warmup_epochs: 5
  gradient_clip: 1.0
  use_amp: true                       # Mixed precision

  # Grouped learning rates
  use_group_lrs: true
  head_lr_scale: 3.0                  # Heads: 3× base LR
  alpha_lr_scale: 0.3                 # Alpha params: 0.3× base LR

  # Loss
  loss_fn: bce_uncertain
  use_class_weights: true
  pos_weight_clip: 10.0

  # Distributed
  use_ddp: false

  # Checkpointing
  save_best_only: true
  monitor_metric: val/macro_auc
```

#### Augmentation Settings
```yaml
dataset:
  use_augmentation: true
  aug_preset: anatomy_safe_v2         # Recommended (no flips)
  aug_params:
    anatomy_safe_v2:
      p_affine: 0.35
      rot_deg: 10.0
      translate_xy: 5.0
      scale_min: 0.95
      scale_max: 1.05
      p_gamma: 0.30
      p_noise: 0.30
```

## Training

### Basic Training
```bash
python train.py experiment=dinov3_gated_fusion
```

### With Custom Paths
```bash
python train.py experiment=dinov3_gated_fusion \
  paths.disease_config=/path/to/disease_config_final.py \
  paths.pack_root=/path/to/packs \
  paths.labels_csv=/path/to/labels.csv
```

### Distributed Training (Multi-GPU)
```bash
torchrun --nproc_per_node=8 train.py \
  experiment=dinov3_gated_fusion \
  training.use_ddp=true
```

### With WandB Logging
```bash
python train.py experiment=dinov3_gated_fusion \
  logging.use_wandb=true \
  logging.wandb_project=janus_runs
```

### Model Variants
```bash
# Baseline (no anatomical priors)
python train.py experiment=dinov3_baseline_gap

# Masked Attention
python train.py experiment=dinov3_masked_attn

# Scalar Fusion
python train.py experiment=dinov3_scalar_fusion

# Gated Fusion (default)
python train.py experiment=dinov3_gated_fusion
```

### Advanced Options
```bash
# Backbone size
python train.py experiment=dinov3_gated_fusion model.variant=L

# Visual pooling strategy
python train.py experiment=dinov3_gated_fusion model.visual_pooling=gap

# Learning rate tuning
python train.py experiment=dinov3_gated_fusion \
  training.learning_rate=1e-4 \
  training.warmup_epochs=3

# Debug: Test LR weight loading
python train.py experiment=dinov3_gated_fusion \
  model.debug_scalar_only=true
```

## Inference

### Basic Inference
```bash
python inference.py \
  experiment=dinov3_gated_fusion \
  paths.checkpoint=/path/to/checkpoints/model_best.pt
```

### Distributed Inference
```bash
torchrun --nproc_per_node=4 inference.py \
  experiment=dinov3_gated_fusion \
  paths.checkpoint=/path/to/checkpoint.pt \
  training.use_ddp=true
```

### Different Splits
```bash
# Validation set
python inference.py ... paths.test_ids=/path/to/val_ids.txt

# Custom split
python inference.py ... paths.test_ids=/path/to/custom_ids.txt
```

### Output
Inference saves to the checkpoint's parent directory:
- `test_predictions.csv`: Per-case disease probabilities
- `test_metrics.json`: AUC, AUPRC per disease

## SLURM Scripts

### Training
```bash
sbatch scripts/slurm_train.sub
```

### Inference
```bash
sbatch scripts/slurm_inference.sub
```

## Project Structure

```
janus/
├── train.py                    # Main training script
├── inference.py                # Inference script
├── losses.py                   # BCE with uncertain label handling
│
├── configs/
│   ├── config.yaml             # Default configuration
│   ├── disease_config.py       # Dynamic disease config loader
│   └── experiment/
│       ├── dinov3_baseline_gap.yaml
│       ├── dinov3_masked_attn.yaml
│       ├── dinov3_scalar_fusion.yaml
│       └── dinov3_gated_fusion.yaml
│
├── models/
│   └── radioprior_model.py     # All model architectures
│
├── datamodules/
│   ├── dataset.py              # RadioPriorDataset
│   ├── feature_bank.py         # Scalar feature loading/normalization
│   └── augmentation.py         # Data augmentation
│
└── scripts/
    ├── slurm_train.sub         # SLURM training script
    ├── slurm_inference.sub     # SLURM inference script
    └── compute_feature_stats.py
```

## Key Technical Details

### Visual Processing
1. CT volume (160×224×224) → 2.5D tri-slices (axial+coronal+sagittal)
2. DINOv3 extracts 768d features per patch token
3. Pooling: GAP or disease-specific masked attention

### Attention Strategies
| Strategy | Description | Example Disease |
|----------|-------------|-----------------|
| single | Focus on one organ | hepatomegaly (liver) |
| union | Multiple organs combined | ascites (liver+spleen+kidneys) |
| comparative | Separate pooling, concatenated | hepatic_steatosis (liver vs spleen) |
| roi | Precise bounding box | appendicitis (appendix ROI) |
| global | Full CT scan | free_air |

### Scalar Features
- Z-score normalized using precomputed statistics
- Disease-specific feature selection from `disease_config_final.py`
- Supports base features + derived features (ratios, differences)

### Learnable Parameters
- **Temperature (tau)**: Controls attention sharpness [0.2, 2.0]
- **Mask bias**: Inside/outside attention priors
- **Alpha**: Mixture weight for dual-head mode

## Troubleshooting

### Missing Keys Warning
```
Missing keys: 90 (ok if you changed model options like visual_pooling)
```
This occurs when checkpoint and model configs differ (e.g., `use_residual` mismatch). Ensure configs match.

### Environment Variable Errors
```
InterpolationKeyError: oc.env:SCRATCH
```
Export required environment variables before running:
```bash
export SCRATCH=/scratch/your_path
export IPREDICT_ROOT=/path/to/ipredict
export NETID=your_netid
```

### GPU Memory Issues
Enable gradient checkpointing:
```bash
python train.py model.use_gradient_checkpointing=true
```

### Empty Validation Batches (DDP)
Ensure validation dataset size is divisible by (batch_size × num_gpus) or the code handles remainders.

## Citation

If you use this code, please cite:
```

```

## License


