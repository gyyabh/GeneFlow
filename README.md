# GeneFlow: From Deterministic Mapping to Probability Flow


## Overview

GeneFlow is a conditional flow-matching generative framework that models the conditional distribution of spot-level gene expression given H&E histopathology images — without requiring single-cell RNA-seq references. It consists of three core components:

- **AMCF** (Adaptive Multi-scale Conditional Fusion): integrates multi-resolution histological features from UNI and CONCH foundation models via globally learned attention weights.
- **GTVN** (Gene-Tokenized Velocity Network): a DiT-style Transformer that models the velocity field of gene expression evolution.
- **DT-CFM** (Discrete-Time Conditional Flow Matching): a discrete-time variant of I-CFM that reduces optimization variance by supervising at fixed time steps {0.25, 0.5, 0.75}.

GeneFlow achieves state-of-the-art PCC-10 and PCC-50 across all four HEST-1k benchmarks, while completing full test-set inference in under 5 minutes using 100-step Euler integration.

---

## Environment Setup

### Requirements

- Python 3.10+
- CUDA 11.8+
- PyTorch 2.x

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/gyyabh/Geneflow.git
cd Geneflow

# 2. Create a conda environment
conda create -n geneflow python=3.10
conda activate geneflow

# 3. Install PyTorch (adjust cuda version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install remaining dependencies
pip install -r requirements.txt
```
---

## Project Structure

```
Geneflow/
│
├── model/                          # Core model package
│   ├── __init__.py
│   ├── models.py                  # StemModel (GTVN + AMCF)            # EMA update, flow sampling (Euler ODE)
│   └── train_helper.py               # (legacy, for Stem baseline)
│
├── train.py                       # Main training script (GeneFlow)
├── s_train.py                  # Training script 
├── sample_flow.py                 # Inference script → outputs .pt / .npy
├── export_SPA125_cond.py          # Build condition embeddings for test slide
│
├── compute_metrics.py             # Evaluate PCC-10/50/200, MSE, MAE, RVD
├── plot_variance.py               # Draw gene variance comparison figure (Fig. 4)
│
├── hest1k_datasets/               # Dataset root (not included, see Data Preparation)
│   ├── kidney/
│   │   ├── st/                    # .h5ad files (gene expression per slide)
│   │   ├── processed_data/
│   │   │   ├── 1spot_uni_ebd/     # UNI embeddings (.pt, per resolution)
│   │   │   ├── 1spot_conch_ebd/   # CONCH embeddings (.pt, per resolution)
│   │   │   ├── all_slide_lst.txt  # List of all slide IDs
│   │   │   └── HMHVG.txt          # 200 selected gene names
│   ├── HER2ST/
│   ├── PRAD/
│   └── MouseBrain/
│
└── kidney_results/                # Output directory (created at runtime)
    └── runs/
        └── 000/
            ├── checkpoints/       # Model checkpoints (.pt)
            └── samples/           # Generated predictions (.pt / .npy)
```

---

## Data Preparation

### 1. Download HEST-1k datasets

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="MahmoodLab/hest",
    repo_type="dataset",
    local_dir="./hest1k_datasets",
)
```

Datasets used in this paper:

| Dataset | Tissue | Test Slide |
|---------|--------|-----------|
| HER2ST | Breast cancer | SPA148 |
| PRAD Visium | Prostate cancer | MEND145 |
| Kidney Visium | Kidney | NCBI697 |
| Mouse Brain | Mouse brain | NCBI667 |

### 2. Pre-compute UNI + CONCH embeddings

Embeddings at three resolutions (224×224, 112×112, 56×56) must be pre-extracted using frozen UNI and CONCH encoders. Place them under:

```
processed_data/1spot_uni_ebd/{slide_id}_uni.pt          # 224×224
processed_data/1spot_uni_ebd/{slide_id}_uni_112.pt      # 112×112
processed_data/1spot_uni_ebd/{slide_id}_uni_56.pt       # 56×56
processed_data/1spot_conch_ebd/{slide_id}_conch.pt
processed_data/1spot_conch_ebd/{slide_id}_conch_112.pt
processed_data/1spot_conch_ebd/{slide_id}_conch_56.pt
```

Each `.pt` file is a tensor of shape `(N_spots, dim)` where `dim` is the encoder output dimension.

### 3. Build condition embedding for test slide

```bash
python export_SPA125_cond.py
```

This concatenates UNI + CONCH across 3 resolutions into a single `(N_spots, 4608)` condition tensor, saved as `{slide_id}_cond.pt`.

---

## Training

```bash
# Single GPU
python train.py \
    --data_path      ./hest1k_datasets/kidney/ \
    --result_path    ./kidney_results/ \
    --slide_out      NCBI697 \
    --gene_list      HMHVG.txt \
    --epochs         4000 \
    --batch_size     256 \
    --lr             1e-4 \
    --DiT_num_blocks 12 \
    --hidden_size    384 \
    --num_heads      6 \
    --cond_size      4608

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 train.py [same args]
```

### Key training arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--DiT_num_blocks` | 12 | Number of DiT blocks (L) |
| `--hidden_size` | 384 | Hidden dimension (H) |
| `--num_heads` | 6 | Attention heads |
| `--cond_size` | 4608 | Condition embedding dimension |
| `--epochs` | 4000 | Total training epochs |
| `--batch_size` | 256 | Batch size (spots) |
| `--lr` | 1e-4 | Learning rate (AdamW) |
| `--lambda_aux` | 1.0 | Auxiliary per-scale loss weight |

Checkpoints are saved every 25,000 steps to `{result_path}/runs/000/checkpoints/`.

---

## Inference

### Step 1: Build condition embeddings (if not already done)

```bash
python export_SPA125_cond.py
```

### Step 2: Run inference

```bash
python sample_flow.py \
    --checkpoint     ./kidney_results/runs/000/checkpoints/0400000.pt \
    --cond_path      ./kidney_results/runs/000/samples/NCBI697_cond.pt \
    --output_dir     ./kidney_results/runs/000/samples/ \
    --input_gene_size 200 \
    --cond_size      4608 \
    --DiT_num_blocks 12 \
    --hidden_size    384 \
    --num_heads      6 \
    --num_steps      100 \
    --device         cuda:0
```

This runs **100-step Euler ODE integration** and saves predictions as both `.pt` and `.npy`.

### Output format

- `generated_samples_{ckpt_name}_{N}sample.pt` — PyTorch tensor, shape `(N_spots, 200)`
- `generated_samples_{ckpt_name}_{N}sample.npy` — NumPy array, same shape

---

## Evaluation

Compute PCC-10, PCC-50, PCC-200, MSE, MAE on the held-out test slide:

```bash
python compute_metrics.py \
    --sample_path  ./kidney_results/runs/000/samples/generated_samples_0400000_20sample.pt \
    --data_path    ./hest1k_datasets/kidney/processed_data/ \
    --ori_st_path  ./hest1k_datasets/kidney/st/ \
    --slide_out    NCBI697 \
    --gene_list    HMHVG.txt \
    --topk         10 50 200
```

### Metric definitions

| Metric | Description |
|--------|-------------|
| **PCC-k** | Mean Pearson correlation over top-k most variable genes (macro-average per gene) |
| **MSE** | Mean squared error across all spots and all 200 genes |
| **MAE** | Mean absolute error across all spots and all 200 genes |
| **RVD** | Relative variance deviation: measures how well the model preserves gene expression variance |

---

## Visualization

### Gene variance comparison (Figure 4 in paper)

```bash
python plot_variance.py
```

Configure paths at the top of the script:

```python
DATA_PATH   = "./hest1k_datasets/kidney/"
SLIDE_OUT   = "NCBI697"
OUTPUT_PDF  = "./figure4_variance.pdf"

PRED_PATHS = {
    "GeneFlow (Ours)": "./kidney_results/runs/000/samples/generated_samples_xxx.npy",
    "HisToGene":       "./baselines/histogene_pred.npy",
    "Stem":            "./baselines/stem_pred.npy",
    "STFlow":          "./baselines/stflow_pred.npy",
}
```

Outputs a 2×4 subplot figure (normalized and absolute variance, upper and lower rows).

---

## Pretrained Models

Pretrained checkpoints for all four datasets will be released at:

```
https://github.com/gyyabh/GeneFlow
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhang2025geneflow,
  title     = {From Deterministic Mapping to Probability Flow: Learning Spatial Transcriptomic Distributions from Histopathology},
  author    = {Zhang, Peng and Bai, Jinwen and Zhang, Jiarui and Li, Jinyan and Wang, Wenjian},
  journal   = {Bioinformatics},
  year      = {2025},
}
```

---


## License

This project is licensed under the MIT License.
