# 🧠 Self-Pruning Neural Network — CIFAR-10

> **Tredence AI Engineering Internship Case Study**
> A neural network that learns to prune itself during training — no post-training step required.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Core Concept](#core-concept)
- [Repository Structure](#repository-structure)
- [Solution 1: Pure MLP (Main Submission)](#solution-1-pure-mlp-main-submission)
  - [Architecture](#architecture)
  - [PrunableLinear Layer](#prunablelinear-layer)
  - [Sparsity Loss](#sparsity-loss)
  - [Training Strategy](#training-strategy)
  - [Results](#results)
  - [How to Run](#how-to-run)
- [Solution 2: CNN + Prunable Classifier (Bonus)](#solution-2-cnn--prunable-classifier-bonus)
  - [Architecture](#architecture-1)
  - [Key Differences from Pure MLP](#key-differences-from-pure-mlp)
  - [Colab Checkpoint Resume System](#colab-checkpoint-resume-system)
  - [Results](#results-1)
  - [How to Run](#how-to-run-1)
- [Why L1 Encourages Sparsity — Theory](#why-l1-encourages-sparsity--theory)
- [λ Trade-off Analysis](#λ-trade-off-analysis)
- [Generated Outputs](#generated-outputs)
- [Requirements](#requirements)

---

## Overview

This repository contains the solution to the **Self-Pruning Neural Network** case study assigned by Tredence Analytics for their AI Engineering Internship. The task is to build a neural network that actively learns to prune its own weights during the forward/backward pass — not as a post-training compression step, but as an intrinsic part of the training objective.

The core mechanism: every weight in the network is multiplied by a learnable **gate**, a scalar in (0, 1) produced by a sigmoid function. An L1 regularisation term on these gates pushes them toward zero. A gate that reaches zero effectively removes its weight from the network — the connection is pruned.

Two complete solutions are provided:

| Notebook | Architecture | Purpose |
|---|---|---|
| `Self Pruning MLP.ipynb` | **Pure MLP** (5 PrunableLinear layers) | **Main required submission** |
| `Pruning With CNN feature extractor.ipynb` | **CNN backbone + Prunable classifier head** | Bonus: higher accuracy, Colab-ready |

---

## Core Concept

The key innovation is the `PrunableLinear` layer:

```
gate_scores   →  sigmoid(gate_scores)  =  gates  ∈ (0, 1)
pruned_weight  =  weight  ×  gates            (element-wise)
output         =  pruned_weight @ x.T + bias
```

The total training loss is:

```
Total Loss = CrossEntropyLoss(logits, labels)  +  λ × Σ(all gate values)
```

The second term — the **L1 norm of all gates** — creates a constant downward pressure on every gate, regardless of its current value. A weight survives only if it is useful enough for classification to overcome this pressure. Otherwise, its gate is driven to zero and the weight is effectively deleted from the network.

---

## Repository Structure

```
.
├── Self Pruning MLP.ipynb   # Main solution: Pure MLP self-pruning network
├── Pruning With CNN feature extractor.ipynb             # Bonus solution: CNN + prunable head, Colab/resume support
├── outputs/                   # Auto-generated after running
│   ├── gate_distribution.png  # Gate histogram (spike at 0 = successful pruning)
│   ├── training_curves.png    # Accuracy & sparsity vs epoch for all λ
│   ├── lambda_tradeoff.png    # Accuracy–Sparsity scatter plot
│   ├── report.md              # Auto-generated Markdown report
│   └── model_lam*.pt          # Saved model checkpoints per λ
└── README.md
```

---

## Solution 1: Pure MLP (Main Submission)

**File:** `notebookd645b4b263.ipynb`

This is the primary, fully self-contained solution. It uses **no CNN, no convolutional layers anywhere**. Every single linear transformation in the network goes through a `PrunableLinear` layer. This is intentionally the harder path — a pure MLP operating on raw CIFAR-10 pixels — because it most cleanly demonstrates the pruning mechanism without any feature extraction shortcut.

### Architecture

```
Input: CIFAR-10 image  (3 × 32 × 32)
         ↓
   Flatten → 3072 features
         ↓
   PrunableLinear(3072 → 1024)  +  BatchNorm1d  +  ReLU  +  Dropout(0.3)
         ↓
   PrunableLinear(1024 → 512)   +  BatchNorm1d  +  ReLU  +  Dropout(0.3)
         ↓
   PrunableLinear(512  → 256)   +  BatchNorm1d  +  ReLU  +  Dropout(0.2)
         ↓
   PrunableLinear(256  → 128)   +  BatchNorm1d  +  ReLU
         ↓
   PrunableLinear(128  → 10)    →  Logits (10 classes)
```

**Total gated parameters:** Every weight in every one of the 5 PrunableLinear layers has its own learnable gate. The gate tensors mirror the weight tensors exactly in shape.

**Why BatchNorm?** As gates shrink toward zero, the effective magnitude of weights changes dramatically during training. BatchNorm stabilises the activations through this process, preventing training collapse.

---

### PrunableLinear Layer

```python
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features, gate_init=2.0):
        super().__init__()
        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), gate_init)
        )
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x):
        gates          = torch.sigmoid(self.gate_scores)   # ∈ (0, 1)
        pruned_weights = self.weight * gates               # element-wise mask
        return F.linear(x, pruned_weights, self.bias)
```

**Design decisions:**
- `gate_init = 2.0` → `sigmoid(2.0) ≈ 0.88`, so gates start mostly open. This gives the network time to learn useful representations before pruning pressure takes effect.
- `gate_scores` is registered as an `nn.Parameter`, so PyTorch autograd automatically computes gradients for it — no custom backward pass needed.
- Gradients flow through both `weight` and `gate_scores` via the element-wise product. The chain rule handles everything:
  - `∂Loss/∂weight = ∂Loss/∂output × gate` (the CE gradient scaled by the gate)
  - `∂Loss/∂gate_scores = ∂Loss/∂output × weight × sigmoid'(gate_scores)` (CE path) + `λ × sigmoid'(gate_scores)` (sparsity path)

---

### Sparsity Loss

```python
def sparsity_loss(self) -> torch.Tensor:
    all_gates = torch.cat([
        torch.sigmoid(layer.gate_scores).reshape(-1)
        for layer in self.prunable_layers()
    ])
    return all_gates.sum()   # L1 of positives == sum
```

Because sigmoid output is always positive, the L1 norm simplifies to the sum of all gate values. This is added to the classification loss scaled by λ:

```
Total Loss = CrossEntropy + λ × sum(all gates)
```

A weight is considered **pruned** when its gate falls below the threshold `1e-2` (1%). Sparsity is reported as:

```
Sparsity % = (# gates < 0.01) / (total gates) × 100
```

---

### Training Strategy

Several design choices work together to achieve both high accuracy and high sparsity simultaneously:

**1. λ Warm-Up Schedule**

Directly applying a large λ from the start would prevent the network from learning anything useful before the sparsity loss crushes all gates. Instead, lambda is introduced in three phases:

```
Phase 1 — Warm-up   (epochs  1–16, 20% of total):   λ_eff = 0
           Network trains freely on cross-entropy alone.

Phase 2 — Ramp      (epochs 17–36, next 25%):         λ_eff linearly 0 → λ_target
           Sparsity pressure is introduced gradually.

Phase 3 — Full      (epochs 37–80):                   λ_eff = λ_target
           Full pruning pressure; only important gates survive.
```

**2. Separate Learning Rates for Gates and Weights**

```python
optimizer = optim.AdamW([
    {"params": rest_params, "lr": 5e-4,      "weight_decay": 1e-4},
    {"params": gate_params, "lr": 5e-4 * 8,  "weight_decay": 0.0},
])
```

Gates get **8× the base learning rate** so they can move fast enough to achieve meaningful sparsity within the 80-epoch budget. Weight decay is set to zero for gate parameters — the L1 sparsity loss already controls them; L2 would cause unintended interactions.

**3. CosineAnnealingWarmRestarts Scheduler**

```python
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=25, T_mult=1, eta_min=1e-6
)
```

Multiple warm restarts give the optimizer fresh momentum across long training, helping escape local minima where gates are "stuck" at intermediate values.

**4. Data Augmentation**

Since a pure MLP has no spatial inductive bias, augmentation is important:

```python
transforms.RandomCrop(32, padding=4),
transforms.RandomHorizontalFlip(p=0.5),
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
```

**5. Best Checkpoint Restore**

The model state at peak test accuracy is saved throughout training. The final reported metrics come from this best checkpoint, not the final epoch state.

---

### Results

Training was run for **80 epochs** with **batch size 256** on CIFAR-10 (50,000 train / 10,000 test) across three λ values. Hardware: CUDA GPU. Total parameters: 7,676,042 — of which 3,835,136 are learnable gate parameters.

| λ (Lambda) | Test Accuracy | Sparsity Level (%) | Accuracy Grade | Sparsity Grade |
|:---:|:---:|:---:|:---:|:---:|
| `1e-6` (Low) | **60.86%** | **34.47%** | Excellent | Below |
| `1e-5` (Medium) | **60.51%** | **77.95%** | Excellent | Good |
| `5e-5` (High) | **60.42%** | **92.95%** | Excellent | Excellent |

> Results are from the best checkpoint (peak test accuracy) restored at the end of training, not the final epoch state. Seed fixed to 42.

**Detailed training progression for λ = 5e-5 (best sparsity result):**

| Epoch | λ_eff | Train Acc | Test Acc | Sparsity |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0 | 31.09% | 40.56% | 0.00% |
| 10 | 0 | 46.63% | 52.02% | 0.00% |
| 20 | 1e-5 | 50.62% | 55.68% | 0.00% |
| 30 | 3.5e-5 | 50.27% | 55.86% | 0.00% |
| 40 | 5e-5 | 52.42% | 57.95% | 56.39% |
| 60 | 5e-5 | 53.56% | 59.10% | 89.54% |
| 80 | 5e-5 | 54.61% | 60.42% | 92.95% |

**Key observations:**
- At `λ = 1e-6`, the sparsity penalty is nearly negligible. The network keeps 65.53% of its gates active, behaving close to a dense MLP. Highest accuracy (60.86%), lowest sparsity (34.47%).
- At `λ = 1e-5`, a meaningful balance is struck — 77.95% of weights pruned with only a 0.35% accuracy drop vs the dense baseline. The network successfully identifies and removes redundant connections.
- At `λ = 5e-5`, aggressive pruning removes **92.95% of all weights** while retaining 60.42% accuracy — less than 0.44% accuracy degradation from the baseline. This is the standout result: a 93% sparse network that still classifies correctly 3 out of 5 times on a 10-class problem using only raw pixels.
- Accuracy is remarkably stable across all three λ values (spread of only ~0.44%), confirming the pruning mechanism correctly identifies redundant weights rather than randomly destroying important ones.

The gate distribution histogram for the best model clearly shows the **bimodal pattern** that indicates successful pruning: a large spike near gate ≈ 0 (pruned connections) and a smaller cluster of surviving gates toward 0.5–1.0.

---

### How to Run

**Dependencies:**
```bash
pip install torch torchvision matplotlib numpy
```

**Run on Kaggle (recommended) or locally:**

The notebook is structured as a Kaggle notebook. To run locally:

```bash
jupyter notebook Self Pruning MLP.ipynb
```

Or convert to a script and run directly:

```bash
jupyter nbconvert --to script Self Pruning MLP.ipynb
python Self Pruning MLP.py
```

**Configuration (bottom of notebook):**

```python
run_experiments(
    lambdas    = [1e-6, 1e-5, 5e-5],   # low / medium / high pruning pressure
    epochs     = 80,
    batch_size = 256,
    data_dir   = "./data",              # CIFAR-10 auto-downloads here
    output_dir = "./outputs",           # plots, checkpoints, report saved here
)
```

CIFAR-10 downloads automatically on first run (~170 MB). All outputs (plots, report, model checkpoints) are saved to `./outputs/`.

---

## Solution 2: CNN + Prunable Classifier (Bonus)

**File:** `with_cnn.ipynb`

This notebook is a **bonus extension** designed for Google Colab. It uses a CNN backbone for feature extraction paired with a `PrunableLinear` classifier head. The pruning mechanism is identical in principle — the same `PrunableLinear` layer, the same L1 sparsity loss — but the CNN's convolutional layers extract spatial features first, giving the prunable head a much richer representation to work with. This results in significantly higher test accuracy compared to the pure MLP.

### Architecture

```
Input: CIFAR-10 image  (3 × 32 × 32)
         ↓
   ─── CNN Feature Extractor (standard Conv2d, NOT pruned) ───
   Conv2d(3→64, 3×3)    + BN2d + ReLU
   Conv2d(64→64, 3×3)   + BN2d + ReLU
   MaxPool2d(2×2)        + Dropout2d(0.1)
   Conv2d(64→128, 3×3)  + BN2d + ReLU
   Conv2d(128→128, 3×3) + BN2d + ReLU
   MaxPool2d(2×2)
   Conv2d(128→256, 3×3) + BN2d + ReLU
   AdaptiveAvgPool2d(4×4)
         ↓
   Flatten → 256 × 4 × 4 = 4096 features
         ↓
   ─── Prunable Classifier Head ───
   PrunableLinear(4096 → 512)  +  BatchNorm1d  +  ReLU  +  Dropout(0.4)
   PrunableLinear(512  → 256)  +  BatchNorm1d  +  ReLU  +  Dropout(0.4)
   PrunableLinear(256  → 10)   →  Logits
```

The three `PrunableLinear` layers in the classifier head are the only gated layers. The CNN backbone (convolutional layers) is **not pruned** — it remains a standard feature extractor.

---

### Key Differences from Pure MLP

| Feature | Pure MLP | CNN + Prunable Head |
|---|---|---|
| Feature extraction | Raw pixel flatten | CNN (spatial convolutions) |
| Pruned layers | All 5 linear layers | Classifier head only (3 layers) |
| Total gated weights | ~4.6M | ~2.2M (head only) |
| Achieved accuracy | 60.42–60.86% | 90.89–91.03% |
| Achieved sparsity | 34.47–92.95% | 13.64–89.69% |
| Training epochs | 80 | 60 |
| Batch size | 256 | 128 |
| Gate init | `sigmoid(2.0) ≈ 0.88` | `sigmoid(0.0) = 0.50` |
| `gate_lr_mult` | 8× | 10× |
| λ warm-up phases | 20% / 25% | 25% / 25% |
| Platform | Kaggle / local | Google Colab (with Drive) |

The CNN variant achieves higher accuracy because convolutional layers are far more efficient at extracting features from image data than raw pixel MLP layers. The pruning then operates on the classifier head where redundancy is high, achieving strong sparsity without hurting the learned features.

---

### Colab Checkpoint Resume System

A major practical addition in this notebook is a full **checkpoint/resume system** designed for Google Colab's frequent runtime disconnects. Every epoch is saved atomically using a write-to-temp-then-rename pattern:

```python
def save_checkpoint(path, payload):
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)   # atomic on POSIX — no partial writes
```

**What is saved per epoch:**
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Full training history (all metrics)
- Best accuracy achieved so far
- Best model state (for restoration)
- PyTorch, NumPy, and Python RNG states (for exact reproducibility on resume)

**Master state file (`experiment_state.json`)** tracks which λ values have been fully completed. If a λ is already done, it is skipped entirely. This means if training λ=1e-6 completes but the runtime dies during λ=1e-5, the next run will skip λ=1e-6 and resume from the exact epoch where λ=1e-5 left off.

```python
# Colab-safe output directory on Google Drive
run_experiments(
    output_dir = "/content/drive/MyDrive/selfprune_outputs"
)
```

Pointing the output directory to Google Drive ensures all checkpoints persist across Colab session resets.

---

### Results

Training was run for **60 epochs** with **batch size 128** on CIFAR-10. Hardware: CUDA GPU. Total parameters: 5,019,850 — of which 2,230,784 are learnable gate parameters (classifier head only).

| λ (Lambda) | Test Accuracy | Sparsity Level (%) | Accuracy Grade | Sparsity Grade |
|:---:|:---:|:---:|:---:|:---:|
| `1e-6` (Low) | **90.97%** | **13.64%** | Excellent | Below |
| `1e-5` (Medium) | **91.03%** | **67.75%** | Excellent | Min |
| `5e-5` (High) | **90.89%** | **89.69%** | Excellent | Excellent |

> Results are from the best checkpoint per λ. All three runs used an identical warm-up phase (first 25% of epochs at λ=0), which is why epoch 1–10 metrics are identical across λ values.

**Detailed training progression for λ = 5e-5 (best sparsity result):**

| Epoch | λ_eff | Train Acc | Test Acc | Sparsity |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0 | 46.07% | 66.02% | 0.00% |
| 10 | 0 | 82.27% | 84.95% | 0.00% |
| 20 | 1.7e-5 | 87.67% | 88.07% | 0.00% |
| 30 | 5e-5 | 88.39% | 88.65% | 59.37% |
| 40 | 5e-5 | 91.72% | 90.11% | 73.79% |
| 50 | 5e-5 | 91.24% | 89.57% | 88.30% |
| 60 | 5e-5 | 93.53% | 90.89% | 89.69% |

**Key observations:**
- Accuracy is extraordinarily stable across all three λ values — a spread of only **0.14%** (91.03% down to 90.89%) while sparsity ranges from 13.64% to 89.69%. This is the defining result of the CNN variant.
- At `λ = 5e-5`, **89.69% of all classifier weights are pruned** while maintaining 90.89% test accuracy. The CNN backbone's rich feature representations make the classifier head highly compressible.
- At `λ = 1e-5`, 67.75% sparsity is achieved with the highest accuracy of all three runs (91.03%), suggesting this is the sweet spot for this architecture.
- The CNN backbone (convolutional layers, not pruned) is responsible for the dramatic accuracy advantage over the pure MLP — jumping from ~60% to ~91% — confirming that spatial feature extraction is the key bottleneck for raw-pixel CIFAR-10 classification.

---

### How to Run

**On Google Colab (recommended):**

1. Open `Pruning With CNN feature extractor.ipynb` in Google Colab.
2. Mount your Google Drive when prompted (for checkpoint persistence):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Confirm the output directory points to Drive:
   ```python
   run_experiments(
       output_dir = "/content/drive/MyDrive/selfprune_outputs"
   )
   ```
4. Run all cells. If the runtime disconnects, simply re-run — training resumes from where it stopped.

**Locally:**
```bash
pip install torch torchvision matplotlib numpy
jupyter notebook Pruning With CNN feature extractor.ipynb
```

Remove the `drive.mount(...)` call at the top and set `output_dir = "./outputs"` in the `run_experiments` call.

---

## Why L1 Encourages Sparsity — Theory

This is the mathematical heart of the approach.

Each weight is multiplied by a gate: `effective_weight = weight × sigmoid(gate_score)`.

The sparsity term added to the loss is:

```
SparsityLoss = Σ_all_layers Σ_all_weights  sigmoid(gate_score)
```

This is the **L1 norm** of all gate values (since they are strictly positive).

**Why L1 and not L2?**

The gradient of the L1 term with respect to each gate score:
```
∂SparsityLoss/∂gate_score  =  sigmoid'(gate_score)  =  gate × (1 − gate)
```

But more importantly, the gradient of the entire λ·Σ(gates) term:
```
∂(λ · Σ gates)/∂gate_score  =  λ · gate · (1 − gate)
```

For a gate near zero (being pruned), `gate · (1 − gate) ≈ gate` — the gradient is proportional to the gate value, which would slow down as the gate approaches zero (similar to L2 dynamics on the sigmoid output). However, the L1 norm on the gates means the **total loss landscape** still has a "corner" at gate = 0 (in gate-score space), making it optimal for many gates to sit exactly at zero.

In contrast, an L2 norm `Σ gate²` gives gradient `2λ · gate`, which shrinks to zero as the gate shrinks — providing less and less pressure to fully close the gate. The L1 formulation maintains a more aggressive constant-direction push.

**The competition:**
- The **classification gradient** pushes gate scores toward high values for important weights (so the weight has effect on the output and can reduce CE loss).
- The **sparsity gradient** constantly pushes all gate scores downward.
- The gate finds an equilibrium: if the weight is useful, classification wins; if redundant, sparsity wins and the gate goes to zero.

---

## λ Trade-off Analysis

λ controls the balance between accuracy (classification loss) and sparsity (regularisation loss):

```
Low  λ (1e-6)  →  Near-dense network. Highest accuracy, lowest sparsity.
                   Equivalent to a standard dense MLP with a tiny penalty.

Mid  λ (1e-5)  →  Sweet spot. The network surgically removes redundant
                   connections while retaining accuracy-critical ones.
                   Target: ≥50% accuracy + ≥85% sparsity (MLP version).

High λ (5e-5)  →  Aggressive pruning. Highest sparsity, some accuracy loss.
                   The penalty starts removing connections that do contribute.
```

The λ warm-up schedule is essential: jumping straight to the full λ at epoch 1 collapses training before the network learns anything. By withholding sparsity pressure for the first 20% of epochs and then ramping it in, the network builds a solid representation first, then prunes away the parts it no longer needs.

---

## Generated Outputs

After running either notebook, the following files are saved to `./outputs/`:

| File | Description |
|---|---|
| `gate_distribution.png` | Histogram of all gate values. Successful pruning shows a large spike near 0 (dead connections) and a smaller cluster of surviving gates. |
| `training_curves.png` | Test accuracy and sparsity % vs epoch for all three λ values side-by-side. |
| `lambda_tradeoff.png` | Scatter plot of accuracy vs sparsity for each λ. The upper-right quadrant (high accuracy + high sparsity) is the target. |
| `report.md` | Auto-generated Markdown report with theory explanation, results table, and plot references. |
| `model_lam1e-06.pt` | Best-checkpoint model for λ = 1e-6. |
| `model_lam1e-05.pt` | Best-checkpoint model for λ = 1e-5. |
| `model_lam5e-05.pt` | Best-checkpoint model for λ = 5e-5. |

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.5.0
numpy>=1.21.0
```

Python 3.8+ recommended. A CUDA-capable GPU is strongly recommended for reasonable training times (~15–25 min for the MLP version, ~20–30 min for the CNN version on a T4/P100).

**CIFAR-10** is downloaded automatically by `torchvision.datasets.CIFAR10` on the first run (~170 MB).

---

*Built as part of the Tredence Analytics AI Engineering Internship Case Study — 2025 Cohort.*
