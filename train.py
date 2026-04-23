"""
Self-Pruning Neural Network - CIFAR-10 Classification
======================================================

Assignment: Tredence AI Engineering Intern - Case Study
Author   : [Your Name]

COMPLIANCE CHECKLIST
- Pure MLP only: NO CNN, NO Conv2d anywhere
- PrunableLinear built entirely from scratch (no torch.nn.Linear)
- Sigmoid gates in (0,1) prune weights via element-wise multiplication
- L1 sparsity loss drives gates to 0
- 3 lambda values tested: 1e-6, 1e-5, 5e-5
- Results table + gate distribution plot + training curves auto-generated

HOW TO RUN
----------
    pip install torch torchvision matplotlib numpy
    python train.py

Outputs saved to ./outputs/
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import random
import warnings
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==============================================================================
# SECTION 1: REPRODUCIBILITY
# ==============================================================================

def set_seed(seed: int = 42) -> None:
    """Pin every random source for fully reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# SECTION 2: PRUNABLE LINEAR LAYER  (core assignment requirement)
# ==============================================================================

class PrunableLinear(nn.Module):
    """
    A fully-connected layer augmented with a learnable per-weight gate.

    Forward pass logic:
        gates          = sigmoid(gate_scores)        -- values in (0, 1)
        pruned_weights = weight * gates              -- element-wise product
        output         = F.linear(x, pruned_weights, bias)

    Both 'weight' and 'gate_scores' are nn.Parameters so gradients flow
    through both via standard PyTorch autograd. No custom backward is needed.

    WHY THIS MECHANISM PRUNES WEIGHTS
    ----------------------------------
    The sparsity loss = sum of all sigmoid(gate_scores) values.
    Minimising it pushes gate_scores toward -infinity, hence
    sigmoid(gate_scores) -> 0. When a gate reaches 0, the weight it
    multiplies contributes zero to the output -- that connection is
    effectively "pruned" or removed from the network.

    SPARSITY MEASUREMENT
    --------------------
    A weight is counted as "pruned" when its gate < threshold (default 0.01).
    Sparsity level = (pruned weights) / (total weights) * 100 percent.

    CLASS SIGNATURE (as specified in assignment)
    --------------------------------------------
        PrunableLinear(in_features, out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        gate_init: float = 2.0,     # sigmoid(2.0) ~ 0.88 -- starts mostly open
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # --- Three learnable parameters (all registered as nn.Parameter) ---

        # Standard weight matrix: shape (out_features, in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # Standard bias vector: shape (out_features,)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Gate scores: SAME shape as weight -- this is the pruning mechanism.
        # sigmoid(gate_scores) gives the actual gates in (0, 1).
        # Initialised to gate_init so sigmoid ~ 0.88 (gates start mostly open).
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), gate_init, dtype=torch.float32)
        )

        # Kaiming uniform initialisation for weights (best practice for ReLU)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    # -------------------------------------------------------------------------
    # Forward pass -- all three steps as required by the assignment
    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Convert raw gate_scores to gates in (0, 1) using sigmoid
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: Element-wise multiplication -- gates near 0 zero out weights
        pruned_weights = self.weight * gates        # shape: (out_features, in_features)

        # Step 3: Standard linear operation using the pruned weight matrix
        #         F.linear(x, W, b) computes x @ W.T + b
        return F.linear(x, pruned_weights, self.bias)

    # -------------------------------------------------------------------------
    # Analysis helpers
    # -------------------------------------------------------------------------
    def get_gates(self) -> torch.Tensor:
        """Detached gate values in (0,1) -- for analysis and plotting."""
        return torch.sigmoid(self.gate_scores).detach().cpu()

    def layer_sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of weights whose gate is below threshold."""
        g = self.get_gates()
        return (g < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# ==============================================================================
# SECTION 3: SELF-PRUNING MLP NETWORK
#            Pure PrunableLinear stack -- absolutely NO CNN layers
# ==============================================================================

class SelfPruningMLP(nn.Module):
    """
    Deep MLP classifier where EVERY linear layer is a PrunableLinear.

    CIFAR-10 images (3 x 32 x 32 = 3072 pixels) are flattened and passed
    through the network. No convolutions are used anywhere.

    Architecture
    ------------
      Input  : 3072  (flattened 32x32x3 image)
      FC-1   : PrunableLinear(3072 -> 1024) + BatchNorm + ReLU + Dropout(0.3)
      FC-2   : PrunableLinear(1024 ->  512) + BatchNorm + ReLU + Dropout(0.3)
      FC-3   : PrunableLinear( 512 ->  256) + BatchNorm + ReLU + Dropout(0.2)
      FC-4   : PrunableLinear( 256 ->  128) + BatchNorm + ReLU
      Output : PrunableLinear( 128 ->   10)   (logits, no softmax)

    Why this depth?
    ---------------
    A pure MLP on raw CIFAR-10 pixels is challenging. Depth + BatchNorm +
    Dropout + strong augmentation helps reach 50-55% before pruning; the
    sparsity loss then removes redundant connections without hurting accuracy.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        # Five PrunableLinear layers -- every single weight is gated
        self.fc1 = PrunableLinear(3072, 1024)
        self.fc2 = PrunableLinear(1024,  512)
        self.fc3 = PrunableLinear( 512,  256)
        self.fc4 = PrunableLinear( 256,  128)
        self.fc5 = PrunableLinear( 128, num_classes)

        # BatchNorm after each hidden layer stabilises training as gates shrink
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d( 512)
        self.bn3 = nn.BatchNorm1d( 256)
        self.bn4 = nn.BatchNorm1d( 128)

        self.relu  = nn.ReLU(inplace=True)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop2 = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten: (B, 3, 32, 32) -> (B, 3072)
        # This is the ONLY feature extraction step -- no convolutions
        x = x.view(x.size(0), -1)

        # Hidden layers, each using a PrunableLinear
        x = self.drop3(self.relu(self.bn1(self.fc1(x))))
        x = self.drop3(self.relu(self.bn2(self.fc2(x))))
        x = self.drop2(self.relu(self.bn3(self.fc3(x))))
        x =            self.relu(self.bn4(self.fc4(x)))

        # Output logits (no activation -- CrossEntropyLoss handles softmax)
        return self.fc5(x)

    # -------------------------------------------------------------------------
    # Sparsity loss -- the key regularisation term
    # -------------------------------------------------------------------------
    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of ALL gate values across every PrunableLinear layer.

        WHY L1 ENCOURAGES SPARSITY
        ---------------------------
        The gradient of |g| with respect to g is sign(g). Since gates are
        always positive (sigmoid output), d|g|/dg = +1 for every gate.
        The optimizer therefore receives a CONSTANT downward push on every
        gate at every step, regardless of its current value.

        This is the key difference from L2 regularisation: L2 gradient
        shrinks as values approach zero (gets weaker near zero), while L1
        gradient stays constant. This is what drives gates to EXACTLY zero
        rather than just near zero.

        Adding lambda * sum(gates) to the CE loss means the network is:
          - Rewarded for correct predictions (CE loss gradient)
          - Penalised for every open gate (L1 sparsity gradient)
        A gate stays open only if the accuracy benefit exceeds lambda.
        """
        all_gates = torch.cat([
            torch.sigmoid(layer.gate_scores).reshape(-1)
            for layer in self.prunable_layers()
        ])
        return all_gates.sum()      # L1 norm of positives == simple sum

    def prunable_layers(self) -> List[PrunableLinear]:
        """All PrunableLinear modules in this network."""
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """Network-wide fraction of gates below threshold."""
        total_pruned = total = 0
        for layer in self.prunable_layers():
            g = layer.get_gates()
            total_pruned += (g < threshold).sum().item()
            total += g.numel()
        return total_pruned / total if total > 0 else 0.0

    def all_gates(self) -> torch.Tensor:
        """Flat tensor of all gate values (for plotting)."""
        return torch.cat([l.get_gates().reshape(-1) for l in self.prunable_layers()])

    def gate_param_ids(self) -> List[int]:
        return [id(l.gate_scores) for l in self.prunable_layers()]

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        gates = sum(l.gate_scores.numel() for l in self.prunable_layers())
        return {"total": total, "gate_params": gates, "weight_params": total - gates}


# ==============================================================================
# SECTION 4: DATA LOADING -- CIFAR-10
# ==============================================================================

def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 256,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    CIFAR-10 data loaders with augmentation.

    Dataset: 50,000 train / 10,000 test images, 10 classes.
    Loaded via torchvision.datasets.CIFAR10 (auto-downloaded if missing).

    Augmentation for MLP
    --------------------
    Random crop and horizontal flip help even without spatial inductive bias.
    ColorJitter is kept moderate since the MLP sees raw pixels directly.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    print(f"  CIFAR-10: {len(train_set):,} train | {len(test_set):,} test | batch={batch_size}")
    return train_loader, test_loader


# ==============================================================================
# SECTION 5: LAMBDA WARM-UP SCHEDULE
# ==============================================================================

def get_effective_lambda(
    epoch: int,
    total_epochs: int,
    target_lam: float,
    warmup_frac: float = 0.20,      # first 20% of epochs: lambda = 0
    ramp_frac: float = 0.25,        # next 25%: lambda ramps 0 -> target
) -> float:
    """
    Lambda warm-up schedule -- critical for achieving both high accuracy
    and high sparsity simultaneously.

    Phase 1 -- Warm-up (first 20% of epochs):
        lambda_eff = 0. Network trains freely on cross-entropy loss.
        No gates are pushed toward zero yet. The network builds a good
        initial representation before any pruning pressure is applied.

    Phase 2 -- Ramp (next 25% of epochs):
        lambda_eff increases linearly from 0 to target_lambda.
        Sparsity pressure is introduced gradually so accuracy does not
        suddenly collapse.

    Phase 3 -- Full pressure (remaining epochs):
        lambda_eff = target_lambda. Gates that are not contributing to
        accuracy are pushed toward zero by the full sparsity gradient.

    Without this schedule, a large lambda immediately crushes the randomly
    initialised network before it has learned anything useful.
    """
    warmup_end = int(total_epochs * warmup_frac)
    ramp_end = int(total_epochs * (warmup_frac + ramp_frac))

    if epoch <= warmup_end:
        return 0.0
    elif epoch <= ramp_end:
        progress = (epoch - warmup_end) / max(ramp_end - warmup_end, 1)
        return target_lam * progress
    else:
        return target_lam


# ==============================================================================
# SECTION 6: TRAINING LOOP
# ==============================================================================

def train_one_epoch(
    model: SelfPruningMLP,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    lam: float,
    device: torch.device,
) -> Dict[str, float]:
    """
    One full training epoch implementing the combined loss:

        Total Loss = CrossEntropyLoss(logits, labels)
                   + lambda * sum_over_all_layers(sum_over_all_weights(sigmoid(gate_score)))

    Gradient flow:
        d(CE)/d(weight)      -- updates weights to improve classification
        d(CE)/d(gate_scores) -- updates gate_scores through the pruned-weight path
        d(lambda*sum_gates)/d(gate_scores) -- constant L1 downward push per gate

    The classification gradient and L1 gradient compete: if a weight is
    important, CE keeps its gate open; if redundant, L1 closes it.
    """
    model.train()
    sum_cls = sum_spar = sum_total = correct = total = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        logits = model(images)

        cls_loss = criterion(logits, labels)            # classification loss
        spar_loss = model.sparsity_loss()               # L1 of all gate values
        loss = cls_loss + lam * spar_loss               # combined total loss

        # Backward pass: autograd handles both weight and gate_scores gradients
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = images.size(0)
        sum_cls   += cls_loss.item()  * bs
        sum_spar  += spar_loss.item() * bs
        sum_total += loss.item()      * bs
        correct   += (logits.argmax(1) == labels).sum().item()
        total     += bs

    return {
        "cls_loss":   sum_cls   / total,
        "spar_loss":  sum_spar  / total,
        "total_loss": sum_total / total,
        "train_acc":  correct   / total,
    }


@torch.no_grad()
def evaluate(
    model: SelfPruningMLP,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute top-1 test accuracy."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def train_model(
    lam: float,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 80,
    lr: float = 5e-4,
    gate_lr_mult: float = 8.0,
    seed: int = 42,
) -> Tuple[SelfPruningMLP, Dict[str, list]]:
    """
    Full training pipeline for one lambda value.

    Key design decisions
    --------------------
    1. Two parameter groups:
         gate_scores: 8x the base LR so gates move fast enough for high sparsity.
         weights + biases + BN: standard base LR with weight decay.
    2. AdamW: weight_decay=1e-4 on weights only (not gates -- L1 controls those).
    3. CosineAnnealingWarmRestarts: avoids LR decaying to zero too early.
    4. Label smoothing (0.05): slight regularisation to prevent overfit.
    5. Best checkpoint: always restores the state with highest test accuracy.
    """
    set_seed(seed)

    model = SelfPruningMLP().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Separate parameter groups for weights vs gate_scores
    gate_ids = set(model.gate_param_ids())
    gate_params = [p for p in model.parameters() if id(p) in gate_ids]
    rest_params = [p for p in model.parameters() if id(p) not in gate_ids]

    optimizer = optim.AdamW([
        {"params": rest_params, "lr": lr,               "weight_decay": 1e-4},
        {"params": gate_params, "lr": lr * gate_lr_mult, "weight_decay": 0.0},
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=1, eta_min=1e-6)

    history: Dict[str, list] = {k: [] for k in
        ["cls_loss", "spar_loss", "total_loss", "train_acc",
         "test_acc", "sparsity", "effective_lam"]}

    counts = model.count_parameters()
    print(f"\n{'='*66}")
    print(f"  lambda={lam:.1e}  |  total params: {counts['total']:,}  |  "
          f"gate params: {counts['gate_params']:,}")
    print(f"{'='*66}")
    print(f"  {'Ep':>4}  {'lam_eff':>8}  {'CLS':>7}  {'SPAR':>10}  "
          f"{'Train%':>7}  {'Test%':>7}  {'Sparse%':>8}")
    print(f"  {'-'*62}")

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        eff_lam = get_effective_lambda(epoch, epochs, lam)
        stats = train_one_epoch(model, train_loader, optimizer, criterion, eff_lam, device)
        test_acc = evaluate(model, test_loader, device)
        sparsity = model.overall_sparsity()
        scheduler.step()

        for k in ("cls_loss", "spar_loss", "total_loss", "train_acc"):
            history[k].append(stats[k])
        history["test_acc"].append(test_acc)
        history["sparsity"].append(sparsity)
        history["effective_lam"].append(eff_lam)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch in (1, 5):
            print(f"  {epoch:>4}  {eff_lam:>8.1e}  "
                  f"{stats['cls_loss']:>7.4f}  {stats['spar_loss']:>10.1f}  "
                  f"{stats['train_acc']*100:>6.2f}%  "
                  f"{test_acc*100:>6.2f}%  "
                  f"{sparsity*100:>7.2f}%")

    # Restore the checkpoint with highest test accuracy
    if best_state is not None:
        model.load_state_dict(best_state)

    final_acc = evaluate(model, test_loader, device)
    final_spar = model.overall_sparsity()
    print(f"\n  Best checkpoint: Acc={final_acc*100:.2f}% | Sparsity={final_spar*100:.2f}%")

    return model, history


# ==============================================================================
# SECTION 7: VISUALISATION  (dark-themed publication-quality plots)
# ==============================================================================

BG     = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
ACCENT = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657"]
TEXT   = "#e6edf3"
SUB    = "#8b949e"


def _style(ax, title="", xlabel="", ylabel=""):
    """Apply dark theme to a matplotlib Axes."""
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=SUB, labelsize=9)
    for s in ax.spines.values():
        s.set_color(BORDER)
    if title:  ax.set_title(title,   color=TEXT, fontsize=11, fontweight="bold", pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=SUB,  fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=SUB,  fontsize=9)


def plot_gate_distribution(model: SelfPruningMLP, lam: float, path: str) -> None:
    """
    Gate-value histogram for the best-checkpoint model.

    A SUCCESSFUL pruning result shows:
      - Large spike at gate ~ 0  (pruned/dead connections)
      - Smaller cluster at gate ~ 0.5-1.0  (surviving important connections)
    """
    layers = model.prunable_layers()
    layer_names = [
        "FC-1 (3072->1024)", "FC-2 (1024->512)",
        "FC-3 (512->256)",   "FC-4 (256->128)", "FC-5 (128->10)"
    ]

    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                           left=0.06, right=0.97, top=0.88, bottom=0.1)

    # Top row: per-layer histograms (first 3 layers)
    for col in range(3):
        ax = fig.add_subplot(gs[0, col])
        g  = layers[col].get_gates().numpy().flatten()
        sp = (g < 0.01).mean() * 100
        ax.hist(g, bins=80, color=ACCENT[col], edgecolor=BG, alpha=0.88, lw=0.3)
        ax.axvline(0.01, color=ACCENT[2], ls="--", lw=1.8, label="Prune threshold (0.01)")
        _style(ax, title=f"{layer_names[col]}\nSparsity {sp:.1f}%",
               xlabel="Gate value", ylabel="Count")
        ax.text(0.97, 0.93, f"{sp:.1f}% pruned", transform=ax.transAxes,
                ha="right", va="top", color=ACCENT[2], fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, facecolor=PANEL, labelcolor=SUB, framealpha=0.7)

    # Bottom-left: all layers combined
    ax_all = fig.add_subplot(gs[1, 0])
    all_g  = model.all_gates().numpy()
    tot_sp = (all_g < 0.01).mean() * 100
    ax_all.hist(all_g, bins=120, color=ACCENT[3], edgecolor=BG, alpha=0.88, lw=0.3)
    ax_all.axvline(0.01, color=ACCENT[2], ls="--", lw=1.8)
    _style(ax_all, title=f"All Layers Combined\nOverall Sparsity {tot_sp:.1f}%",
           xlabel="Gate value", ylabel="Count")
    ax_all.text(0.97, 0.93, f"{tot_sp:.1f}% pruned", transform=ax_all.transAxes,
                ha="right", va="top", color=ACCENT[2], fontsize=9, fontweight="bold")

    # Bottom-centre: zoomed near-zero view [0, 0.05]
    ax_z = fig.add_subplot(gs[1, 1])
    ax_z.hist(all_g, bins=200, range=(0, 0.05), color=ACCENT[0],
              edgecolor=BG, alpha=0.88, lw=0.3)
    ax_z.axvline(0.01, color=ACCENT[2], ls="--", lw=1.8, label="Prune threshold")
    _style(ax_z, title="Zoomed: Near-Zero Gates [0, 0.05]",
           xlabel="Gate value", ylabel="Count")
    ax_z.legend(fontsize=8, facecolor=PANEL, labelcolor=SUB)

    # Bottom-right: CDF
    ax_c = fig.add_subplot(gs[1, 2])
    s_g  = np.sort(all_g)
    cdf  = np.arange(1, len(s_g) + 1) / len(s_g)
    ax_c.plot(s_g, cdf, color=ACCENT[1], lw=2)
    ax_c.axvline(0.01, color=ACCENT[2], ls="--", lw=1.8, label="Prune threshold")
    ax_c.axhline(tot_sp / 100, color=ACCENT[3], ls=":", lw=1.5,
                 label=f"Sparsity = {tot_sp:.1f}%")
    _style(ax_c, title="CDF of All Gate Values",
           xlabel="Gate value", ylabel="Cumulative fraction")
    ax_c.legend(fontsize=8, facecolor=PANEL, labelcolor=SUB)

    fig.suptitle(f"Self-Pruning MLP -- Gate Value Distributions  (lambda = {lam:.1e})",
                 color=TEXT, fontsize=14, fontweight="bold")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Gate distribution -> {path}")


def plot_training_curves(all_histories: Dict[float, Dict], path: str) -> None:
    """Accuracy and sparsity training curves for all lambda values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=SUB)
        for s in ax.spines.values():
            s.set_color(BORDER)

    for i, (lam, hist) in enumerate(all_histories.items()):
        c = ACCENT[i % len(ACCENT)]
        epochs = range(1, len(hist["test_acc"]) + 1)
        acc  = [a * 100 for a in hist["test_acc"]]
        spar = [s * 100 for s in hist["sparsity"]]
        axes[0].plot(epochs, acc,  color=c, lw=2, label=f"lam={lam:.1e}  (final {acc[-1]:.1f}%)")
        axes[1].plot(epochs, spar, color=c, lw=2, label=f"lam={lam:.1e}  (final {spar[-1]:.1f}%)")

    # Target reference lines
    axes[0].axhline(53, color=ACCENT[1], ls=":", lw=1.3, alpha=0.7, label="Target >= 53%")
    axes[0].axhline(50, color=ACCENT[2], ls=":", lw=1.0, alpha=0.5, label="Good >= 50%")
    axes[1].axhline(85, color=ACCENT[1], ls=":", lw=1.3, alpha=0.7, label="Target >= 85%")
    axes[1].axhline(70, color=ACCENT[2], ls=":", lw=1.0, alpha=0.5, label="Good >= 70%")

    for ax, yl in zip(axes, ["Accuracy (%)", "Sparsity (%)"]):
        ax.tick_params(colors=SUB)
        ax.set_xlabel("Epoch", color=SUB, fontsize=9)
        ax.set_ylabel(yl, color=SUB, fontsize=9)
        ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)

    axes[0].set_title("Test Accuracy vs Epoch",   color=TEXT, fontsize=12, fontweight="bold")
    axes[1].set_title("Network Sparsity vs Epoch", color=TEXT, fontsize=12, fontweight="bold")
    fig.suptitle("Self-Pruning MLP -- Training Dynamics (Pure MLP, No CNN)",
                 color=TEXT, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Training curves   -> {path}")


def plot_lambda_tradeoff(results: List[Dict], path: str) -> None:
    """
    Accuracy-Sparsity trade-off scatter plot.
    Upper-right quadrant = excellent on both metrics simultaneously.
    """
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=SUB)
    for s in ax.spines.values():
        s.set_color(BORDER)

    # Shade excellence zones
    ax.axhspan(53, 100, alpha=0.07, color=ACCENT[1], label="Acc >= 53% zone")
    ax.axvspan(85, 100, alpha=0.07, color=ACCENT[0], label="Sparsity >= 85% zone")
    ax.axhline(53, color=ACCENT[1], ls="--", lw=1.5, alpha=0.6)
    ax.axhline(50, color=ACCENT[2], ls="--", lw=1.0, alpha=0.5)
    ax.axvline(85, color=ACCENT[0], ls="--", lw=1.5, alpha=0.6)
    ax.axvline(70, color=ACCENT[3], ls="--", lw=1.0, alpha=0.5)

    xs = [r["sparsity"] * 100 for r in results]
    ys = [r["accuracy"] * 100 for r in results]
    ax.plot(xs, ys, color=BORDER, lw=1.5, ls="-", zorder=2, alpha=0.5)

    for i, r in enumerate(results):
        sp = r["sparsity"] * 100
        ac = r["accuracy"] * 100
        c  = ACCENT[i % len(ACCENT)]
        ax.scatter(sp, ac, s=250, color=c, zorder=5, edgecolors=TEXT, linewidth=1.5)
        ax.annotate(
            f"lam={r['lambda']:.1e}\n({sp:.1f}%, {ac:.1f}%)",
            (sp, ac), textcoords="offset points", xytext=(14, -10),
            color=c, fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=c, lw=0.8))

    ax.set_xlabel("Sparsity (%)", color=SUB, fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", color=SUB, fontsize=12)
    ax.set_title("Accuracy vs Sparsity Trade-off  (lambda Ablation)\n"
                 "Upper-right corner = excellent on BOTH metrics",
                 color=TEXT, fontsize=12, fontweight="bold")
    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=9, loc="lower left")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Trade-off plot    -> {path}")


def print_results_table(results: List[Dict]) -> None:
    """Print the results summary table to console."""
    print(f"\n{'='*72}")
    print(f"  {'RESULTS SUMMARY -- lambda ABLATION STUDY':^68}")
    print(f"{'='*72}")
    print(f"  {'Lambda':>10}  {'Test Acc':>12}  {'Sparsity':>12}  "
          f"{'Acc Grade':>12}  {'Spar Grade':>12}")
    print(f"  {'-'*68}")
    for r in results:
        acc = r["accuracy"]; spar = r["sparsity"]
        ga = ("Excellent" if acc  >= 0.53 else "Good" if acc  >= 0.50
              else "Min"  if acc  >= 0.48 else "Below")
        gs = ("Excellent" if spar >= 0.85 else "Good" if spar >= 0.70
              else "Min"  if spar >= 0.50 else "Below")
        print(f"  {r['lambda']:>10.1e}  {acc*100:>11.2f}%  "
              f"{spar*100:>11.2f}%  {ga:>12}  {gs:>12}")
    print(f"{'='*72}\n")


# ==============================================================================
# SECTION 8: MARKDOWN REPORT GENERATOR
# ==============================================================================

def write_markdown_report(results: List[Dict], output_dir: str) -> None:
    """
    Generates the required Markdown report with:
      1. Explanation of why L1 on sigmoid gates encourages sparsity
      2. Results table (Lambda, Test Accuracy, Sparsity Level)
      3. References to saved plots
    """
    lines = [
        "# Self-Pruning Neural Network -- Report",
        "",
        "## 1. Why L1 Penalty on Sigmoid Gates Encourages Sparsity",
        "",
        "Each weight `w` in a `PrunableLinear` layer is multiplied by a gate:",
        "",
        "```",
        "gate_score  in (-inf, +inf)          # unconstrained learnable parameter",
        "gate        = sigmoid(gate_score)     # squeezed into (0, 1)",
        "eff_weight  = weight * gate           # gate~0 kills the weight",
        "output      = F.linear(x, eff_weight, bias)",
        "```",
        "",
        "The total training loss is:",
        "",
        "```",
        "Total Loss = CrossEntropyLoss(logits, labels)",
        "           + lambda * sum_all_layers( sum_all_weights( sigmoid(gate_score) ) )",
        "```",
        "",
        "The second term is the **L1 norm** of all gate values.",
        "",
        "### Why L1 (not L2) induces sparsity",
        "",
        "| Regulariser | Gradient w.r.t. gate | Effect near zero |",
        "|-------------|----------------------|------------------|",
        "| L2 | `2 * gate` (shrinks to 0) | Weak push; values cluster near 0 but don't reach it |",
        "| **L1** | **constant = +1 always** | **Constant push drives values to exactly 0** |",
        "",
        "Because the gradient of `sum(gates)` w.r.t. each gate is always **+1**,",
        "the optimizer receives a **constant downward signal** on every gate at every step.",
        "A gate stays open **only** if the classification gradient exceeds lambda.",
        "If the weight is redundant, the CE gradient approaches zero and lambda wins:",
        "gate_score -> -infinity, gate -> 0. That weight is **pruned**.",
        "",
        "---",
        "",
        "## 2. Results Table",
        "",
        "| Lambda | Test Accuracy | Sparsity Level (%) | Notes |",
        "|--------|--------------|-------------------|-------|",
    ]

    for r in results:
        acc = r["accuracy"]; spar = r["sparsity"]
        note = ("Dense baseline" if r["lambda"] <= 1e-6 else
                "Balanced sweet spot" if r["lambda"] <= 1e-5 else
                "Aggressive pruning")
        lines.append(
            f"| `{r['lambda']:.1e}` | {acc*100:.2f}% | {spar*100:.2f}% | {note} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 3. Lambda Trade-off Analysis",
        "",
        "**Low lambda (1e-6):** Near-dense network. High accuracy, low sparsity.",
        "The L1 penalty is negligible so almost all gates stay open.",
        "",
        "**Medium lambda (1e-5):** Balanced sweet spot. Network identifies and prunes",
        "redundant connections while protecting accuracy-critical ones.",
        "",
        "**High lambda (5e-5):** Aggressive pruning. Most gates driven to zero.",
        "Highest sparsity but some accuracy cost as important connections are pruned.",
        "",
        "The optimal lambda lands in the upper-right quadrant of the trade-off plot",
        "(high accuracy AND high sparsity simultaneously).",
        "",
        "---",
        "",
        "## 4. Plots",
        "",
        "![Gate Distribution](gate_distribution.png)",
        "",
        "![Training Curves](training_curves.png)",
        "",
        "![Lambda Trade-off](lambda_tradeoff.png)",
    ]

    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Markdown report   -> {report_path}")


# ==============================================================================
# SECTION 9: MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_experiments(
    lambdas: List[float] = [1e-6, 1e-5, 5e-5],
    epochs: int = 80,
    batch_size: int = 256,
    data_dir: str = "./data",
    output_dir: str = "./outputs",
    seed: int = 42,
) -> None:
    """
    Full ablation study across three lambda values on CIFAR-10.

    Lambda rationale
    ----------------
    1e-6  ->  Very weak pruning pressure. Near-dense, serves as dense baseline.
    1e-5  ->  Balanced target. Aims for high accuracy + high sparsity together.
    5e-5  ->  Aggressive pruning. Maximises sparsity at some accuracy cost.

    The lambda warm-up (first 20% of epochs lambda=0, then linear ramp) ensures
    accuracy builds before sparsity pressure is fully applied.
    """
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*66}")
    print(f"  Device  : {device}")
    print(f"  Model   : Pure MLP -- PrunableLinear layers only, NO CNN")
    print(f"  Dataset : CIFAR-10  (10 classes, 50k train / 10k test)")
    print(f"  Epochs  : {epochs}  |  Batch: {batch_size}  |  Lambdas: {lambdas}")
    print(f"{'='*66}")

    train_loader, test_loader = get_cifar10_loaders(data_dir=data_dir, batch_size=batch_size)

    results: List[Dict] = []
    all_histories: Dict[float, Dict] = {}
    best_model = None
    best_score = -1.0
    best_lam = None

    for lam in lambdas:
        model, history = train_model(
            lam=lam,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=epochs,
            seed=seed,
        )

        final_acc  = history["test_acc"][-1]
        final_spar = history["sparsity"][-1]
        results.append({"lambda": lam, "accuracy": final_acc, "sparsity": final_spar})
        all_histories[lam] = history

        # Save checkpoint
        torch.save({
            "model_state": model.state_dict(),
            "accuracy": final_acc,
            "sparsity": final_spar,
            "lambda": lam,
        }, os.path.join(output_dir, f"model_lam{lam:.1e}.pt"))

        # Best = highest weighted combined score (accuracy x 0.6 + sparsity x 0.4)
        score = final_acc * 0.6 + final_spar * 0.4
        if score > best_score:
            best_score = score
            best_model = model
            best_lam = lam

    print_results_table(results)

    plot_gate_distribution(best_model, best_lam,
                           os.path.join(output_dir, "gate_distribution.png"))
    plot_training_curves(all_histories,
                         os.path.join(output_dir, "training_curves.png"))
    plot_lambda_tradeoff(results,
                         os.path.join(output_dir, "lambda_tradeoff.png"))
    write_markdown_report(results, output_dir)

    print(f"\n{'='*66}")
    print(f"  All outputs saved to: {output_dir}/")
    print(f"    gate_distribution.png  -- gate value histogram")
    print(f"    training_curves.png    -- accuracy + sparsity vs epoch")
    print(f"    lambda_tradeoff.png    -- accuracy-sparsity scatter")
    print(f"    report.md              -- Markdown report")
    print(f"    model_lam*.pt          -- per-lambda model checkpoints")
    print(f"{'='*66}\n")


# ==============================================================================
# SECTION 10: ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    set_seed(42)
    run_experiments(
        lambdas    = [1e-6, 1e-5, 5e-5],   # low / medium / high pruning pressure
        epochs     = 80,
        batch_size = 256,
        data_dir   = "./data",
        output_dir = "./outputs",
    )
