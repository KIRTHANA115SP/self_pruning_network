# 🧠 Self-Pruning Neural Network on CIFAR-10

> **Tredence AI Engineering Internship – Case Study Submission**

A feed-forward neural network that **learns to prune its own weights during training** — no post-training surgery required. Built with PyTorch on the CIFAR-10 image classification benchmark.

---

## 💡 Core Idea

Standard pruning removes weights *after* training. This project takes it further — every weight in the network is paired with a learnable **gate parameter**. During training, an L1 sparsity penalty pressures most gates toward zero, effectively silencing unimportant weights on the fly.

```
gates        = sigmoid(gate_scores)       # learnable, ∈ (0, 1)
pruned_weight = weight ⊙ gates            # element-wise masking
output        = pruned_weight · x + bias  # standard linear op
```

---

## 📁 Project Structure

```
📦 tredence-ai-case-study
 ├── self_pruning_network.py   # Full implementation (model + training + evaluation)
 ├── report.md                 # Analysis report with results and explanation
 ├── results_table.csv         # Lambda vs Accuracy vs Sparsity (generated on run)
 ├── gate_distribution.png     # Gate value histogram plot (generated on run)
 └── README.md
```

---

## ⚙️ How It Works

### 1. `PrunableLinear` Layer
A custom replacement for `nn.Linear` with an extra `gate_scores` parameter tensor (same shape as `weight`). In the forward pass, gates are computed via sigmoid and multiplied element-wise with the weights — allowing gradients to flow through both `weight` and `gate_scores`.

### 2. Sparsity Loss (L1)
```
Total Loss = CrossEntropyLoss + λ × SparsityLoss

SparsityLoss = Σ sigmoid(gate_scores)   # sum of all gate values
```
The L1 norm exerts a **constant gradient** on every gate regardless of its magnitude, pushing gates all the way to zero — unlike L2 which only shrinks them. This creates genuinely sparse (pruned) networks.

### 3. Training
- Optimizer: Adam (lr=1e-3, cosine annealing)
- Epochs: 25
- Three experiments with λ ∈ `{1e-5, 1e-4, 1e-3}` to show the sparsity–accuracy trade-off

---

## 🚀 Quickstart

### Install dependencies
```bash
pip install torch torchvision matplotlib
```

### Run the experiment
```bash
python self_pruning_network.py
```

CIFAR-10 (~170MB) is downloaded automatically on the first run.

---

## 📊 Results

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|--------|:-----------------:|:------------------:|
| 1e-5   | ~57–60            | ~10–20             |
| 1e-4   | ~53–57            | ~40–60             |
| 1e-3   | ~45–50            | ~75–90             |

> Higher λ → more weights pruned → lower accuracy. The λ knob lets you dial in the compression you need.

### Gate Distribution (best model)

After training, gate values show a clear **bimodal distribution**:
- 📌 **Spike near 0** — weights pruned by the sparsity penalty
- 📌 **Cluster near 0.5–1.0** — weights kept because they matter for classification

![Gate Distribution](gate_distribution.png)

---

## 🏗️ Network Architecture

```
Input (3072)
    ↓
PrunableLinear(3072 → 1024) → BatchNorm → ReLU → Dropout(0.3)
    ↓
PrunableLinear(1024 → 512)  → BatchNorm → ReLU → Dropout(0.3)
    ↓
PrunableLinear(512 → 256)   → BatchNorm → ReLU → Dropout(0.3)
    ↓
PrunableLinear(256 → 10)    [classifier]
    ↓
Output (10 classes)
```

All four linear layers participate in pruning — every weight in the network has a gate.

---

## 📄 Output Files

| File | Description |
|------|-------------|
| `results_table.csv` | Experiment results for all λ values |
| `gate_distribution.png` | Histogram of gate values for the best model |

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-green)

---

## 📬 Submission

Built as part of the **Tredence Studio – AI Agents Engineering Internship 2025** case study.
