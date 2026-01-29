# Graph Hölder Networks (GHN)

[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue.svg)](https://icml.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)


Official implementation for ICML 2026 submission: *Graph Hölder Networks*.

---

## 🎯 Overview

Graph Neural Networks (GNNs) are vulnerable to adversarial perturbations on both node features and graph topology. Existing certified defenses rely on **1-Lipschitz constraints**, which suffer from limited expressivity and gradient pathologies in deep architectures.

**Graph Hölder Networks (GHN)** relax Lipschitz continuity to **α-Hölder continuity** (α < 1), achieving:

| Property | Lipschitz Networks | **GHN (Ours)** |
|----------|-------------------|----------------|
| Certified Radius Scaling | R ∝ γ (linear) | **R ∝ γ^{1/α^L} (super-linear)** |
| Expressivity | Limited (cannot approximate \|x\|) | **Universal Approximation** |
| Deep Network Training | Gradient vanishing/exploding | **Depth-uniform boundedness** |
| Accuracy vs. Robustness | Trade-off | **No trade-off** |

### Key results (Table 1)

| Method | Cora Acc | Cora ACR | Improvement |
|--------|----------|----------|-------------|
| GCN | 81.5% | — | — |
| GroupSort-GCN | 76.2% | 0.051 | baseline |
| GNNCert | 79.1% | 0.063 | 1.0× |
| **GHN (Ours)** | **81.2%** | **0.147** | **2.3×** |

GHN achieves **2.3× larger certified radius** than the strongest baseline while matching standard GCN accuracy.

---

## 📦 Installation

### Requirements
- Python ≥ 3.8
- PyTorch ≥ 2.0
- PyTorch Geometric ≥ 2.4
- CUDA (optional, for GPU acceleration)

### Quick install

```bash
# Clone repository
git clone https://github.com/anonymous/ghn.git
cd ghn

# Install dependencies (GPU)
make install

# Or install dependencies (CPU only)
make install-cpu

# Verify installation
make test
```

### Manual installation

```bash
pip install torch torchvision
pip install torch-geometric torch-scatter torch-sparse
pip install numpy scipy tqdm matplotlib seaborn ogb
```

---

## 🚀 Quick start

### Using Makefile

```bash
# Train and evaluate GHN on Cora
make train MODEL=ghn

# Run quick experiments
make table1-quick
```

---

## 📁 Project structure

```
ghn/
├── models/
│   ├── activations.py      # α-RePU activation (Definition 3.2)
│   ├── ghn.py              # Graph Hölder Network (Definition 3.3-3.4)
│   ├── baselines.py        # GCN, GAT, SGC
│   ├── lipschitz.py        # Spectral-GCN, GroupSort-GCN, PairNorm-GCN
│   ├── certified.py        # Randomized Smoothing, GNNCert
│   └── empirical.py        # GNNGuard, RobustGCN
├── data/
│   └── datasets.py         # Planetoid, OGB loaders
├── certify/
│   └── certification.py    # Certified radius computation (Corollary 3.6)
├── attacks/
│   └── __init__.py         # PGD, FGSM, Nettack, Metattack
├── utils/
│   ├── training.py         # Training loop, early stopping
│   ├── metrics.py          # Accuracy, ACR, certified accuracy
│   └── analysis.py         # NSR, MAD analysis
├── configs/
│   └── default.py          # Hyperparameter configurations
├── experiments/
│   └── main.py             # Paper experiment runner
├── Makefile                # Build and experiment commands
└── requirements.txt
```

---

## 🔬 Models

### Available models

| Model | Type | Certificate | Reference |
|-------|------|-------------|-----------|
| `ghn` | **GHN** | ✅ Hölder | **Ours** |
| `gcn` | Standard | ❌ | Kipf & Welling, 2016 |
| `gat` | Standard | ❌ | Veličković et al., 2017 |
| `sgc` | Standard | ❌ | Wu et al., 2019 |
| `spectral_gcn` | Lipschitz | ✅ Lipschitz | Miyato et al., 2018 |
| `groupsort_gcn` | Lipschitz | ✅ Lipschitz | Anil et al., 2019 |
| `pairnorm_gcn` | Lipschitz | ✅ Lipschitz | Zhao & Akoglu, 2019 |
| `randomized_smoothing` | Probabilistic | ✅ Probabilistic | Cohen et al., 2019 |
| `gnncert` | Partitioning | ✅ Deterministic | Xia et al., 2024 |
| `gnnguard` | Empirical | ❌ | Zhang & Zitnik, 2020 |
| `robustgcn` | Empirical | ❌ | Zhu et al., 2019 |

### Model instantiation

```python
from models import get_model, MODEL_REGISTRY

# List available models
print(list(MODEL_REGISTRY.keys()))

# Create model
model = get_model(
    'ghn',
    in_features=1433,
    out_features=7,
    hidden_features=64,
    num_layers=2,
    alpha=0.8,
    c=1e-4,
    dropout=0.5,
)
```

---

## 📐 Mathematical background

### α-RePU activation (Definition 3.2)


$$\sigma_{\alpha,c}(x):=
\begin{cases}
(x + c)^\alpha, & \text{if } x \ge 0,\\[4pt]
c^\alpha,       & \text{if } x < 0.
\end{cases}
$$

**Properties:**
- **α-Hölder continuous** with seminorm $[\sigma_{\alpha,c}]_\alpha=1$.
- **Sub-linear response**: dampens large perturbations
- Reduces to ReLU when $\alpha\rightarrow 1, c\rightarrow 0$.

### Graph Hölder Layer (Definition 3.3)

$$
  \mathbf{H}^{(l+1)}=
  \Sigma_{\alpha,c}\bigl(
    \hat{\mathbf{A}} \mathbf{H}^{(l)} \mathbf{W}^{(l)}
    + \mathbf{1}_n (\mathbf{b}^{(l)})^\top
  \bigr),
$$

where $\hat{\mathbf{A}} =\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}$ is the symmetric normalized adjacency.

### Certified Radius (Corollary 3.6)

For a node with classification margin γ:

| Network Type | Certified Radius |
|--------------|------------------|
| Lipschitz (α=1) | R = γ / (2K) |
| **Hölder (α<1)** | **R = (γ / 2C_net)^{1/α^L}** |

Since 1/α^L > 1, the Hölder radius scales **super-linearly** with margin, providing significantly larger certificates for high-confidence predictions.

### Network Hölder Constant

$$C_{\mathrm{net}} = \prod_{l=0}^{L-1} ||W^{(l)}||_{\infty}^{\alpha^{L-l}}.$$

The $\ell_\infty$ formulation ensures **dimension-free** certificates that don't degrade with graph size.

---

## 📊 Reproducing Paper Experiments

### Experiment-to-Paper Mapping

| Command | Paper Reference | Description |
|---------|-----------------|-------------|
| `make table1` | Table 1 | Clean accuracy + ACR on citation networks |
| `make figure1` | Figure 2 | Certified accuracy vs. perturbation radius |
| `make figure2` | Figure 4 (Appendix) | Margin-radius scaling (log-log) |
| `make table3` | Table 2 | Nettack + Metattack structural attacks |
| `make table4` | Table 3 | Bernoulli edge deletion |
| `make figure3` | Figure 3 | NSR analysis (geometric compression) |
| `make table5` | Table 4 (Appendix) | Deep network trainability + MAD |
| `make table6` | Table 6 (Appendix) | Ablation: effect of α |
| `make table7` | Table 7 (Appendix) | Ablation: effect of depth L |
| `make table8` | Table 8 (Appendix) | Spectral normalization ablation |
| `make table9` | Table 5 (Appendix) | Scalability on ogbn-arxiv |

### Running Experiments

```bash
# Quick version (3 seeds, ~10 min)
make table1-quick

# Full version (10 seeds, ~2 hours)
make table1

# Run ALL experiments
make all-exp        # Full (~6 hours)
make all-exp-quick  # Quick (~30 min)

# Generate figures
make plot-all
```

### Using Python script

```bash
# Single experiment
python experiments/main.py --experiment table1 \
    --datasets cora citeseer pubmed \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --gpu 0

# Ablation study
python experiments/main.py --experiment table6 \
    --seeds 0 1 2 3 4 \
    --gpu 0

# All experiments
python experiments/main.py --experiment all --gpu 0
```

---

## ⚙️ Configuration

### Default hyperparameters

```python
# GHN Model (configs/default.py)
GHN_CONFIG = {
    'hidden_features': 64,
    'num_layers': 2,
    'alpha': 0.8,           # Hölder exponent
    'c': 1e-4,              # α-RePU smoothing constant
    'dropout': 0.5,
    'use_batch_norm': False,
}

# Training
TRAINING_CONFIG = {
    'optimizer': 'adam',
    'lr': 0.01,
    'weight_decay': 5e-4,
    'epochs': 200,
    'patience': 20,         # Early stopping
}
```

### Ablation guidance

| Parameter | Recommended Range | Notes |
|-----------|-------------------|-------|
| α | 0.7 - 0.9 | α=0.8 balances accuracy and robustness |
| L (depth) | 2 - 4 | Deeper → larger α^L → larger certificates |
| c | 1e-6 - 1e-2 | Minimal sensitivity |
| hidden_dim | 64 - 256 | Standard GNN guidance |

---

## 📈 Expected results

### Table 1: Main results

| Method | Cora Acc | Cora ACR | Citeseer Acc | Citeseer ACR | PubMed Acc | PubMed ACR |
|--------|----------|----------|--------------|--------------|------------|------------|
| GCN | 81.5±0.5 | — | 70.3±0.6 | — | 79.0±0.4 | — |
| GAT | 83.0±0.7 | — | 72.5±0.5 | — | 79.5±0.3 | — |
| Spectral-GCN | 78.4±0.6 | 0.042 | 67.8±0.8 | 0.035 | 76.1±0.5 | 0.038 |
| GroupSort-GCN | 76.2±0.8 | 0.051 | 66.4±0.9 | 0.043 | 74.8±0.6 | 0.046 |
| GNNCert | 79.1±0.7 | 0.063 | 68.9±0.8 | 0.054 | 77.2±0.6 | 0.058 |
| **GHN (Ours)** | **81.2±0.5** | **0.147** | **70.8±0.6** | **0.118** | **79.2±0.4** | **0.132** |

### Ablation: Effect of α (Table 6)

| α | Clean Accuracy | ACR | Notes |
|---|----------------|-----|-------|
| 0.5 | 78.1% | 0.21 | Too aggressive compression |
| 0.6 | 79.4% | 0.19 | |
| 0.7 | 80.5% | 0.17 | |
| **0.8** | **81.2%** | **0.15** | **Recommended** |
| 0.9 | 81.4% | 0.08 | |
| 1.0 | 81.5% | 0.05 | Reduces to Lipschitz |

---

## 🔧 Advanced usage

### Custom certification

```python
from certify.certification import (
    compute_classification_margin,
    compute_holder_certified_radius,
    compute_network_holder_constant,
)

# Compute margin for a specific node
logits = model(x, adj)
margin = compute_classification_margin(logits[node_idx], true_label)

# Compute network Hölder constant
c_net = compute_network_holder_constant(model, alpha=0.8)

# Compute certified radius
alpha_net = 0.8 ** 2  # α^L for 2 layers
radius = compute_holder_certified_radius(margin, c_net, alpha_net)
```

### Adversarial evaluation

```python
from attacks import PGDAttack, Nettack, BernoulliEdgeDeletion

# PGD attack on features
attacker = PGDAttack(model, epsilon=0.1, num_steps=40)
x_adv = attacker.attack(x, adj, labels, test_mask)

# Nettack on structure
nettack = Nettack(model)
adj_adv = nettack.attack(x, adj, labels, target_node, budget=5)

# Random edge deletion
perturber = BernoulliEdgeDeletion(deletion_prob=0.1)
adj_noisy = perturber.perturb(adj, seed=42)
```

### NSR Analysis (Geometric Compression)

```python
from utils.analysis import compute_nsr, compare_nsr_models

# Compute NSR for GHN
nsr_result = compute_nsr(model, x, adj, noise_std=0.1, num_samples=100)
print(f"NSR ratio per layer: {nsr_result['nsr_ratio']}")
# Expected: ~0.8 for GHN (geometric decay)
# Expected: ~1.0 for GCN (constant)
```

---

## 📚 Citation

```bibtex
@inproceedings{anonymous2026ghn,
  title={Graph H{\"o}lder Networks},
  author={Anonymous},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- Datasets: [Planetoid](https://arxiv.org/abs/1603.08861), [Open Graph Benchmark](https://ogb.stanford.edu/)
- Baseline implementations adapted from original papers
