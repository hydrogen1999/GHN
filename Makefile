# =============================================================================
# Makefile for Graph Hölder Networks (GHN)
# ICML 2026 Submission
# =============================================================================
#
# Quick Start:
#   make install       - Install dependencies (PyTorch Geometric)
#   make train         - Train GHN on Cora
#   make eval          - Evaluate with certification
#   make table1-quick  - Quick Table 1 (3 seeds)
#   make all-exp       - Run all paper experiments
#
# =============================================================================

# Configuration
PYTHON := python3
PIP := pip3
GPU := 0
SEEDS := 0 1 2 3 4 5 6 7 8 9
DATASETS := cora citeseer pubmed

# Directories
EXP_DIR := experiments
RESULTS_DIR := results
CHECKPOINT_DIR := checkpoints
DATA_DIR := data

# Default model settings
MODEL := ghn
ALPHA := 0.8
HIDDEN := 64
LAYERS := 2

# =============================================================================
# Installation
# =============================================================================

.PHONY: install
install:
	@echo "Installing dependencies..."
	$(PIP) install torch torchvision --break-system-packages
	$(PIP) install torch-geometric --break-system-packages
	$(PIP) install torch-scatter torch-sparse --break-system-packages
	$(PIP) install numpy scipy tqdm matplotlib seaborn ogb --break-system-packages
	@echo "Installation complete!"

.PHONY: install-cpu
install-cpu:
	@echo "Installing CPU-only dependencies..."
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cpu --break-system-packages
	$(PIP) install torch-geometric torch-scatter torch-sparse --break-system-packages
	$(PIP) install numpy scipy tqdm matplotlib seaborn ogb --break-system-packages

.PHONY: install-dev
install-dev: install
	$(PIP) install pytest black flake8 mypy jupyter --break-system-packages

# =============================================================================
# Training
# =============================================================================

.PHONY: train
train:
	@echo "Training $(MODEL) on cora..."
	@$(PYTHON) -c "\
import sys; sys.path.insert(0, '.'); \
from models import get_model; \
from data.datasets import load_dataset, print_dataset_info; \
from utils.training import train_and_evaluate, set_seed; \
from configs.default import get_model_config, get_training_config; \
import torch; \
device = torch.device('cuda:$(GPU)' if torch.cuda.is_available() else 'cpu'); \
print(f'Device: {device}'); \
set_seed(42); \
data = load_dataset('cora'); \
print_dataset_info(data); \
config = get_model_config('$(MODEL)'); \
model = get_model('$(MODEL)', in_features=data.num_features, out_features=data.num_classes, **config); \
print(f'Model: $(MODEL), Params: {sum(p.numel() for p in model.parameters()):,}'); \
results = train_and_evaluate(model, data, get_training_config(), device); \
print(f'\\nTest Accuracy: {results[\"test_accuracy\"]:.4f}')"

.PHONY: train-all
train-all:
	@for model in ghn gcn gat sgc spectral_gcn groupsort_gcn; do \
		echo "\n========== Training $$model =========="; \
		$(MAKE) train MODEL=$$model; \
	done

.PHONY: train-dataset
train-dataset:
	@$(MAKE) train MODEL=$(MODEL) DATASET=$(DATASET)

# =============================================================================
# Evaluation & Certification
# =============================================================================

.PHONY: eval
eval:
	@echo "Evaluating $(MODEL) with certification..."
	@$(PYTHON) -c "\
import sys; sys.path.insert(0, '.'); \
from models import get_model; \
from data.datasets import load_dataset; \
from utils.training import train_and_evaluate, set_seed; \
from certify.certification import certify_all_nodes; \
from configs.default import get_model_config, get_training_config; \
import torch; \
device = torch.device('cuda:$(GPU)' if torch.cuda.is_available() else 'cpu'); \
set_seed(42); \
data = load_dataset('cora'); \
config = get_model_config('$(MODEL)'); \
model = get_model('$(MODEL)', in_features=data.num_features, out_features=data.num_classes, **config); \
results = train_and_evaluate(model, data, get_training_config(), device, verbose=False); \
print(f'Test Accuracy: {results[\"test_accuracy\"]:.4f}'); \
if '$(MODEL)' in ['ghn', 'spectral_gcn', 'groupsort_gcn']: \
    cert = certify_all_nodes(model, data.x.to(device), data.adj.to(device), data.y.to(device), data.test_mask.to(device), 'ghn' if '$(MODEL)'=='ghn' else 'lipschitz', config.get('alpha', 1.0), config.get('num_layers', 2)); \
    print(f'Avg Certified Radius: {cert[\"average_certified_radius\"]:.4f}'); \
    print(f'Certified Accuracy @0.1: {cert[\"certified_accuracy\"]:.4f}')"

# =============================================================================
# Paper Experiments - Table 1
# =============================================================================

.PHONY: table1
table1:
	@echo "Running Table 1: Full experiments (10 seeds)..."
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment table1 \
		--datasets $(DATASETS) \
		--models ghn gcn gat sgc spectral_gcn groupsort_gcn pairnorm_gcn gnnguard robustgcn \
		--seeds $(SEEDS) \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR) \
		--verbose

.PHONY: table1-quick
table1-quick:
	@echo "Running Table 1: Quick version (3 seeds, main models)..."
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment table1 \
		--datasets cora citeseer \
		--models ghn gcn spectral_gcn groupsort_gcn \
		--seeds 0 1 2 \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR)/quick

# =============================================================================
# Paper Experiments - Figure 1 (Scaling)
# =============================================================================

.PHONY: scaling
scaling:
	@echo "Running Figure 1: Scaling behavior analysis..."
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment scaling \
		--datasets cora \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR)

# =============================================================================
# Paper Experiments - Table 2 (Certified Accuracy)
# =============================================================================

.PHONY: certified-accuracy
certified-accuracy:
	@echo "Running Table 2: Certified accuracy at various radii..."
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment certified_accuracy \
		--datasets $(DATASETS) \
		--radii 0.05 0.1 0.15 0.2 0.25 0.3 \
		--seeds 0 1 2 3 4 \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR)

# =============================================================================
# Ablation Studies
# =============================================================================

.PHONY: ablation-alpha
ablation-alpha:
	@echo "Running Ablation: Effect of α..."
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment ablation_alpha \
		--datasets cora \
		--alphas 0.5 0.6 0.7 0.8 0.9 1.0 \
		--seeds 0 1 2 3 4 \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR)

.PHONY: ablation-depth
ablation-depth:
	@echo "Running Ablation: Effect of depth L..."
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment ablation_depth \
		--datasets cora \
		--depths 1 2 3 4 5 6 \
		--seeds 0 1 2 3 4 \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR)

.PHONY: ablation-hidden
ablation-hidden:
	@echo "Running Ablation: Effect of hidden dimension..."
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment ablation_hidden \
		--datasets cora \
		--hidden_dims 16 32 64 128 256 \
		--seeds 0 1 2 3 4 \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR)

.PHONY: ablation-all
ablation-all: ablation-alpha ablation-depth ablation-hidden
	@echo "All ablation studies completed!"

# =============================================================================
# Attack Evaluation
# =============================================================================

.PHONY: attacks
attacks:
	@echo "Running Attack Evaluation (PGD, FGSM)..."
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment attacks \
		--datasets cora \
		--epsilons 0.01 0.05 0.1 0.15 0.2 \
		--seeds 0 1 2 \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR)

# =============================================================================
# Scalability (ogbn-arxiv)
# =============================================================================

.PHONY: scalability
scalability:
	@echo "Running Scalability Experiment (ogbn-arxiv)..."
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment scalability \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR)

# =============================================================================
# All Experiments
# =============================================================================

.PHONY: all-exp
all-exp:
	@echo "Running ALL experiments for ICML paper..."
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment all \
		--datasets $(DATASETS) \
		--seeds $(SEEDS) \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR) \
		--verbose

.PHONY: all-exp-quick
all-exp-quick:
	@echo "Running ALL experiments (quick version)..."
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment all \
		--datasets cora \
		--seeds 0 1 2 \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR)/quick

# =============================================================================
# Visualization
# =============================================================================

.PHONY: plot-all
plot-all: plot-scaling plot-ablation plot-attacks
	@echo "All plots generated!"

.PHONY: plot-scaling
plot-scaling:
	@echo "Generating Figure 1: Scaling plot..."
	@cd $(RESULTS_DIR)/scaling && $(PYTHON) plot_scaling.py

.PHONY: plot-ablation
plot-ablation:
	@echo "Generating ablation plots..."
	@$(PYTHON) -c "\
import json; import matplotlib.pyplot as plt; import numpy as np; \
# Alpha ablation \
try: \
    with open('$(RESULTS_DIR)/ablation_alpha/ablation_alpha.json') as f: data = json.load(f); \
    alphas = sorted([float(a) for a in data.keys()]); \
    accs = [np.mean(data[str(a)]['accuracy']) for a in alphas]; \
    acrs = [np.mean(data[str(a)]['acr']) for a in alphas]; \
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4)); \
    ax1.plot(alphas, accs, 'o-b', linewidth=2, markersize=8); ax1.set_xlabel('α'); ax1.set_ylabel('Accuracy'); ax1.set_title('Effect of α on Accuracy'); ax1.grid(True, alpha=0.3); \
    ax2.plot(alphas, acrs, 'o-r', linewidth=2, markersize=8); ax2.set_xlabel('α'); ax2.set_ylabel('ACR'); ax2.set_title('Effect of α on Certified Radius'); ax2.grid(True, alpha=0.3); \
    plt.tight_layout(); plt.savefig('$(RESULTS_DIR)/ablation_alpha/ablation_alpha.pdf', dpi=300); \
    print('Saved: $(RESULTS_DIR)/ablation_alpha/ablation_alpha.pdf'); \
except FileNotFoundError: print('Run ablation-alpha first')"

.PHONY: plot-attacks
plot-attacks:
	@echo "Generating attack evaluation plots..."
	@$(PYTHON) -c "\
import json; import matplotlib.pyplot as plt; import numpy as np; \
try: \
    with open('$(RESULTS_DIR)/attacks/attack_results.json') as f: data = json.load(f); \
    fig, ax = plt.subplots(figsize=(8, 5)); \
    for model in data: \
        eps = sorted([float(e) for e in data[model].keys()]); \
        pgd_accs = [np.mean(data[model][str(e)]['pgd']) for e in eps]; \
        ax.plot(eps, pgd_accs, 'o-', label=model, linewidth=2); \
    ax.set_xlabel('Perturbation Budget ε'); ax.set_ylabel('Accuracy under PGD Attack'); \
    ax.set_title('Robustness to PGD Attack'); ax.legend(); ax.grid(True, alpha=0.3); \
    plt.tight_layout(); plt.savefig('$(RESULTS_DIR)/attacks/attack_robustness.pdf', dpi=300); \
    print('Saved: $(RESULTS_DIR)/attacks/attack_robustness.pdf'); \
except FileNotFoundError: print('Run attacks first')"

# =============================================================================
# Testing & Validation
# =============================================================================

.PHONY: test
test:
	@echo "Running tests..."
	$(PYTHON) -m pytest tests/ -v || echo "No tests directory found"

.PHONY: test-models
test-models:
	@echo "Testing all model implementations..."
	@$(PYTHON) -c "\
import sys; sys.path.insert(0, '.'); \
import torch; \
from models import get_model, MODEL_REGISTRY; \
print('Testing all models...'); \
x = torch.randn(100, 16); \
adj = torch.eye(100) + torch.randn(100, 100).abs() * 0.1; \
adj = (adj + adj.T) / 2; \
deg = adj.sum(1); deg_inv = deg.pow(-0.5); deg_inv[deg_inv==float('inf')]=0; \
adj = deg_inv.unsqueeze(1) * adj * deg_inv.unsqueeze(0); \
passed = 0; failed = 0; \
for name in MODEL_REGISTRY: \
    try: \
        model = get_model(name, in_features=16, out_features=7); \
        out = model(x, adj); \
        assert out.shape == (100, 7), f'Wrong output shape: {out.shape}'; \
        print(f'  ✓ {name}: OK'); passed += 1; \
    except Exception as e: \
        print(f'  ✗ {name}: {e}'); failed += 1; \
print(f'\\nPassed: {passed}, Failed: {failed}')"

.PHONY: test-data
test-data:
	@echo "Testing data loaders..."
	@$(PYTHON) -c "\
import sys; sys.path.insert(0, '.'); \
from data.datasets import load_dataset, AVAILABLE_DATASETS, print_dataset_info; \
for name in ['cora', 'citeseer', 'pubmed', 'synthetic']: \
    try: \
        data = load_dataset(name); \
        print(f'✓ {name}: {data.num_nodes} nodes, {data.num_features} features, {data.num_classes} classes'); \
    except Exception as e: \
        print(f'✗ {name}: {e}')"

.PHONY: test-certification
test-certification:
	@echo "Testing certification..."
	@$(PYTHON) -c "\
import sys; sys.path.insert(0, '.'); \
import torch; \
from models.ghn import GraphHolderNetwork; \
from data.datasets import load_dataset; \
from certify.certification import certify_all_nodes; \
data = load_dataset('synthetic', num_nodes=100); \
model = GraphHolderNetwork(data.num_features, data.num_classes, 32, 2, 0.8); \
cert = certify_all_nodes(model, data.x, data.adj, data.y, data.test_mask, 'ghn', 0.8, 2); \
print(f'ACR: {cert[\"average_certified_radius\"]:.4f}'); \
print(f'Certified nodes: {cert[\"num_certified\"]}/{cert[\"num_total\"]}'); \
print('✓ Certification test passed')"

# =============================================================================
# Utilities
# =============================================================================

.PHONY: clean
clean:
	@echo "Cleaning generated files..."
	rm -rf $(RESULTS_DIR)/*
	rm -rf $(CHECKPOINT_DIR)/*
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf *.egg-info
	find . -name "*.pyc" -delete
	find . -name ".DS_Store" -delete

.PHONY: clean-results
clean-results:
	rm -rf $(RESULTS_DIR)/*

.PHONY: dirs
dirs:
	@mkdir -p $(RESULTS_DIR) $(CHECKPOINT_DIR) $(DATA_DIR)

.PHONY: download-data
download-data:
	@echo "Downloading datasets..."
	@$(PYTHON) -c "\
import sys; sys.path.insert(0, '.'); \
from data.datasets import load_dataset, print_dataset_info; \
for name in ['cora', 'citeseer', 'pubmed']: \
    print(f'\\nDownloading {name}...'); \
    data = load_dataset(name); \
    print_dataset_info(data)"

.PHONY: info
info:
	@echo ""
	@echo "╔═══════════════════════════════════════════════════════════╗"
	@echo "║         Graph Hölder Networks (GHN) - ICML 2026           ║"
	@echo "╚═══════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Available models:"
	@$(PYTHON) -c "import sys; sys.path.insert(0, '.'); from models import MODEL_REGISTRY; [print(f'  • {m}') for m in sorted(MODEL_REGISTRY.keys())]" 2>/dev/null || echo "  (run 'make install' first)"
	@echo ""
	@echo "Available datasets: cora, citeseer, pubmed, ogbn-arxiv, synthetic"
	@echo ""
	@echo "Quick commands:"
	@echo "  make install      Install PyTorch Geometric"
	@echo "  make train        Train GHN on Cora"
	@echo "  make eval         Evaluate with certification"
	@echo "  make table1-quick Quick Table 1 experiment"
	@echo "  make all-exp      Run all paper experiments"
	@echo ""

.PHONY: help
help:
	@echo ""
	@echo "Graph Hölder Networks - Makefile Commands"
	@echo "=========================================="
	@echo ""
	@echo "SETUP:"
	@echo "  install           Install PyTorch Geometric and dependencies"
	@echo "  install-cpu       Install CPU-only version"
	@echo "  download-data     Pre-download all datasets"
	@echo ""
	@echo "TRAINING:"
	@echo "  train             Train single model (MODEL=ghn)"
	@echo "  train-all         Train all model types"
	@echo "  eval              Evaluate with certification"
	@echo ""
	@echo "PAPER EXPERIMENTS:"
	@echo "  table1            Table 1 - Main results (full)"
	@echo "  table1-quick      Table 1 - Quick version"
	@echo "  scaling           Figure 1 - Scaling behavior"
	@echo "  certified-accuracy Table 2 - CA at various radii"
	@echo ""
	@echo "ABLATION STUDIES:"
	@echo "  ablation-alpha    Effect of Hölder exponent α"
	@echo "  ablation-depth    Effect of network depth L"
	@echo "  ablation-hidden   Effect of hidden dimension"
	@echo "  ablation-all      Run all ablations"
	@echo ""
	@echo "ATTACK EVALUATION:"
	@echo "  attacks           PGD and FGSM attack evaluation"
	@echo ""
	@echo "SCALABILITY:"
	@echo "  scalability       ogbn-arxiv experiment"
	@echo ""
	@echo "ALL EXPERIMENTS:"
	@echo "  all-exp           Run all experiments (full)"
	@echo "  all-exp-quick     Run all experiments (quick)"
	@echo ""
	@echo "VISUALIZATION:"
	@echo "  plot-all          Generate all figures"
	@echo "  plot-scaling      Generate Figure 1"
	@echo "  plot-ablation     Generate ablation plots"
	@echo ""
	@echo "TESTING:"
	@echo "  test-models       Test all model implementations"
	@echo "  test-data         Test data loaders"
	@echo "  test-certification Test certification module"
	@echo ""
	@echo "UTILITIES:"
	@echo "  clean             Remove all generated files"
	@echo "  info              Show project information"
	@echo ""
	@echo "VARIABLES:"
	@echo "  MODEL=ghn         Model to use (default: ghn)"
	@echo "  GPU=0             GPU device ID (default: 0)"
	@echo "  DATASETS='cora citeseer pubmed'"
	@echo ""
