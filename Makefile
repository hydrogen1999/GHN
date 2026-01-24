# =============================================================================
# Makefile for Graph Hölder Networks (GHN)
# ICML 2026 Submission - Complete Experiments
# =============================================================================
#
# Paper Experiments Mapping:
#   Table 1  → make table1      (Clean accuracy + ACR)
#   Figure 1 → make figure1     (Certified accuracy curves)
#   Figure 2 → make figure2     (Margin-radius scaling)
#   Table 2  → make table2      (PGD attacks)
#   Table 3  → make table3      (Nettack + Metattack)
#   Table 4  → make table4      (Bernoulli edge deletion)
#   Figure 3 → make figure3     (NSR analysis)
#   Table 5  → make table5      (Deep networks + MAD)
#   Table 6  → make table6      (Ablation α)
#   Table 7  → make table7      (Ablation depth)
#   Table 8  → make table8      (Spectral norm ablation)
#   Table 9  → make table9      (ogbn-arxiv scalability)
#
# Quick Start:
#   make install    → Install PyTorch Geometric
#   make test       → Verify installation
#   make all-exp    → Run ALL paper experiments
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

# Default settings
MODEL := ghn

# =============================================================================
# Installation
# =============================================================================

.PHONY: install
install:
	@echo "Installing PyTorch and PyTorch Geometric..."
	$(PIP) install torch torchvision --break-system-packages
	$(PIP) install torch-geometric --break-system-packages
	$(PIP) install torch-scatter torch-sparse --break-system-packages
	$(PIP) install numpy scipy tqdm matplotlib seaborn ogb --break-system-packages
	@echo "✓ Installation complete!"

.PHONY: install-cpu
install-cpu:
	@echo "Installing CPU-only version..."
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cpu --break-system-packages
	$(PIP) install torch-geometric torch-scatter torch-sparse --break-system-packages
	$(PIP) install numpy scipy tqdm matplotlib seaborn ogb --break-system-packages

# =============================================================================
# Quick Tests
# =============================================================================

.PHONY: test
test: test-models test-data
	@echo "✓ All tests passed!"

.PHONY: test-models
test-models:
	@echo "Testing model implementations..."
	@$(PYTHON) -c "\
import sys; sys.path.insert(0, '.'); \
import torch; \
from models import get_model, MODEL_REGISTRY; \
print(f'Available models: {list(MODEL_REGISTRY.keys())}'); \
x = torch.randn(100, 16); \
adj = torch.eye(100); \
passed = 0; \
for name in MODEL_REGISTRY: \
    try: \
        model = get_model(name, in_features=16, out_features=7); \
        out = model(x, adj); \
        assert out.shape == (100, 7); \
        print(f'  ✓ {name}'); passed += 1; \
    except Exception as e: \
        print(f'  ✗ {name}: {e}'); \
print(f'\\nPassed: {passed}/{len(MODEL_REGISTRY)}')"

.PHONY: test-data
test-data:
	@echo "Testing data loaders..."
	@$(PYTHON) -c "\
import sys; sys.path.insert(0, '.'); \
from data.datasets import load_dataset, print_dataset_info; \
for name in ['cora', 'citeseer', 'synthetic']: \
    data = load_dataset(name); \
    print(f'✓ {name}: {data.num_nodes} nodes, {data.num_classes} classes')"

# =============================================================================
# Training
# =============================================================================

.PHONY: train
train:
	@echo "Training $(MODEL) on Cora..."
	@$(PYTHON) -c "\
import sys; sys.path.insert(0, '.'); \
from models import get_model; \
from data.datasets import load_dataset, print_dataset_info; \
from utils.training import train_and_evaluate, set_seed; \
from configs.default import get_model_config, get_training_config; \
import torch; \
device = torch.device('cuda:$(GPU)' if torch.cuda.is_available() else 'cpu'); \
set_seed(42); \
data = load_dataset('cora'); \
print_dataset_info(data); \
config = get_model_config('$(MODEL)'); \
model = get_model('$(MODEL)', in_features=data.num_features, out_features=data.num_classes, **config); \
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}'); \
results = train_and_evaluate(model, data, get_training_config(), device); \
print(f'Test Accuracy: {results[\"test_accuracy\"]*100:.2f}%')"

# =============================================================================
# TABLE 1: Clean Accuracy and ACR (Section 1.2)
# =============================================================================

.PHONY: table1
table1:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "TABLE 1: Clean Accuracy and Average Certified Radius"
	@echo "═══════════════════════════════════════════════════════════════"
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment table1 \
		--datasets $(DATASETS) \
		--seeds $(SEEDS) \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR)

.PHONY: table1-quick
table1-quick:
	@echo "TABLE 1 (Quick: 3 seeds, 2 datasets)..."
	@mkdir -p $(RESULTS_DIR)/quick
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment table1 \
		--datasets cora citeseer \
		--seeds 0 1 2 \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR)/quick

# =============================================================================
# FIGURE 1: Certified Accuracy Curves (Section 1.2)
# =============================================================================

.PHONY: figure1
figure1:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "FIGURE 1: Certified Accuracy vs Perturbation Radius"
	@echo "═══════════════════════════════════════════════════════════════"
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment figure1 \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR)

# =============================================================================
# FIGURE 2: Margin-Radius Scaling (Section 1.3)
# =============================================================================

.PHONY: figure2
figure2:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "FIGURE 2: Margin-Radius Scaling Analysis"
	@echo "═══════════════════════════════════════════════════════════════"
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment figure2 \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR)

# =============================================================================
# TABLE 2: PGD Attacks (Section 1.4)
# =============================================================================

.PHONY: table2
table2:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "TABLE 2: Accuracy under PGD Attacks"
	@echo "═══════════════════════════════════════════════════════════════"
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment table2 \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR)

# =============================================================================
# TABLE 3: Structural Attacks - Nettack + Metattack (Section 1.5)
# =============================================================================

.PHONY: table3
table3:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "TABLE 3: Structural Attack Robustness"
	@echo "═══════════════════════════════════════════════════════════════"
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment table3 \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR)

# =============================================================================
# TABLE 4: Bernoulli Edge Deletion (Section 1.5)
# =============================================================================

.PHONY: table4
table4:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "TABLE 4: Bernoulli Edge Deletion (Theorem 3.5)"
	@echo "═══════════════════════════════════════════════════════════════"
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment table4 \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR)

# =============================================================================
# FIGURE 3: NSR Analysis (Section 1.6)
# =============================================================================

.PHONY: figure3
figure3:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "FIGURE 3: Noise-to-Signal Ratio Analysis"
	@echo "═══════════════════════════════════════════════════════════════"
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment figure3 \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR)

# =============================================================================
# TABLE 5: Deep Networks + MAD (Section 1.7)
# =============================================================================

.PHONY: table5
table5:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "TABLE 5: Deep Network Trainability (Oversmoothing)"
	@echo "═══════════════════════════════════════════════════════════════"
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment table5 \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR)

# =============================================================================
# TABLE 6: Ablation α (Section 1.8)
# =============================================================================

.PHONY: table6
table6:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "TABLE 6: Ablation - Effect of Hölder Exponent α"
	@echo "═══════════════════════════════════════════════════════════════"
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment table6 \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR)

# =============================================================================
# TABLE 7: Ablation Depth (Section 1.8)
# =============================================================================

.PHONY: table7
table7:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "TABLE 7: Ablation - Effect of Network Depth L"
	@echo "═══════════════════════════════════════════════════════════════"
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment table7 \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR)

# =============================================================================
# TABLE 8: Spectral Normalization Ablation (Section 1.8)
# =============================================================================

.PHONY: table8
table8:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "TABLE 8: Spectral Normalization Ablation"
	@echo "═══════════════════════════════════════════════════════════════"
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment table8 \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR)

# =============================================================================
# TABLE 9: Scalability ogbn-arxiv (Section 1.9)
# =============================================================================

.PHONY: table9
table9:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "TABLE 9: Scalability (ogbn-arxiv)"
	@echo "═══════════════════════════════════════════════════════════════"
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment table9 \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR)

# =============================================================================
# Run ALL Experiments
# =============================================================================

.PHONY: all-exp
all-exp:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Running ALL Paper Experiments"
	@echo "═══════════════════════════════════════════════════════════════"
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment all \
		--datasets $(DATASETS) \
		--seeds $(SEEDS) \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR)

.PHONY: all-exp-quick
all-exp-quick:
	@echo "Running ALL experiments (quick version)..."
	@mkdir -p $(RESULTS_DIR)/quick
	$(PYTHON) $(EXP_DIR)/main.py \
		--experiment all \
		--datasets cora \
		--seeds 0 1 2 \
		--gpu $(GPU) \
		--output_dir $(RESULTS_DIR)/quick

# =============================================================================
# Ablation Studies (Combined)
# =============================================================================

.PHONY: ablation-all
ablation-all: table6 table7 table8
	@echo "✓ All ablation studies completed!"

# =============================================================================
# Visualization
# =============================================================================

.PHONY: plot-all
plot-all:
	@echo "Generating all figures..."
	@cd $(RESULTS_DIR) && \
	for script in plot_figure*.py; do \
		if [ -f "$$script" ]; then \
			echo "Running $$script..."; \
			$(PYTHON) "$$script"; \
		fi; \
	done
	@echo "✓ Plots saved to $(RESULTS_DIR)/"

# =============================================================================
# Utilities
# =============================================================================

.PHONY: clean
clean:
	rm -rf $(RESULTS_DIR)/*
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf *.pyc
	@echo "✓ Cleaned"

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

# =============================================================================
# Help
# =============================================================================

.PHONY: help
help:
	@echo ""
	@echo "╔═══════════════════════════════════════════════════════════════╗"
	@echo "║     Graph Hölder Networks - ICML 2026 Experiments             ║"
	@echo "╚═══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "SETUP:"
	@echo "  make install        Install PyTorch Geometric"
	@echo "  make test           Verify all models work"
	@echo ""
	@echo "PAPER EXPERIMENTS:"
	@echo "  make table1         Table 1: Clean accuracy + ACR"
	@echo "  make figure1        Figure 1: Certified accuracy curves"
	@echo "  make figure2        Figure 2: Margin-radius scaling"
	@echo "  make table2         Table 2: PGD attacks"
	@echo "  make table3         Table 3: Nettack + Metattack"
	@echo "  make table4         Table 4: Bernoulli edge deletion"
	@echo "  make figure3        Figure 3: NSR analysis"
	@echo "  make table5         Table 5: Deep networks + MAD"
	@echo "  make table6         Table 6: Ablation α"
	@echo "  make table7         Table 7: Ablation depth"
	@echo "  make table8         Table 8: Spectral norm ablation"
	@echo "  make table9         Table 9: ogbn-arxiv scalability"
	@echo ""
	@echo "COMBINED:"
	@echo "  make all-exp        Run ALL experiments (full)"
	@echo "  make all-exp-quick  Run ALL experiments (quick)"
	@echo "  make ablation-all   Run all ablation studies"
	@echo "  make plot-all       Generate all figures"
	@echo ""
	@echo "UTILITIES:"
	@echo "  make train          Train single model"
	@echo "  make download-data  Pre-download datasets"
	@echo "  make clean          Remove generated files"
	@echo ""
	@echo "OPTIONS:"
	@echo "  GPU=0               GPU device (default: 0)"
	@echo "  MODEL=ghn           Model for training"
	@echo ""
