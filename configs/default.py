"""
Default Configuration for Graph Hölder Networks Experiments

This configuration follows the experimental setup in the paper:
- 2-layer architectures with 64 hidden units
- Adam optimizer with lr=0.01, weight_decay=5e-4
- 200 epochs with early stopping (patience=20)
- α=0.8, c=1e-4 for GHN
"""

# =============================================================================
# Model Configurations
# =============================================================================

GHN_CONFIG = {
    'hidden_features': 64,
    'num_layers': 2,
    'alpha': 0.8,
    'c': 1e-4,
    'dropout': 0.5,
    'use_batch_norm': False,
}

GCN_CONFIG = {
    'hidden_features': 64,
    'num_layers': 2,
    'dropout': 0.5,
}

GAT_CONFIG = {
    'hidden_features': 8,
    'num_heads': 8,
    'dropout': 0.6,
}

SGC_CONFIG = {
    'k_hops': 2,
}

SPECTRAL_GCN_CONFIG = {
    'hidden_features': 64,
    'num_layers': 2,
    'dropout': 0.5,
}

GROUPSORT_GCN_CONFIG = {
    'hidden_features': 64,
    'num_layers': 2,
    'group_size': 2,
    'dropout': 0.5,
}

PAIRNORM_GCN_CONFIG = {
    'hidden_features': 64,
    'num_layers': 2,
    'dropout': 0.5,
    'pairnorm_scale': 1.0,
}

GNNGUARD_CONFIG = {
    'hidden_features': 64,
    'num_layers': 2,
    'dropout': 0.5,
    'prune_threshold': 0.1,
}

ROBUSTGCN_CONFIG = {
    'hidden_features': 64,
    'num_layers': 2,
    'dropout': 0.5,
    'gamma': 1.0,
}

RANDOMIZED_SMOOTHING_CONFIG = {
    'sigma': 0.25,
    'n_samples': 1000,
    'n_abstain': 100,
    'alpha': 0.001,
}

GNNCERT_CONFIG = {
    'partition_size': 16,
    'hash_type': 'random',
}

# =============================================================================
# Training Configurations
# =============================================================================

TRAINING_CONFIG = {
    'optimizer': 'adam',
    'lr': 0.01,
    'weight_decay': 5e-4,
    'epochs': 200,
    'patience': 20,
}

# =============================================================================
# Dataset Configurations
# =============================================================================

DATASET_CONFIGS = {
    'cora': {
        'name': 'cora',
        'num_nodes': 2708,
        'num_edges': 5429,
        'num_features': 1433,
        'num_classes': 7,
    },
    'citeseer': {
        'name': 'citeseer',
        'num_nodes': 3327,
        'num_edges': 4732,
        'num_features': 3703,
        'num_classes': 6,
    },
    'pubmed': {
        'name': 'pubmed',
        'num_nodes': 19717,
        'num_edges': 44338,
        'num_features': 500,
        'num_classes': 3,
    },
    'ogbn-arxiv': {
        'name': 'ogbn-arxiv',
        'num_nodes': 169343,
        'num_edges': 1166243,
        'num_features': 128,
        'num_classes': 40,
    },
}

# =============================================================================
# Experiment Configurations
# =============================================================================

EXPERIMENT_CONFIG = {
    'seeds': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # 10 seeds
    'datasets': ['cora', 'citeseer', 'pubmed'],
    'models': [
        'ghn',
        'gcn',
        'gat',
        'sgc',
        'spectral_gcn',
        'groupsort_gcn',
        'pairnorm_gcn',
        'gnnguard',
        'robustgcn',
    ],
    'certified_methods': ['randomized_smoothing', 'gnncert'],
    'radius_thresholds': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
}

# =============================================================================
# Helper Functions
# =============================================================================

def get_model_config(model_name: str) -> dict:
    """Get configuration for a model by name."""
    configs = {
        'ghn': GHN_CONFIG,
        'gcn': GCN_CONFIG,
        'gat': GAT_CONFIG,
        'sgc': SGC_CONFIG,
        'spectral_gcn': SPECTRAL_GCN_CONFIG,
        'groupsort_gcn': GROUPSORT_GCN_CONFIG,
        'pairnorm_gcn': PAIRNORM_GCN_CONFIG,
        'gnnguard': GNNGUARD_CONFIG,
        'robustgcn': ROBUSTGCN_CONFIG,
        'randomized_smoothing': {**GCN_CONFIG, **RANDOMIZED_SMOOTHING_CONFIG},
        'gnncert': {**GCN_CONFIG, **GNNCERT_CONFIG},
    }
    
    model_name = model_name.lower()
    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}")
    
    return configs[model_name].copy()


def get_dataset_config(dataset_name: str) -> dict:
    """Get configuration for a dataset by name."""
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return DATASET_CONFIGS[dataset_name].copy()


def get_training_config() -> dict:
    """Get training configuration."""
    return TRAINING_CONFIG.copy()


def get_experiment_config() -> dict:
    """Get experiment configuration."""
    return EXPERIMENT_CONFIG.copy()


# List of all model names
MODELS = [
    'ghn',
    'gcn', 'gat', 'sgc',
    'spectral_gcn', 'groupsort_gcn', 'pairnorm_gcn',
    'randomized_smoothing', 'gnncert',
    'gnnguard', 'robustgcn',
]
DATASETS = list(DATASET_CONFIGS.keys())