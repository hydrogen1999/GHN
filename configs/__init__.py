"""
Configuration module for Graph Hölder Networks.
"""

from .default import (
    get_model_config,
    get_training_config,
    get_dataset_config,
    get_experiment_config,
    MODELS,
    DATASETS,
)

__all__ = [
    'get_model_config',
    'get_training_config',
    'get_dataset_config',
    'get_experiment_config',
    'MODELS',
    'DATASETS',
]
