"""
Data Loading Package using PyTorch Geometric

Provides utilities for loading citation network datasets
and OGB datasets for node classification.
"""

from .datasets import (
    GraphData,
    load_dataset,
    load_planetoid,
    load_ogbn_arxiv,
    load_synthetic,
    compute_normalized_adjacency,
    get_dataset_statistics,
    print_dataset_info,
    AVAILABLE_DATASETS,
)

__all__ = [
    'GraphData',
    'load_dataset',
    'load_planetoid',
    'load_ogbn_arxiv',
    'load_synthetic',
    'compute_normalized_adjacency',
    'get_dataset_statistics',
    'print_dataset_info',
    'AVAILABLE_DATASETS',
]
