"""
Dataset Loaders using PyTorch Geometric

Provides unified interface for loading graph datasets:
- Planetoid: Cora, Citeseer, PubMed
- OGB: ogbn-arxiv
- Synthetic: for testing
"""

import os
from typing import Optional, Tuple, Dict, Any, NamedTuple
from pathlib import Path

import torch
from torch import Tensor
import numpy as np

# PyTorch Geometric imports
try:
    from torch_geometric.datasets import Planetoid, CitationFull, Amazon, Coauthor
    from torch_geometric.transforms import NormalizeFeatures
    from torch_geometric.utils import to_dense_adj, degree
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("Warning: PyTorch Geometric not installed. Using fallback loaders.")

# OGB import
try:
    from ogb.nodeproppred import PygNodePropPredDataset
    HAS_OGB = True
except ImportError:
    HAS_OGB = False


# Available datasets
AVAILABLE_DATASETS = ['cora', 'citeseer', 'pubmed', 'ogbn-arxiv', 'cora_full', 
                      'computers', 'photo', 'cs', 'physics', 'synthetic']


class GraphData(NamedTuple):
    """
    Unified graph data container.
    
    Attributes:
        x: Node feature matrix [num_nodes, num_features]
        edge_index: Edge indices [2, num_edges] (PyG format)
        adj: Dense adjacency matrix [num_nodes, num_nodes] (normalized)
        y: Node labels [num_nodes]
        train_mask: Training node mask
        val_mask: Validation node mask
        test_mask: Test node mask
        num_nodes: Number of nodes
        num_features: Number of node features
        num_classes: Number of classes
        num_edges: Number of edges
        name: Dataset name
    """
    x: Tensor
    edge_index: Tensor
    adj: Tensor
    y: Tensor
    train_mask: Tensor
    val_mask: Tensor
    test_mask: Tensor
    num_nodes: int
    num_features: int
    num_classes: int
    num_edges: int
    name: str


def compute_normalized_adjacency(
    edge_index: Tensor,
    num_nodes: int,
    add_self_loops: bool = True,
    normalization: str = 'sym',
) -> Tensor:
    """
    Compute normalized adjacency matrix from edge_index.
    
    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes
        add_self_loops: Whether to add self-loops
        normalization: 'sym' for D^{-1/2}AD^{-1/2}, 'rw' for D^{-1}A
        
    Returns:
        Normalized dense adjacency matrix [num_nodes, num_nodes]
    """
    # Convert to dense adjacency
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
    
    # Add self-loops
    if add_self_loops:
        adj = adj + torch.eye(num_nodes, device=adj.device)
    
    # Compute degree matrix
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    if normalization == 'sym':
        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        adj_normalized = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
    elif normalization == 'rw':
        # Random walk normalization: D^{-1} A
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        adj_normalized = deg_inv.unsqueeze(1) * adj
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    
    return adj_normalized


def load_planetoid(
    name: str,
    root: str = './data',
    split: str = 'public',
    normalize_features: bool = True,
) -> GraphData:
    """
    Load Planetoid dataset (Cora, Citeseer, PubMed).
    
    Args:
        name: Dataset name ('cora', 'citeseer', 'pubmed')
        root: Root directory for data
        split: Split type ('public', 'random', 'full')
        normalize_features: Whether to normalize features
        
    Returns:
        GraphData namedtuple
    """
    if not HAS_PYG:
        raise ImportError("PyTorch Geometric required for Planetoid datasets")
    
    # Apply transforms
    transform = NormalizeFeatures() if normalize_features else None
    
    # Load dataset
    dataset = Planetoid(
        root=root,
        name=name.capitalize() if name != 'pubmed' else 'PubMed',
        split=split,
        transform=transform,
    )
    
    data = dataset[0]
    
    # Compute normalized adjacency
    adj = compute_normalized_adjacency(
        data.edge_index,
        data.num_nodes,
        add_self_loops=True,
        normalization='sym',
    )
    
    return GraphData(
        x=data.x,
        edge_index=data.edge_index,
        adj=adj,
        y=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
        num_nodes=data.num_nodes,
        num_features=data.num_features,
        num_classes=dataset.num_classes,
        num_edges=data.edge_index.size(1),
        name=name,
    )


def load_ogbn_arxiv(
    root: str = './data',
    normalize_features: bool = True,
) -> GraphData:
    """
    Load ogbn-arxiv dataset from Open Graph Benchmark.
    
    Returns:
        GraphData namedtuple
    """
    if not HAS_OGB:
        raise ImportError("OGB required for ogbn-arxiv. Install with: pip install ogb")
    
    if not HAS_PYG:
        raise ImportError("PyTorch Geometric required for ogbn-arxiv")
    
    # Load dataset
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=root)
    data = dataset[0]
    
    # Get splits
    split_idx = dataset.get_idx_split()
    
    # Create masks
    num_nodes = data.num_nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[split_idx['train']] = True
    val_mask[split_idx['valid']] = True
    test_mask[split_idx['test']] = True
    
    # Normalize features
    x = data.x
    if normalize_features:
        x = x / (x.sum(dim=1, keepdim=True) + 1e-8)
    
    # Compute normalized adjacency (may be memory-intensive for large graphs)
    adj = compute_normalized_adjacency(
        data.edge_index,
        num_nodes,
        add_self_loops=True,
        normalization='sym',
    )
    
    return GraphData(
        x=x,
        edge_index=data.edge_index,
        adj=adj,
        y=data.y.squeeze(),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=num_nodes,
        num_features=data.x.size(1),
        num_classes=dataset.num_classes,
        num_edges=data.edge_index.size(1),
        name='ogbn-arxiv',
    )


def load_citation_full(
    name: str,
    root: str = './data',
    normalize_features: bool = True,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> GraphData:
    """
    Load full citation dataset (Cora_Full, etc.).
    """
    if not HAS_PYG:
        raise ImportError("PyTorch Geometric required")
    
    transform = NormalizeFeatures() if normalize_features else None
    dataset = CitationFull(root=root, name=name, transform=transform)
    data = dataset[0]
    
    # Create random splits
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    adj = compute_normalized_adjacency(data.edge_index, num_nodes)
    
    return GraphData(
        x=data.x,
        edge_index=data.edge_index,
        adj=adj,
        y=data.y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=num_nodes,
        num_features=data.num_features,
        num_classes=dataset.num_classes,
        num_edges=data.edge_index.size(1),
        name=name,
    )


def load_amazon(
    name: str,
    root: str = './data',
    normalize_features: bool = True,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> GraphData:
    """
    Load Amazon dataset (Computers, Photo).
    """
    if not HAS_PYG:
        raise ImportError("PyTorch Geometric required")
    
    transform = NormalizeFeatures() if normalize_features else None
    dataset = Amazon(root=root, name=name.capitalize(), transform=transform)
    data = dataset[0]
    
    # Create random splits
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    adj = compute_normalized_adjacency(data.edge_index, num_nodes)
    
    return GraphData(
        x=data.x,
        edge_index=data.edge_index,
        adj=adj,
        y=data.y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=num_nodes,
        num_features=data.num_features,
        num_classes=dataset.num_classes,
        num_edges=data.edge_index.size(1),
        name=name,
    )


def load_coauthor(
    name: str,
    root: str = './data',
    normalize_features: bool = True,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> GraphData:
    """
    Load Coauthor dataset (CS, Physics).
    """
    if not HAS_PYG:
        raise ImportError("PyTorch Geometric required")
    
    transform = NormalizeFeatures() if normalize_features else None
    dataset = Coauthor(root=root, name=name.upper(), transform=transform)
    data = dataset[0]
    
    # Create random splits
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    adj = compute_normalized_adjacency(data.edge_index, num_nodes)
    
    return GraphData(
        x=data.x,
        edge_index=data.edge_index,
        adj=adj,
        y=data.y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=num_nodes,
        num_features=data.num_features,
        num_classes=dataset.num_classes,
        num_edges=data.edge_index.size(1),
        name=name,
    )


def load_synthetic(
    num_nodes: int = 1000,
    num_features: int = 64,
    num_classes: int = 5,
    edge_prob: float = 0.02,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> GraphData:
    """
    Generate synthetic graph dataset for testing.
    
    Args:
        num_nodes: Number of nodes
        num_features: Feature dimension
        num_classes: Number of classes
        edge_prob: Edge probability (Erdős-Rényi)
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        seed: Random seed
        
    Returns:
        GraphData namedtuple
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate features (random normal)
    x = torch.randn(num_nodes, num_features)
    x = x / (x.norm(dim=1, keepdim=True) + 1e-8)  # Normalize
    
    # Generate labels (random)
    y = torch.randint(0, num_classes, (num_nodes,))
    
    # Generate edges (Erdős-Rényi with class homophily)
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Higher probability for same-class edges
            p = edge_prob * 3 if y[i] == y[j] else edge_prob
            if np.random.random() < p:
                edges.append([i, j])
                edges.append([j, i])  # Undirected
    
    if len(edges) == 0:
        # Ensure at least some edges
        for i in range(num_nodes):
            j = (i + 1) % num_nodes
            edges.append([i, j])
            edges.append([j, i])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Create splits
    indices = torch.randperm(num_nodes)
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    # Compute normalized adjacency
    adj = compute_normalized_adjacency(edge_index, num_nodes)
    
    return GraphData(
        x=x,
        edge_index=edge_index,
        adj=adj,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=num_nodes,
        num_features=num_features,
        num_classes=num_classes,
        num_edges=edge_index.size(1),
        name='synthetic',
    )


def load_dataset(
    name: str,
    root: str = './data',
    **kwargs,
) -> GraphData:
    """
    Load dataset by name.
    
    Args:
        name: Dataset name (case-insensitive)
        root: Root directory for data
        **kwargs: Additional arguments for specific loaders
        
    Returns:
        GraphData namedtuple
    """
    name = name.lower().replace('-', '_')
    
    if name in ['cora', 'citeseer', 'pubmed']:
        return load_planetoid(name, root, **kwargs)
    
    elif name == 'ogbn_arxiv':
        return load_ogbn_arxiv(root, **kwargs)
    
    elif name == 'cora_full':
        return load_citation_full('Cora', root, **kwargs)
    
    elif name in ['computers', 'photo']:
        return load_amazon(name, root, **kwargs)
    
    elif name in ['cs', 'physics']:
        return load_coauthor(name, root, **kwargs)
    
    elif name == 'synthetic':
        return load_synthetic(**kwargs)
    
    else:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available: {AVAILABLE_DATASETS}"
        )


def get_dataset_statistics(data: GraphData) -> Dict[str, Any]:
    """
    Compute dataset statistics.
    
    Args:
        data: GraphData namedtuple
        
    Returns:
        Dictionary with statistics
    """
    # Edge statistics
    num_edges = data.edge_index.size(1) // 2  # Undirected
    avg_degree = data.edge_index.size(1) / data.num_nodes
    
    # Class distribution
    class_counts = torch.bincount(data.y, minlength=data.num_classes)
    class_dist = class_counts.float() / data.num_nodes
    
    # Split sizes
    train_size = data.train_mask.sum().item()
    val_size = data.val_mask.sum().item()
    test_size = data.test_mask.sum().item()
    
    # Feature statistics
    feature_mean = data.x.mean().item()
    feature_std = data.x.std().item()
    feature_sparsity = (data.x == 0).float().mean().item()
    
    # Homophily (edge homophily ratio)
    src, dst = data.edge_index
    same_class = (data.y[src] == data.y[dst]).float().mean().item()
    
    return {
        'name': data.name,
        'num_nodes': data.num_nodes,
        'num_edges': num_edges,
        'num_features': data.num_features,
        'num_classes': data.num_classes,
        'avg_degree': avg_degree,
        'class_distribution': class_dist.tolist(),
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'feature_sparsity': feature_sparsity,
        'homophily': same_class,
    }


def print_dataset_info(data: GraphData):
    """Print formatted dataset information."""
    stats = get_dataset_statistics(data)
    
    print(f"\nDataset: {stats['name']}")
    print("=" * 40)
    print(f"Nodes:      {stats['num_nodes']:,}")
    print(f"Edges:      {stats['num_edges']:,}")
    print(f"Features:   {stats['num_features']}")
    print(f"Classes:    {stats['num_classes']}")
    print(f"Avg Degree: {stats['avg_degree']:.2f}")
    print(f"Homophily:  {stats['homophily']:.3f}")
    print(f"Splits:     {stats['train_size']}/{stats['val_size']}/{stats['test_size']} "
          f"(train/val/test)")
