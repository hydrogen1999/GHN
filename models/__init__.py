"""
Graph Hölder Networks - Models Package

This package contains:
- GHN: Graph Hölder Network (main contribution)
- Baselines: GCN, GAT, SGC (standard GNNs)
- Lipschitz: Spectral-GCN, GroupSort-GCN, PairNorm-GCN (constrained GNNs)
- Certified: Randomized Smoothing, GNNCert (certified defenses)
- Empirical: GNNGuard, RobustGCN (empirical defenses)
"""

from .activations import (
    AlphaRePU,
    GroupSort,
    MaxMin,
    get_activation,
)

from .ghn import (
    GraphHolderLayer,
    GraphHolderNetwork,
)

from .baselines import (
    GCN,
    GCNLayer,
    GAT,
    MultiHeadAttention,
    SGC,
)

from .lipschitz import (
    SpectralGCN,
    SpectralGCNLayer,
    GroupSortGCN,
    GroupSortGCNLayer,
    PairNormGCN,
    PairNorm,
    SpectralNorm,
)

from .certified import (
    RandomizedSmoothing,
    GNNCert,
    CertifiedRadiusComputer,
)

from .empirical import (
    GNNGuard,
    GNNGuardLayer,
    RobustGCN,
    GaussianLayer,
    AdaptiveAggregation,
)


# Model registry for easy instantiation
MODEL_REGISTRY = {
    # Main contribution
    'ghn': GraphHolderNetwork,
    
    # Standard baselines
    'gcn': GCN,
    'gat': GAT,
    'sgc': SGC,
    
    # Lipschitz-constrained
    'spectral_gcn': SpectralGCN,
    'groupsort_gcn': GroupSortGCN,
    'pairnorm_gcn': PairNormGCN,
    
    # Empirical defenses
    'gnnguard': GNNGuard,
    'robustgcn': RobustGCN,
}


def get_model(name: str, **kwargs):
    """
    Get model by name.
    
    Args:
        name: Model name (case-insensitive)
        **kwargs: Model constructor arguments
        
    Returns:
        Instantiated model
    """
    name_lower = name.lower()
    
    if name_lower not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    
    return MODEL_REGISTRY[name_lower](**kwargs)


def get_certified_wrapper(
    base_model,
    method: str = 'none',
    **kwargs,
):
    """
    Wrap a model with certification method.
    
    Args:
        base_model: Base GNN model
        method: One of 'none', 'smoothing', 'gnncert'
        **kwargs: Wrapper-specific arguments
        
    Returns:
        Wrapped model (or original if method='none')
    """
    if method == 'none':
        return base_model
    elif method == 'smoothing':
        return RandomizedSmoothing(base_model, **kwargs)
    elif method == 'gnncert':
        return GNNCert(base_model, **kwargs)
    else:
        raise ValueError(f"Unknown certification method: {method}")


__all__ = [
    # Activations
    'AlphaRePU',
    'GroupSort',
    'MaxMin',
    'get_activation',
    
    # GHN
    'GraphHolderLayer',
    'GraphHolderNetwork',
    
    # Baselines
    'GCN',
    'GCNLayer',
    'GAT',
    'MultiHeadAttention',
    'SGC',
    
    # Lipschitz
    'SpectralGCN',
    'SpectralGCNLayer',
    'GroupSortGCN',
    'GroupSortGCNLayer',
    'PairNormGCN',
    'PairNorm',
    'SpectralNorm',
    
    # Certified
    'RandomizedSmoothing',
    'GNNCert',
    'CertifiedRadiusComputer',
    
    # Empirical
    'GNNGuard',
    'GNNGuardLayer',
    'RobustGCN',
    'GaussianLayer',
    'AdaptiveAggregation',
    
    # Utilities
    'MODEL_REGISTRY',
    'get_model',
    'get_certified_wrapper',
]
