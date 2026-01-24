"""
Certification Package

Provides utilities for computing certified robustness radii
and evaluating certified accuracy.
"""

from .certification import (
    compute_classification_margin,
    compute_holder_certified_radius,
    compute_lipschitz_certified_radius,
    compute_network_holder_constant,
    compute_network_lipschitz_constant,
    certify_single_node,
    certify_all_nodes,
    compare_scaling_behavior,
    CertificationCache,
)

__all__ = [
    'compute_classification_margin',
    'compute_holder_certified_radius',
    'compute_lipschitz_certified_radius',
    'compute_network_holder_constant',
    'compute_network_lipschitz_constant',
    'certify_single_node',
    'certify_all_nodes',
    'compare_scaling_behavior',
    'CertificationCache',
]
