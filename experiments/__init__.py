"""
Experiments module for Graph Hölder Networks.
"""

from .main import (
    run_table1_experiment,
    run_scaling_experiment,
    run_ablation_alpha,
    run_ablation_depth,
)

__all__ = [
    'run_table1_experiment',
    'run_scaling_experiment',
    'run_ablation_alpha',
    'run_ablation_depth',
]
