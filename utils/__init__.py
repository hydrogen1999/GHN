"""
Utilities Package

Provides training utilities, metrics, and evaluation functions.
"""

from .training import (
    Trainer,
    EarlyStopping,
    create_optimizer,
    train_and_evaluate,
    set_seed,
    ExperimentLogger,
)

from .metrics import (
    accuracy,
    certified_accuracy_at_radius,
    average_certified_radius,
    margin_statistics,
    compute_confusion_matrix,
    per_class_accuracy,
    f1_score,
    MetricTracker,
    compare_methods,
)

# Analysis utilities (NSR, MAD, etc.)
try:
    from .analysis import (
        compute_nsr,
        compare_nsr_models,
        compute_mad,
        compute_mad_for_model,
        compute_certified_accuracy_curve,
        compare_certified_accuracy_curves,
        analyze_gradient_flow,
        compute_margin_distribution,
        compute_spectral_norms,
        compute_holder_constant,
    )
    HAS_ANALYSIS = True
except ImportError:
    HAS_ANALYSIS = False

__all__ = [
    # Training
    'Trainer',
    'EarlyStopping',
    'create_optimizer',
    'train_and_evaluate',
    'set_seed',
    'ExperimentLogger',
    
    # Metrics
    'accuracy',
    'certified_accuracy_at_radius',
    'average_certified_radius',
    'margin_statistics',
    'compute_confusion_matrix',
    'per_class_accuracy',
    'f1_score',
    'MetricTracker',
    'compare_methods',
]
