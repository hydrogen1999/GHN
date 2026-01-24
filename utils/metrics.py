"""
Metrics and Evaluation Utilities
"""

import torch
import numpy as np
from torch import Tensor
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def accuracy(
    logits: Tensor,
    labels: Tensor,
    mask: Optional[Tensor] = None,
) -> float:
    """
    Compute classification accuracy.
    
    Args:
        logits: Prediction logits [n, C]
        labels: Ground truth labels [n]
        mask: Optional boolean mask [n]
        
    Returns:
        Accuracy (0-1)
    """
    preds = logits.argmax(dim=-1)
    
    if mask is not None:
        preds = preds[mask]
        labels = labels[mask]
    
    return (preds == labels).float().mean().item()


def certified_accuracy_at_radius(
    certified_radii: np.ndarray,
    radius_threshold: float,
    total_nodes: int,
) -> float:
    """
    Compute certified accuracy at a given radius.
    
    Args:
        certified_radii: Array of certified radii (0 for incorrect predictions)
        radius_threshold: Minimum radius threshold
        total_nodes: Total number of test nodes
        
    Returns:
        Certified accuracy (fraction of nodes certified at this radius)
    """
    return np.sum(certified_radii >= radius_threshold) / total_nodes


def average_certified_radius(
    certified_radii: np.ndarray,
    only_positive: bool = True,
) -> float:
    """
    Compute average certified radius.
    
    Args:
        certified_radii: Array of certified radii
        only_positive: Only average over positive (correct) predictions
        
    Returns:
        Average certified radius
    """
    if only_positive:
        positive_radii = certified_radii[certified_radii > 0]
        return float(np.mean(positive_radii)) if len(positive_radii) > 0 else 0.0
    else:
        return float(np.mean(certified_radii))


def margin_statistics(margins: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics over classification margins.
    
    Args:
        margins: Array of classification margins
        
    Returns:
        Dictionary with mean, std, median, min, max
    """
    positive_margins = margins[margins > 0]
    
    return {
        'mean': float(np.mean(margins)),
        'std': float(np.std(margins)),
        'median': float(np.median(margins)),
        'min': float(np.min(margins)),
        'max': float(np.max(margins)),
        'positive_mean': float(np.mean(positive_margins)) if len(positive_margins) > 0 else 0.0,
        'positive_fraction': float(np.sum(margins > 0) / len(margins)),
    }


def compute_confusion_matrix(
    logits: Tensor,
    labels: Tensor,
    num_classes: int,
    mask: Optional[Tensor] = None,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        logits: Prediction logits [n, C]
        labels: Ground truth labels [n]
        num_classes: Number of classes
        mask: Optional boolean mask [n]
        
    Returns:
        Confusion matrix [C, C]
    """
    preds = logits.argmax(dim=-1)
    
    if mask is not None:
        preds = preds[mask]
        labels = labels[mask]
    
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    
    return cm


def per_class_accuracy(
    logits: Tensor,
    labels: Tensor,
    num_classes: int,
    mask: Optional[Tensor] = None,
) -> Dict[int, float]:
    """
    Compute per-class accuracy.
    
    Returns:
        Dictionary mapping class index to accuracy
    """
    cm = compute_confusion_matrix(logits, labels, num_classes, mask)
    
    per_class = {}
    for c in range(num_classes):
        total = cm[c].sum()
        correct = cm[c, c]
        per_class[c] = correct / total if total > 0 else 0.0
    
    return per_class


def f1_score(
    logits: Tensor,
    labels: Tensor,
    num_classes: int,
    average: str = 'macro',
    mask: Optional[Tensor] = None,
) -> float:
    """
    Compute F1 score.
    
    Args:
        logits: Prediction logits [n, C]
        labels: Ground truth labels [n]
        num_classes: Number of classes
        average: 'macro' or 'micro'
        mask: Optional boolean mask [n]
        
    Returns:
        F1 score
    """
    cm = compute_confusion_matrix(logits, labels, num_classes, mask)
    
    if average == 'micro':
        # Micro: global TP, FP, FN
        tp = np.diag(cm).sum()
        fp = cm.sum() - np.diag(cm).sum()
        fn = fp  # For multi-class
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    else:  # macro
        f1_scores = []
        for c in range(num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        return float(np.mean(f1_scores))


class MetricTracker:
    """
    Track metrics over training.
    """
    
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def update(self, **kwargs):
        """Update metrics."""
        for name, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[name].append(value)
    
    def get_mean(self, name: str, last_n: Optional[int] = None) -> float:
        """Get mean of metric."""
        values = self.metrics[name]
        if last_n is not None:
            values = values[-last_n:]
        return float(np.mean(values)) if values else 0.0
    
    def get_std(self, name: str, last_n: Optional[int] = None) -> float:
        """Get std of metric."""
        values = self.metrics[name]
        if last_n is not None:
            values = values[-last_n:]
        return float(np.std(values)) if values else 0.0
    
    def get_best(self, name: str, mode: str = 'max') -> Tuple[float, int]:
        """Get best value and epoch."""
        values = self.metrics[name]
        if not values:
            return 0.0, -1
        
        if mode == 'max':
            idx = int(np.argmax(values))
        else:
            idx = int(np.argmin(values))
        
        return values[idx], idx
    
    def get_last(self, name: str) -> float:
        """Get last value."""
        values = self.metrics[name]
        return values[-1] if values else 0.0
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return dict(self.metrics)


def compare_methods(
    results: Dict[str, Dict[str, float]],
) -> str:
    """
    Format comparison table.
    
    Args:
        results: Dictionary mapping method name to metrics
        
    Returns:
        Formatted table string
    """
    lines = []
    
    # Header
    metrics = ['test_accuracy', 'average_certified_radius']
    header = f"{'Method':<20} " + " ".join(f"{m:<25}" for m in metrics)
    lines.append(header)
    lines.append("-" * len(header))
    
    # Rows
    for method, method_results in results.items():
        row = f"{method:<20} "
        for metric in metrics:
            value = method_results.get(metric, 'N/A')
            if isinstance(value, float):
                row += f"{value:<25.4f}"
            else:
                row += f"{str(value):<25}"
        lines.append(row)
    
    return "\n".join(lines)
