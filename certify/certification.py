"""
Certification Module

Provides utilities for computing certified robustness radii
and certified accuracy metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from torch import Tensor
from tqdm import tqdm
import torch.nn.functional as F


def compute_classification_margin(
    logits: Tensor,
    true_label: int,
) -> float:
    """
    Compute classification margin γ = f_y - max_{k≠y} f_k.
    
    Args:
        logits: Output logits [C]
        true_label: Ground truth label
        
    Returns:
        Classification margin (negative if misclassified)
    """
    logits = logits.detach().cpu()
    pred_label = logits.argmax().item()
    
    logits_sorted, _ = torch.sort(logits, descending=True)
    
    if pred_label == true_label:
        # Correct prediction
        true_logit = logits[true_label]
        # Runner-up is second highest
        if logits_sorted[0] == true_logit:
            runner_up = logits_sorted[1]
        else:
            runner_up = logits_sorted[0]
        margin = (true_logit - runner_up).item()
    else:
        # Incorrect prediction
        margin = (logits[true_label] - logits_sorted[0]).item()
    
    return margin


def compute_holder_certified_radius(
    margin: float,
    holder_constant: float,
    alpha_net: float,
) -> float:
    """
    Compute certified radius for Hölder networks.
    
    R = (γ / (2 * C_net))^{1/α_net}
    
    This scales super-linearly with margin when α_net < 1.
    
    Args:
        margin: Classification margin γ
        holder_constant: Network Hölder constant C_net
        alpha_net: Global Hölder exponent α^L
        
    Returns:
        Certified L2 radius
    """
    if margin <= 0:
        return 0.0
    
    return (margin / (2 * holder_constant)) ** (1 / alpha_net)


def compute_lipschitz_certified_radius(
    margin: float,
    lipschitz_constant: float,
) -> float:
    """
    Compute certified radius for Lipschitz networks.
    
    R = γ / (2K)
    
    This scales linearly with margin.
    
    Args:
        margin: Classification margin γ
        lipschitz_constant: Network Lipschitz constant K
        
    Returns:
        Certified L2 radius
    """
    if margin <= 0:
        return 0.0
    
    return margin / (2 * lipschitz_constant)


def _power_iteration_spectral_norm(weight: Tensor, num_iters: int = 10) -> float:
    """
    Compute spectral norm of a weight matrix using power iteration.
    
    For matrix W of shape (m, n):
    - u ∈ R^m (left singular vector)
    - v ∈ R^n (right singular vector)
    - Power iteration: v = W^T u / ||W^T u||, u = W v / ||W v||
    - σ = u^T W v
    
    Args:
        weight: Weight matrix of shape (m, n)
        num_iters: Number of power iterations
        
    Returns:
        Estimated spectral norm
    """
    m, n = weight.shape
    device = weight.device
    
    # Initialize u (left singular vector, size m)
    u = torch.randn(m, device=device)
    u = F.normalize(u, dim=0)
    
    with torch.no_grad():
        for _ in range(num_iters):
            # v = W^T u / ||W^T u||  (v has size n)
            v = torch.mv(weight.t(), u)
            v = F.normalize(v, dim=0)
            # u = W v / ||W v||  (u has size m)
            u = torch.mv(weight, v)
            u = F.normalize(u, dim=0)
        
        # σ = u^T W v
        sigma = torch.dot(u, torch.mv(weight, v))
    
    return abs(sigma.item())


def compute_network_holder_constant(
    model,
    alpha: float,
) -> float:
    """
    Compute network Hölder constant C_net = Π_l ||W^{(l)}||_2^α.
    
    Args:
        model: GHN model with layers attribute
        alpha: Layer-wise Hölder exponent
        
    Returns:
        Network Hölder constant
    """
    c_net = 1.0
    
    for layer in model.layers:
        if hasattr(layer, 'get_spectral_norm'):
            spectral_norm = layer.get_spectral_norm()
            c_net *= spectral_norm ** alpha
        elif hasattr(layer, 'weight'):
            # Fallback: estimate spectral norm via power iteration
            weight = layer.weight.data
            spectral_norm = _power_iteration_spectral_norm(weight)
            c_net *= spectral_norm ** alpha
    
    # Include readout layer if exists
    if hasattr(model, 'readout'):
        weight = model.readout.weight.data
        # nn.Linear stores weight as (out_features, in_features)
        readout_norm = _power_iteration_spectral_norm(weight)
        c_net *= readout_norm
    
    return c_net


def compute_network_lipschitz_constant(model) -> float:
    """
    Compute network Lipschitz constant K = Π_l ||W^{(l)}||_2.
    
    Args:
        model: Model with layers attribute
        
    Returns:
        Network Lipschitz constant
    """
    k = 1.0
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2:
            weight = param.data
            spectral_norm = _power_iteration_spectral_norm(weight)
            k *= spectral_norm
    
    return k


@torch.no_grad()
def certify_single_node(
    model,
    x: Tensor,
    adj: Tensor,
    node_idx: int,
    true_label: int,
    model_type: str = 'ghn',
    alpha: float = 0.8,
    num_layers: int = 2,
) -> Tuple[bool, float, float]:
    """
    Certify a single node.
    
    Args:
        model: Trained model
        x: Node features [n, d]
        adj: Adjacency matrix [n, n]
        node_idx: Target node index
        true_label: Ground truth label
        model_type: One of 'ghn', 'lipschitz', 'gcn'
        alpha: Hölder exponent (for GHN)
        num_layers: Network depth
        
    Returns:
        (is_correct, margin, certified_radius)
    """
    model.eval()
    
    # Forward pass
    logits = model(x, adj)
    node_logits = logits[node_idx]
    
    # Check correctness
    pred_label = node_logits.argmax().item()
    is_correct = pred_label == true_label
    
    # Compute margin
    margin = compute_classification_margin(node_logits, true_label)
    
    if not is_correct:
        return False, margin, 0.0
    
    # Compute certified radius based on model type
    if model_type == 'ghn':
        alpha_net = alpha ** num_layers
        c_net = compute_network_holder_constant(model, alpha)
        radius = compute_holder_certified_radius(margin, c_net, alpha_net)
    elif model_type == 'lipschitz':
        k = compute_network_lipschitz_constant(model)
        radius = compute_lipschitz_certified_radius(margin, k)
    else:
        # For standard GCN, compute Lipschitz constant post-hoc
        k = compute_network_lipschitz_constant(model)
        radius = compute_lipschitz_certified_radius(margin, k)
    
    return True, margin, radius


@torch.no_grad()
def certify_all_nodes(
    model,
    x: Tensor,
    adj: Tensor,
    labels: Tensor,
    test_mask: Tensor,
    model_type: str = 'ghn',
    alpha: float = 0.8,
    num_layers: int = 2,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Certify all test nodes and compute metrics.
    
    Args:
        model: Trained model
        x: Node features [n, d]
        adj: Adjacency matrix [n, n]
        labels: Node labels [n]
        test_mask: Boolean mask for test nodes [n]
        model_type: Model type for certification
        alpha: Hölder exponent
        num_layers: Network depth
        verbose: Show progress bar
        
    Returns:
        Dictionary with metrics:
        - clean_accuracy
        - certified_accuracy@{r} for various radii
        - average_certified_radius (ACR)
    """
    model.eval()
    
    test_indices = test_mask.nonzero().squeeze(-1).tolist()
    
    correct_count = 0
    certified_radii = []
    margins = []
    
    iterator = tqdm(test_indices, desc="Certifying") if verbose else test_indices
    
    all_predictions = []
    for idx in iterator:
        true_label = labels[idx].item()
        
        is_correct, margin, radius = certify_single_node(
            model, x, adj, idx, true_label,
            model_type=model_type,
            alpha=alpha,
            num_layers=num_layers,
        )
        pred_label = model(x, adj)[idx].argmax().item()
        all_predictions.append(pred_label)
        
        if is_correct:
            correct_count += 1
            certified_radii.append(radius)
            margins.append(margin)
        else:
            certified_radii.append(0.0)
            margins.append(margin)
    n_test = len(test_indices)
    n_correct = correct_count
    
    # Metrics
    clean_accuracy = n_correct / n_test
    
    # Average certified radius (over correctly classified)
    acr = np.mean(certified_radii) if certified_radii else 0.0
    
    # Certified accuracy at various radii
    certified_radii = np.array(certified_radii)
    radii_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    results = {
        'clean_accuracy': clean_accuracy,
        'average_certified_radius': acr,
        'num_test': n_test,
        'num_correct': n_correct,
        'radii': certified_radii,
        'predictions': all_predictions,
        'margins': margins
    }
    
    for r in radii_thresholds:
        cert_acc = np.sum(certified_radii >= r) / n_test
        results[f'certified_accuracy@{r}'] = cert_acc
    
    # Margin statistics
    if margins:
        results['mean_margin'] = np.mean(margins)
        results['median_margin'] = np.median(margins)
        results['std_margin'] = np.std(margins)
    
    return results


def compare_scaling_behavior(
    ghn_model,
    lipschitz_model,
    x: Tensor,
    adj: Tensor,
    labels: Tensor,
    test_mask: Tensor,
    alpha: float,
    num_layers: int,
) -> Dict[str, np.ndarray]:
    """
    Compare certified radius scaling between Hölder and Lipschitz.
    """
    # Helper to compute margins and radii
    def get_data(model, model_type):
        model.eval()
        with torch.no_grad():
            logits = model(x, adj)
            
        test_indices = test_mask.nonzero().squeeze(-1)
        margins = []
        radii = []
        
        for idx in test_indices:
            idx = idx.item()
            true_label = labels[idx].item()
            # Reuse single node logic or inline it for speed
            node_logits = logits[idx]
            margin = compute_classification_margin(node_logits, true_label)
            
            if model_type == 'ghn':
                c_net = compute_network_holder_constant(model, alpha)
                radius = compute_holder_certified_radius(margin, c_net, alpha**num_layers)
            else:
                k = compute_network_lipschitz_constant(model)
                radius = compute_lipschitz_certified_radius(margin, k)
            
            margins.append(margin)
            radii.append(radius)
        return np.array(margins), np.array(radii)

    ghn_margins, ghn_radii = get_data(ghn_model, 'ghn')
    lip_margins, lip_radii = get_data(lipschitz_model, 'lipschitz')
    
    return {
        'margins': ghn_margins,
        'ghn_radii': ghn_radii,
        'lipschitz_radii': lip_radii,
    }


class CertificationCache:
    """
    Cache for certification results to avoid recomputation.
    """
    
    def __init__(self):
        self._cache = {}
    
    def get(self, key: str) -> Optional[Dict]:
        return self._cache.get(key)
    
    def set(self, key: str, value: Dict):
        self._cache[key] = value
    
    def clear(self):
        self._cache.clear()
    
    @staticmethod
    def make_key(
        model_name: str,
        dataset_name: str,
        seed: int,
    ) -> str:
        return f"{model_name}_{dataset_name}_{seed}"