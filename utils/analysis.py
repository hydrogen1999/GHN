"""
Analysis Utilities for Graph Hölder Networks

Implements diagnostic metrics from the paper:
1. NSR (Noise-to-Signal Ratio) - Section 1.6
2. MAD (Mean Average Distance) - Section 1.7
3. Gradient flow analysis
4. Certified accuracy curves
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional
import numpy as np


# =============================================================================
# Noise-to-Signal Ratio (NSR) Analysis - Section 1.6
# =============================================================================

def compute_layer_representations(
    model: nn.Module,
    x: Tensor,
    adj: Tensor,
) -> List[Tensor]:
    """
    Extract intermediate representations from each layer.
    
    Args:
        model: GNN model (must have .layers attribute)
        x: Input features
        adj: Normalized adjacency
        
    Returns:
        List of representations [H^(0), H^(1), ..., H^(L)]
    """
    representations = [x]
    h = x
    
    # Handle different model architectures
    if hasattr(model, 'layers'):
        for layer in model.layers:
            h = layer(h, adj)
            representations.append(h.clone())
    elif hasattr(model, 'conv1') and hasattr(model, 'conv2'):
        # 2-layer model with conv1, conv2
        h = model.conv1(h, adj) if hasattr(model.conv1, '__call__') else F.relu(h @ model.conv1.weight.T)
        representations.append(h.clone())
        h = model.conv2(h, adj) if hasattr(model.conv2, '__call__') else h @ model.conv2.weight.T
        representations.append(h.clone())
    else:
        # Fallback: just return input and output
        with torch.no_grad():
            out = model(x, adj)
        representations.append(out)
    
    return representations


def compute_nsr(
    model: nn.Module,
    x: Tensor,
    adj: Tensor,
    noise_std: float = 0.1,
    num_samples: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Compute layer-wise Noise-to-Signal Ratio.
    
    NSR^(l) = ||H^(l)(X + ξ) - H^(l)(X)||_F / ||H^(l)(X)||_F
    
    where ξ ~ N(0, σ²I) is input noise.
    
    Args:
        model: GNN model
        x: Clean input features
        adj: Normalized adjacency
        noise_std: Standard deviation σ of input noise
        num_samples: Number of noise samples for estimation
        
    Returns:
        Dictionary with:
            - 'nsr_per_layer': NSR at each layer [L+1]
            - 'nsr_ratio': NSR decay ratio between consecutive layers
            - 'clean_norms': Frobenius norms of clean representations
    """
    model.eval()
    device = x.device
    
    # Get clean representations
    with torch.no_grad():
        clean_reps = compute_layer_representations(model, x, adj)
    
    num_layers = len(clean_reps)
    nsr_accumulator = [[] for _ in range(num_layers)]
    
    # Estimate NSR via Monte Carlo
    for _ in range(num_samples):
        # Add Gaussian noise to input
        noise = torch.randn_like(x) * noise_std
        x_noisy = x + noise
        
        with torch.no_grad():
            noisy_reps = compute_layer_representations(model, x_noisy, adj)
        
        for l in range(num_layers):
            diff_norm = (noisy_reps[l] - clean_reps[l]).norm('fro').item()
            clean_norm = clean_reps[l].norm('fro').item() + 1e-8
            nsr_accumulator[l].append(diff_norm / clean_norm)
    
    # Compute statistics
    nsr_per_layer = np.array([np.mean(nsr_accumulator[l]) for l in range(num_layers)])
    nsr_std = np.array([np.std(nsr_accumulator[l]) for l in range(num_layers)])
    
    # Compute decay ratio between consecutive layers
    nsr_ratio = []
    for l in range(1, num_layers):
        if nsr_per_layer[l-1] > 1e-8:
            nsr_ratio.append(nsr_per_layer[l] / nsr_per_layer[l-1])
        else:
            nsr_ratio.append(1.0)
    nsr_ratio = np.array(nsr_ratio)
    
    clean_norms = np.array([clean_reps[l].norm('fro').item() for l in range(num_layers)])
    
    return {
        'nsr_per_layer': nsr_per_layer,
        'nsr_std': nsr_std,
        'nsr_ratio': nsr_ratio,
        'clean_norms': clean_norms,
        'num_layers': num_layers,
    }


def compare_nsr_models(
    models: Dict[str, nn.Module],
    x: Tensor,
    adj: Tensor,
    noise_std: float = 0.1,
    num_samples: int = 100,
) -> Dict[str, Dict]:
    """
    Compare NSR across multiple models (Figure 3).
    
    Args:
        models: Dictionary mapping model names to models
        x: Input features
        adj: Normalized adjacency
        noise_std: Noise standard deviation
        num_samples: Monte Carlo samples
        
    Returns:
        Dictionary mapping model names to NSR results
    """
    results = {}
    for name, model in models.items():
        results[name] = compute_nsr(model, x, adj, noise_std, num_samples)
    return results


# =============================================================================
# Mean Average Distance (MAD) - Section 1.7
# =============================================================================

def compute_mad(embeddings: Tensor) -> float:
    """
    Compute Mean Average Distance (MAD) for representation diversity.
    
    MAD = (1 / n(n-1)) * Σ_{i≠j} ||h̄_i - h̄_j||_2
    
    where h̄_i = h_i / ||h_i||_2 is ℓ2-normalized embedding.
    
    Lower MAD indicates representation collapse (oversmoothing).
    
    Args:
        embeddings: Node embeddings [num_nodes, embed_dim]
        
    Returns:
        MAD value (scalar)
    """
    # L2 normalize embeddings
    norms = embeddings.norm(dim=1, keepdim=True) + 1e-8
    normalized = embeddings / norms
    
    # Compute pairwise distances
    # ||h̄_i - h̄_j||_2^2 = 2 - 2 * h̄_i · h̄_j (since ||h̄||=1)
    similarity = normalized @ normalized.T
    distances = torch.sqrt(2 - 2 * similarity.clamp(-1, 1) + 1e-8)
    
    # Mean of off-diagonal elements
    n = embeddings.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=embeddings.device)
    mad = distances[mask].mean().item()
    
    return mad


def compute_mad_for_model(
    model: nn.Module,
    x: Tensor,
    adj: Tensor,
) -> float:
    """
    Compute MAD for a model's final layer embeddings.
    
    Args:
        model: GNN model
        x: Input features
        adj: Normalized adjacency
        
    Returns:
        MAD value
    """
    model.eval()
    
    with torch.no_grad():
        # Get embeddings before final classification layer
        if hasattr(model, 'get_embeddings'):
            embeddings = model.get_embeddings(x, adj)
        else:
            # Try to get second-to-last layer output
            reps = compute_layer_representations(model, x, adj)
            embeddings = reps[-2] if len(reps) > 1 else reps[-1]
    
    return compute_mad(embeddings)


def analyze_oversmoothing(
    model_class,
    model_kwargs: Dict,
    depths: List[int],
    x: Tensor,
    adj: Tensor,
    labels: Tensor,
    train_mask: Tensor,
    val_mask: Tensor,
    test_mask: Tensor,
    train_config: Dict,
    device: torch.device,
    num_seeds: int = 3,
) -> Dict[str, Dict]:
    """
    Analyze oversmoothing across network depths (Table 5).
    
    Args:
        model_class: Model class to instantiate
        model_kwargs: Base kwargs for model (without num_layers)
        depths: List of depths to evaluate
        x, adj, labels, masks: Data
        train_config: Training configuration
        device: Computation device
        num_seeds: Number of random seeds
        
    Returns:
        Dictionary with accuracy and MAD for each depth
    """
    from utils.training import train_and_evaluate, set_seed
    
    results = {}
    
    for depth in depths:
        results[depth] = {'accuracy': [], 'mad': []}
        
        for seed in range(num_seeds):
            set_seed(seed)
            
            # Create model with specified depth
            kwargs = {**model_kwargs, 'num_layers': depth}
            model = model_class(**kwargs)
            
            # Create simple data container
            class DataContainer:
                pass
            data = DataContainer()
            data.x = x
            data.adj = adj
            data.y = labels
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask
            data.num_features = x.size(1)
            data.num_classes = labels.max().item() + 1
            
            # Train
            train_results = train_and_evaluate(model, data, train_config, device, verbose=False)
            
            # Compute MAD
            model.to(device)
            mad = compute_mad_for_model(model, x.to(device), adj.to(device))
            
            results[depth]['accuracy'].append(train_results['test_accuracy'])
            results[depth]['mad'].append(mad)
        
        # Compute statistics
        results[depth]['accuracy_mean'] = np.mean(results[depth]['accuracy'])
        results[depth]['accuracy_std'] = np.std(results[depth]['accuracy'])
        results[depth]['mad_mean'] = np.mean(results[depth]['mad'])
        results[depth]['mad_std'] = np.std(results[depth]['mad'])
    
    return results


# =============================================================================
# Certified Accuracy Curves - Figure 1
# =============================================================================

def compute_certified_accuracy_curve(
    certified_radii: Tensor,
    predictions: Tensor,
    labels: Tensor,
    radii_range: Optional[np.ndarray] = None,
    num_points: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute certified accuracy as a function of perturbation radius.
    
    CA@r = fraction of test nodes that are:
        1. Correctly classified on clean input
        2. Have certified radius R_i >= r
    
    Args:
        certified_radii: Certified radius for each test node
        predictions: Model predictions on clean input
        labels: True labels
        radii_range: Range of r values (optional)
        num_points: Number of points in the curve
        
    Returns:
        (radii, certified_accuracies) arrays for plotting
    """
    if radii_range is None:
        max_r = certified_radii.max().item() * 1.2
        radii_range = np.linspace(0, max_r, num_points)
    
    # Correct predictions mask
    correct = (predictions == labels)
    
    certified_accuracies = []
    for r in radii_range:
        # Certified at radius r: correct AND radius >= r
        certified_at_r = (correct & (certified_radii >= r)).float().mean().item()
        certified_accuracies.append(certified_at_r)
    
    return radii_range, np.array(certified_accuracies)


def compare_certified_accuracy_curves(
    results: Dict[str, Dict],
    radii_range: Optional[np.ndarray] = None,
    num_points: int = 50,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compare certified accuracy curves across multiple methods (Figure 1).
    
    Args:
        results: Dictionary mapping method names to certification results
                 Each result should have 'radii', 'predictions', 'labels'
        radii_range: Common range of r values
        num_points: Number of points
        
    Returns:
        Dictionary mapping method names to (radii, ca) tuples
    """
    # Determine common radii range
    if radii_range is None:
        max_r = max(
            np.max(res['radii']) for res in results.values()
        ) * 1.2
        radii_range = np.linspace(0, max_r, num_points)
    
    curves = {}
    for name, res in results.items():
        radii, ca = compute_certified_accuracy_curve(
            torch.tensor(res['radii']),
            torch.tensor(res['predictions']),
            torch.tensor(res['labels']),
            radii_range,
        )
        curves[name] = (radii, ca)
    
    return curves


# =============================================================================
# Gradient Flow Analysis - Section 1.6
# =============================================================================

def analyze_gradient_flow(
    model: nn.Module,
    x: Tensor,
    adj: Tensor,
    labels: Tensor,
    train_mask: Tensor,
) -> Dict[str, np.ndarray]:
    """
    Analyze gradient magnitude distribution across neurons.
    
    GHN exhibits "sparse but strong" gradients:
    - Strong gradients for small pre-activations
    - Attenuated gradients for large-magnitude neurons
    
    Args:
        model: GNN model
        x: Input features
        adj: Adjacency matrix
        labels: True labels
        train_mask: Training mask
        
    Returns:
        Dictionary with gradient statistics
    """
    model.train()
    
    # Forward pass with gradient tracking
    logits = model(x, adj)
    loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    
    # Collect gradients and pre-activations
    grad_magnitudes = []
    preact_magnitudes = []
    
    for name, param in model.named_parameters():
        if param.grad is None:
            loss.backward(retain_graph=True)
        
        if 'weight' in name and param.grad is not None:
            grad_magnitudes.extend(param.grad.abs().flatten().tolist())
    
    model.zero_grad()
    
    # Compute statistics
    grad_magnitudes = np.array(grad_magnitudes)
    
    return {
        'grad_mean': np.mean(grad_magnitudes),
        'grad_std': np.std(grad_magnitudes),
        'grad_median': np.median(grad_magnitudes),
        'grad_percentiles': np.percentile(grad_magnitudes, [10, 25, 50, 75, 90]),
        'grad_magnitudes': grad_magnitudes,
    }


# =============================================================================
# Margin Analysis
# =============================================================================

def compute_margin_distribution(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
) -> Dict[str, float]:
    """
    Compute classification margin distribution.
    
    γ_i = f_y(x_i) - max_{k≠y} f_k(x_i)
    
    Args:
        logits: Model logits [num_nodes, num_classes]
        labels: True labels [num_nodes]
        mask: Node mask
        
    Returns:
        Dictionary with margin statistics
    """
    logits_masked = logits[mask]
    labels_masked = labels[mask]
    
    # Get score for true class
    true_scores = logits_masked.gather(1, labels_masked.unsqueeze(1)).squeeze()
    
    # Get max score for other classes
    logits_masked_copy = logits_masked.clone()
    logits_masked_copy.scatter_(1, labels_masked.unsqueeze(1), -float('inf'))
    other_max = logits_masked_copy.max(dim=1).values
    
    # Margin
    margins = true_scores - other_max
    
    return {
        'mean': margins.mean().item(),
        'std': margins.std().item(),
        'median': margins.median().item(),
        'min': margins.min().item(),
        'max': margins.max().item(),
        'positive_fraction': (margins > 0).float().mean().item(),
        'margins': margins.detach().cpu().numpy(),
    }


# =============================================================================
# Spectral Norm Analysis - Section 1.8
# =============================================================================

def compute_spectral_norms(model: nn.Module) -> Dict[str, float]:
    """
    Compute spectral norms of all weight matrices.
    
    Args:
        model: GNN model
        
    Returns:
        Dictionary mapping layer names to spectral norms
    """
    norms = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            # Power iteration for spectral norm
            with torch.no_grad():
                W = param.data
                u = torch.randn(W.size(0), device=W.device)
                u = u / u.norm()
                
                for _ in range(10):
                    v = W.T @ u
                    v = v / v.norm()
                    u = W @ v
                    u = u / u.norm()
                
                spectral_norm = (u @ W @ v).item()
                norms[name] = abs(spectral_norm)
    
    # Also compute product (Lipschitz constant for α=1)
    if norms:
        norms['product'] = np.prod(list(norms.values()))
        norms['max'] = max(norms.values())
    
    return norms


def compute_holder_constant(
    model: nn.Module,
    alpha: float,
    num_nodes: int,
    hidden_dims: List[int],
) -> float:
    """
    Compute network Hölder constant C_net.
    
    C_net = Π_{l=1}^L ||W^(l)||_2^α
    
    Note: The full formula includes dimension factors, but for
    comparison purposes we use the simplified version.
    
    Args:
        model: GNN model
        alpha: Hölder exponent
        num_nodes: Number of nodes n
        hidden_dims: Hidden dimensions [d_1, d_2, ...]
        
    Returns:
        C_net value
    """
    spectral_norms = compute_spectral_norms(model)
    
    # Extract weight norms in order
    weight_norms = []
    for name in sorted(spectral_norms.keys()):
        if 'weight' in name and name != 'product' and name != 'max':
            weight_norms.append(spectral_norms[name])
    
    # Compute C_net = Π ||W||_2^α
    c_net = 1.0
    for norm in weight_norms:
        c_net *= norm ** alpha
    
    return c_net
