"""
Adversarial Attack Implementations for Graph Neural Networks

Implements:
1. Feature attacks: PGD, FGSM
2. Structural attacks: Nettack (targeted), Metattack (global poisoning)
3. Random perturbations: Bernoulli edge deletion

References:
- Nettack: Zügner et al., "Adversarial Attacks on Neural Networks for Graph Data" (KDD 2018)
- Metattack: Zügner & Günnemann, "Adversarial Attacks on Graph Neural Networks via Meta Learning" (ICLR 2019)
- PGD: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (ICLR 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List, Dict
import numpy as np
from tqdm import tqdm


# =============================================================================
# Feature Attacks
# =============================================================================

class PGDAttack:
    """
    Projected Gradient Descent attack for node features.
    
    Reference: Madry et al., "Towards Deep Learning Models Resistant to 
               Adversarial Attacks" (ICLR 2018)
    
    Args:
        model: Target model
        epsilon: Perturbation budget (ℓ2 norm)
        alpha: Step size
        num_steps: Number of PGD iterations
        num_restarts: Number of random restarts
        random_start: Whether to use random initialization
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.1,
        alpha: float = None,
        num_steps: int = 40,
        num_restarts: int = 5,
        random_start: bool = True,
    ):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha if alpha is not None else epsilon / 10
        self.num_steps = num_steps
        self.num_restarts = num_restarts
        self.random_start = random_start
    
    def attack(
        self,
        x: Tensor,
        adj: Tensor,
        labels: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        """
        Perform PGD attack on node features.
        
        Args:
            x: Node features [num_nodes, num_features]
            adj: Normalized adjacency matrix
            labels: True labels
            target_mask: Mask for nodes to attack
            
        Returns:
            Perturbed features achieving maximum loss
        """
        best_x_adv = x.clone()
        best_loss = -float('inf')
        
        for restart in range(self.num_restarts):
            x_adv = x.clone().detach()
            
            if self.random_start:
                # Random initialization within epsilon ball (ℓ2)
                noise = torch.randn_like(x_adv)
                noise = noise / noise.norm(dim=-1, keepdim=True) * self.epsilon
                noise = noise * torch.rand(x_adv.size(0), 1, device=x_adv.device)
                x_adv = x_adv + noise
            
            for step in range(self.num_steps):
                x_adv.requires_grad = True
                
                logits = self.model(x_adv, adj)
                loss = F.cross_entropy(logits[target_mask], labels[target_mask])
                
                grad = torch.autograd.grad(loss, x_adv)[0]
                
                # Gradient ascent step (maximize loss)
                x_adv = x_adv.detach() + self.alpha * grad / (grad.norm(dim=-1, keepdim=True) + 1e-8)
                
                # Project back to ℓ2 epsilon ball
                delta = x_adv - x
                delta_norm = delta.norm(dim=-1, keepdim=True)
                delta = torch.where(
                    delta_norm > self.epsilon,
                    delta * self.epsilon / delta_norm,
                    delta
                )
                x_adv = x + delta
            
            # Check if this restart achieved higher loss
            with torch.no_grad():
                logits = self.model(x_adv, adj)
                final_loss = F.cross_entropy(logits[target_mask], labels[target_mask])
                
                if final_loss > best_loss:
                    best_loss = final_loss
                    best_x_adv = x_adv.clone()
        
        return best_x_adv.detach()


class FGSMAttack:
    """
    Fast Gradient Sign Method attack.
    
    Reference: Goodfellow et al., "Explaining and Harnessing 
               Adversarial Examples" (ICLR 2015)
    """
    
    def __init__(self, model: nn.Module, epsilon: float = 0.1):
        self.model = model
        self.epsilon = epsilon
    
    def attack(
        self,
        x: Tensor,
        adj: Tensor,
        labels: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        """Perform FGSM attack (single-step PGD)."""
        x_adv = x.clone().detach()
        x_adv.requires_grad = True
        
        logits = self.model(x_adv, adj)
        loss = F.cross_entropy(logits[target_mask], labels[target_mask])
        
        grad = torch.autograd.grad(loss, x_adv)[0]
        
        # ℓ2 normalized gradient step
        x_adv = x_adv.detach() + self.epsilon * grad / (grad.norm(dim=-1, keepdim=True) + 1e-8)
        
        return x_adv


# =============================================================================
# Structural Attacks
# =============================================================================

class Nettack:
    """
    Nettack: Targeted adversarial attack on graph structure.
    
    Reference: Zügner et al., "Adversarial Attacks on Neural Networks 
               for Graph Data" (KDD 2018)
    
    Simplified implementation focusing on edge perturbations.
    
    Args:
        model: Target model (surrogate)
        num_perturbations: Budget Δ (number of edge modifications)
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_perturbations: Optional[int] = None,
    ):
        self.model = model
        self.num_perturbations = num_perturbations
    
    def attack(
        self,
        x: Tensor,
        adj: Tensor,
        labels: Tensor,
        target_node: int,
        budget: Optional[int] = None,
    ) -> Tensor:
        """
        Perform Nettack on a single target node.
        
        Args:
            x: Node features
            adj: Dense adjacency matrix (unnormalized)
            labels: True labels
            target_node: Index of node to attack
            budget: Perturbation budget Δ (default: deg(target) + 2)
            
        Returns:
            Perturbed (normalized) adjacency matrix
        """
        num_nodes = adj.size(0)
        
        # Default budget: deg(v) + 2
        if budget is None:
            budget = self.num_perturbations or (int(adj[target_node].sum().item()) + 2)
        
        # Work with unnormalized adjacency
        adj_pert = adj.clone()
        
        # Greedily select edge perturbations
        for _ in range(budget):
            best_flip = None
            best_score = -float('inf')
            
            # Compute gradient-based scores for candidate edges
            candidates = torch.randperm(num_nodes)[:min(100, num_nodes)]
            
            for j in candidates.tolist():
                if j == target_node:
                    continue
                
                # Compute score for flipping edge (target, j)
                adj_temp = adj_pert.clone()
                
                # Flip edge
                if adj_temp[target_node, j] > 0.5:
                    adj_temp[target_node, j] = 0
                    adj_temp[j, target_node] = 0
                else:
                    adj_temp[target_node, j] = 1
                    adj_temp[j, target_node] = 1
                
                # Normalize
                adj_norm = self._normalize_adj(adj_temp)
                
                # Score: increase in loss for target node
                with torch.no_grad():
                    logits = self.model(x, adj_norm)
                    # Score: negative margin (want to decrease correct class score)
                    true_class = labels[target_node].item()
                    score = -logits[target_node, true_class].item()
                    score += logits[target_node].max().item()
                
                if score > best_score:
                    best_score = score
                    best_flip = j
            
            # Apply best flip
            if best_flip is not None:
                if adj_pert[target_node, best_flip] > 0.5:
                    adj_pert[target_node, best_flip] = 0
                    adj_pert[best_flip, target_node] = 0
                else:
                    adj_pert[target_node, best_flip] = 1
                    adj_pert[best_flip, target_node] = 1
        
        return self._normalize_adj(adj_pert)
    
    def _normalize_adj(self, adj: Tensor) -> Tensor:
        """Symmetric normalization: D^{-1/2} A D^{-1/2}"""
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)


class Metattack:
    """
    Metattack: Global graph poisoning via meta-gradients.
    
    Reference: Zügner & Günnemann, "Adversarial Attacks on Graph Neural 
               Networks via Meta Learning" (ICLR 2019)
    
    Simplified greedy approximation.
    
    Args:
        model: Target model (used as surrogate)
        budget_pct: Fraction of edges to modify
    """
    
    def __init__(
        self,
        model: nn.Module,
        budget_pct: float = 0.05,
    ):
        self.model = model
        self.budget_pct = budget_pct
    
    def attack(
        self,
        x: Tensor,
        adj: Tensor,
        labels: Tensor,
        train_mask: Tensor,
    ) -> Tensor:
        """
        Perform Metattack (global poisoning).
        
        Args:
            x: Node features
            adj: Dense adjacency matrix (unnormalized)
            labels: Labels for all nodes
            train_mask: Training node mask
            
        Returns:
            Poisoned (normalized) adjacency matrix
        """
        num_edges = int(adj.sum().item() / 2)
        budget = int(num_edges * self.budget_pct)
        
        adj_pert = adj.clone()
        num_nodes = adj.size(0)
        
        # Greedy edge perturbation
        for _ in tqdm(range(budget), desc="Metattack", leave=False):
            best_flip = None
            best_score = -float('inf')
            
            # Sample candidate edge pairs
            candidates_i = torch.randint(0, num_nodes, (50,))
            candidates_j = torch.randint(0, num_nodes, (50,))
            
            for i, j in zip(candidates_i.tolist(), candidates_j.tolist()):
                if i >= j:
                    continue
                
                adj_temp = adj_pert.clone()
                
                # Flip edge
                if adj_temp[i, j] > 0.5:
                    adj_temp[i, j] = 0
                    adj_temp[j, i] = 0
                else:
                    adj_temp[i, j] = 1
                    adj_temp[j, i] = 1
                
                adj_norm = self._normalize_adj(adj_temp)
                
                # Score: training loss (want to increase)
                with torch.no_grad():
                    logits = self.model(x, adj_norm)
                    score = F.cross_entropy(logits[train_mask], labels[train_mask]).item()
                
                if score > best_score:
                    best_score = score
                    best_flip = (i, j)
            
            if best_flip is not None:
                i, j = best_flip
                if adj_pert[i, j] > 0.5:
                    adj_pert[i, j] = 0
                    adj_pert[j, i] = 0
                else:
                    adj_pert[i, j] = 1
                    adj_pert[j, i] = 1
        
        return self._normalize_adj(adj_pert)
    
    def _normalize_adj(self, adj: Tensor) -> Tensor:
        """Symmetric normalization."""
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)


class BernoulliEdgeDeletion:
    """
    Random edge deletion following Bernoulli model.
    
    Each existing edge is independently deleted with probability p.
    Used to verify Theorem 3.5 (probabilistic stability bound).
    
    Args:
        deletion_prob: Probability p of deleting each edge
    """
    
    def __init__(self, deletion_prob: float = 0.1):
        self.deletion_prob = deletion_prob
    
    def perturb(
        self,
        adj: Tensor,
        seed: Optional[int] = None,
    ) -> Tensor:
        """
        Apply Bernoulli edge deletion.
        
        Args:
            adj: Dense adjacency matrix (unnormalized, symmetric)
            seed: Random seed for reproducibility
            
        Returns:
            Perturbed (normalized) adjacency matrix
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        num_nodes = adj.size(0)
        
        # Create mask for existing edges (upper triangular)
        edge_mask = (adj > 0.5).float()
        upper_mask = torch.triu(edge_mask, diagonal=1)
        
        # Sample deletion mask
        delete_mask = torch.bernoulli(
            torch.ones_like(upper_mask) * self.deletion_prob
        ) * upper_mask
        
        # Make symmetric
        delete_mask = delete_mask + delete_mask.t()
        
        # Apply deletion
        adj_pert = adj.clone()
        adj_pert[delete_mask > 0.5] = 0
        
        return self._normalize_adj(adj_pert)
    
    def _normalize_adj(self, adj: Tensor) -> Tensor:
        """Symmetric normalization."""
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)


# =============================================================================
# Attack Evaluation Utilities
# =============================================================================

def evaluate_under_attack(
    model: nn.Module,
    x: Tensor,
    adj: Tensor,
    labels: Tensor,
    test_mask: Tensor,
    attack: str,
    attack_params: Dict,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model under specified attack.
    
    Args:
        model: Model to evaluate
        x: Node features
        adj: Adjacency matrix
        labels: True labels
        test_mask: Test node mask
        attack: Attack type ('pgd', 'fgsm', 'nettack', 'metattack', 'bernoulli')
        attack_params: Attack-specific parameters
        device: Computation device
        
    Returns:
        Dictionary with clean and adversarial accuracy
    """
    model.eval()
    model.to(device)
    x = x.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    test_mask = test_mask.to(device)
    
    # Clean accuracy
    with torch.no_grad():
        logits_clean = model(x, adj)
        clean_acc = (logits_clean[test_mask].argmax(dim=-1) == labels[test_mask]).float().mean().item()
    
    # Apply attack
    if attack == 'pgd':
        attacker = PGDAttack(model, **attack_params)
        x_adv = attacker.attack(x, adj, labels, test_mask)
        with torch.no_grad():
            logits_adv = model(x_adv, adj)
            adv_acc = (logits_adv[test_mask].argmax(dim=-1) == labels[test_mask]).float().mean().item()
    
    elif attack == 'fgsm':
        attacker = FGSMAttack(model, **attack_params)
        x_adv = attacker.attack(x, adj, labels, test_mask)
        with torch.no_grad():
            logits_adv = model(x_adv, adj)
            adv_acc = (logits_adv[test_mask].argmax(dim=-1) == labels[test_mask]).float().mean().item()
    
    elif attack == 'bernoulli':
        perturber = BernoulliEdgeDeletion(**attack_params)
        # Average over multiple perturbations
        adv_accs = []
        for seed in range(10):
            adj_pert = perturber.perturb(adj, seed=seed)
            with torch.no_grad():
                logits_adv = model(x, adj_pert)
                acc = (logits_adv[test_mask].argmax(dim=-1) == labels[test_mask]).float().mean().item()
                adv_accs.append(acc)
        adv_acc = np.mean(adv_accs)
        adv_std = np.std(adv_accs)
        return {
            'clean_accuracy': clean_acc,
            'adversarial_accuracy': adv_acc,
            'adversarial_std': adv_std,
        }
    
    else:
        raise ValueError(f"Unknown attack: {attack}")
    
    return {
        'clean_accuracy': clean_acc,
        'adversarial_accuracy': adv_acc,
    }


def run_nettack_evaluation(
    model: nn.Module,
    x: Tensor,
    adj_unnorm: Tensor,
    labels: Tensor,
    test_mask: Tensor,
    num_targets: int = 100,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Evaluate model under Nettack on random target nodes.
    
    Args:
        model: Model to evaluate
        x: Node features
        adj_unnorm: Unnormalized adjacency matrix
        labels: True labels
        test_mask: Test node mask
        num_targets: Number of target nodes to attack
        device: Computation device
        
    Returns:
        Dictionary with clean and adversarial accuracy
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    x = x.to(device)
    adj_unnorm = adj_unnorm.to(device)
    labels = labels.to(device)
    
    # Get test nodes that are correctly classified
    adj_norm = _normalize_adj(adj_unnorm, device)
    with torch.no_grad():
        logits = model(x, adj_norm)
        preds = logits.argmax(dim=-1)
    
    test_indices = test_mask.nonzero().squeeze(-1)
    correct_test = test_indices[(preds[test_indices] == labels[test_indices])]
    
    # Sample targets
    if len(correct_test) > num_targets:
        target_indices = correct_test[torch.randperm(len(correct_test))[:num_targets]]
    else:
        target_indices = correct_test
    
    # Attack each target
    attacker = Nettack(model)
    successes = 0
    
    for target in tqdm(target_indices.tolist(), desc="Nettack", leave=False):
        budget = int(adj_unnorm[target].sum().item()) + 2
        adj_pert = attacker.attack(x, adj_unnorm, labels, target, budget=budget)
        
        with torch.no_grad():
            logits_adv = model(x, adj_pert)
            if logits_adv[target].argmax().item() != labels[target].item():
                successes += 1
    
    return {
        'clean_accuracy': len(target_indices) / len(target_indices),  # All targets were clean-correct
        'adversarial_accuracy': 1 - successes / len(target_indices),
        'num_targets': len(target_indices),
    }


def _normalize_adj(adj: Tensor, device: torch.device) -> Tensor:
    """Symmetric normalization helper."""
    adj = adj + torch.eye(adj.size(0), device=device)
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
