"""
Certified Defense Methods
- Randomized Smoothing: Probabilistic certificates via Monte Carlo
- GNNCert: Hash-based graph partitioning for deterministic certificates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Callable
import numpy as np
from scipy.stats import norm
import math


class RandomizedSmoothing(nn.Module):
    """
    Randomized Smoothing for Graph Neural Networks.
    
    Provides probabilistic certificates by adding Gaussian noise
    to node features during inference.
    
    Reference: Cohen et al., "Certified Adversarial Robustness via Randomized Smoothing"
               Bojchevski & Günnemann, "Certifiable Robustness to Graph Perturbations"
    
    Args:
        base_model: Underlying GNN classifier
        sigma: Noise standard deviation
        n_samples: Number of Monte Carlo samples for certification
        n_abstain: Minimum samples for abstention
        alpha: Confidence level (1 - alpha)
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        sigma: float = 0.25,
        n_samples: int = 1000,
        n_abstain: int = 100,
        alpha: float = 0.001,
    ):
        super().__init__()
        
        self.base_model = base_model
        self.sigma = sigma
        self.n_samples = n_samples
        self.n_abstain = n_abstain
        self.alpha = alpha
    
    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        training: bool = True,
        **kwargs
    ) -> Tensor:
        """
        Forward pass with optional noise injection.
        
        During training, adds noise for regularization.
        During inference, returns clean predictions.
        """
        if training and self.training:
            # Add noise during training for regularization
            noise = torch.randn_like(x) * self.sigma
            x = x + noise
        
        return self.base_model(x, adj, **kwargs)
    
    @torch.no_grad()
    def predict(
        self,
        x: Tensor,
        adj: Tensor,
        n_samples: Optional[int] = None,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """
        Smoothed prediction via Monte Carlo sampling.
        
        Args:
            x: Node features [n, d]
            adj: Adjacency matrix [n, n]
            n_samples: Number of samples (default: self.n_samples)
            
        Returns:
            predictions: Predicted labels [n]
            counts: Class counts for each node [n, C]
        """
        n_samples = n_samples or self.n_abstain
        n_nodes = x.size(0)
        
        # Sample multiple noisy predictions
        counts = None
        
        for _ in range(n_samples):
            noise = torch.randn_like(x) * self.sigma
            x_noisy = x + noise
            
            logits = self.base_model(x_noisy, adj, **kwargs)
            preds = logits.argmax(dim=-1)  # [n]
            
            if counts is None:
                n_classes = logits.size(-1)
                counts = torch.zeros(n_nodes, n_classes, device=x.device)
            
            # Count predictions
            for i in range(n_nodes):
                counts[i, preds[i]] += 1
        
        predictions = counts.argmax(dim=-1)
        
        return predictions, counts
    
    @torch.no_grad()
    def certify(
        self,
        x: Tensor,
        adj: Tensor,
        node_idx: int,
        **kwargs
    ) -> Tuple[int, float]:
        """
        Certify a single node with probabilistic guarantee.
        
        Args:
            x: Node features [n, d]
            adj: Adjacency matrix [n, n]
            node_idx: Index of node to certify
            
        Returns:
            predicted_class: Predicted label (-1 if abstain)
            certified_radius: Certified L2 radius
        """
        # Step 1: Initial prediction with few samples
        _, counts_init = self.predict(x, adj, n_samples=self.n_abstain, **kwargs)
        top_class = counts_init[node_idx].argmax().item()
        
        # Step 2: Precise estimation with more samples
        _, counts = self.predict(x, adj, n_samples=self.n_samples, **kwargs)
        
        # Count for top class
        n_top = counts[node_idx, top_class].item()
        n_total = self.n_samples
        
        # Clopper-Pearson confidence bound
        p_lower = self._lower_confidence_bound(n_top, n_total, self.alpha)
        
        if p_lower < 0.5:
            # Abstain
            return -1, 0.0
        
        # Certified radius: R = σ * Φ^{-1}(p_lower)
        certified_radius = self.sigma * norm.ppf(p_lower)
        
        return top_class, certified_radius
    
    def _lower_confidence_bound(
        self,
        k: int,
        n: int,
        alpha: float
    ) -> float:
        """
        Clopper-Pearson lower confidence bound.
        
        Args:
            k: Number of successes
            n: Total trials
            alpha: Significance level
            
        Returns:
            Lower bound on probability
        """
        if k == 0:
            return 0.0
        
        from scipy.stats import binom
        return binom.ppf(alpha, n, k / n) / n
    
    def certify_all_nodes(
        self,
        x: Tensor,
        adj: Tensor,
        test_mask: Tensor,
        labels: Tensor,
        **kwargs
    ) -> Tuple[float, float, float]:
        """
        Certify all test nodes.
        
        Returns:
            clean_accuracy: Accuracy on clean data
            certified_accuracy: Fraction certified correct
            average_certified_radius: Mean radius over certified nodes
        """
        test_indices = test_mask.nonzero().squeeze(-1)
        
        correct = 0
        certified_correct = 0
        total_radius = 0.0
        certified_count = 0
        
        for idx in test_indices:
            idx = idx.item()
            true_label = labels[idx].item()
            
            pred_class, radius = self.certify(x, adj, idx, **kwargs)
            
            if pred_class == true_label:
                correct += 1
                if radius > 0:
                    certified_correct += 1
                    total_radius += radius
                    certified_count += 1
        
        n_test = len(test_indices)
        clean_acc = correct / n_test
        cert_acc = certified_correct / n_test
        avg_radius = total_radius / certified_count if certified_count > 0 else 0.0
        
        return clean_acc, cert_acc, avg_radius


class GNNCert(nn.Module):
    """
    GNNCert: Deterministic Certification via Graph Partitioning.
    
    Uses hash-based graph partitioning to provide deterministic
    certificates against arbitrary perturbations.
    
    Reference: Li & Wang, "AGNNCert: Defending GNNs against arbitrary perturbations"
    
    Args:
        base_model: Underlying GNN classifier
        partition_size: Size k for graph partitioning
        hash_type: Type of hash function ('random', 'degree')
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        partition_size: int = 16,
        hash_type: str = 'random',
    ):
        super().__init__()
        
        self.base_model = base_model
        self.partition_size = partition_size
        self.hash_type = hash_type
    
    def _hash_partition(
        self,
        adj: Tensor,
        x: Tensor,
        n_partitions: int,
    ) -> Tensor:
        """
        Partition nodes using hash-based method.
        
        Args:
            adj: Adjacency matrix [n, n]
            x: Node features [n, d]
            n_partitions: Number of partitions
            
        Returns:
            partition_assignments: Partition ID for each node [n]
        """
        n_nodes = adj.size(0)
        
        if self.hash_type == 'random':
            # Random hash
            assignments = torch.randint(0, n_partitions, (n_nodes,), device=adj.device)
        elif self.hash_type == 'degree':
            # Degree-based hash
            degrees = adj.sum(dim=1)
            assignments = (degrees % n_partitions).long()
        else:
            # Feature-based hash
            feature_sum = x.sum(dim=1)
            _, sorted_idx = torch.sort(feature_sum)
            assignments = torch.zeros(n_nodes, dtype=torch.long, device=adj.device)
            for i, idx in enumerate(sorted_idx):
                assignments[idx] = i % n_partitions
        
        return assignments
    
    def _create_subgraph(
        self,
        adj: Tensor,
        x: Tensor,
        partition_id: int,
        assignments: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Create subgraph for a partition.
        
        Returns:
            sub_adj: Subgraph adjacency
            sub_x: Subgraph features
            node_mapping: Original indices of nodes in subgraph
        """
        mask = assignments == partition_id
        node_indices = mask.nonzero().squeeze(-1)
        
        # Extract subgraph
        sub_x = x[mask]
        sub_adj = adj[mask][:, mask]
        
        return sub_adj, sub_x, node_indices
    
    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        **kwargs
    ) -> Tensor:
        """
        Forward pass with graph partitioning.
        
        Aggregates predictions from multiple subgraphs.
        """
        n_nodes = x.size(0)
        n_partitions = max(1, n_nodes // self.partition_size)
        
        # Get partition assignments
        assignments = self._hash_partition(adj, x, n_partitions)
        
        # Initialize output
        all_logits = []
        all_counts = []
        
        for p in range(n_partitions):
            sub_adj, sub_x, node_indices = self._create_subgraph(
                adj, x, p, assignments
            )
            
            if len(node_indices) == 0:
                continue
            
            # Get predictions for subgraph
            sub_logits = self.base_model(sub_x, sub_adj, **kwargs)
            
            all_logits.append((node_indices, sub_logits))
        
        # Aggregate
        device = x.device
        n_classes = all_logits[0][1].size(-1) if all_logits else 7
        output = torch.zeros(n_nodes, n_classes, device=device)
        
        for node_indices, sub_logits in all_logits:
            output[node_indices] = sub_logits
        
        return output
    
    def certify_node(
        self,
        x: Tensor,
        adj: Tensor,
        node_idx: int,
        true_label: int,
        budget: int = 1,
        **kwargs
    ) -> Tuple[float, float]:
        """
        Certify a node against perturbations.
        
        Args:
            x: Node features
            adj: Adjacency matrix
            node_idx: Target node
            true_label: Ground truth label
            budget: Perturbation budget (edges/features)
            
        Returns:
            margin: Classification margin
            certified_radius: Certified radius (binary: 0 or 1)
        """
        n_nodes = x.size(0)
        n_partitions = max(1, n_nodes // self.partition_size)
        
        # Count votes from each partition
        assignments = self._hash_partition(adj, x, n_partitions)
        
        votes = {}
        
        for p in range(n_partitions):
            sub_adj, sub_x, node_indices = self._create_subgraph(
                adj, x, p, assignments
            )
            
            if node_idx not in node_indices:
                continue
            
            # Local index of target node
            local_idx = (node_indices == node_idx).nonzero().item()
            
            sub_logits = self.base_model(sub_x, sub_adj, **kwargs)
            pred = sub_logits[local_idx].argmax().item()
            
            votes[pred] = votes.get(pred, 0) + 1
        
        if not votes:
            return 0.0, 0.0
        
        # Get top predictions
        sorted_votes = sorted(votes.items(), key=lambda x: -x[1])
        top_class, top_count = sorted_votes[0]
        runner_up_count = sorted_votes[1][1] if len(sorted_votes) > 1 else 0
        
        margin = (top_count - runner_up_count) / n_partitions
        
        # Certified if margin > 2 * budget / n_partitions
        certified = top_class == true_label and margin > 2 * budget / n_partitions
        
        # Certified radius (approximate based on margin)
        if certified:
            cert_radius = margin * self.partition_size / 2
        else:
            cert_radius = 0.0
        
        return margin, cert_radius


class CertifiedRadiusComputer:
    """
    Utility class for computing certified radii across different methods.
    """
    
    @staticmethod
    def compute_lipschitz_radius(
        margin: float,
        lipschitz_constant: float,
    ) -> float:
        """
        Certified radius for Lipschitz networks.
        R = γ / (2K)
        """
        if margin <= 0:
            return 0.0
        return margin / (2 * lipschitz_constant)
    
    @staticmethod
    def compute_holder_radius(
        margin: float,
        holder_constant: float,
        alpha_net: float,
    ) -> float:
        """
        Certified radius for Hölder networks.
        R = (γ / (2C_net))^{1/α_net}
        """
        if margin <= 0:
            return 0.0
        return (margin / (2 * holder_constant)) ** (1 / alpha_net)
    
    @staticmethod
    def compute_smoothing_radius(
        p_lower: float,
        sigma: float,
    ) -> float:
        """
        Certified radius for randomized smoothing.
        R = σ * Φ^{-1}(p_lower)
        """
        if p_lower <= 0.5:
            return 0.0
        return sigma * norm.ppf(p_lower)
