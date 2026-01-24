"""
Empirical Defense Methods (No Formal Certificates)
- GNNGuard: Prunes adversarial edges via neighbor similarity
- RobustGCN: Uses Gaussian distributions to absorb perturbations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math


class GNNGuardLayer(nn.Module):
    """
    GNNGuard Layer - Attention-based defense against adversarial edges.
    
    Key idea: Learn to down-weight edges between dissimilar nodes,
    which are likely adversarial.
    
    Reference: Zhang & Zitnik, "GNNGuard: Defending GNNs against Adversarial Attacks"
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        prune_threshold: Threshold for edge pruning (0 = no pruning)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prune_threshold: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.prune_threshold = prune_threshold
        
        # Main transformation
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Attention mechanism for edge filtering
        self.attention = nn.Linear(in_features * 2, 1)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)
    
    def compute_neighbor_similarity(
        self,
        x: Tensor,
        adj: Tensor,
    ) -> Tensor:
        """
        Compute cosine similarity between connected nodes.
        
        Returns:
            similarity_adj: Adjacency weighted by similarity [n, n]
        """
        # Normalize features
        x_norm = F.normalize(x, p=2, dim=-1)
        
        # Cosine similarity
        sim = torch.mm(x_norm, x_norm.t())
        
        # Apply to adjacency
        similarity_adj = adj * sim
        
        return similarity_adj
    
    def compute_attention_weights(
        self,
        x: Tensor,
        adj: Tensor,
    ) -> Tensor:
        """
        Compute attention weights for edges.
        
        Down-weights edges between dissimilar nodes.
        """
        n = x.size(0)
        device = x.device
        
        # Get edge indices
        edge_mask = adj > 0
        
        # For each edge (i, j), compute attention
        attention_scores = torch.zeros(n, n, device=device)
        
        for i in range(n):
            neighbors = edge_mask[i].nonzero().squeeze(-1)
            if len(neighbors) == 0:
                continue
            
            # Concatenate source and target features
            x_i = x[i].unsqueeze(0).expand(len(neighbors), -1)
            x_j = x[neighbors]
            
            concat = torch.cat([x_i, x_j], dim=-1)
            scores = self.attention(concat).squeeze(-1)
            
            attention_scores[i, neighbors] = scores
        
        # Softmax over neighbors
        attention_weights = torch.where(
            edge_mask,
            F.softmax(attention_scores.masked_fill(~edge_mask, -1e9), dim=-1),
            torch.zeros_like(attention_scores)
        )
        
        return attention_weights
    
    @staticmethod
    def normalize_adjacency(adj: Tensor) -> Tensor:
        n = adj.size(0)
        adj = adj + torch.eye(n, device=adj.device)
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        return degree_inv_sqrt.unsqueeze(1) * adj * degree_inv_sqrt.unsqueeze(0)
    
    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        adj_normalized: bool = False,
    ) -> Tensor:
        """
        Forward pass with defense mechanism.
        """
        n = x.size(0)
        device = x.device
        
        # Add self-loops
        adj_with_self = adj + torch.eye(n, device=device)
        
        # Compute neighbor similarity
        similarity = self.compute_neighbor_similarity(x, adj_with_self)
        
        # Prune low-similarity edges
        if self.prune_threshold > 0:
            mask = similarity > self.prune_threshold
            adj_filtered = adj_with_self * mask.float()
        else:
            adj_filtered = adj_with_self * (similarity > 0).float()
        
        # Re-weight using similarity
        adj_weighted = adj_filtered * F.relu(similarity)
        
        # Normalize
        degree = adj_weighted.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        adj_norm = degree_inv_sqrt.unsqueeze(1) * adj_weighted * degree_inv_sqrt.unsqueeze(0)
        
        # Standard GCN operation
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj_norm, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class GNNGuard(nn.Module):
    """
    GNNGuard - Full network with edge filtering defense.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 64,
        out_features: int = 7,
        num_layers: int = 2,
        dropout: float = 0.5,
        prune_threshold: float = 0.1,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        self.layers.append(GNNGuardLayer(
            in_features, hidden_features, prune_threshold
        ))
        
        for _ in range(num_layers - 2):
            self.layers.append(GNNGuardLayer(
                hidden_features, hidden_features, prune_threshold
            ))
        
        self.layers.append(GNNGuardLayer(
            hidden_features, out_features, prune_threshold
        ))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, adj: Tensor, **kwargs) -> Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adj)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.layers[-1](x, adj)
        return x


class GaussianLayer(nn.Module):
    """
    Gaussian-based GCN layer for RobustGCN.
    
    Represents node features as Gaussian distributions (mean, variance)
    to absorb perturbations through variance estimation.
    
    Reference: Zhu et al., "Robust GCN against Adversarial Attacks"
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        
        # Mean transformation
        self.weight_mean = nn.Parameter(torch.empty(in_features, out_features))
        # Variance transformation
        self.weight_var = nn.Parameter(torch.empty(in_features, out_features))
        
        if bias:
            self.bias_mean = nn.Parameter(torch.empty(out_features))
            self.bias_var = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias_mean', None)
            self.register_parameter('bias_var', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_mean)
        nn.init.xavier_uniform_(self.weight_var)
        if self.bias_mean is not None:
            nn.init.zeros_(self.bias_mean)
            nn.init.zeros_(self.bias_var)
    
    @staticmethod
    def normalize_adjacency(adj: Tensor) -> Tensor:
        n = adj.size(0)
        adj = adj + torch.eye(n, device=adj.device)
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        return degree_inv_sqrt.unsqueeze(1) * adj * degree_inv_sqrt.unsqueeze(0)
    
    def forward(
        self,
        mean: Tensor,
        variance: Tensor,
        adj: Tensor,
        adj_normalized: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with Gaussian representations.
        
        Args:
            mean: Mean features [n, d]
            variance: Variance features [n, d]
            adj: Adjacency matrix [n, n]
            
        Returns:
            new_mean: Updated mean [n, d']
            new_variance: Updated variance [n, d']
        """
        if not adj_normalized:
            adj = self.normalize_adjacency(adj)
        
        # Transform mean
        mean_support = torch.matmul(mean, self.weight_mean)
        new_mean = torch.matmul(adj, mean_support)
        
        if self.bias_mean is not None:
            new_mean = new_mean + self.bias_mean
        
        # Transform variance
        # Variance propagates through adjacency squared
        var_support = torch.matmul(variance, self.weight_var.pow(2))
        adj_sq = adj.pow(2)
        new_variance = torch.matmul(adj_sq, var_support)
        
        if self.bias_var is not None:
            new_variance = new_variance + self.bias_var.pow(2)
        
        # Ensure variance is positive
        new_variance = F.softplus(new_variance) + 1e-6
        
        return new_mean, new_variance


class RobustGCN(nn.Module):
    """
    Robust Graph Convolutional Network.
    
    Uses Gaussian distributions to model node features,
    allowing the network to absorb perturbations through
    variance estimation.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 64,
        out_features: int = 7,
        num_layers: int = 2,
        dropout: float = 0.5,
        gamma: float = 1.0,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.gamma = gamma
        
        # Initial variance
        self.init_var = nn.Parameter(torch.ones(1) * 0.1)
        
        self.layers = nn.ModuleList()
        self.layers.append(GaussianLayer(in_features, hidden_features))
        
        for _ in range(num_layers - 2):
            self.layers.append(GaussianLayer(hidden_features, hidden_features))
        
        self.layers.append(GaussianLayer(hidden_features, out_features))
        
        self.dropout = nn.Dropout(dropout)
    
    def _attention_weighted_aggregation(
        self,
        mean: Tensor,
        variance: Tensor,
    ) -> Tensor:
        """
        Attention-based aggregation using variance.
        Down-weights high-variance (uncertain) features.
        """
        # Attention weights inversely proportional to variance
        attention = 1.0 / (variance + 1e-6)
        attention = attention / attention.sum(dim=-1, keepdim=True)
        
        return mean * attention
    
    def forward(self, x: Tensor, adj: Tensor, **kwargs) -> Tensor:
        """
        Forward pass with Gaussian propagation.
        """
        # Initialize variance
        mean = x
        variance = self.init_var.expand_as(x)
        
        # Normalize adjacency once
        adj_norm = GaussianLayer.normalize_adjacency(adj)
        
        for i, layer in enumerate(self.layers[:-1]):
            mean, variance = layer(mean, variance, adj_norm, adj_normalized=True)
            
            # ReLU on mean, keep variance positive
            mean = F.relu(mean)
            
            # Sample from Gaussian during training for regularization
            if self.training:
                eps = torch.randn_like(mean)
                mean = mean + self.gamma * torch.sqrt(variance) * eps
            
            mean = self.dropout(mean)
        
        # Final layer
        mean, variance = self.layers[-1](mean, variance, adj_norm, adj_normalized=True)
        
        # Return mean as logits
        return mean
    
    def get_uncertainty(
        self,
        x: Tensor,
        adj: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get predictions with uncertainty estimates.
        
        Returns:
            logits: Prediction logits [n, C]
            uncertainty: Variance per node [n, C]
        """
        mean = x
        variance = self.init_var.expand_as(x)
        
        adj_norm = GaussianLayer.normalize_adjacency(adj)
        
        for i, layer in enumerate(self.layers[:-1]):
            mean, variance = layer(mean, variance, adj_norm, adj_normalized=True)
            mean = F.relu(mean)
        
        mean, variance = self.layers[-1](mean, variance, adj_norm, adj_normalized=True)
        
        return mean, variance


class AdaptiveAggregation(nn.Module):
    """
    Adaptive aggregation layer that learns to filter noisy neighbors.
    Can be combined with any GNN backbone.
    """
    
    def __init__(
        self,
        in_features: int,
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.temperature = temperature
        
        # Learnable query for attention
        self.query = nn.Linear(in_features, in_features)
        self.key = nn.Linear(in_features, in_features)
    
    def forward(
        self,
        x: Tensor,
        adj: Tensor,
    ) -> Tensor:
        """
        Compute adaptive adjacency weights.
        
        Returns:
            weighted_adj: Attention-weighted adjacency [n, n]
        """
        Q = self.query(x)
        K = self.key(x)
        
        # Scaled dot-product attention
        attention = torch.matmul(Q, K.t()) / math.sqrt(Q.size(-1))
        
        # Mask with adjacency
        n = x.size(0)
        adj_with_self = adj + torch.eye(n, device=adj.device)
        attention = attention.masked_fill(adj_with_self == 0, -1e9)
        
        # Temperature scaling
        attention = attention / self.temperature
        
        # Softmax
        attention_weights = F.softmax(attention, dim=-1)
        
        return attention_weights
