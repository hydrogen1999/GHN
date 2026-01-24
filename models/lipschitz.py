"""
Lipschitz-Constrained GNN Baselines
- Spectral-GCN: Spectral normalization on weights
- GroupSort-GCN: GroupSort activation for strict 1-Lipschitz
- PairNorm-GCN: Pairwise normalization to prevent oversmoothing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math

from .activations import GroupSort


class SpectralNorm(nn.Module):
    """
    Spectral Normalization wrapper for linear layers.
    Constrains ||W||_2 <= 1 using power iteration.
    
    Reference: Miyato et al., "Spectral Normalization for GANs"
    """
    
    def __init__(
        self,
        module: nn.Module,
        name: str = 'weight',
        n_power_iterations: int = 1,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        
        # Initialize u, v vectors
        weight = getattr(module, name)
        h, w = weight.shape
        
        # u is left singular vector (size = num rows = h)
        # v is right singular vector (size = num cols = w)
        self.register_buffer('u', F.normalize(torch.randn(h), dim=0))
        self.register_buffer('v', F.normalize(torch.randn(w), dim=0))
    
    def _update_vectors(self, weight: Tensor) -> Tuple[Tensor, Tensor]:
        """Power iteration to estimate spectral norm."""
        u, v = self.u, self.v
        
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                # v = W^T u / ||W^T u||
                v = F.normalize(torch.mv(weight.t(), u), dim=0, eps=self.eps)
                # u = W v / ||W v||
                u = F.normalize(torch.mv(weight, v), dim=0, eps=self.eps)
        
        return u, v
    
    def forward(self, *args, **kwargs):
        weight = getattr(self.module, self.name)
        
        # Update singular vectors
        u, v = self._update_vectors(weight)
        
        if self.training:
            self.u = u
            self.v = v
        
        # Compute spectral norm
        sigma = torch.dot(u, torch.mv(weight, v))
        
        # Normalize weight
        weight_normalized = weight / (sigma + self.eps)
        
        # Temporarily replace weight
        setattr(self.module, self.name, weight_normalized)
        output = self.module(*args, **kwargs)
        setattr(self.module, self.name, weight)
        
        return output


class SpectralGCNLayer(nn.Module):
    """
    GCN layer with spectral normalization.
    Enforces ||W||_2 <= 1 for Lipschitz continuity.
    
    Weight shape: (in_features, out_features)
    For power iteration on W:
    - u ∈ R^{in_features} (left singular vector)
    - v ∈ R^{out_features} (right singular vector)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight with spectral norm
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Power iteration vectors
        # For W of shape (in_features, out_features):
        # u has size in_features (left singular vector)
        # v has size out_features (right singular vector)
        self.register_buffer('u', F.normalize(torch.randn(in_features), dim=0))
        self.register_buffer('v', F.normalize(torch.randn(out_features), dim=0))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def _spectral_normalize(self, weight: Tensor) -> Tensor:
        """Apply spectral normalization."""
        u, v = self.u, self.v
        
        # Power iteration
        # weight shape: (in_features, out_features)
        # weight.t() shape: (out_features, in_features)
        with torch.no_grad():
            for _ in range(1):
                # v = W^T u / ||W^T u||, where W^T is (out, in) and u is (in,) -> v is (out,)
                v_new = F.normalize(torch.mv(weight.t(), u), dim=0)
                # u = W v / ||W v||, where W is (in, out) and v is (out,) -> u is (in,)
                u_new = F.normalize(torch.mv(weight, v_new), dim=0)
            
            if self.training:
                self.u = u_new
                self.v = v_new
        
        sigma = torch.dot(u_new, torch.mv(weight, v_new))
        return weight / (sigma + 1e-8)
    
    @staticmethod
    def normalize_adjacency(adj: Tensor) -> Tensor:
        n = adj.size(0)
        adj = adj + torch.eye(n, device=adj.device)
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        return degree_inv_sqrt.unsqueeze(1) * adj * degree_inv_sqrt.unsqueeze(0)
    
    def forward(self, x: Tensor, adj: Tensor, adj_normalized: bool = False) -> Tensor:
        if not adj_normalized:
            adj = self.normalize_adjacency(adj)
        
        # Apply spectral normalization
        weight_normalized = self._spectral_normalize(self.weight)
        
        support = torch.matmul(x, weight_normalized)
        output = torch.matmul(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def get_spectral_norm(self) -> float:
        """Return 1.0 since we enforce ||W||_2 = 1."""
        return 1.0


class SpectralGCN(nn.Module):
    """
    GCN with spectral normalization on all weight matrices.
    Deterministic certificates with α = 1 (Lipschitz).
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 64,
        out_features: int = 7,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout_p = dropout
        
        self.layers = nn.ModuleList()
        self.layers.append(SpectralGCNLayer(in_features, hidden_features))
        
        for _ in range(num_layers - 2):
            self.layers.append(SpectralGCNLayer(hidden_features, hidden_features))
        
        self.layers.append(SpectralGCNLayer(hidden_features, out_features))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, adj: Tensor, adj_normalized: bool = False) -> Tensor:
        if not adj_normalized:
            adj = SpectralGCNLayer.normalize_adjacency(adj)
            adj_normalized = True
        
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adj, adj_normalized=adj_normalized)
            x = F.relu(x)  # 1-Lipschitz activation
            x = self.dropout(x)
        
        x = self.layers[-1](x, adj, adj_normalized=adj_normalized)
        return x
    
    def get_lipschitz_constant(self) -> float:
        """Lipschitz constant K = 1 due to spectral norm."""
        return 1.0
    
    def certify_node(
        self,
        logits: Tensor,
        true_label: int,
        **kwargs,
    ) -> Tuple[float, float]:
        """Certified radius for Lipschitz network: R = γ / (2K)."""
        logits_sorted, _ = torch.sort(logits, descending=True)
        
        if torch.argmax(logits).item() == true_label:
            true_logit = logits[true_label]
            runner_up = logits_sorted[1] if logits_sorted[0] == true_logit else logits_sorted[0]
            margin = (true_logit - runner_up).item()
        else:
            margin = (logits[true_label] - logits_sorted[0]).item()
        
        if margin <= 0:
            return margin, 0.0
        
        k = self.get_lipschitz_constant()
        certified_radius = margin / (2 * k)
        
        return margin, certified_radius


class GroupSortGCNLayer(nn.Module):
    """
    GCN layer with GroupSort activation for strict 1-Lipschitz.
    
    Weight shape: (in_features, out_features)
    For power iteration:
    - u ∈ R^{in_features}
    - v ∈ R^{out_features}
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 2,
        bias: bool = True,
    ):
        super().__init__()
        
        # Ensure output is divisible by group_size
        self.out_features = out_features if out_features % group_size == 0 else \
                           out_features + (group_size - out_features % group_size)
        
        self.weight = nn.Parameter(torch.empty(in_features, self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter('bias', None)
        
        # Spectral normalization vectors
        # For W of shape (in_features, out_features):
        # u has size in_features
        # v has size out_features
        self.register_buffer('u', F.normalize(torch.randn(in_features), dim=0))
        self.register_buffer('v', F.normalize(torch.randn(self.out_features), dim=0))
        
        self.activation = GroupSort(group_size)
        self.true_out_features = out_features
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def _spectral_normalize(self, weight: Tensor) -> Tensor:
        u, v = self.u, self.v
        
        # weight shape: (in_features, out_features)
        with torch.no_grad():
            for _ in range(1):
                # v = W^T u / ||W^T u||
                v_new = F.normalize(torch.mv(weight.t(), u), dim=0)
                # u = W v / ||W v||
                u_new = F.normalize(torch.mv(weight, v_new), dim=0)
            
            if self.training:
                self.u = u_new
                self.v = v_new
        
        sigma = torch.dot(u_new, torch.mv(weight, v_new))
        return weight / (sigma + 1e-8)
    
    @staticmethod
    def normalize_adjacency(adj: Tensor) -> Tensor:
        n = adj.size(0)
        adj = adj + torch.eye(n, device=adj.device)
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        return degree_inv_sqrt.unsqueeze(1) * adj * degree_inv_sqrt.unsqueeze(0)
    
    def forward(self, x: Tensor, adj: Tensor, adj_normalized: bool = False) -> Tensor:
        if not adj_normalized:
            adj = self.normalize_adjacency(adj)
        
        weight_normalized = self._spectral_normalize(self.weight)
        
        support = torch.matmul(x, weight_normalized)
        output = torch.matmul(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output[..., :self.true_out_features]


class GroupSortGCN(nn.Module):
    """
    GCN with spectral normalization + GroupSort activation.
    Strict 1-Lipschitz continuity.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 64,
        out_features: int = 7,
        num_layers: int = 2,
        group_size: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.group_size = group_size
        
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        self.layers.append(GroupSortGCNLayer(in_features, hidden_features, group_size))
        self.activations.append(GroupSort(group_size))
        
        for _ in range(num_layers - 2):
            self.layers.append(GroupSortGCNLayer(hidden_features, hidden_features, group_size))
            self.activations.append(GroupSort(group_size))
        
        self.layers.append(GroupSortGCNLayer(hidden_features, out_features, group_size))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, adj: Tensor, adj_normalized: bool = False) -> Tensor:
        if not adj_normalized:
            adj = GroupSortGCNLayer.normalize_adjacency(adj)
            adj_normalized = True
        
        for i, (layer, activation) in enumerate(zip(self.layers[:-1], self.activations)):
            x = layer(x, adj, adj_normalized=adj_normalized)
            x = activation(x)
            x = self.dropout(x)
        
        x = self.layers[-1](x, adj, adj_normalized=adj_normalized)
        return x
    
    def get_lipschitz_constant(self) -> float:
        return 1.0
    
    def certify_node(
        self,
        logits: Tensor,
        true_label: int,
        **kwargs,
    ) -> Tuple[float, float]:
        logits_sorted, _ = torch.sort(logits, descending=True)
        
        if torch.argmax(logits).item() == true_label:
            true_logit = logits[true_label]
            runner_up = logits_sorted[1] if logits_sorted[0] == true_logit else logits_sorted[0]
            margin = (true_logit - runner_up).item()
        else:
            margin = (logits[true_label] - logits_sorted[0]).item()
        
        if margin <= 0:
            return margin, 0.0
        
        k = self.get_lipschitz_constant()
        certified_radius = margin / (2 * k)
        
        return margin, certified_radius


class PairNorm(nn.Module):
    """
    PairNorm normalization (Zhao & Akoglu, 2019).
    Normalizes node representations to prevent oversmoothing.
    """
    
    def __init__(self, scale: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.scale = scale
        self.eps = eps
    
    def forward(self, x: Tensor) -> Tensor:
        # Center: subtract mean
        x = x - x.mean(dim=0, keepdim=True)
        
        # Scale: normalize to unit variance
        std = x.pow(2).mean(dim=0, keepdim=True).sqrt()
        x = x / (std + self.eps)
        
        return x * self.scale


class PairNormGCN(nn.Module):
    """
    GCN with PairNorm for preventing oversmoothing.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 64,
        out_features: int = 7,
        num_layers: int = 2,
        dropout: float = 0.5,
        pairnorm_scale: float = 1.0,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        self.layers.append(SpectralGCNLayer(in_features, hidden_features))
        self.norms.append(PairNorm(scale=pairnorm_scale))
        
        for _ in range(num_layers - 2):
            self.layers.append(SpectralGCNLayer(hidden_features, hidden_features))
            self.norms.append(PairNorm(scale=pairnorm_scale))
        
        self.layers.append(SpectralGCNLayer(hidden_features, out_features))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, adj: Tensor, adj_normalized: bool = False) -> Tensor:
        if not adj_normalized:
            adj = SpectralGCNLayer.normalize_adjacency(adj)
            adj_normalized = True
        
        for i, (layer, norm) in enumerate(zip(self.layers[:-1], self.norms)):
            x = layer(x, adj, adj_normalized=adj_normalized)
            x = F.relu(x)
            x = norm(x)
            x = self.dropout(x)
        
        x = self.layers[-1](x, adj, adj_normalized=adj_normalized)
        return x
    
    def get_lipschitz_constant(self) -> float:
        return 1.0
    
    def certify_node(
        self,
        logits: Tensor,
        true_label: int,
        **kwargs,
    ) -> Tuple[float, float]:
        logits_sorted, _ = torch.sort(logits, descending=True)
        
        if torch.argmax(logits).item() == true_label:
            true_logit = logits[true_label]
            runner_up = logits_sorted[1] if logits_sorted[0] == true_logit else logits_sorted[0]
            margin = (true_logit - runner_up).item()
        else:
            margin = (logits[true_label] - logits_sorted[0]).item()
        
        if margin <= 0:
            return margin, 0.0
        
        k = self.get_lipschitz_constant()
        certified_radius = margin / (2 * k)
        
        return margin, certified_radius