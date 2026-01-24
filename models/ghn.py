"""
Graph Hölder Network (GHN) - Core Architecture
Based on Definitions 3.3 and 3.4 in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List
import math

from .activations import AlphaRePU, get_activation


class GraphHolderLayer(nn.Module):
    """
    Graph Hölder Layer (Definition 3.3).
    
    H^{(l+1)} = Σ_{α,c}(Â H^{(l)} W^{(l)} + 1_n (b^{(l)})^T)
    
    Key properties:
    - Uses α-RePU activation for α-Hölder continuity
    - Symmetric normalized adjacency Â = D̃^{-1/2} Ã D̃^{-1/2}
    - Layer-wise Hölder constant C_l = (n·d_{l+1})^{(1-α)/2} · B_l^α
    
    Args:
        in_features: Input feature dimension d_l
        out_features: Output feature dimension d_{l+1}
        alpha: Hölder exponent (0 < α <= 1)
        c: Smoothing constant for α-RePU
        bias: Whether to include bias term
        cached: Whether to cache normalized adjacency
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        alpha: float = 0.8,
        c: float = 1e-4,
        bias: bool = True,
        cached: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.c = c
        self.cached = cached
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # α-RePU activation
        self.activation = AlphaRePU(alpha=alpha, c=c)
        
        # Cache for normalized adjacency
        self._cached_adj_norm = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Glorot initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    @staticmethod
    def normalize_adjacency(
        adj: Tensor,
        add_self_loops: bool = True,
    ) -> Tensor:
        """
        Compute symmetric normalized adjacency matrix.
        
        Â = D̃^{-1/2} Ã D̃^{-1/2}
        
        where Ã = A + I_n (with self-loops)
        
        Args:
            adj: Adjacency matrix [n, n] (dense or sparse)
            add_self_loops: Whether to add self-loops
            
        Returns:
            Normalized adjacency matrix [n, n]
        """
        n = adj.size(0)
        device = adj.device
        
        # Add self-loops: Ã = A + I_n
        if add_self_loops:
            if adj.is_sparse:
                identity = torch.sparse_coo_tensor(
                    torch.arange(n, device=device).unsqueeze(0).repeat(2, 1),
                    torch.ones(n, device=device),
                    (n, n)
                )
                adj_tilde = adj + identity
            else:
                adj_tilde = adj + torch.eye(n, device=device)
        else:
            adj_tilde = adj
        
        # Compute degree: D̃_ii = Σ_j Ã_ij
        if adj_tilde.is_sparse:
            degree = torch.sparse.sum(adj_tilde, dim=1).to_dense()
        else:
            degree = adj_tilde.sum(dim=1)
        
        # D̃^{-1/2}
        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        
        # Symmetric normalization: D̃^{-1/2} Ã D̃^{-1/2}
        if adj_tilde.is_sparse:
            # For sparse: scale rows and columns
            adj_tilde = adj_tilde.to_dense()
        
        # D^{-1/2} A D^{-1/2} = (D^{-1/2})^T * A * D^{-1/2}
        adj_norm = degree_inv_sqrt.unsqueeze(1) * adj_tilde * degree_inv_sqrt.unsqueeze(0)
        
        return adj_norm
    
    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        adj_normalized: bool = False,
    ) -> Tensor:
        """
        Forward pass of Graph Hölder Layer.
        
        Args:
            x: Node features [n, d_l]
            adj: Adjacency matrix [n, n]
            adj_normalized: If True, assume adj is already normalized
            
        Returns:
            Updated node features [n, d_{l+1}]
        """
        # Normalize adjacency if needed
        if adj_normalized:
            adj_norm = adj
        elif self.cached and self._cached_adj_norm is not None:
            adj_norm = self._cached_adj_norm
        else:
            adj_norm = self.normalize_adjacency(adj)
            if self.cached:
                self._cached_adj_norm = adj_norm
        
        # Message passing: Â H W
        support = torch.matmul(x, self.weight)  # [n, d_{l+1}]
        output = torch.matmul(adj_norm, support)  # [n, d_{l+1}]
        
        # Add bias
        if self.bias is not None:
            output = output + self.bias
        
        # Apply α-RePU activation
        output = self.activation(output)
        
        return output
    
    def get_spectral_norm(self) -> float:
        """
        Compute spectral norm ||W||_2 of weight matrix.
        Used for certified radius computation (Eq. 2).
        """
        with torch.no_grad():
            # Power iteration for spectral norm estimation
            u = torch.randn(self.in_features, device=self.weight.device)
            for _ in range(10):  # 10 iterations as in paper
                v = F.normalize(torch.mv(self.weight.t(), u), dim=0)
                u = F.normalize(torch.mv(self.weight, v), dim=0)
            sigma = torch.dot(u, torch.mv(self.weight, v))
            return sigma.item()
    
    def get_holder_constant(self, n_nodes: int) -> float:
        """
        Compute single-layer Hölder constant C_l (Lemma A.5).
        
        C_l = (n · d_{l+1})^{(1-α)/2} · ||W||_2^α
        
        Args:
            n_nodes: Number of nodes in the graph
            
        Returns:
            Layer Hölder constant
        """
        spectral_norm = self.get_spectral_norm()
        dim_factor = (n_nodes * self.out_features) ** ((1 - self.alpha) / 2)
        return dim_factor * (spectral_norm ** self.alpha)
    
    def clear_cache(self):
        """Clear cached normalized adjacency."""
        self._cached_adj_norm = None


class GraphHolderNetwork(nn.Module):
    """
    Graph Hölder Network (GHN) - Definition 3.4.
    
    A deep GNN with α-Hölder continuity for certified adversarial robustness.
    
    Key properties:
    - Global Hölder exponent: α_net = α^L
    - Network Hölder constant: C_net = Π_l C_l^{α^{L-1-l}}
    - Certified radius: R = (γ / 2C_net)^{1/α_net} (super-linear scaling)
    
    Args:
        in_features: Input feature dimension
        hidden_features: Hidden layer dimension(s)
        out_features: Output dimension (number of classes)
        num_layers: Number of GHN layers
        alpha: Hölder exponent (0 < α < 1)
        c: Smoothing constant for α-RePU
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 64,
        out_features: int = 7,
        num_layers: int = 2,
        alpha: float = 0.8,
        c: float = 1e-4,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.alpha = alpha
        self.c = c
        self.dropout_p = dropout
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(GraphHolderLayer(
            in_features, hidden_features, alpha=alpha, c=c
        ))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GraphHolderLayer(
                hidden_features, hidden_features, alpha=alpha, c=c
            ))
        
        # Output layer (no activation, just linear readout)
        self.readout = nn.Linear(hidden_features, out_features)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        adj_normalized: bool = False,
    ) -> Tensor:
        """
        Forward pass through GHN.
        
        Args:
            x: Node features [n, d_in]
            adj: Adjacency matrix [n, n]
            adj_normalized: Whether adjacency is pre-normalized
            
        Returns:
            Logits [n, C]
        """
        # Normalize adjacency once
        if not adj_normalized:
            adj = GraphHolderLayer.normalize_adjacency(adj)
            adj_normalized = True
        
        # Pass through GHN layers
        h = x
        for i, layer in enumerate(self.layers):
            h = self.dropout(h) if i > 0 else h
            h = layer(h, adj, adj_normalized=adj_normalized)
        
        # Readout layer
        h = self.dropout(h)
        out = self.readout(h)
        
        return out
    
    def get_network_holder_constant(self, n_nodes: int) -> float:
        """
        Compute network Hölder constant C_net (Eq. 2 in paper).
        
        C_net = Π_{l=1}^L ||W^{(l)}||_2^α
        
        Note: Simplified from Corollary 3.5 assuming unit Hölder seminorm of activation.
        
        Args:
            n_nodes: Number of nodes
            
        Returns:
            Network Hölder constant
        """
        c_net = 1.0
        for layer in self.layers:
            spectral_norm = layer.get_spectral_norm()
            c_net *= spectral_norm ** self.alpha
        
        # Include readout layer
        with torch.no_grad():
            u = torch.randn(self.hidden_features, device=self.readout.weight.device)
            for _ in range(10):
                v = F.normalize(torch.mv(self.readout.weight.t(), u), dim=0)
                u = F.normalize(torch.mv(self.readout.weight, v), dim=0)
            readout_norm = torch.dot(u, torch.mv(self.readout.weight, v)).item()
        
        c_net *= readout_norm
        
        return c_net
    
    def get_alpha_net(self) -> float:
        """Get global Hölder exponent α_net = α^L."""
        return self.alpha ** self.num_layers
    
    def certify_node(
        self,
        logits: Tensor,
        true_label: int,
        n_nodes: int,
    ) -> Tuple[float, float]:
        """
        Compute certified radius for a single node (Corollary 3.6).
        
        R_i = (γ_i / 2C_net)^{1/α_net}
        
        Args:
            logits: Output logits for node i [C]
            true_label: Ground truth label
            n_nodes: Number of nodes in graph
            
        Returns:
            (margin, certified_radius) tuple
        """
        # Classification margin: γ = f_y - max_{k≠y} f_k
        logits_sorted, _ = torch.sort(logits, descending=True)
        
        if torch.argmax(logits).item() == true_label:
            # Correct prediction
            true_logit = logits[true_label]
            # Runner-up logit
            if logits_sorted[0] == true_logit:
                runner_up = logits_sorted[1]
            else:
                runner_up = logits_sorted[0]
            margin = (true_logit - runner_up).item()
        else:
            # Incorrect prediction, margin is negative
            margin = (logits[true_label] - logits_sorted[0]).item()
        
        if margin <= 0:
            return margin, 0.0
        
        # Compute certified radius
        c_net = self.get_network_holder_constant(n_nodes)
        alpha_net = self.get_alpha_net()
        
        certified_radius = (margin / (2 * c_net)) ** (1 / alpha_net)
        
        return margin, certified_radius
    
    def clear_cache(self):
        """Clear all layer caches."""
        for layer in self.layers:
            layer.clear_cache()
