"""
Standard GNN Baselines: GCN, GAT, SGC
These serve as accuracy upper bounds (no robustness guarantees).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math


class GCNLayer(nn.Module):
    """
    Graph Convolutional Layer (Kipf & Welling, 2016).
    
    H^{(l+1)} = ReLU(Â H^{(l)} W^{(l)})
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to include bias
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
        
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    @staticmethod
    def normalize_adjacency(adj: Tensor, add_self_loops: bool = True) -> Tensor:
        """Symmetric normalized adjacency: D^{-1/2} A D^{-1/2}"""
        n = adj.size(0)
        device = adj.device
        
        if add_self_loops:
            adj = adj + torch.eye(n, device=device)
        
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        
        return degree_inv_sqrt.unsqueeze(1) * adj * degree_inv_sqrt.unsqueeze(0)
    
    def forward(self, x: Tensor, adj: Tensor, adj_normalized: bool = False) -> Tensor:
        if not adj_normalized:
            adj = self.normalize_adjacency(adj)
        
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def get_spectral_norm(self) -> float:
        """Compute spectral norm for post-hoc Lipschitz analysis."""
        with torch.no_grad():
            u = torch.randn(self.in_features, device=self.weight.device)
            for _ in range(10):
                v = F.normalize(torch.mv(self.weight.t(), u), dim=0)
                u = F.normalize(torch.mv(self.weight, v), dim=0)
            return torch.dot(u, torch.mv(self.weight, v)).item()


class GCN(nn.Module):
    """
    Graph Convolutional Network (Kipf & Welling, 2016).
    Standard 2-layer GCN with ReLU activation.
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
        self.layers.append(GCNLayer(in_features, hidden_features))
        
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_features, hidden_features))
        
        self.layers.append(GCNLayer(hidden_features, out_features))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, adj: Tensor, adj_normalized: bool = False) -> Tensor:
        if not adj_normalized:
            adj = GCNLayer.normalize_adjacency(adj)
            adj_normalized = True
        
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adj, adj_normalized=adj_normalized)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.layers[-1](x, adj, adj_normalized=adj_normalized)
        return x
    
    def get_lipschitz_constant(self) -> float:
        """Compute post-hoc Lipschitz constant K = Π_l ||W^{(l)}||_2."""
        k = 1.0
        for layer in self.layers:
            k *= layer.get_spectral_norm()
        return k


class MultiHeadAttention(nn.Module):
    """Multi-head attention for GAT."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 8,
        dropout: float = 0.6,
        concat: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        
        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.a_src = nn.Parameter(torch.empty(num_heads, out_features))
        self.a_dst = nn.Parameter(torch.empty(num_heads, out_features))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
    
    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        n = x.size(0)
        
        # Linear transformation: [n, in] -> [n, heads * out]
        h = self.W(x).view(n, self.num_heads, self.out_features)  # [n, heads, out]
        
        # Attention scores
        e_src = (h * self.a_src).sum(dim=-1)  # [n, heads]
        e_dst = (h * self.a_dst).sum(dim=-1)  # [n, heads]
        
        # Compute attention coefficients
        e = e_src.unsqueeze(1) + e_dst.unsqueeze(0)  # [n, n, heads]
        e = self.leaky_relu(e)
        
        # Mask with adjacency (including self-loops)
        mask = (adj + torch.eye(n, device=adj.device)) > 0
        e = e.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        
        # Softmax over neighbors
        alpha = F.softmax(e, dim=1)  # [n, n, heads]
        alpha = self.dropout(alpha)
        
        # Aggregate
        h = h.permute(1, 0, 2)  # [heads, n, out]
        alpha = alpha.permute(2, 0, 1)  # [heads, n, n]
        out = torch.bmm(alpha, h)  # [heads, n, out]
        out = out.permute(1, 0, 2)  # [n, heads, out]
        
        if self.concat:
            return out.reshape(n, -1)  # [n, heads * out]
        else:
            return out.mean(dim=1)  # [n, out]


class GAT(nn.Module):
    """
    Graph Attention Network (Veličković et al., 2017).
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 8,
        out_features: int = 7,
        num_heads: int = 8,
        dropout: float = 0.6,
    ):
        super().__init__()
        
        self.layer1 = MultiHeadAttention(
            in_features, hidden_features, num_heads=num_heads,
            dropout=dropout, concat=True
        )
        self.layer2 = MultiHeadAttention(
            hidden_features * num_heads, out_features, num_heads=1,
            dropout=dropout, concat=False
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, adj: Tensor, **kwargs) -> Tensor:
        x = self.dropout(x)
        x = self.layer1(x, adj)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.layer2(x, adj)
        return x


class SGC(nn.Module):
    """
    Simplified Graph Convolution (Wu et al., 2019).
    
    Removes non-linearities between layers:
    Y = softmax(Â^K X W)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int = 7,
        k_hops: int = 2,
    ):
        super().__init__()
        
        self.k_hops = k_hops
        self.linear = nn.Linear(in_features, out_features)
    
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
        
        # K-hop propagation: Â^K X
        for _ in range(self.k_hops):
            x = torch.matmul(adj, x)
        
        return self.linear(x)
    
    def get_lipschitz_constant(self) -> float:
        """Post-hoc Lipschitz constant."""
        with torch.no_grad():
            u = torch.randn(self.linear.in_features, device=self.linear.weight.device)
            for _ in range(10):
                v = F.normalize(torch.mv(self.linear.weight.t(), u), dim=0)
                u = F.normalize(torch.mv(self.linear.weight, v), dim=0)
            return torch.dot(u, torch.mv(self.linear.weight, v)).item()
