"""
α-RePU Activation Function for Graph Hölder Networks
Based on Definition 3.2 in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AlphaRePU(nn.Module):
    """
    α-Rectified Power Unit (α-RePU) activation function.
    
    σ_{α,c}(x) = (x + c)^α  if x >= 0
                 c^α         if x < 0
    
    This activation is α-Hölder continuous with Hölder constant 1.
    
    Args:
        alpha: Hölder exponent (0 < α <= 1). Default: 0.8
        c: Smoothing constant (c > 0). Default: 1e-4
    """
    
    def __init__(self, alpha: float = 0.8, c: float = 1e-4):
        super().__init__()
        assert 0 < alpha <= 1, f"α must be in (0, 1], got {alpha}"
        assert c > 0, f"c must be positive, got {c}"
        
        self.alpha = alpha
        self.c = c
        self._c_alpha = c ** alpha  # Precompute c^α
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply α-RePU activation element-wise.
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Activated tensor of same shape
        """
        # For numerical stability, use torch.where
        positive_part = torch.pow(x + self.c, self.alpha)
        return torch.where(x >= 0, positive_part, self._c_alpha * torch.ones_like(x))
    
    def holder_seminorm(self) -> float:
        """Return the Hölder seminorm [σ_{α,c}]_α = 1."""
        return 1.0
    
    def extra_repr(self) -> str:
        return f'alpha={self.alpha}, c={self.c}'


class GroupSort(nn.Module):
    """
    GroupSort activation for 1-Lipschitz networks.
    Sorts each group of neurons to maintain Lipschitz continuity.
    
    Reference: Anil et al., "Sorting out Lipschitz function approximation"
    """
    
    def __init__(self, group_size: int = 2):
        super().__init__()
        self.group_size = group_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [..., features]
        *batch_dims, features = x.shape
        
        if features % self.group_size != 0:
            # Pad if needed
            pad_size = self.group_size - (features % self.group_size)
            x = F.pad(x, (0, pad_size))
            features = x.shape[-1]
        
        # Reshape to groups and sort
        x = x.view(*batch_dims, features // self.group_size, self.group_size)
        x, _ = torch.sort(x, dim=-1, descending=True)
        x = x.view(*batch_dims, features)
        
        return x[..., :features]
    
    def extra_repr(self) -> str:
        return f'group_size={self.group_size}'


class MaxMin(nn.Module):
    """
    MaxMin activation: pairs of neurons output (max, min).
    A special case of GroupSort with group_size=2.
    Maintains 1-Lipschitz property.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *batch_dims, features = x.shape
        assert features % 2 == 0, "Features must be even for MaxMin"
        
        x = x.view(*batch_dims, features // 2, 2)
        x_max = torch.max(x, dim=-1).values
        x_min = torch.min(x, dim=-1).values
        
        return torch.cat([x_max, x_min], dim=-1)


def get_activation(name: str, **kwargs) -> nn.Module:
    """
    Factory function to get activation by name.
    
    Args:
        name: One of 'relu', 'alpha_repu', 'groupsort', 'maxmin', 'tanh', 'sigmoid'
        **kwargs: Additional arguments for the activation
        
    Returns:
        Activation module
    """
    activations = {
        'relu': nn.ReLU,
        'alpha_repu': AlphaRePU,
        'groupsort': GroupSort,
        'maxmin': MaxMin,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'leaky_relu': nn.LeakyReLU,
        'elu': nn.ELU,
    }
    
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
    
    return activations[name.lower()](**kwargs)
