"""
Training Utilities

Provides trainer class and utilities for training GNN models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from typing import Dict, Optional, Tuple, List, Callable
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0,
        mode: str = 'min',
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None
    
    def __call__(
        self,
        score: float,
        model: nn.Module,
    ) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            model: Model to save state
            
        Returns:
            True if should stop
        """
        if self.mode == 'min':
            score = -score
        
        if self.best_score is None:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        
        return self.early_stop
    
    def load_best(self, model: nn.Module):
        """Load best model state."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class Trainer:
    """
    Trainer class for GNN models.
    
    Args:
        model: Model to train
        optimizer: Optimizer instance
        device: Training device
        log_interval: Logging frequency
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device = torch.device('cpu'),
        log_interval: int = 10,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.log_interval = log_interval
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
    
    def train_epoch(
        self,
        x: Tensor,
        adj: Tensor,
        labels: Tensor,
        train_mask: Tensor,
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            (loss, accuracy)
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        logits = self.model(x, adj)
        
        # Loss only on training nodes
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Accuracy
        preds = logits[train_mask].argmax(dim=-1)
        acc = (preds == labels[train_mask]).float().mean().item()
        
        return loss.item(), acc
    
    @torch.no_grad()
    def evaluate(
        self,
        x: Tensor,
        adj: Tensor,
        labels: Tensor,
        mask: Tensor,
    ) -> Tuple[float, float]:
        """
        Evaluate model.
        
        Returns:
            (loss, accuracy)
        """
        self.model.eval()
        
        logits = self.model(x, adj)
        
        loss = F.cross_entropy(logits[mask], labels[mask]).item()
        preds = logits[mask].argmax(dim=-1)
        acc = (preds == labels[mask]).float().mean().item()
        
        return loss, acc
    
    def fit(
        self,
        x: Tensor,
        adj: Tensor,
        labels: Tensor,
        train_mask: Tensor,
        val_mask: Tensor,
        epochs: int = 200,
        early_stopping: Optional[EarlyStopping] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            x: Node features
            adj: Adjacency matrix
            labels: Node labels
            train_mask: Training mask
            val_mask: Validation mask
            epochs: Number of epochs
            early_stopping: Early stopping callback
            verbose: Print progress
            
        Returns:
            Training history
        """
        # Move data to device
        x = x.to(self.device)
        adj = adj.to(self.device)
        labels = labels.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        
        iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)
        
        for epoch in iterator:
            # Train
            train_loss, train_acc = self.train_epoch(x, adj, labels, train_mask)
            
            # Validate
            val_loss, val_acc = self.evaluate(x, adj, labels, val_mask)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Logging
            if verbose and (epoch + 1) % self.log_interval == 0:
                tqdm.write(
                    f"Epoch {epoch+1:03d}: "
                    f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                    f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
                )
            
            # Early stopping
            if early_stopping is not None:
                if early_stopping(val_loss, self.model):
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    early_stopping.load_best(self.model)
                    break
        
        return self.history
    
    def save_checkpoint(
        self,
        path: str,
        extra_info: Optional[Dict] = None,
    ):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }
        
        if extra_info is not None:
            checkpoint.update(extra_info)
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']


def create_optimizer(
    model: nn.Module,
    name: str = 'adam',
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    **kwargs,
) -> optim.Optimizer:
    """
    Create optimizer by name.
    
    Args:
        model: Model to optimize
        name: Optimizer name ('adam', 'sgd', 'adamw')
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)
        
    Returns:
        Optimizer instance
    """
    optimizers = {
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'adamw': optim.AdamW,
        'rmsprop': optim.RMSprop,
    }
    
    name = name.lower()
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")
    
    return optimizers[name](
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        **kwargs,
    )


def train_and_evaluate(
    model: nn.Module,
    data,
    config: Dict,
    device: torch.device,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Train and evaluate a model with given config.
    
    Args:
        model: Model to train
        data: GraphData namedtuple
        config: Training configuration
        device: Training device
        verbose: Print progress
        
    Returns:
        Dictionary with final metrics
    """
    # Create optimizer
    optimizer = create_optimizer(
        model,
        name=config.get('optimizer', 'adam'),
        lr=config.get('lr', 0.01),
        weight_decay=config.get('weight_decay', 5e-4),
    )
    
    # Create trainer
    trainer = Trainer(model, optimizer, device)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.get('patience', 20),
        mode='min',
    )
    
    # Train
    start_time = time.time()
    history = trainer.fit(
        x=data.x,
        adj=data.adj,
        labels=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        epochs=config.get('epochs', 200),
        early_stopping=early_stopping,
        verbose=verbose,
    )
    train_time = time.time() - start_time
    
    # Final evaluation
    x = data.x.to(device)
    adj = data.adj.to(device)
    labels = data.y.to(device)
    
    test_loss, test_acc = trainer.evaluate(
        x, adj, labels, data.test_mask.to(device)
    )
    
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'train_time': train_time,
        'best_val_loss': min(history['val_loss']),
        'best_val_acc': max(history['val_acc']),
        'epochs_trained': len(history['train_loss']),
    }
    
    return results


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ExperimentLogger:
    """Logger for experiments."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
    
    def log(self, result: Dict, name: str = None):
        """Log experiment result."""
        result['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        if name:
            result['name'] = name
        
        self.results.append(result)
        
        # Save to file
        log_path = self.log_dir / 'results.json'
        with open(log_path, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.results:
            return {}
        
        # Group by model
        by_model = {}
        for r in self.results:
            model = r.get('model', 'unknown')
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(r)
        
        summary = {}
        for model, results in by_model.items():
            accs = [r['test_accuracy'] for r in results if 'test_accuracy' in r]
            acrs = [r['average_certified_radius'] for r in results if 'average_certified_radius' in r]
            
            summary[model] = {
                'mean_accuracy': np.mean(accs) if accs else None,
                'std_accuracy': np.std(accs) if accs else None,
                'mean_acr': np.mean(acrs) if acrs else None,
                'std_acr': np.std(acrs) if acrs else None,
                'n_runs': len(results),
            }
        
        return summary
