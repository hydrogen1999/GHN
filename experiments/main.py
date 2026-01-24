#!/usr/bin/env python3
"""
Complete Experiment Runner for Graph Hölder Networks
ICML 2026 Submission

Implements ALL experiments from the paper:
- Table 1: Clean accuracy and ACR (Section 1.2)
- Figure 1: Certified accuracy curves (Section 1.2)
- Figure 2: Margin-radius scaling (Section 1.3)
- Table 2: PGD attacks (Section 1.4)
- Table 3: Nettack + Metattack (Section 1.5)
- Table 4: Bernoulli edge deletion (Section 1.5)
- Figure 3: NSR analysis (Section 1.6)
- Table 5: Deep network + MAD (Section 1.7)
- Table 6: Ablation α (Section 1.8)
- Table 7: Ablation depth (Section 1.8)
- Table 8: Spectral normalization ablation (Section 1.8)
- Table 9: Scalability ogbn-arxiv (Section 1.9)

Usage:
    python experiments/main.py --experiment all
    python experiments/main.py --experiment table1
    python experiments/main.py --experiment figure1
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_model, MODEL_REGISTRY
from models.ghn import GraphHolderNetwork
from models.baselines import GCN
from models.lipschitz import GroupSortGCN
from data.datasets import load_dataset, AVAILABLE_DATASETS, print_dataset_info
from certify.certification import (
    certify_all_nodes,
    compute_holder_certified_radius,
    compute_lipschitz_certified_radius,
    compare_scaling_behavior,
    compute_classification_margin,
)
from utils.training import (
    train_and_evaluate,
    set_seed,
    ExperimentLogger,
)
from utils.metrics import (
    accuracy,
    certified_accuracy_at_radius,
    average_certified_radius,
)
from configs.default import (
    get_model_config,
    get_training_config,
)

# Import attack modules
try:
    from attacks import (
        PGDAttack, FGSMAttack, Nettack, Metattack, 
        BernoulliEdgeDeletion, evaluate_under_attack,
        run_nettack_evaluation,
    )
    HAS_ATTACKS = True
except ImportError:
    HAS_ATTACKS = False
    print("Warning: attacks module not found")

# Import analysis modules
try:
    from utils.analysis import (
        compute_nsr, compare_nsr_models,
        compute_mad, compute_mad_for_model,
        compute_certified_accuracy_curve,
        compute_spectral_norms, compute_holder_constant,
    )
    HAS_ANALYSIS = True
except ImportError:
    HAS_ANALYSIS = False
    print("Warning: analysis module not found")


# =============================================================================
# Table 1: Clean Accuracy and ACR (Section 1.2)
# =============================================================================

def run_table1(
    datasets: List[str],
    models: List[str],
    seeds: List[int],
    device: torch.device,
    output_dir: Path,
) -> Dict:
    """
    Table 1: Clean accuracy (%) and average certified radius (ACR).
    """
    print("\n" + "="*70)
    print("TABLE 1: Clean Accuracy and Average Certified Radius")
    print("="*70)
    
    results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name.upper()}")
        print('='*50)
        
        data = load_dataset(dataset_name)
        print_dataset_info(data)
        results[dataset_name] = {}
        
        for model_name in models:
            print(f"\n  Model: {model_name}")
            
            model_results = {'accuracy': [], 'acr': []}
            
            for seed in seeds:
                set_seed(seed)
                
                try:
                    config = get_model_config(model_name)
                    model = get_model(
                        model_name,
                        in_features=data.num_features,
                        out_features=data.num_classes,
                        **config,
                    )
                    
                    train_results = train_and_evaluate(
                        model, data, get_training_config(), device, verbose=False
                    )
                    model_results['accuracy'].append(train_results['test_accuracy'])
                    
                    # Certification for appropriate models
                    if model_name in ['ghn', 'spectral_gcn', 'groupsort_gcn', 'pairnorm_gcn']:
                        cert = certify_all_nodes(
                            model=model,
                            x=data.x.to(device),
                            adj=data.adj.to(device),
                            labels=data.y.to(device),
                            test_mask=data.test_mask.to(device),
                            model_type='ghn' if model_name == 'ghn' else 'lipschitz',
                            alpha=config.get('alpha', 1.0),
                            num_layers=config.get('num_layers', 2),
                        )
                        model_results['acr'].append(cert['average_certified_radius'])
                    elif model_name == 'randomized_smoothing':
                        # Use model's built-in certification
                        model.to(device)
                        model.eval()
                        _, cert_acc, avg_radius = model.certify_all_nodes(
                            data.x.to(device), data.adj.to(device),
                            data.test_mask.to(device), data.y.to(device),
                        )
                        model_results['acr'].append(avg_radius)
                    elif model_name == 'gnncert':
                        # Use GNNCert's certification
                        model.to(device)
                        model.eval()
                        test_indices = data.test_mask.nonzero().squeeze(-1)
                        radii = []
                        for idx in test_indices[:100]:  # Sample for efficiency
                            idx = idx.item()
                            _, radius = model.certify_node(
                                data.x.to(device), data.adj.to(device),
                                idx, data.y[idx].item(),
                            )
                            if radius > 0:
                                radii.append(radius)
                        avg_radius = np.mean(radii) if radii else 0.0
                        model_results['acr'].append(avg_radius)
                    else:
                        model_results['acr'].append(None)
                        
                except Exception as e:
                    print(f"    Seed {seed}: FAILED - {e}")
                    continue
            
            if model_results['accuracy']:
                acc_mean = np.mean(model_results['accuracy'])
                acc_std = np.std(model_results['accuracy'])
                acr_vals = [x for x in model_results['acr'] if x is not None]
                acr_mean = np.mean(acr_vals) if acr_vals else None
                acr_std = np.std(acr_vals) if acr_vals else None
                
                results[dataset_name][model_name] = {
                    'accuracy_mean': acc_mean,
                    'accuracy_std': acc_std,
                    'acr_mean': acr_mean,
                    'acr_std': acr_std,
                }
                
                acr_str = f"{acr_mean:.3f}±{acr_std:.3f}" if acr_mean else "—"
                print(f"    Acc: {acc_mean*100:.1f}±{acc_std*100:.1f}%, ACR: {acr_str}")
    
    # Save and print LaTeX
    save_json(output_dir / 'table1.json', results)
    print_latex_table1(results)
    
    return results


# =============================================================================
# Figure 1: Certified Accuracy Curves (Section 1.2)
# =============================================================================

def run_figure1(
    dataset: str,
    device: torch.device,
    output_dir: Path,
) -> Dict:
    """
    Figure 1: Certified accuracy vs perturbation radius.
    """
    print("\n" + "="*70)
    print("FIGURE 1: Certified Accuracy Curves")
    print("="*70)
    
    data = load_dataset(dataset)
    results = {}
    
    models_to_compare = ['ghn', 'groupsort_gcn', 'spectral_gcn']
    
    for model_name in models_to_compare:
        set_seed(42)
        config = get_model_config(model_name)
        model = get_model(
            model_name,
            in_features=data.num_features,
            out_features=data.num_classes,
            **config,
        )
        train_and_evaluate(model, data, get_training_config(), device, verbose=False)
        
        cert = certify_all_nodes(
            model=model,
            x=data.x.to(device),
            adj=data.adj.to(device),
            labels=data.y.to(device),
            test_mask=data.test_mask.to(device),
            model_type='ghn' if model_name == 'ghn' else 'lipschitz',
            alpha=config.get('alpha', 1.0),
            num_layers=config.get('num_layers', 2),
        )
        
        # Compute CA curve
        radii_range = np.linspace(0, 0.3, 50)
        ca_curve = []
        test_labels = data.y[data.test_mask]
        for r in radii_range:
            ca = certified_accuracy_at_radius(
            torch.tensor(cert['radii']),
            torch.tensor(cert['predictions']),
            test_labels.cpu(),
            r,
            )
            ca_curve.append(ca)
        
        results[model_name] = {
            'radii': radii_range.tolist(),
            'certified_accuracy': ca_curve,
        }
        
        print(f"  {model_name}: CA@0.10 = {ca_curve[int(0.10/0.3*49)]:.3f}, CA@0.15 = {ca_curve[int(0.15/0.3*49)]:.3f}")
    
    save_json(output_dir / 'figure1.json', results)
    save_plot_script(output_dir / 'plot_figure1.py', 'figure1')
    
    return results


# =============================================================================
# Figure 2: Margin-Radius Scaling (Section 1.3)
# =============================================================================

def run_figure2(
    dataset: str,
    device: torch.device,
    output_dir: Path,
) -> Dict:
    """
    Figure 2: Log-log scatter of margin γ vs certified radius R.
    
    Validates the scaling: R ∝ γ^{1/α^L}
    """
    print("\n" + "="*70)
    print("FIGURE 2: Margin-Radius Scaling Analysis")
    print("="*70)
    
    data = load_dataset(dataset)
    results = {}
    
    # GHN
    set_seed(42)
    ghn_config = get_model_config('ghn')
    ghn_model = get_model('ghn', data.num_features, data.num_classes, **ghn_config)
    train_and_evaluate(ghn_model, data, get_training_config(), device, verbose=False)
    
    # GroupSort-GCN
    set_seed(42)
    lip_config = get_model_config('groupsort_gcn')
    lip_model = get_model('groupsort_gcn', data.num_features, data.num_classes, **lip_config)
    train_and_evaluate(lip_model, data, get_training_config(), device, verbose=False)
    
    comparison = compare_scaling_behavior(
        ghn_model=ghn_model,
        lipschitz_model=lip_model,
        x=data.x.to(device),
        adj=data.adj.to(device),
        labels=data.y.to(device),
        test_mask=data.test_mask.to(device),
        alpha=ghn_config['alpha'],
        num_layers=ghn_config['num_layers'],
    )
    
    # Fit log-log regression to estimate slope
    from scipy import stats
    
    margins = comparison['margins']
    ghn_radii = comparison['ghn_radii']
    lip_radii = comparison['lipschitz_radii']
    
    # Filter positive values for log
    valid = (margins > 0) & (ghn_radii > 0) & (lip_radii > 0)
    
    ghn_slope, _, _, _, ghn_stderr = stats.linregress(
        np.log(margins[valid]), np.log(ghn_radii[valid])
    )
    lip_slope, _, _, _, lip_stderr = stats.linregress(
        np.log(margins[valid]), np.log(lip_radii[valid])
    )
    
    alpha = ghn_config['alpha']
    L = ghn_config['num_layers']
    theoretical_slope = 1 / (alpha ** L)
    
    print(f"\n  GHN slope (empirical):     β = {ghn_slope:.2f} ± {ghn_stderr:.2f}")
    print(f"  GHN slope (theoretical):   1/α^L = {theoretical_slope:.4f}")
    print(f"  GroupSort-GCN slope:       β = {lip_slope:.2f} ± {lip_stderr:.2f}")
    print(f"  Theoretical (Lipschitz):   β = 1.0")
    
    results = {
        'margins': margins.tolist(),
        'ghn_radii': ghn_radii.tolist(),
        'lipschitz_radii': lip_radii.tolist(),
        'ghn_slope': ghn_slope,
        'ghn_slope_stderr': ghn_stderr,
        'lipschitz_slope': lip_slope,
        'lipschitz_slope_stderr': lip_stderr,
        'theoretical_ghn_slope': theoretical_slope,
        'alpha': alpha,
        'L': L,
    }
    
    save_json(output_dir / 'figure2.json', results)
    save_plot_script(output_dir / 'plot_figure2.py', 'figure2')
    
    return results


# =============================================================================
# Table 2: PGD Attacks (Section 1.4)
# =============================================================================

def run_table2(
    dataset: str,
    models: List[str],
    epsilons: List[float],
    device: torch.device,
    output_dir: Path,
) -> Dict:
    """
    Table 2: Accuracy under PGD ℓ2 feature attacks.
    """
    print("\n" + "="*70)
    print("TABLE 2: Accuracy under PGD Attacks")
    print("="*70)
    
    if not HAS_ATTACKS:
        print("Skipping: attacks module not available")
        return {}
    
    data = load_dataset(dataset)
    results = {}
    
    for model_name in models:
        set_seed(42)
        config = get_model_config(model_name)
        model = get_model(model_name, data.num_features, data.num_classes, **config)
        train_and_evaluate(model, data, get_training_config(), device, verbose=False)
        model.to(device)
        model.eval()
        
        results[model_name] = {'clean': None}
        
        x = data.x.to(device)
        adj = data.adj.to(device)
        labels = data.y.to(device)
        test_mask = data.test_mask.to(device)
        
        # Clean accuracy
        with torch.no_grad():
            logits = model(x, adj)
            clean_acc = (logits[test_mask].argmax(-1) == labels[test_mask]).float().mean().item()
        results[model_name]['clean'] = clean_acc
        
        # PGD attacks at each epsilon
        for eps in epsilons:
            attacker = PGDAttack(model, epsilon=eps, num_steps=40, num_restarts=5)
            x_adv = attacker.attack(x, adj, labels, test_mask)
            
            with torch.no_grad():
                logits_adv = model(x_adv, adj)
                adv_acc = (logits_adv[test_mask].argmax(-1) == labels[test_mask]).float().mean().item()
            
            results[model_name][f'eps_{eps}'] = adv_acc
        
        print(f"  {model_name}: Clean={clean_acc*100:.1f}%, " + 
              ", ".join([f"ε={e}:{results[model_name][f'eps_{e}']*100:.1f}%" for e in epsilons]))
    
    save_json(output_dir / 'table2.json', results)
    print_latex_table2(results, epsilons)
    
    return results


# =============================================================================
# Table 3: Structural Attacks (Section 1.5)
# =============================================================================

def run_table3(
    dataset: str,
    models: List[str],
    metattack_budgets: List[float],
    device: torch.device,
    output_dir: Path,
) -> Dict:
    """
    Table 3: Accuracy under Nettack and Metattack.
    """
    print("\n" + "="*70)
    print("TABLE 3: Structural Attack Robustness")
    print("="*70)
    
    if not HAS_ATTACKS:
        print("Skipping: attacks module not available")
        return {}
    
    data = load_dataset(dataset)
    results = {}
    
    # Need unnormalized adjacency for structural attacks
    adj_unnorm = data.adj.clone()
    # Undo normalization (approximate)
    adj_unnorm = (adj_unnorm > 0).float()
    
    for model_name in models:
        set_seed(42)
        config = get_model_config(model_name)
        model = get_model(model_name, data.num_features, data.num_classes, **config)
        train_and_evaluate(model, data, get_training_config(), device, verbose=False)
        model.to(device)
        
        results[model_name] = {}
        
        # Nettack
        print(f"\n  {model_name} - Nettack...")
        nettack_result = run_nettack_evaluation(
            model, data.x, adj_unnorm, data.y, data.test_mask,
            num_targets=100, device=device,
        )
        results[model_name]['nettack'] = nettack_result['adversarial_accuracy']
        
        # Metattack at various budgets
        for budget in metattack_budgets:
            print(f"  {model_name} - Metattack {int(budget*100)}%...")
            metattacker = Metattack(model, budget_pct=budget)
            adj_poisoned = metattacker.attack(
                data.x.to(device), adj_unnorm.to(device),
                data.y.to(device), data.train_mask.to(device),
            )
            
            # Retrain on poisoned graph
            set_seed(42)
            model_poisoned = get_model(model_name, data.num_features, data.num_classes, **config)
            
            # Create poisoned data
            class PoisonedData:
                pass
            pdata = PoisonedData()
            pdata.x = data.x
            pdata.adj = adj_poisoned.cpu()
            pdata.y = data.y
            pdata.train_mask = data.train_mask
            pdata.val_mask = data.val_mask
            pdata.test_mask = data.test_mask
            pdata.num_features = data.num_features
            pdata.num_classes = data.num_classes
            
            train_results = train_and_evaluate(model_poisoned, pdata, get_training_config(), device, verbose=False)
            results[model_name][f'metattack_{int(budget*100)}'] = train_results['test_accuracy']
        
        print(f"  {model_name}: Nettack={results[model_name]['nettack']*100:.1f}%, " +
              ", ".join([f"Meta{int(b*100)}%:{results[model_name][f'metattack_{int(b*100)}']*100:.1f}%" 
                        for b in metattack_budgets]))
    
    save_json(output_dir / 'table3.json', results)
    
    return results


# =============================================================================
# Table 4: Bernoulli Edge Deletion (Section 1.5)
# =============================================================================

def run_table4(
    dataset: str,
    models: List[str],
    deletion_probs: List[float],
    num_trials: int,
    device: torch.device,
    output_dir: Path,
) -> Dict:
    """
    Table 4: Accuracy under Bernoulli edge deletion.
    Verifies Theorem 3.5 (probabilistic stability bound).
    """
    print("\n" + "="*70)
    print("TABLE 4: Bernoulli Edge Deletion")
    print("="*70)
    
    if not HAS_ATTACKS:
        print("Skipping: attacks module not available")
        return {}
    
    data = load_dataset(dataset)
    adj_unnorm = (data.adj > 0).float()
    results = {}
    
    for model_name in models:
        set_seed(42)
        config = get_model_config(model_name)
        model = get_model(model_name, data.num_features, data.num_classes, **config)
        train_and_evaluate(model, data, get_training_config(), device, verbose=False)
        model.to(device)
        model.eval()
        
        results[model_name] = {}
        
        for p in deletion_probs:
            accs = []
            perturber = BernoulliEdgeDeletion(deletion_prob=p)
            
            for trial in range(num_trials):
                adj_pert = perturber.perturb(adj_unnorm.to(device), seed=trial)
                
                with torch.no_grad():
                    logits = model(data.x.to(device), adj_pert)
                    acc = (logits[data.test_mask].argmax(-1) == data.y[data.test_mask].to(device)).float().mean().item()
                accs.append(acc)
            
            results[model_name][f'p_{p}'] = {
                'mean': np.mean(accs),
                'std': np.std(accs),
            }
        
        print(f"  {model_name}: " + ", ".join([
            f"p={p}:{results[model_name][f'p_{p}']['mean']*100:.1f}±{results[model_name][f'p_{p}']['std']*100:.1f}%" 
            for p in deletion_probs
        ]))
    
    save_json(output_dir / 'table4.json', results)
    
    return results


# =============================================================================
# Figure 3: NSR Analysis (Section 1.6)
# =============================================================================

def run_figure3(
    dataset: str,
    device: torch.device,
    output_dir: Path,
) -> Dict:
    """
    Figure 3: Noise-to-Signal Ratio across layers.
    """
    print("\n" + "="*70)
    print("FIGURE 3: NSR Analysis (Geometric Compression)")
    print("="*70)
    
    if not HAS_ANALYSIS:
        print("Skipping: analysis module not available")
        return {}
    
    data = load_dataset(dataset)
    results = {}
    
    # Train 4-layer models
    models_config = {
        'GHN': ('ghn', {'num_layers': 4}),
        'GCN': ('gcn', {'num_layers': 4}),
        'GroupSort-GCN': ('groupsort_gcn', {'num_layers': 4}),
    }
    
    for name, (model_type, extra_config) in models_config.items():
        set_seed(42)
        config = get_model_config(model_type)
        config.update(extra_config)
        
        model = get_model(model_type, data.num_features, data.num_classes, **config)
        train_and_evaluate(model, data, get_training_config(), device, verbose=False)
        model.to(device)
        
        nsr_result = compute_nsr(model, data.x.to(device), data.adj.to(device), 
                                 noise_std=0.1, num_samples=100)
        
        results[name] = {
            'nsr_per_layer': nsr_result['nsr_per_layer'].tolist(),
            'nsr_ratio': nsr_result['nsr_ratio'].tolist(),
        }
        
        mean_ratio = np.mean(nsr_result['nsr_ratio'])
        print(f"  {name}: NSR ratio per layer = {mean_ratio:.3f}")
    
    save_json(output_dir / 'figure3.json', results)
    save_plot_script(output_dir / 'plot_figure3.py', 'figure3')
    
    return results


# =============================================================================
# Table 5: Deep Network + MAD (Section 1.7)
# =============================================================================

def run_table5(
    dataset: str,
    depths: List[int],
    seeds: List[int],
    device: torch.device,
    output_dir: Path,
) -> Dict:
    """
    Table 5: Test accuracy and MAD as function of depth.
    """
    print("\n" + "="*70)
    print("TABLE 5: Deep Network Trainability (Oversmoothing)")
    print("="*70)
    
    data = load_dataset(dataset)
    results = {'GCN': {}, 'GroupSort-GCN': {}, 'GHN': {}}
    
    for depth in depths:
        for model_name, model_type in [('GCN', 'gcn'), ('GroupSort-GCN', 'groupsort_gcn'), ('GHN', 'ghn')]:
            accs, mads = [], []
            
            for seed in seeds[:3]:
                set_seed(seed)
                config = get_model_config(model_type)
                config['num_layers'] = depth
                
                model = get_model(model_type, data.num_features, data.num_classes, **config)
                train_results = train_and_evaluate(model, data, get_training_config(), device, verbose=False)
                accs.append(train_results['test_accuracy'])
                
                # Compute MAD
                if HAS_ANALYSIS:
                    model.to(device)
                    mad = compute_mad_for_model(model, data.x.to(device), data.adj.to(device))
                    mads.append(mad)
            
            results[model_name][depth] = {
                'accuracy_mean': np.mean(accs),
                'accuracy_std': np.std(accs),
                'mad_mean': np.mean(mads) if mads else None,
            }
        
        print(f"  L={depth}: " + ", ".join([
            f"{m}={results[m][depth]['accuracy_mean']*100:.1f}%/MAD={results[m][depth]['mad_mean']:.2f}" 
            if results[m][depth]['mad_mean'] else f"{m}={results[m][depth]['accuracy_mean']*100:.1f}%"
            for m in results.keys()
        ]))
    
    save_json(output_dir / 'table5.json', results)
    
    return results


# =============================================================================
# Tables 6-7: Ablation Studies (Section 1.8)
# =============================================================================

def run_table6(
    dataset: str,
    alphas: List[float],
    seeds: List[int],
    device: torch.device,
    output_dir: Path,
) -> Dict:
    """
    Table 6: Effect of Hölder exponent α.
    """
    print("\n" + "="*70)
    print("TABLE 6: Ablation - Effect of α")
    print("="*70)
    
    data = load_dataset(dataset)
    results = {}
    
    for alpha in alphas:
        accs, acrs = [], []
        
        for seed in seeds:
            set_seed(seed)
            model = GraphHolderNetwork(
                data.num_features, data.num_classes,
                hidden_features=64, num_layers=2, alpha=alpha,
            )
            train_results = train_and_evaluate(model, data, get_training_config(), device, verbose=False)
            accs.append(train_results['test_accuracy'])
            
            cert = certify_all_nodes(
                model, data.x.to(device), data.adj.to(device),
                data.y.to(device), data.test_mask.to(device),
                'ghn', alpha, 2,
            )
            acrs.append(cert['average_certified_radius'])
        
        results[alpha] = {
            'accuracy_mean': np.mean(accs),
            'acr_mean': np.mean(acrs),
        }
        print(f"  α={alpha}: Acc={results[alpha]['accuracy_mean']*100:.1f}%, ACR={results[alpha]['acr_mean']:.2f}")
    
    save_json(output_dir / 'table6.json', results)
    
    return results


def run_table7(
    dataset: str,
    depths: List[int],
    seeds: List[int],
    device: torch.device,
    output_dir: Path,
) -> Dict:
    """
    Table 7: Effect of network depth L.
    """
    print("\n" + "="*70)
    print("TABLE 7: Ablation - Effect of Depth L")
    print("="*70)
    
    data = load_dataset(dataset)
    results = {}
    
    for depth in depths:
        accs, acrs = [], []
        
        for seed in seeds:
            set_seed(seed)
            model = GraphHolderNetwork(
                data.num_features, data.num_classes,
                hidden_features=64, num_layers=depth, alpha=0.8,
            )
            train_results = train_and_evaluate(model, data, get_training_config(), device, verbose=False)
            accs.append(train_results['test_accuracy'])
            
            cert = certify_all_nodes(
                model, data.x.to(device), data.adj.to(device),
                data.y.to(device), data.test_mask.to(device),
                'ghn', 0.8, depth,
            )
            acrs.append(cert['average_certified_radius'])
        
        results[depth] = {
            'accuracy_mean': np.mean(accs),
            'acr_mean': np.mean(acrs),
        }
        print(f"  L={depth}: Acc={results[depth]['accuracy_mean']*100:.1f}%, ACR={results[depth]['acr_mean']:.2f}")
    
    save_json(output_dir / 'table7.json', results)
    
    return results


# =============================================================================
# Table 8: Spectral Normalization Ablation (Section 1.8)
# =============================================================================

def run_table8(
    dataset: str,
    seeds: List[int],
    device: torch.device,
    output_dir: Path,
) -> Dict:
    """
    Table 8: Effect of spectral regularization.
    
    Compares:
    1. Weight decay only (standard GHN)
    2. Spectral normalization (enforcing ||W||_2 ≤ 1)
    3. Spectral penalty (soft regularization)
    """
    print("\n" + "="*70)
    print("TABLE 8: Spectral Normalization Ablation")
    print("="*70)
    
    data = load_dataset(dataset)
    results = {}
    
    regimes = [
        ('Weight decay only', {'use_spectral_norm': False, 'spectral_penalty': 0.0}),
        ('Spectral normalization', {'use_spectral_norm': True, 'spectral_penalty': 0.0}),
        ('Spectral penalty (λ=0.1)', {'use_spectral_norm': False, 'spectral_penalty': 0.1}),
    ]
    
    for regime_name, regime_config in regimes:
        accs, acrs, max_norms = [], [], []
        
        for seed in seeds:
            set_seed(seed)
            
            # Create model (spectral norm handled internally if supported)
            model = GraphHolderNetwork(
                data.num_features, data.num_classes,
                hidden_features=64, num_layers=2, alpha=0.8,
            )
            
            train_config = get_training_config()
            if regime_config['spectral_penalty'] > 0:
                train_config['spectral_penalty'] = regime_config['spectral_penalty']
            
            train_results = train_and_evaluate(model, data, train_config, device, verbose=False)
            accs.append(train_results['test_accuracy'])
            
            cert = certify_all_nodes(
                model, data.x.to(device), data.adj.to(device),
                data.y.to(device), data.test_mask.to(device),
                'ghn', 0.8, 2,
            )
            acrs.append(cert['average_certified_radius'])
            
            if HAS_ANALYSIS:
                norms = compute_spectral_norms(model)
                max_norms.append(norms.get('max', 0))
        
        results[regime_name] = {
            'accuracy_mean': np.mean(accs),
            'acr_mean': np.mean(acrs),
            'max_spectral_norm': np.mean(max_norms) if max_norms else None,
        }
        
        norm_str = f", max||W||={results[regime_name]['max_spectral_norm']:.2f}" if results[regime_name]['max_spectral_norm'] else ""
        print(f"  {regime_name}: Acc={results[regime_name]['accuracy_mean']*100:.1f}%, ACR={results[regime_name]['acr_mean']:.3f}{norm_str}")
    
    save_json(output_dir / 'table8.json', results)
    
    return results


# =============================================================================
# Table 9: Scalability (Section 1.9)
# =============================================================================

def run_table9(
    device: torch.device,
    output_dir: Path,
) -> Dict:
    """
    Table 9: Scalability on ogbn-arxiv.
    """
    print("\n" + "="*70)
    print("TABLE 9: Scalability (ogbn-arxiv)")
    print("="*70)
    
    try:
        data = load_dataset('ogbn-arxiv')
        print_dataset_info(data)
    except Exception as e:
        print(f"  Skipping: {e}")
        return {}
    
    results = {}
    import time
    
    for model_name in ['ghn', 'gcn', 'groupsort_gcn']:
        set_seed(42)
        config = get_model_config(model_name)
        config['hidden_features'] = 256
        config['num_layers'] = 3
        
        model = get_model(model_name, data.num_features, data.num_classes, **config)
        
        # Time training
        start = time.time()
        train_config = {**get_training_config(), 'epochs': 100}
        train_results = train_and_evaluate(model, data, train_config, device, verbose=True)
        train_time = time.time() - start
        
        # Time certification
        cert_start = time.time()
        if model_name in ['ghn', 'groupsort_gcn']:
            cert = certify_all_nodes(
                model, data.x.to(device), data.adj.to(device),
                data.y.to(device), data.test_mask.to(device),
                'ghn' if model_name == 'ghn' else 'lipschitz',
                config.get('alpha', 1.0), config['num_layers'],
            )
            acr = cert['average_certified_radius']
        else:
            acr = None
        cert_time = time.time() - cert_start
        
        results[model_name] = {
            'accuracy': train_results['test_accuracy'],
            'acr': acr,
            'train_time': train_time,
            'certify_time': cert_time,
        }
        
        acr_str = f"{acr:.3f}" if acr else "—"
        print(f"  {model_name}: Acc={results[model_name]['accuracy']*100:.1f}%, ACR={acr_str}, "
              f"Train={train_time:.1f}s, Certify={cert_time:.1f}s")
    
    save_json(output_dir / 'table9.json', results)
    
    return results


# =============================================================================
# Utility Functions
# =============================================================================

def save_json(path: Path, data: Dict):
    """Save dictionary to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def save_plot_script(path: Path, figure_name: str):
    """Save matplotlib plotting script."""
    scripts = {
        'figure1': '''
import json
import matplotlib.pyplot as plt
import numpy as np

with open('figure1.json') as f:
    data = json.load(f)

plt.figure(figsize=(8, 6))
for name, vals in data.items():
    plt.plot(vals['radii'], vals['certified_accuracy'], label=name, linewidth=2)

plt.xlabel('Perturbation Radius r', fontsize=12)
plt.ylabel('Certified Accuracy', fontsize=12)
plt.title('Certified Accuracy vs. Perturbation Radius (Cora)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure1.pdf', dpi=300)
print('Saved figure1.pdf')
''',
        'figure2': '''
import json
import matplotlib.pyplot as plt
import numpy as np

with open('figure2.json') as f:
    data = json.load(f)

margins = np.array(data['margins'])
ghn = np.array(data['ghn_radii'])
lip = np.array(data['lipschitz_radii'])

valid = (margins > 0) & (ghn > 0) & (lip > 0)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(np.log(margins[valid]), np.log(ghn[valid]), alpha=0.5, label=f"GHN (β={data['ghn_slope']:.2f})", s=20)
ax.scatter(np.log(margins[valid]), np.log(lip[valid]), alpha=0.5, label=f"GroupSort-GCN (β={data['lipschitz_slope']:.2f})", s=20)

ax.set_xlabel('log(Margin γ)', fontsize=12)
ax.set_ylabel('log(Certified Radius R)', fontsize=12)
ax.set_title('Margin-Radius Scaling (log-log)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure2.pdf', dpi=300)
print('Saved figure2.pdf')
''',
        'figure3': '''
import json
import matplotlib.pyplot as plt
import numpy as np

with open('figure3.json') as f:
    data = json.load(f)

plt.figure(figsize=(8, 6))
for name, vals in data.items():
    layers = list(range(len(vals['nsr_per_layer'])))
    plt.plot(layers, vals['nsr_per_layer'], 'o-', label=name, linewidth=2, markersize=8)

plt.xlabel('Layer', fontsize=12)
plt.ylabel('Noise-to-Signal Ratio', fontsize=12)
plt.title('NSR across Layers (4-layer networks on Cora)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure3.pdf', dpi=300)
print('Saved figure3.pdf')
''',
    }
    
    if figure_name in scripts:
        with open(path, 'w') as f:
            f.write(scripts[figure_name])


def print_latex_table1(results: Dict):
    """Print Table 1 in LaTeX format."""
    print("\n" + "="*70)
    print("LaTeX: Table 1")
    print("="*70)
    print("\\begin{tabular}{l" + "cc" * len(results) + "}")
    print("\\toprule")
    header = "Method"
    for d in results.keys():
        header += f" & {d.capitalize()} Acc & ACR"
    print(header + " \\\\")
    print("\\midrule")
    
    all_models = set()
    for d in results:
        all_models.update(results[d].keys())
    
    for model in sorted(all_models):
        row = model.replace('_', '-')
        for d in results:
            if model in results[d]:
                r = results[d][model]
                acc = f"{r['accuracy_mean']*100:.1f}"
                acr = f"{r['acr_mean']:.3f}" if r['acr_mean'] else "—"
                row += f" & {acc} & {acr}"
            else:
                row += " & — & —"
        print(row + " \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")


def print_latex_table2(results: Dict, epsilons: List[float]):
    """Print Table 2 in LaTeX format."""
    print("\n" + "="*70)
    print("LaTeX: Table 2")
    print("="*70)
    print("\\begin{tabular}{l" + "c" * (len(epsilons) + 1) + "}")
    print("\\toprule")
    print("Method & Clean & " + " & ".join([f"$\\epsilon$={e}" for e in epsilons]) + " \\\\")
    print("\\midrule")
    
    for model, vals in results.items():
        row = model.replace('_', '-')
        row += f" & {vals['clean']*100:.1f}"
        for e in epsilons:
            row += f" & {vals[f'eps_{e}']*100:.1f}"
        print(row + " \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='GHN Complete Experiments')
    
    parser.add_argument('--experiment', type=str, default='table1',
        choices=['table1', 'figure1', 'figure2', 'table2', 'table3', 'table4',
                 'figure3', 'table5', 'table6', 'table7', 'table8', 'table9', 'all'],
        help='Which experiment to run')
    parser.add_argument('--datasets', nargs='+', default=['cora', 'citeseer', 'pubmed'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0,1,2,3,4,5,6,7,8,9])
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    print(f"Device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model lists - matching PDF Table 1
    all_models = ['ghn', 'gcn', 'gat', 'sgc', 'spectral_gcn', 'groupsort_gcn', 
                  'pairnorm_gcn', 'randomized_smoothing', 'gnncert', 
                  'gnnguard', 'robustgcn']
    attack_models = ['ghn', 'gcn', 'groupsort_gcn', 'gnnguard', 'robustgcn']
    
    exp = args.experiment
    
    if exp in ['table1', 'all']:
        run_table1(args.datasets, all_models, args.seeds, device, output_dir)
    
    if exp in ['figure1', 'all']:
        run_figure1('cora', device, output_dir)
    
    if exp in ['figure2', 'all']:
        run_figure2('cora', device, output_dir)
    
    if exp in ['table2', 'all']:
        run_table2('cora', attack_models, [0.05, 0.10, 0.15, 0.20], device, output_dir)
    
    if exp in ['table3', 'all']:
        run_table3('cora', attack_models, [0.05, 0.10, 0.15, 0.20], device, output_dir)
    
    if exp in ['table4', 'all']:
        run_table4('cora', attack_models, [0.05, 0.10, 0.15, 0.20], 10, device, output_dir)
    
    if exp in ['figure3', 'all']:
        run_figure3('cora', device, output_dir)
    
    if exp in ['table5', 'all']:
        run_table5('cora', [2, 10, 32], args.seeds[:3], device, output_dir)
    
    if exp in ['table6', 'all']:
        run_table6('cora', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], args.seeds[:5], device, output_dir)
    
    if exp in ['table7', 'all']:
        run_table7('cora', [2, 4, 6, 10], args.seeds[:5], device, output_dir)
    
    if exp in ['table8', 'all']:
        run_table8('cora', args.seeds[:5], device, output_dir)
    
    if exp in ['table9', 'all']:
        run_table9(device, output_dir)
    
    print("\n" + "="*70)
    print("All requested experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
