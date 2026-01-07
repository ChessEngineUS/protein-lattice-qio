"""
Visualization utilities for protein folding simulations.

Provides plotting functions for conformations, energy landscapes, and
algorithm comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List


def plot_conformation(coords: np.ndarray, sequence: str, title: str = "Protein Conformation") -> plt.Figure:
    """
    Plot a protein conformation on a lattice.
    
    Args:
        coords: Array of shape (N, D) with coordinates
        sequence: Protein sequence (for coloring H/P residues)
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    dims = coords.shape[1]
    
    if dims == 2:
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot backbone
        ax.plot(coords[:, 0], coords[:, 1], 'k-', alpha=0.3, linewidth=2)
        
        # Plot residues with colors
        colors = ['red' if s == 'H' else 'blue' for s in sequence]
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=200, 
                  edgecolors='black', linewidth=2, zorder=10)
        
        # Add labels
        for i, (x, y) in enumerate(coords):
            ax.text(x, y, str(i), ha='center', va='center', 
                   fontsize=8, fontweight='bold', color='white')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
    elif dims == 3:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot backbone
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 
               'k-', alpha=0.3, linewidth=2)
        
        # Plot residues
        colors = ['red' if s == 'H' else 'blue' for s in sequence]
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                  c=colors, s=200, edgecolors='black', linewidth=2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
    
    plt.tight_layout()
    return fig


def plot_energy_trajectory(trajectories: Dict[str, List[float]]) -> plt.Figure:
    """
    Plot energy trajectories for multiple algorithms.
    
    Args:
        trajectories: Dict mapping algorithm name to energy trajectory
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, trajectory in trajectories.items():
        ax.plot(trajectory, label=name, linewidth=2)
    
    ax.set_xlabel('Optimization Step', fontsize=12)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title('Energy Convergence Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_comparison(results: Dict, sequences: List[str]) -> plt.Figure:
    """
    Plot comparison of algorithm performance.
    
    Args:
        results: Dict mapping algorithm name to result dict
        sequences: List of test sequences
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Final energies
    names = list(results.keys())
    energies = [results[name]["energy"] for name in names]
    
    ax1.bar(names, energies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_ylabel('Final Energy', fontsize=12)
    ax1.set_title('Final Energy Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Convergence steps (when energy stops improving)
    steps = [len(results[name]["trajectory"]) for name in names]
    
    ax2.bar(names, steps, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_ylabel('Optimization Steps', fontsize=12)
    ax2.set_title('Algorithm Steps', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig
