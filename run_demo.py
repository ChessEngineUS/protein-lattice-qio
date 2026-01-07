#!/usr/bin/env python3
"""Main entry point for the protein lattice folding demo.

This script:
1. Generates synthetic HP sequences
2. Runs multiple optimization algorithms
3. Produces visualizations and saves results to outputs/

Usage:
    python run_demo.py
"""

import os
import sys
import logging
import json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lattice import SquareLattice2D, CubicLattice3D
from energy import HPEnergyFunction
from optimizer import GreedyOptimizer, SimulatedAnnealingOptimizer, QuantumInspiredOptimizer
from visualization import plot_conformation, plot_energy_trajectory, plot_comparison
from utils import set_seed, detect_device, generate_hp_sequence

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the complete demo pipeline."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    set_seed(42)
    
    device = detect_device()
    logger.info(f"Using device: {device}")
    
    logger.info("Generating synthetic HP sequences...")
    sequences = [
        generate_hp_sequence(20, h_ratio=0.5),
        generate_hp_sequence(30, h_ratio=0.6),
        generate_hp_sequence(25, h_ratio=0.55),
    ]
    
    logger.info(f"Generated {len(sequences)} sequences: {sequences}")
    
    test_sequence = sequences[1]
    logger.info(f"Running detailed analysis on sequence: {test_sequence}")
    
    lattice = SquareLattice2D()
    energy_fn = HPEnergyFunction()
    
    optimizers = {
        "Greedy": GreedyOptimizer(device=device),
        "Simulated Annealing": SimulatedAnnealingOptimizer(
            initial_temp=10.0,
            final_temp=0.1,
            steps=5000,
            device=device
        ),
        "Quantum-Inspired": QuantumInspiredOptimizer(
            initial_temp=10.0,
            final_temp=0.1,
            steps=5000,
            tunnel_rate=0.1,
            device=device
        ),
    }
    
    results = {}
    for name, optimizer in optimizers.items():
        logger.info(f"Running {name} optimizer...")
        coords, energy, trajectory = optimizer.optimize(
            test_sequence, lattice, energy_fn
        )
        results[name] = {
            "coords": coords,
            "energy": energy,
            "trajectory": trajectory
        }
        logger.info(f"{name} final energy: {energy:.4f}")
        
        fig = plot_conformation(coords, test_sequence, title=f"{name} - Energy: {energy:.4f}")
        fig.savefig(output_dir / f"conformation_{name.replace(' ', '_').lower()}.png", dpi=150)
        logger.info(f"Saved conformation plot for {name}")
    
    logger.info("Plotting energy trajectories...")
    fig = plot_energy_trajectory(
        {name: res["trajectory"] for name, res in results.items()}
    )
    fig.savefig(output_dir / "energy_trajectories.png", dpi=150)
    
    logger.info("Plotting algorithm comparison...")
    fig = plot_comparison(results, sequences)
    fig.savefig(output_dir / "algorithm_comparison.png", dpi=150)
    
    summary = {
        "sequence": test_sequence,
        "sequence_length": len(test_sequence),
        "device": str(device),
        "algorithms": {
            name: {
                "final_energy": float(res["energy"]),
                "steps": len(res["trajectory"])
            }
            for name, res in results.items()
        }
    }
    
    with open(output_dir / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Demo complete! Results saved to {output_dir}")
    logger.info(f"Best energy: {min(res['energy'] for res in results.values()):.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
