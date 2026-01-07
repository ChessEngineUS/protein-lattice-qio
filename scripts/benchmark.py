#!/usr/bin/env python3
"""Automated benchmarking script for protein lattice folding.

Measures:
- Wall-clock time for optimization algorithms
- Peak memory usage
- Solution quality (final energy)
- Device detection (GPU vs CPU)

Outputs:
- outputs/benchmark_results.json
- outputs/benchmark_plot.png

Usage:
    python scripts/benchmark.py
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
import psutil

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lattice import SquareLattice2D
from energy import HPEnergyFunction
from optimizer import GreedyOptimizer, SimulatedAnnealingOptimizer, QuantumInspiredOptimizer
from utils import set_seed, detect_device, generate_hp_sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def benchmark_optimizer(optimizer, sequence: str, lattice, energy_fn, name: str) -> dict:
    """Benchmark a single optimizer.
    
    Returns dict with timing, memory, and quality metrics.
    """
    logger.info(f"Benchmarking {name}...")
    
    mem_before = get_memory_usage()
    
    start_time = time.time()
    coords, energy, trajectory = optimizer.optimize(sequence, lattice, energy_fn)
    elapsed_time = time.time() - start_time
    
    mem_after = get_memory_usage()
    mem_used = mem_after - mem_before
    
    results = {
        "algorithm": name,
        "sequence_length": len(sequence),
        "final_energy": float(energy),
        "elapsed_time_seconds": elapsed_time,
        "memory_used_mb": mem_used,
        "optimization_steps": len(trajectory),
        "convergence_step": _find_convergence_step(trajectory)
    }
    
    logger.info(f"{name} completed in {elapsed_time:.2f}s with energy {energy:.4f}")
    
    return results


def _find_convergence_step(trajectory: list) -> int:
    """Find step where energy stops improving significantly."""
    if len(trajectory) < 10:
        return len(trajectory)
    
    threshold = 0.001
    for i in range(10, len(trajectory)):
        recent_improvement = abs(trajectory[i] - trajectory[i-10]) / 10
        if recent_improvement < threshold:
            return i
    
    return len(trajectory)


def main():
    """Run comprehensive benchmarks."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    set_seed(42)
    
    device = detect_device()
    
    test_sequences = [
        generate_hp_sequence(15, h_ratio=0.5),
        generate_hp_sequence(20, h_ratio=0.55),
        generate_hp_sequence(25, h_ratio=0.6),
    ]
    
    logger.info(f"Testing on {len(test_sequences)} sequences")
    
    lattice = SquareLattice2D()
    energy_fn = HPEnergyFunction()
    
    all_results = []
    
    for seq_idx, sequence in enumerate(test_sequences):
        logger.info(f"\n=== Testing sequence {seq_idx + 1}/{len(test_sequences)} (length={len(sequence)}) ===")
        
        optimizers = [
            (GreedyOptimizer(device=device), "Greedy"),
            (SimulatedAnnealingOptimizer(
                initial_temp=10.0, 
                final_temp=0.1, 
                steps=1000,
                device=device
            ), "Simulated Annealing"),
            (QuantumInspiredOptimizer(
                initial_temp=10.0,
                final_temp=0.1,
                steps=1000,
                tunnel_rate=0.1,
                device=device
            ), "Quantum-Inspired"),
        ]
        
        for optimizer, name in optimizers:
            result = benchmark_optimizer(optimizer, sequence, lattice, energy_fn, name)
            result["sequence_index"] = seq_idx
            result["sequence"] = sequence
            all_results.append(result)
    
    benchmark_data = {
        "device": device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": all_results
    }
    
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    # Generate comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    algorithms = ["Greedy", "Simulated Annealing", "Quantum-Inspired"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Plot 1: Runtime vs sequence length
    ax = axes[0, 0]
    for alg, color in zip(algorithms, colors):
        alg_data = [r for r in all_results if r["algorithm"] == alg]
        lengths = [r["sequence_length"] for r in alg_data]
        times = [r["elapsed_time_seconds"] for r in alg_data]
        ax.plot(lengths, times, 'o-', label=alg, color=color, linewidth=2, markersize=8)
    ax.set_xlabel("Sequence Length", fontsize=11)
    ax.set_ylabel("Runtime (seconds)", fontsize=11)
    ax.set_title("Runtime vs Sequence Length", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Final energy comparison
    ax = axes[0, 1]
    for seq_idx in range(len(test_sequences)):
        seq_data = [r for r in all_results if r["sequence_index"] == seq_idx]
        energies = [r["final_energy"] for r in seq_data]
        x_pos = np.arange(len(algorithms)) + seq_idx * 0.25
        ax.bar(x_pos, energies, width=0.25, 
               label=f"Seq {seq_idx+1} (L={seq_data[0]['sequence_length']})")
    ax.set_xticks(np.arange(len(algorithms)) + 0.25)
    ax.set_xticklabels(algorithms)
    ax.set_ylabel("Final Energy", fontsize=11)
    ax.set_title("Solution Quality Comparison", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Memory usage
    ax = axes[1, 0]
    for alg, color in zip(algorithms, colors):
        alg_data = [r for r in all_results if r["algorithm"] == alg]
        lengths = [r["sequence_length"] for r in alg_data]
        memory = [r["memory_used_mb"] for r in alg_data]
        ax.plot(lengths, memory, 'o-', label=alg, color=color, linewidth=2, markersize=8)
    ax.set_xlabel("Sequence Length", fontsize=11)
    ax.set_ylabel("Memory Used (MB)", fontsize=11)
    ax.set_title("Memory Usage vs Sequence Length", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Convergence speed
    ax = axes[1, 1]
    for alg, color in zip(algorithms, colors):
        alg_data = [r for r in all_results if r["algorithm"] == alg]
        lengths = [r["sequence_length"] for r in alg_data]
        conv_steps = [r["convergence_step"] for r in alg_data]
        ax.plot(lengths, conv_steps, 'o-', label=alg, color=color, linewidth=2, markersize=8)
    ax.set_xlabel("Sequence Length", fontsize=11)
    ax.set_ylabel("Steps to Convergence", fontsize=11)
    ax.set_title("Convergence Speed", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "benchmark_plot.png"
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Benchmark plot saved to {plot_path}")
    
    logger.info("\n=== BENCHMARK SUMMARY ===")
    logger.info(f"Device: {device}")
    for alg in algorithms:
        alg_data = [r for r in all_results if r["algorithm"] == alg]
        avg_time = np.mean([r["elapsed_time_seconds"] for r in alg_data])
        avg_energy = np.mean([r["final_energy"] for r in alg_data])
        logger.info(f"{alg}: avg_time={avg_time:.2f}s, avg_energy={avg_energy:.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
