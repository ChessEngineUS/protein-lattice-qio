"""Protein Lattice Folding with Quantum-Inspired Optimization.

A toolkit for simulating protein folding on lattice models using
quantum-inspired optimization algorithms.
"""

__version__ = "1.0.0"
__author__ = "ChessEngineUS"

from .lattice import SquareLattice2D, CubicLattice3D
from .energy import HPEnergyFunction
from .optimizer import GreedyOptimizer, SimulatedAnnealingOptimizer, QuantumInspiredOptimizer

__all__ = [
    "SquareLattice2D",
    "CubicLattice3D",
    "HPEnergyFunction",
    "GreedyOptimizer",
    "SimulatedAnnealingOptimizer",
    "QuantumInspiredOptimizer",
]
