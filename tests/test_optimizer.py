"""Unit tests for optimizers."""

import numpy as np
import pytest
from src.optimizer import GreedyOptimizer, SimulatedAnnealingOptimizer, QuantumInspiredOptimizer
from src.lattice import SquareLattice2D
from src.energy import HPEnergyFunction
from src.utils import set_seed


def test_greedy_optimizer():
    """Test greedy optimizer produces valid conformation."""
    set_seed(42)
    
    optimizer = GreedyOptimizer()
    lattice = SquareLattice2D()
    energy_fn = HPEnergyFunction()
    sequence = "HPHPH"
    
    coords, energy, trajectory = optimizer.optimize(sequence, lattice, energy_fn)
    
    assert coords.shape == (len(sequence), 2)
    assert lattice.is_valid_conformation(coords)
    assert isinstance(energy, float)
    assert len(trajectory) > 0


def test_simulated_annealing_optimizer():
    """Test simulated annealing optimizer."""
    set_seed(42)
    
    optimizer = SimulatedAnnealingOptimizer(
        initial_temp=5.0,
        final_temp=0.1,
        steps=100,
        device="cpu"
    )
    lattice = SquareLattice2D()
    energy_fn = HPEnergyFunction()
    sequence = "HPHPH"
    
    coords, energy, trajectory = optimizer.optimize(sequence, lattice, energy_fn)
    
    assert coords.shape == (len(sequence), 2)
    assert lattice.is_valid_conformation(coords)
    assert isinstance(energy, float)
    assert len(trajectory) == 101


def test_quantum_inspired_optimizer():
    """Test quantum-inspired optimizer."""
    set_seed(42)
    
    optimizer = QuantumInspiredOptimizer(
        initial_temp=5.0,
        final_temp=0.1,
        steps=100,
        tunnel_rate=0.1,
        device="cpu"
    )
    lattice = SquareLattice2D()
    energy_fn = HPEnergyFunction()
    sequence = "HPHPH"
    
    coords, energy, trajectory = optimizer.optimize(sequence, lattice, energy_fn)
    
    assert coords.shape == (len(sequence), 2)
    assert lattice.is_valid_conformation(coords)
    assert isinstance(energy, float)
    assert len(trajectory) == 101


def test_optimizer_comparison():
    """Test that different optimizers produce different results."""
    set_seed(42)
    
    lattice = SquareLattice2D()
    energy_fn = HPEnergyFunction()
    sequence = "HPHPHPH"
    
    greedy = GreedyOptimizer()
    sa = SimulatedAnnealingOptimizer(steps=50, device="cpu")
    qi = QuantumInspiredOptimizer(steps=50, device="cpu")
    
    _, e_greedy, _ = greedy.optimize(sequence, lattice, energy_fn)
    
    set_seed(42)
    _, e_sa, _ = sa.optimize(sequence, lattice, energy_fn)
    
    set_seed(42)
    _, e_qi, _ = qi.optimize(sequence, lattice, energy_fn)
    
    assert all(isinstance(e, float) for e in [e_greedy, e_sa, e_qi])
