"""Optimization algorithms for protein lattice folding.

Implements:
- Greedy search baseline
- Simulated annealing (classical)
- Quantum-inspired annealing (with tunneling)

References:
- Kirkpatrick et al. (1983), DOI: 10.1126/science.220.4598.671
  Optimization by simulated annealing
- Kadowaki & Nishimori (1998), DOI: 10.1103/PhysRevE.58.5355
  Quantum annealing in the transverse Ising model
"""

from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np
import torch
import logging
from tqdm import tqdm

from .lattice import Lattice
from .energy import EnergyFunction

logger = logging.getLogger(__name__)


class Optimizer(ABC):
    """Abstract base class for optimization algorithms."""
    
    def __init__(self, device: str = "cpu"):
        """Initialize optimizer with device."""
        self.device = device
    
    @abstractmethod
    def optimize(self, sequence: str, lattice: Lattice, 
                energy_fn: EnergyFunction) -> Tuple[np.ndarray, float, List[float]]:
        """Optimize protein conformation.
        
        Args:
            sequence: Protein sequence string (e.g., "HPHPPHHPHH")
            lattice: Lattice model
            energy_fn: Energy function
            
        Returns:
            best_coords: Best conformation found (N, D)
            best_energy: Energy of best conformation
            trajectory: Energy values over optimization steps
        """
        pass
    
    def _initialize_conformation(self, sequence: str, lattice: Lattice) -> np.ndarray:
        """Initialize a random self-avoiding conformation.
        
        Uses a self-avoiding random walk on the lattice.
        """
        n = len(sequence)
        dims = lattice.get_dimensions()
        coords = np.zeros((n, dims), dtype=int)
        
        occupied = {tuple(coords[0])}
        
        for i in range(1, n):
            neighbors = lattice.get_neighbors(tuple(coords[i-1]))
            available = [n for n in neighbors if tuple(n) not in occupied]
            
            if not available:
                occupied.remove(tuple(coords[i-1]))
                if i > 1:
                    i -= 1
                continue
            
            next_coord = available[np.random.randint(len(available))]
            coords[i] = next_coord
            occupied.add(tuple(next_coord))
        
        return coords


class GreedyOptimizer(Optimizer):
    """Greedy search baseline optimizer.
    
    Builds conformation step-by-step, choosing the move that minimizes
    immediate energy at each step.
    """
    
    def optimize(self, sequence: str, lattice: Lattice, 
                energy_fn: EnergyFunction) -> Tuple[np.ndarray, float, List[float]]:
        """Optimize using greedy search."""
        n = len(sequence)
        dims = lattice.get_dimensions()
        coords = np.zeros((n, dims), dtype=int)
        
        trajectory = []
        occupied = {tuple(coords[0])}
        
        for i in range(1, n):
            neighbors = lattice.get_neighbors(tuple(coords[i-1]))
            available = [n for n in neighbors if tuple(n) not in occupied]
            
            if not available:
                available = lattice.get_neighbors(tuple(coords[i-1]))
            
            best_energy = float('inf')
            best_coord = available[0]
            
            for candidate in available:
                coords[i] = candidate
                contacts = lattice.calculate_contacts(coords[:i+1], sequence[:i+1])
                energy = energy_fn.calculate(coords[:i+1], sequence[:i+1], contacts)
                
                if energy < best_energy:
                    best_energy = energy
                    best_coord = candidate
            
            coords[i] = best_coord
            occupied.add(tuple(best_coord))
            trajectory.append(best_energy)
        
        contacts = lattice.calculate_contacts(coords, sequence)
        final_energy = energy_fn.calculate(coords, sequence, contacts)
        
        return coords, final_energy, trajectory


class SimulatedAnnealingOptimizer(Optimizer):
    """Simulated annealing optimizer.
    
    Classical optimization inspired by metallurgical annealing process.
    
    Reference:
    Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983).
    Science, 220(4598), 671-680. DOI: 10.1126/science.220.4598.671
    """
    
    def __init__(self, initial_temp: float = 10.0, final_temp: float = 0.1,
                 steps: int = 5000, device: str = "cpu"):
        """Initialize simulated annealing optimizer.
        
        Args:
            initial_temp: Starting temperature
            final_temp: Final temperature
            steps: Number of optimization steps
            device: Device for computation
        """
        super().__init__(device)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.steps = steps
    
    def optimize(self, sequence: str, lattice: Lattice, 
                energy_fn: EnergyFunction) -> Tuple[np.ndarray, float, List[float]]:
        """Optimize using simulated annealing."""
        current_coords = self._initialize_conformation(sequence, lattice)
        contacts = lattice.calculate_contacts(current_coords, sequence)
        current_energy = energy_fn.calculate(current_coords, sequence, contacts)
        
        best_coords = current_coords.copy()
        best_energy = current_energy
        
        trajectory = [current_energy]
        
        temperatures = np.linspace(self.initial_temp, self.final_temp, self.steps)
        
        for step in tqdm(range(self.steps), desc="Simulated Annealing", leave=False):
            temp = temperatures[step]
            
            candidate_coords = current_coords.copy()
            move_idx = np.random.randint(1, len(sequence))
            
            neighbors = lattice.get_neighbors(tuple(current_coords[move_idx - 1]))
            available = [n for n in neighbors if tuple(n) not in set(map(tuple, current_coords))]
            
            if available:
                candidate_coords[move_idx] = available[np.random.randint(len(available))]
                
                if lattice.is_valid_conformation(candidate_coords):
                    contacts = lattice.calculate_contacts(candidate_coords, sequence)
                    candidate_energy = energy_fn.calculate(candidate_coords, sequence, contacts)
                    
                    delta_e = candidate_energy - current_energy
                    if delta_e < 0 or np.random.random() < np.exp(-delta_e / temp):
                        current_coords = candidate_coords
                        current_energy = candidate_energy
                        
                        if current_energy < best_energy:
                            best_coords = current_coords.copy()
                            best_energy = current_energy
            
            trajectory.append(best_energy)
        
        return best_coords, best_energy, trajectory


class QuantumInspiredOptimizer(Optimizer):
    """Quantum-inspired annealing optimizer with tunneling.
    
    Extends classical simulated annealing with quantum-inspired tunneling
    that allows escaping local minima through non-local moves.
    
    Reference:
    Kadowaki, T., & Nishimori, H. (1998). Physical Review E, 58(5), 5355.
    DOI: 10.1103/PhysRevE.58.5355
    """
    
    def __init__(self, initial_temp: float = 10.0, final_temp: float = 0.1,
                 steps: int = 5000, tunnel_rate: float = 0.1, device: str = "cpu"):
        """Initialize quantum-inspired optimizer.
        
        Args:
            initial_temp: Starting temperature
            final_temp: Final temperature
            steps: Number of optimization steps
            tunnel_rate: Probability of quantum tunneling move
            device: Device for computation
        """
        super().__init__(device)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.steps = steps
        self.tunnel_rate = tunnel_rate
    
    def optimize(self, sequence: str, lattice: Lattice, 
                energy_fn: EnergyFunction) -> Tuple[np.ndarray, float, List[float]]:
        """Optimize using quantum-inspired annealing."""
        current_coords = self._initialize_conformation(sequence, lattice)
        contacts = lattice.calculate_contacts(current_coords, sequence)
        current_energy = energy_fn.calculate(current_coords, sequence, contacts)
        
        best_coords = current_coords.copy()
        best_energy = current_energy
        
        trajectory = [current_energy]
        
        temperatures = np.linspace(self.initial_temp, self.final_temp, self.steps)
        
        for step in tqdm(range(self.steps), desc="Quantum-Inspired Annealing", leave=False):
            temp = temperatures[step]
            
            if np.random.random() < self.tunnel_rate:
                segment_start = np.random.randint(1, len(sequence) - 1)
                segment_end = min(segment_start + np.random.randint(2, 5), len(sequence))
                
                candidate_coords = current_coords.copy()
                for i in range(segment_start, segment_end):
                    neighbors = lattice.get_neighbors(tuple(candidate_coords[i-1]))
                    available = [n for n in neighbors if tuple(n) not in set(map(tuple, candidate_coords[:i]))]
                    if available:
                        candidate_coords[i] = available[np.random.randint(len(available))]
            else:
                candidate_coords = current_coords.copy()
                move_idx = np.random.randint(1, len(sequence))
                
                neighbors = lattice.get_neighbors(tuple(current_coords[move_idx - 1]))
                available = [n for n in neighbors if tuple(n) not in set(map(tuple, current_coords))]
                
                if available:
                    candidate_coords[move_idx] = available[np.random.randint(len(available))]
            
            if lattice.is_valid_conformation(candidate_coords):
                contacts = lattice.calculate_contacts(candidate_coords, sequence)
                candidate_energy = energy_fn.calculate(candidate_coords, sequence, contacts)
                
                delta_e = candidate_energy - current_energy
                if delta_e < 0 or np.random.random() < np.exp(-delta_e / temp):
                    current_coords = candidate_coords
                    current_energy = candidate_energy
                    
                    if current_energy < best_energy:
                        best_coords = current_coords.copy()
                        best_energy = current_energy
            
            trajectory.append(best_energy)
        
        return best_coords, best_energy, trajectory
