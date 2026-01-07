"""Energy function implementations for protein folding simulations.

Implements the HP (Hydrophobic-Polar) model and extensible energy framework.

References:
- Lau & Dill (1989), DOI: 10.1073/pnas.86.6.2050
  A lattice statistical mechanics model of the conformational and sequence spaces of proteins
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
import torch


class EnergyFunction(ABC):
    """Abstract base class for energy functions."""
    
    @abstractmethod
    def calculate(self, coords: np.ndarray, sequence: str, contacts: List[Tuple[int, int]]) -> float:
        """Calculate the energy of a conformation.
        
        Args:
            coords: Array of shape (N, D) with conformation coordinates
            sequence: Protein sequence string
            contacts: List of (i, j) contact pairs
            
        Returns:
            Energy value (lower is better)
        """
        pass


class HPEnergyFunction(EnergyFunction):
    """HP (Hydrophobic-Polar) model energy function.
    
    Energy is based on hydrophobic contacts:
    - H-H contact: -1 (favorable)
    - H-P contact: 0
    - P-P contact: 0
    
    The goal is to maximize H-H contacts (minimize energy).
    
    Reference:
    Lau, K. F., & Dill, K. A. (1989). PNAS, 86(6), 2050-2054.
    DOI: 10.1073/pnas.86.6.2050
    """
    
    def __init__(self, hh_energy: float = -1.0):
        """Initialize HP energy function.
        
        Args:
            hh_energy: Energy value for H-H contacts (typically negative)
        """
        self.hh_energy = hh_energy
    
    def calculate(self, coords: np.ndarray, sequence: str, contacts: List[Tuple[int, int]]) -> float:
        """Calculate HP model energy."""
        energy = 0.0
        
        for i, j in contacts:
            if sequence[i] == 'H' and sequence[j] == 'H':
                energy += self.hh_energy
        
        return energy
    
    def calculate_batch(self, coords_batch: torch.Tensor, sequence: str, 
                       lattice) -> torch.Tensor:
        """Calculate energies for a batch of conformations (GPU-accelerated).
        
        Args:
            coords_batch: Tensor of shape (B, N, D) where B is batch size
            sequence: Protein sequence string
            lattice: Lattice object for contact calculation
            
        Returns:
            Tensor of shape (B,) with energy values
        """
        batch_size = coords_batch.shape[0]
        energies = torch.zeros(batch_size, device=coords_batch.device)
        
        for b in range(batch_size):
            coords_np = coords_batch[b].cpu().numpy()
            contacts = lattice.calculate_contacts(coords_np, sequence)
            energies[b] = self.calculate(coords_np, sequence, contacts)
        
        return energies
