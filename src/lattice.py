"""Lattice model definitions for protein folding simulations.

Implements 2D square and 3D cubic lattices with self-avoiding walk constraints.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class Lattice(ABC):
    """Abstract base class for lattice models."""
    
    @abstractmethod
    def get_neighbors(self, coord: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Get valid neighboring positions on the lattice."""
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """Return the dimensionality of the lattice."""
        pass
    
    def is_valid_conformation(self, coords: np.ndarray) -> bool:
        """Check if a conformation is valid (self-avoiding).
        
        Args:
            coords: Array of shape (N, D) where N is sequence length, D is dimensions
            
        Returns:
            True if conformation is self-avoiding, False otherwise
        """
        unique_coords = set(map(tuple, coords))
        return len(unique_coords) == len(coords)
    
    def calculate_contacts(self, coords: np.ndarray, sequence: str) -> List[Tuple[int, int]]:
        """Calculate non-local contacts in a conformation.
        
        A contact is defined as two non-consecutive residues that are adjacent
        on the lattice.
        
        Args:
            coords: Array of shape (N, D) with conformation coordinates
            sequence: Protein sequence string
            
        Returns:
            List of (i, j) tuples representing contacts where i < j
        """
        contacts = []
        n = len(coords)
        
        for i in range(n):
            for j in range(i + 2, n):
                distance = np.sum(np.abs(coords[i] - coords[j]))
                if distance == 1:
                    contacts.append((i, j))
        
        return contacts


class SquareLattice2D(Lattice):
    """2D square lattice with 4-connectivity."""
    
    def get_neighbors(self, coord: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get 4-connected neighbors on 2D square lattice."""
        x, y = coord
        return [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]
    
    def get_dimensions(self) -> int:
        return 2


class CubicLattice3D(Lattice):
    """3D cubic lattice with 6-connectivity."""
    
    def get_neighbors(self, coord: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Get 6-connected neighbors on 3D cubic lattice."""
        x, y, z = coord
        return [
            (x + 1, y, z),
            (x - 1, y, z),
            (x, y + 1, z),
            (x, y - 1, z),
            (x, y, z + 1),
            (x, y, z - 1),
        ]
    
    def get_dimensions(self) -> int:
        return 3
