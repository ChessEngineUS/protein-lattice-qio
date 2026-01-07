"""Unit tests for lattice models."""

import numpy as np
import pytest
from src.lattice import SquareLattice2D, CubicLattice3D


def test_square_lattice_dimensions():
    """Test that 2D lattice reports correct dimensions."""
    lattice = SquareLattice2D()
    assert lattice.get_dimensions() == 2


def test_square_lattice_neighbors():
    """Test neighbor generation on 2D lattice."""
    lattice = SquareLattice2D()
    neighbors = lattice.get_neighbors((0, 0))
    
    assert len(neighbors) == 4
    assert (1, 0) in neighbors
    assert (-1, 0) in neighbors
    assert (0, 1) in neighbors
    assert (0, -1) in neighbors


def test_cubic_lattice_dimensions():
    """Test that 3D lattice reports correct dimensions."""
    lattice = CubicLattice3D()
    assert lattice.get_dimensions() == 3


def test_cubic_lattice_neighbors():
    """Test neighbor generation on 3D lattice."""
    lattice = CubicLattice3D()
    neighbors = lattice.get_neighbors((0, 0, 0))
    
    assert len(neighbors) == 6
    assert (1, 0, 0) in neighbors
    assert (-1, 0, 0) in neighbors


def test_valid_conformation():
    """Test validation of self-avoiding conformation."""
    lattice = SquareLattice2D()
    
    valid_coords = np.array([[0, 0], [1, 0], [1, 1], [2, 1]])
    assert lattice.is_valid_conformation(valid_coords)
    
    invalid_coords = np.array([[0, 0], [1, 0], [1, 1], [1, 0]])
    assert not lattice.is_valid_conformation(invalid_coords)


def test_calculate_contacts():
    """Test contact calculation."""
    lattice = SquareLattice2D()
    sequence = "HPHPH"
    
    coords = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
    contacts = lattice.calculate_contacts(coords, sequence)
    assert len(contacts) == 0
    
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 2]])
    contacts = lattice.calculate_contacts(coords, sequence)
    assert len(contacts) >= 1
    assert (0, 3) in contacts or (1, 3) in contacts
