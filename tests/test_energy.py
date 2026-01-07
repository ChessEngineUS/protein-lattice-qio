"""Unit tests for energy functions."""

import numpy as np
import pytest
from src.energy import HPEnergyFunction
from src.lattice import SquareLattice2D


def test_hp_energy_no_contacts():
    """Test HP energy with no contacts."""
    energy_fn = HPEnergyFunction()
    sequence = "HPHPH"
    coords = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
    
    contacts = []
    energy = energy_fn.calculate(coords, sequence, contacts)
    
    assert energy == 0.0


def test_hp_energy_hh_contact():
    """Test HP energy with H-H contact."""
    energy_fn = HPEnergyFunction(hh_energy=-1.0)
    sequence = "HH"
    coords = np.array([[0, 0], [1, 0]])
    
    contacts = []
    energy = energy_fn.calculate(coords, sequence, contacts)
    assert energy == 0.0
    
    sequence = "HPPH"
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    contacts = [(0, 3)]
    energy = energy_fn.calculate(coords, sequence, contacts)
    assert energy == -1.0


def test_hp_energy_hp_contact():
    """Test HP energy with H-P contact (should be 0)."""
    energy_fn = HPEnergyFunction()
    sequence = "HP"
    coords = np.array([[0, 0], [1, 0]])
    
    contacts = [(0, 1)]
    energy = energy_fn.calculate(coords, sequence, contacts)
    assert energy == 0.0


def test_hp_energy_multiple_contacts():
    """Test HP energy with multiple H-H contacts."""
    energy_fn = HPEnergyFunction(hh_energy=-1.0)
    sequence = "HPHHPH"
    coords = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]])
    
    contacts = [(0, 2), (2, 5)]
    energy = energy_fn.calculate(coords, sequence, contacts)
    assert energy == -2.0
