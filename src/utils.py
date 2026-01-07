"""Utility functions for protein lattice folding simulations.

Provides helper functions for random seed setting, device detection,
and sequence generation.
"""

import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def detect_device() -> str:
    """Detect available compute device (GPU/CPU).
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = 'cpu'
        logger.info("No GPU detected, using CPU")
    
    return device


def generate_hp_sequence(length: int, h_ratio: float = 0.5, seed: int = None) -> str:
    """Generate a random HP sequence.
    
    Args:
        length: Sequence length
        h_ratio: Ratio of hydrophobic (H) residues
        seed: Optional random seed
        
    Returns:
        HP sequence string (e.g., "HPHPPHHPHH")
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_h = int(length * h_ratio)
    n_p = length - n_h
    
    residues = ['H'] * n_h + ['P'] * n_p
    np.random.shuffle(residues)
    
    return ''.join(residues)


def calculate_sequence_hydrophobicity(sequence: str) -> float:
    """Calculate the hydrophobicity ratio of a sequence.
    
    Args:
        sequence: HP sequence string
        
    Returns:
        Ratio of H residues (0.0 to 1.0)
    """
    return sequence.count('H') / len(sequence)
