# Quantum-Inspired Optimization for Protein Folding Lattice Models

[![CI](https://github.com/ChessEngineUS/protein-lattice-qio/actions/workflows/ci.yml/badge.svg)](https://github.com/ChessEngineUS/protein-lattice-qio/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository provides a complete, runnable toolkit for simulating protein folding on 2D and 3D lattice models using quantum-inspired optimization algorithms. It demonstrates how classical optimization techniques inspired by quantum mechanics (simulated annealing with tunneling-inspired perturbations) can tackle NP-hard combinatorial problems in structural biology.

**Why it's useful for STEM:**
- **Education**: Interactive visualizations help students understand the protein folding problem, energy landscapes, and optimization heuristics.
- **Research**: Provides a modular framework for testing new folding algorithms, comparing energy functions, and benchmarking performance.
- **Reproducibility**: All experiments run in Google Colab (free GPU) or on local CPU with pinned dependencies and automated benchmarks.

**Scientific Background:**
Protein folding on lattice models is a simplified yet computationally challenging representation of the real folding process [Dill et al., 1995, DOI: 10.1110/ps.4.3.561](https://doi.org/10.1110/ps.4.3.561). The HP (Hydrophobic-Polar) model captures essential hydrophobic interactions [Lau & Dill, 1989, DOI: 10.1073/pnas.86.6.2050](https://doi.org/10.1073/pnas.86.6.2050). Quantum-inspired optimization has shown promise for combinatorial problems [Kadowaki & Nishimori, 1998, DOI: 10.1103/PhysRevE.58.5355](https://doi.org/10.1103/PhysRevE.58.5355).

## Features

- **Multiple lattice types**: 2D square, 3D cubic
- **Energy models**: HP model, contact energy, custom extensible models
- **Optimization algorithms**: Simulated annealing, quantum-inspired annealing (with tunneling), greedy baseline
- **GPU acceleration**: Batch evaluation of candidate conformations using PyTorch (auto-detects GPU/CPU)
- **Visualization**: Interactive 3D plots of folding trajectories and energy landscapes
- **Benchmarking**: Automated measurement of runtime, convergence, and solution quality
- **Synthetic datasets**: Generates realistic HP sequences programmatically (no external downloads required for basic runs)
- **External dataset support**: Optional fetching of curated sequences from ProteinNet/CASP subsets

## Quick Start (Google Colab)

1. Open `notebooks/colab_run.ipynb` in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChessEngineUS/protein-lattice-qio/blob/main/notebooks/colab_run.ipynb)
2. Run the first cell to install dependencies:
   ```python
   !pip install -r requirements.txt
   ```
3. Execute cells sequentially. The notebook auto-detects GPU and adjusts batch sizes accordingly.
4. Outputs (plots, benchmark results) are saved to `outputs/` and displayed inline.

**Expected runtime (Colab free tier T4 GPU):** ~3-5 minutes for demo sequences (to be measured by running scripts/benchmark.py).

## Local Installation & Usage

### Prerequisites
- Python 3.10 or higher
- (Optional) CUDA-compatible GPU for acceleration

### Installation
```bash
git clone https://github.com/ChessEngineUS/protein-lattice-qio.git
cd protein-lattice-qio
pip install -r requirements.txt
```

### Running the Demo
```bash
python run_demo.py
```
This will:
1. Generate synthetic HP sequences
2. Run optimization algorithms (greedy, simulated annealing, quantum-inspired)
3. Produce visualizations in `outputs/`
4. Save energy trajectories and final conformations

### Running Benchmarks
To measure actual performance on your hardware:
```bash
python scripts/benchmark.py
```
Results are saved to:
- `outputs/benchmark_results.json` (timing, memory, solution quality)
- `outputs/benchmark_plot.png` (comparative visualization)

### Running Tests
```bash
pytest tests/ -v
```

### CPU vs GPU Behavior
- **GPU detected**: Uses batch sizes of 128-512 for parallel conformation evaluation
- **CPU only**: Automatically reduces batch size to 16-32 to avoid memory issues
- **Runtime difference**: GPU typically 5-10× faster on sequences of length 20-50

Actual runtime ranges must be measured by running `scripts/benchmark.py` on your specific hardware.

## Repository Structure

```
protein-lattice-qio/
├── README.md
├── LICENSE
├── CITATION.md
├── requirements.txt
├── run_demo.py                 # Main entry point
├── notebooks/
│   └── colab_run.ipynb        # Interactive Colab notebook
├── src/
│   ├── __init__.py
│   ├── lattice.py             # Lattice model definitions
│   ├── energy.py              # Energy function implementations
│   ├── optimizer.py           # Optimization algorithms
│   ├── visualization.py       # Plotting utilities
│   └── utils.py               # Helper functions
├── scripts/
│   ├── fetch_and_verify.py    # Download & verify external datasets
│   └── benchmark.py           # Automated benchmarking
├── tests/
│   ├── __init__.py
│   ├── test_lattice.py
│   ├── test_energy.py
│   └── test_optimizer.py
├── outputs/                    # Generated outputs (created at runtime)
│   ├── sources.json           # Metadata for external resources
│   ├── benchmark_results.json
│   └── benchmark_plot.png
└── .github/
    └── workflows/
        └── ci.yml             # Continuous integration
```

## Extending the Code

### Adding a New Energy Function
Edit `src/energy.py` and subclass `EnergyFunction`:
```python
class MyEnergyFunction(EnergyFunction):
    def calculate(self, coords, sequence):
        # Your energy calculation
        return energy_value
```

### Adding a New Optimizer
Edit `src/optimizer.py` and subclass `Optimizer`:
```python
class MyOptimizer(Optimizer):
    def optimize(self, sequence, lattice, energy_fn):
        # Your optimization logic
        return best_coords, best_energy, trajectory
```

## Citations & References

If you use this code in your research, please cite:

```bibtex
@software{protein_lattice_qio_2026,
  author = {ChessEngineUS},
  title = {Quantum-Inspired Optimization for Protein Folding Lattice Models},
  year = {2026},
  url = {https://github.com/ChessEngineUS/protein-lattice-qio}
}
```

**Key scientific references:**
1. Dill, K. A. (1995). Theory for the folding and stability of globular proteins. *Protein Science*, 4(3), 561-602. [DOI: 10.1110/ps.4.3.561](https://doi.org/10.1110/ps.4.3.561)
2. Lau, K. F., & Dill, K. A. (1989). A lattice statistical mechanics model of the conformational and sequence spaces of proteins. *PNAS*, 86(6), 2050-2054. [DOI: 10.1073/pnas.86.6.2050](https://doi.org/10.1073/pnas.86.6.2050)
3. Kadowaki, T., & Nishimori, H. (1998). Quantum annealing in the transverse Ising model. *Physical Review E*, 58(5), 5355. [DOI: 10.1103/PhysRevE.58.5355](https://doi.org/10.1103/PhysRevE.58.5355)

## License

MIT License. See `LICENSE` for details.

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

## Contact

For questions or collaboration, open an issue at [https://github.com/ChessEngineUS/protein-lattice-qio/issues](https://github.com/ChessEngineUS/protein-lattice-qio/issues).