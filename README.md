# Geometric Lattice Model for the Cusp-Core Problem

## Project Description
This repository contains the Python simulation code used for the research note, "A Geometric Lattice Model for the Cusp-Core Problem," published in the AAS Research Notes.

A phenomenological lattice model is implemented to explore the formation of galactic cores and cusps. Particles interact with a dynamic curvature field, and a density-dependent repulsive force simulates stellar feedback. The repository includes 2D lattice simulations, 3D lattice simulations, and static lattice control runs (no curvature feedback). Supporting modules and notebooks allow interactive exploration.

**Core finding:** The repulsion strength (`Krep`) determines whether a stable, low-density core (analogous to a dwarf galaxy) or a dynamic, high-density cusp (analogous to a massive galaxy) forms. Static lattice runs confirm that two-way curvature–density interaction is essential for core formation.

---

## Getting Started

### Prerequisites
- Python 3.x  
- numpy  
- matplotlib  

Install required libraries:

```bash
pip install numpy matplotlib
```
# Running the Simulations:

# 2D Simulation
```bash
python scripts/2D_simulation.py
```
# 3D Simulation
```bash
python scripts/3D_simulation.py
```
# Static Lattice Simulation
```bash
python scripts/static_lattice_sim.py
```
---

# Geometric Lattice Model – README

## Outputs
All results are saved in the `results/` folder, including plots and CSV data.

## Key Results
- 2D and 3D sweeps show the lattice phase transition: low `Krep` → cores; high `Krep` → cusps
- Radial profiles confirm low-density cores in dwarf analogs and high-density cusps in massive analogs
- Static lattice runs highlight the necessity of curvature–density feedback for core formation

## Code and Notebooks Details

### Code Organization
All simulation scripts are in the `scripts/` folder:
- `2D_simulation.py` – 2D lattice simulations and parameter sweeps
- `3D_simulation.py` – 3D lattice simulations and parameter sweeps
- `static_lattice_sim.py` – Static lattice control runs

Supporting routines and notebooks:
- `3d.py`, `matrix_simulation.py`, `matrix_simulation2_0.py` – Core lattice computation routines
- `notebooks/` – Optional Jupyter notebooks for interactive visualization

### Results Folder
- `results/2D_krep_sweep.png` and `2D_krep_sweep_data_.csv` – 2D sweep results
- `results/3D_krep_sweep.png` and `3d_krep_sweep_data.csv` – 3D sweep results
- `results/radial_profiles.png` – Normalized radial density profiles

### Notebooks
Notebooks allow exploration of particle dynamics, curvature fields, and radial density profiles.

## License
MIT License. See the [LICENSE](LICENSE) file.

## Contact
For questions or feedback, please open an issue or contact [Dr Mike](mailto:mjay10016@gmail.com)

