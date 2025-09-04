# Geometric Lattice Model for the Cusp-Core Problem

## Project Description
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

#Dwarf Single Run Radial Density
```bash
python scripts/dwarf_single_run_with_slopes.py
```
---

# Geometric Lattice Model – README

## Outputs
All results are saved in the `results/` folder, including plots and CSV data.

## Key Results
- 2D and 3D sweeps show the lattice phase transition: low `Krep` → cores; high `Krep` → cusps
- Radial profiles confirm low-density cores in dwarf analogs and high-density cusps in massive analogs
- Static lattice runs highlight the necessity of curvature–density feedback for core formation
- Dwarf single-run simulations confirm that low-density cores emerge naturally from the lattice physics, without sweeping parameters.


### Results Folder Overview

| File / Folder | Description |
|---------------|-------------|
| `results/2D_krep_sweep.png` | 2D lattice parameter sweep plot |
| `results/2D_krep_sweep_data_.csv` | 2D sweep data |
| `results/3D_krep_sweep.png` | 3D lattice parameter sweep plot |
| `results/3d_krep_sweep_data.csv` | 3D sweep data |
| `results/3D_radial_density_profile.png` | Radial density profile from 3D sweep |
| `results/dwarf_radial_density_profile_data.csv` | Dwarf single-run radial density data |
| `scripts/` | Simulation scripts for 2D, 3D, dwarf single-run, and static lattice |
| `notebooks/` | Optional Jupyter notebooks for visualization and exploration |

---

### Code Organization

All simulation scripts are in the `scripts/` folder:

- `2D_simulation.py` – 2D lattice simulations and parameter sweeps  
- `3D_simulation.py` – 3D lattice simulations and parameter sweeps  
- `dwarf_single_run_with_slopes.py` – Single-run dwarf galaxy simulation  
- `static_lattice_sim.py` – Static lattice control runs  

Supporting routines:

- `3d.py`, `matrix_simulation.py`, `matrix_simulation2_0.py` – Core lattice computation routines  

Notebooks in the `notebooks/` folder allow interactive exploration of:

- Particle dynamics  
- Curvature field evolution  
- Radial density profiles


## License
MIT License. See the [LICENSE](LICENSE) file.

## Contact
For questions or feedback, please open an issue or contact [Dr Mike](mailto:mjay10016@gmail.com)

