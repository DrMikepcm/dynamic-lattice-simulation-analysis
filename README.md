# Geometric Lattice Model for the Cusp-Core Problem

## Project Description
This repository contains the Python simulation code used for the research note, "A Geometric Lattice Model for the Cusp-Core Problem," published in the AAS Research Notes.

The script, `simulation.py`, implements a phenomenological model to explore the formation of galactic cores and cusps. It models the gravitational influence of a central mass using a dynamic curvature field and introduces a density-dependent repulsive force to simulate stellar feedback.

**Core finding:** The repulsion strength (`Krep`) determines whether a stable, low-density core (analogous to a dwarf galaxy) or a dynamic, high-density cusp (analogous to a massive galaxy) forms.

---

## Getting Started

### Prerequisites
- Python 3.x  
- numpy  
- matplotlib  

Install the required libraries using pip:

```bash
pip install numpy matplotlib
```

---

## Running the Simulation
Run the script:

```bash
python simulation.py
```

The script will run two main simulations and a parameter sweep, printing progress updates to the console. It will save two plots, `radial_profiles.png` and `krep_sweep.png`, to the same directory.

---

## Key Results
- **`radial_profiles.png`** – Normalized radial density profiles for the "Dwarf Galaxy Analog" and "Massive Galaxy Analog," showing a low-density core in the dwarf analog and a dense cusp in the massive analog.  
- **`krep_sweep.png`** – Maximum central curvature (`maxC`) as a function of repulsion strength (`Krep`). Illustrates a phase transition: high `Krep` leads to a stable core, low `Krep` produces a dynamic, unstable cusp.

---

## Code Structure
All code is in `simulation.py`:

- **Helper Functions** – Grid operations, particle deposition, interpolation.  
- **`run_main_sim`** – Runs the full simulation for radial density profiles.  
- **`run_sweep_sim`** – Runs the parameter sweep, returning maximum central curvature.  
- **Main Execution Block (`if __name__ == "__main__":`)** – Orchestrates simulations and generates plots.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact
For questions or feedback, please open an issue or contact [Dr Mike](mailto:mjay10016@gmail.com).
