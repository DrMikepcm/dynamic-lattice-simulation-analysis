### `dynamic-lattice-simulation-analysis`

This repository contains data and scripts for analyzing a dynamic lattice simulation. The simulation models particle density over time and distance.

***

### Files

* `simulation_results.csv`: The core dataset with columns for **Step**, **Distance**, and **Density**.
* `data_parser.py`: A Python script to parse raw simulation output into a CSV file.
* `graph_generator.py`: A Python script to create graphs from the CSV data.

***

### Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/dynamic-lattice-simulation-analysis.git](https://github.com/your-username/dynamic-lattice-simulation-analysis.git)
    cd dynamic-lattice-simulation-analysis
    ```
2.  **Run the simulation:**
    ```bash
    python simulation.py
    ```
3.  **Generate graphs (after the simulation creates data):**
    ```bash
    pip install pandas matplotlib
    python graph_generator.py
    ```
    
