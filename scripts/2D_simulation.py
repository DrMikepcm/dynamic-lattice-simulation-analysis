import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# A Geometric Lattice Model for the Cusp-Core Problem
#
# This script runs a 2D particle-on-a-grid simulation to explore the formation
# of galaxy cores and cusps. It models particles interacting with a dynamic
# curvature field and a density-dependent repulsive force. The script
# includes a parameter sweep for the repulsion strength (Krep) to demonstrate
# a phase transition from a stable core to a dynamic cusp.
#
# The parameters used here correspond to the dwarf and massive galaxy analogs
# discussed in the accompanying research note.
# =============================================================================

# --- Simulation parameters ---
# Core parameters for the dwarf galaxy analog
N = 128            # Grid size
NP = 2000          # Number of particles
DT = 0.05          # Time step
STEPS = 1000       # Total simulation steps
Krep_dwarf = 1.0   # Repulsion strength for dwarf analog
v_th_dwarf = 0.1   # Thermal velocity for dwarf analog

# Core parameters for the massive galaxy analog
NP_massive = 20000
Krep_massive = 0.2
v_th_massive = 0.05

# Common parameters
PRINT_EVERY = 50
s_src = 2.5
lam = 0.005
D_C = 0.08
mu = 2.0
r_rep = 3

# --- Helper functions ---
def laplacian(F):
    """Calculates the discrete Laplacian of a 2D field."""
    return (np.roll(F,1,0)+np.roll(F,-1,0)+np.roll(F,1,1)+np.roll(F,-1,1)-4*F)

def cic_deposit(xp, yp):
    """Deposits particle positions onto a grid using Cloud-in-Cell interpolation."""
    rho = np.zeros((N,N))
    i0 = np.floor(xp).astype(int) % N
    j0 = np.floor(yp).astype(int) % N
    i1 = (i0+1)%N; j1=(j0+1)%N
    tx = xp - np.floor(xp); ty = yp - np.floor(yp)
    w00=(1-tx)*(1-ty); w10=tx*(1-ty); w01=(1-tx)*ty; w11=tx*ty
    np.add.at(rho, (i0,j0), w00)
    np.add.at(rho, (i1,j0), w10)
    np.add.at(rho, (i0,j1), w01)
    np.add.at(rho, (i1,j1), w11)
    return rho

def grad_xy(F):
    """Calculates the central finite difference gradient of a 2D field."""
    Fx = (np.roll(F,-1,1)-np.roll(F,1,1))/2.0
    Fy = (np.roll(F,-1,0)-np.roll(F,1,0))/2.0
    return Fx, Fy

def bilinear_sample(F, xp, yp):
    """Samples a 2D field at particle positions using bilinear interpolation."""
    i0 = np.floor(xp).astype(int) % N
    j0 = np.floor(yp).astype(int) % N
    i1 = (i0+1)%N; j1=(j0+1)%N
    tx = xp - np.floor(xp)
    ty = yp - np.floor(yp)
    f00 = F[i0,j0]; f10 = F[i1,j0]; f01 = F[i0,j1]; f11 = F[i1,j1]
    return (1-tx)*(1-ty)*f00 + tx*(1-ty)*f10 + (1-tx)*ty*f01 + tx*ty*f11

def local_density(xp, yp, r=r_rep):
    """Calculates the local density around each particle using a circle of radius r."""
    Np = len(xp)
    rho_local = np.zeros(Np)
    r2 = r**2
    for i in range(Np):
        dx = (xp - xp[i] + N/2) % N - N/2
        dy = (yp - yp[i] + N/2) % N - N/2
        dist2 = dx**2 + dy**2
        rho_local[i] = np.sum(dist2 <= r2) / (np.pi*r2)
    return rho_local

def repulsion_accel(xp, yp, Krep, r=r_rep):
    """Calculates the repulsive acceleration based on local density and Krep."""
    rho_local = local_density(xp, yp, r)
    ax = np.zeros_like(xp)
    ay = np.zeros_like(yp)
    r2 = r**2
    for i in range(len(xp)):
        dx = (xp[i] - xp + N/2) % N - N/2
        dy = (yp[i] - yp + N/2) % N - N/2
        dist2 = dx**2 + dy**2
        mask = (dist2 > 0) & (dist2 <= r2)
        ax[i] += Krep * np.sum(dx[mask]/np.sqrt(dist2[mask]) * rho_local[mask])
        ay[i] += Krep * np.sum(dy[mask]/np.sqrt(dist2[mask]) * rho_local[mask])
    return ax, ay

def get_radial_profile(xp, yp, bins=20):
    """Calculates the radial density profile from particle positions."""
    cx, cy = N/2, N/2
    dx = (xp - cx + N/2) % N - N/2
    dy = (yp - cy + N/2) % N - N/2
    dist = np.sqrt(dx**2 + dy**2)
    radii = np.linspace(0, N/2, bins+1)
    hist, _ = np.histogram(dist, bins=radii)
    area = np.pi * (radii[1:]**2 - radii[:-1]**2)
    return radii[:-1], hist / area

# --- Simulation runners ---
def run_main_sim(Krep, NP_val, v_th_val):
    """Runs a single simulation and returns the final particle positions."""
    rng = np.random.default_rng(12345)
    cx, cy = N/2, N/2
    sigma = 10
    xp = rng.normal(cx, sigma, size=NP_val) % N
    yp = rng.normal(cy, sigma, size=NP_val) % N
    vx = np.zeros(NP_val)
    vy = np.zeros(NP_val)
    C = np.zeros((N,N))

    for step in range(1, STEPS+1):
        rho = cic_deposit(xp, yp)
        C += DT*(s_src*rho - lam*C + D_C*laplacian(C))
        Cx, Cy = grad_xy(C)
        ax_curv = mu * bilinear_sample(Cx, xp, yp)
        ay_curv = mu * bilinear_sample(Cy, xp, yp)
        ax_rep, ay_rep = repulsion_accel(xp, yp, Krep)
        ax_therm = rng.normal(0, v_th_val, size=NP_val)
        ay_therm = rng.normal(0, v_th_val, size=NP_val)
        vx += (ax_curv + ax_rep + ax_therm)*DT
        vy += (ay_curv + ay_rep + ay_therm)*DT
        xp = (xp + vx*DT) % N
        yp = (yp + vy*DT) % N
    return xp, yp

def run_sweep_sim(Krep):
    """Runs a single simulation for the sweep and returns the max central curvature."""
    rng = np.random.default_rng(12345)
    cx, cy = N/2, N/2
    sigma = 10
    xp = rng.normal(cx, sigma, size=NP) % N
    yp = rng.normal(cy, sigma, size=NP) % N
    vx = np.zeros(NP)
    vy = np.zeros(NP)
    C = np.zeros((N,N))
    max_C_history = []

    for step in range(1, STEPS+1):
        rho = cic_deposit(xp, yp)
        C += DT*(s_src*rho - lam*C + D_C*laplacian(C))
        Cx, Cy = grad_xy(C)
        ax_curv = mu * bilinear_sample(Cx, xp, yp)
        ay_curv = mu * bilinear_sample(Cy, xp, yp)
        ax_rep, ay_rep = repulsion_accel(xp, yp, Krep)
        ax_therm = rng.normal(0, v_th_dwarf, size=NP)
        ay_therm = rng.normal(0, v_th_dwarf, size=NP)
        vx += (ax_curv + ax_rep + ax_therm)*DT
        vy += (ay_curv + ay_rep + ay_therm)*DT
        xp = (xp + vx*DT) % N
        yp = (yp + vy*DT) % N
        # Record max central curvature
        center_pixel = (int(cx), int(cy))
        max_C_history.append(C[center_pixel])
    return np.max(max_C_history)

# --- Main execution block ---
if __name__ == "__main__":
    # --- Part 1: Run and plot the main simulations ---
    print("Running Dwarf Galaxy Analog...")
    xp_dwarf, yp_dwarf = run_main_sim(Krep_dwarf, NP, v_th_dwarf)
    print("Running Massive Galaxy Analog...")
    xp_massive, yp_massive = run_main_sim(Krep_massive, NP_massive, v_th_massive)
    
    r_dwarf, profile_dwarf = get_radial_profile(xp_dwarf, yp_dwarf)
    r_massive, profile_massive = get_radial_profile(xp_massive, yp_massive)

    plt.figure(figsize=(10, 6))
    plt.plot(r_dwarf, profile_dwarf, label='Dwarf Analog (Core)', color='blue')
    plt.plot(r_massive, profile_massive, label='Massive Analog (Cusp)', color='red')
    plt.xlabel('Radius')
    plt.ylabel('Normalized Density')
    plt.title('Radial Density Profiles')
    plt.legend()
    plt.grid(True)
    plt.savefig('radial_profiles.png')
    plt.show()
    print("Radial density profiles saved to 'radial_profiles.png'")

    # --- Part 2: Run the Krep parameter sweep ---
    print("\nStarting Krep parameter sweep...")
    Krep_values = np.linspace(0.01, 2.0, 20)
    maxC_values = []

    for krep in Krep_values:
        print(f"Running simulation with Krep={krep:.3f}...")
        max_c = run_sweep_sim(krep)
        maxC_values.append(max_c)
    
    maxC_values = np.array(maxC_values)
    
    # --- Part 3: Plot the sweep results ---
    plt.figure(figsize=(10,6))
    plt.plot(Krep_values, maxC_values, marker='o', linestyle='-', color='green', label='Max Central Curvature')
    stable_threshold = 0.2
    plt.axhline(y=stable_threshold, color='red', linestyle='--', label='Stability threshold')
    plt.xlabel("Krep (Repulsion Strength)")
    plt.ylabel("Max Central Curvature")
    plt.title("Core Stability as a Function of Repulsion Strength")
    plt.legend()
    plt.grid(True)
    plt.savefig('krep_sweep.png')
    plt.show()
    print("Krep sweep results saved to 'krep_sweep.png'")

    print("\nAll simulations and plots complete.")
