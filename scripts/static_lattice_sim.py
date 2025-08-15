# =============================================================================
# Static 2D Lattice Simulation for Core–Cusp Formation (Cmax vs Repulsion Sweep)
#
# This script implements a static 2D particle-on-a-grid lattice model to study
# galaxy core and cusp formation. Particles interact with a fixed curvature
# field and a density-dependent repulsive force. The simulation sweeps the
# repulsion strength parameter (Krep) to evaluate its effect on the maximum
# central curvature (Cmax), illustrating the absence of a sweet spot in the
# static lattice.
#
# Features:
# - Core parameters for 2D lattice
# - Cloud-in-Cell deposition of particles
# - Bilinear interpolation of curvature gradients
# - Local density-dependent repulsive acceleration
# - Sweep of Krep and plotting of Cmax vs Krep
#
# Outputs:
# - Plot of maximum central curvature as a function of repulsion strength
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# --- Simulation parameters ---
N = 128          # Grid size
NP = 2000        # Number of particles
DT = 0.05        # Time step
STEPS = 1000     # Simulation steps
v_th = 0.1       # Thermal velocity
PRINT_EVERY = 200
s_src = 2.5
lam = 0.005
D_C = 0.08
mu = 2.0
r_rep = 3

# --- Helper functions ---
def laplacian(F):
    return np.roll(F,1,0)+np.roll(F,-1,0)+np.roll(F,1,1)+np.roll(F,-1,1)-4*F

def cic_deposit(xp, yp):
    rho = np.zeros((N,N))
    i0 = np.floor(xp).astype(int) % N
    j0 = np.floor(yp).astype(int) % N
    i1, j1 = (i0+1)%N, (j0+1)%N
    tx, ty = xp - np.floor(xp), yp - np.floor(yp)
    w00=(1-tx)*(1-ty); w10=tx*(1-ty); w01=(1-tx)*ty; w11=tx*ty
    np.add.at(rho, (i0,j0), w00)
    np.add.at(rho, (i1,j0), w10)
    np.add.at(rho, (i0,j1), w01)
    np.add.at(rho, (i1,j1), w11)
    return rho

def grad_xy(F):
    Fx = (np.roll(F,-1,1)-np.roll(F,1,1))/2.0
    Fy = (np.roll(F,-1,0)-np.roll(F,1,0))/2.0
    return Fx, Fy

def bilinear_sample(F, xp, yp):
    i0 = np.floor(xp).astype(int) % N
    j0 = np.floor(yp).astype(int) % N
    i1, j1 = (i0+1)%N, (j0+1)%N
    tx, ty = xp - np.floor(xp), yp - np.floor(yp)
    f00 = F[i0,j0]; f10 = F[i1,j0]; f01 = F[i0,j1]; f11 = F[i1,j1]
    return (1-tx)*(1-ty)*f00 + tx*(1-ty)*f10 + (1-tx)*ty*f01 + tx*ty*f11

def local_density(xp, yp, r=r_rep):
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
    rho_local = local_density(xp, yp, r)
    ax = np.zeros_like(xp)
    ay = np.zeros_like(yp)
    r2 = r**2
    for i in range(len(xp)):
        dx = (xp[i] - xp + N/2) % N - N/2
        dy = (yp[i] - yp + N/2) % N - N/2
        dist2 = dx**2 + dy**2
        mask = (dist2>0) & (dist2 <= r2)
        ax[i] += Krep*np.sum(dx[mask]/np.sqrt(dist2[mask])*rho_local[mask])
        ay[i] += Krep*np.sum(dy[mask]/np.sqrt(dist2[mask])*rho_local[mask])
    return ax, ay

# --- Sweep simulation ---
def run_sweep(Krep):
    rng = np.random.default_rng(12345)
    cx, cy = N/2, N/2
    sigma = 10
    xp = rng.normal(cx, sigma, NP) % N
    yp = rng.normal(cy, sigma, NP) % N
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
        ax_therm = rng.normal(0, v_th, NP)
        ay_therm = rng.normal(0, v_th, NP)
        vx += (ax_curv + ax_rep + ax_therm)*DT
        vy += (ay_curv + ay_rep + ay_therm)*DT
        xp = (xp + vx*DT) % N
        yp = (yp + vy*DT) % N
        max_C_history.append(C[N//2,N//2])
    return np.max(max_C_history)

# --- Main execution ---
if __name__=="__main__":
    Krep_values = np.linspace(0.01, 2.0, 20)
    maxC_values = []

    for Krep in Krep_values:
        print(f"Running static simulation with Krep={Krep:.3f}")
        maxC_values.append(run_sweep(Krep))

    # Plot Cmax vs Krep
    plt.figure(figsize=(10,6))
    plt.plot(Krep_values, maxC_values, marker='o', linestyle='-', color='purple')
    plt.xlabel("Krep (Repulsion Strength)")
    plt.ylabel("Max Central Curvature (Cmax)")
    plt.title("Static Lattice: Max Central Curvature vs Repulsion")
    plt.grid(True)
    plt.savefig("static_cmax_vs_krep.png")
    plt.show()
    print("Cmax vs Krep plot saved as 'static_cmax_vs_krep.png'")
