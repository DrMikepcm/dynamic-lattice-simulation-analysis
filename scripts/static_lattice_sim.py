"""
# 2D Static Lattice Simulation for Core–Cusp Formation

This script runs a 2D particle-on-a-grid simulation to explore
the formation of galaxy cores and cusps. It models particles 
interacting with a static curvature field and a density-dependent 
repulsive force.

It includes:
- Dwarf and massive galaxy analogs
- Tracking of max central curvature over time
- Plotting of maxC vs simulation steps

Author: [Your Name]
Repository: https://github.com/yourusername/galaxy-lattice
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# --- Simulation parameters ---
N = 128
NP_dwarf = 2000
NP_massive = 20000
DT = 0.05
STEPS = 1000
Krep_dwarf = 1.0
v_th_dwarf = 0.1
Krep_massive = 0.2
v_th_massive = 0.05
PRINT_EVERY = 50
s_src = 2.5
lam = 0.005
D_C = 0.08
mu = 2.0
r_rep = 3

OUT_DIR = "static_2d_output"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Helper functions ---
def laplacian(F):
    return (np.roll(F,1,0)+np.roll(F,-1,0)+np.roll(F,1,1)+np.roll(F,-1,1)-4*F)

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
        mask = (dist2 > 0) & (dist2 <= r2)
        ax[i] += Krep * np.sum(dx[mask]/np.sqrt(dist2[mask]) * rho_local[mask])
        ay[i] += Krep * np.sum(dy[mask]/np.sqrt(dist2[mask]) * rho_local[mask])
    return ax, ay

# --- Simulation runner ---
def run_sim(NP_val, Krep, v_th, seed=12345):
    rng = np.random.default_rng(seed)
    xp = rng.normal(N/2, 10, NP_val) % N
    yp = rng.normal(N/2, 10, NP_val) % N
    vx, vy = np.zeros(NP_val), np.zeros(NP_val)
    C = np.zeros((N,N))
    maxC_history = []

    for step in range(1, STEPS+1):
        rho = cic_deposit(xp, yp)
        C += DT*(s_src*rho - lam*C + D_C*laplacian(C))
        Cx, Cy = grad_xy(C)
        ax_curv = mu * bilinear_sample(Cx, xp, yp)
        ay_curv = mu * bilinear_sample(Cy, xp, yp)
        ax_rep, ay_rep = repulsion_accel(xp, yp, Krep)
        ax_therm = rng.normal(0, v_th, NP_val)
        ay_therm = rng.normal(0, v_th, NP_val)
        vx += (ax_curv + ax_rep + ax_therm) * DT
        vy += (ay_curv + ay_rep + ay_therm) * DT
        xp = (xp + vx*DT) % N
        yp = (yp + vy*DT) % N
        maxC_history.append(C[N//2,N//2])
        if step % PRINT_EVERY == 0:
            print(f"Step {step}/{STEPS}: max central curvature = {maxC_history[-1]:.4f}")

    return xp, yp, np.array(maxC_history)

# --- Main execution ---
if __name__=="__main__":
    xp_d, yp_d, maxC_d = run_sim(NP_dwarf, Krep_dwarf, v_th_dwarf)
    xp_m, yp_m, maxC_m = run_sim(NP_massive, Krep_massive, v_th_massive)

    # Save data
    np.save(os.path.join(OUT_DIR,"positions_dwarf.npy"), np.vstack([xp_d, yp_d]))
    np.save(os.path.join(OUT_DIR,"maxC_dwarf.npy"), maxC_d)
    np.save(os.path.join(OUT_DIR,"positions_massive.npy"), np.vstack([xp_m, yp_m]))
    np.save(os.path.join(OUT_DIR,"maxC_massive.npy"), maxC_m)

    # Plot max central curvature
    plt.figure(figsize=(10,6))
    plt.plot(np.arange(1, STEPS+1), maxC_d, label="Dwarf Analog", color='blue')
    plt.plot(np.arange(1, STEPS+1), maxC_m, label="Massive Analog", color='red')
    plt.xlabel("Time step")
    plt.ylabel("Max Central Curvature")
    plt.title("Max Central Curvature over Time (2D Static Lattice)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR,"maxC_vs_step.png"))
    plt.show()
    print(f"Plot saved to '{OUT_DIR}/maxC_vs_step.png'")
