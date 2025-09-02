"""
Galaxy Core–Cusp Simulation and Radial Profile Analysis
-------------------------------------------------------

This script simulates particle evolution in a 3D lattice under the influence of:
  - A curvature-driven large-scale attractive field (diffusion coefficient D_C).
  - A localized density-dependent repulsive force (strength K_rep).
  - Thermal noise and stochastic motion.

The simulation produces a particle distribution representing a dwarf-galaxy analog.
From this, the script computes:
  - Radial density profile (ρ vs. r).
  - Inner and outer log–log slopes of the density profile, fitted in specified radial ranges.

Outputs:
  1. Radial density profile plot.
  2. Radial density profile with fitted inner/outer slopes overplotted.
  3. Printed table of (radius, density) and slope values.

Usage:
  Run directly with Python. Adjust simulation parameters (N, STEPS, D_C_choice,
  Krep_choice, etc.) to explore core–cusp transitions.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Simulation parameters ---
N = 48              # lattice size (3D grid)
STEPS = 200         # time steps
DT = 0.05           # time step size

NP_dwarf = 2500     # particles for dwarf galaxy
v_th = 0.12         # thermal noise
D_C_choice = 0.05   # diffusion coefficient
Krep_choice = 0.08  # repulsion (use 0.01 for cusp, 0.08 for core example)
s_src = 2.5         # source for curvature evolution
lam = 0.01          # curvature decay
mu = 1.0            # curvature force scaling
r_rep = 2           # repulsion interaction radius
alpha = 0.5         # repulsion ramp exponent
rng_seed = 12345    # random seed

# --- Helper functions ---
def laplacian(F):
    return (np.roll(F,1,0)+np.roll(F,-1,0)+
            np.roll(F,1,1)+np.roll(F,-1,1)+
            np.roll(F,1,2)+np.roll(F,-1,2) - 6*F)

def cic_deposit(xp, yp, zp):
    rho = np.zeros((N,N,N))
    i0 = np.floor(xp).astype(int) % N
    j0 = np.floor(yp).astype(int) % N
    k0 = np.floor(zp).astype(int) % N
    i1, j1, k1 = (i0+1)%N, (j0+1)%N, (k0+1)%N
    tx, ty, tz = xp - np.floor(xp), yp - np.floor(yp), zp - np.floor(zp)
    w000 = (1-tx)*(1-ty)*(1-tz)
    w100 = tx*(1-ty)*(1-tz)
    w010 = (1-tx)*ty*(1-tz)
    w001 = (1-tx)*(1-ty)*tz
    w101 = tx*(1-ty)*tz
    w011 = (1-tx)*ty*tz
    w110 = tx*ty*(1-tz)
    w111 = tx*ty*tz
    np.add.at(rho, (i0,j0,k0), w000)
    np.add.at(rho, (i1,j0,k0), w100)
    np.add.at(rho, (i0,j1,k0), w010)
    np.add.at(rho, (i0,j0,k1), w001)
    np.add.at(rho, (i1,j0,k1), w101)
    np.add.at(rho, (i0,j1,k1), w011)
    np.add.at(rho, (i1,j1,k0), w110)
    np.add.at(rho, (i1,j1,k1), w111)
    return rho

def grad_xyz(F):
    Fx = (np.roll(F,-1,0)-np.roll(F,1,0))/2
    Fy = (np.roll(F,-1,1)-np.roll(F,1,1))/2
    Fz = (np.roll(F,-1,2)-np.roll(F,1,2))/2
    return Fx, Fy, Fz

def trilinear_sample(F, xp, yp, zp):
    i0 = np.floor(xp).astype(int) % N
    j0 = np.floor(yp).astype(int) % N
    k0 = np.floor(zp).astype(int) % N
    i1, j1, k1 = (i0+1)%N, (j0+1)%N, (k0+1)%N
    tx, ty, tz = xp - np.floor(xp), yp - np.floor(yp), zp - np.floor(zp)
    f000, f100 = F[i0,j0,k0], F[i1,j0,k0]
    f010, f001 = F[i0,j1,k0], F[i0,j0,k1]
    f101, f011 = F[i1,j0,k1], F[i0,j1,k1]
    f110, f111 = F[i1,j1,k0], F[i1,j1,k1]
    c00 = f000*(1-tx)+f100*tx
    c01 = f001*(1-tx)+f101*tx
    c10 = f010*(1-tx)+f110*tx
    c11 = f011*(1-tx)+f111*tx
    c0 = c00*(1-ty)+c10*ty
    c1 = c01*(1-ty)+c11*ty
    return c0*(1-tz)+c1*tz

def minimum_image(dx):
    return (dx + N/2) % N - N/2

def radial_profile_from_particles(xp, yp, zp, nbins=24, rmax=None):
    cx = cy = cz = N/2.0
    dx = minimum_image(xp - cx)
    dy = minimum_image(yp - cy)
    dz = minimum_image(zp - cz)
    r = np.sqrt(dx*dx + dy*dy + dz*dz)
    if rmax is None:
        rmax = N/2.0
    edges = np.linspace(0.5, rmax, nbins+1)
    counts, _ = np.histogram(r, bins=edges)
    r3 = edges**3
    vol = (4.0/3.0)*np.pi*(r3[1:] - r3[:-1])
    rho = counts / np.maximum(vol, 1e-9)
    r_mid = 0.5*(edges[1:] + edges[:-1])
    rho = np.where(rho <= 0, np.nan, rho)
    return r_mid, rho

# --- Initialize particles and curvature field ---
rng = np.random.default_rng(rng_seed)
xp = rng.normal(N/2, 5, NP_dwarf) + rng.normal(0, 2, NP_dwarf)
yp = rng.normal(N/2, 5, NP_dwarf) + rng.normal(0, 2, NP_dwarf)
zp = rng.normal(N/2, 5, NP_dwarf) + rng.normal(0, 2, NP_dwarf)
vx = np.zeros(NP_dwarf); vy = np.zeros(NP_dwarf); vz = np.zeros(NP_dwarf)
C = np.zeros((N,N,N))

# --- Main simulation loop ---
for step in range(STEPS):
    Krep_step = Krep_choice * ((step+1)/STEPS)**alpha
    D_C_step  = D_C_choice * (step/100) if step < 100 else D_C_choice
    v_th_dynamic = v_th * (1 + 5*np.exp(-step/50))
    rho = cic_deposit(xp, yp, zp)
    C += DT*(s_src*rho - lam*C + D_C_step*laplacian(C))
    Cx, Cy, Cz = grad_xyz(C)
    ax = mu * trilinear_sample(Cx, xp, yp, zp)
    ay = mu * trilinear_sample(Cy, xp, yp, zp)
    az = mu * trilinear_sample(Cz, xp, yp, zp)
    dx = minimum_image(xp[:,None] - xp[None,:])
    dy = minimum_image(yp[:,None] - yp[None,:])
    dz = minimum_image(zp[:,None] - zp[None,:])
    dist2 = dx*dx + dy*dy + dz*dz
    mask = (dist2 > 0) & (dist2 <= r_rep**2)
    invr = np.zeros_like(dist2)
    invr[mask] = 1.0 / np.sqrt(dist2[mask])
    ax += Krep_step * np.sum(dx * mask * invr, axis=1)
    ay += Krep_step * np.sum(dy * mask * invr, axis=1)
    az += Krep_step * np.sum(dz * mask * invr, axis=1)
    ax += rng.normal(0, v_th_dynamic, NP_dwarf)
    ay += rng.normal(0, v_th_dynamic, NP_dwarf)
    az += rng.normal(0, v_th_dynamic, NP_dwarf)
    vx += DT * ax; vy += DT * ay; vz += DT * az
    xp = (xp + DT * vx) % N
    yp = (yp + DT * vy) % N
    zp = (zp + DT * vz) % N

# --- Radial density profile ---
r_mid, rho_shell = radial_profile_from_particles(xp, yp, zp, nbins=24)

plt.figure(figsize=(6,5))
plt.loglog(r_mid, rho_shell, marker='o', linestyle='-', color='tab:blue')
plt.xlabel('Radius (r)')
plt.ylabel('Density (ρ)')
plt.title('Radial Density Profile (Single Dwarf Run)')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.show()

# --- Slope fitting ---
inner_mask = (r_mid > 1) & (r_mid < 0.25 * N / 2)
outer_mask = (r_mid > 0.35 * N / 2) & (r_mid < 0.60 * N / 2)

def fit_loglog_slope(r, rho, mask):
    valid = mask & np.isfinite(rho) & (rho > 0)
    if np.sum(valid) < 3:
        return np.nan, np.array([np.nan, np.nan])
    x = np.log10(r[valid]); y = np.log10(rho[valid])
    coeffs = np.polyfit(x, y, 1)
    return coeffs[0], coeffs

alpha_in, coeffs_in = fit_loglog_slope(r_mid, rho_shell, inner_mask)
alpha_out, coeffs_out = fit_loglog_slope(r_mid, rho_shell, outer_mask)

plt.figure(figsize=(6,5))
plt.loglog(r_mid, rho_shell, 'o-', label='Radial Density Profile')

# Plot fitted inner slope
if not np.isnan(alpha_in):
    x_fit_inner = r_mid[inner_mask & np.isfinite(rho_shell) & (rho_shell > 0)]
    y_fit_inner = 10 ** np.polyval(coeffs_in, np.log10(x_fit_inner))
    plt.plot(x_fit_inner, y_fit_inner, 'r--', label=f'Inner slope α={-alpha_in:.2f}')

# Plot fitted outer slope
if not np.isnan(alpha_out):
    x_fit_outer = r_mid[outer_mask & np.isfinite(rho_shell) & (rho_shell > 0)]
    y_fit_outer = 10 ** np.polyval(coeffs_out, np.log10(x_fit_outer))
    plt.plot(x_fit_outer, y_fit_outer, 'g--', label=f'Outer slope α={-alpha_out:.2f}')

plt.xlabel('Radius (r)')
plt.ylabel('Density (ρ)')
plt.title('3D Radial Density Profile with Inner/Outer Slopes')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.show()

# --- Print raw data with slopes ---
print("Radius\tDensity")
for r, dens in zip(r_mid, rho_shell):
    print(f"{r:.4f}\t{dens:.5e}")

print(f"\nFitted inner slope: {-alpha_in:.3f}")
print(f"Fitted outer slope: {-alpha_out:.3f}")
