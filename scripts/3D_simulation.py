#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# quick_3d_sweep_with_thresholds.py
# ------------------------------------------------------------
# 3D Cusp–Core Sweep for Dwarf & Massive Analogs
# ------------------------------------------------------------
# Description:
# - Runs 3D lattice particle simulations for dwarfs & massive galaxies
# - Sweeps over Krep (repulsion) and D_C (diffusion coefficient)
# - Computes MaxC (proxy for central curvature/density) vs Krep
# - Computes radial density profiles and inner/outer log-log slopes
# - Automatically detects core vs cusp transitions using slope differences
# - Saves MaxC tables, slope diagnostics, and thresholds
# - Generates publication-ready figure with all 6 curves + horizontal threshold lines
#
# Requirements:
#   numpy, matplotlib, os
#
# Output:
#   - quick_sweep_output/<gal>_DC_<D_C>.txt  (MaxC tables)
#   - quick_sweep_output/slopes_<gal>_DC_<D_C>.txt (slope diagnostics)
#   - quick_sweep_output/thresholds_by_slope.txt
#   - quick_sweep_output/quick_sweep_summary_with_thresholds.png
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------
# Output
# ----------------------------
OUT_DIR = "quick_sweep_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Simulation settings (fast but representative)
# ----------------------------
N = 32
STEPS = 200
DT = 0.05

# Dwarf & Massive particle counts
NP_dwarf = 500
NP_massive = 1000

# Thermal noise (base)
v_th_dwarf = 0.12
v_th_massive = 0.08

# Lattice params
s_src = 2.5
lam = 0.01
mu = 1.0
r_rep = 2

# Sweep parameters
D_C_values = [0.01, 0.02, 0.05]
Krep_values = np.linspace(0.01, 2.0, 15)

# Dynamics knobs
alpha = 0.5        # ramp exponent for Krep_step
vth_A = 5.0        # early noise amplitude multiplier
vth_tau = 50.0     # noise decay timescale
rng_seed = 12345

# ----------------------------
# Helper functions
# ----------------------------
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

def fit_loglog_slope(r, rho, rmin, rmax):
    mask = (r >= rmin) & (r <= rmax) & np.isfinite(rho) & (rho > 0)
    if mask.sum() < 3:
        return np.nan
    x = np.log(r[mask])
    y = np.log(rho[mask])
    return np.polyfit(x, y, 1)[0]

def classify_core_by_slopes(r, rho, delta_alpha_cut=0.3):
    R = N/2.0
    alpha_in  = fit_loglog_slope(r, rho, rmin=1.0,        rmax=0.25*R)
    alpha_out = fit_loglog_slope(r, rho, rmin=0.35*R,     rmax=0.60*R)
    if np.isnan(alpha_in) or np.isnan(alpha_out):
        return alpha_in, alpha_out, False
    delta = abs(alpha_out) - abs(alpha_in)
    return alpha_in, alpha_out, (delta >= delta_alpha_cut)

# ----------------------------
# Core simulation routine
# ----------------------------
def run_one_run(NP_val, Krep_base, v_th_base, D_C_choice, seed=rng_seed):
    rng = np.random.default_rng(seed)
    xp = rng.normal(N/2, 5, NP_val) + rng.normal(0, 2, NP_val)
    yp = rng.normal(N/2, 5, NP_val) + rng.normal(0, 2, NP_val)
    zp = rng.normal(N/2, 5, NP_val) + rng.normal(0, 2, NP_val)
    vx = np.zeros(NP_val); vy = np.zeros(NP_val); vz = np.zeros(NP_val)
    C = np.zeros((N,N,N))
    maxC_history = []

    for step in range(STEPS):
        Krep_step = Krep_base * ((step+1)/STEPS)**alpha
        D_C_step  = D_C_choice * (step/100) if step < 100 else D_C_choice
        v_th_dynamic = v_th_base * (1 + vth_A * np.exp(-step/vth_tau))

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

        ax += rng.normal(0, v_th_dynamic, NP_val)
        ay += rng.normal(0, v_th_dynamic, NP_val)
        az += rng.normal(0, v_th_dynamic, NP_val)

        vx += DT * ax; vy += DT * ay; vz += DT * az
        xp = (xp + DT * vx) % N
        yp = (yp + DT * vy) % N
        zp = (zp + DT * vz) % N

        maxC_history.append(C[N//2, N//2, N//2])

    r_mid, rho_shell = radial_profile_from_particles(xp, yp, zp, nbins=24)
    return np.max(maxC_history), r_mid, rho_shell

# ----------------------------
# Run sweeps, save data, compute thresholds
# ----------------------------
results = {}
thresholds = {}
slope_tables = {}

for D_C_choice in D_C_values:
    print(f"\n=== Running sweep for D_C = {D_C_choice:.4f} ===")
    for gal, NP_val, vth in [("dwarf", NP_dwarf, v_th_dwarf),
                             ("massive", NP_massive, v_th_massive)]:

        maxC_list = []
        all_slopes = []
        core_found = False
        Krep_threshold = np.nan

        for k in Krep_values:
            Cmax, r_mid, rho_shell = run_one_run(NP_val, k, vth, D_C_choice, seed=rng_seed)
            maxC_list.append(Cmax)
            a_in, a_out, is_core = classify_core_by_slopes(r_mid, rho_shell, delta_alpha_cut=0.3)
            all_slopes.append((k, a_in, a_out, is_core))
            print(f"{gal:7s} D_C={D_C_choice:.3f} Krep={k:.3f}  MaxC={Cmax:.4f}  alpha_in={a_in:.3f} alpha_out={a_out:.3f} core={is_core}")

            if (not core_found) and is_core:
                Krep_threshold = k
                core_found = True

        maxC_arr = np.array(maxC_list)
        fname = os.path.join(OUT_DIR, f"{gal}_DC_{D_C_choice:.3f}.txt")
        np.savetxt(fname, np.column_stack([Krep_values, maxC_arr]),
                   header="Krep\tMaxC", fmt="%.6f")
        print(f"Saved table: {fname}")

        results[(gal, D_C_choice)] = {"Krep": Krep_values, "MaxC": maxC_arr}
        thresholds[(gal, D_C_choice)] = Krep_threshold
        slope_tables[(gal, D_C_choice)] = np.array(all_slopes, dtype=float)

# ----------------------------
# Plot all curves + horizontal thresholds
# ----------------------------
plt.figure(figsize=(10,6))
color_map = {"dwarf": "tab:blue", "massive": "tab:red"}
linestyles = {"dwarf": "-", "massive": "--"}

for gal in ["dwarf", "massive"]:
    for D_C_choice in D_C_values:
        dat = results[(gal, D_C_choice)]
        label = f"{gal.capitalize()} $D_C={D_C_choice:.2f}$"
        plt.plot(dat["Krep"], dat["MaxC"],
                 linestyles[gal], marker='o', color=color_map[gal],
                 alpha=0.9, label=label)

        kthr = thresholds[(gal, D_C_choice)]
        if not np.isnan(kthr):
            plt.axhline(kthr, color=color_map[gal], linestyle=":", alpha=0.5,
                        label=f"{gal.capitalize()} threshold Δ|α|≥0.3")

plt.xlabel("Krep (Repulsion)")
plt.ylabel("MaxC (Max Central Curvature/Density)")
plt.title("Effect of Repulsion and Curvature Diffusion on Central Density in 3D Lattice Simulations")
plt.grid(True, alpha=0.3)
plt.legend(ncol=2)
plt.tight_layout()
figfile = os.path.join(OUT_DIR, "quick_sweep_summary_with_thresholds.png")
plt.savefig(figfile, dpi=300)
plt.show()
print(f"\nSaved summary figure: {figfile}")

# ----------------------------
# Save thresholds to disk
# ----------------------------
thr_file = os.path.join(OUT_DIR, "thresholds_by_slope.txt")
with open(thr_file, "w") as f:
    f.write("galaxy,DC,Krep_threshold(Δ|α|≥0.3)\n")
    for gal in ["dwarf", "massive"]:
        for D_C_choice in D_C_values:
            kthr = thresholds[(gal, D_C_choice)]
            f.write(f"{gal},{D_C_choice:.3f},{'' if np.isnan(kthr) else f'{kthr:.3f}'}\n")
print("Saved thresholds:", thr_file)

# ----------------------------
# Optional: Save slope diagnostics
# ----------------------------
for gal in ["dwarf", "massive"]:
    for D_C_choice in D_C_values:
        arr = slope_tables[(gal, D_C_choice)]
        fn = os.path.join(OUT_DIR, f"slopes_{gal}_DC_{D_C_choice:.3f}.txt")
        np.savetxt(fn, arr, header="Krep alpha_in alpha_out is_core(0/1)", fmt="%.6f")
