# When Mathematics Predicts Panic — Disaster Reaction Simulation 

This repository contains a full simulation of human reaction during a disaster across multiple zones, built around an extended SIR-like behavioral model and simple spatial flows. It produces illustrative figures for several scenarios and summary plots that relate evacuation time and panic ratio to initial conditions and bottleneck capacities.

The code is self-contained (no SciPy) and uses a custom RK4 integrator with NumPy and Matplotlib only.

- Main script: `simulation_panic.py`
- Language of code and comments: French (with mathematical notation and some English inline)


## Overview

Behavioral compartments (per zone):
- n: normal
- r: reflexive
- i: intelligent
- p: panic
- s: rescued (safe)
- d: deceased (if mortality is enabled)

Dynamics include:
- Imitation/contagion terms between behaviors using smoothed nonlinearities:
  - Ξ(r, i), Θ(r, p), Υ(i, p) with Δ(x) = x²/(1 + x²)
- Time forcing via smooth gate functions (cosine doors) Ψ(t) and Φ(t)
- Spatial flows between three zones with capacity and openings L_{jk}
- Capacity saturation: (1 − N_j / Nmax_j)
- Runge–Kutta 4 (RK4) integration

Three ready-to-run scenarios produce:
- 4-panel plots (total + 3 zones) for the 5 tracked states (n, r, i, p, s)
- Time-function plots for Ψ(t) and Φ(t)
- Summary curves across scenarios:
  - Evacuation time Tevac vs. stair capacity (or n3(0) variants)
  - Panic ratio p3(40) / N3_total(40) vs. n3(0)

Example zones used in the script:
- [1] Jardin Majorelle
- [2] Musée Berbère (stair/bottleneck)
- [3] Rue Yves St. Laurent

## Requirements

- Python 3.9+
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install -U numpy matplotlib
```

## Running the simulation

Execute the main script:

```bash
python simulation_panic.py
```

What you will see:
- For each scenario, a figure with 4 panels showing the evolution of n, r, i, p, s (total and per zone)
- A figure with Ψ(t) and Φ(t)
- Two summary figures:
  - Evacuation time vs. capacity (or an equivalent bottleneck sweep)
  - Panic ratio at 40 minutes in zone 3 vs. initial n3(0)

The script uses the `main()` function at the bottom of `simulation_panic.py` which:
1. Creates default temporal parameters (`TemporalParams`)
2. Creates default spatial parameters (`SpaceParams`)
3. Runs multiple built-in scenarios via `build_scenarios()` and `run_scenario()`
4. Produces summary plots with `plot_evacuation_curve_multi()` and `plot_panic_ratio_curve_multi()`

## Configuration

Key dataclasses you can adjust in the script:

- TemporalParams
  - a1, a2, b1, b2, c1, c2: internal transition rates
  - Mr, Mi, Mp: stabilizing losses
  - t0, t1, t2, t3: time thresholds for gates Ψ, Φ
  - alpha1, alpha2, beta1, beta2, gamma1, gamma2: imitation weights
  - eps: numerical epsilon
  - pi_r, pi_i, pi_p: optional mortality rates (per minute)
  - k_ir, k_pr, k_pi: behavioral back-transitions (e.g., panic -> reflexive)

- SpaceParams
  - S1, S2, S3: zone surface areas (m²)
  - L12, L23: opening lengths (m)
  - Vr, Vi, Vp: walking speeds (m/s) for r, i, p groups
  - capacity_density: persons per m²
  - stair_capacity_factor: additional cap on zone 2 (bottleneck)

Scenario definition is encapsulated in the `Scenario` structure returned by `build_scenarios(space)` and used in `run_scenario()`.

## Outputs

- Interactive Matplotlib figures (no files are saved by default). You can add `plt.savefig(...)` calls to persist figures.
- Time arrays and state arrays are kept in memory inside each plotting function; adapt as needed if you wish to export CSV/NPY.

## Reproducibility and Performance

- Deterministic ODE integration via RK4 with fixed step sizes set inside the plotting routines.
- For parameter sweeps (e.g., panic ratios and evacuation times), the script loops across grids and capacities. Adjust grid sizes for speed/accuracy trade-offs.



