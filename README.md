# When Mathematics Predicts Panic — Disaster Reaction Simulation 

This repository contains a full simulation of human reaction during a disaster across multiple zones, built around an extended SIR-like behavioral model and simple spatial flows. It produces illustrative figures for several scenarios and summary plots that relate evacuation time and panic ratio to initial conditions and bottleneck capacities.

The code is self-contained (no SciPy) and uses a custom RK4 integrator with NumPy and Matplotlib only.

- Main script: `simulation_panic.py`



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





