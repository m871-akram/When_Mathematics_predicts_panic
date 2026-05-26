# When Mathematics Predicts Panic — Disaster Reaction Simulation 

This repository contains the simulation of human reaction during a disaster across multiple zones, built around an extended SIR-like behavioral model and simple spatial flows. It produces illustrative plots and data series for evacuation outcomes, making it easy to tune behavioral switches and spatial constraints.

- Main script: `simulation_panic.py`

## Behavioral Compartments (per zone)

- $n$ : normal
- $r$ : reflexive
- $i$ : intelligent
- $p$ : panic
- $s$ : rescued (safe)
- $d$ : deceased (if mortality is enabled)

## Dynamics include

- Imitation/contagion terms between behaviors using smoothed nonlinearities:
  - $\Xi(r, i)$
  - $\Theta(r, p)$
  - $\Upsilon(i, p)$ 
  - with $\Delta(x) = \frac{x^2}{1 + x^2}$
- Time forcing via smooth gate functions (cosine doors):
  - $\Psi(t)$ and $\Phi(t)$
- Spatial flows between three zones with capacity and openings $L_{jk}$
- Capacity saturation: $1 - \frac{N_j}{N^{\max}_j}$
- Runge–Kutta 4 (RK4) integration

## Scenarios

Three ready-to-run scenarios produce:
- 4-panel plots (total + 3 zones) for the 5 tracked states $(n, r, i, p, s)$
- Time-function plots for $\Psi(t)$ and $\Phi(t)$
- Summary curves across scenarios:
  - Evacuation time $T_\text{evac}$ vs. stair capacity (or $n_3(0)$ variants)
  - Panic ratio $\frac{p_3(40)}{N_{3,\text{total}}(40)}$ vs. $n_3(0)$

Example zones used in the script:

- **[1] Jardin Majorelle**  
- **[2] Musée Berbère** (stair/bottleneck)
- **[3] Rue Yves St. Laurent**

---

## Configuration

**Key dataclasses you can adjust in the script:**

- **TemporalParams**
  - $a_1$, $a_2$, $b_1$, $b_2$, $c_1$, $c_2$: internal transition rates
  - $M_r$, $M_i$, $M_p$: stabilizing losses
  - $t_0$, $t_1$, $t_2$, $t_3$: time thresholds for gates $\Psi$, $\Phi$
  - $\alpha_1$, $\alpha_2$, $\beta_1$, $\beta_2$, $\gamma_1$, $\gamma_2$: imitation weights
  - $\varepsilon$: numerical epsilon
  - $\pi_r$, $\pi_i$, $\pi_p$: optional mortality rates (per minute)
  - $k_{ir}$, $k_{pr}$, $k_{pi}$: behavioral back-transitions (e.g. panic $\to$ reflexive)

- **SpaceParams**
  - $S_1$, $S_2$, $S_3$: zone surface areas (m²)
  - $L_{12}$, $L_{23}$: opening lengths (m)
  - $V_r$, $V_i$, $V_p$: walking speeds (m/s) for $r$, $i$, $p$ groups
  - `capacity_density`: persons per m²
  - `stair_capacity_factor`: additional cap on zone 2 (bottleneck)

Scenario definition is encapsulated in the `Scenario` structure returned by `build_scenarios(space)` and used in `run_scenario()`.
