#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def smooth_gate(t: float, t0: float, t1: float) -> float:
    """
    Porte temporelle lissée en cosinus :
      0,                t < t0
      1/2 - 1/2 cos(pi*(t-t0)/(t1-t0)),  t0 ≤ t ≤ t1
      1,                t > t1
    """
    if t < t0:
        return 0.0
    if t > t1:
        return 1.0
    return 0.5 - 0.5 * math.cos(math.pi * (t - t0) / (t1 - t0))


def Delta(x: float) -> float:
    """Δ(x) = x² / (1 + x²)"""
    return (x * x) / (1.0 + x * x)


# ---------------------------------------------------------------------------
# Paramètres du modèle
# ---------------------------------------------------------------------------

@dataclass
class TemporalParams:
    # Taux de transitions internes
    a1: float = 0.2
    a2: float = 0.25
    b1: float = 0.1
    b2: float = 0.15
    c1: float = 0.005
    c2: float = 0.005
    # Taux de "pertes" (stabilisants)
    Mr: float = 0.001
    Mi: float = 0.001
    Mp: float = 0.001
    # Temps (min) des portes temporelles
    t0: float = 4.0
    t1: float = 10.0
    t2: float = 60.0
    t3: float = 100.0
    # Poids d'imitation
    alpha1: float = 1.0
    alpha2: float = 1.0
    beta1: float = 1.0
    beta2: float = 1.0
    gamma1: float = 1.0
    gamma2: float = 1.0
    # Épsilon numérique
    eps: float = 1e-6
    # Mortality rates (per minute)
    pi_r: float = 0.0005
    pi_i: float = 0.0002
    pi_p: float = 0.001

    # Behavioral regression coefficients (back transitions)
    k_ir: float = 0.02   # intelligent -> reflexive (loss of control)
    k_pr: float = 0.05   # panic -> reflexive
    k_pi: float = 0.01   # panic -> intelligent (calming)


@dataclass
class SpaceParams:
    # Surfaces (m²)
    S1: float = 8000.0
    S2: float = 2000.0
    S3: float = 3500.0
    # Longueurs d'ouverture (m)
    L12: float = 50.0
    L23: float = 30.0
    # Vitesses de marche (m/s)
    Vr: float = 5.0
    Vi: float = 4.0
    Vp: float = 3.0
    # Densité capacité (pers/m²)
    capacity_density: float = 4.0
    stair_capacity_factor: float = 1.0


    def Nmax(self) -> np.ndarray:
        N = np.array([self.S1, self.S2, self.S3]) * self.capacity_density
        N[1] *= self.stair_capacity_factor  # limit for node 2 (stair)
        return N

    def L_matrix(self) -> np.ndarray:
        """Matrice symétrique des ouvertures L_kj (m). Ouvert uniquement entre 1-2 et 2-3."""
        L = np.zeros((3, 3))
        L[0, 1] = L[1, 0] = self.L12
        L[1, 2] = L[2, 1] = self.L23
        return L

    def S_vector(self) -> np.ndarray:
        """Vecteur des surfaces."""
        return np.array([self.S1, self.S2, self.S3])


@dataclass
class Scenario:
    name: str
    # Comptes initiaux par zone pour (n, r, i, p, s)  -> shape (3,5)
    init_zone_counts: np.ndarray
    # Multiplicateur de vitesse (1.0 nominal)
    speed_multiplier: float = 1.0
    # Temps de simulation (minutes) et pas
    t_max_min: float = 120.0
    dt_min: float = 0.1


# ---------------------------------------------------------------------------
# Imitation / contagion
# ---------------------------------------------------------------------------

def Xi(r: float, i: float, n0: float, alpha1: float, alpha2: float, eps: float) -> float:
    # Ξ(r,i) = -α1 Δ( i/(r+eps) ) * 1/n0 + α2 Δ( r/(i+eps) ) * 1/n0
    return -alpha1 * Delta(i / (r + eps)) / n0 + alpha2 * Delta(r / (i + eps)) / n0


def Theta(r: float, p: float, n0: float, beta1: float, beta2: float, eps: float) -> float:
    # Θ(r,p) = -β1 Δ( p/(r+eps) ) * 1/n0 + β2 Δ( r/(p+eps) ) * 1/n0
    return -beta1 * Delta(p / (r + eps)) / n0 + beta2 * Delta(r / (p + eps)) / n0


def Upsilon(i: float, p: float, n0: float, gamma1: float, gamma2: float, eps: float) -> float:
    # Υ(i,p) = -γ1 Δ( p/(i+eps) ) * 1/n0 + γ2 Δ( i/(p+eps) ) * 1/n0
    return -gamma1 * Delta(p / (i + eps)) / n0 + gamma2 * Delta(i / (p + eps)) / n0


# ---------------------------------------------------------------------------
# Flux spatiaux
# ---------------------------------------------------------------------------

def rho(speed: float, L_kj: float, S_k: float) -> float:
    """ρ_kj = (V * L_kj) / S_k"""
    return (speed * L_kj) / S_k if S_k > 0 else 0.0


def compute_flux_terms(pop: np.ndarray,
                       speeds: Tuple[float, float, float],
                       L: np.ndarray,
                       S: np.ndarray,
                       Nmax: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calcule le flux net (entrées - sorties) pour r, i, p dans chaque zone,
    avec effet de congestion fort (bottleneck zone 2).
    """
    N = pop.sum(axis=1)  # total par zone
    r_flux = np.zeros(3)
    i_flux = np.zeros(3)
    p_flux = np.zeros(3)
    Vr, Vi, Vp = speeds

    for j in range(3):
        inflow_r = inflow_i = inflow_p = 0.0
        out_r = out_i = out_p = 0.0

        for k in range(3):
            if k == j or L[k, j] == 0.0:
                continue

            # --- inflow from k → j, limited by congestion in destination j
            congestion = max(0.0, 1.0 - (N[j] / (Nmax[j] + 1e-9))**3)
            inflow_r += congestion * rho(Vr, L[k, j], S[k]) * pop[k, 1]
            inflow_i += congestion * rho(Vi, L[k, j], S[k]) * pop[k, 2]
            inflow_p += congestion * rho(Vp, L[k, j], S[k]) * pop[k, 3]

            # --- outflow from j → k, reduced if destination is congested
            congestion_dest = max(0.0, 1.0 - (N[k] / (Nmax[k] + 1e-9))**3)
            out_r += congestion_dest * rho(Vr, L[j, k], S[j]) * pop[j, 1]
            out_i += congestion_dest * rho(Vi, L[j, k], S[j]) * pop[j, 2]
            out_p += congestion_dest * rho(Vp, L[j, k], S[j]) * pop[j, 3]

        r_flux[j] = inflow_r - out_r
        i_flux[j] = inflow_i - out_i
        p_flux[j] = inflow_p - out_p

    return {'r': r_flux, 'i': i_flux, 'p': p_flux}

# ---------------------------------------------------------------------------
# Système d'EDO (RHS)
# ---------------------------------------------------------------------------

def rhs(t_min: float,
        y: np.ndarray,
        params: TemporalParams,
        space: SpaceParams,
        n0_total: float) -> np.ndarray:
    """
    RHS du système avec mortalité et imitation bidirectionnelle.
    États : n, r, i, p, s, d
    """
    Y = y.reshape(3, 6).copy()
    n, r, i, p, s, d = Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3], Y[:, 4], Y[:, 5]

    # Fonctions de forçage (alert & rescue)
    Psi = smooth_gate(t_min, params.t0, params.t1)  # alerte (n -> r)
    Phi = smooth_gate(t_min, params.t2, params.t3)  # secours (i -> s)

    # Flux spatiaux (inchangé)
    L = space.L_matrix()
    S = space.S_vector()
    Nmax = space.Nmax()
    speeds = (space.Vr, space.Vi, space.Vp)
    flux = compute_flux_terms(Y[:, :5], speeds, L, S, Nmax)

    dY = np.zeros_like(Y)

    for j in range(3):
        # --- Imitation / contagion (bidirectional) ---
        Xi_val = (params.alpha1 * Delta(i[j] / (r[j] + params.eps)) +
                  params.alpha2 * Delta(r[j] / (i[j] + params.eps))) / n0_total
        Th_val = (params.beta1 * Delta(p[j] / (r[j] + params.eps)) +
                  params.beta2 * Delta(r[j] / (p[j] + params.eps))) / n0_total
        Up_val = (params.gamma1 * Delta(p[j] / (i[j] + params.eps)) +
                  params.gamma2 * Delta(i[j] / (p[j] + params.eps))) / n0_total

        # --- Mortality rates ---
        death_r = params.pi_r * r[j]
        death_i = params.pi_i * i[j]
        death_p = params.pi_p * p[j]

        # --- Differential equations ---
        dn = -Psi * n[j]

        dr = (Psi * n[j]
              + params.c1 * i[j]
              + params.c2 * p[j]
              + params.k_ir * i[j]
              + params.k_pr * p[j]
              - (params.a1 + params.a2 + params.Mr + params.pi_r) * r[j]
              + Xi_val * i[j] * r[j]
              + Th_val * p[j] * r[j])

        di = (params.a1 * r[j]
              + params.b1 * p[j]
              + params.k_pi * p[j]
              - (params.c1 + params.b2 + params.Mi + params.pi_i) * i[j]
              - Phi * i[j]
              + Up_val * p[j] * i[j]
              - Xi_val * r[j] * i[j])

        dp = (params.b2 * i[j]
              + params.a2 * r[j]
              - (params.b1 + params.c2 + params.Mp + params.pi_p) * p[j]
              - Up_val * i[j] * p[j]
              - Th_val * r[j] * p[j])

        ds = Phi * i[j]
        dd = death_r + death_i + death_p

        # --- Add flux (spatial coupling) ---
        dr += flux['r'][j]
        di += flux['i'][j]
        dp += flux['p'][j]

        dY[j, 0] = dn
        dY[j, 1] = dr
        dY[j, 2] = di
        dY[j, 3] = dp
        dY[j, 4] = ds
        dY[j, 5] = dd

    return dY.reshape(-1)


# ---------------------------------------------------------------------------
# Intégrateur RK4
# ---------------------------------------------------------------------------

def rk4(f: Callable[[float, np.ndarray], np.ndarray],
        t0: float,
        y0: np.ndarray,
        dt: float,
        n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Intégrateur RK4 à pas fixe.
    Retourne (T, Y) avec T de taille n_steps+1 et Y de shape (n_steps+1, len(y0)).
    """
    t = t0
    y = y0.copy()
    T = np.zeros(n_steps + 1)
    Y = np.zeros((n_steps + 1, len(y0)))
    T[0] = t
    Y[0] = y

    for k in range(1, n_steps + 1):
        k1 = f(t, y)
        k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = f(t + dt,       y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Garde-fou de non-négativité
        y = np.maximum(y, 0.0)
        t = t + dt
        T[k] = t
        Y[k] = y

    return T, Y


# ---------------------------------------------------------------------------
# Graphiques des scénarios (4 panneaux)
# ---------------------------------------------------------------------------

def plot_results(times: np.ndarray, states: np.ndarray, scenario_name: str):

    Y = states.reshape(len(times), 3, 6)
    labels = ['n(t)', 'r(t)', 'i(t)', 'p(t)', 's(t)', 'd(t)']

    # Totaux toutes zones
    totals = Y.sum(axis=1)  # shape (T, 5)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_all = axes[0, 0]
    ax_z1 = axes[0, 1]
    ax_z2 = axes[1, 0]
    ax_z3 = axes[1, 1]

    # Toutes zones combinées
    for k in range(5):
        ax_all.plot(times, totals[:, k], label=labels[k])
    ax_all.set_title("Toute la zone")
    ax_all.set_xlabel("temps (min)")
    ax_all.set_ylabel("Population")
    ax_all.legend()

    # Zone 1
    for k in range(5):
        ax_z1.plot(times, Y[:, 0, k], label=labels[k])
    ax_z1.set_title("Zone [1] : Jardin Majorelle")
    ax_z1.set_xlabel("temps (min)")
    ax_z1.set_ylabel("Population")

    # Zone 2
    for k in range(5):
        ax_z2.plot(times, Y[:, 1, k], label=labels[k])
    ax_z2.set_title("Zone [2] : Musée Berbère")
    ax_z2.set_xlabel("temps (min)")
    ax_z2.set_ylabel("Population")

    # Zone 3
    for k in range(5):
        ax_z3.plot(times, Y[:, 2, k], label=labels[k])
    ax_z3.set_title("Zone [3] : Rue Yves St. Laurent")
    ax_z3.set_xlabel("temps (min)")
    ax_z3.set_ylabel("Population")

    fig.suptitle(f"Simulation – {scenario_name}", fontsize=14)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Graphiques des fonctions du temps Ψ(t) et Φ(t)
# ---------------------------------------------------------------------------

def plot_time_functions(params: TemporalParams):
    t = np.linspace(0, max(params.t3 * 1.1, 100.0), 600)
    Psi = [smooth_gate(tt, params.t0, params.t1) for tt in t]
    Phi = [smooth_gate(tt, params.t2, params.t3) for tt in t]
    plt.figure(figsize=(7, 4))
    plt.plot(t, Psi, label='Ψ(t) : forçage (n→r)')
    plt.plot(t, Phi, label='Φ(t) : passage (i→s)')
    plt.xlabel('temps (min)')
    plt.ylabel('valeur')
    plt.title('Fonctions du temps Ψ(t) et Φ(t)')
    plt.legend()
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------------
# Construction des scénarios
# ---------------------------------------------------------------------------

def build_scenarios(space: SpaceParams) -> List[Scenario]:
    """
    Trois scénarios inspirés de la présentation.
    - Scénario 1 : évacuation normale légère (~300 en zone 1)
    - Scénario 2 : évacuation normale moyenne (~500 total)
    - Scénario 3 : évacuation rapide (~1200 total, proportionnel aux surfaces)
    """
    # Scénario 1
    init1 = np.zeros((3, 6))  # ← 6 colonnes (n, r, i, p, s, d)
    init1[0, 0] = 300.0
    init1[1, 0] = 50.0
    init1[2, 0] = 50.0

    # Scénario 2
    init2 = np.zeros((3, 6))  # ← idem
    init2[0, 0] = 300.0
    init2[1, 0] = 100.0
    init2[2, 0] = 100.0

    # Scénario 3 (répartition proportionnelle aux surfaces, total ~1200)
    total3 = 1200.0
    areas = np.array([space.S1, space.S2, space.S3])
    w = areas / areas.sum()
    pop3 = (w * total3).astype(float)

    init3 = np.zeros((3, 6))  # ← idem
    init3[:, 0] = pop3  # population initiale dans l’état normal (n)
    # r, i, p, s, d = 0 au départ

    return [
        Scenario(name="Scénario 1 (normale légère)", init_zone_counts=init1, speed_multiplier=1.0, t_max_min=120.0, dt_min=0.1),
        Scenario(name="Scénario 2 (normale moyenne)", init_zone_counts=init2, speed_multiplier=1.0, t_max_min=120.0, dt_min=0.1),
        Scenario(name="Scénario 3 (rapide)",          init_zone_counts=init3, speed_multiplier=1.3, t_max_min=120.0, dt_min=0.1),
    ]


# ---------------------------------------------------------------------------
# Exécution d’un scénario (intégration + affichage)
# ---------------------------------------------------------------------------

def run_scenario(scn: Scenario, base_params: TemporalParams, base_space: SpaceParams):
    # Clone des paramètres temporels
    params = TemporalParams(**asdict(base_params))

    # Clone des paramètres spatiaux + ajustement des vitesses
    space = SpaceParams(**asdict(base_space))
    space.Vr *= scn.speed_multiplier
    space.Vi *= scn.speed_multiplier
    space.Vp *= scn.speed_multiplier

    # État initial
    Y0 = scn.init_zone_counts.copy()
    y0 = Y0.reshape(-1)

    # n0_total = somme initiale des individus "normaux" (échelle imitation)
    n0_total = Y0[:, 0].sum()

    # RHS partiellement appliqué
    def f(t: float, y: np.ndarray) -> np.ndarray:
        return rhs(t, y, params, space, n0_total)

    # Intégration
    t0 = 0.0
    n_steps = int(math.ceil(scn.t_max_min / scn.dt_min))
    times, states = rk4(f, t0, y0, scn.dt_min, n_steps)

    # Graphiques 4 panneaux
    plot_results(times, states, scn.name)


# ---------------------------------------------------------------------------
# Graphiques de conclusion multi-courbes (par scénario)
# ---------------------------------------------------------------------------

def plot_evacuation_curve_multi(params: TemporalParams, space: SpaceParams):

    scenarios = build_scenarios(space)

    # On fait varier le "goulot d'étranglement" — l'ouverture effective de l'escalier
    stair_caps = np.linspace(10, 500, 30)  # correspond à N2max ou flux équivalent

    plt.figure(figsize=(7, 5))

    for scn in scenarios:
        Te_list = []

        for stair_cap in stair_caps:
            # Clone de l'espace
            sp = SpaceParams(**asdict(space))

            # ⚙️ On traduit la capacité en facteur sur L12 (la largeur de l’escalier)
            baseN2 = sp.S2 * sp.capacity_density
            sp.L12 = space.L12 * (stair_cap / baseN2)  # effet de saturation
            sp.L23 = space.L23  # inchangé

            # Ajustement des vitesses selon le scénario
            sp.Vr *= scn.speed_multiplier
            sp.Vi *= scn.speed_multiplier
            sp.Vp *= scn.speed_multiplier

            # Conditions initiales
            init = scn.init_zone_counts.copy()
            y0 = init.reshape(-1)
            n0_total = init[:, 0].sum()

            def f(t, y):
                return rhs(t, y, params, sp, n0_total)

            # Intégration sur 60 minutes
            t, sol = rk4(f, 0.0, y0, 0.1, 600)
            Y = sol.reshape(len(t), 3, 6)

            # Zone 1 = plage
            r1, i1, p1 = Y[:, 0, 1], Y[:, 0, 2], Y[:, 0, 3]
            seuil = 0.1 * init[0, 0]

            idx = np.where(r1 + i1 + p1 < seuil)[0]
            Te = t[idx[0]] if len(idx) > 0 else 60.0
            Te_list.append(Te)

        # Tracé
        plt.plot(stair_caps, Te_list, lw=2, label=scn.name)

    plt.xlim(0, 500)
    plt.ylim(10, 100)
    plt.axhline(30, ls='--', c='r', lw=1.5)
    plt.axhline(15, ls='--', c='r', lw=1.5)
    plt.xlabel(r'Capacité maximale de la zone 2 ($N_{2}^{max}$)', fontsize=13)
    plt.ylabel(r'Temps d\'évacuation $T_{evac}$ (minutes)', fontsize=13)
    plt.title('Temps d\'évacuation selon la capacité de la zone 2 (escalier)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_panic_ratio_curve_multi(params: TemporalParams, space: SpaceParams):
    """
    Taux de panique p₃(40) / N₃_total(40) vs n₃(0)
    où N₃_total = n+r+i+p+s en zone 3 à t=40 min.
    """
    scenarios = build_scenarios(space)
    n3_grid = np.linspace(0, 400, 60)

    plt.figure(figsize=(7, 5))
    for scn in scenarios:
        sp = SpaceParams(**asdict(space))
        sp.Vr *= scn.speed_multiplier
        sp.Vi *= scn.speed_multiplier
        sp.Vp *= scn.speed_multiplier

        ratios = []
        for n3_0 in n3_grid:
            init = scn.init_zone_counts.copy()
            init[2, 0] = n3_0
            y0 = init.reshape(-1)
            n0_total = init[:, 0].sum()

            def f(t, y): return rhs(t, y, params, sp, n0_total)

            # 40 minutes = 200 pas de 0.2 min
            t, sol = rk4(f, 0.0, y0, 0.2, 200)
            Y = sol.reshape(len(t), 3, 6)
            idx = np.argmin(np.abs(t - 40.0))
            N3_tot = Y[idx, 2, :].sum()
            p3 = Y[idx, 2, 3]
            ratios.append(p3 / (N3_tot + 1e-12))

        plt.plot(n3_grid, ratios, label=scn.name)

    plt.xlabel('n₃(0)')
    plt.ylabel('p₃(40 min) / N₃_total(40 min)')
    plt.title('Taux de panique à 40 min selon n₃(0) — par scénario')
    plt.legend()
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------

def main():
    # Paramètres de base
    params = TemporalParams(
        a1=0.2, a2=0.25, b1=0.1, b2=0.15,
        c1=0.005, c2=0.005,
        Mr=0.001, Mi=0.001, Mp=0.001,
        t0=4.0, t1=10.0, t2=60.0, t3=100.0,
        alpha1=1.0, alpha2=1.0, beta1=1.0, beta2=1.0, gamma1=1.0, gamma2=1.0,
        eps=1e-6
    )

    space = SpaceParams(
        S1=8000.0, S2=2000.0, S3=3500.0,
        L12=50.0, L23=30.0,
        Vr=5.0, Vi=4.0, Vp=3.0,
        capacity_density=4.0
    )

    # 1) Simulations de scénarios (graphiques 4 panneaux)
    for scn in build_scenarios(space):
        run_scenario(scn, params, space)

    # 2) Fonctions du temps
    plot_time_functions(params)

    # 3) Graphiques de conclusion multi-courbes
    plot_evacuation_curve_multi(params, space)
    plot_panic_ratio_curve_multi(params, space)


if __name__ == "__main__":
    main()