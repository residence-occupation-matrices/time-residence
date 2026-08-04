#!/usr/bin/env python3
# ruff: noqa: E741
"""Run a synthetic multi-patch SEIR experiment with a random ROM."""

import random

import matplotlib.pyplot as plt
import numpy as np
from odeintw import odeintw
from project_paths import OUTPUT_ROOT, prepare_output


def systemSEIR(M, t, beta, gama, pstar):
    S, E, I, R = M
    Ntilde = np.transpose(pstar) @ Nbar
    dS_dt = (
        Lambda
        - np.diag(S)
        @ pstar
        @ np.diag(beta)
        @ np.linalg.inv(np.diag(Ntilde))
        @ np.transpose(pstar)
        @ I
        - np.diag(mu) @ S
        + np.diag(tau) @ R
    )
    dE_dt = (
        np.diag(S)
        @ pstar
        @ np.diag(beta)
        @ np.linalg.inv(np.diag(Ntilde))
        @ np.transpose(pstar)
        @ I
        - np.diag(kappa + mu) @ E
    )
    dI_dt = np.diag(kappa) @ E - np.diag(gamma + phi + mu) @ I
    dR_dt = np.diag(gamma) @ I - np.diag(tau + mu) @ R
    return np.array([dS_dt, dE_dt, dI_dt, dR_dt])


n = 20

rng = np.random.default_rng(2025)
random.seed(2025)
Lambda = np.ones(n) * 9
kappa = np.ones(n) * 1 / 7
beta = rng.uniform(low=0, high=3, size=(n))
mu = np.ones(n) * 1 / 140
gamma = np.ones(n) * 1 / 14
phi = np.ones(n) * 0.003
tau = np.ones(n) * 1 / 10
Nbar = Lambda * 1 / mu


E_initial = np.zeros(n)

I_initial = rng.integers(low=0, high=10, size=(n))

R_initial = np.zeros(n)
S_initial = Nbar - (I_initial + E_initial + R_initial)
M_initial = np.array([S_initial, E_initial, I_initial, R_initial])

precision = 1000000


def f(n):
    matrix = []
    for _ in range(n):
        lineLst = []
        total = 0
        crtPrec = precision
        for _ in range(n - 1):
            val = random.randrange(crtPrec)
            total += val
            lineLst.append(float(val) / precision)
            crtPrec -= val
        lineLst.append(float(precision - total) / precision)
        matrix.append(lineLst)
    return matrix


p = np.array(f(n))

t = np.linspace(0, 200, 100)
alfa = rng.uniform(size=(n,))


def build_pstar(p):
    result = np.zeros_like(p)
    for i in range(n):
        result[i, i] = 1 - alfa[i]
        for j in range(n):
            result[i, j] += alfa[i] * p[i, j]
    return result


pstar = build_pstar(p)

sol = odeintw(systemSEIR, M_initial, t, args=(beta, gamma, pstar))

fig, ax = plt.subplots(figsize=(10, 10))
for index, (label, color) in enumerate(
    (
        ("Susceptible", "C0"),
        ("Exposed", "C1"),
        ("Infected", "C3"),
        ("Recovered", "C2"),
    )
):
    lines = ax.plot(t, sol[:, index, :], color=color, alpha=0.55)
    lines[0].set_label(label)
ax.set_xlabel("Time")
ax.set_ylabel("Population")
ax.legend()
fig.savefig(prepare_output(OUTPUT_ROOT / "random_full_simulation.png"), dpi=350)
