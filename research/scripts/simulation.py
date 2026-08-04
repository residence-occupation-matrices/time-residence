#!/usr/bin/env python3
# ruff: noqa: E741
"""
Created on Thu Sep 30 14:35:54 2021

@author: albertakuno
"""

import random

import matplotlib.pyplot as plt
import numpy as np
from odeintw import odeintw


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


# n=residence_matrix.shape[0]
n = 20

rng = np.random.default_rng()
Lambda = np.ones(n) * 9
kappa = np.ones(n) * 1 / 7
beta = rng.uniform(low=0, high=3, size=(n))
mu = np.ones(n) * 1 / 140
gamma = np.ones(n) * 1 / 14
phi = np.ones(n) * 0.003
tau = np.ones(n) * 1 / 10
Nbar = Lambda * 1 / mu


# E_initial = rng.integers(low=0, high=20, size=(n))
E_initial = np.zeros(n)

I_initial = rng.integers(low=0, high=10, size=(n))

R_initial = np.zeros(n)
S_initial = Nbar - (I_initial + E_initial + R_initial)
M_initial = np.array([S_initial, E_initial, I_initial, R_initial])

# alfa=np.linspace(0,1,20)
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

# n=residence_matrix.shape[0]
t = np.linspace(0, 200, 100)
alfa = np.random.default_rng().uniform(size=(n,))


def build_pstar(p):
    result = np.zeros_like(p)
    # p = np.array(f(20))
    for i in range(n):
        result[i, i] = 1 - alfa[i]
        for j in range(n):
            result[i, j] += alfa[i] * p[i, j]
    return result


# pstar=pstar(residence_matrix)

pstar = build_pstar(p)

sol = odeintw(systemSEIR, M_initial, t, args=(beta, gamma, pstar))

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(t, sol[:, 0, :])  # all susceptible curves
ax.plot(t, sol[:, 1, :])  # all exposed curves
ax.plot(t, sol[:, 2, :])  # all infected curves
ax.plot(t, sol[:, 3, :])  # all recovered curves
fig.savefig("random_full_simulation.png", dpi=350)
# R0i_P=np.zeros(n)

# for i in range(n):
#     R0_j=(beta[i]*kappa[i])/((kappa[i]+mu[i])*(gamma[i]+phi[i]+mu[i]))
#     for j in range(n):
#         R0i_P[i]=beta[i]*np.sum((1/beta[j])*(pstar[:,i])[i]*R0_j)

# v=(kappa+mu)*(gamma+phi+mu)
# for i in range(n):
#     temp=np.sum(pstar[:,i]*kappa/v)
#     R0i_P[i]=beta[i]*temp
