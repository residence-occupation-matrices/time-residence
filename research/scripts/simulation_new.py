# ruff: noqa: E741
"""Explore a multi-patch model with three infectious compartments."""

import json
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from odeintw import odeintw
from project_paths import DATA_ROOT, OUTPUT_ROOT, prepare_output
from scipy.integrate import solve_ivp
from tqdm import tqdm

rng = np.random.default_rng()


def SEIR_system(M, t, var_dict, pstar):
    S, E, Ia, Is, Iw, R = M
    S, E, Ia, Is, Iw, R = (
        S.reshape(-1, 1),
        E.reshape(-1, 1),
        Ia.reshape(-1, 1),
        Is.reshape(-1, 1),
        Iw.reshape(-1, 1),
        R.reshape(-1, 1),
    )
    I = Ia + Is + Iw
    N_tilde = pstar.T @ var_dict["N_bar"]
    beta, gamma, kappa, mu, psi, tau, eta_a, eta_s, eta_w = (
        var_dict["beta"].reshape(-1, 1),
        var_dict["gamma"].reshape(-1, 1),
        var_dict["kappa"].reshape(-1, 1),
        var_dict["mu"].reshape(-1, 1),
        var_dict["psi"].reshape(-1, 1),
        var_dict["tau"].reshape(-1, 1),
        var_dict["eta_a"].reshape(-1, 1),
        var_dict["eta_s"].reshape(-1, 1),
        var_dict["eta_w"].reshape(-1, 1),
    )

    dSdt = (
        mu * (S + E + Ia + Is + Iw + R)
        - np.diag(
            S.reshape(
                -1,
            )
        )
        @ pstar
        @ np.diag(
            beta.reshape(
                -1,
            )
        )
        @ np.linalg.inv(
            np.diag(
                N_tilde.reshape(
                    -1,
                )
            )
        )
        @ pstar.T
        @ I
        - np.diag(
            mu.reshape(
                -1,
            )
        )
        @ S
        + np.diag(
            tau.reshape(
                -1,
            )
        )
        @ R
    )

    dEdt = (
        np.diag(
            S.reshape(
                -1,
            )
        )
        @ pstar
        @ np.diag(
            beta.reshape(
                -1,
            )
        )
        @ np.linalg.inv(
            np.diag(
                N_tilde.reshape(
                    -1,
                )
            )
        )
        @ pstar.T
        @ I
        - np.diag(
            (kappa + mu).reshape(
                -1,
            )
        )
        @ E
    )

    dIadt = (
        np.diag(
            eta_a.reshape(
                -1,
            )
        )
        @ np.diag(
            kappa.reshape(
                -1,
            )
        )
        @ E
        - np.diag(
            (gamma + psi + mu).reshape(
                -1,
            )
        )
        @ Ia
    )

    dIsdt = (
        np.diag(
            eta_s.reshape(
                -1,
            )
        )
        @ np.diag(
            kappa.reshape(
                -1,
            )
        )
        @ E
        - np.diag(
            (gamma + psi + mu).reshape(
                -1,
            )
        )
        @ Is
    )

    dIwdt = (
        np.diag(
            eta_w.reshape(
                -1,
            )
        )
        @ np.diag(
            kappa.reshape(
                -1,
            )
        )
        @ E
        - np.diag(
            (gamma + psi + mu).reshape(
                -1,
            )
        )
        @ Iw
    )

    dRdt = (
        np.diag(
            gamma.reshape(
                -1,
            )
        )
        @ I
        - np.diag(
            (tau + mu).reshape(
                -1,
            )
        )
        @ R
    )

    dSdt, dEdt, dIadt, dIsdt, dIwdt, dRdt = (
        dSdt.reshape(
            -1,
        ),
        dEdt.reshape(
            -1,
        ),
        dIadt.reshape(
            -1,
        ),
        dIsdt.reshape(
            -1,
        ),
        dIwdt.reshape(
            -1,
        ),
        dRdt.reshape(
            -1,
        ),
    )
    return np.array([dSdt, dEdt, dIadt, dIsdt, dIwdt, dRdt])


def initialise_variables(n):
    var_dict = {}
    var_dict["mu"] = np.ones((n, 1)) * (1 / (70 * 365))
    var_dict["lamda"] = var_dict["mu"]
    var_dict["N_bar"] = var_dict["lamda"] * 1 / var_dict["mu"]
    var_dict["beta"] = rng.uniform(low=1.1, high=2.3, size=(n, 1))
    var_dict["gamma"] = np.ones((n, 1)) * 0.13
    var_dict["kappa"] = np.ones((n, 1)) * 0.196078
    var_dict["psi"] = np.ones((n, 1)) * 0.11
    var_dict["tau"] = np.ones((n, 1)) * (1 / 365)
    var_dict["eta_a"] = np.ones((n, 1)) * 0.8
    var_dict["eta_s"] = np.ones((n, 1)) * 0.1213
    var_dict["eta_w"] = 1 - var_dict["eta_a"] - var_dict["eta_s"]

    var_dict["E0"] = np.zeros((n, 1))
    var_dict["Ia0"] = np.zeros((n, 1))
    var_dict["Is0"] = np.zeros((n, 1))
    var_dict["Iw0"] = np.zeros((n, 1))
    indices_asymp = rng.integers(low=0, high=n, size=10)
    indices_symp = rng.integers(low=0, high=n, size=3)
    indices_severe = rng.integers(low=0, high=n, size=1)
    for i in indices_asymp:
        var_dict["Ia0"][i] = rng.integers(low=0, high=7)
    for i in indices_symp:
        var_dict["Is0"][i] = rng.integers(low=0, high=4)
    for i in indices_severe:
        var_dict["Iw0"][i] = rng.integers(low=0, high=3)
    var_dict["R0"] = np.zeros((n, 1))
    var_dict["S0"] = var_dict["N_bar"] - (
        var_dict["E0"] + var_dict["Ia0"] + var_dict["Is0"] + var_dict["Iw0"] + var_dict["R0"]
    )
    var_dict["M0"] = np.array(
        [
            var_dict["S0"],
            var_dict["E0"],
            var_dict["Ia0"],
            var_dict["Is0"],
            var_dict["Iw0"],
            var_dict["R0"],
        ]
    )

    return var_dict


def generate_random_p(n):
    p = rng.random(size=(n, n))
    p = p / p.sum(axis=1).reshape(-1, 1)
    return p


def generate_random_alpha(n):
    return rng.uniform(low=0, high=1, size=(n, 1))


def generate_random_pstar(n):
    p = generate_random_p(n)
    alpha = generate_random_alpha(n)
    pstar = p * alpha.reshape(-1, 1) + np.diag(
        1
        - alpha.reshape(
            -1,
        )
    )
    return pstar


def generate_actual_pstar(p, alpha):
    pstar = p * alpha.reshape(-1, 1) + np.diag(
        1
        - alpha.reshape(
            -1,
        )
    )
    return pstar


def calculate_global_reproduction_number(pstar, var_dict):
    G = (
        np.diag(
            var_dict["N_bar"].reshape(
                -1,
            )
        )
        @ pstar
        @ np.diag(
            var_dict["beta"].reshape(
                -1,
            )
        )
        @ np.linalg.inv(
            np.diag(
                (var_dict["N_bar"].T @ pstar).reshape(
                    -1,
                )
            )
        )
        @ pstar.T
    )
    V = np.diag(
        var_dict["kappa"].reshape(
            -1,
        )
    ) @ np.linalg.inv(
        np.diag(
            (
                (var_dict["kappa"] + var_dict["mu"])
                * (var_dict["gamma"] + var_dict["psi"] + var_dict["mu"])
            ).reshape(
                -1,
            )
        )
    )
    eigvals = np.linalg.eigvals(G @ V)
    R_0 = np.abs(eigvals).max()
    return R_0


def calculate_individual_reproduction_numbers(var_dict):
    R_0 = (var_dict["beta"] * var_dict["kappa"]) / (
        (var_dict["kappa"] + var_dict["mu"])
        * (var_dict["gamma"] + var_dict["psi"] + var_dict["mu"])
    )
    return R_0


def check_varying_n(low=200, high=520, step=1, ax=None):
    for n in tqdm(range(low, high, step)):
        start = time.time()
        t = np.linspace(0, 600, 300)
        var_dict = initialise_variables(n)
        pstar = generate_random_pstar(n)
        sol = odeintw(SEIR_system, var_dict["M0"], t, args=(var_dict, pstar))
        fig, ax = plt.subplots(2, 2, figsize=(14, 14))
        ax[0][0].plot(t, sol[:, 0, :, 0])
        ax[0][0].set_title("Susceptible")
        ax[0][1].plot(t, sol[:, 1, :, 0])
        ax[0][1].set_title("Exposed")
        ax[1][0].plot(t, sol[:, 2, :, 0])
        ax[1][0].plot(t, sol[:, 3, :, 0])
        ax[1][0].set_title("Infected")
        ax[1][1].plot(t, sol[:, 4, :, 0])
        ax[1][1].set_title("Recovered")
        fig.savefig(
            prepare_output(OUTPUT_ROOT / "final_pop_experiment" / f"random_simulation_n_{n}.png"),
            dpi=350,
        )
        plt.close(fig)
        end = time.time()
        global_R0 = calculate_global_reproduction_number(pstar, var_dict)
        individual_r0 = calculate_individual_reproduction_numbers(var_dict)
        R_0_bound = individual_r0.min() < global_R0 < individual_r0.max()
        tqdm.write(
            f"n: {n}, t: {end - start}, R_0: {global_R0}, "
            f"I_n: {sol[-1, 2, :, 0].sum()}, R_0 bound: {R_0_bound}"
        )
    return ax


def polygon_under_graph(xlist, ylist):
    """Return vertices for the area under a line with ascending x-values."""
    return [(xlist[0], 0.0), *zip(xlist, ylist, strict=True), (xlist[-1], 0.0)]


def plot3d(data, t, ax=None):
    zs = range(data.shape[2])
    verts = []
    for i in zs:
        ys = data[:, 0, i, 0]
        verts.append(polygon_under_graph(t, ys))

    if not ax:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 12))

    poly = PolyCollection(verts, facecolors=["r", "g", "b", "y"], alpha=0.2)
    ax.add_collection3d(poly, zs=zs, zdir="y")
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(0, len(zs))
    return ax


def check_varying_time_full_n():
    residence_matrix = json.load(
        open(
            DATA_ROOT / "normal_avg_res_mat" / "albert_avg_res_mat_First_First.json",
        )
    )["residence_matrix"]
    residence_matrix = np.array(residence_matrix)
    alpha = json.load(
        open(
            DATA_ROOT / "alpha_values" / "albert_alpha_values_First_First.json",
        )
    )["alpha_values"]
    alpha = np.array(alpha)

    n = residence_matrix.shape[0]
    var_dict = initialise_variables(n)
    pstar = generate_actual_pstar(residence_matrix, alpha)

    for t_end in tqdm(range(20, 220, 10)):
        start = time.time()
        t = np.linspace(0, t_end, t_end // 2)
        sol = odeintw(SEIR_system, var_dict["M0"], t, args=(var_dict, pstar))
        json.dump(
            {"sol": sol.tolist()},
            open(
                prepare_output(OUTPUT_ROOT / f"forward_simulation_t_{t_end}.json"),
                "w",
            ),
            indent=2,
        )

        fig, ax = plt.subplots(2, 2, figsize=(14, 14))
        ax[0][0].plot(t, sol[:, 0, :, 0])
        ax[0][0].set_title("Susceptible")
        ax[0][1].plot(t, sol[:, 1, :, 0])
        ax[0][1].set_title("Exposed")
        ax[1][0].plot(t, sol[:, 2, :, 0])
        ax[1][0].set_title("Infected")
        ax[1][1].plot(t, sol[:, 3, :, 0])
        ax[1][1].set_title("Recovered")
        fig.savefig(
            prepare_output(OUTPUT_ROOT / "varying_time" / f"full_simulation_t_{t_end}.png"),
            dpi=350,
        )
        plt.close(fig)
        end = time.time()
        tqdm.write(f"Time taken - t: {t_end}, t: {end - start}")
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": "3d"})
        ax = plot3d(sol, t, ax)
        fig.savefig(
            prepare_output(OUTPUT_ROOT / "varying_time" / f"full_simulation_t_{t_end}.png"),
            dpi=350,
        )
        pickle.dump(
            fig,
            open(
                prepare_output(
                    OUTPUT_ROOT / "varying_time" / f"dump_full_simulation_t_{t_end}.png.pkl"
                ),
                "wb",
            ),
        )
        plt.close(fig)
    return


def compare_single_sir_model():
    n = 2
    multipatch_var_dict = initialise_variables(n)
    beta_mean = multipatch_var_dict["beta"].mean()
    t = np.arange(0, 120)
    pstar = generate_random_pstar(n)
    sol_multipatch = odeintw(
        SEIR_system, multipatch_var_dict["M0"], t, args=(multipatch_var_dict, pstar)
    )
    sol_multipatch_sum = {
        "S": sol_multipatch[:, 0, :, 0].sum(axis=1),
        "E": sol_multipatch[:, 1, :, 0].sum(axis=1),
        "I": sol_multipatch[:, 2, :, 0].sum(axis=1),
        "R": sol_multipatch[:, 3, :, 0].sum(axis=1),
    }

    var_dict = {}
    var_dict["beta"] = beta_mean
    var_dict["gamma"] = 1 / 14
    var_dict["kappa"] = 1 / 7
    var_dict["lamda"] = 9
    var_dict["mu"] = 1 / 140
    var_dict["psi"] = 0.003
    var_dict["tau"] = 1 / 10
    var_dict["N_bar"] = n * var_dict["lamda"] * 1 / var_dict["mu"]

    var_dict["E0"] = 0
    var_dict["I0"] = rng.integers(low=0, high=10, size=(1,))[0]
    var_dict["R0"] = 0
    var_dict["S0"] = var_dict["N_bar"] - (var_dict["E0"] + var_dict["I0"] + var_dict["R0"])
    var_dict["M0"] = np.array([var_dict["S0"], var_dict["E0"], var_dict["I0"], var_dict["R0"]])

    def SEIR_single(t, M, var_dict):
        S, E, I, R = M
        dSdt = (
            var_dict["lamda"]
            - beta_mean * S * I / var_dict["N_bar"]
            - var_dict["mu"] * S
            + var_dict["tau"] * R
        )
        dEdt = beta_mean * S * I / var_dict["N_bar"] - (var_dict["kappa"] + var_dict["mu"]) * E
        dIdt = var_dict["kappa"] * E - (var_dict["gamma"] + var_dict["psi"] + var_dict["mu"]) * I
        dRdt = var_dict["gamma"] * I - (var_dict["tau"] + var_dict["mu"]) * R

        return [dSdt, dEdt, dIdt, dRdt]

    sol_single = solve_ivp(
        SEIR_single, (t.min(), t.max()), var_dict["M0"], t_eval=t, args=(var_dict,)
    )
    sol_single_dict = {
        "S": sol_single.y[0, :],
        "E": sol_single.y[1, :],
        "I": sol_single.y[2, :],
        "R": sol_single.y[3, :],
    }

    fig, ax = plt.subplots(2, 2, figsize=(14, 14))
    ax[0][0].plot(sol_multipatch_sum["S"], "k-", label="Multipatch sum")
    ax[0][0].plot(sol_single_dict["S"], "r-", label="Single SEIR")
    ax[0][0].grid()
    ax[0][0].legend()
    ax[0][0].set_title("Susceptibles")
    ax[0][1].plot(sol_multipatch_sum["E"], "k-", label="Multipatch sum")
    ax[0][1].plot(sol_single_dict["E"], "r-", label="Single SEIR")
    ax[0][1].grid()
    ax[0][1].legend()
    ax[0][1].set_title("Exposed")
    ax[1][0].plot(sol_multipatch_sum["I"], "k-", label="Multipatch sum")
    ax[1][0].plot(sol_single_dict["I"], "r-", label="Single SEIR")
    ax[1][0].grid()
    ax[1][0].legend()
    ax[1][0].set_title("Infected")
    ax[1][1].plot(sol_multipatch_sum["R"], "k-", label="Multipatch sum")
    ax[1][1].plot(sol_single_dict["R"], "r-", label="Single SEIR")
    ax[1][1].grid()
    ax[1][1].legend()
    ax[1][1].set_title("Recovered")

    fig.savefig(prepare_output(OUTPUT_ROOT / "comparison_single_multi.png"), dpi=350)
    return


def final_experiment():
    res_mat = json.load(
        open(
            DATA_ROOT / "normal_avg_res_mat" / "albert_avg_res_mat_First_First.json",
        )
    )
    residence_matrix = np.array(res_mat["residence_matrix"])
    alpha = json.load(
        open(
            DATA_ROOT / "alpha_values" / "albert_alpha_values_First_First.json",
        )
    )["alpha_values"]
    alpha = np.array(alpha)
    agebs_to_remove = res_mat["agebs_to_remove"]
    n = residence_matrix.shape[0]
    multipatch_var_dict = initialise_variables(n)
    temp = json.load(open(DATA_ROOT / "ageb-population-mapping.json"))
    population_dict = {int(k): v for k, v in temp.items()}
    population = np.array(sorted(population_dict.items(), key=lambda item: item[0]))[:582, 1]
    final_pop = np.delete(population, agebs_to_remove, 0)
    multipatch_var_dict["N_bar"] = final_pop.reshape(-1, 1)
    multipatch_var_dict["lamda"] = multipatch_var_dict["mu"] * multipatch_var_dict["N_bar"]
    multipatch_var_dict["E0"] = np.zeros((n, 1))
    multipatch_var_dict["R0"] = np.zeros((n, 1))
    multipatch_var_dict["S0"] = multipatch_var_dict["N_bar"] - (
        multipatch_var_dict["E0"]
        + multipatch_var_dict["Ia0"]
        + multipatch_var_dict["Is0"]
        + multipatch_var_dict["Iw0"]
        + multipatch_var_dict["R0"]
    )
    multipatch_var_dict["M0"] = np.array(
        [
            multipatch_var_dict["S0"],
            multipatch_var_dict["E0"],
            multipatch_var_dict["Ia0"],
            multipatch_var_dict["Is0"],
            multipatch_var_dict["Iw0"],
            multipatch_var_dict["R0"],
        ]
    )
    pstar = generate_actual_pstar(residence_matrix, alpha)

    start = time.time()
    t = np.arange(0, 120)
    sol = odeintw(SEIR_system, multipatch_var_dict["M0"], t, args=(multipatch_var_dict, pstar))
    end = time.time()
    print(f"Entire simulation took: t - {end - start}")
    json.dump(
        {"sol": sol.tolist()},
        open(
            prepare_output(OUTPUT_ROOT / "final_pop_exp" / f"forward_simulation_n_{n}.json"),
            "w",
        ),
        indent=2,
    )

    def SEIR_single(t, M, var_dict):
        S, E, Ia, Is, Iw, R = M
        I = Ia + Is + Iw
        N_bar = var_dict["M0"].sum()
        dSdt = (
            var_dict["lamda"]
            - var_dict["beta"] * S * I / N_bar
            - var_dict["mu"] * S
            + var_dict["tau"] * R
        )
        dEdt = var_dict["beta"] * S * I / N_bar - (var_dict["kappa"] + var_dict["mu"]) * E
        dIadt = (
            var_dict["eta_a"] * var_dict["kappa"] * E
            - (var_dict["gamma"] + var_dict["psi"] + var_dict["mu"]) * Ia
        )
        dIsdt = (
            var_dict["eta_s"] * var_dict["kappa"] * E
            - (var_dict["gamma"] + var_dict["psi"] + var_dict["mu"]) * Is
        )
        dIwdt = (
            var_dict["eta_w"] * var_dict["kappa"] * E
            - (var_dict["gamma"] + var_dict["psi"] + var_dict["mu"]) * Iw
        )
        dRdt = var_dict["gamma"] * I - (var_dict["tau"] + var_dict["mu"]) * R

        return [dSdt, dEdt, dIadt, dIsdt, dIwdt, dRdt]

    beta_mean = multipatch_var_dict["beta"].mean()
    max_pop_inflow = 105
    beta_max_inflow = multipatch_var_dict["beta"][max_pop_inflow][0]
    print(f"Beta mean: {beta_mean}")
    print(f"Beta (max pop inflow (ageb: {max_pop_inflow})): {beta_max_inflow}")

    var_dict = {}
    var_dict["beta"] = beta_mean
    var_dict["N_bar"] = final_pop.sum()
    var_dict["gamma"] = 0.13
    var_dict["kappa"] = 0.196078
    var_dict["mu"] = 1 / (70 * 365)
    var_dict["lamda"] = var_dict["mu"] * var_dict["N_bar"]
    var_dict["psi"] = 0.11
    var_dict["tau"] = 1 / 365
    var_dict["eta_a"] = 0.8
    var_dict["eta_s"] = 0.1213
    var_dict["eta_w"] = 1 - var_dict["eta_a"] - var_dict["eta_s"]

    var_dict["E0"] = 0
    var_dict["Ia0"] = 0
    var_dict["Is0"] = 1
    var_dict["Iw0"] = 0
    var_dict["R0"] = 0
    var_dict["S0"] = var_dict["N_bar"] - (
        var_dict["E0"] + var_dict["Ia0"] + var_dict["Is0"] + var_dict["Iw0"] + var_dict["R0"]
    )
    var_dict["M0"] = np.array(
        [
            var_dict["S0"],
            var_dict["E0"],
            var_dict["Ia0"],
            var_dict["Is0"],
            var_dict["Iw0"],
            var_dict["R0"],
        ]
    )

    start = time.time()
    sol_single = solve_ivp(
        SEIR_single, (t.min(), t.max()), var_dict["M0"], t_eval=t, args=(var_dict,), method="Radau"
    )
    end = time.time()
    print(f"Single SEIR simulation: t - {end - start}")

    var_dict["beta"] = beta_max_inflow
    var_dict["N_bar"] = final_pop[max_pop_inflow]
    var_dict["lamda"] = var_dict["mu"] * var_dict["N_bar"]
    var_dict["S0"] = var_dict["N_bar"] - (
        var_dict["E0"] + var_dict["Ia0"] + var_dict["Is0"] + var_dict["Iw0"] + var_dict["R0"]
    )
    var_dict["M0"] = np.array(
        [
            var_dict["S0"],
            var_dict["E0"],
            var_dict["Ia0"],
            var_dict["Is0"],
            var_dict["Iw0"],
            var_dict["R0"],
        ]
    )
    start = time.time()
    sol_single_max = solve_ivp(
        SEIR_single, (t.min(), t.max()), var_dict["M0"], t_eval=t, args=(var_dict,), method="Radau"
    )
    end = time.time()
    print(f"Single SEIR simulation: t - {end - start}")

    json.dump(
        {"sol": sol_single.y.tolist(), "sol_beta_max_inflow": sol_single_max.y.tolist()},
        open(
            prepare_output(OUTPUT_ROOT / "final_pop_exp" / f"single_forward_simulation_n_{n}.json"),
            "w",
        ),
        indent=2,
    )
    return


def main():
    exp = "f"
    if exp == "n":
        check_varying_n(low=10, high=50, step=10)
    elif exp == "t":
        check_varying_time_full_n()
    elif exp == "c":
        compare_single_sir_model()
    elif exp == "f":
        final_experiment()
    return


if __name__ == "__main__":
    main()
