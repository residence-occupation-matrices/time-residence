#!/usr/bin/env python3
# ruff: noqa: E741
"""Generate the Figure 8 SEIRS comparisons from period-specific ROM inputs."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from odeintw import odeintw
from project_paths import DATA_ROOT, EPIDEMIOLOGY_CSV, OUTPUT_ROOT, prepare_output

rng = np.random.default_rng()


def SEIR_system(M, t, var_dict, pstar):
    """Evaluate the multi-patch SEIRS system used for Figure 8."""
    S, E, I, Y, R = M
    S, E, I, Y, R = (
        S.reshape(-1, 1),
        E.reshape(-1, 1),
        I.reshape(-1, 1),
        Y.reshape(-1, 1),
        R.reshape(-1, 1),
    )
    N_tilde = pstar.T @ var_dict["N_bar"]
    beta, gamma, kappa, lamda, mu, phi, tau = (
        var_dict["beta"].reshape(-1, 1),
        var_dict["gamma"].reshape(-1, 1),
        var_dict["kappa"].reshape(-1, 1),
        var_dict["lamda"].reshape(-1, 1),
        var_dict["mu"].reshape(-1, 1),
        var_dict["phi"].reshape(-1, 1),
        var_dict["tau"].reshape(-1, 1),
    )

    dSdt = (
        lamda
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

    dIdt = (
        np.diag(
            kappa.reshape(
                -1,
            )
        )
        @ E
        - np.diag(
            (gamma + phi + mu).reshape(
                -1,
            )
        )
        @ I
    )

    dYdt = (
        np.diag(
            kappa.reshape(
                -1,
            )
        )
        @ E
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

    dSdt, dEdt, dIdt, dYdt, dRdt = (
        dSdt.reshape(
            -1,
        ),
        dEdt.reshape(
            -1,
        ),
        dIdt.reshape(
            -1,
        ),
        dYdt.reshape(
            -1,
        ),
        dRdt.reshape(
            -1,
        ),
    )
    return np.array([dSdt, dEdt, dIdt, dYdt, dRdt])


# Period 1. P1A uses the SF inputs because P1A and P2A share the same dates.
residence_matrix_FF = np.load(
    DATA_ROOT / "working_avg_res_mat" / "working_avg_resmat_Second_First.npy"
)
residence_matrix_FS = np.load(
    DATA_ROOT / "working_avg_res_mat" / "working_avg_resmat_First_Second.npy"
)
n_FF = residence_matrix_FF.shape[0]
n_FS = residence_matrix_FS.shape[0]

dat_one_alpha_FF = pd.read_csv(DATA_ROOT / "one_alpha_POBTOT" / "alphas_SF.csv")
dat_one_alpha_FS = pd.read_csv(DATA_ROOT / "one_alpha_POBTOT" / "alphas_FS.csv")

working_agebs_FF = np.where(dat_one_alpha_FF["proporcion"].notna())[0]
working_agebs_FS = np.where(dat_one_alpha_FS["proporcion"].notna())[0]


common_CVE_AGEB_FF = dat_one_alpha_FF["CVE_AGEB"][working_agebs_FF].reset_index()
common_CVE_AGEB_FF["New_index_FF"] = range(0, 0 + len(working_agebs_FF))

common_CVE_AGEB_FS = dat_one_alpha_FS["CVE_AGEB"][working_agebs_FS].reset_index()
common_CVE_AGEB_FS["New_index_FS"] = range(0, 0 + len(working_agebs_FS))


common_CVE_AGEB_indices = pd.merge(
    common_CVE_AGEB_FF, common_CVE_AGEB_FS, how="inner", on="CVE_AGEB"
)

list_FF_FS = ["1746", "1799", "1835", "1869", "1837", "1943", "1958"]

print(common_CVE_AGEB_indices[common_CVE_AGEB_indices["CVE_AGEB"].isin(list_FF_FS)])


mod_dat_one_alphas_FF = dat_one_alpha_FF.dropna().reset_index(drop=True)

mod_dat_one_alphas_FS = dat_one_alpha_FS.dropna().reset_index(drop=True)


alpha_FF = np.ones(n_FF) - np.array(mod_dat_one_alphas_FF["proporcion"])
alpha_FS = np.ones(n_FS) - np.array(mod_dat_one_alphas_FS["proporcion"])

N_bar_FF = np.array(mod_dat_one_alphas_FF["POBTOT"]).reshape(n_FF, 1)

N_bar_FS = np.array(mod_dat_one_alphas_FS["POBTOT"]).reshape(n_FS, 1)


df_4zones = pd.read_csv(EPIDEMIOLOGY_CSV)

df_4zones["CLASCOVID19"] = np.where(
    (df_4zones["CLASCOVID19"] == "SOSPECHOSO") & (df_4zones["En base Sonora"] == "verdadero"),
    "CONF LAB",
    df_4zones["CLASCOVID19"],
)

date_init_case = df_4zones.loc[
    (df_4zones["CLASCOVID19"] == "CONF LAB")
    & (df_4zones["En base Sonora"] == "verdadero")
    & (df_4zones["LOCRESI"] == "HERMOSILLO"),
    "FECINISI",
].min()

CVE_AGEB_init = (
    df_4zones.loc[
        (df_4zones["FECINISI"] == date_init_case)
        & (df_4zones["CLASCOVID19"] == "CONF LAB")
        & (df_4zones["En base Sonora"] == "verdadero")
        & (df_4zones["LOCRESI"] == "HERMOSILLO"),
        "CVE_AGEB",
    ]
    .value_counts()
    .to_frame("counts")
    .reset_index()
)
CVE_AGEB_init = CVE_AGEB_init.rename(columns={"index": "CVE_AGEB"})
CVE_AGEB_array = np.array(CVE_AGEB_init["CVE_AGEB"]).astype(str)

AGEB_indices_FF = np.where(mod_dat_one_alphas_FF["CVE_AGEB"].isin(CVE_AGEB_array))[0]
AGEB_indices_FS = np.where(mod_dat_one_alphas_FS["CVE_AGEB"].isin(CVE_AGEB_array))[0]


E0_FF = np.zeros((n_FF, 1))
E0_FF[AGEB_indices_FF] = 1

E0_FS = np.zeros((n_FS, 1))
E0_FS[AGEB_indices_FS] = 1

I0_FF = np.zeros((n_FF, 1))
I0_FF[AGEB_indices_FF] = 1

I0_FS = np.zeros((n_FS, 1))
I0_FS[AGEB_indices_FS] = 1


def initialise_variables_FF(n):
    var_dict = {}
    var_dict["beta"] = np.ones((n, 1)) * 1.5
    var_dict["gamma"] = np.ones((n, 1)) * 1 / 14
    var_dict["kappa"] = np.ones((n, 1)) * 1 / 7
    var_dict["N_bar"] = N_bar_FF
    var_dict["mu"] = np.ones((n, 1)) * 0.06 / (1000 * 365)
    var_dict["lamda"] = var_dict["N_bar"] * var_dict["mu"]
    var_dict["phi"] = np.ones((n, 1)) * 0.0003
    var_dict["tau"] = np.ones((n, 1)) * 1 / 180
    var_dict["E0"] = E0_FF
    var_dict["I0"] = I0_FF
    var_dict["Y0"] = np.zeros((n, 1))
    var_dict["R0"] = np.zeros((n, 1))
    var_dict["S0"] = var_dict["N_bar"] - (var_dict["E0"] + var_dict["I0"] + var_dict["R0"])
    var_dict["M0"] = np.array(
        [var_dict["S0"], var_dict["E0"], var_dict["I0"], var_dict["Y0"], var_dict["R0"]]
    )

    return var_dict


def initialise_variables_FS(n):
    var_dict = {}
    var_dict["beta"] = np.ones((n, 1)) * 1.5
    var_dict["gamma"] = np.ones((n, 1)) * 1 / 14
    var_dict["kappa"] = np.ones((n, 1)) * 1 / 7
    var_dict["N_bar"] = N_bar_FS
    var_dict["mu"] = np.ones((n, 1)) * 0.06 / (1000 * 365)
    var_dict["lamda"] = var_dict["N_bar"] * var_dict["mu"]
    var_dict["phi"] = np.ones((n, 1)) * 0.0003
    var_dict["tau"] = np.ones((n, 1)) * 1 / 180
    var_dict["E0"] = E0_FS
    var_dict["I0"] = I0_FS
    var_dict["Y0"] = np.zeros((n, 1))
    var_dict["R0"] = np.zeros((n, 1))
    var_dict["S0"] = var_dict["N_bar"] - (var_dict["E0"] + var_dict["I0"] + var_dict["R0"])
    var_dict["M0"] = np.array(
        [var_dict["S0"], var_dict["E0"], var_dict["I0"], var_dict["Y0"], var_dict["R0"]]
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


t = np.linspace(0, 200, 100)
pstar_FF = generate_actual_pstar(residence_matrix_FF, alpha_FF)
pstar_FS = generate_actual_pstar(residence_matrix_FS, alpha_FS)

var_dict_FF = initialise_variables_FF(n_FF)

var_dict_FS = initialise_variables_FS(n_FS)

# Solve the Period 1 models.

sol_FF = odeintw(SEIR_system, var_dict_FF["M0"], t, args=(var_dict_FF, pstar_FF))

sol_FS = odeintw(SEIR_system, var_dict_FS["M0"], t, args=(var_dict_FS, pstar_FS))

# Figure 8A: AGEB-level infection differences.

I0_FF = sol_FF[:, 2, :, 0]

I0_FS = sol_FS[:, 2, :, 0]

commonI0_FF = I0_FF[:, np.array(common_CVE_AGEB_indices["New_index_FF"])]

commonI0_FS = I0_FS[:, np.array(common_CVE_AGEB_indices["New_index_FS"])]


fig, ax = plt.subplots()
ax.plot(t, np.subtract(commonI0_FF, commonI0_FS))
plt.ylabel("Difference")
plt.xlabel("time")
plt.grid(True)
fig.savefig(
    prepare_output(OUTPUT_ROOT / "difference_FP_FP_and_FP_SP_used_SF_resmat.png"),
    format="png",
    dpi=300,
)
plt.close(fig)

# Fraction of AGEB differences below zero at day 30.303 (index 15).

dif_FF_FS = np.subtract(commonI0_FF, commonI0_FS)

np.count_nonzero(dif_FF_FS[15,] < 0.0) / np.shape(dif_FF_FS)[1]

# Figure 8B: citywide infection difference.
sum_commonI0_FF = np.sum(commonI0_FF, axis=1)

sum_commonI0_FS = np.sum(commonI0_FS, axis=1)

fig, ax = plt.subplots()
ax.plot(t, np.subtract(sum_commonI0_FF, sum_commonI0_FS))
plt.ylabel("Difference")
plt.xlabel("time")
plt.grid(True)
fig.savefig(
    prepare_output(OUTPUT_ROOT / "difference_global_FP_FP_and_FP_SP_used_SF_resmat.png"),
    format="png",
    dpi=300,
)
plt.close(fig)

# Period 2.
residence_matrix_SF = np.load(
    DATA_ROOT / "working_avg_res_mat" / "working_avg_resmat_Second_First.npy"
)
residence_matrix_SS = np.load(
    DATA_ROOT / "working_avg_res_mat" / "working_avg_resmat_Second_Second.npy"
)
n_SF = residence_matrix_SF.shape[0]
n_SS = residence_matrix_SS.shape[0]

dat_one_alpha_SF = pd.read_csv(DATA_ROOT / "one_alpha_POBTOT" / "alphas_SF.csv")
dat_one_alpha_SS = pd.read_csv(DATA_ROOT / "one_alpha_POBTOT" / "alphas_SS.csv")

working_agebs_SF = np.where(dat_one_alpha_SF["proporcion"].notna())[0]
working_agebs_SS = np.where(dat_one_alpha_SS["proporcion"].notna())[0]


common_CVE_AGEB_SF = dat_one_alpha_SF["CVE_AGEB"][working_agebs_SF].reset_index()
common_CVE_AGEB_SF["New_index_SF"] = range(0, 0 + len(working_agebs_SF))

common_CVE_AGEB_SS = dat_one_alpha_SS["CVE_AGEB"][working_agebs_SS].reset_index()
common_CVE_AGEB_SS["New_index_SS"] = range(0, 0 + len(working_agebs_SS))


common_CVE_AGEB_indices_SF_SS = pd.merge(
    common_CVE_AGEB_SF, common_CVE_AGEB_SS, how="inner", on="CVE_AGEB"
)


list_SF_SS = ["593", "1144", "1835", "1958", "323A", "330A", "3314", "490A", "565A", "572A", "6953"]

print(common_CVE_AGEB_indices_SF_SS[common_CVE_AGEB_indices_SF_SS["CVE_AGEB"].isin(list_SF_SS)])


mod_dat_one_alphas_SF = dat_one_alpha_SF.dropna().reset_index(drop=True)

mod_dat_one_alphas_SS = dat_one_alpha_SS.dropna().reset_index(drop=True)


alpha_SF = np.ones(n_SF) - np.array(mod_dat_one_alphas_SF["proporcion"])
alpha_SS = np.ones(n_SS) - np.array(mod_dat_one_alphas_SS["proporcion"])

N_bar_SF = np.array(mod_dat_one_alphas_SF["POBTOT"]).reshape(n_SF, 1)

N_bar_SS = np.array(mod_dat_one_alphas_SS["POBTOT"]).reshape(n_SS, 1)


AGEB_indices_SF = np.where(mod_dat_one_alphas_SF["CVE_AGEB"].isin(CVE_AGEB_array))[0]
AGEB_indices_SS = np.where(mod_dat_one_alphas_SS["CVE_AGEB"].isin(CVE_AGEB_array))[0]


E0_SF = np.zeros((n_SF, 1))
E0_SF[AGEB_indices_SF] = 1

E0_SS = np.zeros((n_SS, 1))
E0_SS[AGEB_indices_SS] = 1

I0_SF = np.zeros((n_SF, 1))
I0_SF[AGEB_indices_SF] = 1

I0_SS = np.zeros((n_SS, 1))
I0_SS[AGEB_indices_SS] = 1


def initialise_variables_SF(n):
    var_dict = {}
    var_dict["beta"] = np.ones((n, 1)) * 1.5
    var_dict["gamma"] = np.ones((n, 1)) * 1 / 14
    var_dict["kappa"] = np.ones((n, 1)) * 1 / 7
    var_dict["N_bar"] = N_bar_SF
    var_dict["mu"] = np.ones((n, 1)) * 0.06 / (1000 * 365)
    var_dict["lamda"] = var_dict["N_bar"] * var_dict["mu"]
    var_dict["phi"] = np.ones((n, 1)) * 0.0003
    var_dict["tau"] = np.ones((n, 1)) * 1 / 180
    var_dict["E0"] = E0_SF
    var_dict["I0"] = I0_SF
    var_dict["Y0"] = np.zeros((n, 1))
    var_dict["R0"] = np.zeros((n, 1))
    var_dict["S0"] = var_dict["N_bar"] - (var_dict["E0"] + var_dict["I0"] + var_dict["R0"])
    var_dict["M0"] = np.array(
        [var_dict["S0"], var_dict["E0"], var_dict["I0"], var_dict["Y0"], var_dict["R0"]]
    )

    return var_dict


def initialise_variables_SS(n):
    var_dict = {}
    var_dict["beta"] = np.ones((n, 1)) * 1.5
    var_dict["gamma"] = np.ones((n, 1)) * 1 / 14
    var_dict["kappa"] = np.ones((n, 1)) * 1 / 7
    var_dict["N_bar"] = N_bar_SS
    var_dict["mu"] = np.ones((n, 1)) * 0.06 / (1000 * 365)
    var_dict["lamda"] = var_dict["N_bar"] * var_dict["mu"]
    var_dict["phi"] = np.ones((n, 1)) * 0.0003
    var_dict["tau"] = np.ones((n, 1)) * 1 / 180
    var_dict["E0"] = E0_SS
    var_dict["I0"] = I0_SS
    var_dict["Y0"] = np.zeros((n, 1))
    var_dict["R0"] = np.zeros((n, 1))
    var_dict["S0"] = var_dict["N_bar"] - (var_dict["E0"] + var_dict["I0"] + var_dict["R0"])
    var_dict["M0"] = np.array(
        [var_dict["S0"], var_dict["E0"], var_dict["I0"], var_dict["Y0"], var_dict["R0"]]
    )

    return var_dict


pstar_SF = generate_actual_pstar(residence_matrix_SF, alpha_SF)
pstar_SS = generate_actual_pstar(residence_matrix_SS, alpha_SS)

var_dict_SF = initialise_variables_SF(n_SF)

var_dict_SS = initialise_variables_SS(n_SS)

# Solve the Period 2 models.

sol_SF = odeintw(SEIR_system, var_dict_SF["M0"], t, args=(var_dict_SF, pstar_SF))

sol_SS = odeintw(SEIR_system, var_dict_SS["M0"], t, args=(var_dict_SS, pstar_SS))

# Figure 8C: AGEB-level infection differences.

I0_SF = sol_SF[:, 2, :, 0]

I0_SS = sol_SS[:, 2, :, 0]

commonI0_SF = I0_SF[:, np.array(common_CVE_AGEB_indices_SF_SS["New_index_SF"])]

commonI0_SS = I0_SS[:, np.array(common_CVE_AGEB_indices_SF_SS["New_index_SS"])]

fig, ax = plt.subplots()
ax.plot(t, np.subtract(commonI0_SF, commonI0_SS))
plt.ylabel("Difference")
plt.xlabel("time")
plt.grid(True)
fig.savefig(
    prepare_output(OUTPUT_ROOT / "difference_SP_FP_and_SP_SP.png"),
    format="png",
    dpi=300,
)
plt.close(fig)

# Fraction of AGEB differences below zero at day 30.303 (index 15).

dif_SF_SS = np.subtract(commonI0_SF, commonI0_SS)

np.count_nonzero(dif_SF_SS[15,] < 0.0) / np.shape(dif_SF_SS)[1]

# Figure 8D: citywide infection difference.
sum_commonI0_SF = np.sum(commonI0_SF, axis=1)

sum_commonI0_SS = np.sum(commonI0_SS, axis=1)

fig, ax = plt.subplots()
ax.plot(t, np.subtract(sum_commonI0_SF, sum_commonI0_SS))
plt.ylabel("Difference")
plt.grid(True)
plt.xlabel("time")
fig.savefig(
    prepare_output(OUTPUT_ROOT / "difference_global_SP_FP_and_SP_SP.png"),
    format="png",
    dpi=300,
)
plt.close(fig)

# Period 3.
residence_matrix_TF = np.load(
    DATA_ROOT / "working_avg_res_mat" / "working_avg_resmat_Third_First.npy"
)
residence_matrix_TS = np.load(
    DATA_ROOT / "working_avg_res_mat" / "working_avg_resmat_Third_Second.npy"
)
n_TF = residence_matrix_TF.shape[0]
n_TS = residence_matrix_TS.shape[0]

dat_one_alpha_TF = pd.read_csv(DATA_ROOT / "one_alpha_POBTOT" / "alphas_TF.csv")
dat_one_alpha_TS = pd.read_csv(DATA_ROOT / "one_alpha_POBTOT" / "alphas_TS.csv")

working_agebs_TF = np.where(dat_one_alpha_TF["proporcion"].notna())[0]
working_agebs_TS = np.where(dat_one_alpha_TS["proporcion"].notna())[0]


common_CVE_AGEB_TF = dat_one_alpha_TF["CVE_AGEB"][working_agebs_TF].reset_index()
common_CVE_AGEB_TF["New_index_TF"] = range(0, 0 + len(working_agebs_TF))

common_CVE_AGEB_TS = dat_one_alpha_TS["CVE_AGEB"][working_agebs_TS].reset_index()
common_CVE_AGEB_TS["New_index_TS"] = range(0, 0 + len(working_agebs_TS))


common_CVE_AGEB_indices_TF_TS = pd.merge(
    common_CVE_AGEB_TF, common_CVE_AGEB_TS, how="inner", on="CVE_AGEB"
)


mod_dat_one_alphas_TF = dat_one_alpha_TF.dropna().reset_index(drop=True)

mod_dat_one_alphas_TS = dat_one_alpha_TS.dropna().reset_index(drop=True)


alpha_TF = np.ones(n_TF) - np.array(mod_dat_one_alphas_TF["proporcion"])
alpha_TS = np.ones(n_TS) - np.array(mod_dat_one_alphas_TS["proporcion"])

N_bar_TF = np.array(mod_dat_one_alphas_TF["POBTOT"]).reshape(n_TF, 1)

N_bar_TS = np.array(mod_dat_one_alphas_TS["POBTOT"]).reshape(n_TS, 1)


AGEB_indices_TF = np.where(mod_dat_one_alphas_TF["CVE_AGEB"].isin(CVE_AGEB_array))[0]
AGEB_indices_TS = np.where(mod_dat_one_alphas_TS["CVE_AGEB"].isin(CVE_AGEB_array))[0]


E0_TF = np.zeros((n_TF, 1))
E0_TF[AGEB_indices_TF] = 1

E0_TS = np.zeros((n_TS, 1))
E0_TS[AGEB_indices_TS] = 1

I0_TF = np.zeros((n_TF, 1))
I0_TF[AGEB_indices_TF] = 1

I0_TS = np.zeros((n_TS, 1))
I0_TS[AGEB_indices_TS] = 1


def initialise_variables_TF(n):
    var_dict = {}
    var_dict["beta"] = np.ones((n, 1)) * 1.5
    var_dict["gamma"] = np.ones((n, 1)) * 1 / 14
    var_dict["kappa"] = np.ones((n, 1)) * 1 / 7
    var_dict["N_bar"] = N_bar_TF
    var_dict["mu"] = np.ones((n, 1)) * 0.06 / (1000 * 365)
    var_dict["lamda"] = var_dict["N_bar"] * var_dict["mu"]
    var_dict["phi"] = np.ones((n, 1)) * 0.0003
    var_dict["tau"] = np.ones((n, 1)) * 1 / 180
    var_dict["E0"] = E0_TF
    var_dict["I0"] = I0_TF
    var_dict["Y0"] = np.zeros((n, 1))
    var_dict["R0"] = np.zeros((n, 1))
    var_dict["S0"] = var_dict["N_bar"] - (var_dict["E0"] + var_dict["I0"] + var_dict["R0"])
    var_dict["M0"] = np.array(
        [var_dict["S0"], var_dict["E0"], var_dict["I0"], var_dict["Y0"], var_dict["R0"]]
    )

    return var_dict


def initialise_variables_TS(n):
    var_dict = {}
    var_dict["beta"] = np.ones((n, 1)) * 1.5
    var_dict["gamma"] = np.ones((n, 1)) * 1 / 14
    var_dict["kappa"] = np.ones((n, 1)) * 1 / 7
    var_dict["N_bar"] = N_bar_TS
    var_dict["mu"] = np.ones((n, 1)) * 0.06 / (1000 * 365)
    var_dict["lamda"] = var_dict["N_bar"] * var_dict["mu"]
    var_dict["phi"] = np.ones((n, 1)) * 0.0003
    var_dict["tau"] = np.ones((n, 1)) * 1 / 180
    var_dict["E0"] = E0_TS
    var_dict["I0"] = I0_TS
    var_dict["Y0"] = np.zeros((n, 1))
    var_dict["R0"] = np.zeros((n, 1))
    var_dict["S0"] = var_dict["N_bar"] - (var_dict["E0"] + var_dict["I0"] + var_dict["R0"])
    var_dict["M0"] = np.array(
        [var_dict["S0"], var_dict["E0"], var_dict["I0"], var_dict["Y0"], var_dict["R0"]]
    )

    return var_dict


pstar_TF = generate_actual_pstar(residence_matrix_TF, alpha_TF)
pstar_TS = generate_actual_pstar(residence_matrix_TS, alpha_TS)

var_dict_TF = initialise_variables_TF(n_TF)

var_dict_TS = initialise_variables_TS(n_TS)

# Solve the Period 3 models.
sol_TF = odeintw(SEIR_system, var_dict_TF["M0"], t, args=(var_dict_TF, pstar_TF))

sol_TS = odeintw(SEIR_system, var_dict_TS["M0"], t, args=(var_dict_TS, pstar_TS))

# Figure 8E: AGEB-level infection differences.
I0_TF = sol_TF[:, 2, :, 0]

I0_TS = sol_TS[:, 2, :, 0]

commonI0_TF = I0_TF[:, np.array(common_CVE_AGEB_indices_TF_TS["New_index_TF"])]

commonI0_TS = I0_TS[:, np.array(common_CVE_AGEB_indices_TF_TS["New_index_TS"])]

fig, ax = plt.subplots()
ax.plot(t, np.subtract(commonI0_TF, commonI0_TS))
plt.ylabel("Difference")
plt.xlabel("time")
plt.grid(True)
fig.savefig(
    prepare_output(OUTPUT_ROOT / "difference_TP_FP_and_TP_SP.png"),
    format="png",
    dpi=300,
)
plt.close(fig)

# Fraction of AGEB differences below zero at day 30.303 (index 15).

dif_TF_TS = np.subtract(commonI0_TF, commonI0_TS)


np.count_nonzero(dif_TF_TS[15,] < 0.0) / np.shape(dif_TF_TS)[1]


# Figure 8F: citywide infection difference.
sum_commonI0_TF = np.sum(commonI0_TF, axis=1)

sum_commonI0_TS = np.sum(commonI0_TS, axis=1)

fig, ax = plt.subplots()
ax.plot(t, np.subtract(sum_commonI0_TF, sum_commonI0_TS))
plt.ylabel("Difference")
plt.xlabel("time")
plt.grid(True)
fig.savefig(
    prepare_output(OUTPUT_ROOT / "difference_global_TP_FP_and_TP_SP.png"),
    format="png",
    dpi=300,
)
plt.close(fig)
