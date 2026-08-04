# Mathematical and numerical methods

This document specifies the conventions implemented by the
`time_residence` package. Equation numbers refer to the associated article.

## Notation

Let $n$ denote the number of AGEBs in a model input. The same ordered AGEB
identifier array indexes every matrix row, matrix column, mobility fraction,
and population value.

- $N_i$: census population resident in AGEB $i$.
- $\alpha_i$: fraction of sampled residents of $i$ observed at least once
  outside $i$.
- $P=(p_{ij})$: residence-occupation matrix for residents who leave their home
  AGEB.
- $P^*=(p^*_{ij})$: effective residence-occupation matrix for the complete
  population.

Rows represent residence AGEBs and columns represent destination AGEBs. Thus
$p_{ij}$ is the fraction of time that residents of $i$ spend in $j$.

## Local time and residence assignment

Input timestamps are parsed as UTC and converted to `America/Hermosillo`,
which remains at UTC−07:00 during the study period. Period selection uses
inclusive local calendar dates. Nighttime observations satisfy

$$
22{:}00 \leq t < 24{:}00
\quad\text{or}\quad
00{:}00 \leq t < 06{:}00.
$$

For a device, let $M_{\mathrm{all}}$ and $M_{\mathrm{night}}$ be the sets of
modal AGEB indices in the complete and nighttime records. Residence candidates
are selected in this order:

1. $M_{\mathrm{all}}\cap M_{\mathrm{night}}$, when the intersection is
   nonempty;
2. $M_{\mathrm{night}}$, when it is nonempty; or
3. $M_{\mathrm{all}}$.

If several candidates remain, AGEB $i$ is selected with probability
proportional to $N_i$. The random seed is derived from the device identifier
and a user-supplied base seed.

Pings with `ageb_index = -1` are spatially unassigned and do not count as
observations outside the residence AGEB.

## Brownian-bridge occupation density

Consider consecutive projected observations $(z_k,t_k)$ and
$(z_{k+1},t_{k+1})$, where $z=(x,y)$,
$T_k=t_{k+1}-t_k$, and $u=(t-t_k)/T_k$. The conditional location density is
bivariate normal with mean

$$
\mu_k(t)=z_k+u(z_{k+1}-z_k)
$$

and isotropic variance

$$
v_k(t)=T_k u(1-u)\sigma^2+
\left[(1-u)^2+u^2\right]\delta^2.
$$

Here $\sigma$ is the Brownian standard-deviation rate and $\delta$ is the
location-error standard deviation. With projected coordinates in metres and
time in seconds, $\sigma$ has units $\mathrm{m}/\sqrt{\mathrm{s}}$ and
$\delta$ has units $\mathrm{m}$.

### Parameter likelihood

For the observed increments $X_k=Z(t_{k+1})-Z(t_k)$, Equation (9) gives a
zero-mean Gaussian likelihood with tridiagonal covariance

$$
[\Sigma_X]_{kk}=\Delta t_k\sigma^2+2\delta^2,
\qquad
[\Sigma_X]_{k,k+1}=[\Sigma_X]_{k+1,k}=-\delta^2.
$$

The likelihood is evaluated with a banded Cholesky factorization. The command
line supports either estimation of $\sigma$ for fixed $\delta$ or joint
estimation of both positive parameters in log space. The fixed default is
$\delta=28.85\ \mathrm{m}$, the value used by the study scripts. The article
does not report a numerical value for $\delta$.

### Occupation integral

For polygon $A_j$, the device-level occupation fraction is

$$
q_j=\frac{1}{T}
\sum_k
\int_{t_k}^{t_{k+1}}
\int_{A_j}
\phi_2\!\left(z;\mu_k(t),v_k(t)I\right)\,dz\,dt,
\qquad
T=t_m-t_1.
$$

The spatial and temporal integrals use uniform Monte Carlo samples. Spatial
points are drawn from each polygon's bounding box and evaluated with a
point-in-polygon indicator. The default is 1,000 paired space-time samples per
consecutive-ping interval, matching Section 5.2 of the article. Each device
receives a random stream derived from its identifier and the base seed.

The resulting vector is required to be finite and nonnegative, then normalized
over the AGEBs in the model input. The occupation estimator uses the pairwise
Brownian-bridge density in Equations (3)–(4). Equation (8), which conditions on
all observations, is not evaluated by this estimator.

## Population mobility parameters

For residence AGEB $i$, let $\mathcal{R}_i$ be the sampled residents and let
$\mathcal{L}_i\subseteq\mathcal{R}_i$ be those observed at least once outside
their home AGEB. Then

$$
\alpha_i=\frac{|\mathcal{L}_i|}{|\mathcal{R}_i|},
\qquad
p_{ij}=\frac{1}{|\mathcal{L}_i|}
\sum_{d\in\mathcal{L}_i}q_{dj}.
$$

If $\mathcal{L}_i$ is empty, row $i$ of $P$ is the identity row. In that case
$\alpha_i=0$, so the row does not contribute to away-from-home occupation.
The effective matrix is

$$
P^*=\operatorname{diag}(\alpha)P+
\operatorname{diag}(1-\alpha),
$$

or, entrywise,

$$
p^*_{ij}=\alpha_i p_{ij}+(1-\alpha_i)\mathbf{1}\{i=j\}.
$$

Both $P$ and $P^*$ are row-stochastic.

## Multi-patch SEIRS model

Let $S_i$, $E_i$, $I_i$, and $R_i$ denote susceptible, exposed, infected, and
recovered residents of AGEB $i$. The total and infected populations present in
destination $j$ are

$$
\widetilde N_j=\sum_i p^*_{ij}N_i,
\qquad
\widetilde I_j=\sum_i p^*_{ij}I_i.
$$

For homogeneous contact rate $\beta$, incidence among residents of $i$ is

$$
\mathcal{F}_i=
S_i\sum_j p^*_{ij}\beta
\frac{\widetilde I_j}{\widetilde N_j}.
$$

The differential equations are

$$
\begin{aligned}
\dot S_i &= \mu N_i-\mathcal F_i-\mu S_i+\tau R_i,\\
\dot E_i &= \mathcal F_i-(\kappa+\mu)E_i,\\
\dot I_i &= \kappa E_i-(\gamma+\psi+\mu)I_i,\\
\dot R_i &= \gamma I_i-(\tau+\mu)R_i.
\end{aligned}
$$

The Figure 8 script also defines a cumulative-incidence state $Y$ with
$\dot Y_i=\kappa E_i$. Since $Y$ does not enter the equations above, the
package solver integrates only $(S,E,I,R)$.

### Default epidemic parameters

All rates are per day.

| Symbol | Default | Meaning | Source |
|---|---:|---|---|
| $\beta$ | $1.5$ | contact/transmission rate | Article, Section 5.3.3 |
| $\kappa$ | $1/7$ | exposed-to-infected rate | Article, Section 5.3.3 |
| $\gamma$ | $1/14$ | recovery rate | Article, Section 5.3.3 |
| $\mu$ | $0.06/(1000\times365)$ | natural mortality and per-capita recruitment rate | Article, Section 5.3.3 |
| $\tau$ | $1/180$ | loss-of-immunity rate | Article, Section 5.3.3 |
| $\psi$ | $0.0003$ | disease-induced mortality rate | Figure 8 script; numerical value absent from the article |

Recruitment is $\Lambda_i=\mu N_i$.

### Initial conditions and period comparisons

The default seed AGEB codes are 2956, 3367, 5734, and 6200. At each seed,

$$
E_i(0)=I_i(0)=1.
$$

All other exposed and infected counts, and all recovered counts, start at zero.
Susceptible counts satisfy

$$
S_i(0)=N_i-E_i(0)-I_i(0)-R_i(0).
$$

For comparison $j\in\{1,2,3\}$, calculations use the AGEB identifiers common
to both period-parts. The local and citywide Figure 8 quantities are

$$
D_i^{(j)}(t)=I_i^{(\mathrm{P}j\mathrm{A})}(t)
-I_i^{(\mathrm{P}j\mathrm{B})}(t),
$$

$$
D^{(j)}(t)=\sum_{i\in\text{common AGEBs}}D_i^{(j)}(t).
$$

The default time grid has 100 points on $[0,200]$. The statistic reported near
day 30 uses the closest grid point, day 30.303.
