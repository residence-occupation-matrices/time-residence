# Mathematical and numerical methods

This document states the conventions implemented by the `time_residence`
package. Equation numbers refer to the associated article.

## Notation

Let (n) be the number of retained AGEBs. The same ordered AGEB identifier
array is attached to every matrix row, matrix column, mobility fraction, and
population value.

- (N_i): census population resident in AGEB (i).
- (alpha_i): fraction of sampled residents of (i) with at least one valid
  observed ping outside (i).
- (P=(p_{ij})): residence-occupation matrix conditioned on residents who
  leave their home AGEB.
- (P^*=(p^*_{ij})): effective residence-occupation matrix for the complete
  population.

The implementation uses row-origin convention: row (i) contains the
fractions of time spent in destinations (j=1,\ldots,n) by residents of (i).

## Local-time periods and residence assignment

Input timestamps are parsed as UTC and converted to `America/Hermosillo`,
which remains at UTC−07:00 during the study dates. Periods are selected by
inclusive local calendar date. Nighttime observations satisfy

\[
22{:}00 \leq t < 24{:}00
\quad\text{or}\quad
00{:}00 \leq t < 06{:}00.
\]

For a device, let (M_{\mathrm{all}}) and (M_{\mathrm{night}}) be the sets of
modal observed AGEB indices in all records and nighttime records,
respectively. Residence candidates are selected as follows:

1. use (M_{\mathrm{all}}\cap M_{\mathrm{night}}) when it is nonempty;
2. otherwise use (M_{\mathrm{night}}), if nonempty;
3. otherwise use (M_{\mathrm{all}}).

If more than one candidate remains, AGEB (i) is sampled with probability
proportional to (N_i). The random generator is derived from the device
identifier and a user-supplied base seed. Reordering records or changing worker
scheduling therefore does not change the assignment.

Pings not assigned to a valid AGEB have `ageb_index = -1`. They do not count as
evidence of visiting another AGEB in the estimate of (alpha).

## Brownian-bridge occupation density

For one device, consecutive projected observations are
((z_k,t_k)) and ((z_{k+1},t_{k+1})), where (z=(x,y)),
(T_k=t_{k+1}-t_k), and (u=(t-t_k)/T_k). The conditional location used in
the occupation integral is bivariate normal with mean

\[
\mu_k(t)=z_k+u(z_{k+1}-z_k)
\]

and isotropic variance

\[
v_k(t)=T_k u(1-u)\sigma^2+
\left[(1-u)^2+u^2\right]\delta^2.
\]

Here (sigma) is the Brownian-motion standard-deviation parameter and
(delta) is the location-error standard deviation. Both are in metres when
time is measured in seconds and the trajectory is represented in a metric
projected coordinate system. In particular, the covariance contains
(sigma^2); an additional square root must not be applied after estimating
(sigma).

### Parameter likelihood

For observed increments
(X_k=Z(t_{k+1})-Z(t_k)), Equation (9) gives a zero-mean Gaussian likelihood
with tridiagonal covariance

\[
[\Sigma_X]_{kk}=\Delta t_k\sigma^2+2\delta^2,
\qquad
[\Sigma_X]_{k,k+1}=[\Sigma_X]_{k+1,k}=-\delta^2.
\]

The implementation evaluates this likelihood with banded Cholesky
factorization. It can either estimate (sigma) for a fixed (delta), or
jointly estimate both positive standard deviations in log space. The fixed
default (delta=28.85\) m comes from the research code; the article does not
report a numerical value for this parameter. The command-line flag
`--estimate-location-error` selects the joint formulation.

### Occupation integral

For polygon (A_j), the device-level occupation fraction is

\[
q_j=\frac{1}{T}
\sum_{k}
\int_{t_k}^{t_{k+1}}
\int_{A_j}
\phi_2\!\left(z;\mu_k(t),v_k(t)I\right)\,dz\,dt,
\qquad
T=t_m-t_1.
\]

The spatial and temporal integrals are approximated by uniform Monte Carlo
sampling. Spatial samples are drawn from each polygon's bounding box and
weighted by a point-in-polygon indicator. The default is 1,000 paired
space-time samples per consecutive-ping interval, matching the count reported
in Section 5.2. Device-specific random streams make the result reproducible.
The resulting vector is checked for finite, nonnegative values and normalized
to sum to one over retained AGEBs.

The reusable implementation uses the pairwise Brownian-bridge density in
Equations (3)–(4), as does the computational research script. The article also
derives the full Brownian-motion-with-measurement-error conditional density in
Equation (8). That all-observation conditional density is not substituted
silently for the pairwise estimator here; the distinction is recorded in
`REPRODUCIBILITY.md`.

## Population-level mobility parameters

For residence AGEB (i), let (mathcal R_i) be all sampled residents and
(mathcal L_i\subseteq\mathcal R_i) those with at least one valid ping outside
their home AGEB. Then

\[
\alpha_i=\frac{|\mathcal L_i|}{|\mathcal R_i|},
\qquad
p_{ij}=\frac{1}{|\mathcal L_i|}
\sum_{d\in\mathcal L_i}q_{dj}.
\]

When (mathcal L_i) is empty, row (i) of (P) is set to the identity row.
That row is immaterial because (alpha_i=0), but the convention keeps (P)
row-stochastic. The effective matrix is

\[
P^*=\operatorname{diag}(\alpha)P+
\operatorname{diag}(1-\alpha),
\]

or, entrywise,

\[
p^*_{ij}=\alpha_i p_{ij}+(1-\alpha_i)\mathbf 1\{i=j\}.
\]

Every row of (P) and (P^*) sums to one.

## Multi-patch SEIRS model

Let (S_i,E_i,I_i,R_i) denote susceptible, exposed, infected, and recovered
residents of patch (i). The population present in destination (j) and the
infected population present there are

\[
\widetilde N_j=\sum_i p^*_{ij}N_i,
\qquad
\widetilde I_j=\sum_i p^*_{ij}I_i.
\]

With a homogeneous contact rate (eta), the incidence among residents of
patch (i) is evaluated without constructing or inverting dense diagonal
matrices:

\[
\mathcal F_i=
S_i\sum_j p^*_{ij}\beta\frac{\widetilde I_j}{\widetilde N_j}.
\]

The implemented system is algebraically equivalent to model (2):

\[
\begin{aligned}
\dot S_i &= \mu N_i-\mathcal F_i-\mu S_i+\tau R_i,\\
\dot E_i &= \mathcal F_i-(\kappa+\mu)E_i,\\
\dot I_i &= \kappa E_i-(\gamma+\psi+\mu)I_i,\\
\dot R_i &= \gamma I_i-(\tau+\mu)R_i.
\end{aligned}
\]

The cumulative-incidence variable `Y` in the contributed figure script obeys
(\dot Y_i=\kappa E_i) but does not feed back into these four equations. It is
therefore omitted from the reusable solver; this leaves every (S,E,I,R)
trajectory unchanged and avoids the original nonzero bookkeeping initial
condition.

### Default epidemic parameters

All rates are per day.

| Symbol | Default | Meaning | Source |
|---|---:|---|---|
| (eta) | 1.5 | contact/transmission rate | Article, Section 5.3.3 |
| (kappa) | (1/7) | exposed-to-infected rate | Article, Section 5.3.3 |
| (gamma) | (1/14) | recovery rate | Article, Section 5.3.3 |
| (mu) | (0.06/(1000\times365)) | natural mortality and per-capita recruitment rate | Article, Section 5.3.3 |
| (	au) | (1/180) | loss-of-immunity rate | Article, Section 5.3.3 |
| (psi) | 0.0003 | disease-induced mortality rate | `epidemic_simulation_code.py`; no numerical value appears in the article |

Recruitment is (Lambda_i=mu N_i), giving an almost constant population in
the absence of disease-induced mortality.

### Initial conditions and comparisons

The default seed AGEB codes are 2956, 3367, 5734, and 6200. For each seed
present in a period-part,

\[
E_i(0)=I_i(0)=1,
\]

while all other exposed and infected counts and every recovered count start at
zero. Susceptible counts satisfy
(S_i(0)=N_i-E_i(0)-I_i(0)-R_i(0)).

For period (j\in\{1,2,3\}), only AGEB identifiers present in both parts are
compared. The local and global Figure 8 quantities are

\[
D_i^{(j)}(t)=I_i^{(PjA)}(t)-I_i^{(PjB)}(t),
\qquad
D^{(j)}(t)=\sum_{i\in\text{common AGEBs}}D_i^{(j)}(t).
\]

With 100 points on ([0,200]), array index 15 is day 30.303. This is the point
near (t=30) used for the reported proportions of negative AGEB differences;
it is not the midpoint of the simulation interval.
