# Reproducibility and implementation notes

The repository contains three forms of evidence that serve different purposes:

1. the published article, which defines the scientific method and reported
   results;
2. the scripts and notebooks under `research/`, which record the computations
   contributed to the article; and
3. the tested package under `src/`, which supplies portable interfaces,
   validation, deterministic random streams, and numerical output contracts.

All article scripts are retained together. The package consolidates repeated
operations; it does not erase the individual scripts or assign them a different
authorship status.

## Corrections applied

### Coordinate headers

`calculate-ageb.py` originally selected columns in the order
`id, timestamp, lat, lon, polygon` but supplied the output header
`id, timestamp, lon, lat, polygon`. The values were therefore stored under
interchanged coordinate labels. The script now writes matching headers. The
package reader also detects the known Hermosillo signature and requires an
explicit repair flag, preventing a silent guess.

### Period endpoints and time zone

The research script created UTC midnight bounds, used strict inequalities, and
therefore omitted most of the reported final date while also shifting the local
calendar window by seven hours. Period masks now convert UTC timestamps to
`America/Hermosillo` and include both local endpoint dates. The nighttime
interval is implemented as `[22:00, 06:00)`, rather than including the entire
06:00 hour.

### Portable paths

Workstation-specific paths from Linux, macOS, and remote Jupyter sessions were
removed. Package commands accept paths as arguments. The research scripts use
`research/scripts/project_paths.py`, repository-relative defaults, and optional
environment variables. Output parent directories are created before writing.

### Stable AGEB alignment

The earlier workflow serialized ROMs, alpha vectors, AGEB codes, and population
values separately and relied on matching positional deletions. The canonical
NPZ artifact stores all four together and validates dimensions, identifiers,
probability ranges, row sums, and population values on every load. Figure 8
comparisons align period-parts by `CVE_AGEB`, not by an assumed row number.

### Brownian parameter units

In the maintained research script, the covariance was constructed with
`parameter[0]**2`, after which `calculate_parameters` returned
`sqrt(result.x[0])`. If the optimized variable is (sigma), that second square
root changes the units and causes the occupation variance to use the wrong
scale. The extra square root was removed in both the script and the package.
Tests verify the covariance formula and recovery of a known simulated
(sigma).

The package evaluates the tridiagonal likelihood with a banded Cholesky
factorization. This is algebraically identical to the dense covariance in
Equation (9) but avoids a cubic dense factorization at every objective
evaluation.

### Random-number reproducibility

The previous occupation calculation used one module-level generator from a
multi-process `pandarallel` computation. Draw allocation could change with
worker scheduling. Residence ties and Brownian Monte Carlo calculations now
derive independent deterministic seeds from a base seed and device identifier.

### Definition of alpha and conditional ROM

The article defines (alpha_i) from the presence of at least one ping outside a
resident's home AGEB and computes (P) only among those residents who leave.
The original `estimate-alpha.py` instead classified a device by whether more
than one estimated occupation entry exceeded an arbitrary threshold of 0.05,
and the matrix aggregation included all residents. The package implements the
article's direct-ping definition and conditional average. The earlier script is
retained as part of the research record, but it is not used by
`build-model-input`.

### Figure 8 simulation

The contributed figure script is included under `research/scripts/` and its
matrix/CSV mapping is explicit in the package. The portable implementation:

- stores the four initial seed AGEBs as identifiers rather than positional
  indices;
- removes the `Y` bookkeeping state, which has no effect on (S,E,I,R);
- replaces a dense inverse of a diagonal matrix with elementwise division;
- opens and closes independent axes, preventing local and global curves from
  accumulating in the same image;
- retains complete numerical panel data in addition to the PNG; and
- evaluates the negative-difference fraction at the grid point nearest day 30.

The original expression `difference[15]` on a 100-point grid from 0 to 200
corresponds to day 30.303. Evaluating at the array midpoint would instead use
approximately day 101 and would not match the article's “approximately
(t=30)” statement.

## Explicit article/code distinctions

### Map projection

The article states that geographical information was transformed with the UTM
projection. The research Python scripts use `EPSG:3857` (Web Mercator). For
Hermosillo, the applicable WGS84 UTM zone is 12N (`EPSG:32612`). The package
therefore defaults to `EPSG:32612` and exposes `--projected-crs EPSG:3857` for
comparison with the script calculations. A result should always report which
CRS was used.

### Location-error estimation

Section 5.2 states that the unknown (delta) was estimated under the
Pozdnyakov et al. formulation. The maintained research Python script fixes
`delta_z = 28.85`, while `updated_sigma_estimation.org` contains an experiment
that jointly estimates (sigma) and (delta). The article does not report the
final numerical value or whether it was common across devices. The package
supports both choices:

- fixed (delta=28.85) m by default, matching the executable research script;
- per-trajectory joint estimation with `--estimate-location-error`, matching
  the two-parameter likelihood in Equation (9).

These modes are intentionally distinct. The motion-parameter JSON records the
value used for every device.

### Pairwise and all-observation Brownian densities

The article first gives the pairwise Brownian-bridge density in Equations
(3)–(4), then derives the all-observation BMME conditional density in Equation
(8). The research occupation code evaluates the pairwise density, even though
its parameter-estimation work also considers Equation (9). The package follows
that implemented pairwise occupation integral and documents it in
`METHODS.md`; it does not claim to evaluate the full Equation (8) conditional
density.

### Disease-induced mortality

The article defines (psi_i) but does not state a numerical value for Figure 8.
The contributed simulation script uses `phi = 0.0003`; the package names the
same parameter `psi` and retains 0.0003 as its documented default. It can be
overridden with `--psi`.

### P1A/P2A source substitution

P1A and P2A have identical dates. The figure script therefore reads the
`Second_First` matrix and `alphas_SF.csv` for both. The package preserves this
mapping in one constant and reports it in `DATA_CONTRACT.md`; it is not an
uncommented in-place source edit.

## What is verified automatically

The test suite checks:

- inclusive local-date and half-open nighttime masks;
- explicit repair of the known coordinate inversion;
- deterministic residence tie-breaking;
- direct-ping leaver classification;
- Equation (9) covariance and dense-versus-banded likelihood equality;
- fixed and joint Brownian parameter estimation on simulated trajectories;
- alpha and ROM conditioning on leavers;
- alignment-preserving NPZ serialization;
- (P^*=\operatorname{diag}(\alpha)P+\operatorname{diag}(1-\alpha)) and row
  stochasticity;
- algebraic equality between the optimized SEIRS incidence calculation and the
  original dense matrix expression;
- day-30 comparison logic; and
- a deterministic six-input, three-comparison, six-panel end-to-end run.

Continuous integration runs the tests on Python 3.11 and 3.12.

## What cannot be verified from the public checkout

The public files do not include the 80,582,452 source pings, period eligibility
lists, census geometry/population inputs, SISVER case file, or the six real
matrix/alpha pairs expected by the figure script. Consequently, the following
claims require authorized inputs:

- exact counts of retained devices and residence assignments;
- numerical Brownian parameters and ROM entries;
- Table 4 matrix distances;
- Figure 7 mobility summaries; and
- the published Figure 8 curves and the reported negative fractions of 72.92%,
  71.81%, and 86.61% near day 30.

The synthetic demonstration validates software execution and invariants only.
It must not be cited as a replication of the empirical results.
