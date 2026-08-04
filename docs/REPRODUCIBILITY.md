# Reproducibility

The repository supports two distinct levels of use:

1. **Software verification.** The public test suite and synthetic example
   exercise data validation, model construction, numerical integration, and
   output generation.
2. **Empirical reproduction.** Regenerating the matrices and figures reported
   in the article requires restricted mobility and epidemiological inputs.

The `research/` directory contains the study scripts and notebooks. The
`time_residence` package implements the common data-processing and Figure 8
workflow with command-line interfaces, input validation, and recorded random
seeds.

## Public workflow

The following commands require no restricted data:

```bash
python -m unittest discover -s tests -v
time-residence demo --output-dir outputs/demo
```

The synthetic example writes six model inputs and runs the three period
comparisons. It tests file contracts and numerical execution rather than the
empirical estimates in the article.

## Implementation conventions

### Time and period selection

Source timestamps are interpreted as UTC and converted to
`America/Hermosillo`. Period bounds are inclusive local calendar dates.
Nighttime is `[22:00, 06:00)`.

### Spatial and array indexing

AGEB codes are stored as strings and sorted lexicographically. The same
`ageb_ids` array indexes ROM rows, ROM columns, mobility fractions, and
populations. Period comparisons align arrays by AGEB code.

Coordinate fields and validation rules are specified in the
[data contract](DATA_CONTRACT.md).

### Random numbers

Residence tie-breaking and Brownian Monte Carlo integration derive
device-specific seeds from a base seed and the device identifier. Results are
therefore independent of input row order and parallel worker scheduling.

### Numerical model inputs

Each period-part uses one NPZ file containing:

- AGEB identifiers;
- a row-stochastic ROM;
- mobility fractions;
- census populations; and
- JSON metadata.

The loader verifies dimensions, identifiers, numerical ranges, and row sums
before simulation.

## Article and implementation specifications

### Map projection

The article specifies a UTM projection. For Hermosillo, the corresponding
WGS84 system is UTM zone 12N (`EPSG:32612`), which is the package default. The
study scripts use Web Mercator (`EPSG:3857`). The command-line option
`--projected-crs` selects either system, and the selected CRS should accompany
reported results.

### Location-error parameter

Section 5.2 describes joint estimation of the Brownian parameter $\sigma$ and
location error $\delta$ under the Pozdnyakov et al. formulation. The article
does not report a numerical value for $\delta$. The executable study scripts
use $\delta=28.85\ \mathrm{m}$.

The package provides both modes:

- fixed $\delta=28.85\ \mathrm{m}$; and
- per-trajectory joint estimation selected by
  `--estimate-location-error`.

The motion-parameter JSON records both values for every estimated trajectory.

### Brownian occupation density

Occupation times use the pairwise Brownian-bridge density in Equations (3)–(4).
The all-observation conditional density in Equation (8) is not part of the
occupation estimator. Parameter estimation uses the increment covariance in
Equation (9).

### Mobility fraction and conditional ROM

The model input follows the definitions in the article:

- $\alpha_i$ is the fraction of sampled residents of AGEB $i$ observed at
  least once outside $i$; and
- row $i$ of $P$ averages occupation vectors over those residents.

`research/scripts/estimate-alpha.py` contains a separate threshold-based
exploration. The `build-model-input` command uses the direct-observation
definition above.

### Figure 8 parameters and inputs

The article defines disease-induced mortality $\psi_i$ but does not report its
numerical value for Figure 8. The Figure 8 script sets the corresponding
parameter to 0.0003, which is the package default.

P1A and P2A cover the same dates and use
`working_avg_resmat_Second_First.npy` with `alphas_SF.csv`. The complete file
mapping is listed in the [data contract](DATA_CONTRACT.md).

The Figure 8 script includes a cumulative-incidence state $Y$. Since $Y$ does
not affect $(S,E,I,R)$, the package solver integrates those four compartments.

The default simulation grid has 100 points from day 0 through day 200. The
day-30 statistic uses the closest grid point, day 30.303.

## Automated checks

The test suite covers:

- inclusive local-date and half-open nighttime masks;
- coordinate-header repair;
- deterministic residence assignment;
- direct-observation leaver classification;
- Equation (9) covariance and dense/banded likelihood equivalence;
- fixed and joint Brownian parameter estimation on synthetic trajectories;
- mobility-fraction and ROM conditioning;
- alignment-preserving NPZ serialization;
- row stochasticity of $P^*$;
- equivalence of the matrix and elementwise SEIRS formulations;
- day-30 comparison logic;
- generated files from the six-input synthetic example; and
- Markdown mathematics and control-character checks.

GitHub Actions runs the suite on Python 3.11 and 3.12.

## Restricted-data results

The public repository does not contain:

- source mobile-phone pings;
- period eligibility lists;
- AGEB geometry and population files;
- the SISVER case file; or
- the six empirical matrix/CSV input pairs.

These inputs are required to recover device counts, residence assignments,
Brownian parameters, ROM entries, Table 4 distances, Figure 7 summaries, and
the empirical Figure 8 curves and proportions.
