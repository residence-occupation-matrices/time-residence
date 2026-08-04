# Research scripts and notebooks

This directory contains the scripts and notebooks used in the analyses
associated with the article. They cover data preparation, Brownian-bridge
estimation, exploratory calculations, and epidemic simulations. Input formats
and model structures vary by script.

The supported command-line workflow is implemented in
`src/time_residence/`. It provides shared validation, path handling, and
numerical output formats for the principal analysis stages.

## Scripts

| File | Analytical role |
|---|---|
| `calculate-ageb.py` | assigns pings to sorted AGEB polygons |
| `generate-final-data.py` | selects period-parts and nighttime records |
| `determine-residence.py` | combines complete-record and nighttime residence modes |
| `residence.py` | estimates Brownian parameters and device occupation vectors |
| `runner.py` | applies occupation estimation across devices |
| `create-final-avg-matrix.py` | aggregates device vectors into period-level matrices |
| `estimate-alpha.py` | evaluates threshold-based mobility fractions |
| `epidemic_simulation_code.py` | computes the SEIRS comparisons shown in Figure 8 |
| `ageb-plots.py` | maps residence and ping counts by AGEB |
| `residence_time.py` | implements the Brownian-bridge calculation developed in the corresponding notebook |
| `mobility.py` | evaluates and plots a synthetic Brownian density |
| `simulation.py` | runs a random multi-patch SEIR experiment |
| `simulation_new.py` | studies an extended asymptomatic/symptomatic/severe infection model |
| `bigquery-test.py` | checks access to the source BigQuery table |
| `project_paths.py` | defines shared input and output paths |

`simulation_new.py` uses a compartment structure and parameter set distinct
from the Figure 8 model. The Figure 8 equations are implemented in
`epidemic_simulation_code.py` and by the `time-residence figure8` command.

## Notebooks

| File | Contents |
|---|---|
| `residence-time.org` | development of the pairwise Brownian-bridge occupation calculation |
| `updated_sigma_estimation.org` | comparison of Brownian parameter likelihoods with measurement error |
| `plot_residence_matrix.org` | ROM image comparisons |
| `duplicate-times.org` | duplicate-timestamp investigation |
| `duplicate-times.html` | HTML export of the duplicate-timestamp report |
| `parameter_estimation.org` | crosswalk between Figure 8 script variables and article notation |
| `references.bib` | Brownian-motion references cited by the notebooks |

The notebooks omit device identifiers, device timestamps, and exact device
coordinates. Cells that query mobility records require authorized access to
the confidential data.

## Paths

Scripts read from `data/` and write to `outputs/` at repository root unless an
environment variable supplies another location:

```bash
export TIME_RESIDENCE_DATA_DIR=/path/to/authorized/data
export TIME_RESIDENCE_OUTPUT_DIR=/path/to/results
export TIME_RESIDENCE_EPIDEMIOLOGY_CSV=/path/to/SISVER_Agebs_Zonas.csv
```

Local environment files, mobile identifiers, trajectories, and restricted case
records must remain outside version control.

## Pipenv environment

`environment/Pipfile.python36` and `environment/Pipfile.lock.python36` preserve
the Python 3.6 Pipenv specification and resolved dependencies used for part of
the analysis. The suffix prevents current dependency tooling from treating
this unsupported environment as an installation target. Current installations
use the repository-level `environment.yml` or `pyproject.toml`.
