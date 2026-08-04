# Research scripts and notebooks

This directory contains the complete set of computational materials associated
with the article. The scripts are kept together without separation by
contributor or by the date on which a file entered the repository.

The files document several stages and experiments, so they should not all be
expected to accept the same inputs or implement the same epidemiological model.
For repeatable command-line execution, validation, and tests, use the package
under `src/time_residence/`. The package consolidates these computations while
the files here retain their article-specific structure and context.

## Scripts

| File | Analytical role |
|---|---|
| `calculate-ageb.py` | assigns raw pings to sorted AGEB polygons |
| `generate-final-data.py` | selects period-parts and nighttime records |
| `determine-residence.py` | combines all-record and nighttime modal residence criteria |
| `residence.py` | estimates Brownian parameters and per-device occupation vectors |
| `runner.py` | applies residence estimation across devices |
| `create-final-avg-matrix.py` | aggregates individual vectors into period-level matrices |
| `estimate-alpha.py` | explores threshold-based mobility fractions |
| `epidemic_simulation_code.py` | simulates the article's SEIRS comparisons and Figure 8 |
| `ageb-plots.py` | maps residence and ping counts by AGEB |
| `residence_time.py` | standalone Brownian-bridge calculation generated from the corresponding notebook |
| `mobility.py` | synthetic Brownian-density calculation and visualization |
| `simulation.py` | random multi-patch SEIR experiment |
| `simulation_new.py` | extended asymptomatic/symptomatic/severe infection-model experiments |
| `bigquery-test.py` | checks access to the source BigQuery table |
| `project_paths.py` | shared path configuration for this directory |

`simulation_new.py` uses an extended compartment structure and parameter set
from related modeling work. The SEIRS calculation corresponding to Figure 8 is
implemented by `epidemic_simulation_code.py` and by the tested
`time-residence figure8` command.

## Notebooks

| File | Contents |
|---|---|
| `residence-time.org` | literate development of the pairwise Brownian-bridge occupation calculation |
| `updated_sigma_estimation.org` | comparison of Brownian parameter likelihoods with measurement error |
| `plot_residence_matrix.org` | ROM image comparisons |
| `duplicate-times.org` | duplicate timestamp investigation |
| `duplicate-times.html` | exported duplicate timestamp report |
| `parameter_estimation.org` | short parameter-estimation notes |

Recorded notebook outputs are evidence of the analysis session in which they
were produced. Device-level identifiers, timestamps, and exact coordinates
have been removed from those outputs because the underlying mobility data are
confidential. The notebooks are not executed by continuous integration.

## Paths and protected inputs

The scripts no longer contain workstation-specific absolute paths. By default,
they read from `data/` and write to `outputs/` at repository root. Override
those locations without editing source:

```bash
export TIME_RESIDENCE_DATA_DIR=/path/to/authorized/data
export TIME_RESIDENCE_OUTPUT_DIR=/path/to/results
export TIME_RESIDENCE_EPIDEMIOLOGY_CSV=/path/to/SISVER_Agebs_Zonas.csv
```

The values above are examples only. Do not commit local environment files,
mobile identifiers, raw trajectories, or restricted case records.

## Original environment

`environment/Pipfile` and `environment/Pipfile.lock` record the environment in
which part of the analysis was developed. They target Python 3.6 and are kept
for dependency provenance, not as the current installation method. Use the
repository-level `environment.yml` or `pyproject.toml` for supported execution.
