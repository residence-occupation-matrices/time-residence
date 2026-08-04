# Data contract

The raw observations are protected. This document defines the schemas needed
to run the code with authorized data and the invariants enforced between
pipeline stages.

## Directory convention

The command-line interface accepts any explicit path. Examples use:

```text
data/
├── private/          protected pings, eligible IDs, and case records
├── geometry/         AGEB geometries
├── processed/        period-level intermediate files
└── model_inputs/     aligned NPZ files for the epidemic model

outputs/              generated figures, summaries, and numerical results
```

Both `data/` and generated contents of `outputs/` are ignored by Git. The
scripts in `research/scripts/` use these defaults and also accept the following
environment variables:

| Variable | Purpose |
|---|---|
| `TIME_RESIDENCE_DATA_DIR` | root of private and derived analysis data |
| `TIME_RESIDENCE_OUTPUT_DIR` | root of generated figures and simulations |
| `TIME_RESIDENCE_EPIDEMIOLOGY_CSV` | SISVER AGEB-level case file used to identify initial patches |

See `.env.example` for relative examples. Environment files containing local
or protected paths must not be committed.

## Mobile-phone pings

Canonical CSV columns are:

| Column | Type | Definition |
|---|---|---|
| `id` | string | pseudonymous device identifier |
| `timestamp` | UTC timestamp | ping time; naive values are interpreted as UTC |
| `latitude` | float | WGS84 latitude in degrees |
| `longitude` | float | WGS84 longitude in degrees |

The reader accepts the research names `id_adv`, `lat`, and `lon` and converts
them to the canonical names. Coordinates must be finite and within geographic
ranges.

The original `calculate-ageb.py` selected values in `lat, lon` order but wrote
the header in `lon, lat` order. Files with the Hermosillo inversion signature
have approximately −111 in the field labelled latitude and 29 in the field
labelled longitude. The package rejects that signature unless
`--repair-swapped-coordinates` is supplied. The flag should only be used after
checking the source because a generic coordinate swap cannot be inferred
unambiguously in all regions.

## AGEB geometry and metadata

The geospatial source must:

- declare a coordinate reference system;
- contain a polygon geometry for every AGEB;
- contain a unique code column, `CVE_AGEB` by default; and
- contain a strictly positive census population column, `POBTOT` by default,
  when model inputs are built.

AGEBs are sorted lexicographically by the string representation of
`CVE_AGEB`. The resulting `ageb_index` is contiguous from zero. This same
ordering is used for spatial assignments, individual occupation vectors, ROM
rows and columns, mobility fractions, and population vectors.

`prepare-agebs` writes:

| Column | Type | Definition |
|---|---|---|
| `ageb_index` | integer | stable zero-based matrix position |
| `CVE_AGEB` | string | published AGEB identifier |
| `POBTOT` | numeric | census population |

## Pings with spatial assignment

`assign-agebs` preserves the canonical ping columns and adds:

| Column | Type | Definition |
|---|---|---|
| `CVE_AGEB` | nullable string | containing AGEB code |
| `ageb_index` | integer | stable index, or −1 when no polygon contains the point |

The default predicate is `within`, matching the strict point-in-polygon rule
in the research scripts. A point exactly on a polygon boundary is not assigned.
The spatial join fails if overlapping polygons produce more than one match.

## Eligible-ID table

`select-period --eligible-ids` expects a CSV containing one device-identifier
column. Its default name is `id_adv` and can be changed with `--id-column`.

The article retained devices meeting its weekly minimum-ping criterion. Those
lists are not tracked and cannot be reconstructed without the protected raw
records. Omitting the option changes the analytical sample and must therefore
be reported in any derived analysis.

## Residence table

`assign-residences` writes one row per device:

| Column | Type | Definition |
|---|---|---|
| `id` | string | device identifier |
| `residence_ageb_index` | integer | assigned residence, or −1 if no valid candidate exists |
| `residence_CVE_AGEB` | nullable string | corresponding AGEB code |
| `all_record_modes` | string | comma-separated modal AGEB indices from all pings |
| `night_record_modes` | string | comma-separated modal AGEB indices from nighttime pings |

## Individual occupation vectors

`estimate-individual-roms` writes a JSON object. Each key is a device
identifier and each value is a length-(n) normalized occupation vector in
stable AGEB order:

```json
{
  "device-id": [0.81, 0.04, 0.15]
}
```

The separate motion-parameter JSON contains:

```json
{
  "device-id": {
    "sigma": 12.4,
    "location_error": 28.85
  }
}
```

Devices that cannot be estimated are omitted from these outputs and recorded
in the failure JSON selected by `--failure-output`. These files contain
pseudonymous identifiers and belong under an access-controlled, ignored data
directory.

## Epidemic model input

Each period-part is stored in one compressed NPZ file such as `P1A.npz`. Pickle
is disabled on load. Required members are:

| Member | Shape | Definition |
|---|---:|---|
| `ageb_ids` | `(n,)` | unique AGEB identifiers |
| `rom` | `(n,n)` | row-stochastic ROM conditioned on residents who leave |
| `alpha` | `(n,)` | fraction of sampled residents who leave home AGEB |
| `population` | `(n,)` | strictly positive census populations |
| `metadata_json` | scalar string | period, definitions, counts, and provenance |

Validation rejects non-square matrices, duplicate identifiers, non-finite
numbers, negative probabilities, row sums outside tolerance, alpha values
outside `[0,1]`, and nonpositive populations.

The metadata generated from individual observations records resident and
leaver counts for every retained row. This makes (alpha)'s denominator
auditable without exposing individual trajectories.

## Matrix/CSV inputs used by the Figure 8 script

`import-article-inputs` accepts the file layout used by
`research/scripts/epidemic_simulation_code.py`:

| Period | Matrix | Alpha/population CSV |
|---|---|---|
| P1A | `working_avg_resmat_Second_First.npy` | `alphas_SF.csv` |
| P1B | `working_avg_resmat_First_Second.npy` | `alphas_FS.csv` |
| P2A | `working_avg_resmat_Second_First.npy` | `alphas_SF.csv` |
| P2B | `working_avg_resmat_Second_Second.npy` | `alphas_SS.csv` |
| P3A | `working_avg_resmat_Third_First.npy` | `alphas_TF.csv` |
| P3B | `working_avg_resmat_Third_Second.npy` | `alphas_TS.csv` |

P1A and P2A use the same dates and therefore the same source pair. Each CSV
must contain `CVE_AGEB`, `proporcion`, and `POBTOT`. Rows with missing
`proporcion` are removed in their existing order. The square matrix dimension
must then equal the number of retained rows. The imported mobility fraction is

\[
\alpha_i=1-\texttt{proporcion}_i.
\]

Matrix, identifiers, alpha, and population are immediately stored together in
the canonical NPZ contract; later stages never rely on separate positional
files.

## Figure 8 outputs

For each comparison `P1`, `P2`, or `P3`, the numeric NPZ contains:

| Member | Shape | Definition |
|---|---:|---|
| `time` | `(T,)` | simulation days |
| `ageb_ids` | `(m,)` | identifiers common to both period-parts |
| `per_ageb_difference` | `(T,m)` | infected count in part A minus part B |
| `global_difference` | `(T,)` | row-wise sum over common AGEBs |

`figure8_summary.json` records `common_agebs`, the actual grid time nearest the
requested evaluation day, and the fraction of individual differences below
zero at that time.
