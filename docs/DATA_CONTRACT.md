# Data contract

This document defines the files exchanged between pipeline stages. Paths may
be supplied directly to every command; the examples use the following
repository-relative layout:

```text
data/
├── private/          protected pings, eligible IDs, and case records
├── geometry/         AGEB geometries
├── processed/        period-level intermediate files
└── model_inputs/     NPZ files for the epidemic model

outputs/              figures, summaries, and numerical results
```

Git ignores the contents of `data/` and generated files under `outputs/`. The
study scripts also read these optional environment variables:

| Variable | Purpose |
|---|---|
| `TIME_RESIDENCE_DATA_DIR` | root of private and derived analysis data |
| `TIME_RESIDENCE_OUTPUT_DIR` | root of generated figures and simulations |
| `TIME_RESIDENCE_EPIDEMIOLOGY_CSV` | SISVER AGEB-level case file used to identify initial patches |

Relative examples are provided in `.env.example`.

## Mobile-phone pings

The standard CSV columns are:

| Column | Type | Definition |
|---|---|---|
| `id` | string | pseudonymous device identifier |
| `timestamp` | UTC timestamp | ping time; naive values are interpreted as UTC |
| `latitude` | float | WGS84 latitude in degrees |
| `longitude` | float | WGS84 longitude in degrees |

The aliases `id_adv`, `lat`, and `lon` are accepted and mapped to the standard
names. Coordinates must be finite and satisfy the geographic bounds for
latitude and longitude.

Some source exports contain longitude values in the latitude column and
latitude values in the longitude column. For Hermosillo records, this appears
as values near −111 under latitude and near 29 under longitude. The reader
rejects this configuration by default. After confirming the source schema,
`--repair-swapped-coordinates` exchanges the two fields during import.

## AGEB geometry and metadata

The geospatial source must:

- declare a coordinate reference system;
- contain one polygon geometry per AGEB;
- contain a unique code column, `CVE_AGEB` by default; and
- contain a strictly positive population column, `POBTOT` by default, when
  model inputs are built.

AGEBs are sorted lexicographically by the string representation of
`CVE_AGEB`. The resulting `ageb_index` is zero-based and contiguous. This
ordering indexes spatial assignments, occupation vectors, ROM rows and
columns, mobility fractions, and populations.

`prepare-agebs` writes:

| Column | Type | Definition |
|---|---|---|
| `ageb_index` | integer | zero-based matrix position |
| `CVE_AGEB` | string | AGEB identifier |
| `POBTOT` | numeric | census population |

## Pings with spatial assignment

`assign-agebs` adds:

| Column | Type | Definition |
|---|---|---|
| `CVE_AGEB` | nullable string | containing AGEB code |
| `ageb_index` | integer | AGEB index, or −1 when no polygon contains the point |

The default predicate is `within`. Points on polygon boundaries remain
unassigned. A spatial join that returns more than one polygon for a point is
rejected.

## Eligible-device table

`select-period --eligible-ids` expects a CSV with one device-identifier column.
The default column name is `id_adv` and may be changed with `--id-column`.

The article applies a weekly minimum-ping criterion. Reproducing that sample
requires the corresponding eligible-device table. If the option is omitted,
all devices observed during the selected period are included.

## Residence table

`assign-residences` writes one row per device:

| Column | Type | Definition |
|---|---|---|
| `id` | string | device identifier |
| `residence_ageb_index` | integer | assigned residence, or −1 when no candidate is available |
| `residence_CVE_AGEB` | nullable string | corresponding AGEB code |
| `all_record_modes` | string | comma-separated modal AGEB indices from all pings |
| `night_record_modes` | string | comma-separated modal AGEB indices from nighttime pings |

## Individual occupation vectors

`estimate-individual-roms` writes a JSON object keyed by device identifier.
Each value is a normalized length-$n$ occupation vector in AGEB order:

```json
{
  "device-id": [0.81, 0.04, 0.15]
}
```

The motion-parameter file uses the same keys:

```json
{
  "device-id": {
    "sigma": 12.4,
    "location_error": 28.85
  }
}
```

Failures are recorded in the JSON selected by `--failure-output`. These files
contain pseudonymous identifiers and belong in an access-controlled directory
ignored by Git.

## Epidemic-model input

Each period-part is stored in one compressed NPZ file, for example `P1A.npz`.
Loading uses `allow_pickle=False`. Required members are:

| Member | Shape | Definition |
|---|---:|---|
| `ageb_ids` | $(n,)$ | unique AGEB identifiers |
| `rom` | $(n,n)$ | row-stochastic ROM for residents who leave |
| `alpha` | $(n,)$ | fraction of sampled residents observed outside their home AGEB |
| `population` | $(n,)$ | strictly positive census populations |
| `metadata_json` | scalar string | period, definitions, and resident/leaver counts |

Validation checks matrix dimensions, unique identifiers, finite values,
nonnegative probabilities, row sums, $\alpha\in[0,1]$, and positive
populations. Resident and leaver counts in `metadata_json` document the
denominator used for each $\alpha_i$.

## Figure 8 source files

`import-article-inputs` accepts the file names used by
`research/scripts/epidemic_simulation_code.py`:

| Period | Matrix | Alpha/population CSV |
|---|---|---|
| P1A | `working_avg_resmat_Second_First.npy` | `alphas_SF.csv` |
| P1B | `working_avg_resmat_First_Second.npy` | `alphas_FS.csv` |
| P2A | `working_avg_resmat_Second_First.npy` | `alphas_SF.csv` |
| P2B | `working_avg_resmat_Second_Second.npy` | `alphas_SS.csv` |
| P3A | `working_avg_resmat_Third_First.npy` | `alphas_TF.csv` |
| P3B | `working_avg_resmat_Third_Second.npy` | `alphas_TS.csv` |

P1A and P2A share the 2020-09-21–2020-10-04 window and use the same source
pair. Each CSV must contain `CVE_AGEB`, `proporcion`, and `POBTOT`. Rows with
missing `proporcion` are omitted, and the matrix dimension must equal the
number of remaining rows. The imported mobility fraction is

$$
\alpha_i=1-\texttt{proporcion}_i.
$$

The importer stores the matrix, AGEB identifiers, mobility fractions, and
populations in one NPZ file.

## Figure 8 outputs

For each comparison `P1`, `P2`, or `P3`, the numeric NPZ contains:

| Member | Shape | Definition |
|---|---:|---|
| `time` | $(T,)$ | simulation days |
| `ageb_ids` | $(m,)$ | AGEB identifiers common to both period-parts |
| `per_ageb_difference` | $(T,m)$ | infected count in part A minus part B |
| `global_difference` | $(T,)$ | sum over common AGEBs |

`figure8_summary.json` records the number of common AGEBs, the grid time used
for the requested evaluation day, and the fraction of AGEB-level differences
below zero at that time.
