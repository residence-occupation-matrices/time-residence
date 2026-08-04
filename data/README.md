# Local data directory

Place authorized inputs and derived intermediate files here when using the
repository-relative examples. Git ignores every other file in this directory.

Suggested subdirectories are `private/`, `geometry/`, `processed/`, and
`model_inputs/`. Schemas and confidentiality constraints are documented in
[`docs/DATA_CONTRACT.md`](../docs/DATA_CONTRACT.md).

Do not commit raw mobile-phone pings, pseudonymous device identifiers,
eligibility lists, SISVER records, or other restricted data.
