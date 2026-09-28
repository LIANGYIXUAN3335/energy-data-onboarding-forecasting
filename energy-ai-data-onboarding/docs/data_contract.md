# Data contract

The published forecasting inputs have exactly these columns, in order:

```text
timestamp,building_id,load,site_id,primary_use,condition
```

| Field | Contract |
|---|---|
| `timestamp` | Parseable hourly source-local wall-clock label represented with a UTC marker; see the timezone limitation below |
| `building_id` | Non-empty source building identifier |
| `load` | Finite numeric value in the BDG2-supplied unit, or null when unresolved; no implicit unit conversion |
| `site_id` | Metadata site identifier |
| `primary_use` | Source primary-space-use category normalized to this field name |
| `condition` | Exactly one of `reference`, `corrupted`, or `remediated` |

The logical key is `(timestamp, building_id)` and must be unique. All three
conditions must contain the same key set in the same stable order. A failed or
quarantined value is represented by a null load, never by silently deleting the
key. Metadata joins are many observations to one building record.

Expected source frequency is hourly. Gaps, duplicates, ordering errors,
non-finite and negative loads, long zeros, stuck segments, robust outliers,
level/scale shifts, metadata orphans, group coverage, season coverage, and
timezone availability are audited.

BDG2 raw meter timestamps are local to each site. The committed subset uses
only `US/Eastern` sites. The current implementation does not claim
cross-timezone UTC conversion and explicitly records this limitation in every
run manifest and report. Its machine-readable semantics are
`basis=source_local_wall_clock`, `canonical_marker=UTC`,
`offset_conversion_applied=false`, and `dst_disambiguated=false`.
