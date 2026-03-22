# Bronze Layer

The bronze layer converts the raw MATLAB payload into a structured, long-format time
series stored as parquet. This is a minimal transformation: the data is restructured
but not filtered, aggregated, or enriched. NaN values from the raw data are preserved.

## Related Documents

| Document | Location |
|----------|----------|
| Architecture overview | [../000_overview/architecture.md](../000_overview/architecture.md) |
| Previous layer (Raw) | [../001_raw/raw.md](../001_raw/raw.md) |
| Next layer (Silver) | [../003_silver/silver.md](../003_silver/silver.md) |
| Pipeline operations | [pipeline.md](../../002_pipeline/pipeline.md) |
| Glossary | [glossary.md](../../004_reference/glossary.md) |

## Script

| Field | Value |
|-------|-------|
| Entry script | `scripts/000_raw_to_bronze.py` |
| Canonical implementation | `scripts/stages/raw_to_bronze.py` |
| Entry function | `raw_to_bronze(raw_path=None, output_path=None)` |
| Config source | `scripts/config.py` (`PATHS`, `SCHEMAS`, `VALID_DAY_CLASSES`) |

## Input and Output

| Direction | Path | Format |
|-----------|------|--------|
| Input | `data/000_raw/P_data.mat` | MATLAB `.mat` |
| Output | `data/001_bronze/power_load_1s.parquet` | Apache Parquet |

## Schema

| Column | Dtype | Nullable | Description |
|--------|-------|----------|-------------|
| `timestamp` | `datetime64[ns]` | No | Absolute timestamp at second-level precision |
| `day_class` | `string` | No | Business-day classification (`full`, `half`, `none`) |
| `load` | `float64` | Yes | Power load in watts; `NaN` indicates missing/invalid |

## Row Count

One row per second across all days:

```text
rows = 86,400 seconds/day x 31 days = 2,678,400
```

## Transformation Logic

The script performs the following steps in order:

1. Load the `.mat` file using `scipy.io.loadmat`.
2. Extract `P_data`, `day_data`, and `day_class` arrays.
3. Validate: all three keys present, `P_data` has 86,400 rows, column counts match.
4. Parse date values from `day_data` into `datetime` objects.
5. Melt the wide `P_data` matrix (columns = days) into long format:
   - Each cell becomes one row with `second_of_day`, `date`, and `load`.
6. Compute `timestamp` by combining `date` + `second_of_day` as timedelta.
7. Sort by `timestamp` and reset index.
8. Create a `day_class` lookup from `day_data` and `day_class` arrays.
9. Merge `day_class` onto the time series by date.
10. Select final columns: `timestamp`, `day_class`, `load`.

## Validation

The script validates the following before writing output:

| Check | Raises on failure |
|-------|-------------------|
| `.mat` file exists | `ValueError` with path |
| `P_data`, `day_data`, `day_class` keys present | `ValueError` naming missing key |
| `P_data` is 2D with 86,400 rows | `ValueError` with actual shape |
| Column count consistent across arrays | `ValueError` with all three counts |
| `day_class` values in `{full, half, none}` | `ValueError` listing unexpected values |
| `timestamp` is monotonically increasing | `ValueError` with row index and conflicting timestamps |
| Row count equals `P_data.size` | `ValueError` with expected vs actual |
| Output columns match `SCHEMAS["bronze"]["columns"]` | `ValueError` with schema diff |

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Missing `.mat` file | Fast-fail with path in error message |
| Corrupt `.mat` file | `RuntimeError` wrapping `loadmat` exception |
| Output directory missing | Auto-created via `Path.mkdir(parents=True)` |
| Parquet write failure | `RuntimeError` with path and OS-level message |

## Logging

After successful write, the script logs:
- Total row count
- Timestamp range (min and max)
- Null counts per column
- `day_class` value distribution (full/half/none counts)
- Output file path

## Key Design Decisions

NaN preservation:
- NaN values in `P_data` are carried through to the `load` column without modification.
  The bronze layer does not drop, fill, or interpolate missing values. This preserves
  the raw signal fidelity for downstream analysis and allows the silver layer to make
  informed decisions about NaN handling at each resolution.

Single output file:
- All 31 days are stored in one parquet file rather than one file per day. This
  simplifies downstream reads and avoids partition management overhead for a dataset
  of this size.

No feature engineering:
- The bronze layer adds no derived columns. Its sole purpose is format conversion
  and structural validation. All feature engineering begins at the silver layer.

## EDA Coverage

Exploratory analysis of the bronze layer is documented in `notebooks/001_bronze_eda.ipynb`:
- Day-class breakdown table (date, class, row count, NaN count, NaN percentage per day)
- NaN clustering analysis (consecutive-second runs vs scattered)
- Outlier day detection (deviation from same-class peers)
- All-days overlay plot colored by day_class
- Basic statistics (min, max, mean, median, std overall and per day_class)
- Zero-load period detection



