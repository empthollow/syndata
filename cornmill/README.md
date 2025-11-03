# Synthetic Cornmill Data

Compact synthetic data generator for a corn milling plant. The main script is `cornmill`.

Quick start

```bash
python3 cornmill
```

Output

Files are written under `synthetic_cornmill_data/cornmill/week_<YYYY-MM-DD>/` in CSV format by default.

Parquet support

To enable parquet output, install `pyarrow` or `fastparquet`:

```bash
python -m pip install fastparquet
```

If no parquet engine is available the generator will default to CSV (or write a CSV fallback and raise a helpful error when parquet is requested).
 
Parquet engine detection

When you request parquet output (via `--format parquet` or by setting `CONFIG['output_format']='parquet'`), the generator will attempt to use `pyarrow` first, then `fastparquet`. If neither is installed the generator will write a CSV fallback file and raise an informative error telling you which file was written. Install one of these packages to enable parquet output.
