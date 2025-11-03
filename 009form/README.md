# FIEC-009 Synthetic Data Generator

This directory contains a small synthetic data generator for FIEC-009 style records used for testing data ingestion, validation, and form-filling agents for banking/finance workflows.

There are two entry points:

- `generate_fiec009_data.py` — the core generator that creates two CSV datasets: DEPOSIT and Treasury Liability. It supports seeding and configurable error injection.
- `generate_fiec009_cli.py` — a thin CLI wrapper that exposes easy modes: `clean`, `dirty`, and `paired` plus overall error-rate and seeding.

What it produces
- `fiec009_DEPOSIT_data.csv`
- `fiec009_TREASURY_LIABILITY_data.csv`

For `paired` mode the script writes:
- `out_dir/ground_truth/fiec009_...csv` (clean reference)
- `out_dir/noisy/fiec009_...csv` (deterministic noisy version where each noisy row maps back to the ground-truth row)

Why use this
- Quickly generate realistic-looking FIEC-009 test data for model/agent testing.
- Inject controlled, reproducible errors (misclassification, missing fields, malformed accounts, wrong currency, maturity/interest anomalies) to evaluate robustness.
- Paired mode lets you evaluate detection/correction logic row-by-row.

Dependencies
- Python 3.8+ (tested with Python 3.10+)
- pandas
- numpy

Optional
- `pyarrow` or `fastparquet` to enable Parquet output if you extend the scripts to write parquet files. The current scripts default to CSV.

Install quick requirements

```bash
python3 -m pip install --user pandas numpy
# optional: pip install pyarrow
```

Usage: generate_fiec009_cli.py

The CLI wrapper provides friendly options. Run the help for full options:

```bash
python3 009form/generate_fiec009_cli.py --help
```

Main options
- `--mode {clean,dirty,paired}`
  - `clean` — generate clean ground-truth CSVs (no injected errors).
  - `dirty` — generate noisy CSVs with injected errors according to `--error-rate`.
  - `paired` — generate both clean (into `ground_truth/`) and deterministic noisy datasets (into `noisy/`) so rows map 1:1.
- `--start` — start date (default `2024-01-01`).
- `--end` — end date (default `2024-12-31`).
- `--seed` — RNG seed to make generation deterministic (default `42`). Use the same seed to reproduce results.
- `--out-dir` — output directory (default `.`).
- `--error-rate` — overall percent error rate (float, default `2.0`). This is distributed across different error types by the wrapper; see notes below if you want to tune proportions.

Quick examples

Generate clean ground-truth for 2024 and write to `./gt`:

```bash
python3 009form/generate_fiec009_cli.py --mode clean --start 2024-01-01 --end 2024-12-31 --out-dir ./gt --seed 42
```

Generate noisy data (2% overall error rate) and write to `./dirty`:

```bash
python3 009form/generate_fiec009_cli.py --mode dirty --out-dir ./dirty --error-rate 2.0 --seed 42
```

Generate paired clean + deterministic noisy datasets (ground_truth/ & noisy/):

```bash
python3 009form/generate_fiec009_cli.py --mode paired --out-dir ./paired_output --error-rate 2.0 --seed 42
```

Paired/deterministic behavior
- `paired` mode generates a clean dataset first, then applies deterministic corruption using the provided `--seed` (and seed+1 internally for some steps). That means repeated runs with the same seed produce identical clean and noisy files. This lets you evaluate detection and repair logic at the per-row level.

Tuning error mixes
- The wrapper distributes the overall `--error-rate` into sub-categories (missing fields, wrong currency, invalid accounts, maturity errors, interest scale errors, misclassification). If you want different proportions, edit the mapping near the top of `generate_fiec009_cli.py` where `prop` is defined and re-run.

Where outputs are written
- `--mode clean` or `--mode dirty` writes CSVs directly to the directory passed in `--out-dir`.
- `--mode paired` writes:
  - `out_dir/ground_truth/fiec009_DEPOSIT_data.csv` and `..._TREASURY_LIABILITY_data.csv`
  - `out_dir/noisy/fiec009_DEPOSIT_data.csv` and `..._TREASURY_LIABILITY_data.csv`

Extending the generator
- The core logic lives in `generate_fiec009_data.py`. If you want additional corruptions or different distributions for balances/rates, modify that file.
- Consider adding a `--format` flag and parquet writing if you have `pyarrow`/`fastparquet` installed.

Notes and caveats
- The generator is intended for testing and simulation — the produced data is synthetic and not real account data.
- Keep the seed consistent between paired runs to preserve mapping.

If you want, I can:
- Run a short paired generation for one month and show a few matched rows (clean vs noisy).
- Add separate CLI flags to control individual error-type rates instead of a single overall `--error-rate`.

