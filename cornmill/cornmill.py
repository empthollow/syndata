#!/usr/bin/env python3
"""Synthetic Corn Mill Data Generator (clean)

Single-file smoke-test generator producing weekly CSV partitions for:
- sensors (5-min by default)
- production KPIs (hourly)
- truck arrivals (events)

This version is intentionally compact and validated to avoid parse errors.
"""

from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import random

CONFIG = {
    "months_to_generate": 1,
    "base_start": "2023-01-01",
    "resolutions": {"sensor": "5min", "kpi": "1h", "lab": "1d"},
    "output_format": "csv",
    "out_dir": "synthetic_cornmill_data",
    "random_seed": 42,
}

np.random.seed(CONFIG["random_seed"])
random.seed(CONFIG["random_seed"])

SENSORS = [
    {"id": "feed_rate_tph", "unit": "tons/hr", "base": 30, "std": 3},
    {"id": "motor_power_kw", "unit": "kW", "base": 120, "std": 8},
    {"id": "dryer_temp_c", "unit": "C", "base": 85, "std": 4},
    {"id": "vibration_mm_s", "unit": "mm/s", "base": 1.2, "std": 0.25},
]


def make_index(start, end, freq):
    # some pandas versions don't support 'closed' in date_range; make end exclusive
    end_adj = pd.to_datetime(end) - pd.Timedelta(seconds=1)
    return pd.date_range(start, end_adj, freq=freq)


def daily_profile_fraction(ts: pd.DatetimeIndex):
    hours = ts.hour + ts.minute / 60.0
    return 0.9 + 0.15 * np.sin(2 * np.pi * (hours - 6) / 24)


def generate_sensor_data(start, end, freq):
    ts = make_index(start, end, freq)
    rows = []
    frac = daily_profile_fraction(ts)
    # small weekly effect: slightly lower production on weekends
    weekday = ts.dayofweek
    weekly_factor = 1.0 - 0.05 * (weekday >= 5)
    frac = frac * weekly_factor
    for s in SENSORS:
        noise = np.random.normal(0, s["std"], len(ts))
        values = s["base"] * frac + noise
        df = pd.DataFrame({"timestamp": ts, "sensor_id": s["id"], "unit": s["unit"], "value": values})
        rows.append(df)
    sensors = pd.concat(rows, ignore_index=True)
    sensors.loc[np.random.rand(len(sensors)) < 0.001, "value"] = np.nan
    sensors["status_flag"] = sensors["value"].notna()
    return sensors


def generate_truck_arrivals(start, end, mean_per_day=3):
    # make day range end-exclusive
    end_days = pd.to_datetime(end).normalize() - pd.Timedelta(seconds=1)
    days = pd.date_range(pd.to_datetime(start).normalize(), end_days, freq="D")
    trucks = []
    suppliers = ["SupplierA", "SupplierB", "SupplierC"]
    grades = ["Feed", "Premium", "Standard"]
    for d in days:
        n = np.random.poisson(mean_per_day)
        for i in range(n):
            t = d + pd.to_timedelta(np.random.randint(6 * 60, 18 * 60), unit="m")
            tons = float(max(5, np.random.normal(20, 4)))
            supplier = random.choice(suppliers)
            grade = random.choice(grades)
            # moisture around 14% for inbound grain with some supplier/grade variability
            moisture = float(np.clip(np.random.normal(14.0 + (0.5 if grade=="Feed" else 0.0), 1.2), 10.0, 22.0))
            trucks.append({
                "timestamp_arrival": t,
                "truck_id": f"T{d.strftime('%Y%m%d')}_{i}",
                "tons": tons,
                "supplier": supplier,
                "grade": grade,
                "moisture_pct": moisture,
            })
    return pd.DataFrame(trucks)


def generate_kpis_from_sensors(sensors_df, freq_kpi, trucks_df=None):
    feed = sensors_df[sensors_df["sensor_id"] == "feed_rate_tph"][["timestamp", "value"]].rename(columns={"value": "feed_tph"})
    feed = feed.set_index("timestamp").resample(freq_kpi).mean().ffill().reset_index()
    if feed.empty:
        return pd.DataFrame(columns=["timestamp", "throughput_tph", "yield_pct"])
    feed["throughput_tph"] = feed["feed_tph"] * 0.95
    # base yield
    base_yield = np.clip(np.random.normal(0.92, 0.02, len(feed)), 0.8, 0.98)
    feed["yield_pct"] = base_yield
    # if trucks data is provided, adjust yield based on recent inbound moisture
    if trucks_df is not None and not trucks_df.empty:
        # compute daily average moisture
        trucks_df = trucks_df.copy()
        trucks_df["date"] = pd.to_datetime(trucks_df["timestamp_arrival"]).dt.normalize()
        daily_m = trucks_df.groupby("date")["moisture_pct"].mean()
        # map each kpi timestamp to its date and adjust yield
        kpi_dates = pd.to_datetime(feed["timestamp"]).dt.normalize()
        moisture_for_kpi = kpi_dates.map(lambda d: daily_m.get(d, np.nan))
        # moisture effect: for each percent above baseline reduce yield by 1% of yield
        baseline = 13.5
        effect_per_pct = 0.01
        adj = np.where(~pd.isna(moisture_for_kpi), 1.0 - effect_per_pct * (moisture_for_kpi - baseline), 1.0)
        feed["yield_pct"] = np.clip(feed["yield_pct"] * adj, 0.7, 0.99)
    return feed[["timestamp", "throughput_tph", "yield_pct"]]


def save_df(df: pd.DataFrame, path: Path, fmt: str):
    if fmt == "parquet":
        engine = detect_parquet_engine()
        if engine is None:
            # no engine available: write CSV fallback and raise informative error
            warn_path = path.with_suffix(path.suffix + ".no_parquet_fallback.csv")
            df.to_csv(warn_path, index=False)
            raise RuntimeError(
                f"No parquet engine available (pyarrow/fastparquet). Wrote CSV fallback to {warn_path}."
            )
        else:
            # be explicit about engine for deterministic writes
            df.to_parquet(path, engine=engine, index=False)
    else:
        df.to_csv(path, index=False)


def detect_parquet_engine():
    """Return the parquet engine available or None."""
    try:
        import pyarrow  # type: ignore

        return "pyarrow"
    except Exception:
        try:
            import fastparquet  # type: ignore

            return "fastparquet"
        except Exception:
            return None


## --- equipment state machine (simple MTBF/MTTR) ---
EQUIPMENT = [
    {"id": "mill_1", "mtbf_hours": 300, "mttr_hours": 6},
    {"id": "dryer_1", "mtbf_hours": 480, "mttr_hours": 8},
]


def generate_equipment_states(start, end, freq="1h"):
    idx = make_index(start, end, freq)
    rows = []
    for eq in EQUIPMENT:
        state = "RUN"
        t_until_failure = np.random.exponential(eq["mtbf_hours"])  # hours
        remaining = t_until_failure
        for ts in idx:
            rows.append({"timestamp": ts, "equipment_id": eq["id"], "state": state})
            remaining -= pd.Timedelta(freq).total_seconds() / 3600.0
            if remaining <= 0 and state == "RUN":
                state = "DOWN"
                remaining = np.random.exponential(eq["mttr_hours"])
            elif remaining <= 0 and state == "DOWN":
                state = "RUN"
                remaining = np.random.exponential(eq["mtbf_hours"])
    return pd.DataFrame(rows)


def generate_maintenance_and_alarms(equipment_states: pd.DataFrame):
    # convert DOWN intervals into maintenance workorders and alarms
    workorders = []
    alarms = []
    for eq_id, grp in equipment_states.groupby("equipment_id"):
        grp = grp.sort_values("timestamp")
        down = grp[grp["state"] == "DOWN"]
        if down.empty:
            continue
        # collapse contiguous down blocks
        down["block"] = (down["timestamp"].diff() > pd.Timedelta("2h")).cumsum()
        for _, block in down.groupby("block"):
            start = block["timestamp"].min()
            end = block["timestamp"].max()
            workorders.append({"equipment_id": eq_id, "start": start, "end": end, "type": "repair"})
            alarms.append({"equipment_id": eq_id, "timestamp": start, "severity": "high", "message": "equipment down"})
    return pd.DataFrame(workorders), pd.DataFrame(alarms)


def simulate_inventory(trucks_df: pd.DataFrame, kpis_df: pd.DataFrame, start, end):
    # very small mass-balance: inventory increases by arrivals and decreases by throughput
    days = pd.date_range(pd.to_datetime(start).normalize(), pd.to_datetime(end).normalize(), freq="D")
    records = []
    inventory = 500.0  # starting tons
    for d in days:
        arrivals = trucks_df[(trucks_df["timestamp_arrival"] >= d) & (trucks_df["timestamp_arrival"] < d + pd.Timedelta(days=1))]
        in_tons = arrivals["tons"].sum() if not arrivals.empty else 0.0
        # assume daily throughput is sum of hourly throughput * 24 (approx)
        day_kpis = kpis_df[(kpis_df["timestamp"] >= d) & (kpis_df["timestamp"] < d + pd.Timedelta(days=1))]
        out_tons = day_kpis["throughput_tph"].sum() if not day_kpis.empty else 0.0
        inventory = max(0.0, inventory + in_tons - out_tons)
        records.append({"date": d, "inventory_tons": inventory, "in_tons": in_tons, "out_tons": out_tons})
    return pd.DataFrame(records)


def generate_lab_results(trucks_df: pd.DataFrame, production_kpis: pd.DataFrame):
    # sample a subset of trucks and finished product for basic lab metrics
    if trucks_df.empty:
        return pd.DataFrame(columns=["sample_id", "timestamp", "type", "moisture_pct", "protein_pct"])
    sampled = trucks_df.sample(min(5, len(trucks_df)))
    labs = []
    for i, row in sampled.iterrows():
        labs.append({"sample_id": f"IN_{i}", "timestamp": row["timestamp_arrival"], "type": "incoming", "moisture_pct": float(np.clip(np.random.normal(14.0, 1.0), 10, 22)), "protein_pct": float(np.clip(np.random.normal(8.5, 0.4), 6.0, 12.0))})
    # finished product samples
    if not production_kpis.empty:
        samples = production_kpis.sample(min(3, len(production_kpis)))
        for i, row in samples.iterrows():
            labs.append({"sample_id": f"OUT_{i}", "timestamp": row["timestamp"], "type": "finished", "moisture_pct": float(np.clip(np.random.normal(12.5, 0.3), 10, 16)), "protein_pct": float(np.clip(np.random.normal(8.8, 0.2), 7.5, 9.5))})
    return pd.DataFrame(labs)


def estimate_energy(sensors_df: pd.DataFrame):
    # crude energy estimate from motor_power_kw and dryer_temp
    e = sensors_df[sensors_df["sensor_id"] == "motor_power_kw"][['timestamp', 'value']].rename(columns={'value': 'motor_kw'})
    d = sensors_df[sensors_df["sensor_id"] == "dryer_temp_c"][['timestamp', 'value']].rename(columns={'value': 'dryer_c'})
    merged = pd.merge_asof(e.sort_values('timestamp'), d.sort_values('timestamp'), on='timestamp', direction='nearest')
    merged['energy_kwh'] = merged['motor_kw'] * 1.0 + np.clip((merged['dryer_c'] - 70) * 0.1, 0, None)
    return merged[['timestamp', 'energy_kwh']]


def generate_dataset():
    out_dir = Path(CONFIG["out_dir"]) / "cornmill"
    out_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(CONFIG["base_start"])
    end = start + pd.DateOffset(months=CONFIG["months_to_generate"])
    # week boundaries: make end exclusive similar to other ranges
    weeks = pd.date_range(start, pd.to_datetime(end) - pd.Timedelta(seconds=1), freq="W")
    if len(weeks) < 2:
        weeks = pd.DatetimeIndex([start, end])
    for i in range(len(weeks) - 1):
        ws = weeks[i]
        we = weeks[i + 1]
        sensors = generate_sensor_data(ws, we, CONFIG["resolutions"]["sensor"])
        kpis = generate_kpis_from_sensors(sensors, CONFIG["resolutions"]["kpi"])
        trucks = generate_truck_arrivals(ws, we)
        week_dir = out_dir / f"week_{ws.strftime('%Y-%m-%d')}"
        week_dir.mkdir(parents=True, exist_ok=True)
        save_df(sensors, week_dir / "sensors.csv", CONFIG["output_format"])
        save_df(kpis, week_dir / "production_kpi.csv", CONFIG["output_format"])
        save_df(trucks, week_dir / "trucks.csv", CONFIG["output_format"])
    print("Done. Data written to", out_dir.resolve())


def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Synthetic corn mill data generator")
    p.add_argument("--months", type=int, default=None, help="months to generate (overrides config)")
    p.add_argument("--start", type=str, default=None, help="base start date YYYY-MM-DD")
    p.add_argument("--out-dir", type=str, default=None, help="output directory base")
    p.add_argument("--format", choices=["csv", "parquet"], default=None, help="output format")
    p.add_argument("--smoke", action="store_true", help="run a short smoke test (one week)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.months is not None:
        CONFIG["months_to_generate"] = args.months
    if args.start is not None:
        CONFIG["base_start"] = args.start
    if args.out_dir is not None:
        CONFIG["out_dir"] = args.out_dir
    if args.format is not None:
        CONFIG["output_format"] = args.format
    if args.smoke:
        # run one week only
        CONFIG["months_to_generate"] = 0
    generate_dataset()


if __name__ == "__main__":
    main()
