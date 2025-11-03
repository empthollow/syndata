import pandas as pd
from pathlib import Path
import os

# import functions from cornmill
import cornmill as cm


def test_sensor_generation_small():
    start = pd.Timestamp("2023-01-01")
    end = start + pd.Timedelta(days=1)
    sensors = cm.generate_sensor_data(start, end, "1h")
    assert not sensors.empty
    assert "timestamp" in sensors.columns


def test_truck_arrivals():
    start = pd.Timestamp("2023-01-01")
    end = start + pd.Timedelta(days=3)
    trucks = cm.generate_truck_arrivals(start, end, mean_per_day=1)
    assert isinstance(trucks, pd.DataFrame)


def test_kpi_aggregation():
    start = pd.Timestamp("2023-01-01")
    end = start + pd.Timedelta(days=1)
    sensors = cm.generate_sensor_data(start, end, "5min")
    kpis = cm.generate_kpis_from_sensors(sensors, "1h")
    assert "throughput_tph" in kpis.columns


def test_smoke_end_to_end(tmp_path):
    # run a tiny dataset and ensure files written
    cwd = Path(os.getcwd())
    prev_out = Path(cm.CONFIG["out_dir"]) / "cornmill"
    # ensure a clean tmp output
    cm.CONFIG["out_dir"] = str(tmp_path)
    cm.CONFIG["months_to_generate"] = 0  # only base week
    cm.CONFIG["base_start"] = "2023-01-01"
    cm.generate_dataset()
    out = Path(cm.CONFIG["out_dir"]) / "cornmill"
    assert out.exists()
    # reset not strictly necessary in tests
