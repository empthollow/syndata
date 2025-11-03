import pandas as pd
import tempfile
import shutil
import os
import sys

import runpy
import types
from pathlib import Path

# load cornmill script via runpy (works for executable scripts without .py suffix)
cm_path = Path('/home/shoestring/syndata/cornmill')
if not cm_path.exists():
    raise FileNotFoundError(f"cornmill script not found at {cm_path}")
g = runpy.run_path(str(cm_path))
module = types.ModuleType('cornmill')
for k, v in g.items():
    setattr(module, k, v)
cm = module


def fail(msg):
    print('FAIL:', msg)
    sys.exit(1)


def ok(msg):
    print('OK:', msg)


def test_sensor_generation_small():
    start = pd.Timestamp("2023-01-01")
    end = start + pd.Timedelta(days=1)
    sensors = cm.generate_sensor_data(start, end, "1h")
    if sensors.empty:
        fail('sensors empty')
    if 'timestamp' not in sensors.columns:
        fail('timestamp column missing')
    ok('sensor_generation_small')


def test_truck_arrivals():
    start = pd.Timestamp("2023-01-01")
    end = start + pd.Timedelta(days=3)
    trucks = cm.generate_truck_arrivals(start, end, mean_per_day=1)
    if not isinstance(trucks, pd.DataFrame):
        fail('trucks not dataframe')
    ok('truck_arrivals')


def test_kpi_aggregation():
    start = pd.Timestamp("2023-01-01")
    end = start + pd.Timedelta(days=1)
    sensors = cm.generate_sensor_data(start, end, "5min")
    kpis = cm.generate_kpis_from_sensors(sensors, "1h")
    if 'throughput_tph' not in kpis.columns:
        fail('throughput_tph missing')
    ok('kpi_aggregation')


def test_smoke_end_to_end():
    tmpdir = tempfile.mkdtemp(prefix='cornmill-test-')
    try:
        old_out = cm.CONFIG.get('out_dir')
        cm.CONFIG['out_dir'] = tmpdir
        cm.CONFIG['months_to_generate'] = 0
        cm.CONFIG['base_start'] = '2023-01-01'
        cm.generate_dataset()
        out = os.path.join(tmpdir, 'cornmill')
        if not os.path.exists(out):
            fail('output dir not created')
        ok('smoke_end_to_end')
    finally:
        cm.CONFIG['out_dir'] = old_out
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    test_sensor_generation_small()
    test_truck_arrivals()
    test_kpi_aggregation()
    test_smoke_end_to_end()
    print('ALL TESTS PASSED')
    sys.exit(0)
