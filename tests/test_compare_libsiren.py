"""Validate alignment with the supplied libsiren benchmark protocol."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / 'benchmarks'
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location('compare_libsiren', DIRECTORY / 'compare_libsiren.py')
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


@pytest.mark.parametrize('name', harness.MODELS)
def test_exact_references_have_positive_marginal_variances(name):
    mean, variance = harness.reference(name)
    assert mean.shape == variance.shape
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(variance))
    assert np.all(variance > 0)


@pytest.mark.parametrize('name', ['coin', 'gaussian-gaussian', 'gtree', 'kalman'])
def test_conjugate_examples_match_exact_posterior_means(name):
    result = harness.measure(name, 1, 1234)
    assert result['time_ms'] > 0
    assert result['standardized_mse'] < 1e-20


def test_wheels_has_no_timeout():
    assert harness.run_timeout('wheels', 120) is None
    assert harness.run_timeout('coin', 120) == 120


def test_csv_checkpoint_preserves_pending_and_successful_results(tmp_path):
    import csv
    path = tmp_path / 'raw.csv'
    rows = [dict(benchmark='wheels', particles=10, repeat=0, seed=1234, status='pending'),
            dict(benchmark='coin', particles=1, repeat=0, seed=1234, status='ok', time_ms=1)]
    harness.save_rows(path, rows)
    loaded = list(csv.DictReader(path.open()))
    assert [row['status'] for row in loaded] == ['pending', 'ok']
    assert not path.with_suffix('.csv.tmp').exists()


def test_wheels_particle_grid_is_one_two_three():
    assert harness.WHEELS_PARTICLES == (1, 2, 3)


def test_default_run_uses_requested_grid_and_exports(tmp_path, monkeypatch):
    import csv
    import json
    from types import SimpleNamespace
    calls = []
    exports = []

    def worker(command, **kwargs):
        calls.append((command[-3], int(command[-2]), int(command[-1]), kwargs['timeout']))
        return SimpleNamespace(stdout=json.dumps(dict(time_ms=1, standardized_mse=0, estimate=[0])))

    monkeypatch.setattr(harness.subprocess, 'run', worker)
    monkeypatch.setattr(harness, 'plot', lambda csv_path, output: exports.append((csv_path, output)))
    monkeypatch.setattr(sys, 'argv', ['compare_libsiren.py', '--output', str(tmp_path)])
    harness.main()
    rows = list(csv.DictReader((tmp_path / 'raw.csv').open()))
    assert len(rows) == len(calls) == 190
    wheels = [row for row in rows if row['benchmark'] == 'wheels']
    assert len(wheels) == 15
    assert {int(row['particles']) for row in wheels} == {1, 2, 3}
    assert all(timeout is None for name, count, seed, timeout in calls if name == 'wheels')
    assert len(exports) == 192  # Initial checkpoint, every result, final export.


def test_report_includes_real_values_and_unfinished_runs(tmp_path):
    rows = [dict(status='ok'), dict(status='running')]
    summary = [dict(benchmark='wheels', particles=2, successful_runs=1, failed_runs=0,
                    pending_runs=1, median_time_ms='1250', standardized_rmse='0.02'),
               dict(benchmark='coin', particles=10, successful_runs=0, failed_runs=0,
                    pending_runs=5, median_time_ms='', standardized_rmse='')]
    harness.write_summary_report(rows, summary, tmp_path)
    report = (tmp_path / 'all_examples_summary.md').read_text()
    assert '1 successful' in report and '1 running' in report
    assert '| wheels | 2 | 1 | 0 | 1 | 1,250.000 | 0.02 |' in report
    assert '| coin | 10 | 0 | 0 | 5 | — | — |' in report
