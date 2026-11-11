"""Checks for plan selection and benchmark result alignment."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('compare_modes', Path(__file__).resolve().parents[1] / 'benchmarks/compare_modes.py')
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)



def test_result_alignment_and_mse():
    output = '===== Evaluation Time =====\n0.25\n===== Result =====\n([1, 3], (2, [4, 6]))\n'
    truths = [('trajectory', [0, 1]), ('scalar', [1]), ('other', [4, 4])]
    runtime, errors = harness.parse_output(output, truths)
    assert runtime == .25
    assert errors == [('trajectory', 2.5), ('scalar', 1), ('other', 2)]
    with pytest.raises(ValueError, match='expected'):
        harness.parse_output(output, [('trajectory', [0]), *truths[1:]])
    with pytest.raises(ValueError, match='variable count'):
        harness.parse_output(output, truths[:1])


def test_summary_deduplicates_runtime_and_normalizes_each_output():
    rows = []
    for repeat, runtime in [(0, .01), (1, .03)]:
        for variable, mse, scale in [('a', 4, 2), ('b', 16, 2)]:
            rows.append(dict(example='test', handler='smc', method='ssi',
                             particles='1000', repeat=str(repeat), seed=str(repeat),
                             runtime_seconds=str(runtime), variable=variable,
                             error=str(mse), reference_scale=str(scale), status='ok'))
    result = harness.summarize(rows)
    assert len(result) == 1
    assert result[0]['median_runtime_ms'] == 20
    assert result[0]['mean_rmse'] == 1.5
    assert result[0]['runs'] == 2


def test_failed_runs_are_excluded_from_summary():
    assert harness.summarize([dict(status='failed')]) == []


def test_empirical_reference_uses_only_largest_measured_count():
    results = [({'particles': 1}, [[100], [200, 300]]),
               ({'particles': 1000}, [[2], [4, 6]]),
               ({'particles': 1000}, [[4], [6, 8]])]
    count, reference = harness.empirical_reference(results)
    assert count == 1000
    assert reference == [('output_0', [3]), ('output_1', [5, 7])]


def test_harness_never_launches_extra_reference_runs(tmp_path, monkeypatch):
    from argparse import Namespace
    calls = []
    def invoke(source, args, particles, seed):
        calls.append((particles, seed))
        return f'===== Evaluation Time =====\n0.1\n===== Result =====\n{seed + 1}\n'
    monkeypatch.setattr(harness, 'invoke', invoke)
    args = Namespace(examples=['coin'], references=None, particles=[1, 1000],
                     repeats=10, seed=0, handler='smc', method='ssi')
    harness.run(args, tmp_path / 'results.csv')
    assert len(calls) == 20
    assert {p for p, seed in calls} == {1, 1000}
