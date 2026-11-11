"""Benchmark Original Siren examples using the libsiren runtime/error protocol."""
import argparse
import csv
import json
import math
import itertools
from functools import lru_cache
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / 'examples'
MODELS = ('gaussian-gaussian', 'gtree', 'kalman', 'wheels', 'slam', 'outlier', 'coin', 'gaussian-gaussian-hybrid')
DEFAULT_PARTICLES = (1, 10, 100, 1000, 10000)
WHEELS_PARTICLES = (1, 2, 3)
FIELDS = ('benchmark', 'particles', 'repeat', 'seed', 'status', 'time_ms', 'standardized_mse', 'estimate', 'error')


@lru_cache(maxsize=None)
def reference(name):
    """Exact posterior means and marginal variances for the intended models."""
    if name == 'gaussian-gaussian-hybrid':
        probability = (1 / math.sqrt(11)) / (1 / math.sqrt(11) + 1 / 10)
        return np.array([0.]), np.array([probability * (10 / 11) + (1-probability) * 10])
    if name == 'outlier':
        components = []
        for indicators in itertools.product((False, True), repeat=9):
            mean, variance, log_weight = 0., 100., 0.
            for observation, inlier in zip(range(1, 10), indicators):
                variance += 1
                log_weight += math.log(.9 if inlier else .1)
                observation_variance = variance + 1 if inlier else 100.
                residual = observation - mean if inlier else observation
                log_weight -= .5 * (math.log(2 * math.pi * observation_variance) + residual**2 / observation_variance)
                if inlier:
                    gain = variance / observation_variance
                    mean += gain * residual
                    variance *= 1 - gain
            components.append((mean, variance, log_weight))
        means, variances, logs = np.array(components).T
        weights = np.exp(logs - logs.max()); weights /= weights.sum()
        mean = weights @ means
        # print("mean", mean)
        # print("variance", variances)
        # print("weights", weights)
        return np.array([mean]), np.array([weights @ (variances + means**2) - mean**2])
    if name == 'slam':
        values, weights = [], []
        for cell0, cell1, slip0, slip1 in itertools.product((0, 1), repeat=4):
            x0 = 0 if slip0 else 1
            x1 = x0 if slip1 else max(0, x0 - 1)
            cells = (cell0, cell1)
            weight = .25 * (.1 if slip0 else .9) * (.1 if slip1 else .9)
            weight *= (.1 if cells[x0] else .9) * (.1 if cells[x1] else .9)
            values.append([cell0, cell1, x0, x1]); weights.append(weight)
        values = np.array(values); weights = np.array(weights); weights /= weights.sum()
        mean = weights @ values
        return mean, weights @ (values**2) - mean**2
    if name == 'coin':
        return np.array([.8]), np.array([16 / 1100])  # Beta(8, 2)
    if name == 'gaussian-gaussian':
        return np.array([30 / 11]), np.array([10 / 11])
    if name == 'gtree':
        variance = 1 / (1 / 100 + 1 / 1010 + 1 / 1000)
        return np.array([variance * (2.3 / 1010 - 55 / 1000)]), np.array([variance])
    if name == 'kalman':
        mean, variance = 0., 100.
        for observation in range(1, 10):
            variance += 1
            gain = variance / (variance + 1)
            mean += gain * (observation - mean)
            variance *= 1 - gain
        return np.array([mean]), np.array([variance])
    if name == 'wheels':
        # Joint Gaussian conditioning gives smoothed means for every returned state.
        observations = np.array([[201.79897693, -32.779256728], [206.423095274, -31.9258541027],
                                 [207.900486767, -31.5701198412], [210.298303206, -32.0217383347],
                                 [206.010088006, -31.3001236526], [207.230561559, -35.9919009388],
                                 [207.802456076, -34.6068500903]])
        n = len(observations)
        covariance = 2500 + np.minimum.outer(np.arange(n), np.arange(n))
        prior = np.kron(covariance, np.eye(2))  # interleaved velocity, omega
        h = np.kron(np.eye(n), np.array([[1., -2.], [1., 2.]]))
        noise = np.diag(np.tile([1., .95], n))
        gain = np.linalg.solve(h @ prior @ h.T + noise, h @ prior).T
        mean = gain @ observations.ravel()
        posterior = prior - gain @ h @ prior
        order = np.r_[np.arange(0, 2*n, 2), np.arange(1, 2*n, 2)]
        return mean[order], np.diag(posterior)[order]
    raise ValueError(name)


def measure(name, count, seed):
    # Match the normal CLI evaluation boundary, with a monotonic timer.
    sys.path.insert(0, str(ROOT))
    from siren.parser import parse_program
    from siren.evaluate import SMC
    from siren.inference.ssi import SSIState
    from compare_modes import flatten
    sys.setrecursionlimit(5000)
    program = parse_program((EXAMPLES / f'{name}.si').read_text())
    start = time.perf_counter_ns()
    value, _ = SMC().infer(program, SSIState, str(EXAMPLES), seed=None, n_particles=count)
    elapsed = (time.perf_counter_ns() - start) / 1e6
    estimate = np.array([float(x) for group in flatten(value.v) for x in group])

    truth, variance = reference(name)
    print(
        f"model={name}, particles={count}\n"
        f"estimate={estimate.tolist()}\n"
        f"truth={truth.tolist()}\n"
        f"posterior_variance={variance.tolist()}\n"
        f"standardized_mse={float(np.mean((estimate - truth) ** 2 / variance))}",
        file=sys.stderr,
        flush=True,
    )
    if estimate.shape != truth.shape or not np.all(np.isfinite(estimate)):
        raise ValueError(f'Invalid estimate shape/value: {estimate}')
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError(f'Invalid inference time: {elapsed}')
    # print("standard mse:", float(np.mean((estimate-truth)**2 / variance)))
    # print("elapsed:", elapsed)
    return dict(time_ms=elapsed, standardized_mse=float(np.mean((estimate-truth)**2 / variance)), estimate=estimate.tolist())


def write_summary_report(rows, summary, output):
    """Generate the human-readable summary from the same measured CSV data."""
    statuses = {status: sum(r['status'] == status for r in rows)
                for status in ('ok', 'error', 'timeout', 'pending', 'running')}
    lines = [
        '# Summary of all examples', '',
        'Counts and repetitions below describe the actual measured CSV. '
        'Default settings use five repetitions at each particle count; '
        'wheels uses 1, 2, and 3 particles, and other models use '
        '1, 10, 100, 1,000, and 10,000.', '',
        f"Runs: {statuses['ok']} successful, {statuses['error']} failed, "
        f"{statuses['timeout']} timed out, {statuses['pending']} pending, "
        f"{statuses['running']} running.", '',
        'Runtime is the median across successful repetitions. Standardized RMSE '
        'is the square root of mean standardized squared error across dimensions '
        'and successful repetitions, using exact posterior marginal variances.', '',
        '| Example | Particles | Successful runs | Failed / timed out | Pending / running | Median runtime (ms) | Standardized RMSE |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in summary:
        runtime = f"{float(row['median_time_ms']):,.3f}" if row['median_time_ms'] != '' else '—'
        error = f"{float(row['standardized_rmse']):.6g}" if row['standardized_rmse'] != '' else '—'
        lines.append(f"| {row['benchmark']} | {int(row['particles']):,} | "
                     f"{row['successful_runs']} | {row['failed_runs']} | {row['pending_runs']} | "
                     f"{runtime} | {error} |")
    lines += [
        '', 'Timing includes particle creation, interpreter execution, and weighted '
        'mean evaluation; excludes imports, parsing, and process startup. '
        'The supplied libsiren benchmark excludes particle creation.', '',
        'The existing interpreter uses the same initial seed for every particle. '
        'Flat SLAM/outlier error curves reflect that behavior; this benchmark preserves it.', '',
        'Errors below 1e-12 are shown at 1e-12 in plots only. CSVs preserve actual error values.',
    ]
    (output / 'all_examples_summary.md').write_text('\n'.join(lines) + '\n')


def plot(csv_path, output):
    # Failed runs can store tracebacks larger than the default CSV field limit.
    csv.field_size_limit(sys.maxsize)
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    save_rows(output / 'wheels_raw.csv', [r for r in rows if r['benchmark'] == 'wheels'])
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    def panel(name, ax):
        points = []
        for count in sorted({int(r['particles']) for r in rows if r['benchmark'] == name}):
            valid = [r for r in rows if r['benchmark'] == name and int(r['particles']) == count and r['status'] == 'ok']
            if valid:
                points.append((count, statistics.median(float(r['time_ms']) for r in valid),
                               max(1e-12, math.sqrt(statistics.mean(float(r['standardized_mse']) for r in valid)))))
        if points:
            ax.plot([p[1] for p in points], [p[2] for p in points], 'o-', color='tab:green', label='Siren smc-ssi')
            for p in [points[0]] + ([points[-1]] if len(points) > 1 else []):
                ax.annotate(str(p[0]), (p[1], p[2]), xytext=(5, 8), textcoords='offset points')
            ax.legend(loc='center right')
        else:
            ax.text(.5, .5, 'No successful runs\nSee raw.csv for failures', ha='center', transform=ax.transAxes)
        ax.set(xscale='log', yscale='log', title=name, xlabel='Median inference time (ms)', ylabel='Standardized mean RMSE')
        ax.set_ylim(3e-13, max(10., max((p[2] for p in points), default=1.) * 5))
        ax.margins(x=.12)
        ax.grid(alpha=.2)
        failed = sum(r['status'] in ('error', 'timeout') for r in rows if r['benchmark'] == name)
        pending = sum(r['status'] in ('pending', 'running') for r in rows if r['benchmark'] == name)
        if pending:
            ax.text(.02, .08, f'{pending} run(s) pending / running', transform=ax.transAxes, fontsize=8)
        if failed:
            ax.text(.02, .04, f'{failed} failed / timed out run(s)',
                    transform=ax.transAxes, fontsize=8)
    for name, ax in zip(MODELS, axes.flat):
        panel(name, ax)
        individual, single = plt.subplots(figsize=(7, 5))
        panel(name, single)
        individual.tight_layout()
        individual.savefig(output / f'{name}.png', dpi=180)
        plt.close(individual)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0, .95, 'Endpoint labels denote particle counts.\nErrors below 1e-12 are displayed at 1e-12.\n\nRMSE uses exact posterior marginal standard deviations.\nTiming includes particle creation, interpreter evaluation,\nand weighted mean evaluation; excludes startup, imports, and parsing.\n\nOriginal .si examples and annotations are unchanged.\nExact references match the libsiren benchmark protocol.\nFailures and timeouts are recorded in raw.csv.', va='top', linespacing=1.6)
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['benchmark', 'particles', 'successful_runs', 'failed_runs', 'pending_runs', 'median_time_ms', 'standardized_rmse'])
        writer.writeheader()
        for name in MODELS:
            for count in sorted({int(r['particles']) for r in rows if r['benchmark'] == name}):
                group = [r for r in rows if r['benchmark'] == name and int(r['particles']) == count]
                valid = [r for r in group if r['status'] == 'ok']
                writer.writerow(dict(benchmark=name, particles=count, successful_runs=len(valid), failed_runs=sum(r['status'] in ('error', 'timeout') for r in group),
                                     pending_runs=sum(r['status'] in ('pending', 'running') for r in group),
                                     median_time_ms=statistics.median(float(r['time_ms']) for r in valid) if valid else '',
                                     standardized_rmse=math.sqrt(statistics.mean(float(r['standardized_mse']) for r in valid)) if valid else ''))
    with (output / 'summary.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        summary = list(reader)
        wheels_summary = [r for r in summary if r['benchmark'] == 'wheels']
        with (output / 'wheels_summary.csv').open('w', newline='') as target:
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(wheels_summary)
    write_summary_report(rows, summary, output)
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'siren_runtime_error.{extension}', dpi=180)
    plt.close(fig)


def run_timeout(name, seconds):
    """Wheels runs to completion; other models retain their configured limit."""
    return None if name == 'wheels' else seconds


def save_rows(csv_path, rows):
    temporary = csv_path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmarks', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--particles', nargs='+', type=int, default=DEFAULT_PARTICLES)
    parser.add_argument('--wheels-particles', nargs='+', type=int, choices=range(1, 11), default=WHEELS_PARTICLES, help='Wheels particle counts (default: 1, 2, and 3)')
    parser.add_argument('--all-particles', action='store_true', help='Every integer from 1 through 10000')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--timeout', type=float, default=120, help='Seconds per repetition for models other than wheels (wheels has no timeout)')
    parser.add_argument('--output', type=Path, default=ROOT / 'examples' / 'libsiren_comparison_output')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Keep successful CSV rows and rerun missing or unsuccessful selected runs')
    parser.add_argument('--worker', nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        name, count, seed = args.worker
        print(json.dumps(measure(name, int(count), int(seed))))
        return
    counts = list(range(1, 10001)) if args.all_particles else sorted(set(args.particles))
    if args.repeats < 1 or args.timeout <= 0 or any(p < 1 or p > 10000 for p in counts):
        parser.error('Require positive repeats/timeout and particle counts in 1..10000')
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / 'raw.csv'
    if not args.plot_only:
        rows = []
        if args.resume and csv_path.exists():
            with csv_path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
        if args.resume:
            # Remove superseded wheels configurations from the active CSV.
            rows = [r for r in rows if r['benchmark'] != 'wheels' or
                    (int(r['particles']) in args.wheels_particles and int(r['repeat']) < args.repeats)]
        indices = {(r['benchmark'], int(r['particles']), int(r['repeat']), int(r['seed'])): i
                   for i, r in enumerate(rows)}
        jobs = []
        for name in args.benchmarks:
            for count in (sorted(set(args.wheels_particles)) if name == 'wheels' else counts):
                for repeat in range(args.repeats):
                    seed = args.seed + repeat * 10000
                    key = (name, count, repeat, seed)
                    index = indices.get(key)
                    if index is not None and rows[index]['status'] == 'ok':
                        continue
                    row = dict(benchmark=name, particles=count, repeat=repeat, seed=seed,
                               status='pending', error='Awaiting measurement')
                    if index is None:
                        index = len(rows)
                        indices[key] = index
                        rows.append(row)
                    else:
                        rows[index] = row
                    jobs.append(index)
        config_path = args.output / 'config.json'
        config = json.loads(config_path.read_text()) if args.resume and config_path.exists() else {}
        config.update(benchmarks=list(dict.fromkeys(r['benchmark'] for r in rows)),
                      particles=sorted({int(r['particles']) for r in rows}),
                      repeats=args.repeats, seed=args.seed, timeout=args.timeout,
                      wheels_timeout=None,
                      particles_by_benchmark={name: sorted({int(r['particles']) for r in rows if r['benchmark'] == name})
                                              for name in dict.fromkeys(r['benchmark'] for r in rows)})
        config_path.write_text(json.dumps(config, indent=2))
        save_rows(csv_path, rows)
        plot(csv_path, args.output)
        for index in jobs:
            row = rows[index]
            name, count, repeat, seed = (row[k] for k in ('benchmark', 'particles', 'repeat', 'seed'))
            row.update(status='running', error='Measurement in progress')
            save_rows(csv_path, rows)
            print(f'{name}: particles={count} repeat={int(repeat)+1} running (timeout={run_timeout(name, args.timeout)})', flush=True)
            try:
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), '--worker', name, str(count), str(seed)],
                    capture_output=True, text=True, timeout=run_timeout(name, args.timeout), check=True)
                row.update(json.loads(result.stdout), status='ok', error='')
            except subprocess.TimeoutExpired:
                row.update(status='timeout', error=f'Exceeded {args.timeout}s')
            except subprocess.CalledProcessError as exc:
                row.update(status='error', error=exc.stderr.strip())
            save_rows(csv_path, rows)
            plot(csv_path, args.output)
            print(f'{name}: particles={count} repeat={int(repeat)+1} {row["status"]}', flush=True)
    plot(csv_path, args.output)
    print(f'Results: {args.output.resolve()}')
    print('Generated: raw.csv, summary.csv, all_examples_summary.md, wheels_raw.csv, '
          'wheels_summary.csv, individual model PNGs, and siren_runtime_error.png/.pdf')


if __name__ == '__main__':
    main()
