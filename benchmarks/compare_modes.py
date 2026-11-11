"""Benchmark examples as written, then plot runtime and error from the CSV."""
import argparse
import ast
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import statistics
import textwrap
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['example', 'plan_id', 'annotations', 'handler', 'method',
          'particles', 'repeat', 'seed', 'runtime_seconds', 'variable', 'metric',
          'error', 'reference_scale', 'reference_kind', 'reference_particles', 'status', 'message']


def invoke(source, args, particles, seed):
    command = [sys.executable, '-m', 'siren.siren', str(source),
               '--handler', args.handler, '--method', args.method,
               '--particles', str(particles), '--samples', str(particles),
               '--warmup', str(args.warmup), '--seed', str(seed)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-4000:])
    return result.stdout


def flatten(value):
    if isinstance(value, tuple):
        return [item for part in value for item in flatten(part)]
    return [value if isinstance(value, list) else [value]]


def parse_output(stdout, truths):
    lines = stdout.splitlines()
    runtime = float(lines[lines.index('===== Evaluation Time =====') + 1])
    values = flatten(ast.literal_eval(lines[lines.index('===== Result =====') + 1]))
    if not math.isfinite(runtime) or runtime < 0 or len(values) != len(truths):
        raise ValueError('Invalid runtime or output variable count')
    errors = []
    for (name, expected), actual in zip(truths, values):
        if not expected or len(expected) != len(actual):
            raise ValueError(f'{name}: expected {len(expected)} values, got {len(actual)}')
        error = sum((float(a) - float(b)) ** 2 for a, b in zip(expected, actual)) / len(expected)
        if not math.isfinite(error):
            raise ValueError(f'{name}: non-finite error')
        errors.append((name, error))
    return runtime, errors


def empirical_reference(results):
    """Use only existing highest-particle runs; never launch reference inference."""
    largest = max(row['particles'] for row, values in results)
    values = [value for row, value in results if row['particles'] == largest]
    shape = [len(v) for v in values[0]]
    if any([len(v) for v in value] != shape for value in values):
        raise ValueError('Inconsistent reference output shapes')
    truths = [(f'output_{i}', [statistics.mean(float(value[i][j]) for value in values)
                             for j in range(size)]) for i, size in enumerate(shape)]
    return largest, truths


def run(args, csv_path):
    directory = ROOT / 'examples'
    examples = args.examples or sorted(p.stem for p in directory.glob('*.si'))
    references = json.loads(args.references.read_text()) if args.references else {}
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for example in examples:
            source = directory / f'{example}.si'
            if not source.is_file():
                raise ValueError(f'Example not found: {source}')
            measured = []
            for particles in args.particles:
                for repeat in range(args.repeats):
                    seed = args.seed + repeat
                    row = dict(example=example, plan_id=source.name,
                               annotations='original example', handler=args.handler,
                               method=args.method, metric='mse', particles=particles,
                               repeat=repeat, seed=seed)
                    print(f'{example}: particles={particles}, repeat={repeat}', flush=True)
                    try:
                        stdout = invoke(source, args, particles, seed)
                        lines = stdout.splitlines()
                        values = flatten(ast.literal_eval(lines[lines.index('===== Result =====') + 1]))
                        # Validate numerical output and runtime before using it as reference.
                        runtime, _ = parse_output(stdout, [(str(i), v) for i, v in enumerate(values)])
                        row['runtime_seconds'] = runtime
                        measured.append((row, values))
                    except (RuntimeError, ValueError, SyntaxError) as exc:
                        writer.writerow(dict(row, status='failed', message=str(exc)))
                        stream.flush()
            if not measured:
                continue
            if example in references:
                truths = list(references[example].items())
                reference_kind, reference_particles = 'provided', ''
            else:
                reference_particles, truths = empirical_reference(measured)
                reference_kind = 'empirical_measured_runs'
            for row, values in measured:
                row.update(reference_kind=reference_kind, reference_particles=reference_particles)
                try:
                    if len(values) != len(truths):
                        raise ValueError('Reference output variable count mismatch')
                    outputs = []
                    for (variable, expected), actual in zip(truths, values):
                        if not expected or len(expected) != len(actual):
                            raise ValueError(f'{variable}: reference output length mismatch')
                        error = statistics.mean((float(a) - float(b)) ** 2 for a, b in zip(expected, actual))
                        scale = math.sqrt(statistics.mean(float(v) ** 2 for v in expected)) or 1.0
                        if not math.isfinite(error) or not math.isfinite(scale):
                            raise ValueError('Non-finite error or reference scale')
                        outputs.append(dict(row, variable=variable, error=error,
                                            reference_scale=scale, status='ok'))
                    writer.writerows(outputs)
                except (ValueError, TypeError) as exc:
                    writer.writerow(dict(row, status='failed', message=str(exc)))
                stream.flush()


def summarize(rows):
    """Average normalized output RMSE within a run, then average across repeats.

    Runtime appears once per output in the raw CSV, so deduplicate by run first.
    """
    runs = defaultdict(list)
    for row in rows:
        if row['status'] != 'ok':
            continue
        key = tuple(row[k] for k in ('example', 'handler', 'method', 'particles', 'repeat', 'seed'))
        runs[key].append(row)
    groups = defaultdict(list)
    for key, outputs in runs.items():
        errors = [math.sqrt(float(r['error'])) / float(r.get('reference_scale') or 1) for r in outputs]
        groups[key[:4]].append((float(outputs[0]['runtime_seconds']) * 1000, statistics.mean(errors)))
    return [dict(zip(('example', 'handler', 'method', 'particles'), key),
                 median_runtime_ms=statistics.median(t for t, e in values),
                 mean_rmse=statistics.mean(e for t, e in values), runs=len(values))
            for key, values in sorted(groups.items())]


def plot(csv_path, output):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'siren-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    fields = ['example', 'handler', 'method', 'particles', 'median_runtime_ms', 'mean_rmse', 'runs']
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    standardized = all(r.get('reference_scale') for r in rows if r['status'] == 'ok')
    ylabel = 'Standardized mean RMSE' if standardized else 'Mean RMSE (unscaled legacy CSV)'

    def panel(ax, example):
        for color in ['tab:green']:
            configurations = sorted({(r['handler'], r['method']) for r in summary
                                     if r['example'] == example})
            for handler, method in configurations:
                points = sorted([r for r in summary if r['example'] == example
                                 and r['handler'] == handler and r['method'] == method],
                                key=lambda r: int(r['particles']))
                ax.plot([max(r['median_runtime_ms'], 1e-12) for r in points],
                        [max(r['mean_rmse'], 1e-12) for r in points], 'o-', color=color,
                        label=f'Siren {handler}-{method}', markersize=5)
                endpoints = [points[0]] if len(points) == 1 else [points[0], points[-1]]
                for point in endpoints:
                    ax.annotate(point['particles'],
                                (max(point['median_runtime_ms'], 1e-12), max(point['mean_rmse'], 1e-12)),
                                xytext=(6, 9),
                                textcoords='offset points', fontsize=8)
        ax.set(xscale='log', yscale='log', title=example,
               xlabel='Median inference time (ms)', ylabel=ylabel)
        ax.grid(alpha=.2, which='major')
        if ax.lines:
            ax.legend(fontsize=8, loc='best')
            ax.margins(x=.2, y=.2)
        else:
            ax.text(.5, .5, 'No successful runs', ha='center', transform=ax.transAxes)
        failed = sum(r['status'] != 'ok' for r in rows if r['example'] == example)
        if failed:
            ax.text(.02, .02, f'{failed} failed/unavailable run(s)', transform=ax.transAxes, fontsize=8)

    examples = sorted({r['example'] for r in rows})
    notes = ('Endpoint labels: particle counts. Errors below 1e-12 are displayed at 1e-12.\n'
             'Error: mean across outputs and repetitions of RMSE / reference RMS (zero RMS uses 1).\n'
             'Examples run as written with their original annotations.\n'
             'Empirical references reuse the largest measured particle count; they are not ground truth. Timing includes mean evaluation.')
    if not standardized:
        notes += '\nLegacy CSV lacks reference scales: errors are unscaled RMSE. Rerun for standardized errors.'
    for example in examples:
        fig, ax = plt.subplots(figsize=(7, 5))
        panel(ax, example)
        fig.tight_layout()
        fig.savefig(output / f'{example}.png', dpi=180)
        plt.close(fig)
    if examples:
        columns = 3
        nrows = math.ceil((len(examples) + 1) / columns)
        fig, axes = plt.subplots(nrows, columns, figsize=(18, 4.8 * nrows), squeeze=False)
        for ax, example in zip(axes.flat, examples):
            panel(ax, example)
        extras = list(axes.flat)[len(examples):]
        for ax in extras:
            ax.axis('off')
        notes = '\n\n'.join(textwrap.fill(line, width=48) for line in notes.splitlines())
        extras[0].text(0, .9, notes, transform=extras[0].transAxes, va='top', fontsize=10, wrap=True)
        fig.tight_layout()
        fig.savefig(output / 'siren_comparison.png', dpi=180)
        fig.savefig(output / 'siren_comparison.pdf')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', nargs='+', help='Names from examples/ without .si; default: all examples')
    parser.add_argument('--particles', nargs='+', type=int, default=[1, 10, 100, 1000])
    parser.add_argument('--references', type=Path, help='JSON: example -> output name -> list of reference values')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--handler', choices=['smc', 'mh'], default='smc')
    parser.add_argument('--method', choices=['ssi', 'ds', 'bp'], default='ssi')
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--output', type=Path, default=ROOT / 'examples' / 'comparison_output')
    parser.add_argument('--plot-only', type=Path, metavar='CSV', help='Regenerate plots from an existing CSV')
    args = parser.parse_args()
    if args.repeats < 1 or min(args.particles) < 1 or args.warmup < 0:
        parser.error('repeats and particles must be positive; warmup must be nonnegative')
    csv_path = args.plot_only or args.output / 'runtime_errors.csv'
    if not args.plot_only:
        run(args, csv_path)
    plot(csv_path, args.output / 'plots')
    print(f'CSV: {csv_path}\nPlots: {args.output / "plots"}')


if __name__ == '__main__':
    main()
