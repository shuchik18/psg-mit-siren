# Siren
Siren is a first-order functional probabilistic programming language, implemented with the hybrid inference interface, with support for inference plans. Distributions encoding annotations can be added to random variables to select the representation of the variable's distribution to be used during inference. Siren is also equipped with the inference plan satisfiability analysis, which statically infers if the annotated inference plan is satisfiable. 

The Siren interpreter, including the inference plan satisfiability analysis, is implemented with semi-symbolic inference, delayed sampling, and SMC with belief propagation, with support for particle filtering (Sequential Monte Carlo) and Metropolis-Hastings as the approximate inference backend. It can be extended with other hybrid inference algorithms that can implement the hybrid inference interface. 

## Install
Clone the repository:
```
git clone https://github.com/psg-mit/siren
cd siren
```

### Docker (Recommended)
We also provide a Docker file to build an image. The artifact was tested with Docker Desktop v4.30.0. 
```bash
docker build -t siren .
docker run -it siren
```

### From Source
This software was tested on M1 MacBook and requires Python >= 3.10. To install dependencies:
```bash
pip install -e .
```

## Quickstart
Here is an example of a Siren modeling a Kalman filter:
```ocaml
val make_observations = fun (yobs, xs) ->
  let pre_x = List.hd(xs) in
  let sample x <- gaussian(pre_x, 1.) in
  let () = observe(gaussian(x, 1.), yobs) in

  let () = resample() in
  cons(x, xs)
in

let data = List.range(1, 101) in

let x0 = 0 in
let xs = fold(make_observations, data, [x0]) in
List.rev(xs)
```
The program iterates over a range of values from 1 to 100 (inclusive) as the observed data. The variable `x` is annotated with `sample` to indicate `x` should be represented as samples during inference. To indicate `x` should be represented as a symbolic distribution, replace `sample` with `symbolic`.

To run the inference plan satisfiability analysis, and execute the program if the analysis succeeds:
```bash
siren path/to/program.si -m {method} -p {particles} --analyze
```

For example, to run the analysis and execute the program using the semi-symbolic inference algorithm with 100 particles:
```bash
siren examples/kalman.si -m ssi -p 100 --analyze
```
The analysis will throw an `AnalysisViolatedAnnotationError` and abort without running the program if the annotated inference plan is not satisfiable.

To execute without the analysis:
```bash
siren path/to/program.si -m {method}
```

To run the analysis only:
```bash
siren path/to/program.si -m {method} --analyze-only
```

## Tests
To run the test suite for a quick check everything works:
```bash
python -m pytest tests/
```

## Smoke Test
To do a smoke test that the benchmarks can run correctly:
```bash
cd benchmarks/
python harness.py --output output_kicktires kicktires
```
This should take 1 hour.

The smoke test is configured to only run n=1 iterations for less particles, so the figures will look very distorted but that is expected. To check the visualization script will generate plots:
```bash
python visualize.py --output output_kicktires --example 
python visualize.py --output output_kicktires --example --task timestep
```
These files will be generated 
- `benchmarks/examplegood/output_kicktires/smc_ssi_example_143.png`
- `benchmarks/examplegood/output_kicktires/smc_ssi_example_time_43.png`
- `benchmarks/examplebad/output_kicktires/smc_ssi_example_time_43.png`

```bash
python visualize.py --output output_kicktires --task analysis_table
```
This will output the analysis results table in Table 1 and Table 19, as well as the runtime of the analysis.

```bash
python visualize.py --output output_kicktires --task plot
```
The plots will be located at `benchmarks/{BENCHMARK}/output_kicktires/{HANDLER}_{METHOD}_particles.png` for each BENCHMARK and METHOD and HANDLER.

To check the runtime and accuracy statistics aggregation works correctly:
```bash
 # Compares runtime of inference plans to reach the same accuracy
python visualize.py --output output_kicktires --task compare_time
# Compare accuracy of inference plans given fixed runtime
python visualize.py --output output_kicktires --task compare_accuracy 
```

## Benchmarking
The experiments from the paper were conducted on a 60-vCPU Intel Xeon Cascade Lake (up to 3.9 GHz) node with 240 GB RAM. The full set of experiments in the paper takes about 30 days of computation. The experiments can run on a general-purpose computer as well, requiring only enough computation time. 

### Main Paper Results
Due to the long amount of time needed to compute the full set of benchmarks from the paper, which uses `n=100` iterations per particle setting for each benchmark and method, to only replicate the trends of the main paper figures:
```bash
cd benchmarks/
python harness.py artifact-eval
```
This executes the example for Figure 4 and Figure 5 for `n=10` and the programs for Figure 16 for `n=5`. This will take ~4-5 hours.

Then, to visualize the results for Section 2 Figure 4:
```bash
python visualize.py --example
```
The plot will be located at `benchmarks/examplegood/output/ssi_example_143.png`.

For Figure 5:
```bash
python visualize.py --example --task timestep
```
These commands generate the figures for Section 2. The plots will be located at `benchmarks/examplegood/output_kicktires/smc_ssi_example_time_43.png` and `benchmarks/examplebad/output_kicktires/smc_ssi_example_time_43.png`.

Then, to visualize the results for Section 5 Figure 16:
```bash
python visualize.py --task plot -b outlier noise -m ssi
```
The plot will be located at `benchmarks/outlier/output/smc_ssi_particles.png` and `benchmarks/noise/output/smc_ssi_particles.png`.

Then, to produce Section 5 Table 1:
```bash
python visualize.py --task analysis_table --handlers smc
```

### Full Replication
To perform the full replication of the figures in the paper:
```bash
cd benchmarks/
python harness.py full-replication
python visualize.py --example
python visualize.py --task analysis_table
```

To compute the overall speedup and accuracy comparison from Section 5, and the tables in Appendix F:
```bash
python visualize.py --task compare_time
python visualize.py --task compare_accuracy
```

To visualize the plots of Appendix F and Appendix G:
```bash
python visualize.py --task plot
```
The plot will be located at `benchmarks/{BENCHMARK}/output/{HANDLER}_{METHOD}_particles.png` for each BENCHMARK and METHOD and HANDLER.

This will take ~30 days.

## Syntax
The Siren syntax can be found in `siren/parser.py`.

## Guide
To view a description of the source files in this repository and instructions on how to extend Siren, please see [DESCRIPTION.md](DESCRIPTION.md).

### Example runtime/error harness

From the repository root, with the project Python environment active:

```sh
python benchmarks/compare_modes.py
```

The harness discovers all `.si` programs in **examples/**. By default, each
example runs with 1, 10, 100, and 1,000 particles and 10 repetitions per
particle count, with no timeout. It writes
`examples/comparison_output/runtime_errors.csv` and plots in
`examples/comparison_output/plots/`. `siren_comparison.png` and `.pdf` contain
one panel per example, with one normal SIREN series on logarithmic
axes, matching the comparison figure layout. Individual example PNGs and
`plots/summary.csv` are also generated.

The x axis is median inference time in milliseconds across repetitions. The
y axis is standardized mean RMSE: for each output, take the square root of
MSE and divide by the RMS magnitude of its reference values (use 1 if that
magnitude is zero), then average across outputs and repetitions. This explicit
normalization is used because the supplied screenshot does not define its
standardization formula. Errors below 1e-12 are displayed at 1e-12; the CSV
preserves actual values. Labels on curve endpoints show particle counts.
Runtime is SIREN evaluation time, including result mean evaluation and excluding
startup and parsing. Default output labels are `output_0`, `output_1`, etc.
Legacy CSVs without `reference_scale` produce explicitly labeled unscaled RMSE.

The normal command generates curves automatically using the default particle
counts. To customize the counts:

```sh
python benchmarks/compare_modes.py --particles 1 10 100 1000 10000 --repeats 10
```

Each example runs directly from its original `.si` file with its existing
annotations, just like a normal SIREN invocation. The harness does not generate
sample or symbolic variants or modify source files.

The examples do not ship ground truth. The harness performs **no additional
reference runs**. By default, the reference is the mean of successful measured
runs at the largest configured particle count. The CSV labels this
`empirical_measured_runs`. This measures deviation from an empirical reference,
not accuracy against exact ground truth; it can underestimate error because the
same runs contribute to the reference. With one repetition, the largest-count
run has zero error by construction.

For errors against known expected outputs, use `--references reference.json`:

```json
{"coin": {"xt": [0.8]}, "gaussian-gaussian": {"x": [2.727272727272727]}}
```

Reference entries follow flattened tuple output order, with scalars represented
by one-element lists. Examples missing from the JSON use measured-run references.
Failures are recorded in the CSV. Each new run overwrites its output CSV.

```sh
python benchmarks/compare_modes.py --examples coin kalman --particles 1 4 16 --repeats 3
python benchmarks/compare_modes.py --plot-only examples/comparison_output/runtime_errors.csv
```

Options include `--output PATH`, `--method ssi|ds|bp`, `--handler smc|mh`,
`--warmup 0`, and `--seed 0`. Runs have no timeout. Successive repetitions use successive seeds.

### Matching the libsiren benchmark

Running this command from the repository root automatically creates all CSVs,
plots, and the detailed summary; no separate export command is needed:

```sh
venv/bin/python benchmarks/compare_libsiren.py
```

Defaults: five repetitions at each count; wheels uses **1, 2, and 3** particles
without a timeout; the other seven models use **1, 10, 100, 1000, and 10000**.
The output directory is `examples/libsiren_comparison_output/`, containing:

- `raw.csv` and `summary.csv` for all examples.
- `wheels_raw.csv` and `wheels_summary.csv` for wheels only.
- `all_examples_summary.md`, generated from the actual run data.
- `siren_runtime_error.png` and `.pdf`, plus an individual PNG per model.
- `config.json` with the measurement settings.

Exports update after every completed repetition and at the end. Use `--resume`
to reuse successful measurements or `--plot-only` to regenerate every export
from the existing `raw.csv` without running inference.


```sh
venv/bin/python benchmarks/compare_libsiren.py
```

Runs the eight models from the supplied libsiren benchmark with Original Siren
SMC/SSI, preserving their `.si` sources and annotations. Defaults use 1, 10, 100, 1,000, and 10,000 particles for other models
and 1, 2, and 3 for wheels; five repetitions;
seed `1234 + repeat * 10000`; 120-second timeout per isolated worker for models other than wheels; wheels has no timeout.
No additional reference inference runs are performed.

Results are in `examples/libsiren_comparison_output/`: `raw.csv`, `summary.csv`,
`config.json`, individual model PNGs, and `siren_runtime_error.png` / `.pdf`.
Error is the square root of mean squared posterior-mean error divided by exact
posterior marginal variance, averaged over dimensions and repetitions. The
analytical references are copied from the supplied benchmark, including exact
mixture enumeration for outlier and SLAM and Gaussian smoothing for wheels.
Timing includes interpreter inference, particle creation, and weighted posterior
mean evaluation; it excludes imports, parsing, and process startup. The supplied
compiled benchmark excludes particle creation, so the timing boundaries differ
by that setup cost. Timeouts and failures remain visible in raw results.

```sh
venv/bin/python benchmarks/compare_libsiren.py --benchmarks coin kalman --particles 1 10 100 --repeats 3
venv/bin/python benchmarks/compare_libsiren.py --plot-only
```

`--output PATH`, `--seed`, `--timeout`, and `--all-particles` are also supported.
A new measurement run overwrites its output files. Use `--resume` to retain successful rows, rerun unfinished selected measurements, and preserve other models. CSVs and plots are checkpointed after each completed repetition.

Wheels now uses particle counts **1, 2, and 3**, with five repetitions at each
count and no timeout. Other models retain their original particle grids.
To resume the selected wheels sweep:

```sh
venv/bin/python benchmarks/compare_libsiren.py --benchmarks wheels --wheels-particles 1 2 3 --repeats 5 --resume
```

`--wheels-particles` accepts counts from 1 through 10 and defaults to 1, 2, and 3.
Resuming drops superseded wheels counts from the active CSV and retains all
other models. The previous active CSV was archived as
`raw_before_wheels_1_10.csv`. Outputs include combined `raw.csv` / `summary.csv`
and wheels-only `wheels_raw.csv` / `wheels_summary.csv`, plus `wheels.png`.
Pending and running measurements are labeled until actual results are available.
CSV files and plots update after every completed repetition.

The existing interpreter initializes all particles with the
same seed in `ProbState.__init__`; consequently SLAM and outlier estimates in
this seeded sweep are effectively unchanged as particle counts increase. This
harness preserves that interpreter behavior. Flat error curves therefore do
not demonstrate convergence from independent particles.
