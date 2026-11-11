# Proposed fixes for compare_libsiren.py

This document describes proposed changes only. The Python benchmark and Siren
interpreter have not been changed as part of this documentation task.

At the final inspection, `measure()` already passed `seed=None`. The replacement
below therefore documents the intended behavior; the report text still described
the old same-seed behavior and needs the documentation update in section 2.

## 1. Give particles independent random streams

Inside `measure()`, replace:

```python
value, _ = SMC().infer(program, SSIState, str(EXAMPLES), seed, n_particles=count)
```

with:

```python
# A numeric seed is reused for every particle by the existing interpreter.
# None lets each particle initialize its RNG independently from OS entropy.
value, _ = SMC().infer(
    program,
    SSIState,
    str(EXAMPLES),
    seed=None,
    n_particles=count,
)
```

This is a benchmark-only workaround. It matches the Miking comparison's Siren
adapter. It avoids duplicate particle random streams without editing
`siren/inference/interface.py`.

The tradeoff is reproducibility: the existing numeric `seed` field no longer
describes the random draws. It can remain a job identifier for the current CSV
and resume logic, but must be documented as such. Do not simply write `None`
into that field: the existing resume code parses it with `int()`.

## 2. Correct the generated explanation and configuration

In `write_summary_report()`, replace:

```python
'The existing interpreter uses the same initial seed for every particle. '
'Flat SLAM/outlier error curves reflect that behavior; this benchmark preserves it.'
```

with:

```python
'Particles use independently initialized random streams via seed=None. '
'The recorded seed identifies the job but does not reproduce its random draws.'
```

Clarify the command-line option:

```python
parser.add_argument(
    '--seed', type=int, default=1234,
    help='Base job identifier; particle RNGs use OS entropy',
)
```

Add an explicit policy to the existing `config.update(...)` call:

```python
rng_policy='independent-os-seeded-particles-v1',
```

If implementing a resume guard, reject existing CSVs whose configuration lacks
this policy or has a different policy, before reusing successful rows. This
prevents mixing measurements from the old and new sampling behavior.

## 3. Keep the error calculation

The existing per-run standardized MSE and its aggregation into standardized
RMSE are correct:

```python
# Per run:
standardized_mse = float(np.mean((estimate - truth)**2 / variance))

# Across successful repetitions at one particle count:
standardized_rmse = math.sqrt(statistics.mean(
    float(r['standardized_mse']) for r in valid
))
```

`variance` is the exact posterior marginal variance, not the variance of the
estimated mean. The plot shows error versus milliseconds; it does not divide
error or accuracy by runtime. Keep the plot floor of 1e-12 as display-only.

## 4. Regenerate measurements

After applying the Python changes, use a fresh output directory and omit
`--resume` and `--plot-only` for the first run. Replotting old CSVs cannot fix
their duplicated particle sampling. Failed, pending, and timed-out runs must
remain excluded from accuracy points.

For comparisons against CorePPL, also label the timing boundary: Siren includes
particle creation and weighted symbolic mean evaluation; the supplied CorePPL
inference timer excludes final weighted aggregation. The seed fix does not
remove that timing difference.

## Should gaussian-gaussian-hybrid's graph vary?

Its mean-error curve is expected to remain at zero or numerical roundoff, even
after particle seeding is fixed. Its runtime should generally grow with particle
count and will fluctuate between measurements.

The current model has `input = 0`, samples `b ~ Bernoulli(0.5)`, and uses
`x ~ Gaussian(0, 10)`, with Gaussian parameters written as mean and variance.

| Sampled branch | Conditional posterior for x | Conditional mean |
| --- | --- | --- |
| b is true | Gaussian(0, 10/11) after observing 0 with noise variance 1 | 0 |
| b is false | Gaussian(0, 10), because the observation is independent of x | 0 |

With SSI retaining x symbolically, every particle returns conditional mean 0
regardless of the sampled branch. Any normalized weighted mixture of those
means is also 0. The exact reference mean is 0, so the mean RMSE cannot detect
whether the sampled branch proportions approximate the posterior correctly.

The branch posterior probability is
`P(b=true | input=0) = (1/sqrt(11)) / (1/sqrt(11) + 1/10)`.
The two branches have different variances, but changing their weights does not
change their shared mean.

Therefore a flat hybrid mean-error curve does not prove the particles are
duplicated or that the full posterior is exact. The plotting floor displays
zero errors at 1e-12. SLAM and outlier are more useful checks of the seeding fix.
Their aggregate error should tend to improve with increasing particle budgets,
but finite repetitions do not guarantee monotonic curves.

To study hybrid convergence separately, one could measure posterior branch
probability or posterior second-moment error, or create a separate nonzero-input
experiment with a matching reference. These are additional experiments; do not
change the current input, reference, or metric merely to make its graph slope.
