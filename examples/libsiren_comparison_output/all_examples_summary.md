# Summary of all examples

Counts and repetitions below describe the actual measured CSV. Default settings use five repetitions at each particle count; wheels uses 1, 2, and 3 particles, and other models use 1, 10, 100, 1,000, and 10,000.

Runs: 5 successful, 0 failed, 0 timed out, 0 pending, 0 running.

Runtime is the median across successful repetitions. Standardized RMSE is the square root of mean standardized squared error across dimensions and successful repetitions, using exact posterior marginal variances.

| Example | Particles | Successful runs | Failed / timed out | Pending / running | Median runtime (ms) | Standardized RMSE |
|---|---:|---:|---:|---:|---:|---:|
| outlier | 1 | 5 | 0 | 0 | 2.205 | 0.026021 |

Timing includes particle creation, interpreter execution, and weighted mean evaluation; excludes imports, parsing, and process startup. The supplied libsiren benchmark excludes particle creation.

The existing interpreter uses the same initial seed for every particle. Flat SLAM/outlier error curves reflect that behavior; this benchmark preserves it.

Errors below 1e-12 are shown at 1e-12 in plots only. CSVs preserve actual error values.
