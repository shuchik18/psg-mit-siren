# Original Siren / libsiren protocol benchmark

Eight original `.si` models; SMC/SSI. Other models use particle counts
1, 10, 100, 1000, 10000. Wheels uses only 1, 2, and 3, with five repetitions each.
Seeds: 1234, 11234, 21234, 31234, 41234. Other models have a 120-second timeout;
wheels runs without a timeout.

All 15 wheels measurements succeeded: five repetitions each at 1, 2, and 3
particles, with no timeout. The five earlier one-particle measurements were
retained, and the ten two/three-particle measurements were executed to completion.
All 190 active CSV records are successful, including the 175 retained runs for
other models. No pending, running, failed, or timeout rows remain in the active
CSV. Superseded wheels configurations are excluded from the active sweep.
The previous CSV snapshot is archived as raw_before_wheels_1_10.csv.

Wheels-only exports: wheels_raw.csv, wheels_summary.csv, wheels.png.
Combined exports: raw.csv, summary.csv, siren_runtime_error.png and .pdf.

The combined PNG/PDF and eight individual PNGs plot median inference time
against sqrt(mean standardized MSE across repetitions), using the exact
posterior means and marginal variances from the supplied libsiren script.
Values below 1e-12 are floored for plotting only. Raw values are in raw.csv.

Timing includes Original Siren particle creation; the compiled benchmark
excludes particle creation. Both include program execution and weighted mean
evaluation and exclude imports/startup.

Original Siren currently passes the same seed to every particle in
siren/inference/interface.py, ProbState.__init__. SLAM and outlier estimates
therefore barely change with particle count for a fixed seed. These curves
report existing interpreter behavior; they do not establish convergence from
independent particles. Source models and interpreter behavior were preserved.

Validation: 20 tests passed in test_compare_libsiren.py and
test_compare_modes.py. Verified 190 successful model/count/repetition records, including all 15 wheels records and
visually inspected the combined plot.
