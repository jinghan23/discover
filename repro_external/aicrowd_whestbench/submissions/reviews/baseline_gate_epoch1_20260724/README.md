# Baseline-gated AutoEvolve epoch-1 official submissions

Source run:
`codex_runs/aicrowd_whestbench/whestbench_baseline_gate_epoch1_20260724`.

Both candidates were independently reevaluated on the official Phase 1
`mini` full-100 suite with the subprocess runner, one BLAS thread, and zero
failed MLPs before packaging. The exact estimator and archive SHA-256 values
are pinned in `submissions/registry.json`.

| Submission | Method | Local full-100 adjusted | Official public-50 adjusted | Official MSE |
|---|---|---:|---:|---:|
| #318500 | Antithetic LHS, covariance whitening, exact first-layer moments, analytic final-layer closure | `4.127186617e-7` | `4.525908775e-7` | `4.447715664e-6` |
| #318501 | Full covariance propagation, fixed-radius antithetic sampling, degree-two ReLU control variate | `2.468900577e-6` | `2.111014208e-6` | `1.881607208e-5` |

Both submissions were graded successfully. Neither improved the previously
registered best official adjusted score, `2.787552892e-7` from submission
#316192.
