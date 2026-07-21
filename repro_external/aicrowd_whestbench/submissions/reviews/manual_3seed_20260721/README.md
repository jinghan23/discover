# Manual 3-seed full-100 review

This directory is the durable, Git-safe extract of the corresponding local
`codex_runs/` review. It intentionally contains only the five exact estimator
snapshots, fifteen full-100 JSON reports, the snapshot manifest, and this
decision summary. Raw model-call logs, autonomous workspaces, Codex home data,
and the full search trajectory remain local and ignored.

- Review snapshot: `2026-07-21T08:48:56Z`
- Protocol: official Phase 1 mini full-100, subprocess runner, setup seeds `0/1/2`, BLAS/OMP threads fixed to 1
- Shortlisted methods: 5
- Evaluations: 15 full-100 reports, 1,500 total MLP evaluations, 0 failures
- Official submissions: 1 (`#317723`)

| Candidate | 3-seed adjusted scores | Mean | Sample SD | Mean C/B | Decision |
|---|---|---:|---:|---:|---|
| `german013_auto_r1` (`4649711fe32a`) | `2.542398e-7`, `2.542452e-7`, `2.542824e-7` | `2.542558e-7` | `2.315e-11` | `0.22829` | Do not submit after correlated official drift evidence |
| `german013_auto_r2` (`a246e72ffbfd`) | `2.563439e-7`, `2.563919e-7`, `2.562267e-7` | `2.563208e-7` | `8.501e-11` | `0.12193` | Submitted as `#317723`; official `3.130976e-7` |
| `affinef64_auto_r2` (`66327c7d9e98`) | `2.560232e-7`, `3.248103e-7`, `3.214791e-7` | `3.007709e-7` | `3.879e-8` | `0.22999` | Reject: setup-seed sensitive |
| `affinef64_ttt` (`f66ab7b8be13`) | `2.714235e-7`, `3.299987e-7`, `3.273451e-7` | `3.095891e-7` | `3.308e-8` | `0.17944` | Reject: setup-seed sensitive |
| `rqmc030_auto_r1` (`34eeb3feeaa4`) | `2.829990e-7`, `3.616459e-7`, `3.530940e-7` | `3.325796e-7` | `4.315e-8` | `0.12995` | Reject: setup-seed sensitive |

## Method screening

- `german013_auto_r1`: antithetic Gaussian probes, empirical input whitening, exact first-layer marginal quantile transport, and exact second-layer preactivation moment correction. It is structurally changed from `#317465` (numeric-normalized AST similarity about `0.43`) but remains in the same high-drift quantile/second-moment lineage.
- `german013_auto_r2`: shifted Box-Muller lattice, antithetic pairing, whitening, and strengthened early-layer corrections. It is close to `#317473` (numeric-normalized AST similarity about `0.81`) but improved local full-100 by about `15.1%`.
- `affinef64_auto_r2`: setup-cached spherical directions, angular whitening, radial Rao--Blackwellization, and first/second-layer moment matching. Structurally distinct from `#317463`, but the search seed was unusually favorable.
- `rqmc030_auto_r1`: scrambled Sobol directions with exact early-layer moment controls. Structurally distinct from `#317452`, but the search seed was unusually favorable.
- `affinef64_ttt`: analytic early-layer covariance and Edgeworth/final analytic blending. Strong only at setup seed 0.

`affinef64_auto_r1`, the remaining spherical/layer-rescale/RQMC-r2 candidates, and the weaker TTT states were screened out before 3-seed evaluation because they were either method-near-duplicates or materially worse on full-100.

## Official decision

`german013_auto_r2` was submitted first because its same-lineage predecessor `#317473` had only `-1.07%` local-to-official drift and the new candidate showed a stable `15.1%` local improvement. Submission `#317723` graded at `3.1309757095511477e-7` (secondary `2.544812759879278e-6`), a `+22.1%` adverse drift from the new 3-seed local mean and worse than the registered best `#316192` at `2.787552892175721e-7`.

The new official result and `#317465`'s prior `+18.3%` adverse drift both argue against submitting the correlated `german013_auto_r1` candidate. Its validated package is retained locally, but it was not uploaded.
