# WhestBench Codex-run results, 2026-07-22 to 2026-07-23

This snapshot records the compact, reviewable outputs from two Codex-driven
WhestBench searches. Raw `codex_runs/` workspaces are intentionally excluded:
the two active result roots occupy roughly 2.7 GB and contain transient model
homes, evaluator logs, and duplicate source snapshots.

## Diversity-mode full-100 pipeline

- Git baseline: `5ef1df32`.
- Evaluation protocol: official mini split, first 100 MLPs.
- AutoEvolve protocol: one complete epoch per run, hard six-hour cap.
- Completed outer pools: 18, each with four generated full-100 submissions.
- Every retained outer state had 100 construction records.
- The independent filters launched distinct follow-ups when more than one
  starter survived; the terminal filters found no further credible starter.

Four behaviorally distinct candidates were packaged and submitted to the
public-50 leaderboard:

| ID | Method | Local full-100 | Public-50 | Drift |
|---:|---|---:|---:|---:|
| [317922](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/317922) | primitive-root tent RQMC plus exact first-layer marginal calibration | `3.026566e-7` | `2.852186e-7` | `-5.76%` |
| [317924](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/317924) | pure primitive-root tent RQMC with antithetic ZCA | `3.134437e-7` | `2.887088e-7` | `-7.89%` |
| [317925](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/317925) | exact-radius sampling over 13 signed-Hadamard bases | `3.224360e-7` | `3.596574e-7` | `+11.54%` |
| [317926](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/317926) | base-2 Korobov RQMC plus first-layer marginal calibration | `3.025845e-7` | `3.624420e-7` | `+19.78%` |

The tent construction generalized; the Hadamard and base-2 variants exhibited
large adverse suite drift. Submission 317922 is the best new result from this
pipeline, while the repository-wide registered best remains submission 316192
at `2.787552892e-7`.

## Acronym-diversity Codex runner

The new runner assigns externally generated three-letter codes to independent
idea proposers. A filter selects exactly two proposals per 12-code batch for
one full-100, one-epoch AutoEvolve follow-up. The sweep wrapper traverses codes
lexicographically, persists an atomic cursor, uses a process lock, and resumes
after interrupted supervisors.

Three completed candidates were promoted to public-50:

| ID | Acronym | Local full-100 | Public-50 | Drift |
|---:|---|---:|---:|---:|
| [318093](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/318093) | ADA — Angular Deterministic Antithetics | `3.180688e-7` | `3.441437e-7` | `+8.20%` |
| [318094](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/318094) | AAY — Analytic Antithetic Yoking | `3.419277e-7` | `3.686147e-7` | `+7.80%` |
| [318095](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/318095) | ACL — Analytic Control Linearization | `3.438783e-7` | `3.080306e-7` | `-10.42%` |

ACL was locally weaker than ADA but generalized substantially better, which is
useful evidence for continuing mechanism-diverse rather than score-only search.

## Published artifacts

- `repro/aicrowd_whestbench/run_whest_acronym_pipeline.py`
- `repro/aicrowd_whestbench/run_whest_acronym_sweep.py`
- `tests/test_whest_acronym_pipeline.py`
- `repro_external/aicrowd_whestbench/submissions/candidates/`
- `repro_external/aicrowd_whestbench/submissions/packages/`
- `repro_external/aicrowd_whestbench/submissions/registry.json`
- `repro_external/aicrowd_whestbench/SUBMISSION_TRACKER.md`

## Validation

- `python -m py_compile` passed for both runner scripts.
- Eight acronym-runner unit tests passed.
- All seven newly recorded submission archives passed `whest validate-package`.
- Registry JSON parses successfully and submitted artifact hashes match the
  pinned SHA-256 values.
