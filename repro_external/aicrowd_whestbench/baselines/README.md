# ARC WhestBench starter baselines

These standalone estimators implement the basic strategies from
the official starter kit's
[`algorithm-ideas.md`](https://github.com/AIcrowd/whest-starterkit/blob/main/docs/how-to/algorithm-ideas.md):

- `monte_carlo.py`: progressive plain MC, randomized QMC, radial importance
  sampling, radial control variates, Rao--Blackwellization, antithetic pairing,
  and angular whitening.  The default `Estimator` is the 10% whitened-
  antithetic spherical RQMC + radial Rao--Blackwell stage,
  `O(samples * depth * width^2)`.
- `mean_propagation.py`: diagonal mean/variance propagation,
  `O(depth * width^2)`.
- `covariance_propagation.py`: dense non-central bivariate Gaussian covariance
  plus rank-512 factored cross third/fourth cumulants.  ReLU explicitly
  generates the leading connected `(2,1,1)` Gaussian Wick K3 trees, then
  exactly restores every marginal after rank pruning.  Higher Wick graphs
  remain approximate.
- `cumulant_propagation_k4_exact.py`: standalone NumPy/flopscope port of ARC's
  factored `k_max=4` propagation.  It carries the full all-distinct fourth
  cumulant as `K_ijkl = Sym(sum_r A_ijr B_klr)` and never materializes the
  `width^4` tensor.  This is the correctness reference, not a viable Phase 1
  submission at width 256/depth 32 (see the runtime note below).

All use the official `flopscope` operations and implement the required
`Estimator(BaseEstimator).predict(mlp, budget)` interface.  Run any one on
the pinned Phase 1 mini/full-100 suite with:

```bash
PYTHONPATH=/tmp/whest-official-deps:$PWD \
HF_HOME=/tmp/hf-whest-cache \
python repro/aicrowd_whestbench/evaluate_public_reproduction.py \
  repro_external/aicrowd_whestbench/baselines/mean_propagation.py \
  repro_external/aicrowd_whestbench/baselines/mean_propagation_full100_eval.json
```

## Result-to-path index

下表把“变体名称、可运行入口、结果 JSON”放在一起。所有结果均为 Phase 1
`mini` full-100，除非表中明确写成 first-10。JSON 中的 `code_sha256` 是运行时
代码指纹；同一路径后来继续演化时，旧报告仍保留历史分数，但当前文件不一定能逐字
复原旧版本。

| Family | Variant | Runnable entrypoint | Evaluation report |
|---|---|---|---|
| Analytic | Mean propagation | `mean_propagation.py` | `mean_propagation_full100_eval_20260715.json` |
| Analytic | Covariance gain ablation | historical `covariance_propagation.py` | `covariance_propagation_full100_eval_20260715.json` |
| Analytic | Marginal k3/k4 | historical `covariance_propagation.py` | `covariance_propagation_full100_eval_20260716.json` |
| Analytic | Cross CP k3/k4 | historical `covariance_propagation.py` | `covariance_propagation_cross_k4_first10_eval_20260716.json` |
| Analytic | Cross CP k3/k4 + consistent mean | historical `covariance_propagation.py` | `covariance_propagation_cross_k4_corrected_mean_full100_eval_20260716.json` |
| Analytic | Leading Wick K3 | `covariance_propagation.py` | `covariance_propagation_wick_k3_full100_eval_20260716.json` |
| Monte Carlo | Plain MC | `monte_carlo_plain.py` | `monte_carlo_full100_eval_20260715.json` |
| Monte Carlo | Randomized QMC | `monte_carlo_qmc.py` | `monte_carlo_qmc_full100_eval_20260716.json` |
| Monte Carlo | QMC + radial importance | `monte_carlo_importance.py` | `monte_carlo_importance_full100_eval_20260716.json` |
| Monte Carlo | QMC + importance + control variate | `monte_carlo_control_variate.py` | `monte_carlo_control_variate_full100_eval_20260716.json` |
| Monte Carlo | QMC + radial Rao--Blackwell | `monte_carlo_rao_blackwell.py` | `monte_carlo_rao_blackwell_full100_eval_20260716.json` |
| Hybrid MC | Whitened-antithetic spherical RQMC + RB, 10% | `monte_carlo_whitened_antithetic_rb_budget010.py` | `monte_carlo_whitened_antithetic_rb_budget010_full100_eval_20260716.json` |
| Hybrid MC | Whitened-antithetic spherical RQMC + RB, 15% | `monte_carlo_whitened_antithetic_rb_budget015.py` | `monte_carlo_whitened_antithetic_rb_budget015_full100_eval_20260716.json` |

## Phase 1 full-100 results

Measured on 2026-07-15/16 with the official subprocess runner and
`aicrowd/arc-whestbench-public-2026@v1-phase1`, split `mini`, all 100 MLPs,
width 256, depth 32, and a `2.72e11` FLOP budget per MLP:

| Estimator | Adjusted score | Final-layer MSE | All-layer MSE | Mean tracked FLOPs | Mean effective utilization | Failures |
|---|---:|---:|---:|---:|---:|---:|
| Mean propagation | `9.482215e-5` | `9.482215e-4` | `8.153814e-4` | `1.120666e7` | `0.1086%` | 0/100 |
| Covariance propagation (gain ablation) | `8.366271e-6` | `8.366271e-5` | `5.567462e-5` | `1.616597e9` | `0.7624%` | 0/100 |
| Dense covariance + marginal k3/k4 (archived version) | `7.740976e-6` | `7.740976e-5` | `5.138514e-5` | `3.018019e9` | `6.1999%` | 0/100 |
| Dense covariance + cross CP k3/k4 + consistent mean | `5.868312e-6` | `5.868312e-5` | `3.788036e-5` | `9.423376e9` | `8.3006%` | 0/100 |
| Dense covariance + leading Wick K3 | **`3.614155e-6`** | **`3.166027e-5`** | **`2.083527e-5`** | `1.556094e10` | `11.4208%` | 0/100 |
| Plain Monte Carlo (5,808 samples) | `9.531517e-7` | `9.531517e-6` | `2.563911e-5` | `2.443189e10` | `9.1303%` | 0/100 |
| Randomized QMC (5,782 samples) | `5.261843e-7` | `5.261843e-6` | `1.310876e-5` | `2.443058e10` | `9.0517%` | 0/100 |
| QMC + radial importance sampling (5,774 samples) | `5.218633e-7` | `5.218633e-6` | `1.311060e-5` | `2.445172e10` | `9.1097%` | 0/100 |
| QMC + importance + control variate (5,774 samples) | `5.226706e-7` | `5.226706e-6` | `1.311174e-5` | `2.446296e10` | `9.1941%` | 0/100 |
| QMC + radial Rao--Blackwellization (5,782 samples) | **`5.185848e-7`** | **`5.185848e-6`** | **`1.221593e-5`** | `2.443664e10` | `9.0609%` | 0/100 |
| Whitened-antithetic spherical RQMC + RB, 10% (6,292 samples) | **`3.543378e-7`** | `3.502966e-6` | `6.741515e-6` | `2.714754e10` | `10.1160%` | 0/100 |
| Whitened-antithetic spherical RQMC + RB, 15% (9,464 samples) | `3.763263e-7` | **`2.489837e-6`** | **`4.557369e-6`** | `4.072350e10` | `15.1143%` | 0/100 |

The first five Monte Carlo rows stayed below 10% effective compute and used the
minimum `0.1` multiplier.  The two whitened-antithetic rows use their measured
`0.101160` and `0.151143` mean multipliers.  The current covariance report is
`covariance_propagation_wick_k3_full100_eval_20260716.json`; the consistent
mean ablation is
`covariance_propagation_cross_k4_corrected_mean_full100_eval_20260716.json`,
and the archived marginal-covariance result is in
`covariance_propagation_full100_eval_20260716.json`.

## Fourth-order follow-up

The runnable compressed-cross version was measured on the first 10 Phase 1
MLPs.  Relative to the archived marginal-k3/k4 code on the same 10 networks:

| Version | Adjusted score | Final-layer MSE | All-layer MSE | Mean effective compute | Failures |
|---|---:|---:|---:|---:|---:|
| Marginal k3/k4 | `6.463655e-6` | `6.463655e-5` | `4.133761e-5` | - | 0/10 |
| Rank-512 cross CP k3/k4 | `6.420852e-6` | `6.420852e-5` | `3.977033e-5` | `2.575325e10` | 0/10 |

The report is `covariance_propagation_cross_k4_first10_eval_20260716.json`.
This is only a 0.66% final-layer improvement: propagating old cross cumulants
with a delta gain does not generate all new connected ReLU diagrams.

Centering the Gaussian-closure second moment on the propagated Edgeworth mean
instead of the stale Gaussian mean has a much larger effect.  On full-100 it
reduces adjusted/final-layer MSE by `24.19%` and all-layer MSE by `26.28%`
relative to the archived marginal-k3/k4 version.  Every one of the 100 MLPs
improves; the median per-MLP final-MSE ratio is `0.7665`.

Adding the leading connected ReLU Wick K3 trees gives another structural
improvement.  Relative to the consistent-mean version, full-100 adjusted score
drops by `38.41%`, raw final-layer MSE by `46.05%`, and all-layer MSE by
`45.00%`.  It improves raw final MSE on 97/100 MLPs, with median per-MLP ratio
`0.6256`.  Relative to the original marginal-k3/k4 version, the total adjusted
improvement is `53.31%`.

The exact factored implementation was checked against the upstream PyTorch
reference for three width-6 layers.  Degrees 1-4 agreed to at worst
`1.71e-13`, including diagonal slices and linear contractions.  Its factor
rank grows as `2n, 5n, 8n, ...` (an additional `3n` per layer); at width 256,
depth 32 the two fourth-order factor arrays alone require about 25.5 GB
(23.75 GiB).
An official width-256/depth-32 single-MLP run exceeded the 30-second predict
timeout, so no valid Phase 1 score is claimed for the exact file.

## Progressive Monte Carlo ablation

The five Monte Carlo rows above use the same Phase 1 full-100 suite and nearly
identical tracked compute.  Randomized QMC supplies most of the improvement:
its shifted rank-1 lattice cuts adjusted score by `44.795%` relative to plain
MC.  Exact radial mixture importance sampling improves the full-100 result by
a further `0.821%` relative to QMC.  The fixed diagonal-Gaussian radial control
coefficient slightly regresses (`0.155%` versus importance sampling), so the
control-variate hypothesis is not supported at depth 32.

The radial stage uses the positive homogeneity of the bias-free ReLU network.
Writing a standard Gaussian input as `X = R U`, with independent
`R ~ chi(width)` and uniform direction `U`, every layer obeys
`f(R U) = R f(U)`.  Replacing the sampled radius with exact `E[R]` is therefore
an unbiased Rao--Blackwellization for every layer.  The reported first-layer
row is also replaced by its exact Gaussian ReLU expectation.  On its own, this
gives a `45.593%` adjusted-score reduction versus plain MC.

The thin `monte_carlo_{plain,qmc,importance,control_variate}.py` entrypoints
and `monte_carlo_rao_blackwell.py` select the corresponding ablation class for
the official subprocess runner. Their full reports and the final report are the adjacent
`monte_carlo_*_full100_eval_20260716.json` files.

## Whitened-antithetic spherical RQMC

The follow-up estimator generates half of the sample set with the shifted
rank-1 lattice, normalizes those rows to the sphere, and appends their exact
antipodes.  It then matches the half-batch angular second moment to `I/width`
with a symmetric whitening transform folded into the first weight matrix.  The
antipodal set has exact zero empirical mean, while the whitening removes its
leading angular second-moment error without a sample-sized whitening matmul.
Finally, exact `E[chi(width)]` retains radial Rao--Blackwellization.  The radial
step is an exact unbiased identity; sample-dependent whitening is a consistent
finite-sample moment-matching correction rather than an unbiased identity.

Both requested budgets were evaluated on full-100.  The 15% row reduces raw
final-layer MSE by `28.922%` relative to the 10% row, but its higher compute
multiplier makes adjusted score `6.205%` worse.  The 10% row is therefore the
default `Estimator`; it improves adjusted score by `31.672%` over radial RQMC
and by `62.825%` over plain MC.  The reproducible entrypoints are
`monte_carlo_whitened_antithetic_rb_budget{010,015}.py`, with adjacent dated
full-100 reports.

## Official result and artifact provenance

The 10% estimator was submitted as
[#316625](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316625)
and graded successfully:

| Metric | Value |
|---|---:|
| Local full-100 adjusted | `3.543377641e-7` |
| Official public-50 adjusted | `3.355233812e-7` |
| Official secondary MSE | `3.262751206e-6` |
| Official implied multiplier | `0.1028345` |
| Official relative to local | `-5.31%` |

The archive uploaded for #316625 had SHA-256
`dae36b30f44616518526eaee8e7aec639cf0b9b05e1187416b363b921c7fd11b`.
A parallel session later rebuilt the same local filename, producing archive
SHA-256 `309f97f132bc87c9bd4ec771ecf19b122cda5be03b0a27551e346bfffb1dc778`,
and uploaded that repackaged duplicate as #316628. Both archives contain the
same `estimator.py` bytes, SHA-256
`146a8c786ee31dfefc941712ded736c3ec2625343dd5cd184f80dde915a9d130`;
the archive hash changed because packaging timestamps and manifest metadata
changed. The registry pins the submitted archive hashes, while the live status
of both submissions is maintained in `../SUBMISSION_TRACKER.md`.

Submission #316628 failed without producing a score.  AIcrowd's grading
message was `Error : Evaluation could not complete; please retry`; it did not
report an import, contract, budget, timeout, or estimator exception.  Because
#316625 ran the byte-identical `estimator.py` successfully, this duplicate is
recorded as a platform/evaluation failure rather than an algorithm failure.
No additional retry was submitted because #316625 already provides a valid
official public-50 result for the same estimator.
