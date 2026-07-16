# Public WhestBench Reproductions

| Method | Matching public protocol | Public adjusted | Local full-mini adjusted | Difference |
|---|---|---:|---:|---:|
| German Alfaro #311690 | Phase 1 | `3.357e-7` | `3.834192e-7` | `+14.2%` |
| evaaaz RQMC + RB | Phase 1 | `~4.10e-7` | `4.885414e-7` | `+19.2%` |
| pscamillo #314331 | Phase 1 | `2.45e-6` | `2.574242e-6` | `+5.1%` |
| ascender1729 k3 + k4 | warm-up | `6.65e-7` | `4.975772e-7` | `-25.2%` |

Positive difference means the local score is worse; lower is better. The
cumulant result is warm-up-only and is not comparable with the Phase 1 rows.

## German Alfaro, AIcrowd submission 311690

This directory reproduces the whitened antithetic Monte Carlo estimator
published by German Alfaro for AIcrowd submission `311690`:

- upstream repository: <https://github.com/galfaroi/Can-You-Predict-a-Network-Without-Running-It->
- upstream commit checked: `943a7a50b3350d6b11ac0f119c46d5c1f0876d4e`
- upstream reported public-split adjusted score: `3.357e-7` over 50 MLPs
- local estimator: `german_alfaro_311690_whitened_antithetic.py`
- local full result: `german_alfaro_311690_official_mini_eval_20260713.json`

### Algorithm

For an even sample count `k`, the estimator draws `k/2` float32 Gaussian rows
`u` and constructs the antithetic batch `[u; -u]`. Its mean is exactly zero,
and its covariance can be computed from one side as `u.T @ u / (k/2)`. After a
symmetric eigendecomposition, the inverse covariance square root is folded into
the first-layer weight. This removes a sample-sized whitening matrix multiply.
The folded samples are forwarded through all 32 ReLU layers and averaged at the
final layer. Earlier output rows are zero because Phase 1 scores only the final
row.

The reproduction retains the published `0.155` budget fraction and produces
`k=9826` for width 256, depth 32, and the official `2.72e11` FLOP budget.

### Evaluation rounds

All scored rounds used `aicrowd/arc-whestbench-public-2026@v1-phase1`, split
`mini`, the official subprocess runner, width 256, depth 32, a `2.72e11` per-MLP
FLOP budget, and `lambda=1e11` FLOPs/s.

| Round | MLPs | Adjusted score | Final-layer MSE | Mean effective compute | Failures |
|---|---:|---:|---:|---:|---:|
| contract probe | 1 | `5.028393e-7` | `3.208989e-6` | `4.262161e10` | 0 |
| first-10 | 10 | `2.545944e-7` | `1.637123e-6` | `4.228000e10` | 0 |
| final mini | 100 | `3.834192e-7` | `2.465925e-6` | `4.228989e10` | 0 |

The final run's maximum effective compute was `4.265005e10`, and every MLP
used exactly `4.2099123456e10` tracked FLOPs. It completed all 100 MLPs without
budget, wall-time, combined-budget, or estimator errors. The result meets the
local acceptance target of `4.5e-7`.

The local adjusted score is 14.2% higher than the published `3.357e-7`. This is
consistent with an unbiased finite-sample estimator being measured on a
different evaluation set: the public claim used AIcrowd's 50-MLP public split,
while this report uses the reproducible 100-MLP Hugging Face mini split. The
first-10 score (`2.546e-7`) also shows the size of subset variation. The
production sample count and algorithm match upstream; the local evaluator is
`whestbench 0.12.0rc5` with `flopscope 0.8.0rc5`, which may also differ from the
hosted grader build used for submission 311690.

### Commands

```bash
HF_HOME=/tmp/hf-whest-cache \
PYTHONPATH=/tmp/whest-official-deps:$PWD \
pytest -q tests/test_aicrowd_whestbench.py \
  tests/test_aicrowd_whestbench_public_reproduction.py
```

```bash
HF_HOME=/tmp/hf-whest-cache \
PYTHONPATH=/tmp/whest-official-deps:$PWD \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python repro/aicrowd_whestbench/evaluate_public_reproduction.py \
  repro_external/aicrowd_whestbench/public_reproductions/german_alfaro_311690_whitened_antithetic.py \
  repro_external/aicrowd_whestbench/public_reproductions/german_alfaro_311690_official_mini_eval_20260713.json \
  --n-mlps 100
```

## evaaaz, randomized QMC and Rao-Blackwellization

This reproduction implements the method described in the public AIcrowd forum
post [Unbiased randomized-QMC + Rao-Blackwell for post-ReLU activation
means](https://discourse.aicrowd.com/t/unbiased-randomized-qmc-rao-blackwell-for-post-relu-activation-means-method-unbiasedness-proof-and-where-the-frontier-is/18053):

- source status: public method description, but no executable source or
  submission ID was published;
- public report: adjusted score about `4.10e-7`, `0/100` failures, and
  `C/B` about `0.42` on the Phase 1 public mini split;
- local estimator: `evaaaz_rqmc_rao_blackwell.py`;
- local full result: `evaaaz_rqmc_rao_blackwell_official_mini_eval_20260714.json`.

### Algorithm

The estimator constructs the width-dimensional generator
`g_j = frac(sqrt(p_j))` from the first 256 primes. For each MLP, one uniform
Cranley-Patterson shift `U` is drawn from an RNG seeded by `mlp.seed`. Its
`N` lattice rows are

`u_k = frac(k * g + U), k = 0, ..., N - 1`.

The uniforms are clipped to `[1e-11, 1 - 1e-11]` in float64 and transformed
with `flops.stats.norm.ppf`. These Gaussian inputs are propagated through the
normal 32-layer ReLU MLP, retaining each layer's sample mean. The first output
row is Rao-Blackwellized to its exact value
`sqrt(sum(W0**2, axis=0)) / sqrt(2*pi)`. Later layers still use the sampled
first-layer activations, so the final-layer gain is due to RQMC; the analytic
replacement improves the reported first-layer row without biasing propagation.

The public post does not state its exact sample count. The local implementation
sets tracked work to 41.8% of the FLOP budget, giving one shift with `N=26,858`
rows. Official residual compute raises measured utilization to about 42.36%,
matching the published `C/B` target while staying well below the budget.

### Results

Both local rounds used `aicrowd/arc-whestbench-public-2026@v1-phase1`, split
`mini`, width 256, depth 32, the official subprocess runner, a `2.72e11` per-MLP
FLOP budget, and `lambda=1e11` FLOPs/s.

| Result | MLPs | Adjusted score | Final-layer MSE | Mean C/B | Failures |
|---|---:|---:|---:|---:|---:|
| public forum report | 100 | `~4.10e-7` | not reported | `~0.42` | 0 |
| local first-10 | 10 | `5.387741e-7` | `1.275042e-6` | `0.422715` | 0 |
| local full mini | 100 | `4.885414e-7` | `1.153299e-6` | `0.423625` | 0 |

The full run used exactly `1.13475880917e11` tracked FLOPs per MLP. Mean and
maximum effective compute were `1.15226056423e11` and `1.15459274646e11`,
respectively; all 100 predictions completed without budget, time, combined
budget, or estimator failures.

The local adjusted score is `7.8541e-8`, or 19.2%, above the rounded public
report. A bit-for-bit reproduction is not possible from the post: it omits the
exact `N`, lattice index origin, per-MLP seed derivation, evaluator versions,
and executable code. These choices change the realized error of a single-shift
unbiased estimator without changing its expectation. The local implementation
uses the standard zero-origin rank-1 lattice and direct `mlp.seed`, matches all
published algorithmic details and compute/failure measurements, and is kept
unchanged rather than tuning shifts against the evaluation targets.

The first-10 run wrote `/tmp/evaaaz_rqmc_first10.json` as a temporary diagnostic;
it is recorded above but intentionally not included as a repository artifact.

### Command

```bash
HF_HOME=/tmp/hf-whest-cache \
PYTHONPATH=/tmp/whest-official-deps:$PWD \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python repro/aicrowd_whestbench/evaluate_public_reproduction.py \
  repro_external/aicrowd_whestbench/public_reproductions/evaaaz_rqmc_rao_blackwell.py \
  repro_external/aicrowd_whestbench/public_reproductions/evaaaz_rqmc_rao_blackwell_official_mini_eval_20260714.json \
  --n-mlps 100
```

## pscamillo, AIcrowd submission 314331

The public technical write-up reports a scalar-corrected second-order Gaussian
closure estimator, but does not publish executable code. The reproduction uses
the official full-covariance gain baseline and applies the write-up's
cross-validated `0.9921` factor only to the scored final-layer mean.

- write-up: <https://discourse.aicrowd.com/t/phase-1-write-up-characterizing-a-systematic-scale-bias-in-the-gaussian-closure-estimator-submission-314331/18063>
- public submission score: `2.45e-6` over the hosted 50-MLP public split;
- write-up harness result: `2.66e-6` corrected versus `7.52e-6` uncorrected;
- local estimator: `pscamillo_314331_scalar_corrected_gaussian_closure.py`;
- local full result: `pscamillo_314331_official_mini_eval_20260714.json`.

| Result | MLPs | Adjusted score | Final-layer MSE | Mean effective compute | Failures |
|---|---:|---:|---:|---:|---:|
| public submission | 50 | `2.45e-6` | not reported | not reported | 0 |
| local first-10 | 10 | `1.810315e-6` | `1.810315e-5` | `2.057699e9` | 0 |
| local full mini | 100 | `2.574242e-6` | `2.574242e-5` | `2.026487e9` | 0 |

The local full result is 5.1% above the hosted submission and 3.2% below the
write-up's own harness result. Every network used `1.616597248e9` tracked FLOPs,
the score multiplier stayed at its `0.1` floor, and all 100 networks completed
without failures. This is a close numerical reproduction of both the reported
score and the reported `2.59e-5` raw MSE on the official 100-network benchmark.

## ascender1729, cumulant k3 plus k4 extrapolation

This is a vendored copy of the public single-file estimator at upstream commit
`c6f87fd1e12634447a452f73ebc136c43bf050d5`. Its NumPy backend was adapted to
convert flopscope 0.8.0rc5 results back to base arrays before the upstream
in-place tensor operations. The mathematics is unchanged, and the upstream MIT
license is retained in `ASCENDER1729_LICENSE`.

- repository: <https://github.com/ascender1729/whestbench-cumulant-propagation>;
- reported live warm-up score: `6.65e-7`;
- local estimator: `ascender1729_cumulant_k3_k4_warmup.py`;
- local warm-up result: `ascender1729_cumulant_k3_k4_warmup_official_mini_eval_20260714.json`;
- current Phase 1 probe: `ascender1729_cumulant_k3_k4_phase1_probe1_eval_20260714.json`.

| Result | Protocol | MLPs | Adjusted score | Final-layer MSE | Failures |
|---|---|---:|---:|---:|---:|
| upstream live grader | warm-up, depth 8 | full | `6.65e-7` | not reported | 0 |
| local first-10 | warm-up, depth 8 | 10 | `4.759479e-7` | `9.707086e-7` | 0 |
| local full mini | warm-up, depth 8 | 100 | `4.975772e-7` | `1.021468e-6` | 0 |
| local compatibility probe | Phase 1, depth 32 | 1 | invalid | invalid | 1 timeout |

The valid local warm-up run is 25.2% below the public live-grader score. It
used exactly `1.7505459874e10` tracked FLOPs per network and completed 100/100
networks. The current Phase 1 probe executes the real k3 path but exceeds the
official 30-second predict timeout on its first depth-32 network. A 100-network
Phase 1 run was therefore not performed: it would only repeat a known protocol
failure, and its fallback score would not represent cumulant propagation.

The warm-up command uses the evaluator's non-default protocol flags:

```bash
HF_HOME=/tmp/hf-whest-cache \
PYTHONPATH=/tmp/whest-official-deps:$PWD \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python repro/aicrowd_whestbench/evaluate_public_reproduction.py \
  repro_external/aicrowd_whestbench/public_reproductions/ascender1729_cumulant_k3_k4_warmup.py \
  repro_external/aicrowd_whestbench/public_reproductions/ascender1729_cumulant_k3_k4_warmup_official_mini_eval_20260714.json \
  --n-mlps 100 --revision v1-warmup --flop-budget 68000000000
```
