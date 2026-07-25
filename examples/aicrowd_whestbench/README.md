# ARC WhestBench Task

This directory integrates the
[ARC White-Box Estimation Challenge 2026](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026)
with TTT-Discover using the official `whestbench` and `flopscope` runtimes.

The task predicts per-neuron post-ReLU activation means for random
He-initialized MLPs under standard-normal inputs. Candidates must implement the
official `Estimator(BaseEstimator).predict(mlp, budget)` contract and return a
`flopscope.numpy.ndarray` with shape `(depth, width)`.

## Scoring

Evaluation delegates to `whestbench.scoring.evaluate_estimator`. For each MLP:

```text
effective_compute = flops_used + lambda * residual_wall_time
score = final_layer_mse * max(0.1, effective_compute / flop_budget)
```

The suite score is the mean per-MLP score and is minimized. The official Phase 1
defaults are:

- Dataset: `aicrowd/arc-whestbench-public-2026@v1-phase1`
- Split: `mini`, 100 MLPs
- Shape: width 256, depth 32
- FLOP budget: `272_000_000_000` per MLP
- Residual penalty: `100_000_000_000` FLOPs/second
- Runner: official subprocess runner

The checked-in discovery YAML uses two disjoint gates over the 100 MLPs:

- test gate: rows `[0, 50)`;
- holdout gate: rows `[50, 100)`, evaluated only after the test gate passes.

Both gates require the candidate to improve over the incumbent by more than
`1e-7` by default. Configure the suite sizes and thresholds with
`WHEST_TEST_N_MLPS`, `WHEST_HOLDOUT_N_MLPS`,
`WHEST_TEST_ACCEPTANCE_THRESHOLD`, and
`WHEST_HOLDOUT_ACCEPTANCE_THRESHOLD`.

Failures follow the official semantics: invalid output, exceptions, or compute
exhaustion replace that MLP's prediction with zeros and force the score
multiplier to 1.0.

## Smoke Test

The smoke test uses a tiny generated contest while exercising the same official
runner and scoring implementation. It does not download the public dataset.

```bash
python -m examples.aicrowd_whestbench.env
```

## Discovery

Run the checked-in configuration:

```bash
python repro/run_from_yaml.py \
  --config configs/discovery_runs.yaml \
  --run aicrowd_whestbench
```

The first run downloads and caches the pinned `mini` split (about 850 MB). Set
`WHEST_DATASET` to a local official dataset directory for fully offline
evaluation. `WHEST_DATASET_STREAMING=1` is available for constrained
environments, but materialized evaluation is the official starter-kit default.

To run both TTT-Discover and AutoEvolve with the same official CPU evaluator,
use the reproducibility launcher. AutoEvolve uses the trusted blackbox server;
TTT-Discover uses the in-process evaluator adapter. Neither path requires CUDA.

```bash
# One epoch and one mini MLP on each algorithm.
RUN_WHEST_SMOKE=1 \
  bash repro/aicrowd_whestbench/run_official_mini.sh all

# Use a 50-MLP test gate followed by a disjoint 50-MLP holdout gate.
WHEST_SEARCH_N_MLPS=50 WHEST_HOLDOUT_N_MLPS=50 \
  bash repro/aicrowd_whestbench/run_official_mini.sh all
```

Where Unix sockets or mount namespaces are unavailable, AutoEvolve can run in
isolated one-shot mode and leave final scoring to the outer official evaluator:

```bash
RUN_WHEST_SMOKE=1 \
  bash repro/aicrowd_whestbench/run_official_mini.sh autoevolve-in-process
```

## Official Validation

TTT-Discover scores candidates with the official Python evaluator, but final
submission validation and packaging should still use the official CLI:

```bash
whest validate --estimator estimator.py
whest run \
  --estimator estimator.py \
  --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 \
  --split mini \
  --runner subprocess
whest package --estimator estimator.py
```

Official resources:

- Starter kit: <https://github.com/AIcrowd/whest-starterkit>
- Evaluator: <https://github.com/AIcrowd/whestbench>
- FLOP accounting: <https://github.com/AIcrowd/flopscope>
