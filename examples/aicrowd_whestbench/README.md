# ARC WhestBench Task

This directory adds a TTT-Discover environment for the
[ARC White-Box Estimation Challenge 2026](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026).

The official task asks participants to predict per-neuron post-ReLU activation
means for random He-initialized MLPs under standard-normal inputs. Submissions
are executable Python estimators, and the public leaderboard ranks primarily by
adjusted final-layer MSE, lower is better.

## Local Smoke Test

```bash
python -m examples.aicrowd_whestbench.env
```

The local evaluator is intentionally lightweight and NumPy-only. It builds small
random MLPs, estimates Monte Carlo references, and scores a candidate
`estimate(mlp, budget)` function by final-layer MSE.

Useful knobs:

```bash
WHEST_LOCAL_WIDTH=128 \
WHEST_LOCAL_DEPTH=16 \
WHEST_LOCAL_REFERENCE_SAMPLES=8192 \
WHEST_LOCAL_SEEDS=0,1,2,3 \
python -m examples.aicrowd_whestbench.env
```

## Running Discovery

```python
from examples.aicrowd_whestbench import discover_whestbench

discover_whestbench()
```

## Official Submission Path

Use the official starter kit for competition validation and packaging:

- Starter kit: <https://github.com/AIcrowd/whest-starterkit>
- Challenge page: <https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026>

The official contract expects `estimator.py` to define an `Estimator` class with
`predict(self, mlp, budget)`. In the official harness, use `flopscope.numpy` for
FLOP-counted operations, then validate/package with `whest`.
