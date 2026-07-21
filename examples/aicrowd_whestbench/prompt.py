import os
from pathlib import Path


WHESTBENCH_PROMPT = """You are solving the ARC White-Box Estimation Challenge 2026
(WhestBench) under the official evaluator contract.

Task:
- You receive the weights of a randomly initialized square ReLU MLP.
- Inputs follow a standard normal distribution.
- Predict the per-neuron post-ReLU activation mean for every layer.
- Return a `flopscope.numpy.ndarray` with shape `(mlp.depth, mlp.width)`.

Required submission interface:
- Import `flopscope as flops` for supported distribution functions such as
  `flops.stats.norm.pdf` and `flops.stats.norm.cdf`.
- Import `flopscope.numpy as fnp` for numerical operations.
- Define `class Estimator(BaseEstimator)` with
  `predict(self, mlp, budget)`.
- Optional `setup(self, context)` and `teardown(self)` hooks follow the official
  WhestBench API.
- Seed predict-time randomness from `mlp.seed` and setup-time randomness from
  `context.seed`.
- `flopscope.numpy` does not provide `fnp.erf`; use the supported
  `flops.stats.norm` functions for Gaussian PDF/CDF calculations.
- `flopscope.numpy` arrays are immutable: do not use item assignment or indexed
  in-place updates. Build pieces in Python lists and combine them with
  `fnp.stack`/`fnp.concatenate`, or use whole-array expressions.

Official scoring:
- For each MLP, compute final-layer MSE against the official Monte Carlo target.
- Effective compute is `C = FLOPs + lambda * residual_wall_time`.
- A valid MLP score is `final_layer_mse * max(0.1, C / flop_budget)`.
- The suite score is the arithmetic mean of those per-MLP scores; lower is better.
- Exceptions, invalid shapes/non-finite values, FLOP/time exhaustion, or combined
  budget exhaustion use an all-zero prediction and multiplier 1.0 for that MLP.
- All-layer MSE is diagnostic only.

Do not use plain NumPy or uninstrumented numerical libraries to evade FLOP
accounting. Residual Python or uninstrumented work is charged at the official
lambda rate and can exhaust the combined budget.

Fairness and data isolation:
- Derive predictions only from the `mlp` weights and allowed setup context.
- Do not inspect, search for, or read evaluation datasets, Hugging Face caches,
  saved evaluation reports, target moments, or ground-truth files.
- Do not hardcode values tied to public mini MLPs. Such candidates are invalid
  even if they receive a low local score.

Algorithm directions worth exploring:
- Mean and diagonal-variance propagation through ReLU moments.
- Full, structured, or low-rank covariance propagation.
- Hybrid analytic propagation with compute-budgeted Monte Carlo probes.
- Layer-wise corrections for correlation error in deep networks.
- Allocating FLOPs to layers where final-layer MSE is most sensitive.
"""


WHESTBENCH_PROMPT_BASE = WHESTBENCH_PROMPT


DIVERSITY_MODE_FILE_ENV_VAR = "WHEST_DIVERSITY_MODE_FILE"


def load_diversity_mode_file(mode_file: str | os.PathLike[str] | None) -> str | None:
    """Load one complete diversity-mode prompt block from a UTF-8 text file."""
    if mode_file is None or not str(mode_file).strip():
        return None
    path = Path(mode_file).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"WhestBench diversity mode file not found: {path}")
    block = path.read_text(encoding="utf-8").strip()
    if not block:
        raise ValueError(f"WhestBench diversity mode file is empty: {path}")
    return block


def build_whestbench_prompt(
    mode_file: str | os.PathLike[str] | None = None,
) -> str:
    """Append the diversity-mode file, if configured, to the base prompt."""
    block = load_diversity_mode_file(mode_file)
    if block is None:
        return WHESTBENCH_PROMPT_BASE
    return f"{WHESTBENCH_PROMPT_BASE}\n\n{block}\n"


WHESTBENCH_PROMPT = build_whestbench_prompt(
    os.environ.get(DIVERSITY_MODE_FILE_ENV_VAR)
)
