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

Official public leaderboard top 20 (snapshot retrieved 2026-07-29):
- Adjusted Score and Final Layer MSE are both lower-is-better. They can differ
  because Adjusted Score includes the official compute multiplier.
- Source:
  https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards

| Rank | Participant | Adjusted Score | Final Layer MSE | Entries |
| ---: | :--- | ---: | ---: | ---: |
| 1 | joe_wanza | 0.0000000123 | 0.0000000824 | 594 |
| 2 | dpskv5 | 0.0000000227 | 0.0000000288 | 56 |
| 3 | abhinav_gorrepati | 0.0000000230 | 0.0000002104 | 107 |
| 4 | fklassen | 0.0000000345 | 0.0000001620 | 143 |
| 5 | mliston | 0.0000000463 | 0.0000001680 | 370 |
| 6 | huang_chung_yi | 0.0000000536 | 0.0000000817 | 135 |
| 7 | adrianleb | 0.0000000559 | 0.0000002592 | 21 |
| 8 | ai_innovation | 0.0000000918 | 0.0000001972 | 448 |
| 9 | jtel | 0.0000000938 | 0.0000002039 | 195 |
| 10 | SKIBIDI_TOILET | 0.0000000976 | 0.0000002089 | 246 |
| 11 | Puffi | 0.0000001062 | 0.0000001994 | 118 |
| 12 | neuron | 0.0000001143 | 0.0000002126 | 568 |
| 13 | kaileh57 | 0.0000001159 | 0.0000002089 | 293 |
| 14 | jamespayor | 0.0000001193 | 0.0000001724 | 119 |
| 15 | sweaty_dog | 0.0000001212 | 0.0000001455 | 95 |
| 16 | ednacob | 0.0000001234 | 0.0000002361 | 37 |
| 17 | yanggan_gu | 0.0000001274 | 0.0000002422 | 29 |
| 18 | andrew_epstein | 0.0000001288 | 0.0000002381 | 37 |
| 19 | andrei_bulzan | 0.0000001315 | 0.0000002090 | 133 |
| 20 | williawa | 0.0000001437 | 0.0000007644 | 171 |
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
