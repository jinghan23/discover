from __future__ import annotations

import inspect
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ttt_discover import BaseRewardEvaluator, Environment, State
from ttt_discover.tasks.sandbox_reward_evaluator import run_with_timeout


DEFAULT_PRIMES = (5, 7, 13)
DEFAULT_DIMENSION = 3


@dataclass(frozen=True)
class KakeyaProblemSpec:
    dimension: int
    primes: tuple[int, ...]


def parse_problem_type(problem_type: str | int | None) -> KakeyaProblemSpec:
    """Parse problem types like '', '5,7,13', or '3:5,7,13'."""
    if problem_type in (None, ""):
        return KakeyaProblemSpec(DEFAULT_DIMENSION, DEFAULT_PRIMES)

    text = str(problem_type).strip()
    if ":" in text:
        dim_text, primes_text = text.split(":", 1)
        dimension = int(dim_text.strip())
    else:
        dimension = DEFAULT_DIMENSION
        primes_text = text

    primes = tuple(int(part.strip()) for part in primes_text.split(",") if part.strip())
    if dimension != 3:
        raise ValueError("This Kakeya environment currently supports only d=3")
    if not primes:
        raise ValueError("At least one prime must be provided")
    if any(p < 2 for p in primes):
        raise ValueError(f"Invalid prime list: {primes}")
    return KakeyaProblemSpec(dimension=dimension, primes=primes)


def projective_directions_3d(p: int) -> Iterable[tuple[int, int, int]]:
    for y in range(p):
        for z in range(p):
            yield (1, y, z)
    for z in range(p):
        yield (0, 1, z)
    yield (0, 0, 1)


def normalize_point_array(points: object, p: int, d: int) -> np.ndarray:
    try:
        array = np.asarray(points)
    except Exception as exc:
        raise ValueError(f"Cannot convert construction to ndarray: {exc}") from exc

    if array.ndim != 2 or array.shape[1] != d:
        raise ValueError(f"Expected construction shape (n, {d}), got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.equal(array, np.floor(array))):
            raise ValueError("Construction coordinates must be integers")
        array = array.astype(np.int64)
    else:
        array = array.astype(np.int64, copy=False)
    if np.any(array < 0) or np.any(array >= p):
        raise ValueError(f"Construction coordinates must lie in [0, {p})")
    return array


def point_set(points: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(map(int, row)) for row in points}


def first_missing_direction_3d(points: np.ndarray, p: int) -> tuple[int, int, int] | None:
    points_as_set = point_set(points)
    if not points_as_set:
        return next(projective_directions_3d(p))

    starts = [np.array(start, dtype=np.int64) for start in points_as_set]
    for direction in projective_directions_3d(p):
        v = np.array(direction, dtype=np.int64)
        found_line = False
        for start in starts:
            if all(
                tuple(((start + t * v) % p).tolist()) in points_as_set
                for t in range(p)
            ):
                found_line = True
                break
        if not found_line:
            return direction
    return None


def is_kakeya_set_3d(points: np.ndarray, p: int) -> bool:
    return first_missing_direction_3d(points, p) is None


def evaluate_kakeya_program_outputs(outputs: object, spec: KakeyaProblemSpec) -> dict:
    """Validate one output per prime and return aggregate metrics.

    `outputs` is the return value of the injected `run()` wrapper. It must be a
    dict keyed by prime, where each value is a point array in F_p^3.
    """
    if not isinstance(outputs, dict):
        raise ValueError("run() must return a dict keyed by prime")

    per_prime: list[dict] = []
    densities: list[float] = []
    total_size = 0
    for p in spec.primes:
        if p not in outputs:
            key = str(p)
            if key not in outputs:
                raise ValueError(f"Missing construction for p={p}")
            raw_points = outputs[key]
        else:
            raw_points = outputs[p]

        points = normalize_point_array(raw_points, p, spec.dimension)
        unique_points = point_set(points)
        deduped = np.array(sorted(unique_points), dtype=np.int64)
        missing = first_missing_direction_3d(deduped, p)
        if missing is not None:
            raise ValueError(f"p={p} is missing direction {missing}")

        size = len(unique_points)
        density = size / float(p**spec.dimension)
        total_size += size
        densities.append(density)
        per_prime.append(
            {
                "p": p,
                "dimension": spec.dimension,
                "size": size,
                "ambient_size": p**spec.dimension,
                "density": density,
            }
        )

    average_density = float(np.mean(densities))
    return {
        "average_density": average_density,
        "total_size": total_size,
        "per_prime": per_prime,
    }


class KakeyaRewardEvaluator(BaseRewardEvaluator):
    def __init__(
        self,
        problem_type: str,
        log_dir: str,
        num_cpus_per_task: int = 1,
        fail_score: float = 0.0,
        eval_timeout: int = 120,
        worst_perf_log: float = 1.0,
        env_type: str = "",
    ):
        self.problem_type = problem_type
        self.spec = parse_problem_type(problem_type)
        self.log_dir = log_dir
        self.num_cpus_per_task = max(1, int(num_cpus_per_task))
        self.fail_score = fail_score
        self.eval_timeout = max(1, int(eval_timeout))
        self.worst_perf_log = worst_perf_log
        self.env_type = env_type
        self._last_stdout = ""

    def _extract_code(self, response: str) -> str | None:
        match = re.search(r"```python\s+([\s\S]*?)\s*```", response)
        if match is None:
            return None
        return match.group(1).strip()

    def _get_failure_entry(self, msg: str) -> dict:
        return {
            "reward": self.fail_score,
            "msg": msg,
            "correctness": 0.0,
            "raw_score": self.worst_perf_log,
            "stdout": self._last_stdout,
            "metrics": {"error": msg},
        }

    def _build_program(self, generation: str) -> str:
        generation = re.sub(
            r"(?m)^\s*from\s+__future__\s+import\s+.*\n?",
            "",
            generation,
        )
        spec_src = """class KakeyaProblemSpec:
    def __init__(self, dimension, primes):
        self.dimension = int(dimension)
        self.primes = tuple(int(p) for p in primes)
"""
        helper_src = "\n\n".join(
            [
                spec_src,
                inspect.getsource(projective_directions_3d),
                inspect.getsource(normalize_point_array),
                inspect.getsource(point_set),
                inspect.getsource(first_missing_direction_3d),
                inspect.getsource(is_kakeya_set_3d),
                inspect.getsource(evaluate_kakeya_program_outputs),
            ]
        )
        primes_literal = repr(tuple(int(p) for p in self.spec.primes))
        return f"""from __future__ import annotations

import math
from itertools import product

import numpy as np

KAKEYA_D = {self.spec.dimension}
KAKEYA_PRIMES = {primes_literal}

{helper_src}

{generation.rstrip()}


def run(seed=42, budget_s={self.eval_timeout}, **kwargs):
    d = int(kwargs.get("d", KAKEYA_D))
    primes = tuple(kwargs.get("primes", KAKEYA_PRIMES))
    outputs = {{}}
    for p in primes:
        outputs[int(p)] = search_for_best_construction(int(p), d)
    return outputs
"""

    def get_reward(self, code: str, state: State) -> dict:
        del state
        generation = self._extract_code(code)
        if generation is None:
            return self._get_failure_entry("Cannot extract python code from model response")
        if "def search_for_best_construction" not in generation:
            return self._get_failure_entry(
                "Submission must define search_for_best_construction(p, d)"
            )

        program = self._build_program(generation)
        tmp_dir = os.path.join(self.log_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        program_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".py",
                delete=False,
                mode="w",
                dir=tmp_dir,
            ) as handle:
                program_path = handle.name
                handle.write(program)

            outputs = run_with_timeout(
                program_path,
                "run",
                timeout_seconds=self.eval_timeout,
                cpus=list(range(self.num_cpus_per_task)),
            )
            stdout_path = program_path + ".stdout"
            if os.path.exists(stdout_path):
                with open(stdout_path, "r", encoding="utf-8") as handle:
                    self._last_stdout = handle.read()
        except Exception as exc:
            return self._get_failure_entry(f"Evaluation failed: {exc}")
        finally:
            if program_path is not None:
                try:
                    os.unlink(program_path)
                except OSError:
                    pass
                try:
                    os.unlink(program_path + ".stdout")
                except OSError:
                    pass

        try:
            metrics = evaluate_kakeya_program_outputs(outputs, self.spec)
        except Exception as exc:
            return self._get_failure_entry(f"Invalid construction: {exc}")

        average_density = float(metrics["average_density"])
        reward = 1.0 / max(average_density, 1e-12)
        msg = (
            f"Success; average_density={average_density:.8f}, "
            f"total_size={metrics['total_size']}"
        )
        return {
            "reward": reward,
            "msg": msg,
            "correctness": 1.0,
            "raw_score": average_density,
            "result_construction": metrics["per_prime"],
            "stdout": self._last_stdout,
            "metrics": metrics,
        }


class KakeyaEnv(Environment):
    env_name = "kakeya"
    reward_function = KakeyaRewardEvaluator
    state_type = State
    max_construction_len = 64

    @classmethod
    def prepare_initial_program(
        cls,
        program: str,
        *,
        source_path: str,
        eval_timeout: int,
    ) -> str:
        del source_path, eval_timeout
        if "def search_for_best_construction" not in program:
            raise ValueError(
                "Initial Kakeya program must define search_for_best_construction(p, d)"
            )
        return program

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        spec = parse_problem_type(problem_type)
        per_prime = [
            {
                "p": p,
                "dimension": spec.dimension,
                "size": p**spec.dimension,
                "ambient_size": p**spec.dimension,
                "density": 1.0,
            }
            for p in spec.primes
        ]
        code = """```python
import numpy as np
from itertools import product


def search_for_best_construction(p, d):
    return np.array(list(product(range(p), repeat=d)), dtype=np.int64)
```"""
        return State(
            timestep=-1,
            construction=per_prime,
            code=code,
            value=-1.0,
        )

    def is_maximize(self) -> bool:
        return False

    def check_format(self, parsed_code: str) -> bool:
        if not parsed_code or not parsed_code.strip():
            return False
        return "def search_for_best_construction" in parsed_code

    def get_question(self) -> str:
        spec = parse_problem_type(self.problem_type)
        primes_text = ", ".join(str(p) for p in spec.primes)
        state_ctx = self.initial_state.to_prompt(
            0.30,
            metric_name="average Kakeya density",
            maximize=False,
        )
        budget_s = max(1, int(getattr(self, "eval_timeout", 120)))
        cpus = max(1, int(getattr(self, "num_cpus_per_task", 1)))

        return f"""You are working on finite-field Kakeya sets.

For each tested prime p, construct a small subset K of F_p^3 that contains a full affine line in every projective direction. Smaller sets are better.

The evaluator will test dimension d=3 and primes p in {{{primes_text}}}. It will call:

```python
search_for_best_construction(p, d)
```

Your function must return an array-like object of shape (n, d) with integer coordinates in [0, p). Duplicate rows are allowed but only unique points count toward size.

Rules:
- Return exactly one fenced Python code block.
- Define `search_for_best_construction(p, d)`.
- You may use `math`, `itertools`, and `numpy as np`.
- No filesystem or network IO.
- Your code must finish within {budget_s}s using {cpus} CPU(s).
- The construction must be valid for every tested prime.
- Reward improves as the average density `|K| / p^3` decreases.

Useful baseline: returning the whole space has density 1. Known algebraic constructions can get close to density 1/4 plus lower-order terms.

{state_ctx}

Reason about a construction that covers the three projective direction classes `(1,a,b)`, `(0,1,c)`, and `(0,0,1)`, then return the final program."""
