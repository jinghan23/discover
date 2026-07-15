import json
import os
import re
import shlex
from pathlib import Path

import numpy as np

from ttt_discover import Environment, SandboxRewardEvaluator, State, DiscoverConfig, discover


def verify_c5_solution(h_values: np.ndarray, c5_achieved: float, n_points: int):
    if not isinstance(h_values, np.ndarray):
        try:
            h_values = np.array(h_values, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert h_values to numpy array: {e}")
    
    if len(h_values.shape) != 1:
        raise ValueError(f"h_values must be 1D array, got shape {h_values.shape}")
    
    if h_values.shape[0] != n_points:
        raise ValueError(f"Expected h shape ({n_points},), got {h_values.shape}")
    
    if not np.all(np.isfinite(h_values)):
        raise ValueError("h_values contain NaN or inf values")
    
    bound_tol = 1e-9
    if np.any(h_values < -bound_tol) or np.any(h_values > 1 + bound_tol):
        raise ValueError(f"h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")
    h_values = np.clip(h_values, 0.0, 1.0)
    
    n = n_points
    target_sum = n / 2.0
    current_sum = np.sum(h_values)
    
    if not np.isclose(current_sum, target_sum, rtol=0.0, atol=1e-8):
        h_values = h_values * (target_sum / current_sum)
        if np.any(h_values < -bound_tol) or np.any(h_values > 1 + bound_tol):
            raise ValueError(f"After normalization, h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")
        h_values = np.clip(h_values, 0.0, 1.0)
    
    dx = 2.0 / n_points
    
    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    computed_c5 = np.max(correlation)
    
    if not np.isfinite(computed_c5):
        raise ValueError(f"Computed C5 is not finite: {computed_c5}")
    
    if not np.isclose(computed_c5, c5_achieved, atol=1e-4):
        raise ValueError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")
    
    return computed_c5


def evaluate_erdos_solution(h_values: np.ndarray, c5_bound: float, n_points: int) -> float:
    verify_c5_solution(h_values, c5_bound, n_points)
    return float(c5_bound)


def verify_erdos_solution(result: tuple[np.ndarray, float, int]) -> bool:
    try:
        h_values, c5_bound, n_points = result
        c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)
        if c5_bound <= 0 or np.isnan(c5_bound) or np.isinf(c5_bound):
            return False
    except Exception:
        return False
    return True


class ErdosMinOverlapRewardEvaluator(SandboxRewardEvaluator):
    def get_program_entrypoint(self) -> str:
        return "run"

    def _extract_code(self, response: str) -> str | None:
        match = re.search(r"```python\s+([\s\S]*?)\s*```", response)
        if match is not None:
            return match.group(1).strip()
        source = response.strip()
        return source if "def run(" in source else None

    def preprocess_generation(self, generation, state) -> str:
        import inspect
        verifier_src = inspect.getsource(verify_c5_solution)
        evaluator_src = inspect.getsource(evaluate_erdos_solution)
        numpy_import = "import numpy as np"
        
        base = (
            numpy_import
            + "\n\n"
            + verifier_src
            + "\n\n"
            + evaluator_src
            + "\n\n"
        )
        
        # State with construction is required - no silent fallback
        if state is None:
            raise ValueError(
                "state is required for preprocess_generation. "
                "Use ExperienceSampler to provide initial state with construction."
            )
        if state.construction is not None:
            initial_h_values = f"initial_h_values = np.array({list(state.construction)!r})"
            base += initial_h_values + "\n\n"

        return base + generation

    def get_reward(self, code: str, state: State) -> float:
        output, error_msg = self.execute_code(code, state)
        if error_msg: 
            return self._get_failure_entry(error_msg)

        if not verify_erdos_solution(output):
            return self._get_failure_entry("Invalid solution.")
        h_values, c5_bound, n_points = output
        c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)

        return {
            "reward": float(1.0 / (1e-8 + c5_bound)),
            "correctness": 1.0,
            "raw_score": c5_bound,
            "msg": f"C5 bound: {c5_bound}",
            "result_construction": list(h_values),
            "stdout": getattr(self, '_last_stdout', ''),
        }


class ErdosMinOverlapEnv(Environment):
    reward_function = ErdosMinOverlapRewardEvaluator
    state_type = State
    max_construction_len = 1000

    @classmethod
    def prepare_initial_program(
        cls,
        program: str,
        *,
        source_path: str,
        eval_timeout: int,
    ) -> str:
        if source_path.endswith(".json"):
            data = json.loads(program)
            if "h_values" not in data:
                raise ValueError(
                    f"Initial JSON construction {source_path} must contain h_values"
                )
            h_values = data["h_values"]
            program = (
                "import numpy as np\n\n"
                f"h_values = np.array({h_values!r}, dtype=float)\n"
            )
        if "def run(" in program:
            return program
        if "h_values" not in program:
            raise ValueError(
                f"Initial program {source_path} must define run() or h_values"
            )
        budget_s = max(1, int(eval_timeout))
        return program.rstrip() + f'''


def run(seed=42, budget_s={budget_s}, **kwargs):
    h = np.asarray(h_values, dtype=float).copy()
    n_points = int(h.size)
    dx = 2.0 / n_points
    c5_bound = float(np.max(np.correlate(h, 1.0 - h, mode="full") * dx))
    return h, c5_bound, n_points
'''

    @classmethod
    def create_seed_state_from_initial_program(
        cls,
        program: str,
        *,
        source_path: str,
        eval_timeout: int,
    ) -> State | None:
        del source_path, eval_timeout
        match = re.search(r"```python\s+([\s\S]*?)\s*```", program)
        code = match.group(1).strip() if match is not None else program
        namespace: dict[str, object] = {"np": np}
        exec(code, namespace)
        if "h_values" not in namespace:
            return None
        h_values = np.asarray(namespace["h_values"], dtype=float)
        n_points = int(h_values.size)
        dx = 2.0 / n_points
        c5_bound = float(np.max(np.correlate(h_values, 1.0 - h_values, mode="full") * dx))
        c5_bound = verify_c5_solution(h_values, c5_bound, n_points)
        return State(
            timestep=-1,
            code=program,
            value=-c5_bound,
            construction=list(h_values),
        )

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        rng = np.random.default_rng()
        n_points = rng.integers(40, 100)
        construction = np.ones(n_points) * 0.5
        perturbation = rng.uniform(-0.4, 0.4, n_points)
        perturbation = perturbation - np.mean(perturbation)
        construction = construction + perturbation
        dx = 2.0 / n_points
        correlation = np.correlate(construction, 1 - construction, mode="full") * dx
        c5_bound = float(np.max(correlation))
        return State(timestep=-1, code="", value=-c5_bound, construction=list(construction))

    def _initial_submission_source(self) -> str:
        code = getattr(self.initial_state, "code", "") or ""
        match = re.search(r"```python\s+([\s\S]*?)\s*```", code)
        if match is not None:
            return match.group(1).strip() + "\n"
        if "def run(" in code:
            return code.strip() + "\n"
        budget_s = max(1, int(self.eval_timeout))
        return f'''import numpy as np


def run(seed=42, budget_s={budget_s}, **kwargs):
    h = np.asarray(initial_h_values, dtype=float).copy()
    n_points = int(h.size)
    c5_bound = float(
        np.max(np.correlate(h, 1.0 - h, mode="full") * (2.0 / n_points))
    )
    return h, c5_bound, n_points
'''

    def build_blackbox_autonomous_prompt(
        self,
        *,
        prompt: str,
        workspace: Path,
        eval_timeout: int,
        num_cpus_per_task: int,
        socket_path: str | None = None,
        host: str = "127.0.0.1",
        port: int | None = None,
    ) -> str:
        del num_cpus_per_task
        socket_path = socket_path or os.environ.get("TTT_BLACKBOX_EVAL_SOCKET")
        host = os.environ.get("TTT_BLACKBOX_EVAL_HOST") or host
        env_port = os.environ.get("TTT_BLACKBOX_EVAL_PORT")
        if port is None and env_port:
            port = int(env_port)
        if socket_path is None and port is None:
            raise ValueError(
                "Blackbox autonomous Erdos search requires a socket path or TCP port."
            )

        (workspace / "submission.py").write_text(
            self._initial_submission_source(),
            encoding="utf-8",
        )
        (workspace / "eval_client.py").write_text(
            _erdos_blackbox_eval_client_source(
                problem_type=self.problem_type,
                socket_path=socket_path,
                host=host,
                port=port,
                timeout_s=max(1.0, float(eval_timeout)),
                state=self.initial_state.to_dict(),
            ),
            encoding="utf-8",
        )

        evaluator_cmd = f"cd {shlex.quote(str(workspace))} && python eval_client.py"
        return f"""{prompt}

--- Autonomous Erdos Blackbox Search Mode ---
You may inspect files and run shell commands, but keep all edits inside this workspace:
{workspace}

Editable candidate:
{workspace / "submission.py"}

The local evaluator is a blackbox service. The parent construction is supplied
to the trusted evaluator as `initial_h_values`; evaluator internals are not
available in this workspace.

Run this evaluator after each revision:
{evaluator_cmd}

Do not edit `eval_client.py` to improve a score. Only `submission.py` is a valid
candidate artifact. It must be plain Python source defining `run(...)`.

When done, put the best implementation in:
{workspace / "submission.py"}
"""

    def is_maximize(self) -> bool:
        return False # Minimize upper bound

    def get_question(self) -> str:
        state = self.initial_state
        state_ctx = state.to_prompt(0.3808, metric_name="C₅ bound", maximize=False)
        budget_s = max(1, int(getattr(self, "eval_timeout", 1000)))
        cpus = max(1, int(getattr(self, "num_cpus_per_task", 1)))
        
        # Construct construction section
        construction_section = ""
        if hasattr(state, 'construction') and state.construction is not None and len(state.construction) > 0:
            construction_section = f"""
You may want to start your search from the current construction, which you can access through the `initial_h_values` global variable (n={len(state.construction)} samples).
You are encouraged to explore solutions that use other starting points to prevent getting stuck in a local optimum.
"""

        # Construct code section
        if state.code and state.code.strip():
            code_section = '''Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc. 
Unless you make a meaningful improvement, you will not be rewarded.'''
        else:
            code_section = '''Write code to optimize this construction.'''

        # Construct final prompt
        return f'''You are an expert in harmonic analysis, numerical optimization, and mathematical discovery.
Your task is to find an improved upper bound for the Erdős minimum overlap problem constant C₅.

## Problem

Find a step function h: [0, 2] → [0, 1] that **minimizes** the overlap integral:

$$C_5 = \\max_k \\int h(x)(1 - h(x+k)) dx$$

**Constraints**:
1. h(x) ∈ [0, 1] for all x
2. ∫₀² h(x) dx = 1

**Discretization**: Represent h as n_points samples over [0, 2].
With dx = 2.0 / n_points:
- 0 ≤ h[i] ≤ 1 for all i
- sum(h) * dx = 1 (equivalently: sum(h) == n_points / 2 exactly)

The evaluation computes: C₅ = max(np.correlate(h, 1-h, mode="full") * dx)

Smaller sequences with less than 1k samples are preferred - they are faster to optimize and evaluate.

**Lower C₅ values are better** - they provide tighter upper bounds on the Erdős constant.

## Budget & Resources
- **Time budget**: {budget_s}s for your code to run
- **CPUs**: {cpus} available

## Rules
- Define `run(seed=42, budget_s={budget_s}, **kwargs)` that returns `(h_values, c5_bound, n_points)`
- Use scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math
- Make all helper functions top level, no closures or lambdas
- No filesystem or network IO
- `evaluate_erdos_solution()` and `initial_h_values` (an initial construction, if available) are pre-imported
- Your function must complete within budget_s seconds and return the best solution found

**Lower is better**. Current record: C₅ ≤ 0.38092. Our goal is to find a construction that shows C₅ ≤ 0.38080.

{state_ctx}
{construction_section}
{code_section}
'''


def _erdos_blackbox_eval_client_source(
    *,
    state: dict[str, object],
    problem_type: str = "",
    socket_path: str | None = None,
    host: str = "127.0.0.1",
    port: int | None = None,
    timeout_s: float = 3600.0,
) -> str:
    if socket_path is None and port is None:
        raise ValueError("Either socket_path or port must be provided")

    state_json = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    return f'''#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
import uuid


PROBLEM_TYPE = {problem_type!r}
SOCKET_PATH = {socket_path!r}
HOST = {host!r}
PORT = {port!r}
TIMEOUT_S = {float(timeout_s)!r}
STATE = json.loads({state_json!r})


def _connect() -> socket.socket:
    socket_path = os.environ.get("TTT_BLACKBOX_EVAL_SOCKET") or SOCKET_PATH
    port_value = os.environ.get("TTT_BLACKBOX_EVAL_PORT")
    host_value = os.environ.get("TTT_BLACKBOX_EVAL_HOST") or HOST
    if socket_path:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(TIMEOUT_S)
        sock.connect(socket_path)
        return sock
    port = int(port_value) if port_value else PORT
    if port is None:
        raise SystemExit("missing blackbox eval runner socket or TCP port")
    sock = socket.create_connection((host_value, int(port)), timeout=TIMEOUT_S)
    sock.settimeout(TIMEOUT_S)
    return sock


def _recv_line(sock: socket.socket) -> bytes:
    chunks = []
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
        if b"\\n" in chunk:
            break
    return b"".join(chunks).split(b"\\n", 1)[0]


def main() -> int:
    submission_path = sys.argv[1] if len(sys.argv) > 1 else "submission.py"
    source = Path(submission_path).read_text(encoding="utf-8")
    request = {{
        "request_id": str(uuid.uuid4()),
        "problem_type": PROBLEM_TYPE,
        "submission": source,
        "state": STATE,
    }}
    data = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode() + b"\\n"
    with _connect() as sock:
        sock.sendall(data)
        response_data = _recv_line(sock)
    response = json.loads(response_data.decode("utf-8"))

    message = str(response.get("message") or "")
    if response.get("ok"):
        print(message or "pass")
        if response.get("raw_score") is not None:
            print(f"raw_score {{response['raw_score']}}")
        if response.get("reward") is not None:
            print(f"reward {{response['reward']}}")
        return 0

    stage = response.get("stage") or "eval"
    print(f"FAIL stage={{stage}} message={{message or 'evaluation failed'}}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def discover_erdos_min_overlap():
    config = DiscoverConfig(
        env_type=ErdosMinOverlapEnv,
        problem_type="",
        num_cpus_per_task=1,
        eval_timeout=1100,
        experiment_name=f"test-erdos-min-overlap-run",
        wandb_project="erdos-min-overlap",
    )

    discover(config)
    

if __name__ == "__main__":
    discover_erdos_min_overlap()
