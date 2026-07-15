from __future__ import annotations

import inspect
import itertools
import os
import re
import shlex
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

from ttt_discover import BaseRewardEvaluator, Environment, State
from ttt_discover.tasks.sandbox_reward_evaluator import run_with_timeout


PriorityFn = Callable[[tuple[int, ...], int], float]


def parse_dimension(problem_type: str | int | None) -> int:
    if problem_type in (None, ""):
        return 8
    n = int(problem_type)
    if n < 1 or n > 10:
        raise ValueError(f"Unsupported cap-set dimension: {n}")
    return n


def solve_cap_set(n: int, priority_fn: PriorityFn) -> np.ndarray:
    """Greedily construct a cap set in F_3^n from a priority function."""
    all_vectors = np.array(
        list(itertools.product((0, 1, 2), repeat=n)), dtype=np.int16
    )
    powers = (3 ** np.arange(n - 1, -1, -1)).astype(np.int64)
    priorities = []
    for vector in all_vectors:
        value = float(priority_fn(tuple(int(x) for x in vector), n))
        if np.isnan(value) or value == np.inf:
            raise ValueError(f"priority returned an invalid value: {value}")
        priorities.append(value)
    priorities = np.array(priorities, dtype=float)

    capset = np.empty(shape=(0, n), dtype=np.int16)
    while np.any(priorities != -np.inf):
        max_index = int(np.argmax(priorities))
        vector = all_vectors[None, max_index]
        if capset.size:
            blocking = np.einsum("cn,n->c", (-capset - vector) % 3, powers)
            priorities[blocking] = -np.inf
        priorities[max_index] = -np.inf
        capset = np.concatenate([capset, vector], axis=0)
    return capset.astype(np.int8)


def is_cap_set(vectors: np.ndarray) -> bool:
    """Check that no three selected vectors form an affine line in F_3^n."""
    vectors = np.asarray(vectors)
    if vectors.ndim != 2:
        return False
    if vectors.size == 0:
        return False
    if not np.issubdtype(vectors.dtype, np.integer):
        return False
    if np.any(vectors < 0) or np.any(vectors > 2):
        return False

    _, n = vectors.shape
    powers = np.array([3**j for j in range(n - 1, -1, -1)], dtype=np.int64)
    raveled = np.einsum("in,n->i", vectors.astype(np.int64), powers)
    if len(set(map(int, raveled))) != len(raveled):
        return False

    is_blocked = np.full(shape=3**n, fill_value=False, dtype=bool)
    for i, (new_vector, new_index) in enumerate(zip(vectors, raveled)):
        new_index = int(new_index)
        if is_blocked[new_index]:
            return False
        if i >= 1:
            blocking = np.einsum(
                "nk,k->n",
                (-vectors[:i, :].astype(np.int64) - new_vector[None, :]) % 3,
                powers,
            )
            is_blocked[blocking] = True
        is_blocked[new_index] = True
    return True


def constant_priority(el: tuple[int, ...], n: int) -> float:
    return 0.0


class CapSetPriorityRewardEvaluator(BaseRewardEvaluator):
    def __init__(
        self,
        problem_type: str,
        log_dir: str,
        num_cpus_per_task: int = 1,
        fail_score: float = 0.0,
        eval_timeout: int = 45,
        worst_perf_log: float = 0.0,
        env_type: str = "",
    ):
        self.problem_type = problem_type
        self.log_dir = log_dir
        self.num_cpus_per_task = max(1, int(num_cpus_per_task))
        self.fail_score = fail_score
        self.eval_timeout = eval_timeout
        self.worst_perf_log = worst_perf_log
        self.env_type = env_type
        self._last_stdout = ""

    def _extract_code(self, response: str) -> str | None:
        match = re.search(r"```python\s+([\s\S]*?)\s*```", response)
        if match is None:
            return None
        return match.group(1).strip()

    def _build_program(self, generation: str) -> str:
        n = parse_dimension(self.problem_type)
        helper_src = "\n\n".join(
            [
                inspect.getsource(solve_cap_set),
                inspect.getsource(is_cap_set),
            ]
        )
        return f"""import itertools
import math
import numpy as np
from typing import Callable

PriorityFn = Callable[[tuple[int, ...], int], float]

CAP_SET_N = {n}

{helper_src}

{generation.rstrip()}


def run(seed=42, budget_s={max(1, int(self.eval_timeout))}, **kwargs):
    n = int(kwargs.get("n", CAP_SET_N))
    return solve_cap_set(n, priority)
"""

    def _get_failure_entry(self, msg: str) -> dict:
        return {
            "reward": self.fail_score,
            "msg": msg,
            "correctness": 0.0,
            "raw_score": self.worst_perf_log,
            "stdout": self._last_stdout,
        }

    def get_reward(self, code: str, state: State) -> dict:
        generation = self._extract_code(code)
        if generation is None:
            return self._get_failure_entry("Cannot extract python code from model response")

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

            output = run_with_timeout(
                program_path,
                "run",
                timeout_seconds=max(1, int(self.eval_timeout)),
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

        n = parse_dimension(self.problem_type)
        try:
            vectors = np.asarray(output, dtype=np.int64)
        except (TypeError, ValueError) as exc:
            return self._get_failure_entry(f"Cannot convert output to array: {exc}")

        if vectors.ndim != 2 or vectors.shape[1] != n:
            return self._get_failure_entry(
                f"Expected an array of shape (m, {n}), got {vectors.shape}"
            )
        if not is_cap_set(vectors):
            return self._get_failure_entry("Output is not a valid cap set.")

        size = int(vectors.shape[0])
        return {
            "reward": float(size),
            "msg": f"Success; cap_set_size={size}",
            "correctness": 1.0,
            "raw_score": float(size),
            "result_construction": vectors.astype(int).tolist(),
            "stdout": getattr(self, "_last_stdout", ""),
            "metrics": {"cap_set_size": size, "dimension": n},
        }


class CapSetPriorityEnv(Environment):
    env_name = "cap_set_priority"
    reward_function = CapSetPriorityRewardEvaluator
    state_type = State
    max_construction_len = 1200

    @staticmethod
    def construction_key(construction) -> tuple[tuple[int, ...], ...]:
        """Identify a cap set by its points, independent of greedy selection order."""
        return tuple(
            sorted(
                tuple(int(coordinate) for coordinate in vector)
                for vector in construction
            )
        )

    @classmethod
    def prepare_initial_program(
        cls,
        program: str,
        *,
        source_path: str,
        eval_timeout: int,
    ) -> str:
        if "def priority(" not in program:
            raise ValueError(f"Initial program {source_path} must define priority()")
        return program

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        n = parse_dimension(problem_type)
        vectors = solve_cap_set(n, constant_priority)
        code = """```python
def priority(el, n):
    return 0.0
```"""
        return State(
            timestep=-1,
            construction=vectors.astype(int).tolist(),
            code=code,
            value=float(vectors.shape[0]),
        )

    def is_maximize(self) -> bool:
        return True

    def check_format(self, parsed_code: str) -> bool:
        if not parsed_code or not parsed_code.strip():
            return False
        return "def priority(" in parsed_code

    def _initial_priority_source(self) -> str:
        code = getattr(self.initial_state, "code", "") or ""
        match = re.search(r"```python\s+([\s\S]*?)\s*```", code)
        if match is not None:
            return match.group(1).strip() + "\n"
        if "def priority(" in code:
            return code.strip() + "\n"
        return "def priority(el, n):\n    return 0.0\n"

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
                "Blackbox autonomous cap-set search requires a socket path or TCP port."
            )

        (workspace / "submission.py").write_text(
            self._initial_priority_source(),
            encoding="utf-8",
        )
        (workspace / "eval_client.py").write_text(
            _cap_set_blackbox_eval_client_source(
                problem_type=self.problem_type,
                socket_path=socket_path,
                host=host,
                port=port,
                timeout_s=max(1.0, float(eval_timeout)),
            ),
            encoding="utf-8",
        )

        evaluator_cmd = f"cd {shlex.quote(str(workspace))} && python eval_client.py"
        return f"""{prompt}

--- Autonomous Cap-Set Blackbox Search Mode ---
You may inspect files and run shell commands, but keep all edits inside this workspace:
{workspace}

Editable candidate:
{workspace / "submission.py"}

The local evaluator is a blackbox service. This workspace intentionally contains
only `submission.py` and `eval_client.py`; hidden evaluator internals are not
available here.

Run this evaluator after each revision:
{evaluator_cmd}

Do not edit `eval_client.py` to improve a score. Only `submission.py` is a valid
candidate artifact. `submission.py` must define `priority(el, n)` as plain Python
source, without markdown fences.

When done, put the best implementation in:
{workspace / "submission.py"}
"""

    def get_question(self) -> str:
        state = self.initial_state
        n = parse_dimension(self.problem_type)
        state_ctx = state.to_prompt(600, metric_name=f"cap-set size in F_3^{n}", maximize=True)
        current_size = len(state.construction) if state.construction else 0
        return f"""You are improving a priority function for the cap-set problem.

We greedily build a cap set in F_3^{n}. The evaluator enumerates all vectors in {{0,1,2}}^{n}, sorts them by your priority score, and repeatedly selects the highest-priority vector that has not been blocked by a previously selected pair. Higher final cap-set size is better.

Write one Python program that defines:

```python
def priority(el, n):
    ...
```

`el` is a tuple of ternary coordinates and `n` is the dimension. The evaluator will call `solve_cap_set({n}, priority)` and reward the integer size of the resulting valid cap set.

Rules:
- Return exactly one Python code block.
- Define `priority(el, n)`; helper functions are allowed.
- You may use `math` and `numpy as np`.
- Do not hard-code or return a cap set; only rank vectors through `priority`.
- No filesystem or network IO.

Current greedy cap-set size from the active state: {current_size}.
The baseline constant priority gives 256 in dimension 8; the current seed is intentionally modest and can be improved.

{state_ctx}

Reason about how to change the ranking so the greedy solver admits more vectors, then return the final program as one fenced Python code block."""


def _cap_set_blackbox_eval_client_source(
    *,
    problem_type: str = "",
    socket_path: str | None = None,
    host: str = "127.0.0.1",
    port: int | None = None,
    timeout_s: float = 3600.0,
) -> str:
    if socket_path is None and port is None:
        raise ValueError("Either socket_path or port must be provided")

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
        "submission": "```python\\n" + source.rstrip() + "\\n```",
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
