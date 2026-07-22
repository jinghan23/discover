#!/usr/bin/env python3
"""Protected three-task meta-evaluator for recursive AutoEvolve.

This module is intentionally outside the outer agent's editable scope.  It
checks a committed harness candidate, evaluates it in a detached worktree on
three fresh one-epoch AutoEvolve runs, privately re-scores the exact best
WhestBench public submission, and advances an external incumbent ledger only
when the fixed selection rule accepts the candidate.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = Path("/opt/tiger/recursive_autoevolve_runs")
FIXED_BRANCH = "codex/recursive-autoevolve-task-description"
EDITABLE_PREFIXES = (
    "ttt_discover/algorithms/autoevolve/",
    "ttt_discover/codex_utils/autonomous.py",
)

INNER_EPOCHS = 1
MAX_EVALUATOR_CALLS: int | None = None
TASK_WALL_TIME_SECONDS = 3_600
AUTONOMOUS_SEARCH_SECONDS = 3_000
MODEL_NAME = "gpt-5.5"
GPU_DEVICES = (7,)
WHEST_PUBLIC_SEEDS = tuple(range(0, 50))
WHEST_PRIVATE_SEEDS = tuple(range(50, 100))
WHEST_WIDTH = 64
WHEST_DEPTH = 8
WHEST_REFERENCE_SAMPLES = 4096
ERDOS_INITIAL_SEED = 20260716
ERDOS_INITIAL_POINTS = 96

# Phase-one selection rule.  All task scores are losses (lower is better).
MIN_GEOMEAN_IMPROVEMENT = 1.01
MAX_SINGLE_TASK_REGRESSION = 0.02
MIN_IMPROVED_TASKS = 2


@dataclass(frozen=True)
class TaskSpec:
    name: str
    env_type: str
    problem_type: str
    eval_timeout: int
    cli_timeout: int
    process_timeout: int
    uses_gpu: bool = False


TASK_SPECS = (
    TaskSpec(
        name="erdos",
        env_type="examples.erdos_min_overlap.env:ErdosMinOverlapEnv",
        problem_type="",
        eval_timeout=500,
        cli_timeout=AUTONOMOUS_SEARCH_SECONDS,
        process_timeout=TASK_WALL_TIME_SECONDS,
    ),
    TaskSpec(
        name="arc_whestbench",
        env_type="examples.aicrowd_whestbench.env:WhestBenchEnv",
        problem_type="arc_whestbench_2026",
        eval_timeout=1200,
        cli_timeout=AUTONOMOUS_SEARCH_SECONDS,
        process_timeout=TASK_WALL_TIME_SECONDS,
    ),
    TaskSpec(
        name="kernel",
        env_type="examples.gpu_mode.env:GpuModeEnv",
        problem_type="trimul",
        eval_timeout=1200,
        cli_timeout=AUTONOMOUS_SEARCH_SECONDS,
        process_timeout=TASK_WALL_TIME_SECONDS,
        uses_gpu=True,
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id(role: str, sha: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{role}_{sha[:12]}_{uuid.uuid4().hex[:8]}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    return str(value)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(dict(payload)), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _git(*args: str, cwd: Path = REPO_ROOT, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _resolve_commit(ref: str) -> str:
    return _git("rev-parse", "--verify", f"{ref}^{{commit}}")


def _current_branch() -> str:
    return _git("branch", "--show-current")


def _tracked_worktree_clean() -> bool:
    return not _git("status", "--porcelain", "--untracked-files=no")


def _changed_paths(base_sha: str, candidate_sha: str) -> list[str]:
    output = _git("diff", "--name-only", base_sha, candidate_sha)
    return [line for line in output.splitlines() if line]


def _path_is_editable(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix) for prefix in EDITABLE_PREFIXES)


def validate_candidate_scope(incumbent_sha: str, candidate_sha: str) -> list[str]:
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", incumbent_sha, candidate_sha],
        cwd=REPO_ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError(
            f"incumbent {incumbent_sha} is not an ancestor of candidate {candidate_sha}"
        )
    changed = _changed_paths(incumbent_sha, candidate_sha)
    if not changed:
        raise ValueError("candidate has no tree changes relative to the incumbent")
    forbidden = [path for path in changed if not _path_is_editable(path)]
    if forbidden:
        raise ValueError(
            "candidate changed protected paths: " + ", ".join(sorted(forbidden))
        )
    return changed


def _protocol_payload() -> dict[str, Any]:
    return {
        "version": 2,
        "fixed_branch": FIXED_BRANCH,
        "editable_prefixes": EDITABLE_PREFIXES,
        "inner": {
            "algorithm": "autoevolve",
            "num_epochs": INNER_EPOCHS,
            "max_evaluator_calls": MAX_EVALUATOR_CALLS,
            "task_wall_time_seconds": TASK_WALL_TIME_SECONDS,
            "autonomous_search_seconds": AUTONOMOUS_SEARCH_SECONDS,
            "model": MODEL_NAME,
            "group_size": 1,
            "groups_per_batch": 1,
            "max_concurrent_requests": 1,
        },
        "tasks": [spec.__dict__ for spec in TASK_SPECS],
        "erdos_initial": {
            "seed": ERDOS_INITIAL_SEED,
            "n_points": ERDOS_INITIAL_POINTS,
        },
        "whestbench_manifest": {
            "width": WHEST_WIDTH,
            "depth": WHEST_DEPTH,
            "reference_samples": WHEST_REFERENCE_SAMPLES,
            "public_seeds": WHEST_PUBLIC_SEEDS,
            "private_seeds": WHEST_PRIVATE_SEEDS,
        },
        "kernel": {"task": "trimul", "gpu_devices": GPU_DEVICES},
        "selection": {
            "direction": "lower_is_better",
            "min_geomean_improvement": MIN_GEOMEAN_IMPROVEMENT,
            "max_single_task_regression": MAX_SINGLE_TASK_REGRESSION,
            "min_improved_tasks": MIN_IMPROVED_TASKS,
        },
    }


def _protocol_hash() -> str:
    encoded = json.dumps(
        _protocol_payload(), separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@contextlib.contextmanager
def _global_lock(run_root: Path) -> Iterator[None]:
    run_root.mkdir(parents=True, exist_ok=True)
    path = run_root / ".meta_eval.lock"
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started={_utc_now()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _detached_worktree(run_root: Path, sha: str, run_id: str) -> Iterator[Path]:
    path = run_root / "worktrees" / run_id
    path.parent.mkdir(parents=True, exist_ok=True)
    _git("worktree", "add", "--detach", str(path), sha)
    try:
        yield path
    finally:
        _git("worktree", "remove", "--force", str(path), check=False)
        _git("worktree", "prune", check=False)


@contextlib.contextmanager
def _reserved_gpu(devices: Sequence[int] = GPU_DEVICES) -> Iterator[int]:
    handles: list[Any] = []
    try:
        while True:
            for device in devices:
                path = Path(f"/tmp/recursive_autoevolve_gpu_{device}.lock")
                handle = path.open("a+", encoding="utf-8")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                handle.seek(0)
                handle.truncate()
                handle.write(f"pid={os.getpid()} device={device} started={_utc_now()}\n")
                handle.flush()
                handles.append(handle)
                yield device
                return
            print("Protected GPU 7 is busy; waiting for the device...", flush=True)
            time.sleep(10)
    finally:
        for handle in handles:
            with contextlib.suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()


def _erdos_initial_state() -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(ERDOS_INITIAL_SEED)
    perturbation = rng.uniform(-0.35, 0.35, ERDOS_INITIAL_POINTS)
    perturbation -= float(np.mean(perturbation))
    construction = np.full(ERDOS_INITIAL_POINTS, 0.5) + perturbation
    dx = 2.0 / ERDOS_INITIAL_POINTS
    c5 = float(
        np.max(np.correlate(construction, 1.0 - construction, mode="full") * dx)
    )
    return {
        "type": "State",
        "id": f"protected-erdos-seed-{ERDOS_INITIAL_SEED}",
        "timestep": -1,
        "value": -c5,
        "parent_values": [],
        "parents": [],
        "observation": "",
        "construction": construction.tolist(),
        "code": "",
        "metadata": {
            "protected_seed": ERDOS_INITIAL_SEED,
            "n_points": ERDOS_INITIAL_POINTS,
        },
    }


def _task_environment(spec: TaskSpec, run_dir: Path, gpu_device: int | None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if spec.name == "arc_whestbench":
        env.update(
            {
                "WHEST_LOCAL_WIDTH": str(WHEST_WIDTH),
                "WHEST_LOCAL_DEPTH": str(WHEST_DEPTH),
                "WHEST_LOCAL_REFERENCE_SAMPLES": str(WHEST_REFERENCE_SAMPLES),
                "WHEST_LOCAL_SEEDS": ",".join(map(str, WHEST_PUBLIC_SEEDS)),
                "WHEST_LOCAL_BUDGET": str(int(1e9)),
            }
        )
    if spec.uses_gpu:
        if gpu_device is None:
            raise ValueError("kernel task requires a reserved GPU")
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu_device),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "TORCH_CUDA_ARCH_LIST": "8.0",
                "TRITON_CACHE_DIR": str(run_dir / "triton_cache"),
                "TORCH_EXTENSIONS_DIR": str(run_dir / "torch_extensions"),
            }
        )
    return env


def _inner_config(spec: TaskSpec, run_dir: Path, initial_state: Path | None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "env_type": spec.env_type,
        "problem_type": spec.problem_type,
        "experiment_name": f"recursive-{spec.name}",
        "log_path": str(run_dir),
        "algorithm": "autoevolve",
        "eval_runner": "blackbox",
        "backend": "cli",
        "model_name": MODEL_NAME,
        "cli_command": "codex",
        "cli_sandbox": "danger-full-access",
        "cli_timeout": spec.cli_timeout,
        "max_concurrent_requests": 1,
        "num_epochs": INNER_EPOCHS,
        "max_evaluator_calls": MAX_EVALUATOR_CALLS,
        "group_size": 1,
        "groups_per_batch": 1,
        "num_cpus_per_task": 1,
        "eval_timeout": spec.eval_timeout,
        "timeout": float(spec.eval_timeout + 120),
        "wandb_project": "",
        "remove_constant_reward_groups": False,
    }
    if initial_state is not None:
        config["initial_state_file"] = str(initial_state)
    return {"runs": {"inner": config}}


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def _launch_inner_process(
    worktree: Path,
    spec: TaskSpec,
    run_dir: Path,
    *,
    gpu_device: int | None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    initial_path: Path | None = None
    if spec.name == "erdos":
        initial_path = run_dir / "initial_state.json"
        _atomic_json(initial_path, {"states": [_erdos_initial_state()]})

    config_path = run_dir / "inner_config.json"
    _atomic_json(config_path, _inner_config(spec, run_dir, initial_path))
    stdout_path = run_dir / "launcher.stdout.log"
    stderr_path = run_dir / "launcher.stderr.log"
    command = [
        sys.executable,
        "repro/run_from_yaml.py",
        "--config",
        str(config_path),
        "--run",
        "inner",
    ]
    _atomic_json(
        run_dir / "launch.json",
        {
            "command": command,
            "cwd": str(worktree),
            "gpu_device": gpu_device,
            "started_at": _utc_now(),
        },
    )

    started = time.monotonic()
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=worktree,
            env=_task_environment(spec, run_dir, gpu_device),
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=spec.process_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(process)
            returncode = process.returncode
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_time_seconds": time.monotonic() - started,
        "gpu_device": gpu_device,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def _run_task(worktree: Path, spec: TaskSpec, commit_dir: Path) -> dict[str, Any]:
    run_dir = commit_dir / spec.name
    print(f"[{spec.name}] starting fresh AutoEvolve run", flush=True)
    if spec.uses_gpu:
        with _reserved_gpu() as device:
            process_result = _launch_inner_process(
                worktree, spec, run_dir, gpu_device=device
            )
    else:
        process_result = _launch_inner_process(
            worktree, spec, run_dir, gpu_device=None
        )
    summary = summarize_inner_run(spec.name, run_dir, process_result)
    print(
        f"[{spec.name}] finished valid={summary['valid']} "
        f"calls={summary['evaluator_calls']} score={summary.get('best_score')}",
        flush=True,
    )
    return summary


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _evaluator_calls(run_dir: Path) -> int:
    used = 0
    for row in _read_json_lines(run_dir / "metrics.jsonl"):
        value = row.get("budget/evaluator_calls_used")
        try:
            used = max(used, int(value))
        except (TypeError, ValueError):
            pass
    for row in _read_json_lines(run_dir / "agent_outputs.jsonl"):
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        value = metrics.get("budget/evaluator_calls_server")
        try:
            used = max(used, int(value))
        except (TypeError, ValueError):
            pass
    return used


_TOKEN_RE = re.compile(r"tokens used\s*[:\r\n ]+([0-9][0-9,]*)", re.IGNORECASE)


def _token_usage(run_dir: Path) -> int | None:
    total = 0
    matched = False
    for pattern in ("**/codex.stdout.log", "**/codex.stderr.log"):
        for path in run_dir.glob(pattern):
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in _TOKEN_RE.finditer(text):
                matched = True
                total += int(match.group(1).replace(",", ""))
    return total if matched else None


def _latest_pool(run_dir: Path) -> Path | None:
    pools = sorted(run_dir.glob("autoevolve_pool_step_*.json"))
    return pools[-1] if pools else None


def _best_state(task_name: str, run_dir: Path) -> tuple[dict[str, Any] | None, float | None]:
    pool_path = _latest_pool(run_dir)
    if pool_path is None:
        return None, None
    try:
        payload = json.loads(pool_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    states = payload.get("states") if isinstance(payload, dict) else None
    if not isinstance(states, list):
        return None, None

    candidates: list[tuple[float, dict[str, Any]]] = []
    for state in states:
        if not isinstance(state, dict):
            continue
        value = _finite_float(state.get("value"))
        if value is None:
            continue
        code = str(state.get("code") or "")
        if task_name == "kernel" and "def custom_kernel" not in code:
            continue
        candidates.append((value, state))
    if not candidates:
        return None, None
    value, state = max(candidates, key=lambda pair: pair[0])
    score = -value
    if not math.isfinite(score) or score < 0:
        return None, None
    return state, score


def summarize_inner_run(
    task_name: str,
    run_dir: Path,
    process_result: Mapping[str, Any],
) -> dict[str, Any]:
    calls = _evaluator_calls(run_dir)
    best_state, best_score = _best_state(task_name, run_dir)
    code = str(best_state.get("code") or "") if best_state else ""
    best_submission: str | None = None
    if code.strip():
        submission_path = run_dir / "best_submission.py"
        submission_path.write_text(code, encoding="utf-8")
        best_submission = str(submission_path)

    failures: list[str] = []
    if int(process_result.get("returncode") or 0) != 0:
        failures.append(f"inner process exited {process_result.get('returncode')}")
    if process_result.get("timed_out"):
        failures.append("inner process timed out")
    if MAX_EVALUATOR_CALLS is not None and calls > MAX_EVALUATOR_CALLS:
        failures.append(
            f"evaluator budget exceeded: {calls}/{MAX_EVALUATOR_CALLS}"
        )
    if best_score is None:
        failures.append("no valid scored state")
    if task_name == "arc_whestbench" and not best_submission:
        failures.append("no public submission available for private evaluation")

    return {
        "run_ref": str(run_dir),
        "valid": not failures,
        "failures": failures,
        "best_score": best_score,
        "best_submission": best_submission,
        "evaluator_calls": calls,
        "max_evaluator_calls": MAX_EVALUATOR_CALLS,
        "token_usage": _token_usage(run_dir),
        **dict(process_result),
    }


def _private_whest_worker(candidate: Path, output: Path) -> int:
    os.environ.update(
        {
            "WHEST_LOCAL_WIDTH": str(WHEST_WIDTH),
            "WHEST_LOCAL_DEPTH": str(WHEST_DEPTH),
            "WHEST_LOCAL_REFERENCE_SAMPLES": str(WHEST_REFERENCE_SAMPLES),
            "WHEST_LOCAL_SEEDS": ",".join(map(str, WHEST_PRIVATE_SEEDS)),
            "WHEST_LOCAL_BUDGET": str(int(1e9)),
        }
    )
    from examples.aicrowd_whestbench.env import WhestBenchRewardEvaluator
    from ttt_discover import State

    code = candidate.read_text(encoding="utf-8")
    evaluator = WhestBenchRewardEvaluator(problem_type="arc_whestbench_2026")
    result = evaluator.get_reward(code, State(-1, [], "", 0.0))
    aggregate = {
        "private_best_score": _finite_float(result.get("raw_score")),
        "correctness": _finite_float(result.get("correctness")),
        "reward": _finite_float(result.get("reward")),
        "evaluated_instances": len(WHEST_PRIVATE_SEEDS),
        "private_evaluator_calls": 1,
    }
    _atomic_json(output, aggregate)
    return 0


def _private_whest_eval(worktree: Path, task: dict[str, Any]) -> dict[str, Any]:
    submission = task.get("best_submission")
    if not submission:
        return {"valid": False, "failure": "missing public best submission"}
    run_dir = Path(task["run_ref"])
    output = run_dir / "private_aggregate.json"
    stdout_path = run_dir / "private.stdout.log"
    stderr_path = run_dir / "private.stderr.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_private-whest",
        "--candidate",
        str(submission),
        "--output",
        str(output),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(worktree) + os.pathsep + env.get("PYTHONPATH", "")
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        result = subprocess.run(
            command,
            cwd=worktree,
            env=env,
            stdout=stdout,
            stderr=stderr,
            timeout=3600,
            check=False,
        )
    if result.returncode != 0 or not output.is_file():
        return {
            "valid": False,
            "failure": f"private evaluator exited {result.returncode}",
        }
    payload = json.loads(output.read_text(encoding="utf-8"))
    private_score = _finite_float(payload.get("private_best_score"))
    correctness = _finite_float(payload.get("correctness"))
    valid = private_score is not None and correctness is not None and correctness > 0
    return {"valid": valid, **payload}


def evaluate_commit(run_root: Path, sha: str, *, role: str) -> dict[str, Any]:
    run_id = _run_id(role, sha)
    commit_dir = run_root / "commit_runs" / run_id
    commit_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result_path = commit_dir / "commit_result.json"
    result: dict[str, Any] = {
        "schema_version": 1,
        "role": role,
        "commit_sha": sha,
        "run_ref": str(commit_dir),
        "result_path": str(result_path),
        "started_at": _utc_now(),
        "protocol_hash": _protocol_hash(),
        "protocol": _protocol_payload(),
        "tasks": {},
        "valid": False,
    }
    _atomic_json(result_path, result)

    try:
        with _detached_worktree(run_root, sha, run_id) as worktree:
            task_results: dict[str, dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=len(TASK_SPECS)) as executor:
                futures = {
                    executor.submit(_run_task, worktree, spec, commit_dir): spec.name
                    for spec in TASK_SPECS
                }
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        task_results[name] = future.result()
                    except Exception as exc:
                        task_results[name] = {
                            "run_ref": str(commit_dir / name),
                            "valid": False,
                            "failures": [f"launcher exception: {exc}"],
                            "best_score": None,
                            "evaluator_calls": 0,
                            "max_evaluator_calls": MAX_EVALUATOR_CALLS,
                        }

            whest = task_results["arc_whestbench"]
            private = _private_whest_eval(worktree, whest)
            whest["public_best_score"] = whest.pop("best_score", None)
            whest.update(private)
            public_score = _finite_float(whest.get("public_best_score"))
            private_score = _finite_float(whest.get("private_best_score"))
            if public_score is not None and private_score is not None:
                whest["public_private_gap"] = private_score - public_score
                whest["public_private_ratio"] = (
                    private_score / public_score if public_score > 0 else None
                )
            if not private.get("valid"):
                whest["valid"] = False
                whest.setdefault("failures", []).append(
                    str(private.get("failure") or "private evaluation failed")
                )

            result["tasks"] = task_results
            result["valid"] = all(task.get("valid") for task in task_results.values())
            result["total_evaluator_calls"] = sum(
                int(task.get("evaluator_calls") or 0)
                for task in task_results.values()
            )
            token_values = [
                int(task["token_usage"])
                for task in task_results.values()
                if task.get("token_usage") is not None
            ]
            result["total_token_usage"] = sum(token_values) if token_values else None
            result["compute_usage"] = {
                "task_wall_seconds": {
                    name: task.get("wall_time_seconds")
                    for name, task in task_results.items()
                },
                "kernel_gpu_device": task_results["kernel"].get("gpu_device"),
                "kernel_gpu_allocated_seconds": task_results["kernel"].get(
                    "wall_time_seconds"
                ),
            }
    except Exception as exc:
        result["failure"] = str(exc)
        result["valid"] = False
    finally:
        result["finished_at"] = _utc_now()
        result["wall_time_seconds"] = time.monotonic() - started
        _atomic_json(result_path, result)
    return result


def _selection_losses(evaluation: Mapping[str, Any]) -> dict[str, float] | None:
    tasks = evaluation.get("tasks")
    if not isinstance(tasks, Mapping):
        return None
    raw = {
        "erdos": tasks.get("erdos", {}).get("best_score"),
        "arc_whestbench": tasks.get("arc_whestbench", {}).get(
            "private_best_score"
        ),
        "kernel": tasks.get("kernel", {}).get("best_score"),
    }
    losses: dict[str, float] = {}
    for name, value in raw.items():
        score = _finite_float(value)
        if score is None or score <= 0:
            return None
        losses[name] = score
    return losses


def select_candidate(
    incumbent: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    incumbent_losses = _selection_losses(incumbent)
    candidate_losses = _selection_losses(candidate)
    reasons: list[str] = []
    if not incumbent.get("valid") or incumbent_losses is None:
        reasons.append("incumbent evaluation is invalid")
    if not candidate.get("valid") or candidate_losses is None:
        reasons.append("candidate evaluation is invalid")
    if reasons:
        return {
            "accepted": False,
            "selection_score": None,
            "ratios": {},
            "reasons": reasons,
        }

    assert incumbent_losses is not None and candidate_losses is not None
    ratios = {
        name: incumbent_losses[name] / candidate_losses[name]
        for name in incumbent_losses
    }
    geomean = math.exp(sum(math.log(value) for value in ratios.values()) / len(ratios))
    improved_tasks = sum(value > 1.0 for value in ratios.values())
    worst_ratio = min(ratios.values())
    if geomean < MIN_GEOMEAN_IMPROVEMENT:
        reasons.append(
            f"geomean ratio {geomean:.6f} < {MIN_GEOMEAN_IMPROVEMENT:.6f}"
        )
    if worst_ratio < 1.0 - MAX_SINGLE_TASK_REGRESSION:
        reasons.append(
            f"one-task ratio {worst_ratio:.6f} < "
            f"{1.0 - MAX_SINGLE_TASK_REGRESSION:.6f}"
        )
    if improved_tasks < MIN_IMPROVED_TASKS:
        reasons.append(
            f"improved tasks {improved_tasks} < {MIN_IMPROVED_TASKS}"
        )
    return {
        "accepted": not reasons,
        "selection_score": geomean,
        "ratios": ratios,
        "incumbent_losses": incumbent_losses,
        "candidate_losses": candidate_losses,
        "improved_tasks": improved_tasks,
        "reasons": reasons,
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_incumbent(run_root: Path) -> dict[str, Any] | None:
    path = run_root / "incumbent.json"
    return _load_json(path) if path.is_file() else None


def _record_ledger(run_root: Path, record: Mapping[str, Any]) -> None:
    path = run_root / "ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(dict(record)), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_operator_context(candidate_sha: str) -> None:
    if _current_branch() != FIXED_BRANCH:
        raise ValueError(
            f"meta-evaluation must run on fixed branch {FIXED_BRANCH!r}; "
            f"found {_current_branch()!r}"
        )
    if not _tracked_worktree_clean():
        raise ValueError("tracked files must be clean before meta-evaluation")
    head = _resolve_commit("HEAD")
    if candidate_sha != head:
        raise ValueError(
            f"candidate must be committed HEAD ({head}), got {candidate_sha}"
        )


def bootstrap_incumbent(run_root: Path, sha: str) -> dict[str, Any]:
    if _load_incumbent(run_root) is not None:
        raise ValueError(f"incumbent ledger already exists under {run_root}")
    evaluation = evaluate_commit(run_root, sha, role="baseline")
    if not evaluation.get("valid"):
        raise RuntimeError(
            f"baseline evaluation is invalid: {evaluation.get('result_path')}"
        )
    pointer = {
        "incumbent_sha": sha,
        "evaluation_result": evaluation["result_path"],
        "updated_at": _utc_now(),
        "protocol_hash": _protocol_hash(),
    }
    _atomic_json(run_root / "incumbent.json", pointer)
    _record_ledger(
        run_root,
        {"kind": "baseline", "commit_sha": sha, **pointer},
    )
    return {"baseline": True, **pointer, "evaluation": evaluation}


def evaluate_candidate(run_root: Path, candidate_sha: str) -> dict[str, Any]:
    incumbent_pointer = _load_incumbent(run_root)
    if incumbent_pointer is None:
        parent_sha = _resolve_commit(f"{candidate_sha}^")
        # Reject protected-file edits before spending the baseline budget.  On
        # the first call the candidate's parent is the prospective incumbent.
        validate_candidate_scope(parent_sha, candidate_sha)
        print(
            f"No incumbent ledger found; bootstrapping parent {parent_sha[:12]}",
            flush=True,
        )
        bootstrap_incumbent(run_root, parent_sha)
        incumbent_pointer = _load_incumbent(run_root)
        assert incumbent_pointer is not None

    incumbent_sha = _resolve_commit(str(incumbent_pointer["incumbent_sha"]))
    changed_paths = validate_candidate_scope(incumbent_sha, candidate_sha)
    incumbent_evaluation = _load_json(Path(incumbent_pointer["evaluation_result"]))
    candidate_evaluation = evaluate_commit(run_root, candidate_sha, role="candidate")
    selection = select_candidate(incumbent_evaluation, candidate_evaluation)

    meta_dir = run_root / "meta_results"
    meta_path = meta_dir / f"{_run_id('selection', candidate_sha)}.json"
    summary = {
        "schema_version": 1,
        "candidate_sha": candidate_sha,
        "incumbent_sha": incumbent_sha,
        "accepted": selection["accepted"],
        "selection_score": selection["selection_score"],
        "selection": selection,
        "changed_paths": changed_paths,
        "protocol_hash": _protocol_hash(),
        "candidate_result": candidate_evaluation["result_path"],
        "incumbent_result": incumbent_evaluation["result_path"],
        "tasks": candidate_evaluation.get("tasks", {}),
        "total_evaluator_calls": candidate_evaluation.get("total_evaluator_calls"),
        "total_token_usage": candidate_evaluation.get("total_token_usage"),
        "compute_usage": candidate_evaluation.get("compute_usage"),
        "created_at": _utc_now(),
        "result_path": str(meta_path),
    }
    _atomic_json(meta_path, summary)
    _record_ledger(run_root, summary)

    if summary["accepted"]:
        _atomic_json(
            run_root / "incumbent.json",
            {
                "incumbent_sha": candidate_sha,
                "evaluation_result": candidate_evaluation["result_path"],
                "updated_at": _utc_now(),
                "protocol_hash": _protocol_hash(),
            },
        )
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the protected recursive AutoEvolve meta-evaluator."
    )
    parser.add_argument(
        "--candidate-sha",
        default="HEAD",
        help="Committed candidate ref; must resolve to the current HEAD.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help="External result/ledger root (does not affect scoring).",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Operator-only: evaluate current HEAD as the initial incumbent.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate repository context and print the fixed protocol without runs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(argv if argv is not None else sys.argv[1:])
    if raw and raw[0] == "_private-whest":
        worker = argparse.ArgumentParser(add_help=False)
        worker.add_argument("_command")
        worker.add_argument("--candidate", type=Path, required=True)
        worker.add_argument("--output", type=Path, required=True)
        args = worker.parse_args(raw)
        return _private_whest_worker(args.candidate, args.output)

    args = _parse_args(raw)
    run_root = args.run_root.expanduser().resolve()
    candidate_sha = _resolve_commit(args.candidate_sha)
    _validate_operator_context(candidate_sha)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "candidate_sha": candidate_sha,
                    "run_root": str(run_root),
                    "protocol_hash": _protocol_hash(),
                    "protocol": _protocol_payload(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    with _global_lock(run_root):
        if args.bootstrap:
            result = bootstrap_incumbent(run_root, candidate_sha)
        else:
            result = evaluate_candidate(run_root, candidate_sha)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
