from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import importlib
import json
import logging
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import uuid

from ttt_discover.eval_runners.blackbox_eval.protocol import (
    DEFAULT_MAX_FRAME_BYTES,
    ProtocolError,
    read_frame,
    write_frame,
)


logger = logging.getLogger(__name__)


def _load_object(spec: str) -> Any:
    module_name, sep, attr = spec.partition(":")
    if not sep:
        module_name, sep, attr = spec.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"import spec must be 'module:object': {spec!r}")
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _json_loads_object(value: str | None, *, label: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must decode to a JSON object")
    return parsed


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
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
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except Exception:
            pass
    return str(value)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _result_get(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _compact_text(text: Any, *, max_chars: int) -> str:
    value = "" if text is None else str(text)
    value = "".join(ch if ch == "\n" or ch == "\t" or ord(ch) >= 32 else " " for ch in value)
    value = value.strip()
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 14)].rstrip() + " ...(truncated)"


def _state_from_dict(data: dict[str, Any], state_type: type | None) -> Any:
    if state_type is not None and hasattr(state_type, "from_dict"):
        return state_type.from_dict(data)
    from ttt_discover.algorithms.state import state_from_dict

    return state_from_dict(data, state_type=state_type)


def _build_state_factory(
    *,
    env_type: type | None,
    problem_type: str,
    state_data: dict[str, Any] | None,
) -> Callable[[], Any]:
    state_type = getattr(env_type, "state_type", None) if env_type is not None else None
    if state_data is not None:
        return lambda: _state_from_dict(copy.deepcopy(state_data), state_type)
    if env_type is not None and hasattr(env_type, "create_initial_state"):
        return lambda: env_type.create_initial_state(problem_type)
    return lambda: None


@dataclass(frozen=True)
class VerifierConfig:
    evaluator_type: type
    problem_type: str
    log_dir: Path
    eval_timeout: int
    num_cpus_per_task: int
    evaluator_kwargs: dict[str, Any]
    state_factory: Callable[[], Any]
    allow_request_state: bool = False
    debug_responses: bool = False
    message_max_chars: int = 200
    max_evaluations: int | None = None


class BlackboxVerifier:
    def __init__(self, config: VerifierConfig):
        self.config = config
        self._max_evaluations = config.max_evaluations
        self._eval_count = 0
        self._count_lock = threading.Lock()

    def _try_reserve_call(self) -> tuple[bool, int]:
        """Atomically reserve one evaluator call against the budget.

        Returns (granted, used). The count is incremented *before* the
        evaluator runs and is never rolled back, so a call that later errors
        or times out still consumes budget. Format/protocol failures return
        earlier and never reach here, so they don't consume budget.
        """
        with self._count_lock:
            if (
                self._max_evaluations is not None
                and self._eval_count >= self._max_evaluations
            ):
                return False, self._eval_count
            self._eval_count += 1
            return True, self._eval_count

    def evaluate(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = str(request.get("request_id") or uuid.uuid4())
        response_base = {"request_id": request_id}

        problem_type = str(request.get("problem_type") or self.config.problem_type)
        if self.config.problem_type and problem_type != self.config.problem_type:
            return {
                **response_base,
                "ok": False,
                "stage": "request",
                "message": "unexpected problem_type",
            }

        submission = request.get("submission")
        if not isinstance(submission, str) or not submission.strip():
            return {
                **response_base,
                "ok": False,
                "stage": "request",
                "message": "empty submission",
            }

        state = self.config.state_factory()
        if self.config.allow_request_state and isinstance(request.get("state"), dict):
            state_type = type(state) if state is not None else None
            state = _state_from_dict(copy.deepcopy(request["state"]), state_type)

        granted, budget_used = self._try_reserve_call()
        if not granted:
            return {
                **response_base,
                "ok": False,
                "stage": "server",
                "message": "evaluation unavailable",
                "budget_used": budget_used,
                "budget_exhausted": True,
            }

        request_log_dir = self.config.log_dir / _safe_path_component(request_id)
        request_log_dir.mkdir(parents=True, exist_ok=True)

        evaluator_kwargs = {
            "problem_type": problem_type,
            "log_dir": str(request_log_dir),
            "eval_timeout": self.config.eval_timeout,
            "num_cpus_per_task": self.config.num_cpus_per_task,
            **self.config.evaluator_kwargs,
        }
        try:
            evaluator = self.config.evaluator_type(**evaluator_kwargs)
            result = evaluator.get_reward(submission, state=state)
        except Exception:
            logger.exception("blackbox evaluation failed; request_id=%s", request_id)
            message = "evaluation failed"
            if self.config.debug_responses:
                message = "evaluation failed; see server logs"
            return {
                **response_base,
                "ok": False,
                "stage": "exception",
                "message": message,
                "budget_used": budget_used,
            }

        response = self._response_from_result(response_base, result)
        response["budget_used"] = budget_used
        return response

    def _response_from_result(
        self,
        response_base: dict[str, Any],
        result: Any,
    ) -> dict[str, Any]:
        if isinstance(result, (int, float)):
            reward = float(result)
            raw_score = reward
            correctness = 1.0
            original_msg = ""
        else:
            reward = _safe_float(_result_get(result, "reward"), 0.0)
            raw_score = _safe_float(_result_get(result, "raw_score"), reward)
            correctness = _safe_float(_result_get(result, "correctness"), 1.0)
            original_msg = _result_get(result, "msg", "")
        result_construction = _result_get(result, "result_construction")
        details = _result_get(result, "details", {})

        ok = bool((correctness if correctness is not None else 0.0) > 0.0)
        if ok:
            score_part = raw_score if raw_score is not None else reward
            message = "pass"
            if score_part is not None:
                message = f"pass score={score_part:.6g}"
        else:
            message = "correctness failed"
            if self.config.debug_responses and original_msg:
                message = _compact_text(
                    original_msg,
                    max_chars=self.config.message_max_chars,
                )

        return {
            **response_base,
            "ok": ok,
            "stage": "complete" if ok else "correctness",
            "message": message,
            "reward": reward,
            "raw_score": raw_score,
            "correctness": correctness,
            "result_construction": _json_safe(result_construction),
            "details": _json_safe(details),
        }


def _safe_path_component(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return safe[:96] or uuid.uuid4().hex


@dataclass
class EvalJob:
    request: dict[str, Any]
    future: asyncio.Future


class BlackboxEvalServer:
    def __init__(
        self,
        verifier: BlackboxVerifier,
        *,
        workers: int,
        max_queue: int,
        max_frame_bytes: int,
    ):
        self.verifier = verifier
        self.workers = max(1, int(workers))
        self.queue: asyncio.Queue[EvalJob] = asyncio.Queue(maxsize=max_queue)
        self.max_frame_bytes = max_frame_bytes
        self._worker_tasks: list[asyncio.Task] = []

    async def start_workers(self) -> None:
        self._worker_tasks = [
            asyncio.create_task(self._worker_loop(i), name=f"blackbox-eval-worker-{i}")
            for i in range(self.workers)
        ]

    async def stop_workers(self) -> None:
        for task in self._worker_tasks:
            task.cancel()
        for task in self._worker_tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            request = await read_frame(reader, max_frame_bytes=self.max_frame_bytes)
            loop = asyncio.get_running_loop()
            future: asyncio.Future = loop.create_future()
            await self.queue.put(EvalJob(request=request, future=future))
            response = await future
        except ProtocolError as exc:
            response = {"ok": False, "stage": "request", "message": str(exc)}
        except Exception:
            logger.exception("failed while handling blackbox eval client")
            response = {"ok": False, "stage": "server", "message": "server error"}

        try:
            await write_frame(writer, _json_safe(response))
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _worker_loop(self, worker_idx: int) -> None:
        del worker_idx
        while True:
            job = await self.queue.get()
            try:
                response = await asyncio.to_thread(self.verifier.evaluate, job.request)
            except Exception:
                logger.exception("blackbox worker failed")
                response = {
                    "ok": False,
                    "stage": "server",
                    "message": "server error",
                }
            if not job.future.cancelled():
                job.future.set_result(response)
            self.queue.task_done()


async def _serve(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    env_type = _load_object(args.env_type) if args.env_type else None
    evaluator_type = (
        _load_object(args.evaluator)
        if args.evaluator
        else getattr(env_type, "reward_function", None)
    )
    if evaluator_type is None:
        raise ValueError("provide --evaluator or --env-type with reward_function")

    state_data = None
    if args.state_file:
        state_data = json.loads(Path(args.state_file).read_text(encoding="utf-8"))
    elif args.state_json:
        state_data = json.loads(args.state_json)
    if state_data is not None and not isinstance(state_data, dict):
        raise ValueError("state JSON must be an object")

    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    verifier = BlackboxVerifier(
        VerifierConfig(
            evaluator_type=evaluator_type,
            problem_type=args.problem_type,
            log_dir=log_dir,
            eval_timeout=args.eval_timeout,
            num_cpus_per_task=args.num_cpus_per_task,
            evaluator_kwargs=_json_loads_object(
                args.evaluator_kwargs,
                label="--evaluator-kwargs",
            ),
            state_factory=_build_state_factory(
                env_type=env_type,
                problem_type=args.problem_type,
                state_data=state_data,
            ),
            allow_request_state=args.allow_request_state,
            debug_responses=args.debug_responses,
            message_max_chars=args.message_max_chars,
            max_evaluations=args.max_evaluations,
        )
    )
    app = BlackboxEvalServer(
        verifier,
        workers=args.workers,
        max_queue=args.max_queue,
        max_frame_bytes=args.max_frame_bytes,
    )
    await app.start_workers()

    if args.socket:
        socket_path = Path(args.socket)
        if socket_path.exists():
            socket_path.unlink()
        server = await asyncio.start_unix_server(
            app.handle_client,
            path=str(socket_path),
            limit=args.max_frame_bytes + 1,
        )
        os.chmod(socket_path, 0o600)
        endpoint = str(socket_path)
    else:
        server = await asyncio.start_server(
            app.handle_client,
            host=args.host,
            port=args.port,
            limit=args.max_frame_bytes + 1,
        )
        sock = server.sockets[0] if server.sockets else None
        endpoint = str(sock.getsockname() if sock is not None else (args.host, args.port))

    logger.info("blackbox eval server listening on %s", endpoint)
    try:
        async with server:
            await server.serve_forever()
    finally:
        await app.stop_workers()
        if args.socket:
            with contextlib.suppress(FileNotFoundError):
                Path(args.socket).unlink()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a trusted verifier behind a blackbox IPC endpoint")
    parser.add_argument("--env-type", default=None, help="Import spec for an Environment class")
    parser.add_argument("--evaluator", default=None, help="Import spec for a reward evaluator class")
    parser.add_argument("--problem-type", default="")
    parser.add_argument("--log-dir", default="/tmp/ttt_blackbox_eval_logs")
    parser.add_argument("--eval-timeout", type=int, default=300)
    parser.add_argument("--num-cpus-per-task", type=int, default=1)
    parser.add_argument("--evaluator-kwargs", default=None, help="Extra evaluator kwargs as JSON object")
    parser.add_argument("--state-json", default=None)
    parser.add_argument("--state-file", default=None)
    parser.add_argument("--allow-request-state", action="store_true")
    parser.add_argument("--socket", default="/tmp/ttt_blackbox_eval.sock")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-queue", type=int, default=128)
    parser.add_argument("--max-frame-bytes", type=int, default=DEFAULT_MAX_FRAME_BYTES)
    parser.add_argument("--debug-responses", action="store_true")
    parser.add_argument("--message-max-chars", type=int, default=200)
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=None,
        help="Hard cap on total evaluator.get_reward() executions across all clients",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.port is not None and args.socket == parser.get_default("socket"):
        args.socket = None
    asyncio.run(_serve(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
