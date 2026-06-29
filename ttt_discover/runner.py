"""Bootstrap for Codex-backed discovery algorithms."""

from __future__ import annotations

import asyncio
import os
from argparse import Namespace
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from ttt_discover.config import DiscoverConfig
from ttt_discover.eval_runners import build_eval_runner
from ttt_discover.eval_runners.blackbox_eval.protocol import DEFAULT_MAX_FRAME_BYTES
from ttt_discover.eval_runners.blackbox_eval.server import _serve as serve_blackbox_eval
from ttt_discover.tasks import Task


@asynccontextmanager
async def _maybe_start_blackbox_server(cfg: DiscoverConfig):
    uses_blackbox = cfg.eval_runner == "blackbox" or (
        cfg.eval_runner == "auto" and cfg.algorithm == "autoevolve"
    )
    has_endpoint = bool(
        cfg.blackbox_eval_socket
        or cfg.blackbox_eval_port is not None
        or os.environ.get("TTT_BLACKBOX_EVAL_SOCKET")
        or os.environ.get("TTT_BLACKBOX_EVAL_PORT")
    )
    if not uses_blackbox or has_endpoint:
        yield None
        return

    if cfg.env_type is None:
        raise ValueError("env_type is required to auto-start blackbox eval runner server")

    socket_path = f"/tmp/ttt_blackbox_eval_{os.getpid()}_{id(cfg)}.sock"
    log_dir = os.path.join(cfg.log_path, "blackbox_eval")
    os.makedirs(log_dir, exist_ok=True)

    args = Namespace(
        env_type=f"{cfg.env_type.__module__}:{cfg.env_type.__qualname__}",
        evaluator=None,
        problem_type=cfg.problem_type,
        log_dir=log_dir,
        eval_timeout=cfg.eval_timeout,
        num_cpus_per_task=max(1, int(cfg.num_cpus_per_task)),
        evaluator_kwargs=None,
        state_json=None,
        state_file=None,
        allow_request_state=True,
        socket=socket_path,
        host=cfg.blackbox_eval_host,
        port=None,
        workers=1,
        max_queue=128,
        max_frame_bytes=DEFAULT_MAX_FRAME_BYTES,
        debug_responses=False,
        message_max_chars=200,
        verbose=False,
    )
    server_task = asyncio.create_task(serve_blackbox_eval(args), name="blackbox-eval-server")

    try:
        deadline = asyncio.get_running_loop().time() + 30.0
        path = Path(socket_path)
        while asyncio.get_running_loop().time() < deadline:
            if server_task.done():
                server_task.result()
                raise RuntimeError("blackbox eval runner server stopped before opening its socket")
            if path.exists():
                break
            await asyncio.sleep(0.05)
        else:
            raise TimeoutError(f"blackbox eval runner server did not open socket: {socket_path}")

        yield socket_path
    finally:
        server_task.cancel()
        with suppress(asyncio.CancelledError):
            await server_task
        with suppress(FileNotFoundError):
            Path(socket_path).unlink()


async def main(
    cfg: DiscoverConfig,
    *,
    log_path: str | None = None,
    wandb_name: str | None = None,
) -> None:
    cfg = cfg.to_runtime_config(log_path=log_path, wandb_name=wandb_name)
    task = Task.from_config(cfg)

    async with _maybe_start_blackbox_server(cfg) as managed_blackbox_socket:
        eval_runner = build_eval_runner(
            cfg,
            socket_path=managed_blackbox_socket,
        )

        from ttt_discover.algorithms import run_algorithm

        await run_algorithm(cfg, task=task, eval_runner=eval_runner)


async def _discover_async(config: DiscoverConfig) -> None:
    experiment_name = config.experiment_name or config.wandb_name or "codex-discovery"
    wandb_name = config.wandb_name or experiment_name
    log_path = config.log_path or f"./tinker_log/{experiment_name}"
    os.makedirs(log_path, exist_ok=True)

    await main(
        config,
        log_path=log_path,
        wandb_name=wandb_name,
    )


def discover(config: DiscoverConfig) -> None:
    asyncio.run(_discover_async(config))


__all__ = ["discover", "main"]
