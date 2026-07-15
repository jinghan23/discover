from __future__ import annotations

import json
import os
from pathlib import Path
import socket
from typing import Any
import uuid

from ttt_discover.eval_runners.base import EvalRunner
from ttt_discover.tasks.base import VerifyResult


class BlackboxRunner(EvalRunner):
    NAME = "blackbox"

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        *,
        socket_path: str | None = None,
    ) -> "BlackboxRunner":
        return cls(
            socket_path=socket_path or cfg.blackbox_eval_socket,
            host=cfg.blackbox_eval_host,
            port=cfg.blackbox_eval_port,
            timeout_s=max(1.0, float(cfg.eval_timeout)),
        )

    def __init__(
        self,
        *,
        socket_path: str | None = None,
        host: str = "127.0.0.1",
        port: int | None = None,
        timeout_s: float = 3600.0,
    ):
        self.socket_path = socket_path
        self.host = host
        self.port = port
        self.timeout_s = timeout_s

    def _endpoint(self) -> tuple[str | None, str, int | None]:
        socket_path = os.environ.get("TTT_BLACKBOX_EVAL_SOCKET") or self.socket_path
        host = os.environ.get("TTT_BLACKBOX_EVAL_HOST") or self.host
        env_port = os.environ.get("TTT_BLACKBOX_EVAL_PORT")
        port = int(env_port) if env_port else self.port
        if socket_path is None and port is None:
            raise ValueError("blackbox eval runner requires a socket path or TCP port")
        return socket_path, host, port

    def _connect(self) -> socket.socket:
        socket_path, host, port = self._endpoint()
        if socket_path:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.timeout_s)
            sock.connect(socket_path)
            return sock
        sock = socket.create_connection((host, int(port)), timeout=self.timeout_s)
        sock.settimeout(self.timeout_s)
        return sock

    def evaluate(self, env: Any, generation: str) -> VerifyResult:
        request = {
            "request_id": str(uuid.uuid4()),
            "problem_type": env.problem_type,
            "submission": generation,
        }
        if hasattr(env.state, "to_dict"):
            request["state"] = env.state.to_dict()
        data = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
        with self._connect() as sock:
            sock.sendall(data)
            response_data = _recv_line(sock)
        response = json.loads(response_data.decode("utf-8"))
        ok = bool(response.get("ok"))
        message = str(response.get("message") or "")
        reward = response.get("reward")
        if reward is None:
            reward = 0.0
        correctness = response.get("correctness")
        if correctness is None:
            correctness = 1.0 if ok else 0.0
        raw_score = response.get("raw_score")
        if raw_score is None:
            raw_score = reward
        return VerifyResult.from_reward_dict(
            {
                "reward": float(reward),
                "msg": message,
                "correctness": float(correctness),
                "raw_score": float(raw_score),
                "result_construction": response.get("result_construction"),
                "stdout": message,
                "metrics": {
                    "blackbox/ok": ok,
                    "blackbox/stage": response.get("stage"),
                },
            }
        )

    def build_autonomous_prompt(
        self,
        env: Any,
        *,
        prompt: str,
        workspace: Path,
        eval_timeout: int,
        num_cpus_per_task: int,
    ) -> str | None:
        builder = getattr(env, "build_blackbox_autonomous_prompt", None)
        if builder is None:
            raise ValueError(
                f"{type(env).__name__} does not implement build_blackbox_autonomous_prompt"
            )
        socket_path, host, port = self._endpoint()
        return builder(
            prompt=prompt,
            workspace=workspace,
            eval_timeout=eval_timeout,
            num_cpus_per_task=num_cpus_per_task,
            socket_path=socket_path,
            host=host,
            port=port,
        )


def _recv_line(sock: socket.socket) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
        if b"\n" in chunk:
            break
    data = b"".join(chunks)
    if not data:
        raise ValueError("empty blackbox eval runner response")
    return data.split(b"\n", 1)[0]
