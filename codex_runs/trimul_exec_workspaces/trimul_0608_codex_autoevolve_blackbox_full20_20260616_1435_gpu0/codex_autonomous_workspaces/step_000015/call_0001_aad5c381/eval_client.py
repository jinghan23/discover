#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
import uuid


PROBLEM_TYPE = 'trimul'
SOCKET_PATH = '/tmp/ttt_blackbox_eval_trimul_full20_20260616_1435_gpu0.sock'
HOST = '127.0.0.1'
PORT = None
SUBMISSION_PATH = 'submission.py'
TIMEOUT_S = 1200.0
MAX_SUBMISSION_BYTES = 8388608


def _read_submission(path: str) -> str:
    data = Path(path).read_bytes()
    if len(data) > MAX_SUBMISSION_BYTES:
        raise SystemExit(f"submission is too large: {len(data)} bytes")
    return data.decode("utf-8")


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
        raise SystemExit("missing blackbox evaluator socket or TCP port")
    sock = socket.create_connection((host_value, int(port)), timeout=TIMEOUT_S)
    sock.settimeout(TIMEOUT_S)
    return sock


def _recv_line(sock: socket.socket) -> bytes:
    chunks = []
    total = 0
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > MAX_SUBMISSION_BYTES:
            raise SystemExit("response is too large")
        if b"\n" in chunk:
            break
    data = b"".join(chunks)
    return data.split(b"\n", 1)[0]


def main() -> int:
    submission_file = sys.argv[1] if len(sys.argv) > 1 else SUBMISSION_PATH
    request = {
        "request_id": str(uuid.uuid4()),
        "problem_type": PROBLEM_TYPE,
        "submission": _read_submission(submission_file),
    }
    data = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
    with _connect() as sock:
        sock.sendall(data)
        response_data = _recv_line(sock)
    try:
        response = json.loads(response_data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        print(f"invalid evaluator response: {exc}", file=sys.stderr)
        return 2

    message = str(response.get("message") or "")
    if response.get("ok"):
        print(message or "pass")
        if response.get("raw_score") is not None:
            print(f"raw_score {response['raw_score']}")
        if response.get("reward") is not None:
            print(f"reward {response['reward']}")
        return 0

    stage = response.get("stage") or "eval"
    print(f"FAIL stage={stage} message={message or 'evaluation failed'}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
