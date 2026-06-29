from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_eval_client_source(
    *,
    problem_type: str = "",
    socket_path: str | None = None,
    host: str = "127.0.0.1",
    port: int | None = None,
    submission_path: str = "submission.py",
    timeout_s: float = 3600.0,
    max_submission_bytes: int = 8 * 1024 * 1024,
) -> str:
    """Return a self-contained eval_client.py source string.

    The generated file imports only Python standard-library modules so it can be
    copied into an isolated workspace without exposing repository code.
    """

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
SUBMISSION_PATH = {submission_path!r}
TIMEOUT_S = {float(timeout_s)!r}
MAX_SUBMISSION_BYTES = {int(max_submission_bytes)!r}


def _read_submission(path: str) -> str:
    data = Path(path).read_bytes()
    if len(data) > MAX_SUBMISSION_BYTES:
        raise SystemExit(f"submission is too large: {{len(data)}} bytes")
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
        raise SystemExit("missing blackbox eval runner socket or TCP port")
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
        if b"\\n" in chunk:
            break
    data = b"".join(chunks)
    return data.split(b"\\n", 1)[0]


def main() -> int:
    submission_file = sys.argv[1] if len(sys.argv) > 1 else SUBMISSION_PATH
    request = {{
        "request_id": str(uuid.uuid4()),
        "problem_type": PROBLEM_TYPE,
        "submission": _read_submission(submission_file),
    }}
    data = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode() + b"\\n"
    with _connect() as sock:
        sock.sendall(data)
        response_data = _recv_line(sock)
    try:
        response = json.loads(response_data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        print(f"invalid blackbox eval runner response: {{exc}}", file=sys.stderr)
        return 2

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a thin blackbox eval client")
    parser.add_argument("--problem-type", default="")
    parser.add_argument("--socket", dest="socket_path", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--submission-path", default="submission.py")
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--max-submission-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    source = build_eval_client_source(
        problem_type=args.problem_type,
        socket_path=args.socket_path,
        host=args.host,
        port=args.port,
        submission_path=args.submission_path,
        timeout_s=args.timeout_s,
        max_submission_bytes=args.max_submission_bytes,
    )
    if args.output:
        Path(args.output).write_text(source, encoding="utf-8")
    else:
        sys.stdout.write(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
