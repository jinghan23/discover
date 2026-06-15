from __future__ import annotations

import asyncio
import json
from typing import Any


DEFAULT_MAX_FRAME_BYTES = 8 * 1024 * 1024


class ProtocolError(ValueError):
    pass


def encode_frame(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )


def decode_frame(data: bytes, *, max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES) -> dict[str, Any]:
    if len(data) > max_frame_bytes:
        raise ProtocolError("request frame is too large")
    try:
        payload = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON frame: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("JSON frame must be an object")
    return payload


async def read_frame(
    reader: asyncio.StreamReader,
    *,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> dict[str, Any]:
    try:
        data = await reader.readline()
    except ValueError as exc:
        raise ProtocolError("request frame exceeds stream limit") from exc
    if not data:
        raise ProtocolError("empty request")
    return decode_frame(data, max_frame_bytes=max_frame_bytes)


async def write_frame(
    writer: asyncio.StreamWriter,
    payload: dict[str, Any],
) -> None:
    writer.write(encode_frame(payload))
    await writer.drain()
