"""Text completers backed by Codex."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


async def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except Exception:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    await process.wait()


class TextCompleter:
    async def __call__(self, prompt: str) -> str:
        raise NotImplementedError


class CodexCliTimeoutError(RuntimeError):
    """Raised when `codex exec` times out after writing logs/workspace files."""

    def __init__(
        self,
        *,
        timeout: float | None,
        call_dir: Path | None,
        stdout: str,
        stderr: str,
    ):
        self.timeout = timeout
        self.call_dir = call_dir
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(
            f"codex exec timed out after {timeout}s; "
            f"log_dir={call_dir}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )


@dataclass
class CodexResponseCompleter(TextCompleter):
    """Text completer backed by OpenAI's Responses API."""

    model_name: str | None = "gpt-5.2-codex"
    max_output_tokens: int = 8192
    reasoning_effort: str | None = "medium"
    temperature: float | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    client: Any | None = None
    semaphore: Any | None = None

    async def __call__(self, prompt: str) -> str:
        if self.semaphore is not None:
            async with self.semaphore:
                return await self._call_unlocked(prompt)
        return await self._call_unlocked(prompt)

    async def _call_unlocked(self, prompt: str) -> str:
        if self.client is None:
            from openai import AsyncOpenAI

            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"{self.api_key_env} is not set; it is required for Codex discovery."
                )
            self.client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)

        request: dict[str, Any] = {
            "model": self.model_name or "gpt-5.2-codex",
            "input": prompt,
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        if self.reasoning_effort is not None:
            request["reasoning"] = {"effort": self.reasoning_effort}
        if self.temperature is not None:
            request["temperature"] = self.temperature

        response = await self.client.responses.create(**request)
        return self._extract_response_text(response)

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text

        text_parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text is not None:
                    text_parts.append(text)
                elif isinstance(content, dict) and content.get("text") is not None:
                    text_parts.append(str(content["text"]))
        if text_parts:
            return "\n".join(text_parts)
        raise ValueError("OpenAI response did not contain text output.")


@dataclass
class CodexCliCompleter(TextCompleter):
    """Text completer backed by the local `codex exec` CLI."""

    model_name: str | None = "gpt-5.5"
    reasoning_effort: str | None = "xhigh"
    codex_command: str = "codex"
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "read-only"
    cwd: str | None = None
    timeout: float | None = None
    semaphore: Any | None = None
    ignore_user_config: bool = True
    ignore_rules: bool = True
    append_final_answer_instruction: bool = True
    log_dir: str | None = None
    call_name: str | None = None
    config_overrides: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    skip_git_repo_check: bool = False
    _call_idx: int = field(default=0, init=False, repr=False)

    def _build_prompt(self, prompt: str) -> str:
        if not self.append_final_answer_instruction:
            return prompt
        return (
            prompt
            + "\n\nReturn only the final answer content. Do not edit files or run tools."
        )

    def _build_command(self, output_path: str) -> list[str]:
        cmd = [
            self.codex_command,
            "exec",
            "--ephemeral",
            "--sandbox",
            self.sandbox,
            "-o",
            output_path,
        ]
        if self.cwd:
            cmd.extend(["-C", self.cwd])
        if self.skip_git_repo_check:
            cmd.append("--skip-git-repo-check")
        if self.ignore_user_config:
            cmd.append("--ignore-user-config")
        if self.ignore_rules:
            cmd.append("--ignore-rules")
        cmd.extend(["-m", self.model_name or "gpt-5.5"])
        if self.reasoning_effort:
            cmd.extend([
                "-c",
                f"model_reasoning_effort={json.dumps(self.reasoning_effort)}",
            ])
        for override in self.config_overrides:
            cmd.extend(["-c", override])
        cmd.append("-")
        return cmd

    def _build_process_env(self) -> dict[str, str] | None:
        if self.env is None:
            return None
        process_env = os.environ.copy()
        process_env.update(self.env)
        return process_env

    def _log_env(self, call_dir: Path) -> None:
        keys = (
            "CUDA_VISIBLE_DEVICES",
            "CUDA_DEVICE_ORDER",
            "TORCH_CUDA_ARCH_LIST",
            "PYTHONUNBUFFERED",
            "PYTHONPATH",
            "CUDA_HOME",
            "LD_LIBRARY_PATH",
            "TRITON_CACHE_DIR",
            "TORCH_EXTENSIONS_DIR",
        )
        effective = os.environ.copy()
        if self.env is not None:
            effective.update(self.env)
        logged = {key: effective[key] for key in keys if key in effective}
        if logged:
            (call_dir / "codex_env.json").write_text(
                json.dumps(logged, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    def _next_call_dir(self) -> Path | None:
        if not self.log_dir:
            return None
        self._call_idx += 1
        name = self.call_name or "call"
        safe_name = "".join(
            char if char.isalnum() or char in "._-" else "_"
            for char in name
        )
        call_dir = (
            Path(self.log_dir)
            / f"{safe_name}_call_{self._call_idx:04d}_{uuid.uuid4().hex[:8]}"
        )
        call_dir.mkdir(parents=True, exist_ok=True)
        return call_dir

    async def __call__(self, prompt: str) -> str:
        if self.semaphore is not None:
            async with self.semaphore:
                return await self._call_unlocked(prompt)
        return await self._call_unlocked(prompt)

    async def _call_unlocked(self, prompt: str) -> str:
        prompt = self._build_prompt(prompt)
        call_dir = self._next_call_dir()
        if call_dir is None:
            output_file = tempfile.NamedTemporaryFile(prefix="ttt-discover-codex-", delete=False)
            output_path = output_file.name
            output_file.close()
            stdout_target = asyncio.subprocess.PIPE
            stderr_target = asyncio.subprocess.PIPE
            stdout_log_path = None
            stderr_log_path = None
        else:
            output_path = str(call_dir / "final_response.txt")
            (call_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
            stdout_log_path = call_dir / "codex.stdout.log"
            stderr_log_path = call_dir / "codex.stderr.log"
            stdout_target = stdout_log_path.open("wb")
            stderr_target = stderr_log_path.open("wb")

        cmd = self._build_command(output_path)
        if call_dir is not None:
            (call_dir / "command.json").write_text(
                json.dumps(cmd, indent=2),
                encoding="utf-8",
            )
            self._log_env(call_dir)

        try:
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=stdout_target,
                    stderr=stderr_target,
                    env=self._build_process_env(),
                    start_new_session=True,
                )
                communicate = process.communicate(prompt.encode("utf-8"))
                stdout: bytes | None = None
                stderr: bytes | None = None
                try:
                    if self.timeout is not None:
                        stdout, stderr = await asyncio.wait_for(communicate, timeout=self.timeout)
                    else:
                        stdout, stderr = await communicate
                except asyncio.TimeoutError as exc:
                    await _kill_process_tree(process)
                    stdout_text = _read_log_or_bytes(stdout_log_path, stdout)
                    stderr_text = _read_log_or_bytes(stderr_log_path, stderr)
                    if call_dir is not None:
                        (call_dir / "error.txt").write_text(
                            f"codex exec timed out after {self.timeout}s\n",
                            encoding="utf-8",
                        )
                    raise CodexCliTimeoutError(
                        timeout=self.timeout,
                        call_dir=call_dir,
                        stdout=stdout_text,
                        stderr=stderr_text,
                    ) from exc
                except asyncio.CancelledError:
                    await _kill_process_tree(process)
                    if call_dir is not None:
                        (call_dir / "error.txt").write_text(
                            "codex exec was cancelled\n",
                            encoding="utf-8",
                        )
                    raise

                stdout_text = _read_log_or_bytes(stdout_log_path, stdout)
                stderr_text = _read_log_or_bytes(stderr_log_path, stderr)
                if process.returncode != 0:
                    if call_dir is not None:
                        (call_dir / "error.txt").write_text(
                            f"codex exec failed with exit code {process.returncode}\n",
                            encoding="utf-8",
                        )
                    raise RuntimeError(
                        "codex exec failed with exit code "
                        f"{process.returncode}; log_dir={call_dir}\n"
                        f"STDOUT:\n{stdout_text}\nSTDERR:\n{stderr_text}"
                    )
            finally:
                if stdout_log_path is not None:
                    stdout_target.close()
                if stderr_log_path is not None:
                    stderr_target.close()

            with open(output_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        finally:
            if call_dir is None:
                try:
                    os.remove(output_path)
                except FileNotFoundError:
                    pass


def _read_log_or_bytes(path: Path | None, data: bytes | None) -> str:
    if path is not None:
        try:
            return path.read_bytes().decode(errors="replace")
        except FileNotFoundError:
            return ""
    return (data or b"").decode(errors="replace")
