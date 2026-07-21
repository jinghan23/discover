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


def _write_text_best_effort(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError:
        pass


def _open_log_target(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("wb")


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

    @classmethod
    def from_discover_config(
        cls,
        cfg: Any,
        *,
        semaphore: Any | None,
    ) -> "CodexResponseCompleter":
        return cls(
            model_name=cfg.model_name,
            max_output_tokens=cfg.max_output_tokens or 8192,
            temperature=cfg.temperature,
            api_key_env=cfg.api_key_env,
            base_url=cfg.base_url,
            semaphore=semaphore,
        )

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
    output_read_retries: int = 20
    output_read_retry_delay: float = 0.25
    _call_idx: int = field(default=0, init=False, repr=False)

    @classmethod
    def from_discover_config(
        cls,
        cfg: Any,
        *,
        cwd: str | None,
        semaphore: Any | None,
        step_idx: int,
        group_idx: int,
        sample_idx: int,
    ) -> "CodexCliCompleter":
        return cls(
            model_name=cfg.model_name,
            reasoning_effort=getattr(cfg, "cli_reasoning_effort", "xhigh"),
            codex_command=cfg.cli_command,
            sandbox=cfg.cli_sandbox,
            cwd=cwd,
            timeout=cfg.cli_timeout,
            semaphore=semaphore,
            log_dir=os.path.join(
                cfg.log_path,
                "codex_cli_calls",
                f"step_{step_idx:06d}",
            ),
            call_name=f"group_{group_idx:04d}_sample_{sample_idx:04d}",
            output_read_retries=1 if cfg.cli_command == "claude" else 20,
        )

    def _build_prompt(self, prompt: str) -> str:
        if not self.append_final_answer_instruction:
            return prompt
        return (
            prompt
            + "\n\nReturn only the final answer content. Do not edit files or run tools."
        )

    def _build_command(self, output_path: str) -> list[str]:
        if self.codex_command == "claude":
            cmd = [
                self.codex_command,
                "-p",
                "--no-session-persistence",
                "--output-format",
                "text",
            ]
            if self.model_name:
                cmd.extend(["--model", self.model_name])
            if self.sandbox == "read-only":
                cmd.extend(["--permission-mode", "dontAsk", "--tools", ""])
            elif self.sandbox == "workspace-write":
                cmd.extend([
                    "--permission-mode",
                    "acceptEdits",
                    "--allowedTools",
                    "Bash,Edit,Write,Read,Glob,Grep",
                ])
            elif self.sandbox == "danger-full-access":
                cmd.append("--dangerously-skip-permissions")
            return cmd

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
            _write_text_best_effort(
                call_dir / "codex_env.json",
                json.dumps(logged, indent=2, sort_keys=True),
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
        log_dir = Path(self.log_dir).expanduser().resolve()
        call_dir = (
            log_dir
            / f"{safe_name}_call_{self._call_idx:04d}_{uuid.uuid4().hex[:8]}"
        )
        call_dir.mkdir(parents=True, exist_ok=True)
        return call_dir

    async def _read_output_with_retry(
        self,
        output_path: str,
        stdout_log_path: Path | None,
        stdout: bytes | None,
    ) -> str:
        path = Path(output_path)
        attempts = max(1, self.output_read_retries)
        for i in range(attempts):
            try:
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    return text
            except FileNotFoundError:
                pass
            if i + 1 < attempts:
                await asyncio.sleep(self.output_read_retry_delay)

        stdout_text = _read_log_or_bytes(stdout_log_path, stdout).strip()
        if stdout_text:
            return stdout_text
        raise FileNotFoundError(output_path)

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
            _write_text_best_effort(call_dir / "prompt.txt", prompt)
            stdout_log_path = call_dir / "codex.stdout.log"
            stderr_log_path = call_dir / "codex.stderr.log"
            stdout_target = _open_log_target(stdout_log_path)
            stderr_target = _open_log_target(stderr_log_path)

        cmd = self._build_command(output_path)
        if call_dir is not None:
            _write_text_best_effort(call_dir / "command.json", json.dumps(cmd, indent=2))
            self._log_env(call_dir)

        try:
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=stdout_target,
                    stderr=stderr_target,
                    cwd=self.cwd,
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
                        _write_text_best_effort(
                            call_dir / "error.txt",
                            f"codex exec timed out after {self.timeout}s\n",
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
                        _write_text_best_effort(
                            call_dir / "error.txt",
                            "codex exec was cancelled\n",
                        )
                    raise

                stdout_text = _read_log_or_bytes(stdout_log_path, stdout)
                stderr_text = _read_log_or_bytes(stderr_log_path, stderr)
                if process.returncode != 0:
                    if call_dir is not None:
                        _write_text_best_effort(
                            call_dir / "error.txt",
                            f"codex exec failed with exit code {process.returncode}\n",
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

            return await self._read_output_with_retry(output_path, stdout_log_path, stdout)
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
