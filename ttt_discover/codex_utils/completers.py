"""Text completers backed by Codex."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import tempfile
from dataclasses import dataclass
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
        cmd.append("-")
        return cmd

    async def __call__(self, prompt: str) -> str:
        if self.semaphore is not None:
            async with self.semaphore:
                return await self._call_unlocked(prompt)
        return await self._call_unlocked(prompt)

    async def _call_unlocked(self, prompt: str) -> str:
        prompt = self._build_prompt(prompt)
        output_file = tempfile.NamedTemporaryFile(prefix="ttt-discover-codex-", delete=False)
        output_path = output_file.name
        output_file.close()

        try:
            process = await asyncio.create_subprocess_exec(
                *self._build_command(output_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            communicate = process.communicate(prompt.encode("utf-8"))
            try:
                if self.timeout is not None:
                    stdout, stderr = await asyncio.wait_for(communicate, timeout=self.timeout)
                else:
                    stdout, stderr = await communicate
            except asyncio.TimeoutError as exc:
                await _kill_process_tree(process)
                raise RuntimeError(f"codex exec timed out after {self.timeout}s") from exc
            except asyncio.CancelledError:
                await _kill_process_tree(process)
                raise

            if process.returncode != 0:
                raise RuntimeError(
                    "codex exec failed with exit code "
                    f"{process.returncode}\nSTDOUT:\n{stdout.decode(errors='replace')}\n"
                    f"STDERR:\n{stderr.decode(errors='replace')}"
                )

            with open(output_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        finally:
            try:
                os.remove(output_path)
            except FileNotFoundError:
                pass
