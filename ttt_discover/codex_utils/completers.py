"""
Implementations that correspond to a model or policy that can be sampled from, but with different amounts of additional structure.

The TokenCompleter operates on tokens. This is the version used by RL algorithms, because RL algorithms work on Tokens. The MessageCompleter operates on messages, so it needs to be used with a renderer.

Evals and other code should use the appropriate interface.
"""

import os
import re
import tempfile
from dataclasses import dataclass
import json
from typing import Any, ClassVar, Literal

from ttt_discover.codex_utils.misc_utils import Tokenizer
from ttt_discover.codex_utils.runtime import ModelInput, StopCondition, TokensWithLogprobs

# Interfaces


class TokenCompleter:
    async def __call__(
        self, model_input: ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        raise NotImplementedError


@dataclass
class CodexTokenCompleter(TokenCompleter):
    """
    TokenCompleter backed by OpenAI's Codex models through the Responses API.

    This is intended for sampling-only discovery runs. The API does not return
    token logprobs in the shape expected by the RL trainer, so `maybe_logprobs`
    is intentionally left as None.
    """

    tokenizer: Tokenizer
    model_name: str | None = "gpt-5.2-codex"
    max_output_tokens: int = 8192
    reasoning_effort: str | None = "medium"
    temperature: float | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    client: Any | None = None
    semaphore: Any | None = None

    _GPT_OSS_MESSAGE_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"<\|start\|>(system|user|assistant)(?:<\|channel\|>[^<]+)?<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>)",
        re.DOTALL,
    )

    def _decode_model_input(self, model_input: ModelInput) -> str:
        tokens: list[int] = []
        for chunk in model_input.chunks:
            if not hasattr(chunk, "tokens"):
                raise ValueError("CodexTokenCompleter only supports text token chunks.")
            tokens.extend(chunk.tokens)
        return self.tokenizer.decode(tokens)

    def _parse_gpt_oss_messages(self, prompt_text: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for role, content in self._GPT_OSS_MESSAGE_RE.findall(prompt_text):
            if role == "system" and self._is_gpt_oss_system_prompt(content):
                continue
            messages.append({"role": role, "content": content})
        return messages

    @staticmethod
    def _is_gpt_oss_system_prompt(content: str) -> bool:
        return (
            "Reasoning:" in content
            and "# Valid channels:" in content
            and "Channel must be included for every message" in content
        )

    def _build_input(self, prompt_text: str) -> str | list[dict[str, str]]:
        messages = self._parse_gpt_oss_messages(prompt_text)
        if messages:
            return messages
        return prompt_text

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

    def _append_stop_sequence(self, tokens: list[int], stop: StopCondition) -> list[int]:
        if not stop:
            return tokens
        first_stop = stop[0]
        if isinstance(first_stop, int):
            return tokens + [first_stop]
        return tokens + self.tokenizer.encode(first_stop, add_special_tokens=False)

    async def __call__(
        self,
        model_input: ModelInput,
        stop: StopCondition,
    ) -> TokensWithLogprobs:
        if self.semaphore is not None:
            async with self.semaphore:
                return await self._call_unlocked(model_input, stop)
        return await self._call_unlocked(model_input, stop)

    async def _call_unlocked(
        self,
        model_input: ModelInput,
        stop: StopCondition,
    ) -> TokensWithLogprobs:
        if self.client is None:
            from openai import AsyncOpenAI

            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"{self.api_key_env} is not set; it is required for Codex no-finetune discovery."
                )
            self.client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)

        request: dict[str, Any] = {
            "model": self.model_name or "gpt-5.2-codex",
            "input": self._build_input(self._decode_model_input(model_input)),
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        if self.reasoning_effort is not None:
            request["reasoning"] = {"effort": self.reasoning_effort}
        if self.temperature is not None:
            request["temperature"] = self.temperature

        response = await self.client.responses.create(**request)
        text = self._extract_response_text(response)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return TokensWithLogprobs(
            tokens=self._append_stop_sequence(tokens, stop),
            maybe_logprobs=None,
        )


@dataclass
class CodexCliTokenCompleter(TokenCompleter):
    """
    TokenCompleter backed by the local `codex exec` CLI.

    This uses the logged-in Codex/ChatGPT account instead of `OPENAI_API_KEY`.
    It is slower than a direct API call because each completion starts a Codex
    CLI session, but it matches environments where only the Codex account is
    available.
    """

    tokenizer: Tokenizer
    model_name: str | None = "gpt-5.5"
    reasoning_effort: str | None = "xhigh"
    codex_command: str = "codex"
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "read-only"
    cwd: str | None = None
    timeout: float | None = None
    semaphore: Any | None = None
    ignore_user_config: bool = True
    ignore_rules: bool = True

    def _decode_model_input(self, model_input: ModelInput) -> str:
        return CodexTokenCompleter(tokenizer=self.tokenizer)._decode_model_input(model_input)

    def _build_prompt(self, prompt_text: str) -> str:
        request_input = CodexTokenCompleter(tokenizer=self.tokenizer)._build_input(prompt_text)
        if isinstance(request_input, str):
            return request_input

        if len(request_input) == 1 and request_input[0]["role"] == "user":
            return (
                request_input[0]["content"]
                + "\n\nReturn only the final answer content. Do not edit files or run tools."
            )

        lines: list[str] = []
        for message in request_input:
            role = message["role"].upper()
            lines.append(f"{role}:\n{message['content']}")
        lines.append("Return only the final answer content. Do not edit files or run tools.")
        return "\n\n".join(lines)

    def _append_stop_sequence(self, tokens: list[int], stop: StopCondition) -> list[int]:
        return CodexTokenCompleter(tokenizer=self.tokenizer)._append_stop_sequence(tokens, stop)

    async def __call__(
        self,
        model_input: ModelInput,
        stop: StopCondition,
    ) -> TokensWithLogprobs:
        if self.semaphore is not None:
            async with self.semaphore:
                return await self._call_unlocked(model_input, stop)
        return await self._call_unlocked(model_input, stop)

    async def _call_unlocked(
        self,
        model_input: ModelInput,
        stop: StopCondition,
    ) -> TokensWithLogprobs:
        prompt = self._build_prompt(self._decode_model_input(model_input))
        output_file = tempfile.NamedTemporaryFile(prefix="ttt-discover-codex-", delete=False)
        output_path = output_file.name
        output_file.close()

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

        try:
            import asyncio

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            communicate = process.communicate(prompt.encode("utf-8"))
            if self.timeout is not None:
                stdout, stderr = await asyncio.wait_for(communicate, timeout=self.timeout)
            else:
                stdout, stderr = await communicate

            if process.returncode != 0:
                raise RuntimeError(
                    "codex exec failed with exit code "
                    f"{process.returncode}\nSTDOUT:\n{stdout.decode(errors='replace')}\n"
                    f"STDERR:\n{stderr.decode(errors='replace')}"
                )

            with open(output_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        finally:
            try:
                os.remove(output_path)
            except FileNotFoundError:
                pass

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return TokensWithLogprobs(
            tokens=self._append_stop_sequence(tokens, stop),
            maybe_logprobs=None,
        )
