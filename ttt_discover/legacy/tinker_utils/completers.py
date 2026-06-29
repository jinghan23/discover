"""
Implementations that correspond to a model or policy that can be sampled from, but with different amounts of additional structure.

The TokenCompleter operates on tokens. This is the version used by RL algorithms, because RL algorithms work on Tokens. The MessageCompleter operates on messages, so it needs to be used with a renderer.

Evals and other code should use the appropriate interface.
"""

import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, TypeAlias

import tinker
from ttt_discover.tinker_utils.misc_utils import Tokenizer

# Interfaces

StopCondition: TypeAlias = list[str] | list[int]


@dataclass
class TokensWithLogprobs:
    tokens: list[int]
    maybe_logprobs: list[float] | None
    maybe_mask: list[float] | None = None  # Optional mask: 1.0 = train, 0.0 = don't train

    @property
    def logprobs(self) -> list[float]:
        if self.maybe_logprobs is None:
            raise ValueError("Logprobs are not available")
        return self.maybe_logprobs

    @property
    def mask(self) -> list[float]:
        """Return mask, defaulting to all 1.0 if not provided."""
        if self.maybe_mask is None:
            return [1.0] * len(self.tokens)
        return self.maybe_mask


class TokenCompleter:
    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
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

    def _decode_model_input(self, model_input: tinker.ModelInput) -> str:
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
        model_input: tinker.ModelInput,
        stop: StopCondition,
    ) -> TokensWithLogprobs:
        if self.semaphore is not None:
            async with self.semaphore:
                return await self._call_unlocked(model_input, stop)
        return await self._call_unlocked(model_input, stop)

    async def _call_unlocked(
        self,
        model_input: tinker.ModelInput,
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
    model_name: str | None = None
    codex_command: str = "codex"
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "read-only"
    cwd: str | None = None
    timeout: float | None = None
    semaphore: Any | None = None

    def _decode_model_input(self, model_input: tinker.ModelInput) -> str:
        return CodexTokenCompleter(tokenizer=self.tokenizer)._decode_model_input(model_input)

    def _build_prompt(self, prompt_text: str) -> str:
        request_input = CodexTokenCompleter(tokenizer=self.tokenizer)._build_input(prompt_text)
        if isinstance(request_input, str):
            return request_input

        lines: list[str] = []
        for message in request_input:
            role = message["role"].upper()
            lines.append(f"{role}:\n{message['content']}")
        lines.append(
            "ASSISTANT:\nReturn only the final answer content. Do not edit files or run tools."
        )
        return "\n\n".join(lines)

    def _append_stop_sequence(self, tokens: list[int], stop: StopCondition) -> list[int]:
        return CodexTokenCompleter(tokenizer=self.tokenizer)._append_stop_sequence(tokens, stop)

    async def __call__(
        self,
        model_input: tinker.ModelInput,
        stop: StopCondition,
    ) -> TokensWithLogprobs:
        if self.semaphore is not None:
            async with self.semaphore:
                return await self._call_unlocked(model_input, stop)
        return await self._call_unlocked(model_input, stop)

    async def _call_unlocked(
        self,
        model_input: tinker.ModelInput,
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
        if self.model_name:
            cmd.extend(["-m", self.model_name])
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


@dataclass
class TwoPhaseTokenCompleter(TokenCompleter):
    """
    Two-phase completer for gpt-oss: if Phase 1 exhausts tokens without stop, Phase 2 forces final answer.
    Uses full context window dynamically.
    """
    sampling_client: tinker.SamplingClient
    tokenizer: Tokenizer
    phase1_max_tokens: int  # Phase 1 limit (e.g., 27000)
    temperature: float = 1.0
    context_window: int = 32768
    context_buffer: int = 50

    PHASE2_PREFILL = "\n\n... okay, I am out of thinking tokens. I need to send my final message now."
    # Full marker to transition from analysis to final channel
    GPTOSS_FINAL_MARKER = "<|end|><|start|>assistant<|channel|>final<|message|>"
    # Marker that indicates we're already in the final channel
    GPTOSS_FINAL_CHANNEL_INDICATOR = "<|channel|>final<|message|>"

    def _hit_stop_sequence(self, tokens: list[int], stop: StopCondition) -> bool:
        """Check if the last token(s) match any stop sequence."""
        if not tokens:
            return False
        for s in stop:
            if isinstance(s, int):
                if tokens[-1] == s:
                    return True
            else:
                stop_tokens = self.tokenizer.encode(s, add_special_tokens=False)
                if len(stop_tokens) <= len(tokens) and tokens[-len(stop_tokens):] == stop_tokens:
                    return True
        return False

    def _contains_subsequence(self, tokens: list[int], pattern: str) -> bool:
        """Check if tokens contain the given pattern as a subsequence."""
        pattern_tokens = self.tokenizer.encode(pattern, add_special_tokens=False)
        if len(pattern_tokens) > len(tokens):
            return False
        for i in range(len(tokens) - len(pattern_tokens) + 1):
            if tokens[i:i + len(pattern_tokens)] == pattern_tokens:
                return True
        return False

    async def __call__(self, model_input: tinker.ModelInput, stop: StopCondition) -> TokensWithLogprobs:
        prompt_length = model_input.length
        
        # phase1_max_tokens is the total context budget for phase 1 (prompt + output)
        # This guarantees (context_window - phase1_max_tokens - buffer) tokens for phase 2
        # e.g., context_window = 32768, buffer = 50, prompt_length = 2000, phase1_max_tokens = 25000
        # then, in phase 1, we can generate at most 25000 - 2000 = 23000 tokens
        # in phase 2, we can generate at most 32768 - 2000 - 23000 - 50 = 7718 tokens
        # If prompt_length = 8000, then we can generate at most 25000 - 8000 = 17000 thinking tokens
        phase1_max = self.phase1_max_tokens - prompt_length
        if phase1_max <= 0:
            raise ValueError(f"Prompt length {prompt_length} exceeds phase1_max_tokens {self.phase1_max_tokens}.")
        
        phase1_result = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(stop=stop, max_tokens=phase1_max, temperature=self.temperature),
        )
        phase1_tokens = phase1_result.sequences[0].tokens
        phase1_logprobs = phase1_result.sequences[0].logprobs
        assert phase1_logprobs is not None

        # Check if we hit stop sequence
        if self._hit_stop_sequence(phase1_tokens, stop) or len(phase1_tokens) < phase1_max:
            return TokensWithLogprobs(tokens=phase1_tokens, maybe_logprobs=phase1_logprobs)

        # Phase 2: Didn't hit stop, force completion
        # Phase 2 budget = context_window - prompt - phase1 - buffer
        
        # Already in final channel? Just continue without prefill
        if self._contains_subsequence(phase1_tokens, self.GPTOSS_FINAL_CHANNEL_INDICATOR):
            new_chunks = list(model_input.chunks) + [tinker.types.EncodedTextChunk(tokens=phase1_tokens)]
            phase2_max = self.context_window - prompt_length - len(phase1_tokens) - self.context_buffer
            if phase2_max <= 0:
                return TokensWithLogprobs(tokens=phase1_tokens, maybe_logprobs=phase1_logprobs)
            phase2_result = await self.sampling_client.sample_async(
                prompt=tinker.ModelInput(chunks=new_chunks), num_samples=1,
                sampling_params=tinker.SamplingParams(stop=stop, max_tokens=phase2_max, temperature=self.temperature),
            )
            phase2_tokens = phase2_result.sequences[0].tokens
            phase2_logprobs = phase2_result.sequences[0].logprobs
            assert phase2_logprobs is not None
            return TokensWithLogprobs(tokens=phase1_tokens + phase2_tokens, maybe_logprobs=phase1_logprobs + phase2_logprobs)

        # Need prefill to transition to final channel
        end_token_seq = self.tokenizer.encode("<|end|>", add_special_tokens=False)
        ends_with_end = len(end_token_seq) <= len(phase1_tokens) and phase1_tokens[-len(end_token_seq):] == end_token_seq
        if ends_with_end:
            prefill_text = self.PHASE2_PREFILL + "<|start|>assistant<|channel|>final<|message|>"
        else:
            prefill_text = self.PHASE2_PREFILL + self.GPTOSS_FINAL_MARKER
        prefill_tokens = self.tokenizer.encode(prefill_text, add_special_tokens=False)

        new_chunks = list(model_input.chunks) + [
            tinker.types.EncodedTextChunk(tokens=phase1_tokens),
            tinker.types.EncodedTextChunk(tokens=prefill_tokens),
        ]
        phase2_max = self.context_window - prompt_length - len(phase1_tokens) - len(prefill_tokens) - self.context_buffer
        if phase2_max <= 0:
            return TokensWithLogprobs(
                tokens=phase1_tokens + prefill_tokens,
                maybe_logprobs=phase1_logprobs + [0.0] * len(prefill_tokens),
                maybe_mask=[1.0] * len(phase1_tokens) + [0.0] * len(prefill_tokens),
            )

        phase2_result = await self.sampling_client.sample_async(
            prompt=tinker.ModelInput(chunks=new_chunks), num_samples=1,
            sampling_params=tinker.SamplingParams(stop=stop, max_tokens=phase2_max, temperature=self.temperature),
        )
        phase2_tokens = phase2_result.sequences[0].tokens
        phase2_logprobs = phase2_result.sequences[0].logprobs
        assert phase2_logprobs is not None

        return TokensWithLogprobs(
            tokens=phase1_tokens + prefill_tokens + phase2_tokens,
            maybe_logprobs=phase1_logprobs + [0.0] * len(prefill_tokens) + phase2_logprobs,
            maybe_mask=[1.0] * len(phase1_tokens) + [0.0] * len(prefill_tokens) + [1.0] * len(phase2_tokens),
        )
