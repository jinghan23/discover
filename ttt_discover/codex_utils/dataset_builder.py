from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Sequence
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
import asyncio
import time
import logging
import re
import os
import json
from pathlib import Path

import chz
import ttt_discover.codex_utils.logtree as logtree
import ttt_discover.codex_utils.renderers as renderers
from ttt_discover.codex_utils.runtime import (
    Action,
    Env,
    EnvGroupBuilder,
    ModelInput,
    RLDataset,
    RLDatasetBuilder,
    State,
    StepResult,
    state_from_dict,
)
from ttt_discover.codex_utils.misc_utils import get_tokenizer
from ttt_discover.codex_utils.sampler import StateSampler, get_or_create_sampler_with_default

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """General configuration for dataset and environment creation.

    Provide env_type (for custom envs, pass the class).
    After get_single_problem_dataset_builder(), env_type is set on config for internal use.
    """
    problem_type: str
    env_type: type
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    num_cpus_per_task: int = 1
    eval_timeout: int = 300
    log_path: str = ""
    timeout: float = 8000.0 # Timeout for async grading, not sandbox timeout
    convo_prefix: Any = None
    gpu_mode_score_scale: float = 3000.0
    initial_program_paths: tuple[str, ...] = ()
    initial_pool_paths: tuple[str, ...] = ()
    resume_step: int | None = None
    topk_children: int = 16


class ProblemEnv(Env):
    def __init__(
        self,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        format_coef: float = 0.1,
    ):
        self.renderer = renderer
        self.convo_prefix = convo_prefix or []
        self.format_coef = format_coef

    @property
    def stop_condition(self):
        return self.renderer.get_stop_sequences()

    @abstractmethod
    def get_question(self) -> str:
        pass

    @abstractmethod
    def check_answer(self, sample_str: str) -> bool:
        pass

    @abstractmethod
    def check_format(self, sample_str: str) -> bool:
        pass

    @abstractmethod
    def get_reference_answer(self) -> str:
        pass

    async def initial_observation(self):
        convo = self.convo_prefix + [
            {"role": "user", "content": self.get_question()},
        ]
        return self.renderer.build_generation_prompt(convo), self.stop_condition

    async def step(self, action: Action, *args: Any, **kwargs: Any) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.ensure_text(message["content"])
        correct_format = float(parse_success) and float(self.check_format(content))
        correct_answer = float(self.check_answer(content))
        total_reward = self.format_coef * (correct_format - 1) + correct_answer

        logtree.log_text(f"Problem: {self.get_question()}")
        logtree.log_text(f"Response: {message['content']}")
        logtree.log_text(f"Reference Answer: {self.get_reference_answer()}")
        logtree.log_text(
            f"Format Valid: {'✓' if correct_format else '✗'}, "
            f"Correct: {'✓' if correct_answer else '✗'}, Reward: {total_reward:.2f}"
        )

        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": correct_format,
                "correct": correct_answer,
            },
        )


@dataclass(frozen=True)
class ProblemGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], ProblemEnv]
    num_envs: int
    logging_name: str = "environment"

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    def logging_tags(self) -> list[str]:
        return [self.logging_name]


class SingleProblemDataset(RLDataset):
    def __init__(
        self,
        config: DatasetConfig,
        renderer: renderers.Renderer,
        sampler: StateSampler,
    ):
        self.config = config
        self.batch_size = config.batch_size
        self.group_size = config.group_size
        self.renderer = renderer
        self.problem_type = config.problem_type
        self.sampler = sampler

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        states = self.sampler.sample_states(self.batch_size)
        return [self._make_env_group_builder(state, self.group_size) for state in states]

    def flush(self, step: int | None = None):
        """Flush sampler state to disk. Call after batch completes."""
        self.sampler.flush(step)

    def __len__(self) -> int:
        return 1

    def _make_env_group_builder(
        self, initial_state: State, group_size: int
    ) -> ProblemGroupBuilder:
        """Create an environment group builder using the env type from config."""
        env_type = self.config.env_type
        if env_type is None:
            raise ValueError("config.env_type must be set")
        logging_name = getattr(env_type, "env_name", env_type.__name__)
        return ProblemGroupBuilder(
            env_thunk=partial(
                env_type,
                self.renderer,
                initial_state=initial_state,
                sampler=self.sampler,
                config=self.config,
            ),
            num_envs=group_size,
            logging_name=logging_name,
        )


@chz.chz
class SingleProblemDatasetBuilder(RLDatasetBuilder):
    config: DatasetConfig

    def _latest_sampler_step(self) -> int:
        pattern = Path(self.config.log_path).glob("puct_sampler_step_*.json")
        latest = 0
        for path in pattern:
            match = re.search(r"puct_sampler_step_(\d+)\.json$", path.name)
            if match:
                latest = max(latest, int(match.group(1)))
        return latest

    async def __call__(self) -> SingleProblemDataset:
        if self.config.problem_type is None:
            raise ValueError("problem_type is required")
        if not self.config.log_path:
            raise ValueError("log_path is required for dataset")
        
        tokenizer = get_tokenizer(self.config.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.config.renderer_name, tokenizer=tokenizer)
        
        sampler = self._get_sampler()
        self._seed_initial_pools(sampler)
        self._seed_initial_programs(sampler, renderer)
        
        dataset = SingleProblemDataset(
            config=self.config,
            renderer=renderer,
            sampler=sampler,
        )
        return dataset

    def _get_sampler(self) -> StateSampler:
        """Get the appropriate sampler; env_type is already set on config."""
        resume_step = self.config.resume_step
        if resume_step is None:
            resume_step = self._latest_sampler_step()
        return get_or_create_sampler_with_default(
            log_path=self.config.log_path,
            env_type=self.config.env_type,
            batch_size=self.config.batch_size,
            problem_type=self.config.problem_type,
            resume_step=resume_step if resume_step > 0 else None,
            topk_children=self.config.topk_children,
        )

    def _load_initial_programs(self) -> list[tuple[str, str]]:
        programs: list[tuple[str, str]] = []
        prepare = getattr(self.config.env_type, "prepare_initial_program", None)
        for raw_path in self.config.initial_program_paths:
            path = Path(os.path.expanduser(raw_path))
            if not path.is_absolute():
                path = Path.cwd() / path
            program = path.read_text(encoding="utf-8")
            if prepare is not None:
                program = prepare(
                    program,
                    source_path=str(path),
                    eval_timeout=self.config.eval_timeout,
                )
            if "```python" not in program:
                program = f"```python\n{program.rstrip()}\n```"
            programs.append((str(path), program))
        return programs

    def _resolve_path(self, raw_path: str | os.PathLike) -> Path:
        path = Path(os.path.expanduser(str(raw_path)))
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    def _pool_payload_to_states(
        self,
        payload: Any,
        *,
        path: Path,
        state_type: type,
    ) -> list[State]:
        if isinstance(payload, list):
            raw_states = payload
        elif isinstance(payload, dict):
            if "state" in payload and isinstance(payload["state"], dict):
                raw_states = [payload["state"]]
            elif "states" in payload:
                raw_states = payload["states"]
            elif "initial_states" in payload:
                raw_states = payload["initial_states"]
            elif {"timestep", "construction", "code"}.issubset(payload):
                raw_states = [payload]
            else:
                raise ValueError(f"Unsupported initial pool state file: {path}")
        else:
            raise ValueError(f"Unsupported initial pool format: {path}")

        if not isinstance(raw_states, list):
            raise ValueError(f"Initial pool states must be a list: {path}")
        return [
            state_from_dict(item, state_type=state_type)
            for item in raw_states
            if isinstance(item, dict)
        ]

    def _load_initial_pool_file(self, path: Path, *, state_type: type) -> list[State]:
        store = json.loads(path.read_text(encoding="utf-8"))
        return self._pool_payload_to_states(store, path=path, state_type=state_type)

    def _load_initial_pool_states(self) -> list[State]:
        states: list[State] = []
        state_type = getattr(self.config.env_type, "state_type", State)
        for raw_path in self.config.initial_pool_paths:
            path = self._resolve_path(raw_path)
            loaded = self._load_initial_pool_file(path, state_type=state_type)
            states.extend(loaded)
            logger.info("Loaded %s initial-pool states from %s", len(loaded), path)
        return [state for state in states if state is not None]

    def _seed_initial_pools(self, sampler: StateSampler) -> None:
        if not self.config.initial_pool_paths:
            return

        pool_states = self._load_initial_pool_states()
        if not pool_states:
            return

        add_initial_states = getattr(sampler, "add_initial_states", None)
        if add_initial_states is not None:
            added = add_initial_states(pool_states, save=False)
        else:
            parent_states = sampler.sample_states(1)
            if not parent_states:
                logger.warning("No parent state available for initial pool seeding")
                return
            parents = [parent_states[0]] * len(pool_states)
            before = len(getattr(sampler, "_states", []))
            sampler.update_states(pool_states, parents, save=False)
            added = len(getattr(sampler, "_states", [])) - before

        if added:
            logger.info("Seeded %s states from initial pool", added)
            sampler.flush()

    def _seed_initial_programs(self, sampler: StateSampler, renderer: renderers.Renderer) -> None:
        if not self.config.initial_program_paths:
            return

        if hasattr(sampler, "get_initial_states"):
            parent_states = sampler.get_initial_states()
        else:
            parent_states = sampler.sample_states(1)
        if not parent_states:
            logger.warning("No parent state available for initial program seeding")
            return

        programs = self._load_initial_programs()
        seed_states: list[State] = []
        seed_parents: list[State] = []

        for parent in parent_states:
            env = self.config.env_type(
                renderer,
                initial_state=parent,
                sampler=sampler,
                config=self.config,
            )
            for source_path, program in programs:
                direct_seed = None
                build_seed = getattr(
                    self.config.env_type,
                    "create_seed_state_from_initial_program",
                    None,
                )
                if build_seed is not None:
                    direct_seed = build_seed(
                        program,
                        source_path=source_path,
                        eval_timeout=self.config.eval_timeout,
                    )
                if direct_seed is not None:
                    has_state = getattr(sampler, "has_state", None)
                    if has_state is not None and has_state(direct_seed):
                        logger.info(
                            "Skipping already-seeded initial program %s",
                            source_path,
                        )
                        continue
                    seed_states.append(direct_seed)
                    seed_parents.append(parent)
                    logger.info(
                        "Seeded initial program %s directly with value=%s",
                        source_path,
                        direct_seed.value,
                    )
                    continue

                outs = env._run_verification(
                    program,
                    self.config.problem_type,
                    self.config.log_path,
                    parent,
                )
                if outs.correctness <= 0:
                    logger.warning("Initial program failed verification: %s: %s", source_path, outs.msg)
                    continue
                seed_state = env._create_next_state(-1, program, outs)
                seed_states.append(seed_state)
                seed_parents.append(parent)
                logger.info(
                    "Seeded initial program %s with raw_score=%s reward=%s",
                    source_path,
                    outs.raw_score,
                    outs.reward,
                )

        if seed_states:
            sampler.update_states(seed_states, seed_parents, save=False)
            sampler.flush()


def get_single_problem_dataset_builder(
    config: DatasetConfig,
    **kwargs,
) -> RLDatasetBuilder:
    """
    Unified function to get a single problem dataset builder.
    Custom envs: pass env_type=YourEnv in DatasetConfig; no registry needed.
    """
    if not config.log_path:
        raise ValueError("log_path is required for dataset")

    return SingleProblemDatasetBuilder(config=config)


def last_codeblock_postprocess(input_text, codeblock_seps=['python', 'cpp', 'java', 'cuda'], last_response_strict=True, keep_separators=True):
    """Extract the last code block from input text.
    
    Args:
        input_text: Text to parse
        codeblock_seps: List of language identifiers to look for
        last_response_strict: If True, return empty string for invalid code; otherwise return original text
        keep_separators: If True, return code with ```language wrapper; if False, return code only
    """
    languages_pattern = '|'.join(map(re.escape, codeblock_seps))
    codeblock_start = f'```({languages_pattern})'
    pattern = re.compile(codeblock_start + r'\n(?!```)(.*?)(?:\n```)?(?=\n```|$)', re.DOTALL)
    matches = list(pattern.finditer(input_text))

    if matches:
        last_match = matches[-1]
        language = last_match.group(1)
        code_content = last_match.group(2).rstrip()
        
        # Check if content is empty
        if not code_content or code_content.strip() == '':
            if last_response_strict:
                return ''
            else:
                return input_text
        
        if keep_separators:
            return f'```{language}\n{code_content}\n```'
        else:
            return code_content
    else:
        if last_response_strict:
            return ''
        else:
            return input_text


@dataclass
class VerifyResult:
    reward: float
    msg: str
    correctness: float
    raw_score: float
    result_construction: Any
    stdout: str
    metrics: dict[str, Any] = field(default_factory=dict)

# Shared ThreadPoolExecutor for all environments
SAFE_GRADE_MAX_WORKERS = 4096
SAFE_GRADE_EXECUTOR = ThreadPoolExecutor(max_workers=SAFE_GRADE_MAX_WORKERS)


class Environment(ProblemEnv):

    state_type: State

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        """Create an initial state for rollouts. Override in subclasses that need a different initial state."""
        return State(timestep=-1, construction=None, code="", value=0.0)

    def __init__(
        self,
        renderer: renderers.Renderer,
        initial_state: State,
        sampler,
        config,
    ):
        super().__init__(renderer, convo_prefix=config.convo_prefix)
        
        if initial_state is None:
            raise ValueError("initial_state is required and cannot be None")
        if sampler is None:
            raise ValueError("sampler is required and cannot be None")
        
        self.config = config
        self.timeout = config.timeout
        self.num_cpus_per_task = config.num_cpus_per_task
        self.eval_timeout = config.eval_timeout
        self.log_path = config.log_path
        self.initial_state = initial_state
        self.sampler = sampler
        self.state = initial_state
        self.problem_type = config.problem_type
    
    @abstractmethod
    def get_question(self) -> str:
        """Build prompt from template, injecting previous code from state.
        
        Returns:
            Formatted prompt string
        """
        pass

    def is_maximize(self) -> bool:
        return True
    
    def _create_next_state(
        self,
        step_idx: int,
        parsed_code: str,
        outs: VerifyResult,
    ) -> State:
        """Create the next state from the current step.
        
        Args:
            step_idx: Current step index
            parsed_code: Parsed code from response
            outs: Output dictionary from _verify_code
            
        Returns:
            New State object
        """
        return self.state_type(
            timestep=step_idx,
            construction=outs.result_construction,
            code=parsed_code,
            value=outs.raw_score if self.is_maximize() else -outs.raw_score, # higher = better
            observation=outs.stdout,
        )
    
    def _build_metrics(
        self,
        outs: VerifyResult,
        correct_format: bool,
        message: dict,
        parsed_code: str,
    ) -> dict[str, Any]:
        """Build metrics dictionary for StepResult.
        
        Args:
            outs: VerifyResult from _run_verification
            correct_format: Whether the code format was valid
            message: Parsed message from renderer
            parsed_code: Parsed code string
            
        Returns:
            Metrics dictionary
        """
        correctness = outs.correctness
        return {
            "format": correct_format,
            "reward": outs.reward,
            "correctness": correctness,
            "raw_score": outs.raw_score if correctness > 0 else None,
            "initial_raw_score": self.initial_state.value,
            "msg": outs.msg,
            "prompt": self.get_question(),
            "response": message['content'],
            "parsed_code": parsed_code,
        }
    
    def _get_code_languages(self) -> list[str]:
        """Return list of code block languages to parse. Override if needed."""
        return ["python"]
    
    def _should_keep_code_separators(self) -> bool:
        """Whether to keep ```language separators in parsed code. Override if needed."""
        return True
    
    def check_format(self, parsed_code: str) -> bool:
        """Check if parsed code has valid format."""
        if (parsed_code is None) or (parsed_code.strip() == ''):
            return False
        return True
    
    async def check_answer(self, parsed_code: str, step: int) -> VerifyResult:
        """Check answer asynchronously with timeout."""
        if not self.check_format(parsed_code):
            return VerifyResult(
                reward=0.0,
                msg="Invalid code",
                correctness=0.0,
                raw_score=0.0,
                result_construction=None,
                stdout="",
            )
        
        return await self._safe_grade(parsed_code, step)

    def _run_verification(
        self,
        generation: str,
        problem_type: str,
        log_path: str,
        state: State,
    ) -> VerifyResult:

        task = self.reward_function(problem_type=problem_type, log_dir=log_path, eval_timeout=self.eval_timeout, num_cpus_per_task=self.num_cpus_per_task)
        out = task.get_reward(generation, state=state)

        return VerifyResult(
            reward=out["reward"],
            msg=out["msg"],
            correctness=out["correctness"],
            raw_score=out["raw_score"],
            result_construction=out.get("result_construction", None),
            stdout=out.get("stdout", ""),
            metrics=out.get("metrics", {}),
        )
    
    async def _safe_grade(self, given_answer: str, step: int) -> VerifyResult:
        """Async grader: runs _verify_code in a background thread with asyncio timeout."""
        loop = asyncio.get_running_loop()
        start_time = time.time()
        
        try:
            out = await asyncio.wait_for(
                loop.run_in_executor(
                    SAFE_GRADE_EXECUTOR,
                    partial[VerifyResult](
                        self._run_verification,
                        given_answer,
                        self.problem_type,
                        self.log_path,
                        self.state,
                    )
                ),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.warning(f"Timeout grading: took {elapsed:.1f}s, limit was {self.timeout:.1f}s")
            return VerifyResult(
                reward=0.0, 
                msg="Timeout grading", 
                correctness=0.0, 
                raw_score=0.0, 
                result_construction=None, 
                stdout=""
            )
        except Exception as e:
            import traceback
            error_msg = f"Error grading: {e}\n{traceback.format_exc()}"
            logger.warning(f"Exception while grading: {e}")
            return VerifyResult(
                reward=0.0,
                msg=f"Error grading: {error_msg}",
                correctness=0.0,
                raw_score=0.0,
                result_construction=None,
                stdout="",
            )
        
        return out

    async def step(self, action: Action, step_idx: int) -> StepResult:
        """Process a step: parse response, verify code, compute reward, update state."""
        message, parse_success = self.renderer.parse_response(action)
        response = message["content"]
        
        # Parse code based on environment-specific settings
        languages = self._get_code_languages()
        keep_separators = self._should_keep_code_separators()
        parsed_code = last_codeblock_postprocess(
            response,
            codeblock_seps=languages,
            keep_separators=keep_separators
        )

        correct_format = float(parse_success) and float(self.check_format(parsed_code))
        
        # Verify code
        outs = await self.check_answer(parsed_code, step_idx)
        reward = outs.reward
        correctness = outs.correctness
        raw_score = outs.raw_score
        msg = outs.msg
        
        # Logging
        logtree.log_text(f"Problem: {self.get_question()[:200]}...")
        logtree.log_text(f"Response: {message['content']}")
        logtree.log_text(
            f"Format Valid: {'✓' if correct_format else '✗'}, "
            f"Reward: {reward:.4f}, Correctness: {correctness:.4f}, Raw Score: {raw_score:.4f}, Msg: {msg}"
        )
        
        # Build metrics
        metrics = self._build_metrics(outs, correct_format, message, parsed_code)
        
        # Create step result
        step_result = StepResult(
            reward=reward,
            episode_done=True,
            next_observation=ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )
        
        # Update sampler with new state if we have valid result
        if correctness > 0:
            try:
                next_state = self._create_next_state(step_idx, parsed_code, outs)
                self.sampler.update_states([next_state], [self.initial_state], save=False)
            except Exception as e:
                logger.warning(f"Failed to create next state: {e}")
                if hasattr(self.sampler, 'record_failed_rollout'):
                    self.sampler.record_failed_rollout(self.initial_state)
        elif hasattr(self.sampler, 'record_failed_rollout'):
            # Record that we tried this parent but got no valid child (for PUCT visit counts)
            self.sampler.record_failed_rollout(self.initial_state)
        
        return step_result
    
    def get_reference_answer(self) -> str:
        """Return the reference answer for logging purposes."""
        raise NotImplementedError("Reference answer not available for TTT environments.")


_DELEGATED_HOOKS = (
    "get_question",
    "is_maximize",
    "_create_next_state",
    "_build_metrics",
    "_get_code_languages",
    "_should_keep_code_separators",
    "check_format",
)

_COPIED_ATTRIBUTES = (
    "env_name",
    "reward_function",
    "state_type",
    "max_construction_len",
    "construction_length_limits",
)


def _delegate_method(original_env_type: type, method_name: str):
    original_method = getattr(original_env_type, method_name)

    def delegated(self, *args: Any, **kwargs: Any):
        return original_method(self, *args, **kwargs)

    delegated.__name__ = method_name
    delegated.__qualname__ = f"Codex{original_env_type.__name__}.{method_name}"
    delegated.__doc__ = getattr(original_method, "__doc__", None)
    return delegated


def adapt_environment(original_env_type: type, *, name: str | None = None) -> type[Environment]:
    """Return a Codex-compatible Environment subclass for an existing task."""

    class_name = name or f"Codex{original_env_type.__name__}"
    attrs: dict[str, Any] = {
        "__module__": original_env_type.__module__,
        "__doc__": (
            f"Codex-compatible adapter for `{original_env_type.__module__}."
            f"{original_env_type.__name__}`."
        ),
        "original_env_type": original_env_type,
    }

    for attr_name in _COPIED_ATTRIBUTES:
        if hasattr(original_env_type, attr_name):
            attrs[attr_name] = getattr(original_env_type, attr_name)

    attrs.setdefault("state_type", State)

    @classmethod
    def create_initial_state(cls, problem_type: str):
        return original_env_type.create_initial_state(problem_type)

    attrs["create_initial_state"] = create_initial_state

    if hasattr(original_env_type, "refresh_initial_state"):
        @classmethod
        def refresh_initial_state(cls, state: State, problem_type: str):
            return original_env_type.refresh_initial_state(state, problem_type)

        attrs["refresh_initial_state"] = refresh_initial_state

    if hasattr(original_env_type, "prepare_initial_program"):
        @classmethod
        def prepare_initial_program(cls, program: str, **kwargs: Any):
            return original_env_type.prepare_initial_program(program, **kwargs)

        attrs["prepare_initial_program"] = prepare_initial_program

    if hasattr(original_env_type, "create_seed_state_from_initial_program"):
        @classmethod
        def create_seed_state_from_initial_program(cls, program: str, **kwargs: Any):
            return original_env_type.create_seed_state_from_initial_program(
                program,
                **kwargs,
            )

        attrs["create_seed_state_from_initial_program"] = create_seed_state_from_initial_program

    for method_name in _DELEGATED_HOOKS:
        if method_name in original_env_type.__dict__:
            attrs[method_name] = _delegate_method(original_env_type, method_name)

    return type(class_name, (Environment,), attrs)
