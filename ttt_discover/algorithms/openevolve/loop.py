from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from ttt_discover.algorithms.state import State
from ttt_discover.tasks import Task


_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n([\s\S]*?)```", re.IGNORECASE)


def _safe_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return safe or "artifact"


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def _strip_code_fence(text: str) -> str:
    matches = list(_CODE_BLOCK_RE.finditer(text or ""))
    if not matches:
        return (text or "").strip()
    return matches[-1].group(1).strip()


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _drop_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_none(v) for v in value if v is not None]
    return value


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"Expected a mapping for openevolve_oe, got {type(value)}")


def _resolve_path(path: str | os.PathLike[str], *, base_dir: Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def _env_type_spec(env_type: type) -> str:
    return f"{env_type.__module__}:{env_type.__qualname__}"


def _maybe_create_seed_state(
    task: Task,
    program: str,
    *,
    source_path: str,
    eval_timeout: int,
) -> State | None:
    factory = getattr(task.env_type, "create_seed_state_from_initial_program", None)
    if factory is None:
        return None
    state = factory(
        program,
        source_path=source_path,
        eval_timeout=eval_timeout,
    )
    return state


def _prepare_initial_program(cfg: Any, task: Task, output_dir: Path) -> tuple[Path, State]:
    explicit_path = getattr(cfg, "openevolve_initial_program", None)
    initial_state: State | None = None
    if explicit_path:
        source_path = _resolve_path(explicit_path, base_dir=Path.cwd())
        program = source_path.read_text(encoding="utf-8", errors="replace")
        source_path_label = str(source_path)
    else:
        source_path = output_dir / "initial_state.py"
        initial_state = task.create_initial_state()
        program = initial_state.code
        source_path_label = "<initial_state>"

    program = _strip_code_fence(program)
    if not program:
        raise ValueError(
            "OpenEvolve needs an initial source program. Set "
            "`openevolve_initial_program` or provide non-empty initial state code."
        )

    prepare = getattr(task.env_type, "prepare_initial_program", None)
    if prepare is not None:
        program = prepare(
            program,
            source_path=source_path_label,
            eval_timeout=int(cfg.eval_timeout),
        )

    seed_state = _maybe_create_seed_state(
        task,
        program,
        source_path=source_path_label,
        eval_timeout=int(cfg.eval_timeout),
    )
    if seed_state is not None:
        initial_state = seed_state
    if initial_state is None:
        initial_state = task.create_initial_state()
        if explicit_path and not getattr(initial_state, "code", ""):
            initial_state.code = f"```python\n{program.rstrip()}\n```"

    suffix = source_path.suffix if explicit_path and source_path.suffix == ".py" else ".py"
    initial_program_path = output_dir / f"initial_program{suffix}"
    initial_program_path.write_text(program.rstrip() + "\n", encoding="utf-8")
    return initial_program_path, initial_state


def _trace_path(output_dir: Path, fmt: str) -> Path:
    if fmt == "jsonl":
        return output_dir / "evolution_trace.jsonl"
    if fmt in {"hdf5", "h5"}:
        return output_dir / "evolution_trace.hdf5"
    return output_dir / f"evolution_trace.{fmt}"


def _build_openevolve_config(cfg: Any, Config: type, output_dir: Path, db_dir: Path) -> Any:
    config_path_value = getattr(cfg, "openevolve_config_path", None)
    if config_path_value:
        config_path = _resolve_path(config_path_value, base_dir=Path.cwd())
        config = Config.from_yaml(config_path)
    else:
        config = Config()

    overrides = _as_mapping(getattr(cfg, "openevolve_oe", None))
    if overrides:
        merged = config.to_dict()
        _deep_merge_dict(merged, overrides)
        config = Config.from_dict(_drop_none(merged))

    iterations = int(cfg.num_epochs)
    config.max_iterations = iterations
    if getattr(cfg, "openevolve_checkpoint_interval", None) is not None:
        config.checkpoint_interval = int(cfg.openevolve_checkpoint_interval)
    if getattr(cfg, "openevolve_max_code_length", None) is not None:
        config.max_code_length = int(cfg.openevolve_max_code_length)
    if not config_path_value and "evaluator" not in overrides:
        config.evaluator.cascade_evaluation = False
    config.evaluator.timeout = max(1, int(cfg.eval_timeout))
    config.evaluator.parallel_evaluations = max(1, int(cfg.max_concurrent_requests or 1))
    config.database.db_path = str(db_dir)
    config.log_dir = config.log_dir or str(output_dir / "logs")

    trace_format = str(getattr(cfg, "openevolve_trace_format", "jsonl") or "jsonl")
    config.evolution_trace.enabled = bool(getattr(cfg, "openevolve_trace", False))
    config.evolution_trace.format = trace_format
    config.evolution_trace.output_path = str(_trace_path(output_dir, trace_format))

    api_base = cfg.base_url or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    api_key = os.environ.get(str(cfg.api_key_env or "OPENAI_API_KEY"), "")
    if not api_key and cfg.api_key_env != "OPENAI_API_KEY":
        api_key = os.environ.get("OPENAI_API_KEY", "")
    model = cfg.model_name or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    config.llm.api_base = api_base
    if api_key:
        config.llm.api_key = api_key
    if cfg.temperature is not None:
        config.llm.temperature = float(cfg.temperature)
    if cfg.max_output_tokens is not None:
        config.llm.max_tokens = int(cfg.max_output_tokens)
    if getattr(cfg, "cli_timeout", None) is not None:
        config.llm.timeout = max(1, int(cfg.cli_timeout))

    if not getattr(config.llm, "models", None):
        config.llm.primary_model = model
        config.llm.primary_model_weight = 1.0
        config.llm.rebuild_models()

    config.llm.update_model_params(
        {
            "api_base": config.llm.api_base,
            "api_key": getattr(config.llm, "api_key", None),
            "temperature": config.llm.temperature,
            "top_p": getattr(config.llm, "top_p", None),
            "max_tokens": config.llm.max_tokens,
            "timeout": config.llm.timeout,
            "retries": config.llm.retries,
            "retry_delay": config.llm.retry_delay,
            "random_seed": getattr(config.llm, "random_seed", None),
            "reasoning_effort": getattr(config.llm, "reasoning_effort", None),
            "manual_mode": getattr(config.llm, "manual_mode", False),
        },
        overwrite=False,
    )

    manual_mode = bool(getattr(config.llm, "manual_mode", False))
    if not manual_mode:
        models = list(getattr(config.llm, "models", []) or [])
        models += list(getattr(config.llm, "evaluator_models", []) or [])
        has_any_api_key = bool(getattr(config.llm, "api_key", None)) or any(
            getattr(entry, "api_key", None) for entry in models
        )
        if not has_any_api_key:
            if iterations <= 0:
                dummy = "DUMMY_API_KEY_FOR_ZERO_ITERATIONS"
                config.llm.api_key = dummy
                config.llm.update_model_params({"api_key": dummy}, overwrite=False)
            else:
                raise RuntimeError(
                    "Missing API key for OpenEvolve. Set "
                    f"`{cfg.api_key_env}` or provide `openevolve_oe.llm.api_key`."
                )

    return config


def _export_history(controller: Any, history_dir: Path) -> None:
    history_dir.mkdir(parents=True, exist_ok=True)
    programs = list(controller.database.programs.values())
    programs.sort(
        key=lambda p: (
            int(getattr(p, "iteration_found", 0) or 0),
            float(getattr(p, "timestamp", 0.0) or 0.0),
        )
    )

    with (history_dir / "index.jsonl").open("w", encoding="utf-8") as f:
        for program in programs:
            iter_num = int(getattr(program, "iteration_found", 0) or 0)
            program_dir = history_dir / f"iter_{iter_num:06d}__{program.id}"
            program_dir.mkdir(parents=True, exist_ok=True)

            (program_dir / f"program{controller.file_extension}").write_text(
                program.code,
                encoding="utf-8",
                errors="replace",
            )
            (program_dir / "metrics.json").write_text(
                _safe_json(program.metrics or {}),
                encoding="utf-8",
            )
            meta = {
                "id": program.id,
                "parent_id": program.parent_id,
                "generation": program.generation,
                "timestamp": program.timestamp,
                "iteration_found": program.iteration_found,
                "language": program.language,
                "changes_description": getattr(program, "changes_description", ""),
                "metadata": program.metadata,
                "prompts": getattr(program, "prompts", None),
            }
            (program_dir / "meta.json").write_text(_safe_json(meta), encoding="utf-8")

            artifacts = controller.database.get_artifacts(program.id)
            if artifacts:
                artifacts_dir = program_dir / "artifacts"
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                manifest: dict[str, str] = {}
                used_names: set[str] = set()
                for key, value in artifacts.items():
                    base_name = _safe_filename(str(key))
                    name = base_name
                    i = 1
                    while name in used_names:
                        i += 1
                        name = f"{base_name}__{i}"
                    used_names.add(name)
                    manifest[str(key)] = name
                    artifact_path = artifacts_dir / name
                    if isinstance(value, bytes):
                        artifact_path.write_bytes(value)
                    else:
                        artifact_path.write_text(
                            value if isinstance(value, str) else _safe_json(value),
                            encoding="utf-8",
                            errors="replace",
                        )
                (artifacts_dir / "manifest.json").write_text(
                    _safe_json(manifest),
                    encoding="utf-8",
                )

            record = {
                "iteration": iter_num,
                "id": program.id,
                "parent_id": program.parent_id,
                "generation": program.generation,
                "metrics": program.metrics or {},
            }
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _set_entrypoint_env(cfg: Any, task: Task, initial_state: State) -> dict[str, str | None]:
    root = Path.cwd().resolve()
    old = {
        key: os.environ.get(key)
        for key in (
            "TTT_DISCOVER_ROOT",
            "TTT_OPENEVOLVE_ENV_TYPE",
            "TTT_OPENEVOLVE_PROBLEM_TYPE",
            "TTT_OPENEVOLVE_LOG_DIR",
            "TTT_OPENEVOLVE_EVAL_TIMEOUT",
            "TTT_OPENEVOLVE_NUM_CPUS_PER_TASK",
            "TTT_OPENEVOLVE_TIMEOUT",
            "TTT_OPENEVOLVE_INITIAL_STATE_JSON",
            "PYTHONPATH",
        )
    }
    path_entries = [str(root)]
    if old["PYTHONPATH"]:
        path_entries.append(old["PYTHONPATH"])
    os.environ.update(
        {
            "TTT_DISCOVER_ROOT": str(root),
            "TTT_OPENEVOLVE_ENV_TYPE": _env_type_spec(task.env_type),
            "TTT_OPENEVOLVE_PROBLEM_TYPE": str(task.problem_type),
            "TTT_OPENEVOLVE_LOG_DIR": str(cfg.log_path),
            "TTT_OPENEVOLVE_EVAL_TIMEOUT": str(int(cfg.eval_timeout)),
            "TTT_OPENEVOLVE_NUM_CPUS_PER_TASK": str(int(cfg.num_cpus_per_task)),
            "TTT_OPENEVOLVE_TIMEOUT": str(float(cfg.timeout)),
            "TTT_OPENEVOLVE_INITIAL_STATE_JSON": json.dumps(
                initial_state.to_dict(),
                ensure_ascii=False,
                default=str,
            ),
            "PYTHONPATH": os.pathsep.join(path_entries),
        }
    )
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _store_initial_artifacts(controller: Any) -> None:
    initial_candidates = [
        p for p in controller.database.programs.values() if p.parent_id is None
    ]
    initial_program_obj = None
    if len(initial_candidates) == 1:
        initial_program_obj = initial_candidates[0]
    else:
        for program in initial_candidates:
            if program.code == controller.initial_program_code:
                initial_program_obj = program
                break
    if initial_program_obj is None:
        return
    pending = controller.evaluator.get_pending_artifacts(initial_program_obj.id)
    if pending:
        controller.database.store_artifacts(initial_program_obj.id, pending)


async def run(cfg: Any, *, task: Task) -> None:
    try:
        from openevolve import Config, OpenEvolve
    except Exception as exc:
        raise RuntimeError(
            "OpenEvolve is not importable. Install the optional dependency with "
            "`pip install 'ttt-discover[openevolve]'` or `pip install openevolve==0.2.26`."
        ) from exc

    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    output_dir_value = getattr(cfg, "openevolve_output_dir", None)
    output_dir = (
        _resolve_path(output_dir_value, base_dir=Path.cwd())
        if output_dir_value
        else (Path(cfg.log_path).expanduser().resolve() / "openevolve")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    db_dir = output_dir / "db"
    history_dir = output_dir / "history"

    initial_program_path, initial_state = _prepare_initial_program(cfg, task, output_dir)
    config = _build_openevolve_config(cfg, Config, output_dir, db_dir)
    evaluation_file = Path(__file__).with_name("entrypoint.py").resolve()
    old_env = _set_entrypoint_env(cfg, task, initial_state)
    try:
        controller = OpenEvolve(
            initial_program_path=str(initial_program_path),
            evaluation_file=str(evaluation_file),
            config=config,
            output_dir=str(output_dir),
        )
        best = await controller.run(iterations=int(cfg.num_epochs))
    finally:
        _restore_env(old_env)

    if not best:
        raise RuntimeError("OpenEvolve returned no best program")

    if bool(getattr(cfg, "openevolve_save_db", True)) or bool(
        getattr(cfg, "openevolve_export_history", True)
    ):
        _store_initial_artifacts(controller)

    if bool(getattr(cfg, "openevolve_save_db", True)):
        controller.database.save(str(db_dir), iteration=controller.database.last_iteration)

    if bool(getattr(cfg, "openevolve_export_history", True)):
        _export_history(controller, history_dir)

    metrics = best.metrics or {}
    score = metrics.get("combined_score", metrics.get("score"))
    print(f"Best score: {score}")
    if bool(getattr(cfg, "openevolve_save_db", True)):
        print(f"Saved: {db_dir}")
    if bool(getattr(cfg, "openevolve_export_history", True)):
        print(f"Saved: {history_dir}")
    if bool(getattr(cfg, "openevolve_trace", False)):
        print(f"Trace: {config.evolution_trace.output_path}")
