from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ttt_discover import DiscoverConfig, discover
from ttt_discover.config import DISCOVER_CONFIG_FIELDS


DISCOVER_FIELDS = set(DISCOVER_CONFIG_FIELDS)
LAUNCH_FIELDS = {"env"}
PATH_FIELDS = {"log_path"}

FIELD_ALIASES = {
    "codex_backend": "backend",
    "codex_model_name": "model_name",
    "codex_max_output_tokens": "max_output_tokens",
    "codex_temperature": "temperature",
    "codex_cli_command": "cli_command",
    "codex_cli_sandbox": "cli_sandbox",
    "codex_cli_timeout": "cli_timeout",
    "codex_max_concurrent_requests": "max_concurrent_requests",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ModuleNotFoundError:
        try:
            from omegaconf import OmegaConf
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "YAML config loading requires PyYAML or OmegaConf. "
                "Install `pyyaml` or use the existing `hydra-core` dependency."
            ) from exc
        data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    else:
        data = yaml.safe_load(text)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML value must be a mapping: {path}")
    return data


def _expand_vars(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_vars(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_vars(item) for item in value)
    if isinstance(value, dict):
        return {key: _expand_vars(item) for key, item in value.items()}
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _normalize_run_map(raw_runs: Any) -> dict[str, dict[str, Any]]:
    if isinstance(raw_runs, Mapping):
        runs = {}
        for name, cfg in raw_runs.items():
            if not isinstance(cfg, Mapping):
                raise ValueError(f"Run {name!r} must be a mapping")
            runs[str(name)] = dict(cfg)
        return runs

    if isinstance(raw_runs, list):
        runs = {}
        for idx, item in enumerate(raw_runs):
            if not isinstance(item, Mapping):
                raise ValueError(f"Run entry #{idx} must be a mapping")
            if "name" not in item:
                raise ValueError(f"Run entry #{idx} is missing required key `name`")
            name = str(item["name"])
            cfg = {key: value for key, value in item.items() if key != "name"}
            runs[name] = cfg
        return runs

    raise ValueError("`runs` must be either a mapping or a list of run mappings")


def _split_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    defaults = payload.get("defaults", {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, Mapping):
        raise ValueError("`defaults` must be a mapping when present")

    if "runs" in payload:
        return dict(defaults), _normalize_run_map(payload["runs"])

    single_run = {
        key: value
        for key, value in payload.items()
        if key not in {"defaults", "runs"}
    }
    if not single_run:
        raise ValueError("YAML file must define `runs` or a single top-level run")
    return dict(defaults), {"default": single_run}


def _canonicalize_keys(config: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for key, value in config.items():
        canonical = FIELD_ALIASES.get(str(key), str(key))
        if canonical in out:
            previous = sources[canonical]
            raise ValueError(
                f"Duplicate config field for {canonical!r}: {previous!r} and {key!r}"
            )
        out[canonical] = value
        sources[canonical] = str(key)

    unknown = sorted(set(out) - DISCOVER_FIELDS - LAUNCH_FIELDS)
    if unknown:
        raise ValueError(
            "Unknown DiscoverConfig field(s): "
            + ", ".join(unknown)
            + ". Use fields from ttt_discover/config.py or supported launch fields: "
            + ", ".join(sorted(LAUNCH_FIELDS))
        )
    return out


def _split_launch_options(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str | None]]:
    config_fields = dict(config)
    raw_env = config_fields.pop("env", {})
    unknown_launch = sorted(set(config_fields) - DISCOVER_FIELDS)
    if unknown_launch:
        raise ValueError(
            "Unknown DiscoverConfig field(s): "
            + ", ".join(unknown_launch)
            + ". Use fields from ttt_discover/config.py or supported launch fields: "
            + ", ".join(sorted(LAUNCH_FIELDS))
        )
    if raw_env is None:
        return config_fields, {}
    if not isinstance(raw_env, Mapping):
        raise ValueError("`env` must be a mapping when present")
    return config_fields, {
        str(key): None if value is None else str(value)
        for key, value in raw_env.items()
    }


def _import_object(spec: str) -> Any:
    module_name, sep, attr = spec.partition(":")
    if not sep:
        module_name, sep, attr = spec.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Import spec must be `module:object`: {spec!r}")

    module = importlib.import_module(module_name)
    obj: Any = module
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _resolve_repo_path(path_value: str) -> str:
    expanded = os.path.expanduser(os.path.expandvars(path_value))
    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _coerce_config_values(
    config: Mapping[str, Any],
    *,
    resolve_imports: bool,
) -> dict[str, Any]:
    out = dict(config)

    env_type = out.get("env_type")
    if resolve_imports and isinstance(env_type, str):
        out["env_type"] = _import_object(env_type)

    for field in PATH_FIELDS:
        if field in out and isinstance(out[field], str) and out[field]:
            out[field] = _resolve_repo_path(out[field])

    if out.get("algorithm") == "autoevolve" and "cli_sandbox" not in out:
        out["cli_sandbox"] = "workspace-write"

    return out


def build_config(
    config_path: Path,
    run_name: str | None,
    *,
    resolve_imports: bool = True,
) -> tuple[str, DiscoverConfig, dict[str, str | None]]:
    payload = _expand_vars(_load_yaml(config_path))
    defaults, runs = _split_payload(payload)

    if run_name is None:
        if len(runs) != 1:
            available = ", ".join(sorted(runs))
            raise ValueError(f"--run is required. Available runs: {available}")
        run_name = next(iter(runs))

    if run_name not in runs:
        available = ", ".join(sorted(runs))
        raise KeyError(f"Unknown run {run_name!r}. Available runs: {available}")

    normalized_defaults = _canonicalize_keys(defaults)
    normalized_run = _canonicalize_keys(runs[run_name])
    normalized, launch_env = _split_launch_options(
        _deep_merge(normalized_defaults, normalized_run)
    )
    coerced = _coerce_config_values(normalized, resolve_imports=resolve_imports)
    return run_name, DiscoverConfig(**coerced), launch_env


def _config_for_printing(config: DiscoverConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in DISCOVER_CONFIG_FIELDS:
        value = getattr(config, field)
        if isinstance(value, type):
            out[field] = f"{value.__module__}:{value.__name__}"
        elif isinstance(value, tuple):
            out[field] = list(value)
        else:
            out[field] = value
    return out


def parse_args(
    argv: list[str] | None = None,
    *,
    default_run: str | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a TTT-Discover run from YAML.")
    parser.add_argument(
        "--config",
        "-c",
        default=str(REPO_ROOT / "configs" / "discovery_runs.yaml"),
        help="YAML file containing defaults and runs.",
    )
    parser.add_argument(
        "--run",
        "-r",
        default=default_run,
        help="Run name under the YAML `runs` mapping. Required when multiple runs exist.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List run names from the YAML file and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved DiscoverConfig and exit without launching.",
    )
    return parser.parse_args(argv)


def _apply_launch_env(env: Mapping[str, str | None]) -> None:
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def main(
    argv: list[str] | None = None,
    *,
    default_run: str | None = None,
) -> None:
    args = parse_args(argv, default_run=default_run)
    config_path = Path(args.config).expanduser().resolve()
    payload = _expand_vars(_load_yaml(config_path))
    _defaults, runs = _split_payload(payload)

    if args.list:
        for name in sorted(runs):
            print(name)
        return

    run_name, config, launch_env = build_config(
        config_path,
        args.run,
        resolve_imports=not args.dry_run,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "run": run_name,
                    "env": launch_env,
                    "config": _config_for_printing(config),
                },
                indent=2,
            )
        )
        return

    _apply_launch_env(launch_env)
    os.chdir(REPO_ROOT)
    discover(config)


if __name__ == "__main__":
    main()
