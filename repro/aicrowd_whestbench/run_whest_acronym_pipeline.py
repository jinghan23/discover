#!/usr/bin/env python3
"""Run acronym-constrained idea proposers, one filter, then AutoEvolve.

The three-letter codes are assigned before any model call, either randomly or
by a caller driving a deterministic traversal. Proposers receive only the
WhestBench task description and one code; selected proposals become launch-time
prompt blocks for independent AutoEvolve runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import string
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STARTER = (
    REPO_ROOT
    / "repro_external/aicrowd_whestbench/submissions/candidates/german_budget013.py"
)
RUNNER = REPO_ROOT / "repro/aicrowd_whestbench/run_official_mini.sh"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def generate_codes(count: int) -> list[str]:
    """Generate unique, uniformly random three-letter uppercase codes."""
    codes: set[str] = set()
    alphabet = string.ascii_uppercase
    while len(codes) < count:
        codes.add("".join(secrets.choice(alphabet) for _ in range(3)))
    return sorted(codes)


def configured_codes(default_count: int) -> tuple[list[str], str]:
    """Return explicit caller-assigned codes or a fresh random batch."""
    raw_codes = os.environ.get("RUN_WHEST_ACRONYM_CODES", "").strip()
    if not raw_codes:
        return generate_codes(default_count), "random"

    codes = [item.strip().upper() for item in raw_codes.split(",") if item.strip()]
    if not codes:
        raise ValueError("RUN_WHEST_ACRONYM_CODES did not contain any codes")
    if len(set(codes)) != len(codes):
        raise ValueError("RUN_WHEST_ACRONYM_CODES must contain unique codes")
    invalid = [code for code in codes if re.fullmatch(r"[A-Z]{3}", code) is None]
    if invalid:
        raise ValueError(
            "RUN_WHEST_ACRONYM_CODES entries must be three uppercase letters: "
            + ", ".join(invalid)
        )
    source = os.environ.get(
        "RUN_WHEST_ACRONYM_CODE_SOURCE", "caller_assigned"
    ).strip()
    return codes, source or "caller_assigned"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _method_initials(method_name: str) -> str:
    words = re.findall(r"[A-Za-z]+", method_name)
    if len(words) != 3:
        return ""
    return "".join(word[0].upper() for word in words)


def validate_proposal(payload: dict[str, Any], expected_code: str) -> None:
    code = str(payload.get("code", "")).upper()
    method_name = str(payload.get("method_name", "")).strip()
    one_line = str(payload.get("one_line", "")).strip()
    mode_prompt = str(payload.get("mode_prompt", "")).strip()
    if code != expected_code:
        raise ValueError(f"returned code {code!r}, expected {expected_code!r}")
    if _method_initials(method_name) != expected_code:
        raise ValueError(
            f"method name {method_name!r} does not expand acronym {expected_code}"
        )
    if len(one_line) < 20:
        raise ValueError("one_line is too short")
    if len(mode_prompt) < 400:
        raise ValueError("mode_prompt is too short to guide AutoEvolve")


def _task_description() -> str:
    # Importing this module uses only the standard library and does not load the
    # WhestBench dataset or any prior experiment results.
    sys.path.insert(0, str(REPO_ROOT))
    from examples.aicrowd_whestbench.prompt import WHESTBENCH_PROMPT_BASE

    return WHESTBENCH_PROMPT_BASE.strip()


def _proposer_schema(code: str) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["code", "method_name", "one_line", "mode_prompt"],
        "properties": {
            "code": {"type": "string", "enum": [code]},
            "method_name": {"type": "string", "minLength": 5},
            "one_line": {"type": "string", "minLength": 20},
            "mode_prompt": {"type": "string", "minLength": 400},
        },
    }


def _filter_schema(codes: list[str], keep: int) -> dict[str, Any]:
    verdict = {
        "type": "object",
        "additionalProperties": False,
        "required": ["code", "reason"],
        "properties": {
            "code": {"type": "string", "enum": codes},
            "reason": {"type": "string", "minLength": 40},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["selected", "rejected", "global_reasoning"],
        "properties": {
            "selected": {
                "type": "array",
                "minItems": keep,
                "maxItems": keep,
                "items": verdict,
            },
            "rejected": {
                "type": "array",
                "minItems": len(codes) - keep,
                "maxItems": len(codes) - keep,
                "items": verdict,
            },
            "global_reasoning": {"type": "string", "minLength": 80},
        },
    }


def _proposer_prompt(code: str, task_description: str) -> str:
    return f"""You are one independent WhestBench idea proposer.

An external program assigned this three-letter code before this model call:
{code}

Invent one technically serious estimator method based ONLY on the task
description below. The method must have an English name of exactly three
alphabetic words whose initials, in order, are {code}. The random letters are a
diversity constraint, not a hint toward a prewritten method. Work backward from
the actual estimation problem to give the expansion real technical meaning.

Do not inspect files, use tools, search prior runs, implement code, or assume
knowledge of any predefined diversity modes. Develop the idea yourself.

The mode_prompt will be handed to a downstream coding agent. Make it concrete:
state the mechanism and formulas, why it targets final-layer MSE under the FLOP
score, how to remain valid under flopscope, a feasible implementation recipe,
guards/fallbacks, and ablations that distinguish the core mechanism. Avoid a
generic list of unrelated ideas. The named {code} mechanism must dominate.

TASK DESCRIPTION
================
{task_description}
================

Return only the JSON object required by the supplied schema.
"""


def _filter_prompt(
    proposals: list[dict[str, Any]], task_description: str, keep: int
) -> str:
    proposal_text = json.dumps(proposals, indent=2, sort_keys=True)
    return f"""You are the single independent filter for WhestBench ideas.

Review every proposal against the task description. Select exactly {keep} for
expensive AutoEvolve follow-up. Judge mathematical relevance to the target
expectation, likely final-layer MSE reduction per charged FLOP, feasibility in
the immutable flopscope API, unbiasedness or clearly justified approximation,
implementation risk, and whether the proposal is a focused method rather than
acronym-shaped wordplay. Penalize redundant proposals and unsupported claims.

Give a specific, decision-grade reason for every selected and rejected code.
Do not modify proposals, implement code, inspect files, use tools, or use prior
experiment results. Return only the JSON object required by the schema.

TASK DESCRIPTION
================
{task_description}
================

PROPOSALS
=========
{proposal_text}
=========
"""


def _codex_command(
    codex_command: str,
    model: str,
    reasoning_effort: str,
    workspace: Path,
    output_path: Path,
    schema_path: Path,
) -> list[str]:
    command = [
        codex_command,
        "exec",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
        "-C",
        str(workspace),
        "-m",
        model,
        "-c",
        f"model_reasoning_effort={json.dumps(reasoning_effort)}",
        "--output-schema",
        str(schema_path),
        "-o",
        str(output_path),
        "-",
    ]
    return command


def _start_model_call(
    *,
    name: str,
    workspace: Path,
    prompt: str,
    schema: dict[str, Any],
    codex_command: str,
    model: str,
    reasoning_effort: str,
) -> tuple[subprocess.Popen[str], dict[str, Any]]:
    workspace.mkdir(parents=True, exist_ok=False)
    prompt_path = workspace / "prompt.txt"
    schema_path = workspace / "output_schema.json"
    output_path = workspace / "response.json"
    stdout_path = workspace / "codex.stdout.log"
    stderr_path = workspace / "codex.stderr.log"
    prompt_path.write_text(prompt, encoding="utf-8")
    _atomic_json(schema_path, schema)
    command = _codex_command(
        codex_command,
        model,
        reasoning_effort,
        workspace,
        output_path,
        schema_path,
    )
    _atomic_json(workspace / "command.json", command)
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=workspace,
        stdin=subprocess.PIPE,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
    )
    assert process.stdin is not None
    process.stdin.write(prompt)
    process.stdin.close()
    record = {
        "name": name,
        "pid": process.pid,
        "started_at": _utc_now(),
        "workspace": str(workspace),
        "output": str(output_path),
        "_stdout_handle": stdout_handle,
        "_stderr_handle": stderr_handle,
    }
    return process, record


def _finish_model_call(
    process: subprocess.Popen[str], record: dict[str, Any]
) -> dict[str, Any]:
    returncode = process.wait()
    record.pop("_stdout_handle").close()
    record.pop("_stderr_handle").close()
    record.update(
        {
            "completed_at": _utc_now(),
            "returncode": returncode,
            "status": "completed" if returncode == 0 else "failed",
        }
    )
    return record


def _validate_filter(
    payload: dict[str, Any], available_codes: list[str], keep: int
) -> list[str]:
    selected = payload.get("selected")
    rejected = payload.get("rejected")
    if not isinstance(selected, list) or not isinstance(rejected, list):
        raise ValueError("filter must return selected and rejected arrays")
    selected_codes = [str(item.get("code", "")) for item in selected]
    rejected_codes = [str(item.get("code", "")) for item in rejected]
    if len(selected_codes) != keep or len(set(selected_codes)) != keep:
        raise ValueError(f"filter must select exactly {keep} distinct codes")
    if set(selected_codes + rejected_codes) != set(available_codes):
        raise ValueError("filter verdicts must cover every and only available code")
    if set(selected_codes) & set(rejected_codes):
        raise ValueError("filter selected/rejected sets overlap")
    return selected_codes


def _mode_file(
    proposal: dict[str, Any], filter_reason: str, destination: Path
) -> None:
    code = str(proposal["code"])
    method_name = str(proposal["method_name"])
    content = f"""--- Filter-Selected Acronym Mode ---
Code: `{code}`
Method: `{method_name}`

This method was independently proposed from the task description after code
assigned the acronym `{code}`. Filter rationale: {filter_reason}

{str(proposal["mode_prompt"]).strip()}

Hard constraints for this AutoEvolve run:
- `{code}` / `{method_name}` MUST remain the core mechanism.
- Do not silently replace it with another estimator family.
- Use blackbox evaluations to improve the in-method implementation.
- Return the best valid in-method candidate even if it does not beat the parent.
"""
    destination.write_text(content, encoding="utf-8")


def _autoevolve_env(
    *, code: str, tag: str, run_root: Path, mode_file: Path, starter: Path
) -> dict[str, str]:
    env = os.environ.copy()
    socket_tag = re.sub(r"[^a-zA-Z0-9]", "", tag)[-20:] or "run"
    env.update(
        {
            "PYTHON": os.environ.get("PYTHON", "python"),
            "RUN_WHEST_CODEX_COMMAND": os.environ.get(
                "RUN_WHEST_ACRONYM_CODEX_COMMAND", "codex"
            ),
            "RUN_WHEST_CODEX_MODEL": os.environ.get(
                "RUN_WHEST_ACRONYM_MODEL", "gpt-5.6-sol"
            ),
            "RUN_WHEST_LOG_ROOT": str(run_root),
            "RUN_WHEST_AUTO_EXPERIMENT_NAME": f"autoevolve_{code.lower()}_{tag}",
            # The filtered follow-up is exactly one complete AutoEvolve
            # iteration, evaluated on the full 100-MLP mini suite.
            "RUN_WHEST_AUTO_NUM_EPOCHS": "1",
            "RUN_WHEST_AUTO_GROUP_SIZE": os.environ.get(
                "RUN_WHEST_ACRONYM_AUTO_GROUP_SIZE", "2"
            ),
            "RUN_WHEST_AUTO_GROUPS_PER_BATCH": os.environ.get(
                "RUN_WHEST_ACRONYM_AUTO_GROUPS_PER_BATCH", "1"
            ),
            "RUN_WHEST_AUTO_MAX_CONCURRENT_REQUESTS": os.environ.get(
                "RUN_WHEST_ACRONYM_AUTO_MAX_CONCURRENT_REQUESTS", "2"
            ),
            "RUN_WHEST_AUTO_CLI_TIMEOUT": "21600",
            "WHEST_SEARCH_N_MLPS": "100",
            "WHEST_INITIAL_ESTIMATOR_PATH": str(starter),
            "WHEST_DIVERSITY_MODE_FILE": str(mode_file),
            "TTT_BLACKBOX_EVAL_SOCKET": f"/tmp/whest_acr_{socket_tag}_{code.lower()}.sock",
            "TTT_BLACKBOX_EVAL_LOG_DIR": f"/tmp/whest_acr_{socket_tag}_{code.lower()}_logs",
        }
    )
    reasoning_effort = os.environ.get(
        "RUN_WHEST_ACRONYM_REASONING_EFFORT", "xhigh"
    )
    env["RUN_WHEST_CLI_REASONING_EFFORT"] = reasoning_effort
    return env


def _start_autoevolve(
    *, code: str, tag: str, run_root: Path, mode_file: Path, starter: Path
) -> tuple[subprocess.Popen[str], dict[str, Any]]:
    log_dir = run_root / "operator_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    console_path = log_dir / f"autoevolve_{code.lower()}.console.log"
    console_handle = console_path.open("w", encoding="utf-8")
    env = _autoevolve_env(
        code=code,
        tag=tag,
        run_root=run_root,
        mode_file=mode_file,
        starter=starter,
    )
    run_timeout_s = 21600
    process = subprocess.Popen(
        [
            "timeout",
            "--signal=TERM",
            "--kill-after=60s",
            f"{run_timeout_s}s",
            "bash",
            str(RUNNER),
            "autoevolve",
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=console_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    record = {
        "code": code,
        "pid": process.pid,
        "started_at": _utc_now(),
        "console_log": str(console_path),
        "experiment_name": env["RUN_WHEST_AUTO_EXPERIMENT_NAME"],
        "socket": env["TTT_BLACKBOX_EVAL_SOCKET"],
        "mode_file": str(mode_file),
        "n_mlps": 100,
        "num_epochs": 1,
        "run_timeout_s": run_timeout_s,
        "_console_handle": console_handle,
    }
    return process, record


def _run_pipeline(args: argparse.Namespace) -> int:
    requested_proposer_count = _positive_int_env("RUN_WHEST_ACRONYM_PROPOSERS", 6)
    codes, code_source = configured_codes(requested_proposer_count)
    proposer_count = len(codes)
    keep = _positive_int_env("RUN_WHEST_ACRONYM_KEEP", 2)
    if keep >= proposer_count:
        raise ValueError("RUN_WHEST_ACRONYM_KEEP must be less than proposer count")

    tag = os.environ.get("RUN_WHEST_ACRONYM_TAG") or datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    run_root = Path(
        os.environ.get(
            "RUN_WHEST_ACRONYM_ROOT",
            REPO_ROOT / f"codex_runs/aicrowd_whestbench/acronym_pipeline_{tag}",
        )
    ).resolve()
    starter = Path(
        os.environ.get("RUN_WHEST_ACRONYM_STARTER", DEFAULT_STARTER)
    ).resolve()
    codex_command = os.environ.get("RUN_WHEST_ACRONYM_CODEX_COMMAND", "codex")
    model = os.environ.get("RUN_WHEST_ACRONYM_MODEL", "gpt-5.6-sol")
    reasoning_effort = os.environ.get(
        "RUN_WHEST_ACRONYM_REASONING_EFFORT", "xhigh"
    )
    if run_root.exists():
        raise FileExistsError(f"Pipeline run root already exists: {run_root}")
    if not starter.is_file():
        raise FileNotFoundError(f"Starter not found: {starter}")
    if not RUNNER.is_file():
        raise FileNotFoundError(f"AutoEvolve runner not found: {RUNNER}")
    if shutil.which(codex_command) is None:
        raise FileNotFoundError(f"Codex command not found: {codex_command}")

    task_description = _task_description()
    run_root.mkdir(parents=True)
    (run_root / "task_description.txt").write_text(
        task_description + "\n", encoding="utf-8"
    )
    manifest: dict[str, Any] = {
        "pipeline": "acronym_proposer_filter_autoevolve",
        "created_at": _utc_now(),
        "status": "initialized",
        "tag": tag,
        "run_root": str(run_root),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "proposer_count": proposer_count,
        "filter_keep": keep,
        "codes": codes,
        "code_source": code_source,
        "starter": str(starter),
        "autoevolve_n_mlps": 100,
        "autoevolve_num_epochs": 1,
        "autoevolve_run_timeout_s": 21600,
        "proposers": {},
        "filter": {},
        "autoevolve": {},
    }
    manifest_path = run_root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    print(
        f"[{_utc_now()}] assigned codes ({code_source}): {', '.join(codes)}",
        flush=True,
    )

    if args.dry_run:
        for code in codes:
            workspace = run_root / "proposers" / code.lower()
            workspace.mkdir(parents=True)
            (workspace / "prompt.txt").write_text(
                _proposer_prompt(code, task_description), encoding="utf-8"
            )
            _atomic_json(workspace / "output_schema.json", _proposer_schema(code))
        manifest["status"] = "dry_run_complete"
        manifest["completed_at"] = _utc_now()
        _atomic_json(manifest_path, manifest)
        print(f"[{_utc_now()}] dry run complete: {run_root}", flush=True)
        return 0

    manifest["status"] = "proposers_running"
    proposer_processes: dict[str, tuple[subprocess.Popen[str], dict[str, Any]]] = {}
    for code in codes:
        process, record = _start_model_call(
            name=f"proposer_{code}",
            workspace=run_root / "proposers" / code.lower(),
            prompt=_proposer_prompt(code, task_description),
            schema=_proposer_schema(code),
            codex_command=codex_command,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        proposer_processes[code] = (process, record)
        manifest["proposers"][code] = {
            key: value for key, value in record.items() if not key.startswith("_")
        }
        print(f"[{_utc_now()}] proposer {code} started pid={process.pid}", flush=True)
    _atomic_json(manifest_path, manifest)

    proposals: list[dict[str, Any]] = []
    for code, (process, record) in proposer_processes.items():
        record = _finish_model_call(process, record)
        public_record = {
            key: value for key, value in record.items() if not key.startswith("_")
        }
        try:
            if record["returncode"] != 0:
                raise RuntimeError(f"Codex exited {record['returncode']}")
            proposal = _load_json(Path(record["output"]))
            validate_proposal(proposal, code)
            public_record["status"] = "validated"
            public_record["method_name"] = proposal["method_name"]
            proposals.append(proposal)
            print(
                f"[{_utc_now()}] proposer {code} validated: {proposal['method_name']}",
                flush=True,
            )
        except Exception as exc:
            public_record["status"] = "invalid"
            public_record["error"] = str(exc)
            print(f"[{_utc_now()}] proposer {code} invalid: {exc}", flush=True)
        manifest["proposers"][code] = public_record
        _atomic_json(manifest_path, manifest)

    if len(proposals) <= keep:
        manifest["status"] = "failed_insufficient_valid_proposals"
        manifest["completed_at"] = _utc_now()
        _atomic_json(manifest_path, manifest)
        raise RuntimeError(
            f"Only {len(proposals)} valid proposals; need more than filter_keep={keep}"
        )

    valid_codes = [str(proposal["code"]) for proposal in proposals]
    manifest["status"] = "filter_running"
    filter_process, filter_record = _start_model_call(
        name="filter",
        workspace=run_root / "filter",
        prompt=_filter_prompt(proposals, task_description, keep),
        schema=_filter_schema(valid_codes, keep),
        codex_command=codex_command,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    manifest["filter"] = {
        key: value for key, value in filter_record.items() if not key.startswith("_")
    }
    _atomic_json(manifest_path, manifest)
    print(f"[{_utc_now()}] filter started pid={filter_process.pid}", flush=True)
    filter_record = _finish_model_call(filter_process, filter_record)
    if filter_record["returncode"] != 0:
        raise RuntimeError(f"Filter Codex exited {filter_record['returncode']}")
    filter_payload = _load_json(Path(filter_record["output"]))
    selected_codes = _validate_filter(filter_payload, valid_codes, keep)
    manifest["filter"] = {
        key: value for key, value in filter_record.items() if not key.startswith("_")
    }
    manifest["filter"]["status"] = "validated"
    manifest["filter"]["selected_codes"] = selected_codes
    manifest["filter"]["verdict"] = filter_payload
    _atomic_json(manifest_path, manifest)
    print(f"[{_utc_now()}] filter selected: {', '.join(selected_codes)}", flush=True)

    proposals_by_code = {str(item["code"]): item for item in proposals}
    selected_reasons = {
        str(item["code"]): str(item["reason"])
        for item in filter_payload["selected"]
    }
    mode_dir = run_root / "selected_modes"
    mode_dir.mkdir()
    auto_processes: dict[str, tuple[subprocess.Popen[str], dict[str, Any]]] = {}
    manifest["status"] = "autoevolve_running"
    for code in selected_codes:
        mode_path = mode_dir / f"{code.lower()}.md"
        _mode_file(proposals_by_code[code], selected_reasons[code], mode_path)
        process, record = _start_autoevolve(
            code=code,
            tag=tag,
            run_root=run_root,
            mode_file=mode_path,
            starter=starter,
        )
        auto_processes[code] = (process, record)
        manifest["autoevolve"][code] = {
            key: value for key, value in record.items() if not key.startswith("_")
        }
        print(f"[{_utc_now()}] AutoEvolve {code} started pid={process.pid}", flush=True)
    _atomic_json(manifest_path, manifest)

    failed = False
    for code, (process, record) in auto_processes.items():
        returncode = process.wait()
        record.pop("_console_handle").close()
        record.update(
            {
                "returncode": returncode,
                "completed_at": _utc_now(),
                "status": "completed" if returncode == 0 else "failed",
            }
        )
        failed = failed or returncode != 0
        manifest["autoevolve"][code] = record
        _atomic_json(manifest_path, manifest)
        print(f"[{_utc_now()}] AutoEvolve {code} exited {returncode}", flush=True)

    manifest["status"] = "failed" if failed else "completed"
    manifest["completed_at"] = _utc_now()
    _atomic_json(manifest_path, manifest)
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Acronym proposers, one filter, then WhestBench AutoEvolve"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Generate codes/prompts without model calls"
    )
    args = parser.parse_args()
    try:
        return _run_pipeline(args)
    except Exception as exc:
        print(f"pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
