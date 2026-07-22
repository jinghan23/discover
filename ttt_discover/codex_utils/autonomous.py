"""Codex CLI agent-mode completer used by AutoEvolve."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

from ttt_discover.codex_utils.completers import CodexCliCompleter, TextCompleter

logger = logging.getLogger(__name__)


_TIME_WINDOW_PROTOCOL = """

--- Time-Window Search Protocol ---
Your autonomous search process has a hard 3000-second wall-clock limit. Work
empirically and manage elapsed time, since evaluator runtimes can vary widely.
Evaluate the starting submission early, then explore coherent hypotheses while
keeping a plain-text experiment log. After every improvement, immediately copy
the best valid implementation to a separate checkpoint in the workspace so a
later regression or timeout cannot erase it. Shift from broad exploration to
refining the strongest mechanism as the deadline approaches. By about 2700
seconds, stop starting risky or long experiments, restore the best checkpoint
to `submission.py`, and use the remaining time only for final validation and
robustness checks. Leave the best actually measured implementation, not merely
the most recent one, in `submission.py`. Never fabricate a score.
"""


def _append_time_window_protocol(prompt: str) -> str:
    """Add harness-level deadline and checkpoint discipline to a task prompt."""
    return prompt.rstrip() + _TIME_WINDOW_PROTOCOL


def autonomous_submission_from_workspace(
    completer: TextCompleter,
) -> tuple[str | None, str | None]:
    workspace = getattr(completer, "_workspace", None)
    if workspace is None:
        return None, None

    submission_path = Path(workspace) / "submission.py"
    try:
        code = submission_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read autonomous submission file %s: %s", submission_path, exc)
        return None, str(submission_path)

    if not code.strip():
        logger.warning("Autonomous submission file is empty: %s", submission_path)
        return None, str(submission_path)

    return code, str(submission_path)


class AutonomousCodexCliCompleter(CodexCliCompleter):
    """Codex CLI agent mode: give Codex a writable workspace and eval command."""

    @classmethod
    def from_discover_config(
        cls,
        cfg: Any,
        *,
        task: Any,
        eval_runner: Any,
        env: Any,
        semaphore: Any | None,
        step_idx: int,
    ) -> "AutonomousCodexCliCompleter":
        return cls(
            model_name=cfg.model_name,
            codex_command=cfg.cli_command,
            sandbox=cfg.cli_sandbox,
            cwd=None,
            repo_cwd=os.getcwd(),
            timeout=cfg.cli_timeout,
            semaphore=semaphore,
            log_path=cfg.log_path,
            problem_type=cfg.problem_type,
            step_idx=step_idx,
            eval_timeout=cfg.eval_timeout,
            num_cpus_per_task=max(1, int(cfg.num_cpus_per_task)),
            config_overrides=("shell_environment_policy.inherit=all",),
            isolate_danger_full_access=cfg.cli_sandbox == "danger-full-access",
            skip_git_repo_check=True,
            prompt_builder=task.autonomous_prompt_builder(eval_runner, env),
        )

    def __init__(
        self,
        *,
        log_path: str,
        problem_type: str,
        step_idx: int,
        eval_timeout: int,
        num_cpus_per_task: int,
        repo_cwd: str | os.PathLike[str] | None = None,
        prompt_builder: Callable[..., str] | None = None,
        isolate_danger_full_access: bool = False,
        **kwargs: Any,
    ):
        super().__init__(append_final_answer_instruction=False, **kwargs)
        self.log_path = log_path
        self.problem_type = problem_type
        self.step_idx = step_idx
        self.eval_timeout = eval_timeout
        self.num_cpus_per_task = num_cpus_per_task
        self.repo_cwd = Path(repo_cwd or os.getcwd()).resolve()
        self.prompt_builder = prompt_builder
        self.isolate_danger_full_access = isolate_danger_full_access
        self._call_idx = 0

    def _isolated_workspace(self) -> Path:
        return self.repo_cwd / "workspace"

    def _next_workspace(self) -> Path:
        self._call_idx += 1
        root = (Path(self.log_path) / "autoevolve_workspaces").resolve()
        workspace = (
            root
            / f"step_{max(0, self.step_idx):06d}"
            / f"call_{self._call_idx:04d}_{uuid.uuid4().hex[:8]}"
        )
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "eval_tmp").mkdir(parents=True, exist_ok=True)
        return workspace

    def _build_prompt(self, prompt: str) -> str:
        workspace = self._next_workspace()
        self._workspace = workspace

        if self.prompt_builder is not None:
            full_prompt = self.prompt_builder(
                prompt=prompt,
                workspace=workspace,
                eval_timeout=self.eval_timeout,
                num_cpus_per_task=self.num_cpus_per_task,
            )
            full_prompt = _append_time_window_protocol(full_prompt)
            if self.isolate_danger_full_access:
                full_prompt = full_prompt.replace(
                    str(workspace),
                    str(self._isolated_workspace()),
                )
            (workspace / "prompt.txt").write_text(full_prompt, encoding="utf-8")
            return full_prompt

        matches = re.findall(r"```python\s+([\s\S]*?)\s*```", prompt or "")
        parent_matches = [match for match in matches if "def priority(" in match]
        if not parent_matches:
            parent_source = "def priority(el, n):\n    return 0.0\n"
        else:
            parent_source = parent_matches[-1].strip() + "\n"
        (workspace / "candidate.py").write_text(parent_source, encoding="utf-8")

        candidate = shlex.quote(str(workspace / "candidate.py"))
        eval_dir = shlex.quote(str(workspace / "eval_tmp"))
        python_exe = shlex.quote(sys.executable)
        repo_cwd = shlex.quote(str(self.repo_cwd))
        evaluator_cmd = (
            f"PYTHONPATH={repo_cwd}${{PYTHONPATH:+:$PYTHONPATH}} "
            f"{python_exe} -m repro.cap_set.self_loop_eval "
            f"--candidate {candidate} "
            f"--dimension {shlex.quote(str(self.problem_type))} "
            f"--log-dir {eval_dir} "
            f"--eval-timeout {int(self.eval_timeout)} "
            f"--num-cpus-per-task {int(self.num_cpus_per_task)}"
        )
        full_prompt = f"""{prompt}

--- Autonomous Codex Search Mode ---
You may edit files and run shell commands, but only inside this workspace:
{workspace}

Network IO is not allowed. The current candidate is:
{workspace / "candidate.py"}

Run this reward evaluator after each revision:
{evaluator_cmd}

When done, save the primary best code to:
{workspace / "submission.py"}

You may also add more states to the outer AutoEvolve pool by saving additional
candidate Python files under:
{workspace / "state_pool"}

Create that directory if needed. As an alternative, write candidate_pool.json
with a "candidates" list containing objects like {{"name": "try1", "path": "state_pool/try1.py"}}
or {{"name": "try2", "code": "...python source..."}}.

The outer algorithm will evaluate each candidate itself before adding it.
Finish with exactly one Python code block defining the primary best `priority(el, n)`.
"""
        if self.isolate_danger_full_access:
            full_prompt = full_prompt.replace(
                str(workspace),
                str(self._isolated_workspace()),
            )
        (workspace / "prompt.txt").write_text(full_prompt, encoding="utf-8")
        return full_prompt

    def _next_call_dir(self) -> Path:
        workspace = getattr(self, "_workspace", None)
        if workspace is None:
            raise RuntimeError("Autonomous Codex workspace was not initialized.")
        return workspace

    def _build_command(self, output_path: str) -> list[str]:
        workspace = getattr(self, "_workspace", None)
        if workspace is None:
            raise RuntimeError("Autonomous Codex workspace was not initialized.")
        if self.isolate_danger_full_access:
            return self._build_isolated_command(workspace, output_path)
        original_cwd = self.cwd
        self.cwd = str(workspace)
        try:
            return super()._build_command(output_path)
        finally:
            self.cwd = original_cwd

    def _build_inner_dangerous_command(
        self,
        output_path: str,
        isolated_workspace: Path,
    ) -> list[str]:
        cmd = [
            self.codex_command,
            "exec",
            "--ephemeral",
            "--dangerously-bypass-approvals-and-sandbox",
            "-o",
            output_path,
            "-C",
            str(isolated_workspace),
            "--skip-git-repo-check",
        ]
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

    def _prepare_isolated_codex_home(self, workspace: Path) -> Path:
        codex_home = workspace.parent / f"{workspace.name}_codex_home"
        codex_home.mkdir(parents=True, exist_ok=True)
        auth_src = Path.home() / ".codex" / "auth.json"
        if auth_src.exists():
            shutil.copy2(auth_src, codex_home / "auth.json")
        for name in ("installation_id", "version.json", "models_cache.json"):
            src = Path.home() / ".codex" / name
            if src.exists():
                shutil.copy2(src, codex_home / name)
        return codex_home

    def _build_isolated_command(self, workspace: Path, output_path: str) -> list[str]:
        codex_home = self._prepare_isolated_codex_home(workspace)
        isolated_workspace = self._isolated_workspace()
        isolated_output = str(isolated_workspace / Path(output_path).name)
        inner_cmd = self._build_inner_dangerous_command(
            isolated_output,
            isolated_workspace,
        )

        mask_roots = []
        for root in (self.repo_cwd,):
            root = root.resolve()
            if root.exists() and root not in mask_roots:
                mask_roots.append(root)

        workspace_shell = shlex.quote(str(workspace))
        codex_home_shell = shlex.quote(str(codex_home))
        home = Path.home()
        nvm_shell = shlex.quote(str(home / ".nvm"))
        home_shell = shlex.quote(str(home))
        codex_home_target_shell = shlex.quote(str(home / ".codex"))
        nvm_target_shell = shlex.quote(str(home / ".nvm"))
        isolated_workspace_shell = shlex.quote(str(isolated_workspace))
        mask_roots_shell = " ".join(shlex.quote(str(path)) for path in mask_roots)
        inner_cmd_shell = shlex.join(inner_cmd)
        script = f"""
set -euo pipefail

mount --make-rprivate /

iso_root="$(mktemp -d /tmp/ttt-codex-iso.XXXXXX)"
mkdir -p "$iso_root/ws" "$iso_root/codex_home" "$iso_root/nvm"
mount --bind {workspace_shell} "$iso_root/ws"
mount --bind {codex_home_shell} "$iso_root/codex_home"
mount --bind {nvm_shell} "$iso_root/nvm"

for path in {mask_roots_shell}; do
    if [ -d "$path" ]; then
        mount -t tmpfs tmpfs "$path"
    fi
done
if [ -d {home_shell} ]; then
    mount -t tmpfs tmpfs {home_shell}
fi

mkdir -p {isolated_workspace_shell} {codex_home_target_shell} {nvm_target_shell}
mount --bind "$iso_root/ws" {isolated_workspace_shell}
mount --bind "$iso_root/codex_home" {codex_home_target_shell}
mount --bind "$iso_root/nvm" {nvm_target_shell}

export CODEX_HOME={codex_home_target_shell}
cd {isolated_workspace_shell}
{inner_cmd_shell}
"""
        workspace.joinpath("isolated_inner_command.json").write_text(
            json.dumps(inner_cmd, indent=2),
            encoding="utf-8",
        )
        workspace.joinpath("isolation.json").write_text(
            json.dumps(
                {
                    "mode": "unshare-mount-namespace",
                    "real_workspace": str(workspace),
                    "isolated_workspace": str(isolated_workspace),
                    "codex_home": str(codex_home),
                    "masked_roots": [str(path) for path in mask_roots] + [str(home)],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return ["unshare", "-Ur", "-m", "bash", "-lc", script]
