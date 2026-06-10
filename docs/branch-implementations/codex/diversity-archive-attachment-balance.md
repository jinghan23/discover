# codex/diversity-archive-attachment-balance

## Summary

根据候选的直接 parent 是否仍在当前 archive/candidate 集合中分为 anchored/detached，平衡选择这两类结构位置。

## Branch State

- Worktree: `/opt/tiger/discover-archive-attachment-balance`
- HEAD: `2234282`
- Base used for comparison: `2234282`
- Commits ahead of base: `0`
- Commits behind base: `0`
- Group: `completed_19`
- Implementation location: `uncommitted worktree diff`

## Implemented Behavior

- The summary above is based on reading the changed Python code, added config fields, and sampler/prompt hooks in this branch.
- For sampler branches, the common pattern is to keep baseline PUCT ordering inside a chosen surface/bucket, while changing which surface/bucket gets the next parent pick.
- Tests detected: `9` test functions across untracked test files.

## Added Markers

### Config fields

- `codex_archive_attachment_balance`
- `archive_attachment_balance`

### Constants

- `ARCHIVE_ATTACHMENT_SURFACES`
- `ARCHIVE_ATTACHMENT_SAMPLE_COUNTS_KEY`
- `_ARCHIVE_ATTACHMENT_SURFACE_ORDER`

### Classes

- None

### Functions

- `_zero_archive_attachment_counts`
- `_sanitize_archive_attachment_counts`
- `archive_attachment_surface`
- `_reset_archive_attachment_last`
- `_record_archive_attachment_pick`
- `_choose_archive_attachment_surface`
- `_sample_states_archive_attachment_balanced`
- `is_available`
- `add_pick`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 236 insertions(+), 6 deletions(-)`
- Untracked files: `3`

### Worktree Status

````text
 M ttt_discover/codex_utils/discovery.py
 M ttt_discover/codex_utils/sampler.py
 M ttt_discover/discovery.py
 M ttt_discover/rl/codex_no_finetune.py
?? repro/gpu_mode/
?? repro/run_discovery.py
?? tests/
````
### Tracked Worktree Files

````text
M	ttt_discover/codex_utils/discovery.py
M	ttt_discover/codex_utils/sampler.py
M	ttt_discover/discovery.py
M	ttt_discover/rl/codex_no_finetune.py
````
### Untracked Files

- `repro/gpu_mode/run_0609_archive_attachment_balance.sh (1064 bytes)`
- `repro/run_discovery.py (4327 bytes)`
- `tests/test_codex_archive_attachment_balance.py (13943 bytes)`

### Detected Test Functions

- `tests/test_codex_archive_attachment_balance.py::test_disabled_mode_matches_default_and_has_no_archive_outputs`
- `tests/test_codex_archive_attachment_balance.py::test_classifier_uses_only_parent_edge_and_candidate_membership`
- `tests/test_codex_archive_attachment_balance.py::test_least_sampled_surface_and_tie_order`
- `tests/test_codex_archive_attachment_balance.py::test_within_chosen_surface_follows_puct_order`
- `tests/test_codex_archive_attachment_balance.py::test_multi_parent_updates_counts_and_preserves_lineage_blocking`
- `tests/test_codex_archive_attachment_balance.py::test_fallback_increments_selected_actual_surface_count`
- `tests/test_codex_archive_attachment_balance.py::test_save_resume_restores_and_sanitizes_counts`
- `tests/test_codex_archive_attachment_balance.py::test_enabled_only_table_and_metrics`
- `tests/test_codex_archive_attachment_balance.py::test_config_cli_and_sampler_plumbing`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0609_archive_attachment_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

mode="run"
if [[ "${1:-}" == "run" || "${1:-}" == "dry-run" ]]; then
  mode="$1"
  shift
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-gpu-mode-0609-archive-attachment-balance}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
GROUP_SIZE="${GROUP_SIZE:-1}"
GROUPS_PER_BATCH="${GROUPS_PER_BATCH:-1}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-1200}"
CODEX_CLI_TIMEOUT="${CODEX_CLI_TIMEOUT:-600}"
CODEX_MAX_CONCURRENT_REQUESTS="${CODEX_MAX_CONCURRENT_REQUESTS:-1}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

python repro/run_discovery.py "${mode}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --num-epochs "${NUM_EPOCHS}" \
  --group-size "${GROUP_SIZE}" \
  --groups-per-batch "${GROUPS_PER_BATCH}" \
  --eval-timeout "${EVAL_TIMEOUT}" \
  --codex-cli-timeout "${CODEX_CLI_TIMEOUT}" \
  --codex-max-concurrent-requests "${CODEX_MAX_CONCURRENT_REQUESTS}" \
  --codex-archive-attachment-balance \
  "$@"
````

### `repro/run_discovery.py`

````python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _none_if_empty(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Codex discovery run.")
    parser.add_argument("command", nargs="?", choices=("run", "dry-run"), default="run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--experiment-name", default="gpu-mode-codex-discovery")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--runner", choices=("tinker_rl", "codex_no_finetune"), default="codex_no_finetune")
    parser.add_argument("--problem-type", default="trimul")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--num-cpus-per-task", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=45)
    parser.add_argument("--codex-model-name", default=None)
    parser.add_argument("--codex-backend", choices=("cli", "responses"), default="cli")
    parser.add_argument("--codex-max-output-tokens", type=int, default=8192)
    parser.add_argument("--codex-temperature", type=float, default=None)
    parser.add_argument("--codex-cli-timeout", type=float, default=None)
    parser.add_argument("--codex-max-concurrent-requests", type=int, default=4)
    parser.add_argument(
        "--codex-archive-attachment-balance",
        action="store_true",
        help="Balance Codex parent sampling between detached and anchored archive attachments.",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace, env_type: type):
    from ttt_discover.codex_utils.discovery import DiscoverConfig

    return DiscoverConfig(
        env_type=env_type,
        problem_type=args.problem_type,
        runner=args.runner,
        group_size=args.group_size,
        groups_per_batch=args.groups_per_batch,
        num_epochs=args.num_epochs,
        num_cpus_per_task=args.num_cpus_per_task,
        eval_timeout=args.eval_timeout,
        experiment_name=args.experiment_name,
        wandb_project=_none_if_empty(args.wandb_project),
        codex_model_name=_none_if_empty(args.codex_model_name),
        codex_backend=args.codex_backend,
        codex_max_output_tokens=args.codex_max_output_tokens,
        codex_temperature=args.codex_temperature,
        codex_cli_timeout=args.codex_cli_timeout,
        codex_max_concurrent_requests=args.codex_max_concurrent_requests,
        codex_archive_attachment_balance=args.codex_archive_attachment_balance,
    )


def dry_run_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "runner": args.runner,
        "problem_type": args.problem_type,
        "experiment_name": args.experiment_name,
        "wandb_project": _none_if_empty(args.wandb_project),
        "group_size": args.group_size,
        "groups_per_batch": args.groups_per_batch,
        "num_epochs": args.num_epochs,
        "num_cpus_per_task": args.num_cpus_per_task,
        "eval_timeout": args.eval_timeout,
        "codex_model_name": _none_if_empty(args.codex_model_name),
        "codex_backend": args.codex_backend,
        "codex_max_output_tokens": args.codex_max_output_tokens,
        "codex_temperature": args.codex_temperature,
        "codex_cli_timeout": args.codex_cli_timeout,
        "codex_max_concurrent_requests": args.codex_max_concurrent_requests,
        "codex_archive_attachment_balance": args.codex_archive_attachment_balance,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "dry-run" or args.dry_run:
        print(json.dumps(dry_run_payload(args), indent=2, sort_keys=True))
        return

    from examples.gpu_mode.env import GpuModeEnv
    from ttt_discover.codex_utils.discovery import discover

    discover(build_config(args, env_type=GpuModeEnv))


if __name__ == "__main__":
    main()
````

### `tests/test_codex_archive_attachment_balance.py`

````python
from __future__ import annotations

import json
from pathlib import Path

from ttt_discover.codex_utils import sampler as sampler_mod
from ttt_discover.codex_utils.discovery import DiscoverConfig
from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import (
    ARCHIVE_ATTACHMENT_SAMPLE_COUNTS_KEY,
    ARCHIVE_ATTACHMENT_SURFACES,
    PUCTSampler,
    archive_attachment_surface,
    create_sampler,
)
from ttt_discover.rl.codex_no_finetune import CodexNoFinetuneConfig, _build_sampler


BASE_COLUMNS = [
    "buffer_idx",
    "timestep",
    "value",
    "terminal_value",
    "parent_value",
    "construction_len",
    "observation_len",
    "n",
    "Q",
    "P",
    "bonus",
    "score",
]


class ToyEnv:
    state_type = State

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        return State(timestep=-1, construction=[], code="", value=0.0)


def make_state(
    state_id: str,
    value: float,
    *,
    parents: list | None = None,
    code: str | None = None,
    construction: list | None = None,
    observation: str = "",
) -> State:
    return State(
        id=state_id,
        timestep=0,
        value=value,
        parents=parents or [],
        parent_values=[],
        construction=[state_id] if construction is None else construction,
        code=f"code-{state_id}" if code is None else code,
        observation=observation,
    )


def clone_states(states: list[State]) -> list[State]:
    return [State.from_dict(state.to_dict()) for state in states]


def make_sampler(
    tmp_path: Path,
    states: list[State],
    *,
    enabled: bool = False,
) -> PUCTSampler:
    sampler = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=ToyEnv,
        batch_size=0,
        puct_c=0.0,
        topk_children=0,
        archive_attachment_balance=enabled,
    )
    sampler._states = clone_states(states)
    sampler._initial_states = []
    return sampler


def saved_step(path: Path, step: int = 1) -> dict:
    file_path = path / f"puct_sampler_step_{step:06d}.json"
    return json.loads(file_path.read_text(encoding="utf-8"))


def test_disabled_mode_matches_default_and_has_no_archive_outputs(tmp_path: Path) -> None:
    states = [
        make_state("root", 9.0),
        make_state("child", 10.0, parents=[{"id": "root", "timestep": 0}]),
        make_state("other", 8.0),
    ]

    for num_states, expected_ids in [(1, ["child"]), (2, ["child", "other"])]:
        default_sampler = make_sampler(tmp_path / f"default-{num_states}", states)
        disabled_sampler = make_sampler(
            tmp_path / f"disabled-{num_states}",
            states,
            enabled=False,
        )
        disabled_sampler._archive_attachment_sample_counts = {
            "detached": 7,
            "anchored": 11,
        }

        assert [s.id for s in default_sampler.sample_states(num_states)] == expected_ids
        assert [s.id for s in disabled_sampler.sample_states(num_states)] == expected_ids
        assert default_sampler._last_puct_stats == disabled_sampler._last_puct_stats
        assert disabled_sampler._archive_attachment_sample_counts == {
            "detached": 7,
            "anchored": 11,
        }

        default_columns, _ = default_sampler.get_sample_table()
        disabled_columns, _ = disabled_sampler.get_sample_table()
        assert default_columns == BASE_COLUMNS
        assert disabled_columns == default_columns
        assert not any(col.startswith("archive_attachment") for col in disabled_columns)
        assert not any(
            key.startswith("puct/archive_attachment")
            for key in disabled_sampler.get_sample_stats()
        )

        disabled_sampler.flush(step=1)
        assert ARCHIVE_ATTACHMENT_SAMPLE_COUNTS_KEY not in saved_step(
            tmp_path / f"disabled-{num_states}"
        )


def test_classifier_uses_only_parent_edge_and_candidate_membership() -> None:
    assert ARCHIVE_ATTACHMENT_SURFACES == ("detached", "anchored")
    candidate_ids = {"root", "live_child"}

    root = make_state("root", 1.0)
    live_child = make_state("live_child", 2.0, parents=[{"id": "root"}])
    pruned_child = make_state("pruned_child", 3.0, parents=[{"id": "missing"}])
    malformed = make_state("malformed", 4.0, parents=["not-a-dict"])
    missing_id = make_state("missing_id", 5.0, parents=[{"timestep": 0}])

    assert archive_attachment_surface(root, candidate_ids) == "anchored"
    assert archive_attachment_surface(live_child, candidate_ids) == "anchored"
    assert archive_attachment_surface(pruned_child, candidate_ids) == "detached"
    assert archive_attachment_surface(malformed, candidate_ids) == "anchored"
    assert archive_attachment_surface(missing_id, candidate_ids) == "anchored"

    noisy_a = make_state(
        "same",
        1.0,
        parents=[{"id": "missing"}],
        code="print('a')",
        construction=["a"],
        observation="stdout a",
    )
    noisy_b = make_state(
        "same",
        999.0,
        parents=[{"id": "missing"}],
        code="print('b')",
        construction=["b", "c"],
        observation="stdout b",
    )
    assert archive_attachment_surface(noisy_a, {"same"}) == "detached"
    assert archive_attachment_surface(noisy_b, {"same"}) == "detached"


def test_least_sampled_surface_and_tie_order(tmp_path: Path) -> None:
    states = [
        make_state("anchored", 100.0),
        make_state("detached", 1.0, parents=[{"id": "gone"}]),
    ]

    sampler = make_sampler(tmp_path / "tie", states, enabled=True)
    assert [state.id for state in sampler.sample_states(1)] == ["detached"]
    assert sampler._last_archive_attachment_rows == [
        ("detached", 0, 1, "balanced")
    ]

    sampler = make_sampler(tmp_path / "least", states, enabled=True)
    sampler._archive_attachment_sample_counts = {"detached": 5, "anchored": 0}
    assert [state.id for state in sampler.sample_states(1)] == ["anchored"]
    assert sampler._last_archive_attachment_rows == [
        ("anchored", 0, 1, "balanced")
    ]


def test_within_chosen_surface_follows_puct_order(tmp_path: Path) -> None:
    states = [
        make_state("anchored", 100.0),
        make_state("detached_low", 1.0, parents=[{"id": "gone"}]),
        make_state("detached_high", 10.0, parents=[{"id": "gone"}]),
    ]
    sampler = make_sampler(tmp_path, states, enabled=True)
    sampler._archive_attachment_sample_counts = {"detached": 0, "anchored": 9}

    assert [state.id for state in sampler.sample_states(1)] == ["detached_high"]


def test_multi_parent_updates_counts_and_preserves_lineage_blocking(tmp_path: Path) -> None:
    states = [
        make_state("detached", 100.0, parents=[{"id": "gone"}]),
        make_state("detached_child", 99.0, parents=[{"id": "detached"}]),
        make_state("anchored", 80.0),
        make_state("detached_2", 70.0, parents=[{"id": "gone"}]),
    ]
    sampler = make_sampler(tmp_path, states, enabled=True)

    assert [state.id for state in sampler.sample_states(3)] == [
        "detached",
        "anchored",
        "detached_2",
    ]
    assert sampler._last_archive_attachment_rows == [
        ("detached", 0, 1, "balanced"),
        ("anchored", 0, 1, "balanced"),
        ("detached", 1, 2, "balanced"),
    ]
    assert sampler._archive_attachment_sample_counts == {
        "detached": 2,
        "anchored": 1,
    }


def test_fallback_increments_selected_actual_surface_count(tmp_path: Path) -> None:
    states = [make_state("anchored", 10.0)]
    sampler = make_sampler(tmp_path, states, enabled=True)
    sampler._archive_attachment_sample_counts = {"detached": 0, "anchored": 3}

    assert [state.id for state in sampler.sample_states(1)] == ["anchored"]
    assert sampler._last_archive_attachment_rows == [
        ("anchored", 3, 4, "balanced")
    ]
    assert sampler._archive_attachment_sample_counts == {
        "detached": 0,
        "anchored": 4,
    }
    assert sampler.get_sample_stats()["puct/archive_attachment/fallbacks_last"] == 0

    fallback = make_sampler(tmp_path / "fallback", states, enabled=True)
    fallback._archive_attachment_sample_counts = {"detached": 0, "anchored": 3}
    fallback._choose_archive_attachment_surface = lambda surface_entries: None

    assert [state.id for state in fallback.sample_states(1)] == ["anchored"]
    assert fallback._last_archive_attachment_rows == [
        ("anchored", 3, 4, "fallback")
    ]
    assert fallback._archive_attachment_sample_counts == {
        "detached": 0,
        "anchored": 4,
    }
    assert fallback.get_sample_stats()["puct/archive_attachment/fallbacks_last"] == 1


def test_save_resume_restores_and_sanitizes_counts(tmp_path: Path) -> None:
    states = [
        make_state("detached", 10.0, parents=[{"id": "gone"}]),
        make_state("anchored", 1.0),
    ]
    sampler = make_sampler(tmp_path / "resume", states, enabled=True)
    sampler.sample_states(1)
    sampler.flush(step=1)

    resumed = PUCTSampler(
        file_path=str(tmp_path / "resume" / "puct_sampler.json"),
        env_type=ToyEnv,
        batch_size=0,
        resume_step=1,
        archive_attachment_balance=True,
    )
    assert resumed._archive_attachment_sample_counts == {
        "detached": 1,
        "anchored": 0,
    }

    stored = saved_step(tmp_path / "resume")
    stored.pop(ARCHIVE_ATTACHMENT_SAMPLE_COUNTS_KEY)
    (tmp_path / "resume" / "puct_sampler_step_000002.json").write_text(
        json.dumps(stored),
        encoding="utf-8",
    )
    missing = PUCTSampler(
        file_path=str(tmp_path / "resume" / "puct_sampler.json"),
        env_type=ToyEnv,
        batch_size=0,
        resume_step=2,
        archive_attachment_balance=True,
    )
    assert missing._archive_attachment_sample_counts == {
        "detached": 0,
        "anchored": 0,
    }

    stored[ARCHIVE_ATTACHMENT_SAMPLE_COUNTS_KEY] = {
        "detached": True,
        "anchored": -2,
        "unknown": 99,
    }
    (tmp_path / "resume" / "puct_sampler_step_000003.json").write_text(
        json.dumps(stored),
        encoding="utf-8",
    )
    bad = PUCTSampler(
        file_path=str(tmp_path / "resume" / "puct_sampler.json"),
        env_type=ToyEnv,
        batch_size=0,
        resume_step=3,
        archive_attachment_balance=True,
    )
    assert bad._archive_attachment_sample_counts == {
        "detached": 0,
        "anchored": 0,
    }


def test_enabled_only_table_and_metrics(tmp_path: Path) -> None:
    states = [
        make_state("anchored", 10.0),
        make_state("detached", 1.0, parents=[{"id": "gone"}]),
    ]
    disabled = make_sampler(tmp_path / "disabled", states, enabled=False)
    disabled.sample_states(1)
    disabled_columns, _ = disabled.get_sample_table()
    assert disabled_columns == BASE_COLUMNS
    assert not any(
        key.startswith("puct/archive_attachment")
        for key in disabled.get_sample_stats()
    )

    enabled = make_sampler(tmp_path / "enabled", states, enabled=True)
    enabled.sample_states(1)
    enabled_columns, enabled_rows = enabled.get_sample_table()
    assert enabled_columns == BASE_COLUMNS + [
        "archive_attachment_surface",
        "archive_attachment_count_before",
        "archive_attachment_count_after",
        "archive_attachment_selection_kind",
    ]
    assert enabled_rows[0][-4:] == ("detached", 0, 1, "balanced")
    stats = enabled.get_sample_stats()
    assert stats["puct/archive_attachment/enabled"] == 1.0
    assert stats["puct/archive_attachment/sample_count_detached"] == 1
    assert stats["puct/archive_attachment/sample_count_anchored"] == 0
    assert stats["puct/archive_attachment/last_sampled_detached"] == 1
    assert stats["puct/archive_attachment/available_detached"] == 1
    assert stats["puct/archive_attachment/available_anchored"] == 1


def test_config_cli_and_sampler_plumbing(tmp_path: Path) -> None:
    from repro.run_discovery import dry_run_payload, parse_args

    assert DiscoverConfig(codex_archive_attachment_balance=True).codex_archive_attachment_balance
    default_args = parse_args(["dry-run"])
    explicit_args = parse_args(["--dry-run", "--codex-archive-attachment-balance"])
    assert dry_run_payload(default_args)["codex_archive_attachment_balance"] is False
    explicit_payload = dry_run_payload(explicit_args)
    assert explicit_payload["codex_archive_attachment_balance"] is True
    assert explicit_payload["codex_model_name"] is None
    assert explicit_payload["num_cpus_per_task"] == explicit_args.num_cpus_per_task

    created = create_sampler(
        log_path=str(tmp_path / "factory"),
        env_type=ToyEnv,
        batch_size=0,
        archive_attachment_balance=True,
    )
    assert isinstance(created, PUCTSampler)
    assert created.archive_attachment_balance

    cfg = CodexNoFinetuneConfig(
        env_type=ToyEnv,
        groups_per_batch=0,
        log_path=str(tmp_path / "runner"),
        archive_attachment_balance=True,
    )
    built = _build_sampler(cfg, start_batch=0)
    assert isinstance(built, PUCTSampler)
    assert built.archive_attachment_balance

    repo = Path(__file__).resolve().parents[1]
    run_discovery = (repo / "repro/run_discovery.py").read_text(encoding="utf-8")
    codex_wrapper = (repo / "ttt_discover/codex_utils/discovery.py").read_text(
        encoding="utf-8"
    )
    heavy_wrapper = (repo / "ttt_discover/discovery.py").read_text(encoding="utf-8")

    assert "--codex-archive-attachment-balance" in run_discovery
    assert "codex_archive_attachment_balance" in run_discovery
    assert "codex_archive_attachment_balance: bool = False" in codex_wrapper
    assert "archive_attachment_balance=config.codex_archive_attachment_balance" in codex_wrapper
    assert "codex_archive_attachment_balance: bool = False" in heavy_wrapper
    assert "archive_attachment_balance=config.codex_archive_attachment_balance" in heavy_wrapper
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   2 +
 ttt_discover/codex_utils/sampler.py   | 236 +++++++++++++++++++++++++++++++++-
 ttt_discover/discovery.py             |   2 +
 ttt_discover/rl/codex_no_finetune.py  |   2 +
 4 files changed, 236 insertions(+), 6 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..3dc3cc3 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_archive_attachment_balance: bool = False
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -84,6 +85,7 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         max_concurrent_requests=config.codex_max_concurrent_requests,
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
+        archive_attachment_balance=config.codex_archive_attachment_balance,
         autonomous=config.codex_autonomous,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..da01457 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -16,6 +16,45 @@ from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_ser
 
 logger = logging.getLogger(__name__)
 
+ARCHIVE_ATTACHMENT_SURFACES = ("detached", "anchored")
+ARCHIVE_ATTACHMENT_SAMPLE_COUNTS_KEY = "archive_attachment_sample_counts"
+_ARCHIVE_ATTACHMENT_SURFACE_ORDER = {
+    surface: idx for idx, surface in enumerate(ARCHIVE_ATTACHMENT_SURFACES)
+}
+
+
+def _zero_archive_attachment_counts() -> dict[str, int]:
+    return {surface: 0 for surface in ARCHIVE_ATTACHMENT_SURFACES}
+
+
+def _sanitize_archive_attachment_counts(raw_counts: Any) -> dict[str, int]:
+    counts = _zero_archive_attachment_counts()
+    if not isinstance(raw_counts, dict):
+        return counts
+    for surface in ARCHIVE_ATTACHMENT_SURFACES:
+        value = raw_counts.get(surface, 0)
+        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
+            value = 0
+        counts[surface] = value
+    return counts
+
+
+def archive_attachment_surface(state: State, candidate_ids: set[str]) -> str:
+    parents = getattr(state, "parents", None)
+    if not isinstance(parents, list) or not parents:
+        return "anchored"
+    immediate_parent = parents[0]
+    if not isinstance(immediate_parent, dict):
+        return "anchored"
+    if "id" not in immediate_parent:
+        return "anchored"
+    parent_id = immediate_parent["id"]
+    if parent_id is None or parent_id == "":
+        return "anchored"
+    if str(parent_id) not in candidate_ids:
+        return "detached"
+    return "anchored"
+
 
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
@@ -353,6 +392,7 @@ class PUCTSampler(StateSampler):
         resume_step: int | None = None,
         puct_c: float = 1.0,
         topk_children: int = 2,
+        archive_attachment_balance: bool = False,
     ):
         self.file_path = file_path
         self.env_type = env_type
@@ -361,6 +401,7 @@ class PUCTSampler(StateSampler):
         self.batch_size = batch_size
         self.topk_children = topk_children
         self.puct_c = float(puct_c)
+        self.archive_attachment_balance = bool(archive_attachment_balance)
         
         self._states: list[State] = []
         self._initial_states: list[State] = []
@@ -375,6 +416,11 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._archive_attachment_sample_counts = _zero_archive_attachment_counts()
+        self._last_archive_attachment_rows: list[tuple[str, int, int, str]] = []
+        self._last_archive_attachment_sampled = _zero_archive_attachment_counts()
+        self._last_archive_attachment_available = _zero_archive_attachment_counts()
+        self._last_archive_attachment_fallbacks = 0
         
         if resume_step is not None:
             self._load(resume_step)
@@ -399,6 +445,10 @@ class PUCTSampler(StateSampler):
         self._n = store.get("puct_n", {}) or {}
         self._m = store.get("puct_m", {}) or {}
         self._T = int(store.get("puct_T", 0) or 0)
+        if self.archive_attachment_balance:
+            self._archive_attachment_sample_counts = _sanitize_archive_attachment_counts(
+                store.get(ARCHIVE_ATTACHMENT_SAMPLE_COUNTS_KEY, {})
+            )
 
     def _save(self, step: int):
         save_path = _sampler_file_for_step(self.file_path, step)
@@ -411,6 +461,10 @@ class PUCTSampler(StateSampler):
             "puct_m": self._m,
             "puct_T": self._T,
         }
+        if self.archive_attachment_balance:
+            store[ARCHIVE_ATTACHMENT_SAMPLE_COUNTS_KEY] = (
+                self._archive_attachment_sample_counts
+            )
         with _file_lock(f"{save_path}.lock"):
             _atomic_write_json(save_path, store)
 
@@ -463,6 +517,8 @@ class PUCTSampler(StateSampler):
     def _get_lineage(self, state: State) -> set[str]:
         lineage = {state.id}
         for p in (state.parents or []):
+            if not isinstance(p, dict):
+                continue
             if p.get("id"):
                 lineage.add(str(p["id"]))
         return lineage
@@ -471,6 +527,8 @@ class PUCTSampler(StateSampler):
         children: dict[str, set[str]] = {}
         for s in self._states:
             for p in (s.parents or []):
+                if not isinstance(p, dict):
+                    continue
                 pid = p.get("id")
                 if pid:
                     children.setdefault(str(pid), set()).add(s.id)
@@ -489,6 +547,108 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
+    def _reset_archive_attachment_last(self) -> None:
+        self._last_archive_attachment_rows = []
+        self._last_archive_attachment_sampled = _zero_archive_attachment_counts()
+        self._last_archive_attachment_available = _zero_archive_attachment_counts()
+        self._last_archive_attachment_fallbacks = 0
+
+    def _record_archive_attachment_pick(
+        self,
+        surface: str,
+        selection_kind: str,
+    ) -> tuple[str, int, int, str]:
+        before = self._archive_attachment_sample_counts.get(surface, 0)
+        after = before + 1
+        self._archive_attachment_sample_counts[surface] = after
+        self._last_archive_attachment_sampled[surface] = (
+            self._last_archive_attachment_sampled.get(surface, 0) + 1
+        )
+        row = (surface, before, after, selection_kind)
+        self._last_archive_attachment_rows.append(row)
+        return row
+
+    def _choose_archive_attachment_surface(
+        self,
+        surface_entries: dict[str, list[tuple]],
+    ) -> str | None:
+        available_surfaces = [
+            surface for surface in ARCHIVE_ATTACHMENT_SURFACES if surface_entries[surface]
+        ]
+        if not available_surfaces:
+            return None
+        return min(
+            available_surfaces,
+            key=lambda surface: (
+                self._archive_attachment_sample_counts.get(surface, 0),
+                _ARCHIVE_ATTACHMENT_SURFACE_ORDER[surface],
+            ),
+        )
+
+    def _sample_states_archive_attachment_balanced(
+        self,
+        scores: list[tuple[float, float, State, int, float, float, float]],
+        num_states: int,
+    ) -> tuple[list[State], list[tuple[float, float, State, int, float, float, float]]]:
+        self._reset_archive_attachment_last()
+        if num_states <= 0:
+            return [], []
+
+        candidate_ids = {str(entry[2].id) for entry in scores}
+        surface_by_id = {
+            str(entry[2].id): archive_attachment_surface(entry[2], candidate_ids)
+            for entry in scores
+        }
+        for surface in surface_by_id.values():
+            self._last_archive_attachment_available[surface] += 1
+
+        children_map = self._build_children_map() if num_states > 1 else {}
+        picked: list[State] = []
+        top_scores: list[tuple[float, float, State, int, float, float, float]] = []
+        picked_ids: set[str] = set()
+        blocked_ids: set[str] = set()
+
+        def is_available(entry) -> bool:
+            state_id = str(entry[2].id)
+            if state_id in picked_ids:
+                return False
+            return not (num_states > 1 and state_id in blocked_ids)
+
+        def add_pick(entry, selection_kind: str) -> None:
+            state = entry[2]
+            state_id = str(state.id)
+            surface = surface_by_id[state_id]
+            if selection_kind == "fallback":
+                self._last_archive_attachment_fallbacks += 1
+            self._record_archive_attachment_pick(surface, selection_kind)
+            picked.append(state)
+            top_scores.append(entry)
+            picked_ids.add(state_id)
+            if num_states > 1:
+                blocked_ids.update(self._get_full_lineage(state, children_map))
+
+        while len(picked) < num_states:
+            surface_entries = {surface: [] for surface in ARCHIVE_ATTACHMENT_SURFACES}
+            for entry in scores:
+                if not is_available(entry):
+                    continue
+                surface = surface_by_id[str(entry[2].id)]
+                surface_entries[surface].append(entry)
+            chosen_surface = self._choose_archive_attachment_surface(surface_entries)
+            if chosen_surface is None:
+                break
+            add_pick(surface_entries[chosen_surface][0], "balanced")
+
+        if len(picked) < num_states:
+            for entry in scores:
+                if not is_available(entry):
+                    continue
+                add_pick(entry, "fallback")
+                if len(picked) >= num_states:
+                    break
+
+        return picked, top_scores
+
     def sample_states(self, num_states: int) -> list[State]:
         initial_ids = {s.id for s in self._initial_states}
         candidates = list(self._states)
@@ -501,6 +661,8 @@ class PUCTSampler(StateSampler):
             self._last_sampled_states = picked
             self._last_sampled_indices = []
             self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+            if self.archive_attachment_balance:
+                self._reset_archive_attachment_last()
             return picked
 
         vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
@@ -521,7 +683,12 @@ class PUCTSampler(StateSampler):
 
         scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
 
-        if num_states > 1:
+        if self.archive_attachment_balance:
+            picked, top_scores = self._sample_states_archive_attachment_balanced(
+                scores,
+                num_states,
+            )
+        elif num_states > 1:
             children_map = self._build_children_map()
             picked, top_scores, blocked_ids = [], [], set()
             for entry in scores:
@@ -566,7 +733,11 @@ class PUCTSampler(StateSampler):
         for pid, y in parent_max.items():
             self._m[pid] = max(self._m.get(pid, y), y)
             parent = parent_obj[pid]
-            anc_ids = [pid] + [str(p["id"]) for p in (parent.parents or []) if p.get("id")]
+            anc_ids = [pid] + [
+                str(p["id"])
+                for p in (parent.parents or [])
+                if isinstance(p, dict) and p.get("id")
+            ]
             for aid in anc_ids:
                 self._n[aid] = self._n.get(aid, 0) + 1
             self._T += 1
@@ -628,7 +799,12 @@ class PUCTSampler(StateSampler):
                 by_parent: dict[str, list[State]] = {}
                 no_parent: list[State] = []
                 for s in self._states:
-                    pid = s.parents[0]["id"] if s.parents else None
+                    first_parent = s.parents[0] if s.parents else None
+                    pid = (
+                        first_parent.get("id")
+                        if isinstance(first_parent, dict)
+                        else None
+                    )
                     if pid:
                         by_parent.setdefault(pid, []).append(s)
                     else:
@@ -641,7 +817,11 @@ class PUCTSampler(StateSampler):
             self._finalize_and_save(step)
 
     def record_failed_rollout(self, parent: State):
-        anc_ids = [parent.id] + [str(p["id"]) for p in (parent.parents or []) if p.get("id")]
+        anc_ids = [parent.id] + [
+            str(p["id"])
+            for p in (parent.parents or [])
+            if isinstance(p, dict) and p.get("id")
+        ]
         for aid in anc_ids:
             self._n[aid] = self._n.get(aid, 0) + 1
         self._T += 1
@@ -731,21 +911,61 @@ class PUCTSampler(StateSampler):
         stats.update(_stats(sampled_values, "puct/sampled_value"))
         stats.update(_stats(sampled_timesteps, "puct/sampled_timestep"))
         stats.update(_stats(sampled_constr_lens, "puct/sampled_construction_len"))
+        if self.archive_attachment_balance:
+            stats.update(
+                {
+                    "puct/archive_attachment/enabled": 1.0,
+                    "puct/archive_attachment/sample_count_detached": (
+                        self._archive_attachment_sample_counts["detached"]
+                    ),
+                    "puct/archive_attachment/sample_count_anchored": (
+                        self._archive_attachment_sample_counts["anchored"]
+                    ),
+                    "puct/archive_attachment/last_sampled_detached": (
+                        self._last_archive_attachment_sampled["detached"]
+                    ),
+                    "puct/archive_attachment/last_sampled_anchored": (
+                        self._last_archive_attachment_sampled["anchored"]
+                    ),
+                    "puct/archive_attachment/available_detached": (
+                        self._last_archive_attachment_available["detached"]
+                    ),
+                    "puct/archive_attachment/available_anchored": (
+                        self._last_archive_attachment_available["anchored"]
+                    ),
+                    "puct/archive_attachment/fallbacks_last": (
+                        self._last_archive_attachment_fallbacks
+                    ),
+                }
+            )
         return stats
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
         columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        if self.archive_attachment_balance:
+            columns = columns + [
+                "archive_attachment_surface",
+                "archive_attachment_count_before",
+                "archive_attachment_count_after",
+                "archive_attachment_selection_kind",
+            ]
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        archive_rows = self._last_archive_attachment_rows
+        if len(archive_rows) != len(self._last_sampled_states):
+            archive_rows = [("", 0, 0, "")] * len(self._last_sampled_states)
+        for row_idx, (idx, state, (n, Q, P, bonus, score)) in enumerate(zip(indices, self._last_sampled_states, stats)):
             parent_val = state.parent_values[0] if state.parent_values else None
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            row = (idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score)
+            if self.archive_attachment_balance:
+                row = row + archive_rows[row_idx]
+            rows.append(row)
         return columns, rows
 
 
@@ -756,6 +976,7 @@ def create_sampler(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    archive_attachment_balance: bool = False,
 ) -> StateSampler:
     """Factory function to create samplers. Pass the env type (from config.env_type)."""
     if not log_path:
@@ -768,6 +989,7 @@ def create_sampler(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        archive_attachment_balance=archive_attachment_balance,
     )
 
 
@@ -778,6 +1000,7 @@ def get_or_create_sampler_with_default(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    archive_attachment_balance: bool = False,
 ) -> StateSampler:
     """Get sampler. Initial experience is created via env_type.create_initial_state."""
     return create_sampler(
@@ -787,4 +1010,5 @@ def get_or_create_sampler_with_default(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        archive_attachment_balance=archive_attachment_balance,
     )
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..ee9ee75 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_archive_attachment_balance: bool = False
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -145,6 +146,7 @@ async def discover_impl(config: DiscoverConfig):
             max_concurrent_requests=config.codex_max_concurrent_requests,
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
+            archive_attachment_balance=config.codex_archive_attachment_balance,
             autonomous=config.codex_autonomous,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..5c47307 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    archive_attachment_balance: bool = False
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -960,6 +961,7 @@ def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler
         batch_size=cfg.groups_per_batch,
         resume_step=start_batch if start_batch > 0 else None,
         topk_children=cfg.topk_children,
+        archive_attachment_balance=cfg.archive_attachment_balance,
     )
 
 
````
</details>

