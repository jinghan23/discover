# codex/diversity-parent-family-width-balance

## Summary

统计直接 parent 在 retained archive 中的 sibling/child 数量，将候选分为 only_child/paired_child/wide_family/root_or_seed 后平衡。

## Branch State

- Worktree: `/opt/tiger/discover-parent-family-width-balance`
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

- `codex_parent_family_width_balance`
- `parent_family_width_balance`

### Constants

- `PARENT_FAMILY_WIDTH_SURFACES`
- `_PUCT_PARENT_FAMILY_WIDTH_SAMPLE_COUNTS_KEY`

### Classes

- None

### Functions

- `_zero_parent_family_width_counts`
- `_sanitize_parent_family_width_counts`
- `_immediate_parent_id`
- `_parent_family_widths`
- `_classify_parent_family_width`
- `_parent_family_width_available_counts`
- `_record_parent_family_width_pick`
- `_select_parent_family_width_entry`
- `_sample_parent_family_width_balanced`
- `available_entries`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 238 insertions(+), 3 deletions(-)`
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

- `repro/gpu_mode/run_0609_parent_family_width_balance.sh (1061 bytes)`
- `repro/run_discovery.py (4041 bytes)`
- `tests/test_codex_parent_family_width_balance.py (14108 bytes)`

### Detected Test Functions

- `tests/test_codex_parent_family_width_balance.py::test_disabled_mode_matches_baseline_shape_stats_and_persistence`
- `tests/test_codex_parent_family_width_balance.py::test_classifier_uses_only_retained_direct_parent_links`
- `tests/test_codex_parent_family_width_balance.py::test_least_sampled_surface_beats_higher_puct_and_keeps_surface_order`
- `tests/test_codex_parent_family_width_balance.py::test_equal_counts_use_fixed_surface_order_then_baseline_order`
- `tests/test_codex_parent_family_width_balance.py::test_multi_parent_sampling_updates_counts_and_preserves_lineage_blocking`
- `tests/test_codex_parent_family_width_balance.py::test_fallback_respects_blocking_and_increments_actual_surface`
- `tests/test_codex_parent_family_width_balance.py::test_save_resume_counts_and_sanitize_missing_bad_values`
- `tests/test_codex_parent_family_width_balance.py::test_enabled_only_table_metrics_and_saved_key`
- `tests/test_codex_parent_family_width_balance.py::test_cli_and_config_plumbing_defaults_and_flag_forwarding`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0609_parent_family_width_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="run"
if [[ "${1:-}" == "run" || "${1:-}" == "dry-run" ]]; then
  MODE="$1"
  shift
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-gpu-mode-0609-parent-family-width-balance}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
GROUP_SIZE="${GROUP_SIZE:-1}"
GROUPS_PER_BATCH="${GROUPS_PER_BATCH:-1}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-1200}"
CODEX_CLI_TIMEOUT="${CODEX_CLI_TIMEOUT:-600}"
CODEX_MAX_CONCURRENT_REQUESTS="${CODEX_MAX_CONCURRENT_REQUESTS:-1}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

exec python repro/run_discovery.py "${MODE}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --num-epochs "${NUM_EPOCHS}" \
  --group-size "${GROUP_SIZE}" \
  --groups-per-batch "${GROUPS_PER_BATCH}" \
  --eval-timeout "${EVAL_TIMEOUT}" \
  --codex-cli-timeout "${CODEX_CLI_TIMEOUT}" \
  --codex-max-concurrent-requests "${CODEX_MAX_CONCURRENT_REQUESTS}" \
  --codex-parent-family-width-balance \
  "$@"
````

### `repro/run_discovery.py`

````python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Codex no-finetune discovery.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("run", "dry-run"),
        default="run",
        help="Use 'dry-run' to print the resolved config without launching discovery.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--experiment-name", default="gpu-mode-parent-family-width-balance")
    parser.add_argument("--problem-type", default="trimul")
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--num-cpus-per-task", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=1200)
    parser.add_argument("--codex-backend", choices=("cli", "responses"), default="cli")
    parser.add_argument("--codex-model-name", default=None)
    parser.add_argument("--codex-cli-command", default="codex")
    parser.add_argument("--codex-cli-sandbox", default="read-only")
    parser.add_argument("--codex-cli-timeout", type=float, default=600.0)
    parser.add_argument("--codex-max-concurrent-requests", type=int, default=1)
    parser.add_argument(
        "--codex-parent-family-width-balance",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def config_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "runner": "codex_no_finetune",
        "env_type": "examples.gpu_mode.env.GpuModeEnv",
        "problem_type": args.problem_type,
        "experiment_name": args.experiment_name,
        "wandb_project": None,
        "num_epochs": args.num_epochs,
        "groups_per_batch": args.groups_per_batch,
        "group_size": args.group_size,
        "num_cpus_per_task": args.num_cpus_per_task,
        "eval_timeout": args.eval_timeout,
        "codex_backend": args.codex_backend,
        "codex_model_name": args.codex_model_name,
        "codex_cli_command": args.codex_cli_command,
        "codex_cli_sandbox": args.codex_cli_sandbox,
        "codex_cli_timeout": args.codex_cli_timeout,
        "codex_max_concurrent_requests": args.codex_max_concurrent_requests,
        "codex_autonomous": False,
        "codex_parent_family_width_balance": args.codex_parent_family_width_balance,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    resolved = config_dict(args)

    if args.dry_run or args.command == "dry-run":
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return 0

    from examples.gpu_mode.env import GpuModeEnv
    from ttt_discover.codex_utils.discovery import DiscoverConfig, discover

    config = DiscoverConfig(
        runner="codex_no_finetune",
        env_type=GpuModeEnv,
        problem_type=args.problem_type,
        experiment_name=args.experiment_name,
        wandb_project=None,
        num_epochs=args.num_epochs,
        groups_per_batch=args.groups_per_batch,
        group_size=args.group_size,
        num_cpus_per_task=args.num_cpus_per_task,
        eval_timeout=args.eval_timeout,
        codex_backend=args.codex_backend,
        codex_model_name=args.codex_model_name,
        codex_cli_command=args.codex_cli_command,
        codex_cli_sandbox=args.codex_cli_sandbox,
        codex_cli_timeout=args.codex_cli_timeout,
        codex_max_concurrent_requests=args.codex_max_concurrent_requests,
        codex_autonomous=False,
        codex_parent_family_width_balance=args.codex_parent_family_width_balance,
    )
    discover(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
````

### `tests/test_codex_parent_family_width_balance.py`

````python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import (
    PARENT_FAMILY_WIDTH_SURFACES,
    PUCTSampler,
    create_sampler,
)
from ttt_discover.rl import codex_no_finetune
from ttt_discover.rl.codex_no_finetune import CodexNoFinetuneConfig


COUNT_KEY = "puct_parent_family_width_sample_counts"


class DummyEnv:
    state_type = State
    refreshed: list[str] = []
    initial_idx = 0

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        cls.initial_idx += 1
        return State(
            timestep=0,
            construction=[],
            code="",
            value=0.0,
            id=f"init-{cls.initial_idx}",
        )

    @classmethod
    def refresh_initial_state(cls, state: State, problem_type: str) -> None:
        cls.refreshed.append(state.id)


def state(
    state_id: str,
    value: float,
    *,
    parent_id: str | None = None,
    parents: list | None = None,
    construction: list | None = None,
    code: str | None = None,
    observation: str | None = None,
) -> State:
    if parents is None:
        parents = [] if parent_id is None else [{"id": parent_id, "timestep": 0}]
    return State(
        timestep=1,
        construction=construction if construction is not None else [state_id],
        code=code if code is not None else f"code-{state_id}",
        value=value,
        parent_values=[],
        parents=parents,
        id=state_id,
        observation=observation if observation is not None else f"obs-{state_id}",
    )


def sampler_with_states(
    tmp_path: Path,
    states: list[State],
    *,
    enabled: bool = False,
    initial_states: list[State] | None = None,
) -> PUCTSampler:
    sampler = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        puct_c=0.0,
        topk_children=0,
        parent_family_width_balance=enabled,
    )
    sampler._states = list(states)
    sampler._initial_states = list(initial_states or [])
    return sampler


def step_path(tmp_path: Path, step: int) -> Path:
    return tmp_path / f"puct_sampler_step_{step:06d}.json"


@pytest.mark.parametrize(
    ("num_states", "root_value", "expected_ids", "expected_refresh"),
    [
        (1, 13.0, ["root"], ["root"]),
        (2, 11.0, ["child", "other"], []),
    ],
)
def test_disabled_mode_matches_baseline_shape_stats_and_persistence(
    tmp_path: Path,
    num_states: int,
    root_value: float,
    expected_ids: list[str],
    expected_refresh: list[str],
) -> None:
    DummyEnv.refreshed = []
    root = state("root", root_value)
    child = state("child", 12.0, parent_id="root")
    other = state("other", 10.0)
    sampler = sampler_with_states(
        tmp_path,
        [root, child, other],
        enabled=False,
        initial_states=[root],
    )

    picked = sampler.sample_states(num_states)

    assert [s.id for s in picked] == expected_ids
    assert DummyEnv.refreshed == expected_refresh
    assert [row[0] for row in sampler._last_puct_stats] == [0] * len(expected_ids)
    assert [row[1] for row in sampler._last_puct_stats] == [
        pytest.approx(next(s.value for s in [root, child, other] if s.id == state_id))
        for state_id in expected_ids
    ]
    assert [row[3] for row in sampler._last_puct_stats] == [0.0] * len(expected_ids)
    assert [row[4] for row in sampler._last_puct_stats] == [
        pytest.approx(next(s.value for s in [root, child, other] if s.id == state_id))
        for state_id in expected_ids
    ]

    columns, rows = sampler.get_sample_table()
    assert rows
    assert not any("parent_family_width" in col for col in columns)
    metrics = sampler.get_sample_stats()
    assert not any("parent_family_width" in key for key in metrics)

    sampler.flush(step=1)
    saved = json.loads(step_path(tmp_path, 1).read_text(encoding="utf-8"))
    assert COUNT_KEY not in saved


def test_classifier_uses_only_retained_direct_parent_links(tmp_path: Path) -> None:
    root = state("root", 1.0)
    malformed = state("malformed", 2.0, parents=["not-a-dict"])
    empty_parent = state("empty-parent", 2.5, parents=[{"id": ""}])
    only = state("only", 3.0, parent_id="parent-a")
    paired_1 = state("paired-1", 4.0, parent_id="parent-b")
    paired_2 = state("paired-2", 5.0, parent_id="parent-b")
    wide_1 = state("wide-1", 6.0, parent_id="parent-c")
    wide_2 = state("wide-2", 7.0, parent_id="parent-c")
    wide_3 = state("wide-3", 8.0, parent_id="parent-c")
    sampler = sampler_with_states(
        tmp_path,
        [root, malformed, empty_parent, only, paired_1, paired_2, wide_1, wide_2, wide_3],
        enabled=True,
    )

    widths = sampler._parent_family_widths()

    assert sampler._classify_parent_family_width(root, widths) == ("root_or_seed", 0)
    assert sampler._classify_parent_family_width(malformed, widths) == ("root_or_seed", 0)
    assert sampler._classify_parent_family_width(empty_parent, widths) == ("root_or_seed", 0)
    assert sampler._classify_parent_family_width(only, widths) == ("only_child", 1)
    assert sampler._classify_parent_family_width(paired_1, widths) == ("paired_child", 2)
    assert sampler._classify_parent_family_width(wide_1, widths) == ("wide_family", 3)

    altered = state(
        "only",
        9999.0,
        parent_id="parent-a",
        construction=["forbidden construction change"],
        code="forbidden code change",
        observation="forbidden observation change",
    )
    assert sampler._classify_parent_family_width(altered, widths) == ("only_child", 1)


def test_least_sampled_surface_beats_higher_puct_and_keeps_surface_order(
    tmp_path: Path,
) -> None:
    only_1 = state("only-1", 5.0, parent_id="only-parent-1")
    only_2 = state("only-2", 3.0, parent_id="only-parent-2")
    wide_1 = state("wide-1", 100.0, parent_id="wide-parent")
    wide_2 = state("wide-2", 90.0, parent_id="wide-parent")
    wide_3 = state("wide-3", 80.0, parent_id="wide-parent")
    sampler = sampler_with_states(
        tmp_path,
        [only_1, only_2, wide_1, wide_2, wide_3],
        enabled=True,
    )
    sampler._parent_family_width_sample_counts["wide_family"] = 7

    picked = sampler.sample_states(1)

    assert [s.id for s in picked] == ["only-1"]
    assert sampler._last_parent_family_width_stats == [("only_child", 1, 1, False)]


def test_equal_counts_use_fixed_surface_order_then_baseline_order(tmp_path: Path) -> None:
    root = state("root", 100.0)
    paired_1 = state("paired-1", 90.0, parent_id="paired-parent")
    paired_2 = state("paired-2", 80.0, parent_id="paired-parent")
    wide_1 = state("wide-1", 70.0, parent_id="wide-parent")
    wide_2 = state("wide-2", 60.0, parent_id="wide-parent")
    wide_3 = state("wide-3", 50.0, parent_id="wide-parent")
    only_1 = state("only-1", 5.0, parent_id="only-parent-1")
    only_2 = state("only-2", 4.0, parent_id="only-parent-2")
    sampler = sampler_with_states(
        tmp_path,
        [root, paired_1, paired_2, wide_1, wide_2, wide_3, only_1, only_2],
        enabled=True,
    )

    picked = sampler.sample_states(1)

    assert [s.id for s in picked] == ["only-1"]
    assert sampler._last_parent_family_width_stats == [("only_child", 1, 1, False)]


def test_multi_parent_sampling_updates_counts_and_preserves_lineage_blocking(
    tmp_path: Path,
) -> None:
    root = state("root", 95.0)
    child = state("child", 100.0, parent_id="root")
    only_2 = state("only-2", 99.0, parent_id="only-parent-2")
    paired_1 = state("paired-1", 90.0, parent_id="paired-parent")
    paired_2 = state("paired-2", 80.0, parent_id="paired-parent")
    wide_1 = state("wide-1", 70.0, parent_id="wide-parent")
    wide_2 = state("wide-2", 60.0, parent_id="wide-parent")
    wide_3 = state("wide-3", 50.0, parent_id="wide-parent")
    sampler = sampler_with_states(
        tmp_path,
        [root, child, only_2, paired_1, paired_2, wide_1, wide_2, wide_3],
        enabled=True,
    )

    picked = sampler.sample_states(3)

    assert [s.id for s in picked] == ["child", "paired-1", "wide-1"]
    assert "root" not in [s.id for s in picked]
    assert sampler._last_parent_family_width_stats == [
        ("only_child", 1, 1, False),
        ("paired_child", 2, 1, False),
        ("wide_family", 3, 1, False),
    ]


def test_fallback_respects_blocking_and_increments_actual_surface(tmp_path: Path) -> None:
    root = state("root", 90.0)
    child = state("child", 100.0, parent_id="root")
    other = state("other", 80.0)
    sampler = sampler_with_states(tmp_path, [root, child, other], enabled=True)
    sampler._select_parent_family_width_entry = lambda available, widths: None

    picked = sampler.sample_states(2)

    assert [s.id for s in picked] == ["child", "other"]
    assert sampler._last_parent_family_width_fallbacks == 2
    assert sampler._last_parent_family_width_stats == [
        ("only_child", 1, 1, True),
        ("root_or_seed", 0, 1, True),
    ]


def test_save_resume_counts_and_sanitize_missing_bad_values(tmp_path: Path) -> None:
    only = state("only", 5.0, parent_id="only-parent")
    sampler = sampler_with_states(tmp_path, [only], enabled=True)
    sampler.sample_states(1)
    sampler.flush(step=1)

    resumed = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=1,
        puct_c=0.0,
        topk_children=0,
        parent_family_width_balance=True,
    )
    assert resumed._parent_family_width_sample_counts == {
        "only_child": 1,
        "paired_child": 0,
        "wide_family": 0,
        "root_or_seed": 0,
    }

    saved = json.loads(step_path(tmp_path, 1).read_text(encoding="utf-8"))
    missing = dict(saved)
    missing.pop(COUNT_KEY)
    step_path(tmp_path, 2).write_text(json.dumps(missing), encoding="utf-8")
    missing_resumed = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=2,
        parent_family_width_balance=True,
    )
    assert missing_resumed._parent_family_width_sample_counts == {
        surface: 0 for surface in PARENT_FAMILY_WIDTH_SURFACES
    }

    bad = dict(saved)
    bad[COUNT_KEY] = {
        "only_child": True,
        "paired_child": -1,
        "wide_family": 4,
        "root_or_seed": "2",
        "unknown": 99,
    }
    step_path(tmp_path, 3).write_text(json.dumps(bad), encoding="utf-8")
    bad_resumed = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=3,
        parent_family_width_balance=True,
    )
    assert bad_resumed._parent_family_width_sample_counts == {
        "only_child": 0,
        "paired_child": 0,
        "wide_family": 4,
        "root_or_seed": 0,
    }


def test_enabled_only_table_metrics_and_saved_key(tmp_path: Path) -> None:
    only = state("only", 5.0, parent_id="only-parent")
    sampler = sampler_with_states(tmp_path, [only], enabled=True)

    sampler.sample_states(1)

    columns, rows = sampler.get_sample_table()
    assert rows
    assert columns[-4:] == [
        "parent_family_width_surface",
        "parent_family_width",
        "parent_family_width_count_after",
        "parent_family_width_fallback",
    ]
    metrics = sampler.get_sample_stats()
    assert metrics["puct/parent_family_width_balance_enabled"] == 1
    for surface in PARENT_FAMILY_WIDTH_SURFACES:
        assert f"puct/parent_family_width_sample_count/{surface}" in metrics
        assert f"puct/parent_family_width_available/{surface}" in metrics
    assert metrics["puct/parent_family_width_fallbacks"] == 0

    sampler.flush(step=1)
    saved = json.loads(step_path(tmp_path, 1).read_text(encoding="utf-8"))
    assert saved[COUNT_KEY]["only_child"] == 1


def test_cli_and_config_plumbing_defaults_and_flag_forwarding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_sampler = create_sampler(
        log_path=str(tmp_path / "default"),
        env_type=DummyEnv,
        batch_size=0,
    )
    assert isinstance(default_sampler, PUCTSampler)
    assert default_sampler.parent_family_width_balance is False
    assert CodexNoFinetuneConfig(env_type=DummyEnv).parent_family_width_balance is False

    captured: dict = {}

    def fake_create_sampler(**kwargs):
        captured.update(kwargs)
        return "sampler"

    monkeypatch.setattr(codex_no_finetune, "create_sampler", fake_create_sampler)
    cfg = CodexNoFinetuneConfig(
        env_type=DummyEnv,
        log_path=str(tmp_path),
        parent_family_width_balance=True,
    )
    assert codex_no_finetune._build_sampler(cfg, start_batch=0) == "sampler"
    assert captured["parent_family_width_balance"] is True

    repo = Path(__file__).resolve().parents[1]
    codex_discovery = (repo / "ttt_discover/codex_utils/discovery.py").read_text(
        encoding="utf-8"
    )
    tinker_discovery = (repo / "ttt_discover/discovery.py").read_text(encoding="utf-8")
    run_discovery = (repo / "repro/run_discovery.py").read_text(encoding="utf-8")
    for source in (codex_discovery, tinker_discovery):
        assert "codex_parent_family_width_balance: bool = False" in source
        assert "parent_family_width_balance=config.codex_parent_family_width_balance" in source
    assert "argparse.BooleanOptionalAction" in run_discovery
    assert "--codex-parent-family-width-balance" in run_discovery
    assert '"codex_parent_family_width_balance"' in run_discovery

    from repro.run_discovery import build_parser, config_dict

    default_args = build_parser().parse_args(["dry-run"])
    explicit_args = build_parser().parse_args(["--dry-run", "--codex-parent-family-width-balance"])
    assert config_dict(default_args)["codex_model_name"] is None
    assert config_dict(default_args)["codex_parent_family_width_balance"] is False
    assert config_dict(explicit_args)["codex_parent_family_width_balance"] is True
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   2 +
 ttt_discover/codex_utils/sampler.py   | 235 +++++++++++++++++++++++++++++++++-
 ttt_discover/discovery.py             |   2 +
 ttt_discover/rl/codex_no_finetune.py  |   2 +
 4 files changed, 238 insertions(+), 3 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..9ab3d2c 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_parent_family_width_balance: bool = False
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -85,6 +86,7 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
         autonomous=config.codex_autonomous,
+        parent_family_width_balance=config.codex_parent_family_width_balance,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
         log_path=log_path,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..4f96b50 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -16,6 +16,31 @@ from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_ser
 
 logger = logging.getLogger(__name__)
 
+PARENT_FAMILY_WIDTH_SURFACES = (
+    "only_child",
+    "paired_child",
+    "wide_family",
+    "root_or_seed",
+)
+_PUCT_PARENT_FAMILY_WIDTH_SAMPLE_COUNTS_KEY = "puct_parent_family_width_sample_counts"
+
+
+def _zero_parent_family_width_counts() -> dict[str, int]:
+    return {surface: 0 for surface in PARENT_FAMILY_WIDTH_SURFACES}
+
+
+def _sanitize_parent_family_width_counts(raw_counts: Any) -> dict[str, int]:
+    counts = _zero_parent_family_width_counts()
+    if not isinstance(raw_counts, dict):
+        return counts
+    for surface, value in raw_counts.items():
+        if surface not in counts:
+            continue
+        if type(value) is not int or value < 0:
+            continue
+        counts[surface] = value
+    return counts
+
 
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
@@ -353,6 +378,7 @@ class PUCTSampler(StateSampler):
         resume_step: int | None = None,
         puct_c: float = 1.0,
         topk_children: int = 2,
+        parent_family_width_balance: bool = False,
     ):
         self.file_path = file_path
         self.env_type = env_type
@@ -361,6 +387,7 @@ class PUCTSampler(StateSampler):
         self.batch_size = batch_size
         self.topk_children = topk_children
         self.puct_c = float(puct_c)
+        self.parent_family_width_balance = bool(parent_family_width_balance)
         
         self._states: list[State] = []
         self._initial_states: list[State] = []
@@ -375,6 +402,10 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._parent_family_width_sample_counts: dict[str, int] = _zero_parent_family_width_counts()
+        self._last_parent_family_width_stats: list[tuple[str, int, int, bool]] = []
+        self._last_parent_family_width_available: dict[str, int] = _zero_parent_family_width_counts()
+        self._last_parent_family_width_fallbacks: int = 0
         
         if resume_step is not None:
             self._load(resume_step)
@@ -399,6 +430,10 @@ class PUCTSampler(StateSampler):
         self._n = store.get("puct_n", {}) or {}
         self._m = store.get("puct_m", {}) or {}
         self._T = int(store.get("puct_T", 0) or 0)
+        if self.parent_family_width_balance:
+            self._parent_family_width_sample_counts = _sanitize_parent_family_width_counts(
+                store.get(_PUCT_PARENT_FAMILY_WIDTH_SAMPLE_COUNTS_KEY)
+            )
 
     def _save(self, step: int):
         save_path = _sampler_file_for_step(self.file_path, step)
@@ -411,6 +446,8 @@ class PUCTSampler(StateSampler):
             "puct_m": self._m,
             "puct_T": self._T,
         }
+        if self.parent_family_width_balance:
+            store[_PUCT_PARENT_FAMILY_WIDTH_SAMPLE_COUNTS_KEY] = self._parent_family_width_sample_counts
         with _file_lock(f"{save_path}.lock"):
             _atomic_write_json(save_path, store)
 
@@ -489,6 +526,156 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
+    def _immediate_parent_id(self, state: State) -> str | None:
+        parents = getattr(state, "parents", None) or []
+        if not parents:
+            return None
+        parent = parents[0]
+        if not isinstance(parent, dict):
+            return None
+        parent_id = parent.get("id")
+        if parent_id is None or parent_id == "":
+            return None
+        return str(parent_id)
+
+    def _parent_family_widths(self) -> dict[str, int]:
+        widths: dict[str, int] = {}
+        for state in self._states:
+            parent_id = self._immediate_parent_id(state)
+            if parent_id is None:
+                continue
+            widths[parent_id] = widths.get(parent_id, 0) + 1
+        return widths
+
+    def _classify_parent_family_width(
+        self,
+        state: State,
+        family_widths: dict[str, int] | None = None,
+    ) -> tuple[str, int]:
+        parent_id = self._immediate_parent_id(state)
+        if parent_id is None:
+            return "root_or_seed", 0
+        widths = family_widths if family_widths is not None else self._parent_family_widths()
+        width = widths.get(parent_id, 0)
+        if width <= 1:
+            return "only_child", width
+        if width == 2:
+            return "paired_child", width
+        return "wide_family", width
+
+    def _parent_family_width_available_counts(
+        self,
+        entries: list[tuple[float, float, State, int, float, float, float]],
+        family_widths: dict[str, int],
+    ) -> dict[str, int]:
+        counts = _zero_parent_family_width_counts()
+        for entry in entries:
+            surface, _width = self._classify_parent_family_width(entry[2], family_widths)
+            counts[surface] += 1
+        return counts
+
+    def _record_parent_family_width_pick(
+        self,
+        surface: str,
+        width: int,
+        *,
+        fallback: bool,
+    ) -> tuple[str, int, int, bool]:
+        self._parent_family_width_sample_counts[surface] = (
+            self._parent_family_width_sample_counts.get(surface, 0) + 1
+        )
+        count_after = self._parent_family_width_sample_counts[surface]
+        return surface, width, count_after, fallback
+
+    def _select_parent_family_width_entry(
+        self,
+        available: list[tuple[float, float, State, int, float, float, float]],
+        family_widths: dict[str, int],
+    ) -> tuple[tuple[float, float, State, int, float, float, float], str, int] | None:
+        first_by_surface: dict[str, tuple[tuple[float, float, State, int, float, float, float], int]] = {}
+        for entry in available:
+            surface, width = self._classify_parent_family_width(entry[2], family_widths)
+            if surface not in first_by_surface:
+                first_by_surface[surface] = (entry, width)
+
+        best_surface = None
+        best_key = None
+        for surface_idx, surface in enumerate(PARENT_FAMILY_WIDTH_SURFACES):
+            if surface not in first_by_surface:
+                continue
+            key = (self._parent_family_width_sample_counts.get(surface, 0), surface_idx)
+            if best_key is None or key < best_key:
+                best_key = key
+                best_surface = surface
+        if best_surface is None:
+            return None
+        entry, width = first_by_surface[best_surface]
+        return entry, best_surface, width
+
+    def _sample_parent_family_width_balanced(
+        self,
+        scores: list[tuple[float, float, State, int, float, float, float]],
+        num_states: int,
+    ) -> tuple[list[State], list[tuple[float, float, State, int, float, float, float]]]:
+        family_widths = self._parent_family_widths()
+        children_map = self._build_children_map() if num_states > 1 else {}
+        picked: list[State] = []
+        top_scores: list[tuple[float, float, State, int, float, float, float]] = []
+        picked_ids: set[str] = set()
+        blocked_ids: set[str] = set()
+        self._last_parent_family_width_stats = []
+        self._last_parent_family_width_fallbacks = 0
+
+        def available_entries() -> list[tuple[float, float, State, int, float, float, float]]:
+            return [
+                entry
+                for entry in scores
+                if entry[2].id not in picked_ids
+                and (num_states <= 1 or entry[2].id not in blocked_ids)
+            ]
+
+        self._last_parent_family_width_available = self._parent_family_width_available_counts(
+            available_entries(),
+            family_widths,
+        )
+
+        while len(picked) < num_states:
+            selected = self._select_parent_family_width_entry(available_entries(), family_widths)
+            if selected is None:
+                break
+            entry, surface, width = selected
+            state = entry[2]
+            picked.append(state)
+            top_scores.append(entry)
+            picked_ids.add(state.id)
+            self._last_parent_family_width_stats.append(
+                self._record_parent_family_width_pick(surface, width, fallback=False)
+            )
+            if num_states > 1:
+                blocked_ids.update(self._get_full_lineage(state, children_map))
+
+        if len(picked) < num_states:
+            for entry in scores:
+                state = entry[2]
+                if state.id in picked_ids:
+                    continue
+                if num_states > 1 and state.id in blocked_ids:
+                    continue
+                surface, width = self._classify_parent_family_width(state, family_widths)
+                picked.append(state)
+                top_scores.append(entry)
+                picked_ids.add(state.id)
+                self._last_parent_family_width_fallbacks += 1
+                self._last_parent_family_width_stats.append(
+                    self._record_parent_family_width_pick(surface, width, fallback=True)
+                )
+                if num_states > 1:
+                    blocked_ids.update(self._get_full_lineage(state, children_map))
+                if len(picked) >= num_states:
+                    break
+
+        return picked, top_scores
+
     def sample_states(self, num_states: int) -> list[State]:
         initial_ids = {s.id for s in self._initial_states}
         candidates = list(self._states)
@@ -501,6 +688,17 @@ class PUCTSampler(StateSampler):
             self._last_sampled_states = picked
             self._last_sampled_indices = []
             self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+            if self.parent_family_width_balance:
+                self._last_parent_family_width_available = _zero_parent_family_width_counts()
+                self._last_parent_family_width_fallbacks = 0
+                self._last_parent_family_width_stats = [
+                    self._record_parent_family_width_pick(
+                        "root_or_seed",
+                        0,
+                        fallback=False,
+                    )
+                    for _ in picked
+                ]
             return picked
 
         vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
@@ -521,7 +719,9 @@ class PUCTSampler(StateSampler):
 
         scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
 
-        if num_states > 1:
+        if self.parent_family_width_balance:
+            picked, top_scores = self._sample_parent_family_width_balanced(scores, num_states)
+        elif num_states > 1:
             children_map = self._build_children_map()
             picked, top_scores, blocked_ids = [], [], set()
             for entry in scores:
@@ -731,21 +931,46 @@ class PUCTSampler(StateSampler):
         stats.update(_stats(sampled_values, "puct/sampled_value"))
         stats.update(_stats(sampled_timesteps, "puct/sampled_timestep"))
         stats.update(_stats(sampled_constr_lens, "puct/sampled_construction_len"))
+        if self.parent_family_width_balance:
+            stats["puct/parent_family_width_balance_enabled"] = 1
+            for surface in PARENT_FAMILY_WIDTH_SURFACES:
+                stats[f"puct/parent_family_width_sample_count/{surface}"] = (
+                    self._parent_family_width_sample_counts.get(surface, 0)
+                )
+                stats[f"puct/parent_family_width_available/{surface}"] = (
+                    self._last_parent_family_width_available.get(surface, 0)
+                )
+            stats["puct/parent_family_width_fallbacks"] = self._last_parent_family_width_fallbacks
         return stats
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
         columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        if self.parent_family_width_balance:
+            columns = columns + [
+                "parent_family_width_surface",
+                "parent_family_width",
+                "parent_family_width_count_after",
+                "parent_family_width_fallback",
+            ]
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        parent_family_width_stats = (
+            self._last_parent_family_width_stats
+            if len(self._last_parent_family_width_stats) == len(self._last_sampled_states)
+            else [("root_or_seed", 0, 0, False)] * len(self._last_sampled_states)
+        )
+        for row_idx, (idx, state, (n, Q, P, bonus, score)) in enumerate(zip(indices, self._last_sampled_states, stats)):
             parent_val = state.parent_values[0] if state.parent_values else None
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            row = (idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score)
+            if self.parent_family_width_balance:
+                row = row + parent_family_width_stats[row_idx]
+            rows.append(row)
         return columns, rows
 
 
@@ -756,6 +981,7 @@ def create_sampler(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    parent_family_width_balance: bool = False,
 ) -> StateSampler:
     """Factory function to create samplers. Pass the env type (from config.env_type)."""
     if not log_path:
@@ -768,6 +994,7 @@ def create_sampler(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        parent_family_width_balance=parent_family_width_balance,
     )
 
 
@@ -778,6 +1005,7 @@ def get_or_create_sampler_with_default(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    parent_family_width_balance: bool = False,
 ) -> StateSampler:
     """Get sampler. Initial experience is created via env_type.create_initial_state."""
     return create_sampler(
@@ -787,4 +1015,5 @@ def get_or_create_sampler_with_default(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        parent_family_width_balance=parent_family_width_balance,
     )
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..a46dd6f 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_parent_family_width_balance: bool = False
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -146,6 +147,7 @@ async def discover_impl(config: DiscoverConfig):
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
             autonomous=config.codex_autonomous,
+            parent_family_width_balance=config.codex_parent_family_width_balance,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
             log_path=log_path,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..f11303c 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    parent_family_width_balance: bool = False
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -960,6 +961,7 @@ def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler
         batch_size=cfg.groups_per_batch,
         resume_step=start_batch if start_batch > 0 else None,
         topk_children=cfg.topk_children,
+        parent_family_width_balance=cfg.parent_family_width_balance,
     )
 
 
````
</details>

