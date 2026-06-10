# codex/diversity-birth-cohort-density-balance

## Summary

按同一 birth cohort/parent cohort 的密度分类为 cohort_solo/cohort_small/cohort_dense/seed_root，优先采样历史上较少选择的 cohort density surface。

## Branch State

- Worktree: `/opt/tiger/discover-birth-cohort-density-balance`
- HEAD: `2234282`
- Base used for comparison: `2234282`
- Commits ahead of base: `0`
- Commits behind base: `0`
- Group: `completed_19`
- Implementation location: `uncommitted worktree diff`

## Implemented Behavior

- The summary above is based on reading the changed Python code, added config fields, and sampler/prompt hooks in this branch.
- For sampler branches, the common pattern is to keep baseline PUCT ordering inside a chosen surface/bucket, while changing which surface/bucket gets the next parent pick.
- Tests detected: `8` test functions across untracked test files.

## Added Markers

### Config fields

- `codex_sampler_diversity`
- `sampler_diversity`

### Constants

- `BIRTH_COHORT_DENSITY_SURFACES`
- `BIRTH_COHORT_DENSITY_COUNTS_KEY`
- `SAMPLER_DIVERSITY_MODES`

### Classes

- None

### Functions

- `_empty_birth_cohort_density_counts`
- `_sanitize_birth_cohort_density_counts`
- `_validate_sampler_diversity`
- `_direct_parent_id`
- `_nonnegative_timestep`
- `_birth_cohort_density_sizes`
- `_classify_birth_cohort_density`
- `_reset_birth_cohort_density_last`
- `_record_birth_cohort_density_pick`
- `_choose_birth_cohort_density_entry`
- `_sample_birth_cohort_density`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 282 insertions(+), 4 deletions(-)`
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

- `repro/gpu_mode/run_0609_birth_cohort_density_balance.sh (441 bytes)`
- `repro/run_discovery.py (4233 bytes)`
- `tests/test_codex_birth_cohort_density_balance.py (18321 bytes)`

### Detected Test Functions

- `tests/test_codex_birth_cohort_density_balance.py::test_disabled_mode_keeps_baseline_shape_for_one_and_many`
- `tests/test_codex_birth_cohort_density_balance.py::test_classifier_uses_only_birth_structure`
- `tests/test_codex_birth_cohort_density_balance.py::test_balanced_selector_surface_and_puct_ties`
- `tests/test_codex_birth_cohort_density_balance.py::test_multi_parent_updates_counts_and_blocks_lineage`
- `tests/test_codex_birth_cohort_density_balance.py::test_fallback_rows_increment_classified_surface_and_keep_blocking`
- `tests/test_codex_birth_cohort_density_balance.py::test_persistence_resume_missing_and_bad_counts_are_sanitized`
- `tests/test_codex_birth_cohort_density_balance.py::test_enabled_only_metrics_table_and_factory_validation`
- `tests/test_codex_birth_cohort_density_balance.py::test_cli_and_config_plumbing`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0609_birth_cohort_density_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

MODE="run"
if [[ $# -gt 0 ]]; then
  case "$1" in
    run|dry-run)
      MODE="$1"
      shift
      ;;
  esac
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python "${REPO_ROOT}/repro/run_discovery.py" "${MODE}" \
  --experiment-name gpu-mode-0609-birth-cohort-density-balance \
  --codex-sampler-diversity birth_cohort_density \
  "$@"
````

### `repro/run_discovery.py`

````python
#!/usr/bin/env python3
"""Run or dry-run the Codex no-finetune GPUMode discovery repro."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("run", "dry-run"), default="run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--experiment-name", default="codex-discovery")
    parser.add_argument("--problem-type", default="trimul")
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--num-cpus-per-task", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=1200)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--codex-backend", choices=("cli", "responses"), default="cli")
    parser.add_argument("--codex-model-name", default=None)
    parser.add_argument("--codex-cli-command", default="codex")
    parser.add_argument(
        "--codex-cli-sandbox",
        choices=("read-only", "workspace-write", "danger-full-access"),
        default="read-only",
    )
    parser.add_argument("--codex-cli-timeout", type=float, default=600.0)
    parser.add_argument("--codex-max-concurrent-requests", type=int, default=1)
    parser.add_argument("--codex-autonomous", action="store_true")
    parser.add_argument(
        "--codex-sampler-diversity",
        choices=("none", "birth_cohort_density"),
        default="none",
    )
    return parser


def _payload(args: argparse.Namespace, *, dry_run: bool) -> dict[str, object]:
    return {
        "dry_run": dry_run,
        "runner": "codex_no_finetune",
        "env": "GpuModeEnv",
        "experiment_name": args.experiment_name,
        "problem_type": args.problem_type,
        "num_epochs": args.num_epochs,
        "groups_per_batch": args.groups_per_batch,
        "group_size": args.group_size,
        "num_cpus_per_task": args.num_cpus_per_task,
        "eval_timeout": args.eval_timeout,
        "wandb_project": args.wandb_project,
        "codex_backend": args.codex_backend,
        "codex_model_name": args.codex_model_name,
        "codex_cli_command": args.codex_cli_command,
        "codex_cli_sandbox": args.codex_cli_sandbox,
        "codex_cli_timeout": args.codex_cli_timeout,
        "codex_max_concurrent_requests": args.codex_max_concurrent_requests,
        "codex_autonomous": args.codex_autonomous,
        "codex_sampler_diversity": args.codex_sampler_diversity,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    dry_run = bool(args.dry_run or args.mode == "dry-run")
    payload = _payload(args, dry_run=dry_run)
    if dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from examples.gpu_mode.env import GpuModeEnv
    from ttt_discover.codex_utils.discovery import DiscoverConfig, discover

    config = DiscoverConfig(
        runner="codex_no_finetune",
        env_type=GpuModeEnv,
        problem_type=args.problem_type,
        experiment_name=args.experiment_name,
        wandb_project=args.wandb_project,
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
        codex_autonomous=args.codex_autonomous,
        codex_sampler_diversity=args.codex_sampler_diversity,
    )
    discover(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
````

### `tests/test_codex_birth_cohort_density_balance.py`

````python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import (
    BIRTH_COHORT_DENSITY_COUNTS_KEY,
    BIRTH_COHORT_DENSITY_SURFACES,
    PUCTSampler,
    create_sampler,
    get_or_create_sampler_with_default,
)


ROOT = Path(__file__).resolve().parents[1]


class DummyEnv:
    state_type = State

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        return State(
            timestep=-1,
            construction=[],
            code="initial",
            value=0.0,
            id=f"initial-{problem_type or 'default'}",
        )

    @classmethod
    def refresh_initial_state(cls, state: State, problem_type: str) -> None:
        state.observation = f"refreshed:{problem_type}"


def make_state(
    state_id: str,
    *,
    timestep: int,
    value: float,
    parent_id: str | None = None,
    parents: list[dict] | list[object] | None = None,
) -> State:
    if parents is None:
        parents = [] if parent_id is None else [{"id": parent_id, "timestep": timestep - 1}]
    return State(
        timestep=timestep,
        construction=[state_id],
        code=f"code-{state_id}",
        value=value,
        parent_values=[value - 1.0] if parents else [],
        parents=parents,
        id=state_id,
        observation=f"obs-{state_id}",
    )


def make_sampler(
    tmp_path: Path,
    states: list[State],
    *,
    diversity: str = "none",
) -> PUCTSampler:
    sampler = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        problem_type="p",
        batch_size=0,
        puct_c=0.0,
        topk_children=0,
        sampler_diversity=diversity,
    )
    sampler._states = list(states)
    sampler._initial_states = [state for state in states if not state.parents]
    return sampler


def saved_payload(tmp_path: Path, step: int = 1) -> dict:
    return json.loads((tmp_path / f"puct_sampler_step_{step:06d}.json").read_text())


def test_disabled_mode_keeps_baseline_shape_for_one_and_many(tmp_path: Path) -> None:
    initial = make_state("initial-p", timestep=-1, value=200.0)
    one = make_sampler(tmp_path / "one", [initial], diversity="none")
    assert [state.id for state in one.sample_states(1)] == ["initial-p"]
    assert initial.observation == "refreshed:p"
    assert len(one._last_puct_stats) == 1
    assert one._last_puct_stats[0][0] == 0
    assert one._last_puct_stats[0][-1] == pytest.approx(200.0)
    metrics = one.get_sample_stats()
    columns, rows = one.get_sample_table()
    assert not any(key.startswith("puct/birth_cohort_density/") for key in metrics)
    assert "diversity_surface" not in columns
    assert rows[0][1:3] == (-1, 200.0)
    one.flush(step=1)
    assert BIRTH_COHORT_DENSITY_COUNTS_KEY not in saved_payload(tmp_path / "one")

    parent = make_state("parent", timestep=0, value=100.0, parent_id="root")
    child = make_state("child", timestep=1, value=99.0, parent_id="parent")
    other = make_state("other", timestep=0, value=50.0, parent_id="other-root")
    many = make_sampler(tmp_path / "many", [parent, child, other], diversity="none")
    assert [state.id for state in many.sample_states(2)] == ["parent", "other"]
    assert [stat[-1] for stat in many._last_puct_stats] == [100.0, 50.0]
    metrics = many.get_sample_stats()
    columns, _rows = many.get_sample_table()
    assert not any(key.startswith("puct/birth_cohort_density/") for key in metrics)
    assert "diversity_fallback" not in columns
    many.flush(step=1)
    assert BIRTH_COHORT_DENSITY_COUNTS_KEY not in saved_payload(tmp_path / "many")


def test_classifier_uses_only_birth_structure(tmp_path: Path) -> None:
    root = make_state("root", timestep=0, value=999.0)
    malformed = make_state("malformed", timestep=0, value=1.0, parents=[{"timestep": -1}])
    negative = make_state("negative", timestep=-1, value=1.0, parent_id="p")
    fractional = make_state("fractional", timestep=1, value=1.0, parent_id="p")
    fractional.timestep = 1.5
    solo = make_state("solo", timestep=0, value=1.0, parent_id="p0")
    small_a = make_state("small-a", timestep=1, value=1.0, parent_id="p1")
    small_b = make_state("small-b", timestep=1, value=2.0, parent_id="p2")
    small_c = make_state("small-c", timestep=1, value=3.0, parent_id="p3")
    dense = [
        make_state(f"dense-{idx}", timestep=2, value=float(idx), parent_id=f"pd{idx}")
        for idx in range(4)
    ]
    same_parent_t3 = make_state("same-parent-t3", timestep=3, value=1.0, parent_id="shared")
    same_parent_t4 = make_state("same-parent-t4", timestep=4, value=1.0, parent_id="shared")
    shared_a = make_state("shared-a", timestep=5, value=1.0, parent_id="left")
    shared_b = make_state("shared-b", timestep=5, value=1.0, parent_id="right")

    sampler = make_sampler(
        tmp_path,
        [
            root,
            malformed,
            negative,
            fractional,
            solo,
            small_a,
            small_b,
            small_c,
            *dense,
            same_parent_t3,
            same_parent_t4,
            shared_a,
            shared_b,
        ],
        diversity="birth_cohort_density",
    )
    sizes = sampler._birth_cohort_density_sizes()
    assert sizes[0] == 1
    assert sizes[1] == 3
    assert sizes[2] == 4
    assert sizes[3] == 1
    assert sizes[4] == 1
    assert sizes[5] == 2
    assert sampler._classify_birth_cohort_density(root, sizes) == ("seed_root", 0)
    assert sampler._classify_birth_cohort_density(malformed, sizes) == ("seed_root", 0)
    assert sampler._classify_birth_cohort_density(negative, sizes) == ("seed_root", 0)
    assert sampler._classify_birth_cohort_density(fractional, sizes) == ("seed_root", 0)
    assert sampler._classify_birth_cohort_density(solo, sizes) == ("cohort_solo", 1)
    assert sampler._classify_birth_cohort_density(small_a, sizes) == ("cohort_small", 3)
    assert sampler._classify_birth_cohort_density(dense[0], sizes) == ("cohort_dense", 4)
    assert sampler._classify_birth_cohort_density(same_parent_t3, sizes) == ("cohort_solo", 1)
    assert sampler._classify_birth_cohort_density(same_parent_t4, sizes) == ("cohort_solo", 1)
    assert sampler._classify_birth_cohort_density(shared_a, sizes) == ("cohort_small", 2)

    before = sampler._classify_birth_cohort_density(shared_b, sizes)
    shared_b.code = "completely different code"
    shared_b.observation = "different stdout"
    shared_b.construction = ["different", "construction"]
    shared_b.value = -100000.0
    assert sampler._classify_birth_cohort_density(shared_b, sizes) == before


def test_balanced_selector_surface_and_puct_ties(tmp_path: Path) -> None:
    solo = make_state("solo", timestep=0, value=100.0, parent_id="p0")
    small_hi = make_state("small-hi", timestep=1, value=90.0, parent_id="p1")
    small_lo = make_state("small-lo", timestep=1, value=80.0, parent_id="p2")
    sampler = make_sampler(
        tmp_path / "least-count",
        [solo, small_hi, small_lo],
        diversity="birth_cohort_density",
    )
    sampler._birth_cohort_density_counts.update(
        {
            "cohort_solo": 5,
            "cohort_small": 0,
            "cohort_dense": 5,
            "seed_root": 5,
        }
    )
    assert [state.id for state in sampler.sample_states(1)] == ["small-hi"]
    assert sampler._last_birth_cohort_density_rows == [
        ("cohort_small", 2, 0, 1, False)
    ]

    solo_low = make_state("solo-low", timestep=0, value=10.0, parent_id="p0")
    small_a = make_state("small-a", timestep=1, value=100.0, parent_id="p1")
    small_b = make_state("small-b", timestep=1, value=90.0, parent_id="p2")
    tie = make_sampler(
        tmp_path / "surface-order",
        [small_a, small_b, solo_low],
        diversity="birth_cohort_density",
    )
    assert [state.id for state in tie.sample_states(1)] == ["solo-low"]

    within = make_sampler(
        tmp_path / "within-surface",
        [solo, small_lo, small_hi],
        diversity="birth_cohort_density",
    )
    within._birth_cohort_density_counts.update(
        {
            "cohort_solo": 5,
            "cohort_small": 0,
            "cohort_dense": 5,
            "seed_root": 5,
        }
    )
    assert [state.id for state in within.sample_states(1)] == ["small-hi"]


def test_multi_parent_updates_counts_and_blocks_lineage(tmp_path: Path) -> None:
    parent = make_state("parent", timestep=0, value=100.0, parent_id="root")
    blocked_child = make_state("blocked-child", timestep=1, value=90.0, parent_id="parent")
    small_hi = make_state("small-hi", timestep=2, value=80.0, parent_id="left")
    small_lo = make_state("small-lo", timestep=2, value=70.0, parent_id="right")
    sampler = make_sampler(
        tmp_path,
        [parent, blocked_child, small_hi, small_lo],
        diversity="birth_cohort_density",
    )
    picked = sampler.sample_states(2)
    assert [state.id for state in picked] == ["parent", "small-hi"]
    assert "blocked-child" not in [state.id for state in picked]
    assert sampler._birth_cohort_density_counts["cohort_solo"] == 1
    assert sampler._birth_cohort_density_counts["cohort_small"] == 1
    assert sampler._last_birth_cohort_density_rows == [
        ("cohort_solo", 1, 0, 1, False),
        ("cohort_small", 2, 0, 1, False),
    ]


def test_fallback_rows_increment_classified_surface_and_keep_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dense_parent = make_state("dense-parent", timestep=0, value=100.0, parent_id="root")
    dense_child = make_state("dense-child", timestep=0, value=99.0, parent_id="dense-parent")
    dense_other = make_state("dense-other", timestep=0, value=98.0, parent_id="other")
    dense_more = make_state("dense-more", timestep=0, value=97.0, parent_id="more")
    sampler = make_sampler(
        tmp_path,
        [dense_parent, dense_child, dense_other, dense_more],
        diversity="birth_cohort_density",
    )
    sampler._birth_cohort_density_counts.update(
        {
            "cohort_solo": 0,
            "cohort_small": 5,
            "cohort_dense": 5,
            "seed_root": 0,
        }
    )
    picked = sampler.sample_states(2)
    picked_ids = [state.id for state in picked]
    assert picked_ids == ["dense-parent", "dense-other"]
    assert "dense-child" not in picked_ids
    assert sampler._birth_cohort_density_counts["cohort_dense"] == 7
    assert sampler._last_birth_cohort_density_fallback == 0
    assert [row[-1] for row in sampler._last_birth_cohort_density_rows] == [False, False]

    fallback = make_sampler(
        tmp_path / "forced-fallback",
        [dense_parent, dense_child, dense_other, dense_more],
        diversity="birth_cohort_density",
    )
    monkeypatch.setattr(
        fallback,
        "_choose_birth_cohort_density_entry",
        lambda grouped: None,
    )
    picked = fallback.sample_states(2)
    picked_ids = [state.id for state in picked]
    assert picked_ids == ["dense-parent", "dense-other"]
    assert "dense-child" not in picked_ids
    assert fallback._birth_cohort_density_counts["cohort_dense"] == 2
    assert fallback._last_birth_cohort_density_fallback == 2
    assert [row[-1] for row in fallback._last_birth_cohort_density_rows] == [True, True]


def test_persistence_resume_missing_and_bad_counts_are_sanitized(tmp_path: Path) -> None:
    state = make_state("solo", timestep=0, value=1.0, parent_id="p0")
    sampler = make_sampler(tmp_path, [state], diversity="birth_cohort_density")
    sampler._birth_cohort_density_counts.update(
        {
            "cohort_solo": 3,
            "cohort_small": 2,
            "cohort_dense": 1,
            "seed_root": 4,
        }
    )
    sampler.flush(step=1)

    loaded = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=1,
        topk_children=0,
        sampler_diversity="birth_cohort_density",
    )
    assert loaded._birth_cohort_density_counts == {
        "cohort_solo": 3,
        "cohort_small": 2,
        "cohort_dense": 1,
        "seed_root": 4,
    }

    path = tmp_path / "puct_sampler_step_000001.json"
    payload = json.loads(path.read_text())
    payload.pop(BIRTH_COHORT_DENSITY_COUNTS_KEY)
    path.write_text(json.dumps(payload), encoding="utf-8")
    missing = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=1,
        topk_children=0,
        sampler_diversity="birth_cohort_density",
    )
    assert missing._birth_cohort_density_counts == {
        surface: 0 for surface in BIRTH_COHORT_DENSITY_SURFACES
    }

    payload[BIRTH_COHORT_DENSITY_COUNTS_KEY] = {
        "cohort_solo": True,
        "cohort_small": "2",
        "cohort_dense": -1,
        "seed_root": 8,
        "unknown": 99,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    bad = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=1,
        topk_children=0,
        sampler_diversity="birth_cohort_density",
    )
    assert bad._birth_cohort_density_counts == {
        "cohort_solo": 0,
        "cohort_small": 0,
        "cohort_dense": 0,
        "seed_root": 8,
    }


def test_enabled_only_metrics_table_and_factory_validation(tmp_path: Path) -> None:
    solo = make_state("solo", timestep=0, value=1.0, parent_id="p0")
    enabled = make_sampler(tmp_path / "enabled", [solo], diversity="birth_cohort_density")
    enabled.sample_states(1)
    metrics = enabled.get_sample_stats()
    columns, rows = enabled.get_sample_table()
    assert metrics["puct/birth_cohort_density/enabled"] == 1
    assert metrics["puct/birth_cohort_density/count/cohort_solo"] == 1
    assert metrics["puct/birth_cohort_density/last_available/cohort_solo"] == 1
    assert metrics["puct/birth_cohort_density/last_sampled/cohort_solo"] == 1
    assert metrics["puct/birth_cohort_density/fallback_last"] == 0
    assert columns[-5:] == [
        "diversity_surface",
        "birth_cohort_size",
        "diversity_count_before",
        "diversity_count_after",
        "diversity_fallback",
    ]
    assert rows[0][-5:] == ("cohort_solo", 1, 0, 1, False)

    made = create_sampler(
        log_path=str(tmp_path / "factory"),
        env_type=DummyEnv,
        batch_size=0,
        sampler_diversity="birth_cohort_density",
    )
    assert isinstance(made, PUCTSampler)
    assert made.sampler_diversity == "birth_cohort_density"
    default = get_or_create_sampler_with_default(
        log_path=str(tmp_path / "factory-default"),
        env_type=DummyEnv,
        batch_size=0,
    )
    assert isinstance(default, PUCTSampler)
    assert default.sampler_diversity == "none"
    with pytest.raises(ValueError):
        create_sampler(
            log_path=str(tmp_path / "bad"),
            env_type=DummyEnv,
            batch_size=0,
            sampler_diversity="bad",
        )


def test_cli_and_config_plumbing() -> None:
    default = subprocess.run(
        [sys.executable, str(ROOT / "repro/run_discovery.py"), "dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(default.stdout)
    assert payload["codex_sampler_diversity"] == "none"
    assert payload["codex_model_name"] is None

    enabled = subprocess.run(
        [
            sys.executable,
            str(ROOT / "repro/run_discovery.py"),
            "dry-run",
            "--codex-sampler-diversity",
            "birth_cohort_density",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(enabled.stdout)["codex_sampler_diversity"] == "birth_cohort_density"

    flag = subprocess.run(
        [
            sys.executable,
            str(ROOT / "repro/run_discovery.py"),
            "--dry-run",
            "--codex-sampler-diversity",
            "birth_cohort_density",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(flag.stdout)["codex_sampler_diversity"] == "birth_cohort_density"

    invalid = subprocess.run(
        [
            sys.executable,
            str(ROOT / "repro/run_discovery.py"),
            "dry-run",
            "--codex-sampler-diversity",
            "bad",
        ],
        capture_output=True,
        text=True,
    )
    assert invalid.returncode != 0

    wrapper = subprocess.run(
        [
            "bash",
            str(ROOT / "repro/gpu_mode/run_0609_birth_cohort_density_balance.sh"),
            "dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wrapper_payload = json.loads(wrapper.stdout)
    assert wrapper_payload["experiment_name"] == "gpu-mode-0609-birth-cohort-density-balance"
    assert wrapper_payload["codex_sampler_diversity"] == "birth_cohort_density"

    override = subprocess.run(
        [
            "bash",
            str(ROOT / "repro/gpu_mode/run_0609_birth_cohort_density_balance.sh"),
            "dry-run",
            "--codex-sampler-diversity",
            "none",
            "--experiment-name",
            "override",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    override_payload = json.loads(override.stdout)
    assert override_payload["experiment_name"] == "override"
    assert override_payload["codex_sampler_diversity"] == "none"

    run_discovery = (ROOT / "repro/run_discovery.py").read_text()
    dry_run_prefix = run_discovery.split("if dry_run:", 1)[0]
    assert "from ttt_discover" not in dry_run_prefix
    assert "DiscoverConfig(" not in dry_run_prefix

    assert "sampler_diversity=cfg.sampler_diversity" in (
        ROOT / "ttt_discover/rl/codex_no_finetune.py"
    ).read_text()
    for rel_path in ("ttt_discover/discovery.py", "ttt_discover/codex_utils/discovery.py"):
        text = (ROOT / rel_path).read_text()
        assert 'codex_sampler_diversity: Literal["none", "birth_cohort_density"] = "none"' in text
        assert "sampler_diversity=config.codex_sampler_diversity" in text
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   2 +
 ttt_discover/codex_utils/sampler.py   | 280 +++++++++++++++++++++++++++++++++-
 ttt_discover/discovery.py             |   2 +
 ttt_discover/rl/codex_no_finetune.py  |   2 +
 4 files changed, 282 insertions(+), 4 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..6e12d01 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_sampler_diversity: Literal["none", "birth_cohort_density"] = "none"
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -85,6 +86,7 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
         autonomous=config.codex_autonomous,
+        sampler_diversity=config.codex_sampler_diversity,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
         log_path=log_path,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..657fa5e 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -8,7 +8,7 @@ import os
 from pathlib import Path
 import threading
 import time
-from typing import Any, Callable
+from typing import Any, Callable, Literal
 
 import numpy as np
 
@@ -16,6 +16,64 @@ from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_ser
 
 logger = logging.getLogger(__name__)
 
+BIRTH_COHORT_DENSITY_SURFACES = (
+    "cohort_solo",
+    "cohort_small",
+    "cohort_dense",
+    "seed_root",
+)
+BIRTH_COHORT_DENSITY_COUNTS_KEY = "puct_birth_cohort_density_counts"
+SAMPLER_DIVERSITY_MODES = ("none", "birth_cohort_density")
+
+
+def _empty_birth_cohort_density_counts() -> dict[str, int]:
+    return {surface: 0 for surface in BIRTH_COHORT_DENSITY_SURFACES}
+
+
+def _sanitize_birth_cohort_density_counts(raw: Any) -> dict[str, int]:
+    counts = _empty_birth_cohort_density_counts()
+    if not isinstance(raw, dict):
+        return counts
+    for surface in BIRTH_COHORT_DENSITY_SURFACES:
+        value = raw.get(surface, 0)
+        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
+            counts[surface] = 0
+        else:
+            counts[surface] = int(value)
+    return counts
+
+
+def _validate_sampler_diversity(value: str) -> str:
+    if value not in SAMPLER_DIVERSITY_MODES:
+        raise ValueError(
+            f"sampler_diversity must be one of {SAMPLER_DIVERSITY_MODES}; got {value!r}"
+        )
+    return value
+
+
+def _direct_parent_id(state: State) -> str | None:
+    parents = getattr(state, "parents", None)
+    if not isinstance(parents, list) or not parents:
+        return None
+    parent = parents[0]
+    if not isinstance(parent, dict):
+        return None
+    parent_id = parent.get("id")
+    if parent_id is None or parent_id == "":
+        return None
+    return str(parent_id)
+
+
+def _nonnegative_timestep(state: State) -> int | None:
+    timestep = getattr(state, "timestep", -1)
+    if isinstance(timestep, bool):
+        return None
+    if not isinstance(timestep, (int, np.integer)):
+        return None
+    if int(timestep) < 0:
+        return None
+    return int(timestep)
+
 
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
@@ -353,6 +411,7 @@ class PUCTSampler(StateSampler):
         resume_step: int | None = None,
         puct_c: float = 1.0,
         topk_children: int = 2,
+        sampler_diversity: Literal["none", "birth_cohort_density"] = "none",
     ):
         self.file_path = file_path
         self.env_type = env_type
@@ -361,6 +420,7 @@ class PUCTSampler(StateSampler):
         self.batch_size = batch_size
         self.topk_children = topk_children
         self.puct_c = float(puct_c)
+        self.sampler_diversity = _validate_sampler_diversity(sampler_diversity)
         
         self._states: list[State] = []
         self._initial_states: list[State] = []
@@ -375,6 +435,11 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._birth_cohort_density_counts = _empty_birth_cohort_density_counts()
+        self._last_birth_cohort_density_available = _empty_birth_cohort_density_counts()
+        self._last_birth_cohort_density_sampled = _empty_birth_cohort_density_counts()
+        self._last_birth_cohort_density_fallback = 0
+        self._last_birth_cohort_density_rows: list[tuple[str, int, int, int, bool]] = []
         
         if resume_step is not None:
             self._load(resume_step)
@@ -399,6 +464,10 @@ class PUCTSampler(StateSampler):
         self._n = store.get("puct_n", {}) or {}
         self._m = store.get("puct_m", {}) or {}
         self._T = int(store.get("puct_T", 0) or 0)
+        if self.sampler_diversity == "birth_cohort_density":
+            self._birth_cohort_density_counts = _sanitize_birth_cohort_density_counts(
+                store.get(BIRTH_COHORT_DENSITY_COUNTS_KEY)
+            )
 
     def _save(self, step: int):
         save_path = _sampler_file_for_step(self.file_path, step)
@@ -411,6 +480,10 @@ class PUCTSampler(StateSampler):
             "puct_m": self._m,
             "puct_T": self._T,
         }
+        if self.sampler_diversity == "birth_cohort_density":
+            store[BIRTH_COHORT_DENSITY_COUNTS_KEY] = dict(
+                self._birth_cohort_density_counts
+            )
         with _file_lock(f"{save_path}.lock"):
             _atomic_write_json(save_path, store)
 
@@ -489,6 +562,153 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
+    def _birth_cohort_density_sizes(self) -> dict[int, int]:
+        cohort_sizes: dict[int, int] = {}
+        for state in self._states:
+            if _direct_parent_id(state) is None:
+                continue
+            timestep = _nonnegative_timestep(state)
+            if timestep is None:
+                continue
+            cohort_sizes[timestep] = cohort_sizes.get(timestep, 0) + 1
+        return cohort_sizes
+
+    def _classify_birth_cohort_density(
+        self,
+        state: State,
+        cohort_sizes: dict[int, int],
+    ) -> tuple[str, int]:
+        if _direct_parent_id(state) is None:
+            return "seed_root", 0
+        timestep = _nonnegative_timestep(state)
+        if timestep is None:
+            return "seed_root", 0
+        cohort_size = int(cohort_sizes.get(timestep, 0))
+        if cohort_size <= 1:
+            return "cohort_solo", cohort_size
+        if cohort_size <= 3:
+            return "cohort_small", cohort_size
+        return "cohort_dense", cohort_size
+
+    def _reset_birth_cohort_density_last(self) -> None:
+        self._last_birth_cohort_density_available = _empty_birth_cohort_density_counts()
+        self._last_birth_cohort_density_sampled = _empty_birth_cohort_density_counts()
+        self._last_birth_cohort_density_fallback = 0
+        self._last_birth_cohort_density_rows = []
+
+    def _record_birth_cohort_density_pick(
+        self,
+        state: State,
+        cohort_sizes: dict[int, int],
+        *,
+        fallback: bool,
+    ) -> None:
+        surface, cohort_size = self._classify_birth_cohort_density(state, cohort_sizes)
+        before = self._birth_cohort_density_counts[surface]
+        after = before + 1
+        self._birth_cohort_density_counts[surface] = after
+        self._last_birth_cohort_density_sampled[surface] += 1
+        if fallback:
+            self._last_birth_cohort_density_fallback += 1
+        self._last_birth_cohort_density_rows.append(
+            (surface, cohort_size, before, after, bool(fallback))
+        )
+
+    def _choose_birth_cohort_density_entry(
+        self,
+        grouped: dict[str, list[tuple[float, float, State, int, float, float, float]]],
+    ) -> tuple[float, float, State, int, float, float, float] | None:
+        available_surfaces = [
+            surface
+            for surface in BIRTH_COHORT_DENSITY_SURFACES
+            if grouped[surface]
+        ]
+        if not available_surfaces:
+            return None
+        chosen_surface = min(
+            available_surfaces,
+            key=lambda surface: (
+                self._birth_cohort_density_counts[surface],
+                BIRTH_COHORT_DENSITY_SURFACES.index(surface),
+            ),
+        )
+        return grouped[chosen_surface][0]
+
+    def _sample_birth_cohort_density(
+        self,
+        scores: list[tuple[float, float, State, int, float, float, float]],
+        num_states: int,
+    ) -> tuple[list[State], list[tuple[float, float, State, int, float, float, float]]]:
+        self._reset_birth_cohort_density_last()
+        cohort_sizes = self._birth_cohort_density_sizes()
+        for entry in scores:
+            surface, _cohort_size = self._classify_birth_cohort_density(entry[2], cohort_sizes)
+            self._last_birth_cohort_density_available[surface] += 1
+
+        children_map = self._build_children_map() if num_states > 1 else {}
+        picked: list[State] = []
+        top_scores: list[tuple[float, float, State, int, float, float, float]] = []
+        picked_ids: set[str] = set()
+        blocked_ids: set[str] = set()
+
+        while len(picked) < num_states:
+            available = [
+                entry
+                for entry in scores
+                if entry[2].id not in picked_ids
+                and (num_states <= 1 or entry[2].id not in blocked_ids)
+            ]
+            if not available:
+                break
+
+            grouped: dict[str, list[tuple[float, float, State, int, float, float, float]]] = {
+                surface: [] for surface in BIRTH_COHORT_DENSITY_SURFACES
+            }
+            for entry in available:
+                surface, _cohort_size = self._classify_birth_cohort_density(
+                    entry[2],
+                    cohort_sizes,
+                )
+                grouped[surface].append(entry)
+
+            chosen_entry = self._choose_birth_cohort_density_entry(grouped)
+            if chosen_entry is None:
+                break
+
+            state = chosen_entry[2]
+            picked.append(state)
+            top_scores.append(chosen_entry)
+            picked_ids.add(state.id)
+            self._record_birth_cohort_density_pick(
+                state,
+                cohort_sizes,
+                fallback=False,
+            )
+            if num_states > 1:
+                blocked_ids.update(self._get_full_lineage(state, children_map))
+
+        if len(picked) < num_states:
+            for entry in scores:
+                if len(picked) >= num_states:
+                    break
+                state = entry[2]
+                if state.id in picked_ids:
+                    continue
+                if num_states > 1 and state.id in blocked_ids:
+                    continue
+                picked.append(state)
+                top_scores.append(entry)
+                picked_ids.add(state.id)
+                self._record_birth_cohort_density_pick(
+                    state,
+                    cohort_sizes,
+                    fallback=True,
+                )
+                if num_states > 1:
+                    blocked_ids.update(self._get_full_lineage(state, children_map))
+
+        return picked, top_scores
+
     def sample_states(self, num_states: int) -> list[State]:
         initial_ids = {s.id for s in self._initial_states}
         candidates = list(self._states)
@@ -501,6 +721,15 @@ class PUCTSampler(StateSampler):
             self._last_sampled_states = picked
             self._last_sampled_indices = []
             self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+            if self.sampler_diversity == "birth_cohort_density":
+                self._reset_birth_cohort_density_last()
+                cohort_sizes = self._birth_cohort_density_sizes()
+                for state in picked:
+                    self._record_birth_cohort_density_pick(
+                        state,
+                        cohort_sizes,
+                        fallback=False,
+                    )
             return picked
 
         vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
@@ -521,7 +750,9 @@ class PUCTSampler(StateSampler):
 
         scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
 
-        if num_states > 1:
+        if self.sampler_diversity == "birth_cohort_density":
+            picked, top_scores = self._sample_birth_cohort_density(scores, num_states)
+        elif num_states > 1:
             children_map = self._build_children_map()
             picked, top_scores, blocked_ids = [], [], set()
             for entry in scores:
@@ -731,21 +962,57 @@ class PUCTSampler(StateSampler):
         stats.update(_stats(sampled_values, "puct/sampled_value"))
         stats.update(_stats(sampled_timesteps, "puct/sampled_timestep"))
         stats.update(_stats(sampled_constr_lens, "puct/sampled_construction_len"))
+        if self.sampler_diversity == "birth_cohort_density":
+            stats["puct/birth_cohort_density/enabled"] = 1
+            for surface in BIRTH_COHORT_DENSITY_SURFACES:
+                stats[f"puct/birth_cohort_density/count/{surface}"] = int(
+                    self._birth_cohort_density_counts[surface]
+                )
+                stats[f"puct/birth_cohort_density/last_available/{surface}"] = int(
+                    self._last_birth_cohort_density_available[surface]
+                )
+                stats[f"puct/birth_cohort_density/last_sampled/{surface}"] = int(
+                    self._last_birth_cohort_density_sampled[surface]
+                )
+            stats["puct/birth_cohort_density/fallback_last"] = int(
+                self._last_birth_cohort_density_fallback
+            )
         return stats
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
         columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        if self.sampler_diversity == "birth_cohort_density":
+            columns = columns + [
+                "diversity_surface",
+                "birth_cohort_size",
+                "diversity_count_before",
+                "diversity_count_after",
+                "diversity_fallback",
+            ]
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        if (
+            self.sampler_diversity == "birth_cohort_density"
+            and len(self._last_birth_cohort_density_rows) == len(self._last_sampled_states)
+        ):
+            diversity_rows = self._last_birth_cohort_density_rows
+        else:
+            diversity_rows = [
+                ("seed_root", 0, 0, 0, False)
+                for _state in self._last_sampled_states
+            ]
+        for idx, state, (n, Q, P, bonus, score), diversity_row in zip(indices, self._last_sampled_states, stats, diversity_rows):
             parent_val = state.parent_values[0] if state.parent_values else None
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            row = (idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score)
+            if self.sampler_diversity == "birth_cohort_density":
+                row = row + diversity_row
+            rows.append(row)
         return columns, rows
 
 
@@ -756,8 +1023,10 @@ def create_sampler(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    sampler_diversity: Literal["none", "birth_cohort_density"] = "none",
 ) -> StateSampler:
     """Factory function to create samplers. Pass the env type (from config.env_type)."""
+    sampler_diversity = _validate_sampler_diversity(sampler_diversity)
     if not log_path:
         raise ValueError("log_path is required when using PUCT sampler")
     sampler_path = os.path.join(log_path, "puct_sampler.json")
@@ -768,6 +1037,7 @@ def create_sampler(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        sampler_diversity=sampler_diversity,
     )
 
 
@@ -778,6 +1048,7 @@ def get_or_create_sampler_with_default(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    sampler_diversity: Literal["none", "birth_cohort_density"] = "none",
 ) -> StateSampler:
     """Get sampler. Initial experience is created via env_type.create_initial_state."""
     return create_sampler(
@@ -787,4 +1058,5 @@ def get_or_create_sampler_with_default(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        sampler_diversity=sampler_diversity,
     )
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..a896d7d 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_sampler_diversity: Literal["none", "birth_cohort_density"] = "none"
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -146,6 +147,7 @@ async def discover_impl(config: DiscoverConfig):
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
             autonomous=config.codex_autonomous,
+            sampler_diversity=config.codex_sampler_diversity,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
             log_path=log_path,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..03fd4c9 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    sampler_diversity: Literal["none", "birth_cohort_density"] = "none"
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -960,6 +961,7 @@ def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler
         batch_size=cfg.groups_per_batch,
         resume_step=start_batch if start_batch > 0 else None,
         topk_children=cfg.topk_children,
+        sampler_diversity=cfg.sampler_diversity,
     )
 
 
````
</details>

