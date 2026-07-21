#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash repro/aicrowd_whestbench/run_whest_mode_forcing_claude.sh [--dry-run]

Launches the existing AutoEvolve workflow once per prompt mode. Each launch
gets exactly one fixed WHEST_DIVERSITY_MODE_FILE; no mode is selected from
sample indexes or AutoEvolve actions.

Environment:
  RUN_WHEST_MF_MODES=...   "all" (default) or a comma-separated mode subset
  RUN_WHEST_MF_MODE_DIR=... Directory containing <mode>.md prompt files
  RUN_WHEST_MF_TAG=...     Experiment suffix (default mf1)
  PYTHON=...               Python executable (default .venv/bin/python)

All normal AutoEvolve settings are passed directly to run_official_mini.sh;
use its existing RUN_WHEST_AUTO_*, WHEST_*, and RUN_WHEST_* overrides.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN_ARGS=()
case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN_ARGS+=(--dry-run) ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
    echo "Too many arguments" >&2
    usage >&2
    exit 2
fi

PYTHON_BIN="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
MODE_SPEC="${RUN_WHEST_MF_MODES:-all}"
MODE_DIR="${RUN_WHEST_MF_MODE_DIR:-$REPO_ROOT/examples/aicrowd_whestbench/diversity_modes}"
TAG="${RUN_WHEST_MF_TAG:-mf1}"
EXPERIMENT_PREFIX="${RUN_WHEST_MF_EXPERIMENT_PREFIX:-whestbench_modeforcing_autoevolve_claude}"

ALL_MODES=(
    exact_radial
    rqmc_engine
    even_order_cv
    active_subspace
    frozen_gating_proxy_cv
    budget_economics
    deterministic_closure
)
if [[ "$MODE_SPEC" == "all" ]]; then
    MODES=("${ALL_MODES[@]}")
else
    IFS=',' read -r -a MODES <<<"$MODE_SPEC"
fi
if [[ ${#MODES[@]} -eq 0 ]]; then
    echo "RUN_WHEST_MF_MODES must select at least one mode" >&2
    exit 2
fi

export PYTHON="$PYTHON_BIN"
export RUN_WHEST_CODEX_COMMAND="${RUN_WHEST_CODEX_COMMAND:-claude}"
export RUN_WHEST_CODEX_MODEL="${RUN_WHEST_CODEX_MODEL-}"

for raw_mode in "${MODES[@]}"; do
    diversity_mode="${raw_mode#"${raw_mode%%[![:space:]]*}"}"
    diversity_mode="${diversity_mode%"${diversity_mode##*[![:space:]]}"}"
    mode_file="$MODE_DIR/$diversity_mode.md"
    if [[ ! -s "$mode_file" ]]; then
        echo "Missing or empty diversity mode file: $mode_file" >&2
        exit 2
    fi
    export WHEST_DIVERSITY_MODE_FILE="$mode_file"
    export RUN_WHEST_AUTO_EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_${diversity_mode}_${TAG}"
    export TTT_BLACKBOX_EVAL_SOCKET="/tmp/ttt_whest_mf_${diversity_mode}_${TAG}.sock"
    export TTT_BLACKBOX_EVAL_LOG_DIR="/tmp/ttt_whest_mf_${diversity_mode}_${TAG}_logs"

    echo "[$diversity_mode] launching existing AutoEvolve workflow"
    bash repro/aicrowd_whestbench/run_official_mini.sh \
        autoevolve "${DRY_RUN_ARGS[@]}"
done
