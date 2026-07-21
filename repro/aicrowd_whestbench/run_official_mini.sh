#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash repro/aicrowd_whestbench/run_official_mini.sh [all|ttt|autoevolve|autoevolve-in-process] [--dry-run]

This is a CPU workload; CUDA/GPU settings are intentionally not used.

Environment overrides:
  RUN_WHEST_SMOKE=1                  One epoch/sample
  RUN_WHEST_DRY_RUN=1                Print configs without launching
  RUN_WHEST_LOG_ROOT=...             Discovery log root
  RUN_WHEST_TTT_NUM_EPOCHS=...       Override TTT-Discover epochs
  RUN_WHEST_AUTO_NUM_EPOCHS=...      Override AutoEvolve epochs
  RUN_WHEST_AUTO_MAX_EVALUATOR_CALLS=...  Stop after this many verified candidates
  RUN_WHEST_BLACKBOX_MAX_EVALUATIONS=...  Hard cap shared by agent and outer verifier calls
  RUN_WHEST_BLACKBOX_CACHE=1         Cache exact duplicate submissions
  RUN_WHEST_VERIFIER_WORKERS=...     CPU verifier workers (default 1)
  RUN_WHEST_CODEX_MODEL=...          Empty string uses the CLI's default model
  WHEST_INITIAL_ESTIMATOR_PATH=...    Seed AutoEvolve from an estimator source file
  WHEST_DIVERSITY_MODE_FILE=...       UTF-8 prompt block appended for this launch
  WHEST_SEARCH_N_MLPS=...            MLPs per candidate (default 100)
  WHEST_DEPS_PATH=...                 Directory containing whestbench/flopscope
  HF_HOME=...                         Hugging Face cache (default /tmp/hf-whest-cache)
  TTT_AUTONOMOUS_MASK_PATHS=...       Paths hidden from the AutoEvolve agent
  TTT_BLACKBOX_EVAL_SOCKET=...        AutoEvolve verifier Unix socket
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TARGET="${RUN_WHEST_TARGET:-all}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        all)
            TARGET="all"
            ;;
        ttt|ttt_discover)
            TARGET="ttt_discover"
            ;;
        auto|autoevolve|blackbox|autoevolve_blackbox)
            TARGET="autoevolve_blackbox"
            ;;
        auto-in-process|autoevolve-in-process|autoevolve_in_process)
            TARGET="autoevolve_in_process"
            ;;
        --dry-run)
            RUN_WHEST_DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

PYTHON_BIN="${PYTHON:-python}"
LOG_ROOT="${RUN_WHEST_LOG_ROOT:-codex_runs/aicrowd_whestbench}"
WANDB_PROJECT="${RUN_WHEST_WANDB_PROJECT:-}"
CODEX_MODEL="${RUN_WHEST_CODEX_MODEL-gpt-5.6-sol}"
CODEX_COMMAND="${RUN_WHEST_CODEX_COMMAND:-codex}"
NUM_CPUS_PER_TASK="${RUN_WHEST_NUM_CPUS_PER_TASK:-1}"
EVAL_TIMEOUT="${RUN_WHEST_EVAL_TIMEOUT:-4000}"
SEARCH_N_MLPS="${WHEST_SEARCH_N_MLPS:-100}"
VERIFIER_WORKERS="${RUN_WHEST_VERIFIER_WORKERS:-1}"
BLACKBOX_MAX_EVALUATIONS="${RUN_WHEST_BLACKBOX_MAX_EVALUATIONS:-}"
BLACKBOX_CACHE="${RUN_WHEST_BLACKBOX_CACHE:-0}"

TTT_NUM_EPOCHS="${RUN_WHEST_TTT_NUM_EPOCHS:-50}"
TTT_GROUP_SIZE="${RUN_WHEST_TTT_GROUP_SIZE:-1}"
TTT_GROUPS_PER_BATCH="${RUN_WHEST_TTT_GROUPS_PER_BATCH:-1}"
TTT_CLI_TIMEOUT="${RUN_WHEST_TTT_CLI_TIMEOUT:-900}"
TTT_MAX_CONCURRENT_REQUESTS="${RUN_WHEST_TTT_MAX_CONCURRENT_REQUESTS:-1}"

AUTO_NUM_EPOCHS="${RUN_WHEST_AUTO_NUM_EPOCHS:-20}"
AUTO_GROUP_SIZE="${RUN_WHEST_AUTO_GROUP_SIZE:-2}"
AUTO_GROUPS_PER_BATCH="${RUN_WHEST_AUTO_GROUPS_PER_BATCH:-1}"
AUTO_CLI_TIMEOUT="${RUN_WHEST_AUTO_CLI_TIMEOUT:-7200}"
AUTO_MAX_CONCURRENT_REQUESTS="${RUN_WHEST_AUTO_MAX_CONCURRENT_REQUESTS:-2}"
AUTO_MAX_EVALUATOR_CALLS="${RUN_WHEST_AUTO_MAX_EVALUATOR_CALLS:-null}"

if [[ "${RUN_WHEST_SMOKE:-0}" == "1" ]]; then
    TTT_NUM_EPOCHS="${RUN_WHEST_TTT_NUM_EPOCHS:-1}"
    TTT_GROUP_SIZE="${RUN_WHEST_TTT_GROUP_SIZE:-1}"
    TTT_GROUPS_PER_BATCH="${RUN_WHEST_TTT_GROUPS_PER_BATCH:-1}"
    AUTO_NUM_EPOCHS="${RUN_WHEST_AUTO_NUM_EPOCHS:-1}"
    AUTO_GROUP_SIZE="${RUN_WHEST_AUTO_GROUP_SIZE:-1}"
    AUTO_GROUPS_PER_BATCH="${RUN_WHEST_AUTO_GROUPS_PER_BATCH:-1}"
    AUTO_CLI_TIMEOUT="${RUN_WHEST_AUTO_CLI_TIMEOUT:-1800}"
fi

TTT_EXPERIMENT_NAME="${RUN_WHEST_TTT_EXPERIMENT_NAME:-whestbench_mini_ttt_discover_n${SEARCH_N_MLPS}}"
AUTO_EXPERIMENT_NAME="${RUN_WHEST_AUTO_EXPERIMENT_NAME:-whestbench_mini_autoevolve_blackbox_n${SEARCH_N_MLPS}}"
BLACKBOX_SOCKET="${TTT_BLACKBOX_EVAL_SOCKET:-/tmp/ttt_blackbox_eval_whestbench.sock}"
BLACKBOX_LOG_DIR="${TTT_BLACKBOX_EVAL_LOG_DIR:-/tmp/ttt_blackbox_eval_whestbench_logs}"

export HF_HOME="${HF_HOME:-/tmp/hf-whest-cache}"
export TTT_AUTONOMOUS_MASK_PATHS="${TTT_AUTONOMOUS_MASK_PATHS:-$HF_HOME}"
WHEST_DEPS_PATH="${WHEST_DEPS_PATH:-}"
if [[ -z "$WHEST_DEPS_PATH" && -d /tmp/whest-official-deps ]]; then
    WHEST_DEPS_PATH=/tmp/whest-official-deps
fi
if [[ -n "$WHEST_DEPS_PATH" ]]; then
    export PYTHONPATH="$WHEST_DEPS_PATH:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
else
    export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi

# The blackbox verifier is a sibling process, so it must inherit the suite
# settings directly rather than relying on run_from_yaml.py's child-only env.
export WHEST_DATASET="aicrowd/arc-whestbench-public-2026"
export WHEST_DATASET_REVISION="v1-phase1"
export WHEST_DATASET_SPLIT="mini"
export WHEST_DATASET_STREAMING="0"
export WHEST_N_MLPS="$SEARCH_N_MLPS"
export WHEST_FLOP_BUDGET="272000000000"
export WHEST_LAMBDA_FLOPS_PER_SECOND="100000000000"
export WHEST_WALL_TIME_LIMIT="60"
export WHEST_RUNNER="subprocess"
export WHEST_INITIAL_ESTIMATOR_PATH="${WHEST_INITIAL_ESTIMATOR_PATH:-}"
if [[ "${RUN_WHEST_DRY_RUN:-0}" != "1" ]]; then
    "$PYTHON_BIN" -c 'import flopscope, whestbench' 2>/dev/null || {
        echo "Missing official dependencies. Install the project or set WHEST_DEPS_PATH." >&2
        exit 1
    }
fi

DRY_RUN_ARGS=()
if [[ "${RUN_WHEST_DRY_RUN:-0}" == "1" ]]; then
    DRY_RUN_ARGS+=(--dry-run)
fi

CONFIG_FILE="$(mktemp /tmp/run_whestbench_mini.XXXXXX.yaml)"
SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [[ -n "$SERVER_PID" ]]; then
        rm -f "$BLACKBOX_SOCKET"
    fi
    rm -f "$CONFIG_FILE"
}
trap cleanup EXIT

cat > "$CONFIG_FILE" <<YAML
defaults:
  env_type: examples.aicrowd_whestbench.env:WhestBenchEnv
  problem_type: arc_whestbench_2026
  backend: cli
  model_name: "$CODEX_MODEL"
  cli_command: "$CODEX_COMMAND"
  num_cpus_per_task: $NUM_CPUS_PER_TASK
  eval_timeout: $EVAL_TIMEOUT
  wandb_project: "$WANDB_PROJECT"
  remove_constant_reward_groups: false
  env:
    PYTHONUNBUFFERED: "1"
    HF_HOME: "$HF_HOME"
    WHEST_DIVERSITY_MODE_FILE: "${WHEST_DIVERSITY_MODE_FILE:-}"
    WHEST_DATASET: aicrowd/arc-whestbench-public-2026
    WHEST_DATASET_REVISION: v1-phase1
    WHEST_DATASET_SPLIT: mini
    WHEST_DATASET_STREAMING: "0"
    WHEST_N_MLPS: "$SEARCH_N_MLPS"
    WHEST_FLOP_BUDGET: "272000000000"
    WHEST_LAMBDA_FLOPS_PER_SECOND: "100000000000"
    WHEST_WALL_TIME_LIMIT: "60"
    WHEST_RUNNER: subprocess
    WHEST_INITIAL_ESTIMATOR_PATH: "$WHEST_INITIAL_ESTIMATOR_PATH"

runs:
  ttt_discover:
    algorithm: ttt_discover
    eval_runner: in_process
    experiment_name: "$TTT_EXPERIMENT_NAME"
    log_path: "$LOG_ROOT/$TTT_EXPERIMENT_NAME"
    num_epochs: $TTT_NUM_EPOCHS
    group_size: $TTT_GROUP_SIZE
    groups_per_batch: $TTT_GROUPS_PER_BATCH
    cli_sandbox: read-only
    cli_timeout: $TTT_CLI_TIMEOUT
    max_concurrent_requests: $TTT_MAX_CONCURRENT_REQUESTS

  autoevolve_blackbox:
    algorithm: autoevolve
    eval_runner: blackbox
    experiment_name: "$AUTO_EXPERIMENT_NAME"
    log_path: "$LOG_ROOT/$AUTO_EXPERIMENT_NAME"
    num_epochs: $AUTO_NUM_EPOCHS
    max_evaluator_calls: $AUTO_MAX_EVALUATOR_CALLS
    group_size: $AUTO_GROUP_SIZE
    groups_per_batch: $AUTO_GROUPS_PER_BATCH
    remove_constant_reward_groups: true
    cli_sandbox: danger-full-access
    cli_timeout: $AUTO_CLI_TIMEOUT
    max_concurrent_requests: $AUTO_MAX_CONCURRENT_REQUESTS
    blackbox_eval_socket: "$BLACKBOX_SOCKET"

  autoevolve_in_process:
    algorithm: autoevolve
    eval_runner: in_process
    experiment_name: "$AUTO_EXPERIMENT_NAME"
    log_path: "$LOG_ROOT/$AUTO_EXPERIMENT_NAME"
    num_epochs: $AUTO_NUM_EPOCHS
    group_size: $AUTO_GROUP_SIZE
    groups_per_batch: $AUTO_GROUPS_PER_BATCH
    cli_sandbox: workspace-write
    cli_timeout: $AUTO_CLI_TIMEOUT
    max_concurrent_requests: $AUTO_MAX_CONCURRENT_REQUESTS
YAML

run_config() {
    local run_name="$1"
    "$PYTHON_BIN" repro/run_from_yaml.py \
        --config "$CONFIG_FILE" \
        --run "$run_name" \
        ${DRY_RUN_ARGS[@]+"${DRY_RUN_ARGS[@]}"}
}

start_blackbox_server() {
    if [[ "${RUN_WHEST_DRY_RUN:-0}" == "1" ]]; then
        echo "dry-run: would launch CPU blackbox eval server on socket $BLACKBOX_SOCKET"
        return
    fi

    rm -f "$BLACKBOX_SOCKET"
    mkdir -p "$BLACKBOX_LOG_DIR"
    SERVER_ARGS=(
        --env-type examples.aicrowd_whestbench.env:WhestBenchEnv
        --problem-type arc_whestbench_2026
        --socket "$BLACKBOX_SOCKET"
        --log-dir "$BLACKBOX_LOG_DIR"
        --eval-timeout "$EVAL_TIMEOUT"
        --num-cpus-per-task "$NUM_CPUS_PER_TASK"
        --workers "$VERIFIER_WORKERS"
        --allow-request-state
    )
    if [[ -n "$BLACKBOX_MAX_EVALUATIONS" && "$BLACKBOX_MAX_EVALUATIONS" != "null" ]]; then
        SERVER_ARGS+=(--max-evaluations "$BLACKBOX_MAX_EVALUATIONS")
    fi
    if [[ "$BLACKBOX_CACHE" == "1" ]]; then
        SERVER_ARGS+=(--cache-by-submission)
    fi
    PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -m ttt_discover.eval_runners.blackbox_eval.server \
        "${SERVER_ARGS[@]}" &
    SERVER_PID=$!

    for _ in $(seq 1 120); do
        if [[ -S "$BLACKBOX_SOCKET" ]]; then
            return
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            wait "$SERVER_PID"
            exit 1
        fi
        sleep 0.5
    done

    echo "blackbox eval server did not create socket: $BLACKBOX_SOCKET" >&2
    exit 1
}

case "$TARGET" in
    all)
        run_config ttt_discover
        start_blackbox_server
        run_config autoevolve_blackbox
        ;;
    ttt_discover)
        run_config ttt_discover
        ;;
    autoevolve_blackbox)
        start_blackbox_server
        run_config autoevolve_blackbox
        ;;
    autoevolve_in_process)
        run_config autoevolve_in_process
        ;;
    *)
        echo "Unknown RUN_WHEST_TARGET: $TARGET" >&2
        usage >&2
        exit 2
        ;;
esac
