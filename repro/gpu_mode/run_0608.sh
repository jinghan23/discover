#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash repro/gpu_mode/run_0608.sh [all|ttt|autoevolve|blackbox] [--dry-run]

Environment overrides:
  TTT_GPU=4                         GPU id for both discovery and blackbox eval
  RUN_0608_SMOKE=1                  Run one epoch/sample for quick checks
  RUN_0608_DRY_RUN=1                Print resolved configs without launching
  RUN_0608_LOG_ROOT=...             Log root; run logs go under <root>/<experiment>
  RUN_0608_TTT_NUM_EPOCHS=...       Override TTT Discover epochs
  RUN_0608_AUTO_NUM_EPOCHS=...      Override AutoEvolve epochs
  TTT_BLACKBOX_EVAL_SOCKET=...      Unix socket for the blackbox eval server
  TTT_BLACKBOX_EVAL_LOG_DIR=...     Log directory for blackbox eval requests
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TARGET="${RUN_0608_TARGET:-all}"
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
        --dry-run)
            RUN_0608_DRY_RUN=1
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

if [[ "${RUN_0608_ONLY_AUTONOMOUS:-0}" == "1" ]]; then
    TARGET="autoevolve_blackbox"
fi
if [[ "${RUN_0608_ONLY_TTT:-0}" == "1" ]]; then
    TARGET="ttt_discover"
fi

PYTHON_BIN="${PYTHON:-python}"
GPU="${TTT_GPU:-4}"
LOG_ROOT="${RUN_0608_LOG_ROOT:-codex_runs/trimul_exec_workspaces}"
CUDA_DEVICE_ORDER_VALUE="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
TORCH_CUDA_ARCH_LIST_VALUE="${TORCH_CUDA_ARCH_LIST:-8.0}"
WANDB_PROJECT="${RUN_0608_WANDB_PROJECT:-}"
CODEX_MODEL="${RUN_0608_CODEX_MODEL:-gpt-5.6-sol}"
CODEX_COMMAND="${RUN_0608_CODEX_COMMAND:-codex}"
NUM_CPUS_PER_TASK="${RUN_0608_NUM_CPUS_PER_TASK:-1}"
EVAL_TIMEOUT="${RUN_0608_EVAL_TIMEOUT:-1200}"

if [[ "${RUN_0608_SMOKE:-0}" == "1" ]]; then
    RUN_0608_TTT_NUM_EPOCHS="${RUN_0608_TTT_NUM_EPOCHS:-1}"
    RUN_0608_TTT_GROUP_SIZE="${RUN_0608_TTT_GROUP_SIZE:-1}"
    RUN_0608_TTT_GROUPS_PER_BATCH="${RUN_0608_TTT_GROUPS_PER_BATCH:-1}"
    RUN_0608_TTT_MAX_CONCURRENT_REQUESTS="${RUN_0608_TTT_MAX_CONCURRENT_REQUESTS:-1}"
    RUN_0608_AUTO_NUM_EPOCHS="${RUN_0608_AUTO_NUM_EPOCHS:-1}"
    RUN_0608_AUTO_GROUP_SIZE="${RUN_0608_AUTO_GROUP_SIZE:-1}"
    RUN_0608_AUTO_GROUPS_PER_BATCH="${RUN_0608_AUTO_GROUPS_PER_BATCH:-1}"
    RUN_0608_AUTO_MAX_CONCURRENT_REQUESTS="${RUN_0608_AUTO_MAX_CONCURRENT_REQUESTS:-1}"
fi

TTT_EXPERIMENT_NAME="${RUN_0608_TTT_EXPERIMENT_NAME:-trimul_0608_ttt_discover_gpu${GPU}}"
TTT_NUM_EPOCHS="${RUN_0608_TTT_NUM_EPOCHS:-50}"
TTT_GROUP_SIZE="${RUN_0608_TTT_GROUP_SIZE:-4}"
TTT_GROUPS_PER_BATCH="${RUN_0608_TTT_GROUPS_PER_BATCH:-1}"
TTT_CLI_TIMEOUT="${RUN_0608_TTT_CLI_TIMEOUT:-600}"
TTT_MAX_CONCURRENT_REQUESTS="${RUN_0608_TTT_MAX_CONCURRENT_REQUESTS:-4}"

AUTO_EXPERIMENT_NAME="${RUN_0608_AUTO_EXPERIMENT_NAME:-trimul_0608_codex_autoevolve_blackbox_gpu${GPU}}"
AUTO_NUM_EPOCHS="${RUN_0608_AUTO_NUM_EPOCHS:-20}"
AUTO_GROUP_SIZE="${RUN_0608_AUTO_GROUP_SIZE:-2}"
AUTO_GROUPS_PER_BATCH="${RUN_0608_AUTO_GROUPS_PER_BATCH:-1}"
AUTO_CLI_TIMEOUT="${RUN_0608_AUTO_CLI_TIMEOUT:-7200}"
AUTO_MAX_CONCURRENT_REQUESTS="${RUN_0608_AUTO_MAX_CONCURRENT_REQUESTS:-2}"
BLACKBOX_SOCKET="${TTT_BLACKBOX_EVAL_SOCKET:-/tmp/ttt_blackbox_eval_trimul_gpu${GPU}.sock}"
BLACKBOX_LOG_DIR="${TTT_BLACKBOX_EVAL_LOG_DIR:-/tmp/ttt_blackbox_eval_trimul_gpu${GPU}_logs}"

DRY_RUN_ARGS=()
if [[ "${RUN_0608_DRY_RUN:-0}" == "1" ]]; then
    DRY_RUN_ARGS+=(--dry-run)
fi

CONFIG_FILE="$(mktemp /tmp/run_0608_refactor.XXXXXX.yaml)"
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
  env_type: examples.gpu_mode.env:GpuModeEnv
  problem_type: trimul
  backend: cli
  model_name: "$CODEX_MODEL"
  cli_command: "$CODEX_COMMAND"
  num_cpus_per_task: $NUM_CPUS_PER_TASK
  eval_timeout: $EVAL_TIMEOUT
  wandb_project: "$WANDB_PROJECT"
  remove_constant_reward_groups: true
  env:
    CUDA_VISIBLE_DEVICES: "$GPU"
    CUDA_DEVICE_ORDER: "$CUDA_DEVICE_ORDER_VALUE"
    TORCH_CUDA_ARCH_LIST: "$TORCH_CUDA_ARCH_LIST_VALUE"
    PYTHONUNBUFFERED: "1"

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
    group_size: $AUTO_GROUP_SIZE
    groups_per_batch: $AUTO_GROUPS_PER_BATCH
    cli_sandbox: danger-full-access
    cli_timeout: $AUTO_CLI_TIMEOUT
    max_concurrent_requests: $AUTO_MAX_CONCURRENT_REQUESTS
    blackbox_eval_socket: "$BLACKBOX_SOCKET"
YAML

run_config() {
    local run_name="$1"
    "$PYTHON_BIN" repro/run_from_yaml.py \
        --config "$CONFIG_FILE" \
        --run "$run_name" \
        "${DRY_RUN_ARGS[@]}"
}

start_blackbox_server() {
    if [[ "${RUN_0608_DRY_RUN:-0}" == "1" ]]; then
        echo "dry-run: would launch blackbox eval server on socket $BLACKBOX_SOCKET"
        return
    fi

    rm -f "$BLACKBOX_SOCKET"
    mkdir -p "$BLACKBOX_LOG_DIR"
    CUDA_VISIBLE_DEVICES="$GPU" \
    CUDA_DEVICE_ORDER="$CUDA_DEVICE_ORDER_VALUE" \
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST_VALUE" \
    PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -m ttt_discover.eval_runners.blackbox_eval.server \
        --env-type examples.gpu_mode.env:GpuModeEnv \
        --problem-type trimul \
        --socket "$BLACKBOX_SOCKET" \
        --log-dir "$BLACKBOX_LOG_DIR" \
        --eval-timeout "$EVAL_TIMEOUT" \
        --num-cpus-per-task "$NUM_CPUS_PER_TASK" \
        --allow-request-state &
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
    *)
        echo "Unknown RUN_0608_TARGET: $TARGET" >&2
        usage >&2
        exit 2
        ;;
esac
