#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
GPU="${TTT_GPU:-3}"
SOCK="${TTT_BLACKBOX_EVAL_SOCKET:-/tmp/ttt_blackbox_eval_trimul.sock}"
SERVER_LOG_DIR="${TTT_BLACKBOX_EVAL_LOG_DIR:-/tmp/ttt_blackbox_eval_logs}"
CUDA_DEVICE_ORDER_VALUE="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
TORCH_CUDA_ARCH_LIST_VALUE="${TORCH_CUDA_ARCH_LIST:-8.0}"

DRY_RUN_ARGS=()
if [[ "${RUN_0608_DRY_RUN:-0}" == "1" ]]; then
    DRY_RUN_ARGS+=(--dry-run)
fi

SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ "${RUN_0608_DRY_RUN:-0}" == "1" ]]; then
    echo "dry-run: would launch blackbox eval server on socket $SOCK"
else
    rm -f "$SOCK"
    mkdir -p "$SERVER_LOG_DIR"
    CUDA_VISIBLE_DEVICES="$GPU" \
    CUDA_DEVICE_ORDER="$CUDA_DEVICE_ORDER_VALUE" \
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST_VALUE" \
    "$PYTHON_BIN" -m ttt_discover.blackbox_eval.server \
        --env-type examples.gpu_mode.env:GpuModeEnv \
        --problem-type trimul \
        --socket "$SOCK" \
        --log-dir "$SERVER_LOG_DIR" \
        --eval-timeout 1200 \
        --num-cpus-per-task 1 &
    SERVER_PID=$!

    for _ in $(seq 1 120); do
        if [[ -S "$SOCK" ]]; then
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            wait "$SERVER_PID"
            exit 1
        fi
        sleep 0.5
    done

    if [[ ! -S "$SOCK" ]]; then
        echo "blackbox eval server did not create socket: $SOCK" >&2
        exit 1
    fi
fi

"$PYTHON_BIN" repro/run_discovery.py \
    --task trimul \
    --runner codex_no_finetune \
    --experiment-name trimul_0608_codex_autoevolve_blackbox_gpu3 \
    --log-root codex_runs/trimul_exec_workspaces \
    --gpu "$GPU" \
    --cuda-device-order "$CUDA_DEVICE_ORDER_VALUE" \
    --torch-cuda-arch-list "$TORCH_CUDA_ARCH_LIST_VALUE" \
    --num-epochs 8 \
    --group-size 2 \
    --groups-per-batch 1 \
    --num-cpus-per-task 1 \
    --eval-timeout 1200 \
    --wandb-project "" \
    --codex-backend cli \
    --codex-model-name gpt-5.5 \
    --codex-cli-command codex \
    --codex-cli-sandbox danger-full-access \
    --codex-cli-timeout 7200 \
    --codex-max-concurrent-requests 2 \
    --codex-autonomous \
    --codex-autonomous-blackbox \
    --blackbox-eval-socket "$SOCK" \
    "${DRY_RUN_ARGS[@]}"
