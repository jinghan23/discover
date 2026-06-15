#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
DRY_RUN_ARGS=()
if [[ "${RUN_0608_DRY_RUN:-0}" == "1" ]]; then
    DRY_RUN_ARGS+=(--dry-run)
fi

# TTT Discover: non-auto, many short Codex samples driven by the sampler/evaluator.
# groups-per-batch = parent states sampled per outer round.
# group-size = Codex samples per parent; product is samples/evals per round.
if [[ "${RUN_0608_ONLY_AUTONOMOUS:-0}" != "1" ]]; then
"$PYTHON_BIN" repro/run_discovery.py \
    --task trimul \
    --runner codex_no_finetune \
    --experiment-name trimul_0608_ttt_discover_gpu2 \
    --log-root codex_runs/trimul_exec_workspaces \
    --gpu 2 \
    --cuda-device-order PCI_BUS_ID \
    --torch-cuda-arch-list 8.0 \
    --num-epochs 50 \
    --group-size 8 \
    --groups-per-batch 1 \
    --num-cpus-per-task 1 \
    --eval-timeout 1200 \
    --wandb-project "" \
    --codex-backend cli \
    --codex-model-name gpt-5.5 \
    --codex-cli-command codex \
    --codex-cli-sandbox read-only \
    --codex-cli-timeout 1200 \
    --codex-max-concurrent-requests 4 \
    "${DRY_RUN_ARGS[@]}"
fi

# Codex AutoEvolve: autonomous deep dive, one writable workspace per sample.
# For GPUMode, danger-full-access is wrapped by the completer in an external
# unshare mount namespace: Codex only sees the sample workspace, while CUDA
# remains visible.
# Add RUN_0608_DRY_RUN=1 to validate command plumbing without launching Codex.
# groups-per-batch = parent states sampled per outer round.
# group-size = Codex samples per parent; product is samples/evals per round.
"$PYTHON_BIN" repro/run_discovery.py \
    --task trimul \
    --runner codex_no_finetune \
    --experiment-name trimul_0608_codex_autoevolve_gpu2 \
    --log-root codex_runs/trimul_exec_workspaces \
    --gpu 3 \
    --cuda-device-order PCI_BUS_ID \
    --torch-cuda-arch-list 8.0 \
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
    "${DRY_RUN_ARGS[@]}"
