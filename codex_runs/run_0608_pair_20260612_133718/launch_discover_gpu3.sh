#!/usr/bin/env bash
set -euo pipefail

# Derived from /opt/tiger/discover/repro/gpu_mode/run_0608.sh.
# Source script is left unchanged; this launcher gives the run a fresh log root
# and pins both sequential stages to physical GPU 3.
cd /opt/tiger/discover

LOG_ROOT=/opt/tiger/discover-gpu-kernel-experiments/codex_runs/run_0608_pair_20260612_133718/discover_gpu3
mkdir -p "$LOG_ROOT"

python repro/run_discovery.py \
    --task trimul \
    --runner codex_no_finetune \
    --experiment-name trimul_0608_ttt_discover_discover_20260612_133718_gpu3 \
    --log-root "$LOG_ROOT" \
    --gpu 3 \
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
    --codex-cli-timeout 600 \
    --codex-max-concurrent-requests 4

python repro/run_discovery.py \
    --task trimul \
    --runner codex_no_finetune \
    --experiment-name trimul_0608_codex_autoevolve_discover_20260612_133718_gpu3 \
    --log-root "$LOG_ROOT" \
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
    --codex-autonomous
