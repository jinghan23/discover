#!/usr/bin/env bash
set -euo pipefail

# Fresh restart from step 0, using the current repository logic.
cd /opt/tiger/discover-gpu-kernel-experiments

export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=.byted.org
export PYTHONUNBUFFERED=1

LOG_ROOT=/opt/tiger/discover-gpu-kernel-experiments/codex_runs/run_0616_ttt_discover_reward_shaping_restart_20260616_170538/gpu1
mkdir -p "$LOG_ROOT"

python repro/run_discovery.py \
    --task trimul \
    --runner codex_no_finetune \
    --experiment-name trimul_0616_ttt_discover_reward_shaping_restart_20260616_170538_gpu1 \
    --log-root "$LOG_ROOT" \
    --gpu 1 \
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
    --reward-shaping
