#!/usr/bin/env bash
set -o pipefail

cd /opt/tiger/discover-gpu-kernel-experiments

SESSION_NAME=ttt_discover_0615_gpu0_proxy
LAUNCHER=codex_runs/run_0615_ttt_discover_proxy_20260615_170650/launch_ttt_discover_gpu0_proxy.sh
LOG=codex_runs/run_0615_ttt_discover_proxy_20260615_170650/terminal_ttt_discover_gpu0_proxy.log

export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=.byted.org
export PYTHONUNBUFFERED=1

{
    printf 'tmux session: %s\n' "$SESSION_NAME"
    printf 'command: bash %s\n' "$LAUNCHER"
    printf 'started: %s\n' "$(date --iso-8601=seconds)"
    printf 'http_proxy=%s\n' "$http_proxy"
    printf 'https_proxy=%s\n' "$https_proxy"
    printf 'no_proxy=%s\n' "$no_proxy"
} > "$LOG"

bash "$LAUNCHER" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}

printf 'launcher exited with status %s at %s\n' "$status" "$(date --iso-8601=seconds)" | tee -a "$LOG"
exit "$status"
