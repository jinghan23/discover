# Autonomous Blackbox Smoke Test

Start the trusted GPUMode verifier server:

```bash
SOCK=/tmp/ttt_blackbox_eval_trimul.sock
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_CUDA_ARCH_LIST=8.0 \
.venv/bin/python -m ttt_discover.blackbox_eval.server \
  --env-type examples.gpu_mode.env:GpuModeEnv \
  --problem-type trimul \
  --socket "$SOCK" \
  --log-dir /tmp/ttt_blackbox_eval_logs \
  --eval-timeout 1200 \
  --num-cpus-per-task 1
```

In another shell, run the `run_0608.sh` AutoEvolve path with blackbox enabled:

```bash
PYTHON=.venv/bin/python \
RUN_0608_ONLY_AUTONOMOUS=1 \
RUN_0608_DRY_RUN=1 \
TTT_BLACKBOX_EVAL_SOCKET=/tmp/ttt_blackbox_eval_trimul.sock \
bash repro/gpu_mode/run_0608.sh
```

For a shorter one-sample check using the same autonomous blackbox path:

```bash
SOCK=/tmp/ttt_blackbox_eval_trimul.sock
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_CUDA_ARCH_LIST=8.0 \
.venv/bin/python repro/run_discovery.py \
  --task trimul \
  --runner codex_no_finetune \
  --experiment-name trimul_blackbox_autonomous_smoke \
  --log-root /tmp/ttt_blackbox_autonomous_smoke \
  --gpu 0 \
  --cuda-device-order PCI_BUS_ID \
  --torch-cuda-arch-list 8.0 \
  --num-epochs 1 \
  --group-size 1 \
  --groups-per-batch 1 \
  --num-cpus-per-task 1 \
  --eval-timeout 1200 \
  --wandb-project "" \
  --codex-backend cli \
  --codex-model-name gpt-5.5 \
  --codex-cli-command codex \
  --codex-cli-sandbox danger-full-access \
  --codex-cli-timeout 7200 \
  --codex-max-concurrent-requests 1 \
  --codex-autonomous \
  --codex-autonomous-blackbox \
  --blackbox-eval-socket "$SOCK"
```

Expected workspace contents for each autonomous call include `submission.py`,
`eval_client.py`, `prompt.txt`, `eval_tmp/`, and Codex logs or isolation
metadata. It should not contain `task.yml`, `eval.py`, `reference.py`, `task.py`,
or `libkernelbot/`. The server log should show queued requests and compact
pass/fail/score responses.
