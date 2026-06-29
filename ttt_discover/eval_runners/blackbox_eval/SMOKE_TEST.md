# Blackbox Eval Smoke Test

Start the trusted verifier server outside the Codex workspace:

```bash
SOCK=/tmp/ttt_blackbox_eval_trimul.sock
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_CUDA_ARCH_LIST=8.0 \
python -m ttt_discover.eval_runners.blackbox_eval.server \
  --env-type examples.gpu_mode.env:GpuModeEnv \
  --problem-type trimul \
  --socket "$SOCK" \
  --log-dir /tmp/ttt_blackbox_eval_logs \
  --eval-timeout 1200 \
  --num-cpus-per-task 1
```

In another shell, create a fake Codex workspace with only the thin client and a
known top submission:

```bash
SOCK=/tmp/ttt_blackbox_eval_trimul.sock
WORK=/tmp/ttt_blackbox_eval_ws
rm -rf "$WORK"
mkdir -p "$WORK"
python -m ttt_discover.eval_runners.blackbox_eval.client_template \
  --problem-type trimul \
  --socket "$SOCK" \
  --output "$WORK/eval_client.py"
cp GPUMode/top_solutions/trimul/web_top5/submissions/A100/rank1_A100_submission_782275_josusanmartin.py \
  "$WORK/submission.py"
(cd "$WORK" && python eval_client.py)
```

Expected: the client prints `pass`, `raw_score`, and `reward`. The server logs
should show one queued request and a per-request log directory under
`/tmp/ttt_blackbox_eval_logs/`, while the workspace contains no hidden task files.
