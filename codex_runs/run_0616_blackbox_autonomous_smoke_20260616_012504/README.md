# Autonomous Blackbox Smoke Trajectory

Source run:

- `/tmp/ttt_blackbox_autonomous_smoke/trimul_blackbox_autonomous_smoke`

Command shape:

```bash
python repro/run_discovery.py \
  --task trimul \
  --runner codex_no_finetune \
  --experiment-name trimul_blackbox_autonomous_smoke \
  --log-root /tmp/ttt_blackbox_autonomous_smoke \
  --gpu 0 \
  --num-epochs 1 \
  --group-size 1 \
  --groups-per-batch 1 \
  --codex-backend cli \
  --codex-model-name gpt-5.5 \
  --codex-autonomous \
  --codex-autonomous-blackbox \
  --blackbox-eval-socket /tmp/ttt_blackbox_eval_trimul.sock
```

Result:

- `codex/total_samples`: 1
- `codex/kept_samples`: 1
- `codex/correct_samples`: 1
- `correctness`: 1.0
- `raw_score`: 3664.7516723731055
- `reward`: 0.4093046771239146
- Blackbox evaluator requests during the autonomous call: 84

Notes:

- `*_codex_home/` was intentionally excluded because it contains local Codex auth and cache state.
- The copied workspace intentionally contains `submission.py`, `eval_client.py`, `prompt.txt`, `eval_tmp/`, Codex logs, and isolation metadata.
- The workspace did not contain hidden task files such as `task.yml`, `eval.py`, `reference.py`, `task.py`, or `libkernelbot/`.
