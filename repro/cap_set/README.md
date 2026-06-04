# Cap-Set Priority Discovery

This task wraps the FunSearch-style cap-set greedy solver as a TTT-Discover
`codex_no_finetune` problem. The model edits only `priority(el, n)`.

```bash
TOKENIZERS_PARALLELISM=false .venv/bin/python -m repro.cap_set.run_cap_set_discovery \
  --runner codex_no_finetune \
  --codex-backend cli \
  --experiment-name capset-codex-10 \
  --wandb-project '' \
  --num-epochs 10 \
  --group-size 1 \
  --groups-per-batch 1 \
  --num-cpus-per-task 1 \
  --eval-timeout 45 \
  --codex-max-concurrent-requests 1 \
  --codex-cli-timeout 900
```

By default the runner seeds the sampler with
`initial_pool_400_to_512.json`: 14 verified high-score `F_3^8` states with
sizes `400, 411x3, 412x2, 413, 414x2, 418, 420, 424, 431, 512`.
