# Cap Set Codex Experiments

This launcher runs autonomous Codex cap-set discovery through the shared
AutoEvolve algorithm. The Python side selects parents with the existing PUCT
sampler, gives Codex a workspace plus evaluator command, and verifies the final
returned `priority(el, n)` with the normal environment.

## Example

```bash
TOKENIZERS_PARALLELISM=false .venv/bin/python -m repro.cap_set.run_cap_set_codex_experiment \
  --experiment-name capset-autoevolve \
  --num-epochs 5 \
  --codex-cli-timeout 1800
```

By default this seeds `initial_pool_400_to_512.json`: 14 verified high-score
`F_3^8` states with sizes `400, 411x3, 412x2, 413, 414x2, 418, 420, 424,
431, 512`.

## High-score initial pool

Regenerate the default 400-to-512 pool from saved cap-set sampler snapshots:

```bash
.venv/bin/python -m repro.cap_set.build_initial_pool --verify
```

## Logs

Each run writes to `tinker_log/<experiment-name>/`:

- `metrics.jsonl`: one row per outer step.
- `agent_outputs.jsonl`: final Codex responses and verified rewards.
- `autoevolve_workspaces/`: per-call workspaces with `candidate.py`,
  `prompt.txt`, `eval_tmp/`, and any `best_priority.py` Codex writes.
- `puct_sampler_step_*.json`: sampler pool snapshots.
