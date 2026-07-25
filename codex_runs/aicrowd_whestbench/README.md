# WhestBench search trajectory snapshots

This directory contains curated, point-in-time records of completed discovery
runs.  The local run roots are much larger than the Git snapshot because every
autonomous call also creates a disposable Codex home, evaluation workspace,
dependency cache, and repeated pool snapshots.

## 2026-07-21 snapshot

The committed snapshot includes:

- all 18 completed runs under `multi_initial_20260718/`;
- the six completed `*_ttt` runs under
  `multi_initial_full100_20260720/`;
- the twelve completed `*_auto_r*` runs under
  `multi_initial_full100_20260720/`;
- run-level `metrics.jsonl`, `agent_outputs.jsonl`, score/state streams, and the
  final cumulative pool snapshot;
- model-call prompts, stdout/stderr transcripts, final responses, commands,
  sanitized environment descriptions, and final `submission.py` files;
- initial estimators, completed-run console logs, and submission review logs.

The snapshot intentionally excludes `*_codex_home/`, `eval_tmp/`, Python
caches, dependency/plugin copies, hourly review cycle workspaces, and all but
the latest cumulative `puct_sampler_step_*.json` or
`autoevolve_pool_step_*.json` file per completed run.  The retained JSONL
streams preserve the per-step trajectory; earlier cumulative pool files would
mostly duplicate the final snapshot.

Three autonomous `codex.stderr.log` transcripts that printed a host credential
during process inspection are also excluded.  Their neighboring prompt,
stdout, final response, command metadata, and `submission.py` remain in the
snapshot; the unredacted transcripts remain local and ignored.

## 2026-07-24 baseline-gate snapshot

The `whestbench_baseline_gate_epoch1_20260724/` snapshot records the two
parallel AutoEvolve calls used to validate the 50-MLP test gate followed by the
disjoint 50-MLP holdout gate.  It retains the run-level metric, score, agent
output, and state streams; the final cumulative pool snapshot; and each call's
prompt, command, sanitized environment, stdout/stderr transcript, timeout
marker, and final `submission.py`.

As above, the snapshot excludes the disposable `*_codex_home/` trees,
generated evaluator clients, Python caches, and the earlier cumulative pool
snapshot.  The retained call files were scanned for credential-like strings
before being added.
