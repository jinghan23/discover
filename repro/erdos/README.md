# Erdős Minimum Overlap Reproduction Notes

This directory is a thin local reproduction layer on top of the official
TTT-Discover source tree. It keeps the upstream implementation untouched while
making the paper's Erdős minimum overlap task easier to re-run later.

## Source Snapshot

- Upstream repository: `https://github.com/test-time-training/discover`
- Upstream commit used here: `6c40e82dab9d5de7416ac873ad5cd3106084aaed`
- Official task implementation: `examples/erdos_min_overlap/env.py`
- Official reported best sequence: `results/mathematics/ttt_erdos_sequence.json`
- Official reproduction guide: `docs/reproducing.md`

## What Can Be Checked Without Accelerators

The full discovery run uses Tinker and WANDB credentials and will train
`openai/gpt-oss-120b` or `openai/gpt-oss-20b` through the Tinker API. Without
those credentials or allocated compute, run only the local baseline check:

```bash
python repro/erdos/check_baseline.py
```

Expected output for the bundled TTT-Discover sequence:

```text
sequence: results/mathematics/ttt_erdos_sequence.json
n_points: 600
sum(h): 300.000000000000
target sum: 300.000000000000
min(h): 0.000000000000
max(h): 0.999999729118
c5_bound: 0.380875323218
```

This verifies the saved sequence is in `[0, 1]`, has sum `n_points / 2`, and
matches the paper-side computation:

```python
max(np.correlate(h, 1 - h, mode="full")) / len(h) * 2
```

## Environment Setup For A Full Run

The official guide recommends Python 3.11 for non-GPU-mode tasks. One possible
setup is:

```bash
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[math]"
uv pip install -r requirements/requirements-math.txt
```

Then set secrets locally. Do not commit the filled `.env` file:

```bash
cp repro/erdos/.env.example .env
```

Required variables are:

```bash
export HF_TOKEN="..."
export TINKER_API_KEY="..."
export WANDB_API_KEY="..."
export WANDB_ENTITY="..."
```

The Codex no-finetune runner defaults to the local `codex exec` CLI, so it uses
the already logged-in Codex/ChatGPT account and does not require
`OPENAI_API_KEY` or `TINKER_API_KEY`. It still uses the existing tokenizer,
environment, PUCT sampler, evaluator, and logging code, but skips Tinker
training. If you prefer the direct OpenAI Responses API backend, pass
`--codex-backend responses` and set `OPENAI_API_KEY`.

The Codex runner also loads `initial_pool_reference_plus_codex_20260603.json`
by default. Rebuild that task-specific pool with:

```bash
python -m repro.erdos.build_initial_pool
```

The pool contains generic serialized `State` objects for the public loader;
the Erdős-specific SOURCE_SUMMARY parsing and C5 filtering live in
`build_initial_pool.py`.

## Launching The Paper Task

The official entrypoint is:

```bash
python -m examples.erdos_min_overlap.env
```

This repo also includes a configurable wrapper:

```bash
python -m repro.erdos.run_erdos_discovery \
  --experiment-name erdos-min-overlap-repro \
  --model-name openai/gpt-oss-120b \
  --num-epochs 50 \
  --group-size 64 \
  --groups-per-batch 8 \
  --num-cpus-per-task 1 \
  --eval-timeout 1100
```

For a cheaper smoke run after credentials are available:

```bash
python -m repro.erdos.run_erdos_discovery \
  --experiment-name erdos-min-overlap-smoke \
  --model-name openai/gpt-oss-20b \
  --num-epochs 1 \
  --group-size 2 \
  --groups-per-batch 1 \
  --num-cpus-per-task 1 \
  --eval-timeout 120 \
  --codex-max-concurrent-requests 4
```

To run the Codex no-finetune variant:

```bash
python -m repro.erdos.run_erdos_discovery \
  --runner codex_no_finetune \
  --codex-backend cli \
  --experiment-name erdos-min-overlap-codex-smoke \
  --num-epochs 1 \
  --group-size 2 \
  --groups-per-batch 1 \
  --num-cpus-per-task 1 \
  --eval-timeout 120
```

## Metric To Report

Erdős minimum overlap is a minimization task. Per the official reproduction
guide, track the minimum `raw_score` across all steps:

```text
env/all/raw_score/min
```

The target reported in the README is `C5 <= 0.38080`, and the bundled
TTT-Discover sequence evaluates locally to `0.380875323218`.
