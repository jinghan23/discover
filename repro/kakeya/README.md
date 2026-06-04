# Kakeya TTT-Discover Wrapper

This wraps the finite-field Kakeya construction task as a TTT-Discover
environment.

The model must return one Python code block defining:

```python
def search_for_best_construction(p, d):
    ...
```

The evaluator tests `d=3` and a configurable list of primes. It verifies that
the returned point set contains a full affine line in every projective direction
of `F_p^3`. Lower average density `|K| / p^3` is better.

## Smoke Checks

```bash
.venv/bin/python -m py_compile examples/kakeya/env.py repro/kakeya/run_kakeya_discovery.py
.venv/bin/python repro/kakeya/run_kakeya_discovery.py --help
```

## Run

Small Codex no-finetune run:

```bash
.venv/bin/python repro/kakeya/run_kakeya_discovery.py \
  --runner codex_no_finetune \
  --primes 5,7,13 \
  --num-epochs 10 \
  --group-size 1 \
  --groups-per-batch 1 \
  --wandb-project ''
```

Seed with an existing construction:

```bash
.venv/bin/python repro/kakeya/run_kakeya_discovery.py \
  --runner codex_no_finetune \
  --codex-initial-program repro_external/kakeya_construction/idea_reconstruction/exp3_reconstructed.py \
  --primes 5,7,13 \
  --wandb-project ''
```

