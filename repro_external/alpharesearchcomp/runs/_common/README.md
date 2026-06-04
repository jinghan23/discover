# AlphaResearchComp Run Workspace

This directory keeps local reproduction runs separate from the public source.

Layout:

- `../<problem>/source/`: copied public benchmark files, kept as a local reference.
- `../<problem>/work/`: runnable/editable files for local evaluation or evolution.
- `../<problem>/initial_eval.json`: result from evaluating `work/initial_program.py`.
- `../<problem>/evolve_agent_output/`: checkpoints, logs, and best programs from EvolveAgent.

Useful commands:

```bash
python runs/_common/run_initial_eval.py --all
python runs/_common/run_initial_eval.py MSTD littlewood_polynomials
python runs/_common/run_evolve.py MSTD --iterations 5
```

Notes:

- The public `_source` tree is intentionally left untouched.
- Some public evaluators have import-time `print(evaluate())` calls or hard-coded `/data/...`
  paths. The copies under each `work/` directory are patched for local runs.
- EvolveAgent requires API/model configuration and may need optional dependencies from the
  public repo. Initial evaluation is intentionally independent of that.
