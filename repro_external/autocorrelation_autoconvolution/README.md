# Autocorrelation / Autoconvolution Inequalities

Reproduces public AlphaEvolve and local TTT-Discover scores for the first two autocorrelation/autoconvolution inequality tasks.

- AC1 objective: `2 n max(a*a) / sum(a)^2`; lower is better.
- AC2 objective: `||f*f||_2^2 / (||f*f||_1 ||f*f||_inf)`; higher is better.

Sources:

- https://arxiv.org/abs/2506.13131
- https://arxiv.org/abs/2511.02864
- https://github.com/google-deepmind/alphaevolve_repository_of_problems/blob/main/experiments/autocorrelation_problems/autocorrelation_problems.ipynb
- Local TTT-Discover files: `results/mathematics/ttt_ac1_sequence.json`, `results/mathematics/ttt_ac2_sequence.json`

Run:

```bash
python repro_external/autocorrelation_autoconvolution/reproduce_ac.py
```

To additionally download and verify the current public EinsteinArena/ClaudeExplorer AC2 best solution:

```bash
python repro_external/autocorrelation_autoconvolution/reproduce_ac.py --include-online-ac2-best
```

To run the official AlphaEvolve AC1 final evolved search program itself, with a configurable time budget:

```bash
python repro_external/autocorrelation_autoconvolution/run_official_ac1_evolved_search.py --budget 60
```

The script writes `ac_results.json` in this folder.
