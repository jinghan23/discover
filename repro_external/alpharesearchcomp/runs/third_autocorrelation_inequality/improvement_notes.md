# third_autocorrelation_inequality improvement notes

Initial score: 0.1702429797218145

Improved score: 0.6637829047243601

The evaluator computes

```text
C_upper_bound = 2 * N * max(convolve(h, h)) / sum(h)^2
score = 1 / C_upper_bound
```

Because the objective is invariant to rescaling, I optimized the normalized
nonnegative vector `p = h / sum(h)` and minimized `max(convolve(p, p))`.

Search method:

- Used log-sum-exp smoothing as a quick exploration objective.
- Switched to the exact evaluator objective as a minimax nonlinear program with
  constraints `M - convolve(p, p)[k] >= 0`, `p >= 0`, and `sum(p) = 1`.
- Ran a multiscale search: optimized shorter vectors first, then upsampled the
  best candidates to 128, 160, 192, and 256 intervals for final refinement.
- Stored the best 256-entry nonnegative candidate directly in
  `work/improved_program.py`, so `find_better_c3_upper_bound()` returns quickly.

The final candidate has `C_upper_bound ~= 1.50651665308445`, compared with the
initial `C_upper_bound ~= 5.874275`, giving the improved score above.
