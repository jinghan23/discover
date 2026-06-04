# Autoconvolution Peak Minimization Improvement Notes

## Result

- Initial program: `K=128` cosine-squared candidate.
- Initial evaluator result: `score=0.6614583333333335`, so `mu_inf=1.5118110236220468`.
- Improved program: fixed `K=256` nonnegative step-height vector.
- Improved evaluator result: `score=1.3255744971107202`, so `mu_inf=0.7543898907074958`.

## Method

The discrete evaluator is equivalent to optimizing a probability vector `q`
with `sum(q)=1` under the objective:

```text
mu_inf = K * max(convolve(q, q))
```

I searched directly in that representation. The search used a smooth
log-sum-exp approximation to the maximum convolution bin, optimized over
softmax variables with L-BFGS-B. Candidates were refined by coarse-to-fine
interpolation over grids `64 -> 96 -> 128 -> 192 -> 256`, then polished with
an epigraph-style SLSQP pass on the exact max-bin constraints.

The final submitted program stores the best 256-step vector directly, so
`main()` returns immediately and does not depend on SciPy or runtime search.

## Evaluation

Formal evaluation was run with:

```bash
python /Users/bytedance/Documents/workspace/ttt-discover/repro_external/alpharesearchcomp/runs/_common/run_program_eval.py autoconvolution_peak_minimization work/improved_program.py
```

This wrote `improved_eval.json`.
