# WhestBench MLP variance and rank-stability audit

This directory records an independent check of whether the local mini
full-100 MLPs follow the challenge's documented random-network distribution,
and whether changing the evaluated MLP sample can destabilize method rankings.

The consolidated Chinese report covering the fresh-100 rankings, both
fresh-50 splits, per-method variance, and self-evolution promotion thresholds
is
[`fresh100_rank_variance_summary_20260724.md`](fresh100_rank_variance_summary_20260724.md).

## Result

The local MLP parameters are correct:

- architecture: width 256, depth 32, no bias, ReLU;
- weights: independent `N(0, 2/256)`, rounded to float32;
- seed protocol: the weight RNG is
  `SeedSequence(root_seed).spawn(3)[0]`;
- all 100 public-mini rows were regenerated exactly from their stored root
  seeds, with zero mismatched weight elements;
- empirical weight variance over 209,715,200 parameters was
  `0.00781349551`, versus the target `0.0078125` (about `+0.0127%`, or
  `1.30` approximate standard errors);
- adjacent-layer element correlation was `-8.87e-5`, consistent with
  independence.

The deep networks nevertheless have a broad intrinsic distribution. In the
official public-mini ground truth, mean final-neuron activation variance ranges
from `0.01532` to `0.20245` (`13.21x`), with coefficient of variation `0.674`.
In 100 independently generated MLPs, using two independent 8,192-input Monte
Carlo measurements per network, it ranges from `0.00867` to `0.26844`
(`30.95x`), with coefficient of variation `0.791`. The repeat-based reliability
is `0.99968`, so the observed spread is not Monte Carlo noise.

The spread develops with depth even though mean second moment remains close to
one:

| Layer | Across-MLP CV of second moment | q95 / q05 |
|---:|---:|---:|
| 0 | 0.005 | 1.02 |
| 7 | 0.271 | 2.34 |
| 15 | 0.525 | 4.68 |
| 23 | 0.780 | 10.04 |
| 31 | 1.020 | 16.34 |

For the best 15 methods by local full-100 score, 10,000 bootstrap pseudo-suites
of 50 MLPs give:

- mean Spearman correlation with full-100: `0.766`;
- mean pairwise inversion fraction: `0.198`;
- probability of retaining the local top method: `0.461`;
- adjacent local methods invert with probability roughly `0.31` to `0.49`.

Thus close methods are not reliably ordered by one finite suite. However, the
actual hosted public-50 ordering is more extreme than ordinary resampling from
the local empirical MLP population: Spearman is `-0.068`, pairwise inversion is
`0.524`, and the local top method ranks ninth among these 15. Only `0.03%` of
the bootstrap runs had an equally low Spearman correlation. This is evidence
against explaining all observed drift as plain 50-vs-100 sampling noise.
Repeated search and selection on the same full-100 suite (winner's curse) and
method-specific sensitivity to MLP structure are plausible additional causes.

This audit proves that the public-mini generator and stored weights match the
documented protocol. Hosted weights are hidden, so it cannot directly prove
their byte-level provenance.

## Actual estimator rerun on fresh MLPs

The bootstrap result above was followed by a real out-of-sample evaluation on
two disjoint fresh-50 suites, also analyzed as one fresh-100 suite:

- 100 newly generated width-256, depth-32 MLPs with 100 unique root seeds;
- 15 actual estimator sources, with every source hash matched to its recorded
  local full-100 report or submitted archive;
- 8 independently scrambled Gaussian Sobol replicates of 65,536 points per
  MLP (524,288 total);
- official subprocess runner and Phase-1 FLOP budget;
- 1,500 estimator/MLP predictions, with zero failures;
- per-MLP truth-noise correction estimated from variance across Sobol
  scrambles.

The mean estimated truth noise floor over the fresh 100 was `5.05e-8`. Plain
Monte Carlo at the same total sample count would have an expected noise floor
of `9.93e-8`.

Using corrected FLOPs-only adjusted score (to remove concurrent wall-clock
effects), the combined fresh-100 ranking begins:

| Fresh-100 rank | Local full-100 rank | Fresh-100 score |
|---:|---:|---:|
| 1 | 2 | `3.3755e-7` |
| 2 | 6 | `3.4171e-7` |
| 3 | 7 | `3.5009e-7` |
| 4 | 11 | `3.5576e-7` |
| 5 | 10 | `3.6800e-7` |

The former local rank 1 scored `4.0178e-7` and fell to fresh-100 rank 10.
Across all 15 methods:

- fresh-100 versus local full-100 Spearman: `0.414`;
- fresh-100 versus local pairwise inversion fraction: `0.352`;
- fresh-100 versus hosted public-50 Spearman: `0.175`;
- fresh-100 versus hosted pairwise inversion fraction: `0.438`.

The two independent fresh-50 halves also disagree strongly: their Spearman
correlation is `0.161`, and `43.8%` of method pairs invert. This demonstrates
substantial suite sensitivity directly, without comparing against either the
old local suite or the hosted suite.

A 20,000-resample paired bootstrap makes local rank 2 the fresh-100 winner and
gives it `95.5%` probability of beating the old local winner. The old local
winner has `3.02%` probability of placing in the fresh top 3 and `79.8%`
probability of landing in the bottom half.

Complete records and formatted comparisons:

- [`fresh100_comparison_20260723.md`](fresh100_comparison_20260723.md):
  original local full-100, hosted official public-50, and fresh-100;
- [`fresh50_split_comparison_20260723.md`](fresh50_split_comparison_20260723.md):
  fresh-50 A, fresh-50 B, and their combined fresh-100;
- [`fresh100_mlp_estimator_eval_20260723.json`](fresh100_mlp_estimator_eval_20260723.json):
  complete merged per-MLP record.

The runner is
[`evaluate_fresh_mlp_suite.py`](../../../repro/aicrowd_whestbench/evaluate_fresh_mlp_suite.py).

## Per-method MLP variance and promotion thresholds

Across the fresh 100, the per-MLP score standard deviation ranges from
`2.52e-7` to `4.98e-7`, with coefficients of variation from `0.70` to `1.16`.
Consequently, each method's 100-MLP mean has a two-sided 95% half-width of
roughly `0.50e-7` to `0.99e-7`.

Candidate selection should use paired per-MLP differences against the
incumbent. Across all 105 method pairs, the paired two-sided 95% half-width at
100 MLPs has median `0.76e-7` and 90th percentile `1.00e-7`. The corresponding
one-sided promotion thresholds are `0.64e-7` and `0.83e-7`. At 50 MLPs they
rise to `0.91e-7` and `1.19e-7`.

For adaptive self-evolution, the fixed one-candidate threshold is
anti-conservative because many candidates are tried on the same suite. With 15
candidate comparisons, a Bonferroni one-sided threshold has median `1.06e-7`
at 100 MLPs and `1.53e-7` at 50 MLPs. A held-out, unobserved validation suite
is still required to control repeated suite overfitting.

The complete table and pairwise records are
[`fresh100_method_variance_thresholds_20260724.md`](fresh100_method_variance_thresholds_20260724.md)
and
[`fresh100_method_variance_thresholds_20260724.json`](fresh100_method_variance_thresholds_20260724.json).

## Reproduce

From the repository root:

```bash
PYTHONPATH=/tmp/whest-official-deps:$PWD \
  python repro/aicrowd_whestbench/analyze_mlp_variance.py \
    --audit-official-mini \
    --generated-mlps 100 \
    --mc-samples 8192 \
    --mc-repeats 2 \
    --bootstrap-repeats 10000 \
    --top-methods 15 \
    --output \
      repro_external/aicrowd_whestbench/analysis/mlp_variance_rank_stability_20260723.json
```

The dataset audit needs the official `whestbench` dependencies. Independent
generation and report bootstrapping need only NumPy.

To generate the second disjoint fresh-50 suite and rerun all 15 estimators:

```bash
PYTHONPATH=/tmp/whest-official-deps:$PWD \
  python repro/aicrowd_whestbench/evaluate_fresh_mlp_suite.py \
    --n-estimators 15 \
    --n-mlps 50 \
    --mlp-offset 50 \
    --truth-method sobol \
    --truth-samples-per-repeat 65536 \
    --truth-repeats 8 \
    --truth-workers 7 \
    --estimator-workers 5 \
    --baked-cache \
      repro_external/aicrowd_whestbench/analysis/fresh-part2.baked.pkl \
    --checkpoint-dir \
      repro_external/aicrowd_whestbench/analysis/fresh-part2.checkpoints \
    --output \
      repro_external/aicrowd_whestbench/analysis/fresh-part2.json
```

Use
[`merge_fresh_mlp_evals.py`](../../../repro/aicrowd_whestbench/merge_fresh_mlp_evals.py)
to merge the two disjoint parts and recompute scores, ranks, correlations, and
paired bootstrap statistics over all 100 MLPs.
