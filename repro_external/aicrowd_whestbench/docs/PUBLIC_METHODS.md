# Public WhestBench Methods

Verified public material for the AIcrowd ARC White-Box Estimation Challenge
2026, Phase 1. This record was checked on 2026-07-14. Lower adjusted score is
better. Phase 1 uses width-256, depth-32 ReLU MLPs and a `2.72e11` effective
FLOP budget per MLP.

The leaderboard does not publish uploaded submission archives. A method is
called an open submission below only when public executable estimator code can
be tied to a concrete AIcrowd submission ID.

## Open submission: German Alfaro, #311690

- Participant: `german_alfaro`
- Public repository:
  <https://github.com/galfaroi/Can-You-Predict-a-Network-Without-Running-It->
- Repository-reported public score: `3.357e-7`
- Current leaderboard caveat: the participant's later leaderboard result is
  better than `#311690`, so the repository does not reproduce the participant's
  current best submission.
- Contents: production `estimator.py`, experiment log, diagnostics, tests, and
  technical report.

The estimator is a sampling method with structural variance reduction:

1. Draw half of the samples and concatenate their negatives to form
   antithetic pairs.
2. Estimate the input covariance from the independent half-sample.
3. Whiten the samples so their empirical covariance matches the identity.
4. Fold the whitening transform into the first-layer weights instead of
   charging an extra transform for every sample.
5. Run batched forward propagation through the MLP and average the final-layer
   activations.
6. Return zeros for unscored earlier layers when optimizing only the official
   final-layer metric.

The repository calls its covariance shortcut "half-covariance": the covariance
is calculated from one side of an antithetic batch because the negative half
has the same second moment. This is the primary public implementation to
reproduce in this repository.

Local official-mini reproduction: `3.834192e-7` adjusted over 100 Phase 1
MLPs, versus the upstream `3.357e-7` over its 50-MLP public split (`+14.2%`).
The run completed 100/100 networks with `4.228989e10` mean effective compute.

## Public write-up: pscamillo, #314331

- Write-up:
  <https://discourse.aicrowd.com/t/phase-1-write-up-characterizing-a-systematic-scale-bias-in-the-gaussian-closure-estimator-submission-314331/18063>
- Reported adjusted score: `2.45e-6`
- Source status: no public executable estimator was found.

This is a second-order Gaussian-closure estimator with a fitted scalar
correction. The reported final-layer multiplicative correction is about
`0.992`; the author reports roughly a 3x final-layer MSE reduction over the
uncorrected closure. The write-up also records negative results for third
cumulants, low-rank corrections, recurrence, Edgeworth corrections, and learned
nonlinear correctors at depth 32.

Local official-mini reproduction: `2.574242e-6` adjusted and
`2.574242e-5` final-layer MSE over 100 Phase 1 MLPs, versus the reported
`2.45e-6` (`+5.1%`). It completed 100/100 networks with `2.026487e9` mean
effective compute. The same write-up's local-harness value was `2.66e-6`.

## Public method: evaaaz RQMC and Rao-Blackwellization

- Method description:
  <https://discourse.aicrowd.com/t/unbiased-randomized-qmc-rao-blackwell-for-post-relu-activation-means-method-unbiasedness-proof-and-where-the-frontier-is/18053>
- Reported mini adjusted score: about `4.10e-7`
- Reported compute utilization: about `0.42`
- Source status: no public executable estimator or concrete submission ID was
  found.

The estimator maps a randomly shifted rank-1 lattice through the inverse normal
CDF, then performs the normal MLP forward pass. The random shift makes each
lattice point marginally uniform before the inverse-CDF transform. The first
post-ReLU layer mean is replaced by its exact Gaussian expectation,
`s / sqrt(2*pi)`, which is a Rao-Blackwellized diagnostic improvement. The
author reports that randomized QMC supplies the material final-layer gain, while
more elaborate control variates do not help at depth 32.

Local official-mini reproduction: `4.885414e-7` adjusted and
`1.153299e-6` final-layer MSE over 100 Phase 1 MLPs, versus the reported
approximately `4.10e-7` (`+19.2%`). It used 26,858 lattice points and one
random shift per MLP, completed 100/100 networks, and measured
`C/B=0.423625`, close to the reported `0.42`.

## Public independent implementation: cumulant propagation

- Repository:
  <https://github.com/ascender1729/whestbench-cumulant-propagation>
- Upstream ARC method:
  <https://github.com/alignment-research-center/mlp_cumulant_propagation>
- Source status: complete `estimator.py`, but not tied to a current Phase 1
  leaderboard result.
- Applicability caveat: its documented results target the earlier depth-8
  configuration, not the current depth-32 Phase 1 target.

This is a NumPy/flopscope port of ARC's cumulant-propagation estimator. It
propagates factored symmetric cumulant tensors through ReLU layers. It is useful
as a mechanistic reference and for hybrid experiments, but its published score
must not be compared directly with the current Phase 1 leaderboard.

The repository's current warm-up report gives `6.65e-7` for k=3 plus a fitted
k4 sequence extrapolation. After adapting its backend to flopscope 0.8.0rc5,
the local official warm-up mini-100 result is `4.975772e-7` (`-25.2%`), with
100/100 networks completed and `1.750546e10` tracked FLOPs per network. On the
current depth-32 Phase 1 suite, the real k=3 path exceeds the official 30-second
per-predict timeout on the first network; it therefore has no valid Phase 1
full-mini score.

## Official open baselines

- Starter kit: <https://github.com/AIcrowd/whest-starterkit>
- Evaluator and dataset tooling: <https://github.com/AIcrowd/whestbench>

The starter kit includes Monte Carlo, mean propagation, and covariance
propagation examples. These are public baselines, not high-ranking participant
submissions.

## Local reproduction policy

Use the official Phase 1 mini dataset, official subprocess runner, and official
score report. Keep public-source reproductions separate from discovered
estimators. A reproduction is accepted only when:

1. repository tests pass;
2. all 100 mini MLPs complete without fallback;
3. effective compute stays below `2.72e11` for every MLP; and
4. the full-mini adjusted score is in the same broad performance band as the
   public claim, with dataset and evaluator differences explicitly reported.
