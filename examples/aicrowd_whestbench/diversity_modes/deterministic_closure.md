--- Diversity Mode (FORCED for this launch) ---
Mode: `deterministic_closure`

Pursue a deterministic (or near-deterministic) moment-propagation estimator
that operates at the multiplier floor: below C/B = 0.1 the score multiplier is
pinned at 0.1, so a closure whose final-layer MSE reaches ~2.7e-6 beats the
sampling frontier outright while spending almost no compute. Recipes:
diagonal-plus-low-rank covariance propagation through exact Gaussian ReLU
moment formulas; strictly rank-capped factored third-cumulant corrections with
per-layer FLOP metering (the full k=3 path exceeds the 30-second predict
timeout at depth 32 — meter and cap aggressively); a per-depth scalar
calibration profile (<= 4 constants) applied to propagated moments; optionally
a tiny MC probe (~256 samples) used only to SELECT among 2-3 calibration
profiles per network, never to correct the mean directly (the probe is too
noisy for that). Evidence: plain Gaussian closure with one scalar correction
sits at ~2.57e-5 MSE (10x short of the target); Edgeworth corrections,
learned nonlinear correctors, and recurrence all failed at depth 32 — do not
repeat them unchanged.

Hard constraints for this launch:
- The mechanism above MUST be the core change in your candidate estimator.
- Do NOT pivot to another mode's mechanism; hybrid elements are allowed only if
  this mode's mechanism remains the dominant novelty.
- If evidence suggests this mode cannot beat the parent estimator, still return
  your best in-mode candidate rather than silently substituting a different
  family; a clean in-mode attempt is more valuable to the outer search.
- Where this block conflicts with any earlier generic exploration guidance,
  this block wins.
