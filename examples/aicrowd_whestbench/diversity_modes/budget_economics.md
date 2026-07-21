--- Diversity Mode (FORCED for this launch) ---
Mode: `budget_economics`

Keep the German-current sampler but optimize per-MLP compute economics. The
per-MLP score is final_mse * max(0.1, C/B), so the optimal compute fraction
differs per network and the current estimator only switches between 0.10 and
0.155 on a hard entropy-rank@L8 threshold (40.43). Recipes: (a) two-stage
sampling — a small pilot (~512 samples, counted in C) estimates per-network
variance, then pick the fraction on a grid ~[0.08, 0.30] minimizing predicted
(sigma^2 / K(frac) + floor) * max(0.1, effective fraction), modeling fixed
costs (whitening, probes) explicitly; (b) parity routing — compute
V_odd/V_even from the antithetic pilot pairs at zero extra cost and disable
antithetic (independent draws instead of mirrored pairs) on even-dominated
networks where mirroring wastes half the budget; (c) replace the hard
threshold with a smooth monotone map from probe statistics to fraction.
Evidence: the real budget sweep bottomed near fraction 0.13 and 0.10 scored
WORSE than 0.13 — fixed costs matter. Keep decision rules simple (<= 3
tunable constants, monotone) to avoid overfitting the local mini suite.

Hard constraints for this launch:
- The mechanism above MUST be the core change in your candidate estimator.
- Do NOT pivot to another mode's mechanism; hybrid elements are allowed only if
  this mode's mechanism remains the dominant novelty.
- If evidence suggests this mode cannot beat the parent estimator, still return
  your best in-mode candidate rather than silently substituting a different
  family; a clean in-mode attempt is more valuable to the outer search.
- Where this block conflicts with any earlier generic exploration guidance,
  this block wins.
