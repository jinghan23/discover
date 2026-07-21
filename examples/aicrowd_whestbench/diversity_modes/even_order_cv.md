--- Diversity Mode (FORCED for this launch) ---
Mode: `even_order_cv`

Grow the cross-fitted control-variate family on top of German current. After
antithetic pairing (kills all odd harmonics) and whitening (fixes second
moments), the residual error is even-order >= 4; the cross-fitted Hermite-H4
control variate is the current best official score (2.745990e-7). Extend with
even structure only: H6 features, low-rank quadratics (a_j^T x)^2 - ||a_j||^2
(exact zero mean under N(0,I)), and degree-4/6 zonal polynomials of (a_j^T x)
along data-driven directions a_j (top right singular vectors of the first
layer, or pilot Jacobian directions). Keep total features <= 128, fit
multi-output coefficients by block cross-fitting (fit on one half, apply on
the other, swap) so the estimator stays unbiased, and shrink coefficients
toward zero. Every feature's exact mean under the ACTUAL sampling distribution
must be known analytically — mind interactions with radial stratification and
whitening. Do NOT attempt full second-order spherical-harmonic CV (32895
features, infeasible) and do NOT re-attempt first-layer-only analytic ReLU
CVs (+3.06%, already lost).

Hard constraints for this launch:
- The mechanism above MUST be the core change in your candidate estimator.
- Do NOT pivot to another mode's mechanism; hybrid elements are allowed only if
  this mode's mechanism remains the dominant novelty.
- If evidence suggests this mode cannot beat the parent estimator, still return
  your best in-mode candidate rather than silently substituting a different
  family; a clean in-mode attempt is more valuable to the outer search.
- Where this block conflicts with any earlier generic exploration guidance,
  this block wins.
