--- Diversity Mode (FORCED for this launch) ---
Mode: `exact_radial`

The Phase-1 MLPs are bias-free and apply ReLU to the final layer, so the output
is positively 1-homogeneous: y(r*u) = r*y(u). All radial sampling variance is
therefore removable EXACTLY at zero extra forward cost: for Gaussian samples
x_i with radius R_i = ||x_i||, average y(x_i)/R_i and multiply once by
E[R] = sqrt(2)*Gamma((d+1)/2)/Gamma(d/2) with d = mlp.width. Build this INTO
the German-current structure: keep antithetic pairing and half-sample
covariance whitening folded into the first layer, but delete the chi-squared
radial strata, their ppf computations, and the radial importance weights,
reinvesting the freed FLOPs into more samples. Evidence: sphere-based variants
that also dropped whitening/adaptive budgeting underperformed badly (official
3.355e-7 vs German 2.749e-7), so radialization must augment the German
pipeline, not replace it. Add a cheap numerical homogeneity check in setup as a
guard before relying on the identity.

Hard constraints for this launch:
- The mechanism above MUST be the core change in your candidate estimator.
- Do NOT pivot to another mode's mechanism; hybrid elements are allowed only if
  this mode's mechanism remains the dominant novelty.
- If evidence suggests this mode cannot beat the parent estimator, still return
  your best in-mode candidate rather than silently substituting a different
  family; a clean in-mode attempt is more valuable to the outer search.
- Where this block conflicts with any earlier generic exploration guidance,
  this block wins.
