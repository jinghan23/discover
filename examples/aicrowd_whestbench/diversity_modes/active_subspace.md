--- Diversity Mode (FORCED for this launch) ---
Mode: `active_subspace`

Concentrate sampling effort on the dominant input subspace via CONDITIONAL
sampling, not reweighting. Recipe: spend a small pilot (< 2% of the FLOP
budget, honestly accounted) estimating C = E[J^T J] with 32-64
vector-Jacobian probes through the ReLU masks; take the top r <= 8
eigenvectors A; decompose x = A z + A_perp w with z ~ N(0, I_r); stratify or
low-dimensional-RQMC the z component (r-dimensional low-discrepancy points are
exactly the regime where QMC provably helps) while w stays iid or antithetic;
keep German whitening on the full x. Gate per MLP: engage the subspace path
only when top-r explained energy passes a threshold (~0.7-0.85), otherwise
fall back verbatim to the German-current path so diffuse networks are never
hurt. Evidence: an active-subspace LHS/RQMC REWEIGHTING attempt lost narrowly
(+0.62%, CI crossing 0); the fix is true conditional/preintegration structure
plus the per-MLP gate.

Hard constraints for this launch:
- The mechanism above MUST be the core change in your candidate estimator.
- Do NOT pivot to another mode's mechanism; hybrid elements are allowed only if
  this mode's mechanism remains the dominant novelty.
- If evidence suggests this mode cannot beat the parent estimator, still return
  your best in-mode candidate rather than silently substituting a different
  family; a clean in-mode attempt is more valuable to the outer search.
- Where this block conflicts with any earlier generic exploration guidance,
  this block wins.
