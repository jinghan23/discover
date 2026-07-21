--- Diversity Mode (FORCED for this launch) ---
Mode: `frozen_gating_proxy_cv`

Build a per-sample control variate from a frozen-gating linearization of the
FULL depth-32 network. Recipe: run a few anchor forward passes, record the
ReLU masks along each anchor, and compose the masked chain into a single
width x width matrix A once per anchor (one-time cost ~depth matmuls,
honestly accounted); the per-sample proxy h(x) = x @ A costs one matvec
(~3% of a forward) and has EXACT mean zero under N(0,I). Variant with the
final ReLU kept: h(x) = max(x @ A_pre, 0), whose exact per-column mean is
||a_col|| / sqrt(2*pi). Subtract beta * h with beta fit by cross-fitting on
disjoint sample halves; optionally use 2-8 anchors chosen by clustering pilot
gating signatures. Compose A from the whitened first layer actually used at
predict time. Evidence: a first-layer-only analytic ReLU CV lost (+3.06%)
because it cannot see depth — the proxy MUST compose all layers' masks.

Hard constraints for this launch:
- The mechanism above MUST be the core change in your candidate estimator.
- Do NOT pivot to another mode's mechanism; hybrid elements are allowed only if
  this mode's mechanism remains the dominant novelty.
- If evidence suggests this mode cannot beat the parent estimator, still return
  your best in-mode candidate rather than silently substituting a different
  family; a clean in-mode attempt is more valuable to the outer search.
- Where this block conflicts with any earlier generic exploration guidance,
  this block wins.
