--- Diversity Mode (FORCED for this launch) ---
Mode: `rqmc_engine`

Replace the iid Gaussian half-batch inside the German-current structure with
randomized quasi-Monte Carlo while keeping antithetic mirroring, half-sample
covariance whitening, and the compute-fraction logic. Recipes: scrambled Sobol
points in d = mlp.width dimensions, or a rank-1 Korobov/Fibonacci lattice with
a random shift, mapped through the inverse normal CDF (flops.stats.norm.ppf),
with the scramble/shift seeded from mlp.seed. Evidence: pure lattice RQMC
scored 3.26e-7 official at 30% budget WITHOUT German's structural reductions,
and lattices beat Haar blocks locally by ~10%, but the combination
RQMC x (antithetic + whitening) at compute fraction ~0.10-0.155 is untested.
Do NOT use Haar/orthogonal blocks, tight frames, or Householder orbits — all
already lost on this suite. Account the point-generation and ppf FLOPs
honestly.

Hard constraints for this launch:
- The mechanism above MUST be the core change in your candidate estimator.
- Do NOT pivot to another mode's mechanism; hybrid elements are allowed only if
  this mode's mechanism remains the dominant novelty.
- If evidence suggests this mode cannot beat the parent estimator, still return
  your best in-mode candidate rather than silently substituting a different
  family; a clean in-mode attempt is more valuable to the outer search.
- Where this block conflicts with any earlier generic exploration guidance,
  this block wins.
