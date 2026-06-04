# Spherical Code Improvement Notes

Initial score: `0.5130016277370809`

Improved score: `0.6736467551690192`

Reference score: approximately `0.6736467551690225`

Approach:

- Started from a symmetric 30-point icosidodecahedral-style seed on `S^2`.
- Smoothed the max-pairwise-dot objective with log-sum-exp relaxation to escape the greedy baseline geometry.
- Refined the result with constrained SLSQP on spherical coordinates, minimizing `t` subject to `dot(p_i, p_j) <= t` for all pairs.
- Stored the final deterministic coordinates directly in `work/improved_program.py` so evaluation is fast and does not depend on running the optimizer.

Evaluation:

- Command: `python /Users/bytedance/Documents/workspace/ttt-discover/repro_external/alpharesearchcomp/runs/_common/run_program_eval.py spherical_code work/improved_program.py`
- Result: `score = min_angle = 0.6736467551690192`, `n = 30`, `dimension = 3`.
