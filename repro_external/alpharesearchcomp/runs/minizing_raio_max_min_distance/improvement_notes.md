# minizing_raio_max_min_distance improvement notes

## Scores

- Initial score from `initial_eval.json`: `0.2805918721490253`
- Improved score from `improved_eval.json`: `0.3176350367963276`
- `ratio_n16_d2`: `0.07758415414722268`
- `ratio_n14_d3`: `0.24005088264910493`

## Approach

The evaluator only scores the returned point sets, and the metric
`(min_pairwise_distance / max_pairwise_distance)^2` is invariant under
translation and uniform scaling. I therefore searched offline for compact
unit-min-distance constructions and stored the best deterministic coordinates.

For `(16, 2)`, the starting point was a triangular-lattice cluster, refined with
SLSQP constraints that keep all pairwise distances at least `1` while minimizing
the squared diameter. The final squared diameter is about `12.8892299077`.

For `(14, 3)`, the starting point was an icosahedral/kissing-style cluster with
one interior point, refined with the same constrained diameter minimization. The
final squared diameter is about `4.1657834746`.

`work/improved_program.py` returns these fixed constructions after translating
and uniformly scaling them into `[0, 1]^d`, so evaluation is deterministic and
does not depend on simulated annealing randomness.

## Verification

Final evaluation command:

```bash
python /Users/bytedance/Documents/workspace/ttt-discover/repro_external/alpharesearchcomp/runs/_common/run_program_eval.py minizing_raio_max_min_distance work/improved_program.py
```
