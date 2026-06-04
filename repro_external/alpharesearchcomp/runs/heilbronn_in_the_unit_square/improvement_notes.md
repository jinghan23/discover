# Heilbronn n=16 Improvement Notes

Initial score: 0.008363857977552652

Improved score: 0.02052785923753661

Core idea:
- Use the known 180-degree rotationally symmetric n=16 record pattern as a structured seed.
- Convert the pattern into an exact integer-grid construction: x coordinates on a 31-point scale and y coordinates on a 33-point scale.
- Return the scaled lattice points directly, avoiding runtime random search or optimizer overhead.

The implemented lattice points are:

```text
(8/31, 33/33),  (29/31, 33/33),
(0/31, 31/33),  (21/31, 31/33),
(10/31, 23/33), (31/31, 23/33),
(2/31, 21/33),  (23/31, 21/33),
(8/31, 12/33),  (29/31, 12/33),
(0/31, 10/33),  (21/31, 10/33),
(10/31, 2/33),  (31/31, 2/33),
(2/31, 0/33),   (23/31, 0/33)
```

Evaluation command:

```bash
python /Users/bytedance/Documents/workspace/ttt-discover/repro_external/alpharesearchcomp/runs/_common/run_program_eval.py heilbronn_in_the_unit_square work/improved_program.py
```

Evaluation result:

```json
{
  "score": 0.02052785923753661,
  "scaled_min_area": 0.488745581344794,
  "min_area": 0.02052785923753661,
  "n": 16.0
}
```
