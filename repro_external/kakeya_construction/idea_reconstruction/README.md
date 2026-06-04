# Idea-only reconstruction of AlphaEvolve Kakeya 3D experiments

Source oracle: the official DeepMind notebook
`experiments/finite_field_kakeya_problem/finite_field_kakeya.ipynb`
from <https://github.com/google-deepmind/alphaevolve_repository_of_problems>.

Goal: test whether the final 3D `Code found by AlphaEvolve` programs for
Experiments 1-3 can be reconstructed from a compact idea-level description,
rather than by copying the notebook code.

## Files

- `exp1_reconstructed.py`: reconstruction from the Experiment 1 strip idea.
- `exp2_reconstructed.py`: reconstruction from the Experiment 2 parabolic
  half-set idea.
- `exp3_reconstructed.py`: reconstruction from the Experiment 3 quadratic
  residue / gamma-overlap idea.
- `compare_reconstructions.py`: loads the official notebook cells as oracles,
  imports the reconstructed modules, then compares size, exact point-set
  equality, and Kakeya validity.
- `comparison_results.json`: latest comparison run.

Each reconstructed file exports:

```python
search_for_best_construction(p: int, d: int) -> numpy.ndarray
```

Only `d=3` is supported. The main path is for odd primes; `p=2` uses a simple
valid fallback.

## Idea summaries

Experiment 1:

- Build a 2D strip
  `V_t(c,d)={s(2t+c-s)+t+d : s in F_p}`.
- Use two strips in coordinates `u=y+z` and `v=y-z` to cover directions
  `(1,a,b)`.
- Use one plane strip at `x=0` to cover `(0,1,c)`.
- Add the vertical line `(0,0,z)` for `(0,0,1)`.
- Search the tiny parameter grid `{0,1,1/2,-1}^6` for best overlap.

Experiment 2:

- Use parabolic half-sets
  `u=x^2+L_y x+C_y-s^2`, `v=x^2+L_z x+C_z-t^2`.
- Apply a small invertible transform family to map `(u,v)` to `(y,z)`,
  especially the rotated transform
  `y=(u+v)/2`, `z=(u-v)/2`.
- Add a 2D Kakeya construction in the `x=0` plane plus a vertical line.
- Search constants `{0,1,-1, +/-1/2, +/-1/4}`, linear terms `{0,1,-1}`,
  and four transform types for best overlap.

Experiment 3:

- Let `S={r^2 mod p}`. Use
  `K_A={(x,y,z): y+z+x^2+g1 in S, y-z+x^2+g2 in S}`.
- Add `K_0B={(0,y,z): z+y^2+g3 in S}` and
  `K_0C={(0,g4,z): z in F_p}`.
- Search candidate `gamma` values to maximize overlap between the `x=0`
  slice of `K_A` and the planar pieces.

## Comparison

Command:

```bash
python3 repro_external/kakeya_construction/idea_reconstruction/compare_reconstructions.py --primes 3 5 7 13 17
```

Results:

| experiment | p | official size | reconstructed size | size match | exact set equal | reconstructed valid |
|---|---:|---:|---:|---|---|---|
| exp1 | 3 | 15 | 15 | yes | yes | yes |
| exp1 | 5 | 53 | 53 | yes | yes | yes |
| exp1 | 7 | 128 | 128 | yes | yes | yes |
| exp1 | 13 | 697 | 697 | yes | yes | yes |
| exp1 | 17 | 1481 | 1481 | yes | yes | yes |
| exp2 | 3 | 15 | 15 | yes | yes | yes |
| exp2 | 5 | 53 | 53 | yes | yes | yes |
| exp2 | 7 | 128 | 128 | yes | no | yes |
| exp2 | 13 | 697 | 697 | yes | yes | yes |
| exp2 | 17 | 1481 | 1481 | yes | yes | yes |
| exp3 | 3 | 15 | 15 | yes | yes | yes |
| exp3 | 5 | 53 | 53 | yes | yes | yes |
| exp3 | 7 | 128 | 128 | yes | yes | yes |
| exp3 | 13 | 697 | 697 | yes | yes | yes |
| exp3 | 17 | 1481 | 1481 | yes | yes | yes |

The only exact-set mismatch is Experiment 2 at `p=7`. It has the same size and
passes the Kakeya verifier, so this appears to be a tie-breaking / equivalent
parameter issue rather than a failure of the idea-level reconstruction.

## Interpretation

For these 3D Kakeya experiments, the final AlphaEvolve programs are compactly
describable as algebraic construction families plus small deterministic
parameter searches for overlap. The idea summaries were enough to reproduce
the official sizes and validity across the tested primes. Exact source-level
reproduction still requires low-level choices such as parameter ordering and
tie-breaking, visible in the Experiment 2 `p=7` exact-set mismatch.
