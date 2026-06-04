# Kakeya Construction

Reproduces the explicit finite-field Kakeya construction reported by AlphaEvolve for dimension `d=3`, for primes `p = 1 mod 4`:

```text
|K| = (2 p^3 + 7 p^2 - 1) / 8
    = p^3 / 4 + 7 p^2 / 8 - 1 / 8
```

The script builds the set and brute-force verifies that it contains a line in every projective direction of `F_p^3`.

Sources:

- https://arxiv.org/abs/2511.02864
- https://github.com/google-deepmind/alphaevolve_repository_of_problems
- https://github.com/google-deepmind/alphaevolve_repository_of_problems/blob/main/experiments/finite_field_kakeya_problem/finite_field_kakeya.ipynb

Run:

```bash
python repro_external/kakeya_construction/reproduce_finite_field_kakeya.py
```

The default run verifies `p=5,13,17` and writes `finite_field_kakeya_results.json` in this folder.

To run the literal official notebook cells marked `Code found by AlphaEvolve`, use:

```bash
python repro_external/kakeya_construction/reproduce_alphaevolve_notebook_programs.py
```

That script checks:

- the last evolved program in the 3D section, `cell 4`, for `p=5,13`;
- the literal last evolved program in the notebook, `cell 17` in the 5D section, for `p=3,5,7,11,13,17,19`;
- the notebook's slow 5D verifier only for `p=3,5`.
