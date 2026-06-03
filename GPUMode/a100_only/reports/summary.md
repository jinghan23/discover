# A100-only Verification Summary

All four A100-only reproductions passed local syntax/import/interface checks and CPU smoke tests.

| Problem | CPU smoke | Max abs error | Similarity to A100 top1 |
| --- | --- | ---: | ---: |
| `matmul_v2` | pass | `0` | `82.2%` |
| `trimul` | pass | `1.19e-06` | `13.2%` |
| `vectoradd_v2` | pass | `0` | `60.6%` |
| `vectorsum_v2` | pass | `0` | `51.2%` |

Interpretation:

- `matmul_v2`, `vectoradd_v2`, and `vectorsum_v2` reproduce the A100 top1 fast-path strategy directly.
- `trimul` reproduces the A100 high-level decomposition, but not the full Triton fused-kernel implementation.
- Official A100 GPU evaluator runs are still required because `torch.cuda.is_available()` is false locally.
