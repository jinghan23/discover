# A100-only Reproduction Quality Review

Scope: inspect `GPUMode/a100_only/reproductions/*/submission.py` against the A100 top1 submissions.

Local validation:

- `python -B GPUMode/a100_only/reports/verify_a100_only.py`
- All four reproductions pass syntax/import/interface checks and CPU smoke tests.
- `torch.cuda.is_available()` is false locally, so A100 fast paths were not executed.

## Overall Assessment

| Problem | Quality | Reason |
| --- | --- | --- |
| `matmul_v2` | Good A100 strategy reproduction | Uses the same fixed max-shape cuBLASLt heuristic#2 idea and `torch.mm(out=c)` fallback. |
| `vectoradd_v2` | Good A100 strategy reproduction | Uses the same `N=16384`, `uint4/half2`, fixed-grid CUDA fast path and PyTorch fallback. |
| `vectorsum_v2` | Good A100 strategy reproduction | Uses the same 52M, `float4`, warp/block reduce, global `atomicAdd` strategy. |
| `trimul` | Partial reproduction | Implements the same TriMul math and high-level staged layout, but not the Triton fused kernels. |

## Important Caveats

1. The CUDA extensions are lazy-built on first fast-path use.
   The original A100 submissions build extensions at import time. Lazy build keeps local CPU tests usable, but the first A100 fast-path call may pay compilation cost inside the evaluator.

2. `trimul` is not a faithful low-level A100 top1 reproduction.
   The A100 top1 uses Triton kernels for LN/projection/gate/mask/final fusion and `torch.bmm` for the contraction. The reproduction uses PyTorch operations for most of that pipeline. It is useful for correctness and algorithm study, not for reproducing rank-1 performance.

3. CPU smoke tests only prove fallback/math correctness.
   They do not prove CUDA extension compilation, A100 correctness, stream behavior, or leaderboard performance.

## Per-problem Notes

### `matmul_v2`

The reproduction is close to the A100 top1. It preserves:

- A100 target shape `(4096,4096) @ (4096,5120)`.
- cuBLASLt row-major fp16/fp32-compute plan.
- 32MiB workspace.
- heuristic result index `2`.
- `torch.mm(a, b, out=c)` fallback.

Main risk: lazy build timing.

### `vectoradd_v2`

The reproduction is close to the A100 top1. It preserves:

- `N=16384` specialization.
- 16-byte `uint4` loads/stores.
- four `half2` additions per vector.
- `16384` blocks and `256` threads.
- PyTorch fallback for other sizes.

Main risks: lazy build timing and untested A100 extension compilation.

### `vectorsum_v2`

The reproduction is close to the A100 top1. It preserves:

- `52,428,800` specialization.
- `float4` vector loads.
- `2048` blocks, `256` threads, `25` vector loads per thread.
- warp/block reduction.
- final global `atomicAdd`.

Main risks: lazy build timing, stream behavior, and float32 reduction tolerance on real A100 inputs.

### `trimul`

The reproduction is algorithmically correct but not a strong A100 top1 reproduction. It preserves:

- LayerNorm, five packed projections/gates, mask, triangle contraction, LN2, out gate, final projection.
- hidden-major projection idea.
- batched `torch.bmm` contraction.

It does not preserve:

- Triton fused LN/projection/gate/mask kernels.
- Triton fused final LN/gate/output kernel.
- runner-specific tuning and launch configuration.

Verdict: acceptable as an algorithmic reference; insufficient as a performance reproduction.
