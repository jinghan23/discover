# Semantic Equivalence Review

Scope: compare each H100 top1 submission under `GPUMode/top_solutions/*/submissions/`
with its reproduction under `GPUMode/reproductions/*/submission.py`.

The question here is not whether the source code is copied or textually similar,
but whether the reproduced file implements the same content.

## Summary

| Problem | Same Mathematical Operation | Same Optimization Path | Verdict |
| --- | --- | --- | --- |
| `matmul_v2` | Yes | Yes, for the default path | Same content |
| `vectoradd_v2` | Yes | Yes, with safer fallback/wrapper differences | Same content |
| `vectorsum_v2` | Yes | Mostly yes for the H100 large-size fast path | Same idea, not identical implementation |
| `trimul` | Yes | No, only the high-level layout idea is reproduced | Same algorithmic content, not same top1 low-level implementation |

## `matmul_v2`

Original H100 top1:

- Sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Unpacks `(a, b, c)`.
- Returns `a @ b`.
- Does not write into `c`.

Reproduction:

- Also sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Also unpacks `(a, b, c)`.
- Default path also returns `a @ b`.
- Adds an optional environment-controlled `torch.mm(..., out=c)` debug fallback.

Verdict: same content on the normal path. The added fallback is inactive unless
`GPUMODE_MATMUL_V2_FORCE_OUT=1`, so it does not change the reproduced H100 idea.

## `vectoradd_v2`

Original H100 top1:

- Defines one Triton kernel.
- Flattens contiguous `(N, N)` tensors into a 1D range.
- Uses `BLOCK_SIZE=1024`, `num_warps=8`.
- Loads `A`, loads `B`, stores `A + B` into `C`.
- Uses a mask for tail elements.

Reproduction:

- Defines the same kind of one-kernel Triton flat add.
- Uses `BLOCK_SIZE=1024`, `num_warps=8`.
- Loads `A`, loads `B`, stores `A + B` into `C`.
- Uses a tail mask.
- Adds conservative dtype/device/contiguity checks and PyTorch fallback.
- Supports `(A, B, C)` and a defensive `(A, B)` compatibility path.

Verdict: same content. The wrapper is not identical, but the fast path implements
the same operation with the same key H100 strategy.

## `vectorsum_v2`

Original H100 top1:

- Fast path only for `N == 52,428,800`.
- Builds a CUDA shared library with `nvcc` and calls it through `ctypes`.
- Uses `4288` blocks, `1024` threads, `2` vector loads per thread.
- Uses `float4` when 16-byte aligned and `float2` when only 8-byte aligned.
- Reduces within warp/block, atomically accumulates into 4 scratch slots, then
  the last block writes the scalar output and clears scratch state.
- Falls back to `x.sum(dtype=torch.float32)`.

Reproduction:

- Also fast-paths `N == 52,428,800`.
- Also uses `nvcc + ctypes`.
- Also uses `4288` blocks, `1024` threads, `2` vector loads per thread.
- Also uses `float4` / `float2` depending on alignment.
- Also reduces within warp/block, atomically accumulates into 4 scratch slots,
  uses a completion counter, and lets the last block write the scalar output.
- Uses current CUDA stream explicitly.
- Uses a `float[4]` scratch representation rather than the original `float2[4]`.
- Fallback uses float64-style reference sum instead of original float32 sum.

Verdict: same algorithmic idea and same target fast-path content, but not a
bit-for-bit implementation clone. The fallback differs numerically, although it
is closer to the reference implementation. GPU correctness for the fast path
still needs to be checked on H100.

## `trimul`

Original H100 top1:

- Uses a large C++/CUDA extension.
- LN1 is a custom CUDA kernel.
- Packs five projection/gate weights into a half matrix.
- Computes projection into hidden-major `projT = [5H, M]` through cuBLAS/cuBLASLt.
- Packs/gates/masks left and right through custom CUDA kernels.
- Performs the triangle contraction as a batched Tensor Core GEMM through
  cuBLASLt/cublas, with per-shape algorithm caching and tuning.
- Fuses LN2 + out gate into a custom CUDA kernel.
- Performs final projection through cuBLASLt.

Reproduction:

- Uses PyTorch rather than a C++/CUDA extension.
- Computes LN1 with `torch.nn.functional.layer_norm`.
- Packs the same five projection/gate weights and computes `[5H, M]`.
- Keeps hidden-major layout.
- Applies gate and mask.
- Performs triangle contraction with `torch.bmm`.
- Computes LN2 in PyTorch and applies out gate.
- Performs final projection with PyTorch matmul.

Verdict: same mathematical operator and same high-level decomposition, but not
the same low-level top1 implementation. It is a faithful algorithmic
reproduction, not a faithful performance-path reproduction.

## Overall Answer

If "same content" means the target computation, all four reproductions implement
the same content as the corresponding H100 top1 submissions.

If "same content" means the same low-level top1 implementation strategy:

- `matmul_v2`: yes.
- `vectoradd_v2`: yes, modulo safer wrapper/fallback.
- `vectorsum_v2`: mostly yes for the fast path, but implementation details differ.
- `trimul`: no; it reproduces the algorithm and layout idea, not the fused
  cuBLASLt/custom-CUDA performance path.
