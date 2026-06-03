# vectoradd_v2 reproduction notes

## Reproduced idea

- The implementation follows the H100 top1 strategy described in `GPUMode/insights/vectoradd_v2.md`: flatten the contiguous `(N, N)` FP16 tensors into one 1D buffer and launch a single Triton streaming add kernel.
- It uses a fixed `BLOCK_SIZE = 1024` and `num_warps = 8`, with no autotuning, matching the "stable generic kernel" idea for H100 and avoiding multi-size compile/autotune overhead.
- The kernel writes directly into the preallocated output tensor `C` from the actual evaluator input `(A, B, C)` and returns `C`.
- A Triton mask handles non-multiple sizes such as `127 * 127` and `129 * 129`.

## Simplifications

- This is not a source copy of the leaderboard submission. It is a clean reimplementation of the described one-kernel Triton idea.
- It does not include architecture-specific CUDA inline extensions, half2/vectorized `uint4` loads, hand-written PTX, L2 prefetching, import-time warmup, or size-specific overfitting.
- The wrapper includes conservative checks for CUDA, contiguity, dtype, and equal element counts. If those checks fail, it falls back to `torch.add(a, b, out=out)`.
- If the evaluator ever supplies only `(A, B)`, the reproduction allocates `torch.empty_like(A)` for compatibility, but the intended vectoradd_v2 path is `(A, B, C)`.

## Validation requirements

- Fast path requirements: PyTorch, Triton, a CUDA GPU, and a Triton-compatible NVIDIA driver. The target idea is meant for H100, but the kernel is generic CUDA/Triton and should run on other supported NVIDIA GPUs.
- Official benchmark validation requires the GPUMode evaluator on a CUDA machine. The evaluator clears L2 before timing, so expected performance is dominated by single-pass memory bandwidth.
- On a machine without CUDA/Triton, `custom_kernel(data)` remains runnable via the PyTorch fallback, but that does not validate the H100 performance path.
