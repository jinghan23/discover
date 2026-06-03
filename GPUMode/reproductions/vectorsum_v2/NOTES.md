# vectorsum_v2 H100 Top1 Idea Reproduction

## What is reproduced

- Implements the task contract `custom_kernel((input_tensor, output_tensor)) -> scalar`.
- Targets the H100 top1 strategy from the insight notes: the only CUDA fast path is the largest benchmark size, `N = 52,428,800`, with `float32`, contiguous CUDA input.
- Uses a thin `ctypes` wrapper around an nvcc-built shared object instead of a PyTorch extension wrapper.
- Uses one CUDA kernel for the fast path: fixed `4288` blocks, `1024` threads per block, two vector loads per thread per grid-stride step, warp/block reductions, then a last-block finalization pattern.
- Splits global accumulation across four scratch slots and uses a global completion counter so the last block writes the final scalar and resets scratch state.
- Falls back to PyTorch for all non-fast-path cases, including CPU, small correctness sizes, non-H100 GPUs, missing nvcc, non-contiguous tensors, or unexpected dtype/output shape.

## Simplifications versus the original top1 idea

- This is an independent rewrite from the described idea, not a copy of the top submission source.
- The scratch accumulator is simplified to four scalar `float` slots instead of reproducing the exact `float2` slot layout described in the insight.
- The fast path is intentionally narrow. It does not tune for the other five benchmark sizes.
- Fallback uses a conservative `float64` PyTorch sum converted to `float32` to match the reference more closely; that favors correctness over leaderboard speed outside the H100 fast path.
- Build failures are silent and route to fallback, which is friendlier for local reproduction but can hide whether the fast path actually ran unless profiled.

## Validation requirements

- Fast path requires an NVIDIA GPU with compute capability 9.x or newer, CUDA runtime, PyTorch with CUDA, and `nvcc` available through `PATH`, `CUDA_HOME`, `CUDA_PATH`, or `torch.utils.cpp_extension.CUDA_HOME`.
- The first fast-path call compiles a cached shared library under the system temp directory.
- The official evaluator needs CUDA because `reference.generate_input` creates CUDA tensors. On a machine without an NVIDIA GPU, only import/CPU fallback smoke tests can be run locally.
- Numerical behavior follows the top-solution tradeoff: the fast path does `float32` tree/atomic reduction, while the reference uses `float64` reduction then casts to `float32`. It is expected to pass the original tolerance for the benchmark distribution, but it is not bit-exact.
