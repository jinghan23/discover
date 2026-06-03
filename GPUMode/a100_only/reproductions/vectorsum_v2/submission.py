"""A100-only reproduction for GPUMode vectorsum_v2."""

from __future__ import annotations

import torch

try:
    from task import input_t, output_t
except Exception:
    input_t = tuple[torch.Tensor, torch.Tensor]
    output_t = torch.Tensor

_EXT = None
_BUILD_FAILED = False
_TARGET_N = 52_428_800


CPP_SRC = r"""
#include <torch/extension.h>

torch::Tensor a100_sum(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("a100_sum", &a100_sum, "A100 vectorsum_v2 fast path");
}
"""

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>

constexpr int N_FLOATS = 52428800;
constexpr int BLOCK_SIZE = 256;
constexpr int GRID_SIZE = 2048;
constexpr int FLOAT4_PER_THREAD = 25;
constexpr int STRIDE_FLOAT4 = BLOCK_SIZE * GRID_SIZE;

__global__ void reduce_sum_50m_a100(const float* __restrict__ in, float* __restrict__ out) {
  __shared__ float sdata[8];
  int tid = threadIdx.x;
  int idx = blockIdx.x * BLOCK_SIZE + tid;
  float sum = 0.0f;
  const float4* in4 = reinterpret_cast<const float4*>(in);

  #pragma unroll
  for (int j = 0; j < FLOAT4_PER_THREAD; ++j) {
    float4 v = __ldg(&in4[idx + j * STRIDE_FLOAT4]);
    sum += v.x + v.y + v.z + v.w;
  }

  sum += __shfl_xor_sync(0xffffffff, sum, 16);
  sum += __shfl_xor_sync(0xffffffff, sum, 8);
  sum += __shfl_xor_sync(0xffffffff, sum, 4);
  sum += __shfl_xor_sync(0xffffffff, sum, 2);
  sum += __shfl_xor_sync(0xffffffff, sum, 1);

  if ((tid & 31) == 0) sdata[tid >> 5] = sum;
  __syncthreads();

  if (tid < 32) {
    float wsum = tid < 8 ? sdata[tid] : 0.0f;
    wsum += __shfl_xor_sync(0xffffffff, wsum, 4);
    wsum += __shfl_xor_sync(0xffffffff, wsum, 2);
    wsum += __shfl_xor_sync(0xffffffff, wsum, 1);
    if (tid == 0) atomicAdd(out, wsum);
  }
}

torch::Tensor a100_sum(torch::Tensor x, torch::Tensor out) {
  out.fill_(0);
  reduce_sum_50m_a100<<<GRID_SIZE, BLOCK_SIZE>>>(x.data_ptr<float>(), out.data_ptr<float>());
  return out[0];
}
"""


def _load_ext():
    global _EXT, _BUILD_FAILED
    if _EXT is not None:
        return _EXT
    if _BUILD_FAILED:
        return None
    try:
        from torch.utils.cpp_extension import load_inline

        _EXT = load_inline(
            name="a100_only_vectorsum_v2_float4_atomic",
            cpp_sources=[CPP_SRC],
            cuda_sources=[CUDA_SRC],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            with_cuda=True,
            verbose=False,
        )
    except Exception:
        _BUILD_FAILED = True
        _EXT = None
    return _EXT


def _is_fast_shape(x: torch.Tensor, out: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and out.is_cuda
        and x.dtype == torch.float32
        and out.dtype == torch.float32
        and x.is_contiguous()
        and out.is_contiguous()
        and x.numel() == _TARGET_N
        and out.numel() >= 1
        and x.data_ptr() % 16 == 0
    )


def custom_kernel(data: input_t) -> output_t:
    x, out = data
    if _is_fast_shape(x, out):
        ext = _load_ext()
        if ext is not None:
            return ext.a100_sum(x, out)
    return x.sum(dtype=torch.float32)
