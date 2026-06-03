"""A100-only reproduction for GPUMode vectoradd_v2."""

from __future__ import annotations

import torch

_EXT = None
_BUILD_FAILED = False
_TARGET_N = 16384


CPP_SRC = r"""
#include <torch/extension.h>

torch::Tensor a100_add(torch::Tensor A, torch::Tensor B, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("a100_add", &a100_add, "A100 vectoradd_v2 fast path");
}
"""

CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

constexpr size_t N_SIZE = 16384;
constexpr size_t TOTAL_ELEMENTS = N_SIZE * N_SIZE;

__global__ __launch_bounds__(256)
void add_fp16_n16384_kernel(const half* __restrict__ A,
                            const half* __restrict__ B,
                            half* __restrict__ C) {
  constexpr size_t VEC_SIZE = TOTAL_ELEMENTS / 8;
  constexpr size_t BLOCKS = 16384;
  constexpr size_t THREADS = 256;
  constexpr size_t STRIDE = BLOCKS * THREADS;
  constexpr size_t ITERS = VEC_SIZE / STRIDE;

  size_t idx = blockIdx.x * THREADS + threadIdx.x;
  const uint4* __restrict__ A_vec = reinterpret_cast<const uint4*>(A);
  const uint4* __restrict__ B_vec = reinterpret_cast<const uint4*>(B);
  uint4* __restrict__ C_vec = reinterpret_cast<uint4*>(C);

  #pragma unroll
  for (int i = 0; i < ITERS; ++i) {
    size_t offset = idx + i * STRIDE;
    uint4 a = A_vec[offset];
    uint4 b = B_vec[offset];
    uint4 c;
    half2* a2 = reinterpret_cast<half2*>(&a);
    half2* b2 = reinterpret_cast<half2*>(&b);
    half2* c2 = reinterpret_cast<half2*>(&c);
    c2[0] = __hadd2(a2[0], b2[0]);
    c2[1] = __hadd2(a2[1], b2[1]);
    c2[2] = __hadd2(a2[2], b2[2]);
    c2[3] = __hadd2(a2[3], b2[3]);
    C_vec[offset] = c;
  }
}

torch::Tensor a100_add(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  add_fp16_n16384_kernel<<<N_SIZE, 256>>>(
      reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
      reinterpret_cast<half*>(C.data_ptr<at::Half>()));
  return C;
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
            name="a100_only_vectoradd_v2_uint4_half2",
            cpp_sources=[CPP_SRC],
            cuda_sources=[CUDA_SRC],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-maxrregcount=32"],
            with_cuda=True,
            verbose=False,
        )
    except Exception:
        _BUILD_FAILED = True
        _EXT = None
    return _EXT


def _unpack(data):
    if len(data) == 3:
        return data
    a, b = data
    return a, b, torch.empty_like(a)


def _is_fast_shape(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> bool:
    return (
        a.is_cuda
        and b.is_cuda
        and c.is_cuda
        and a.dtype == b.dtype == c.dtype == torch.float16
        and a.is_contiguous()
        and b.is_contiguous()
        and c.is_contiguous()
        and a.shape == b.shape == c.shape == (_TARGET_N, _TARGET_N)
    )


def custom_kernel(data):
    a, b, c = _unpack(data)
    if _is_fast_shape(a, b, c):
        ext = _load_ext()
        if ext is not None:
            return ext.a100_add(a, b, c)
    torch.add(a, b, out=c)
    return c
