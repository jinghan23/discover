import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import torch
from task import input_t, output_t

_TARGET = 52428800
_LIB = None
_FAIL = False
_ERR = None


def _src() -> str:
    return r"""
#include <cuda_runtime.h>
#include <stdint.h>
namespace {
constexpr int TARGET_N = 52428800;
constexpr int BLOCK_THREADS = 1024;
constexpr int VEC_LOADS_PER_THREAD = 2;
constexpr int NUM_BLOCKS = 4288;
constexpr int NUM_SLOTS = 4;

struct Scratch {
  float2 slots[NUM_SLOTS];
  unsigned int count;
};

__device__ __forceinline__ float2 warp_sum2(float2 v) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v.x += __shfl_down_sync(0xffffffff, v.x, offset);
    v.y += __shfl_down_sync(0xffffffff, v.y, offset);
  }
  return v;
}

template <int BT, int VPT>
__global__ void pass1_slots2(const float2* __restrict__ in2, float* __restrict__ out, Scratch* __restrict__ scratch) {
  constexpr int VECS_PER_TILE = BT * VPT;
  constexpr int TOTAL_TILES = TARGET_N / (VECS_PER_TILE * 2);
  __shared__ float2 warp_sums[BT / 32];
  float2 sum = make_float2(0.f, 0.f);
  int tid = threadIdx.x;

  for (int tile = int(blockIdx.x); tile < TOTAL_TILES; tile += int(gridDim.x)) {
    int base = tile * VECS_PER_TILE + tid;
    #pragma unroll
    for (int i = 0; i < VPT; ++i) {
      float2 v = __ldcs(&in2[base + i * BT]);
      sum.x += v.x;
      sum.y += v.y;
    }
  }

  sum = warp_sum2(sum);
  if ((tid & 31) == 0) warp_sums[tid >> 5] = sum;
  __syncthreads();

  if (tid < 32) {
    float2 block_sum = tid < (BT / 32) ? warp_sums[tid] : make_float2(0.f, 0.f);
    block_sum = warp_sum2(block_sum);
    if (tid == 0) {
      atomicAdd(&scratch->slots[int(blockIdx.x) & 3], block_sum);
      __threadfence();
      unsigned int ticket = atomicAdd(&scratch->count, 1u);
      if (ticket == unsigned(NUM_BLOCKS - 1)) {
        float total = 0.f;
        #pragma unroll
        for (int i = 0; i < NUM_SLOTS; ++i) {
          float2 v = scratch->slots[i];
          total += v.x + v.y;
          scratch->slots[i] = make_float2(0.f, 0.f);
        }
        out[0] = total;
        scratch->count = 0;
      }
    }
  }
}

template <int BT, int VPT>
__global__ void pass1_slots4(const float4* __restrict__ in4, float* __restrict__ out, Scratch* __restrict__ scratch) {
  constexpr int VECS_PER_TILE = BT * VPT;
  constexpr int TOTAL_TILES = TARGET_N / (VECS_PER_TILE * 4);
  __shared__ float2 warp_sums[BT / 32];
  float s0 = 0.f;
  float s1 = 0.f;
  float s2 = 0.f;
  float s3 = 0.f;
  int tid = threadIdx.x;

  for (int tile = int(blockIdx.x); tile < TOTAL_TILES; tile += int(gridDim.x)) {
    int base = tile * VECS_PER_TILE + tid;
    #pragma unroll
    for (int i = 0; i < VPT; ++i) {
      float4 v = __ldcs(&in4[base + i * BT]);
      s0 += v.x;
      s1 += v.y;
      s2 += v.z;
      s3 += v.w;
    }
  }

  float2 sum = make_float2(s0 + s1, s2 + s3);
  sum = warp_sum2(sum);
  if ((tid & 31) == 0) warp_sums[tid >> 5] = sum;
  __syncthreads();

  if (tid < 32) {
    float2 block_sum = tid < (BT / 32) ? warp_sums[tid] : make_float2(0.f, 0.f);
    block_sum = warp_sum2(block_sum);
    if (tid == 0) {
      atomicAdd(&scratch->slots[int(blockIdx.x) & 3], block_sum);
      __threadfence();
      unsigned int ticket = atomicAdd(&scratch->count, 1u);
      if (ticket == unsigned(NUM_BLOCKS - 1)) {
        float total = 0.f;
        #pragma unroll
        for (int i = 0; i < NUM_SLOTS; ++i) {
          float2 v = scratch->slots[i];
          total += v.x + v.y;
          scratch->slots[i] = make_float2(0.f, 0.f);
        }
        out[0] = total;
        scratch->count = 0;
      }
    }
  }
}
}  // namespace

extern "C" int vsum_launch(uint64_t in_ptr, uint64_t out_ptr, unsigned long long n) {
  if (n != TARGET_N) return 7001;

  float* out = reinterpret_cast<float*>(out_ptr);
  static Scratch* scratch = nullptr;
  if (scratch == nullptr) {
    cudaError_t err = cudaMalloc(&scratch, sizeof(Scratch));
    if (err != cudaSuccess) return static_cast<int>(err);
    err = cudaMemset(scratch, 0, sizeof(Scratch));
    if (err != cudaSuccess) return static_cast<int>(err);
  }

  if ((in_ptr & 15ull) == 0ull) {
    const float4* in4 = reinterpret_cast<const float4*>(in_ptr);
    pass1_slots4<BLOCK_THREADS, VEC_LOADS_PER_THREAD><<<NUM_BLOCKS, BLOCK_THREADS>>>(in4, out, scratch);
  } else {
    const float2* in2 = reinterpret_cast<const float2*>(in_ptr);
    pass1_slots2<BLOCK_THREADS, VEC_LOADS_PER_THREAD><<<NUM_BLOCKS, BLOCK_THREADS>>>(in2, out, scratch);
  }

  cudaError_t err = cudaGetLastError();
  return static_cast<int>(err);
}
"""


def _nvcc() -> str:
    home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    cand = os.path.join(home, 'bin', 'nvcc')
    return cand if os.path.exists(cand) else 'nvcc'


def _build_for(dev: torch.device):
    cap = torch.cuda.get_device_capability(dev)
    arch = f'{cap[0]}{cap[1]}'
    source = _src()
    key = hashlib.sha256((source + arch).encode()).hexdigest()[:16]
    root = Path(tempfile.gettempdir()) / 'opt021_local' / key
    cu_path = root / 'vsum.cu'
    so_path = root / 'vsum.so'
    if not so_path.exists():
        root.mkdir(parents=True, exist_ok=True)
        cu_path.write_text(source)
        tmp_path = root / 'vsum.so.tmp'
        cmd = [
            _nvcc(),
            '-O3',
            '-std=c++17',
            '--shared',
            '-Xcompiler',
            '-fPIC',
            f'-gencode=arch=compute_{arch},code=sm_{arch}',
            str(cu_path),
            '-o',
            str(tmp_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr or proc.stdout or 'nvcc failed')
        os.replace(tmp_path, so_path)
    lib = ctypes.CDLL(str(so_path))
    lib.vsum_launch.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_ulonglong]
    lib.vsum_launch.restype = ctypes.c_int
    return lib


def _lib_for(dev: torch.device):
    global _LIB, _FAIL, _ERR
    if _LIB is not None:
        return _LIB
    if _FAIL:
        return None
    try:
        _LIB = _build_for(dev)
    except Exception as exc:
        _FAIL = True
        _ERR = repr(exc)
        _LIB = None
    return _LIB


def _use_fast(data: torch.Tensor, out: torch.Tensor) -> bool:
    return (
        data.is_cuda and out.is_cuda and data.dtype == torch.float32 and out.dtype == torch.float32
        and data.is_contiguous() and out.is_contiguous() and data.numel() == _TARGET and out.numel() == 1
        and data.data_ptr() % 8 == 0
    )


def custom_kernel(data: input_t) -> output_t:
    x, out = data
    if _use_fast(x, out):
        lib = _lib_for(x.device)
        if lib is not None:
            err = lib.vsum_launch(x.data_ptr(), out.data_ptr(), x.numel())
            if err == 0:
                return out.reshape(())
    return x.sum(dtype=torch.float32)
