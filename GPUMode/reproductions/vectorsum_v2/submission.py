import hashlib
import os
import shutil
import subprocess
import tempfile
from ctypes import CDLL, c_int, c_longlong, c_void_p
from pathlib import Path
from typing import Any

import torch

try:
    from task import input_t, output_t
except ImportError:
    input_t = Any
    output_t = torch.Tensor


BIG_N = 52_428_800
H100_BLOCKS = 4288
H100_THREADS = 1024
LOADS_PER_THREAD = 2

_CUDA_SRC = r"""
#include <cuda_runtime.h>

#define NUM_BLOCKS 4288
#define BLOCK_THREADS 1024
#define LOADS_PER_THREAD 2

__device__ float g_slots[4];
__device__ unsigned int g_count;

__inline__ __device__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffff, x, offset);
    }
    return x;
}

__inline__ __device__ float block_reduce_sum(float x) {
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    x = warp_reduce_sum(x);
    if (lane == 0) {
        warp_sums[warp] = x;
    }
    __syncthreads();

    x = (threadIdx.x < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        x = warp_reduce_sum(x);
    }
    return x;
}

__global__ void vectorsum_v4_kernel(const float* __restrict__ x,
                                    float* __restrict__ out,
                                    long long n) {
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    long long n4 = n >> 2;
    long long first = ((long long)blockIdx.x * blockDim.x * LOADS_PER_THREAD)
                    + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x * LOADS_PER_THREAD;

    float acc = 0.0f;
    for (long long base = first; base < n4; base += stride) {
        #pragma unroll
        for (int i = 0; i < LOADS_PER_THREAD; ++i) {
            long long idx = base + (long long)i * blockDim.x;
            if (idx < n4) {
                float4 v = x4[idx];
                acc += v.x + v.y + v.z + v.w;
            }
        }
    }

    float block_sum = block_reduce_sum(acc);
    __shared__ int is_last_block;

    if (threadIdx.x == 0) {
        atomicAdd(&g_slots[blockIdx.x & 3], block_sum);
        __threadfence();
        unsigned int ticket = atomicAdd(&g_count, 1u);
        is_last_block = (ticket == (unsigned int)(gridDim.x - 1));
    }
    __syncthreads();

    if (is_last_block && threadIdx.x == 0) {
        float total = g_slots[0] + g_slots[1] + g_slots[2] + g_slots[3];
        out[0] = total;
        g_slots[0] = 0.0f;
        g_slots[1] = 0.0f;
        g_slots[2] = 0.0f;
        g_slots[3] = 0.0f;
        __threadfence();
        g_count = 0u;
    }
}

__global__ void vectorsum_v2_kernel(const float* __restrict__ x,
                                    float* __restrict__ out,
                                    long long n) {
    const float2* __restrict__ x2 = reinterpret_cast<const float2*>(x);
    long long n2 = n >> 1;
    long long first = ((long long)blockIdx.x * blockDim.x * LOADS_PER_THREAD)
                    + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x * LOADS_PER_THREAD;

    float acc = 0.0f;
    for (long long base = first; base < n2; base += stride) {
        #pragma unroll
        for (int i = 0; i < LOADS_PER_THREAD; ++i) {
            long long idx = base + (long long)i * blockDim.x;
            if (idx < n2) {
                float2 v = x2[idx];
                acc += v.x + v.y;
            }
        }
    }

    float block_sum = block_reduce_sum(acc);
    __shared__ int is_last_block;

    if (threadIdx.x == 0) {
        atomicAdd(&g_slots[blockIdx.x & 3], block_sum);
        __threadfence();
        unsigned int ticket = atomicAdd(&g_count, 1u);
        is_last_block = (ticket == (unsigned int)(gridDim.x - 1));
    }
    __syncthreads();

    if (is_last_block && threadIdx.x == 0) {
        float total = g_slots[0] + g_slots[1] + g_slots[2] + g_slots[3];
        out[0] = total;
        g_slots[0] = 0.0f;
        g_slots[1] = 0.0f;
        g_slots[2] = 0.0f;
        g_slots[3] = 0.0f;
        __threadfence();
        g_count = 0u;
    }
}

extern "C" int launch_vectorsum_h100(const float* x,
                                      float* out,
                                      long long n,
                                      void* stream_ptr,
                                      int use_float4) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (use_float4) {
        vectorsum_v4_kernel<<<NUM_BLOCKS, BLOCK_THREADS, 0, stream>>>(x, out, n);
    } else {
        vectorsum_v2_kernel<<<NUM_BLOCKS, BLOCK_THREADS, 0, stream>>>(x, out, n);
    }
    return (int)cudaGetLastError();
}
"""

_LIB = None
_BUILD_FAILED = False


def _fallback_sum(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float64).sum().to(torch.float32)


def _find_nvcc() -> str | None:
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc

    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        cuda_home = os.environ.get(env_name)
        if cuda_home:
            candidate = Path(cuda_home) / "bin" / "nvcc"
            if candidate.exists():
                return str(candidate)

    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:
        CUDA_HOME = None

    if CUDA_HOME:
        candidate = Path(CUDA_HOME) / "bin" / "nvcc"
        if candidate.exists():
            return str(candidate)

    return None


def _load_cuda_library():
    global _LIB, _BUILD_FAILED
    if _LIB is not None:
        return _LIB
    if _BUILD_FAILED:
        return None

    nvcc = _find_nvcc()
    if nvcc is None:
        _BUILD_FAILED = True
        return None

    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        _BUILD_FAILED = True
        return None

    arch = f"sm_{major}{minor}"
    digest = hashlib.sha256((_CUDA_SRC + arch).encode("utf-8")).hexdigest()[:16]
    build_dir = Path(tempfile.gettempdir()) / f"vectorsum_v2_repro_{digest}_{arch}"
    src_path = build_dir / "vectorsum_v2_repro.cu"
    so_path = build_dir / "vectorsum_v2_repro.so"

    try:
        build_dir.mkdir(parents=True, exist_ok=True)
        if not so_path.exists():
            src_path.write_text(_CUDA_SRC)
            cmd = [
                nvcc,
                "-O3",
                "--use_fast_math",
                "-std=c++17",
                "-shared",
                "-Xcompiler",
                "-fPIC",
                f"-arch={arch}",
                str(src_path),
                "-o",
                str(so_path),
            ]
            subprocess.check_call(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        lib = CDLL(str(so_path))
        lib.launch_vectorsum_h100.argtypes = [
            c_void_p,
            c_void_p,
            c_longlong,
            c_void_p,
            c_int,
        ]
        lib.launch_vectorsum_h100.restype = c_int
        _LIB = lib
        return _LIB
    except Exception:
        _BUILD_FAILED = True
        return None


def _try_h100_fast_path(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor | None:
    if not x.is_cuda or x.dtype != torch.float32:
        return None
    if x.numel() != BIG_N or not x.is_contiguous():
        return None
    if out.numel() < 1 or not out.is_cuda or out.dtype != torch.float32:
        return None

    capability = torch.cuda.get_device_capability(x.device)
    if capability[0] < 9:
        return None

    x_ptr = x.data_ptr()
    out_ptr = out.data_ptr()
    use_float4 = (x_ptr % 16) == 0
    if not use_float4 and (x_ptr % 8) != 0:
        return None

    lib = _load_cuda_library()
    if lib is None:
        return None

    with torch.cuda.device(x.device):
        stream = torch.cuda.current_stream(x.device).cuda_stream
        err = lib.launch_vectorsum_h100(
            c_void_p(x_ptr),
            c_void_p(out_ptr),
            c_longlong(x.numel()),
            c_void_p(stream),
            c_int(1 if use_float4 else 0),
        )
    if err != 0:
        return None
    return out[0]


def custom_kernel(data: input_t) -> output_t:
    x, out = data
    fast = _try_h100_fast_path(x, out)
    if fast is not None:
        return fast
    return _fallback_sum(x)
