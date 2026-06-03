from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

_EXT = None
_ENABLE_STAGE_TIMING = False

_REQUIRED_WEIGHT_KEYS = (
    "norm.weight",
    "norm.bias",
    "left_proj.weight",
    "right_proj.weight",
    "left_gate.weight",
    "right_gate.weight",
    "out_gate.weight",
    "to_out_norm.weight",
    "to_out_norm.bias",
    "to_out.weight",
)


def _get_ext():
    global _EXT
    if _EXT is not None:
        return _EXT

    from torch.utils.cpp_extension import load_inline

    cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDABlas.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>

static inline void _ck(bool ok, const char* msg) {
  if (!ok) { throw std::runtime_error(msg); }
}

static inline void _ck_tensor_cuda_contig(const torch::Tensor& t) {
  _ck(t.is_cuda(), "tensor must be CUDA");
  _ck(t.is_contiguous(), "tensor must be contiguous");
}

static inline void _ck_rowmajor2d(const torch::Tensor& t, const char* name) {
  _ck(t.dim() == 2, "rowmajor2d dim mismatch");
  const int64_t n0 = t.size(0);
  const int64_t n1 = t.size(1);
  if (n0 > 0 && n1 > 0) {
    (void)name;
    _ck(t.stride(1) == 1, "rowmajor2d stride1 mismatch");
    _ck(t.stride(0) == n1, "rowmajor2d stride0 mismatch");
  }
}

static inline void _ck_rowmajor3d(const torch::Tensor& t, const char* name) {
  _ck(t.dim() == 3, "rowmajor3d dim mismatch");
  const int64_t n1 = t.size(1);
  const int64_t n2 = t.size(2);
  if (t.size(0) > 0 && n1 > 0 && n2 > 0) {
    (void)name;
    _ck(t.stride(2) == 1, "rowmajor3d stride2 mismatch");
    _ck(t.stride(1) == n2, "rowmajor3d stride1 mismatch");
    _ck(t.stride(0) == n1 * n2, "rowmajor3d stride0 mismatch");
  }
}

static inline void _ck_cublas(cublasStatus_t st) {
  if (st != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cublas call failed");
  }
}

static inline void _ck_cuda_last(const char* where) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(err));
  }
}

static inline cublasHandle_t _get_handle_tc() {
  cublasHandle_t h = at::cuda::getCurrentCUDABlasHandle();
  static thread_local cublasHandle_t last = nullptr;
  if (h != last) {
    _ck_cublas(cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH));
    last = h;
  }
  return h;
}

static inline cublasComputeType_t _get_ct_fast() {
#if defined(CUBLAS_COMPUTE_32F_FAST_16F)
  return CUBLAS_COMPUTE_32F_FAST_16F;
#else
  return CUBLAS_COMPUTE_32F;
#endif
}

#ifndef TRIMUL_STAGE_TIMING
#define TRIMUL_STAGE_TIMING 0
#endif

#if TRIMUL_STAGE_TIMING
struct _stage_timer {
  cudaEvent_t beg;
  cudaEvent_t end;

  _stage_timer() {
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
  }

  ~_stage_timer() {
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
  }

  __forceinline__ void tic() {
    cudaEventRecord(beg, 0);
  }

  __forceinline__ void toc(const char* tag) {
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, beg, end);
    printf("[trimul_t] %s %.3f ms\\n", tag, ms);
  }
};
#endif

// Sigmoid：保持与参考实现一致的 fast-math 路径
__device__ __forceinline__ float _sigmoid_f(float x) {
  return __fdividef(1.0f, 1.0f + __expf(-x));
}

__device__ __forceinline__ float2 _sigmoid_f2(float2 v) {
  v.x = _sigmoid_f(v.x);
  v.y = _sigmoid_f(v.y);
  return v;
}

__global__ void _mask_gate_lr_fuse_f16_vec4_f32(
    __half* __restrict__ left,
    __half* __restrict__ right,
    const __half* __restrict__ left_gate,
    const __half* __restrict__ right_gate,
    const float* __restrict__ mask,
    int inner) {
  const int d = (int)blockIdx.y;
  const int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  const int col = t << 2;
  if (col >= inner) return;
  const int idx = d * inner + col;

  if (col + 3 < inner) {
    const float4 mv = *(const float4*)(mask + col);
    const float m0 = mv.x;
    const float m1 = mv.y;
    const float m2 = mv.z;
    const float m3 = mv.w;

    const __half2 l2_0 = *(const __half2*)(left + idx);
    const __half2 l2_1 = *(const __half2*)(left + idx + 2);
    const __half2 r2_0 = *(const __half2*)(right + idx);
    const __half2 r2_1 = *(const __half2*)(right + idx + 2);
    const __half2 lg2_0 = *(const __half2*)(left_gate + idx);
    const __half2 lg2_1 = *(const __half2*)(left_gate + idx + 2);
    const __half2 rg2_0 = *(const __half2*)(right_gate + idx);
    const __half2 rg2_1 = *(const __half2*)(right_gate + idx + 2);

    const float2 gl0 = _sigmoid_f2(__half22float2(lg2_0));
    const float2 gl1 = _sigmoid_f2(__half22float2(lg2_1));
    const float2 gr0 = _sigmoid_f2(__half22float2(rg2_0));
    const float2 gr1 = _sigmoid_f2(__half22float2(rg2_1));

    float2 lv0 = __half22float2(l2_0);
    float2 lv1 = __half22float2(l2_1);
    float2 rv0 = __half22float2(r2_0);
    float2 rv1 = __half22float2(r2_1);

    lv0.x = lv0.x * m0 * gl0.x;
    lv0.y = lv0.y * m1 * gl0.y;
    lv1.x = lv1.x * m2 * gl1.x;
    lv1.y = lv1.y * m3 * gl1.y;

    rv0.x = rv0.x * m0 * gr0.x;
    rv0.y = rv0.y * m1 * gr0.y;
    rv1.x = rv1.x * m2 * gr1.x;
    rv1.y = rv1.y * m3 * gr1.y;

    *(__half2*)(left + idx) = __floats2half2_rn(lv0.x, lv0.y);
    *(__half2*)(left + idx + 2) = __floats2half2_rn(lv1.x, lv1.y);
    *(__half2*)(right + idx) = __floats2half2_rn(rv0.x, rv0.y);
    *(__half2*)(right + idx + 2) = __floats2half2_rn(rv1.x, rv1.y);
  } else {
    #pragma unroll
    for (int off = 0; off < 4; ++off) {
      const int c = col + off;
      if (c < inner) {
        const float m = mask[c];
        const int id = idx + off;
        float l = __half2float(left[id]) * m;
        float r = __half2float(right[id]) * m;
        const float gl = _sigmoid_f(__half2float(left_gate[id]));
        const float gr = _sigmoid_f(__half2float(right_gate[id]));
        l *= gl;
        r *= gr;
        left[id] = __float2half_rn(l);
        right[id] = __float2half_rn(r);
      }
    }
  }
}

void apply_mask_gate_lr_f16(torch::Tensor left,
                            torch::Tensor right,
                            torch::Tensor left_gate,
                            torch::Tensor right_gate,
                            torch::Tensor mask) {
  _ck_tensor_cuda_contig(left);
  _ck_tensor_cuda_contig(right);
  _ck_tensor_cuda_contig(left_gate);
  _ck_tensor_cuda_contig(right_gate);
  _ck_tensor_cuda_contig(mask);

  _ck(left.dtype() == torch::kFloat16, "left must be float16");
  _ck(right.dtype() == torch::kFloat16, "right must be float16");
  _ck(left_gate.dtype() == torch::kFloat16, "left_gate must be float16");
  _ck(right_gate.dtype() == torch::kFloat16, "right_gate must be float16");
  _ck(mask.dtype() == torch::kFloat32, "mask must be float32");
  _ck(mask.dim() == 3, "mask must be 3D");
  _ck_rowmajor3d(mask, "mask");

  const int hidden = (int)left.size(0);
  _ck(hidden == 128, "hidden_dim must be 128");
  _ck(right.numel() == left.numel(), "lr size mismatch");
  _ck(left_gate.numel() == left.numel(), "lg size mismatch");
  _ck(right_gate.numel() == left.numel(), "rg size mismatch");

  const int64_t inner64 = mask.numel();
  _ck(inner64 > 0 && inner64 <= INT_MAX, "mask too large");
  const int inner = (int)inner64;
  _ck((int64_t)hidden * (int64_t)inner == left.numel(), "mask/hidden mismatch");

  const int quads = (inner + 3) >> 2;
  const dim3 block(256, 1, 1);
  const dim3 grid((quads + (int)block.x - 1) / (int)block.x, hidden, 1);
  _mask_gate_lr_fuse_f16_vec4_f32<<<grid, block>>>(
      (__half*)left.data_ptr<at::Half>(),
      (__half*)right.data_ptr<at::Half>(),
      (const __half*)left_gate.data_ptr<at::Half>(),
      (const __half*)right_gate.data_ptr<at::Half>(),
      (const float*)mask.data_ptr<float>(),
      inner);
  _ck_cuda_last("mask_gate_lr_f32");
}

// X: [M, K] 行主序（f16）
// W: [N, K] 行主序（f16）
// Y: [M, N] 行主序（f16）
torch::Tensor gemm_f16(torch::Tensor x, torch::Tensor w) {
  _ck_tensor_cuda_contig(x);
  _ck_tensor_cuda_contig(w);
  _ck(x.dtype() == torch::kFloat16, "x must be float16");
  _ck(w.dtype() == torch::kFloat16, "w must be float16");
  _ck(x.dim() == 2, "x must be 2D");
  _ck(w.dim() == 2, "w must be 2D");
  _ck_rowmajor2d(x, "x");
  _ck_rowmajor2d(w, "w");

  const int64_t M64 = x.size(0);
  const int64_t K64 = x.size(1);
  const int64_t N64 = w.size(0);
  _ck(w.size(1) == K64, "w shape mismatch");
  _ck(M64 > 0 && N64 > 0 && K64 > 0, "empty mat");
  _ck(M64 <= INT_MAX && N64 <= INT_MAX && K64 <= INT_MAX, "mat too large");

  auto y = torch::empty({M64, N64}, x.options());

  const int M = (int)M64;
  const int N = (int)N64;
  const int K = (int)K64;

#if TRIMUL_STAGE_TIMING
  printf("[trimul_shape] gemm_f16 M=%d N=%d K=%d\\n", M, N, K);
#endif

  cublasHandle_t handle = _get_handle_tc();
  const cublasComputeType_t ct = _get_ct_fast();

  const float alpha = 1.0f;
  const float beta = 0.0f;

  _ck_cublas(
      cublasGemmEx(
          handle,
          CUBLAS_OP_T, CUBLAS_OP_N,
          N, M, K,
          &alpha,
          w.data_ptr<at::Half>(), CUDA_R_16F, K,
          x.data_ptr<at::Half>(), CUDA_R_16F, K,
          &beta,
          y.data_ptr<at::Half>(), CUDA_R_16F, N,
          ct,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  return y;
}

// A: [B, M, K] 行主序（f16）
// B: [B, N, K] 行主序（f16）
// Y: [B, M, N] 行主序（f16，f32 累加）
void gemm_sb_f16_out(torch::Tensor a, torch::Tensor b, torch::Tensor y) {
  _ck_tensor_cuda_contig(a);
  _ck_tensor_cuda_contig(b);
  _ck_tensor_cuda_contig(y);
  _ck(a.dtype() == torch::kFloat16, "a must be float16");
  _ck(b.dtype() == torch::kFloat16, "b must be float16");
  _ck(y.dtype() == torch::kFloat16, "y must be float16");
  _ck(a.dim() == 3, "a must be 3D");
  _ck(b.dim() == 3, "b must be 3D");
  _ck(y.dim() == 3, "y must be 3D");
  _ck_rowmajor3d(a, "a");
  _ck_rowmajor3d(b, "b");
  _ck_rowmajor3d(y, "y");

  const int64_t B64 = a.size(0);
  const int64_t M64 = a.size(1);
  const int64_t K64 = a.size(2);
  _ck(b.size(0) == B64, "batch mismatch");
  _ck(b.size(2) == K64, "k mismatch");
  const int64_t N64 = b.size(1);
  _ck(y.size(0) == B64 && y.size(1) == M64 && y.size(2) == N64, "y shape mismatch");

  _ck(B64 > 0 && M64 > 0 && N64 > 0 && K64 > 0, "empty batched gemm");
  _ck(B64 <= INT_MAX && M64 <= INT_MAX && N64 <= INT_MAX && K64 <= INT_MAX, "batched gemm too large");

  const int Bc = (int)B64;
  const int M = (int)M64;
  const int N = (int)N64;
  const int K = (int)K64;

#if TRIMUL_STAGE_TIMING
  printf("[trimul_shape] gemm_sb B=%d M=%d N=%d K=%d\\n", Bc, M, N, K);
#endif

  cublasHandle_t handle = _get_handle_tc();
  const cublasComputeType_t ct = _get_ct_fast();

  const float alpha = 1.0f;
  const float beta = 0.0f;

  const long long strideA = (long long)N64 * (long long)K64;
  const long long strideB = (long long)M64 * (long long)K64;
  const long long strideC = (long long)M64 * (long long)N64;

  _ck_cublas(
      cublasGemmStridedBatchedEx(
          handle,
          CUBLAS_OP_T, CUBLAS_OP_N,
          N, M, K,
          &alpha,
          b.data_ptr<at::Half>(), CUDA_R_16F, K, strideA,
          a.data_ptr<at::Half>(), CUDA_R_16F, K, strideB,
          &beta,
          y.data_ptr<at::Half>(), CUDA_R_16F, N, strideC,
          Bc,
          ct,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

__device__ __forceinline__ float _warp_reduce_sum(float v) {
  v += __shfl_down_sync(0xffffffff, v, 16);
  v += __shfl_down_sync(0xffffffff, v, 8);
  v += __shfl_down_sync(0xffffffff, v, 4);
  v += __shfl_down_sync(0xffffffff, v, 2);
  v += __shfl_down_sync(0xffffffff, v, 1);
  return v;
}

template <int D>
__global__ void _ln_fwd_f16_warp4_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    __half* __restrict__ y,
    int rows) {
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warps = (int)blockDim.x >> 5;
  const int row = (int)blockIdx.x * warps + warp;
  if (row >= rows) return;

  const int base = row * D;

  const int off0 = lane << 2;
  float4 v0 = *(const float4*)(x + base + off0);
  float sum = (v0.x + v0.y) + (v0.z + v0.w);
  float sumsq = (v0.x * v0.x + v0.y * v0.y) + (v0.z * v0.z + v0.w * v0.w);

  float4 v1, v2, v3, v4, v5, v6, v7;
  if constexpr (D >= 256) {
    v1 = *(const float4*)(x + base + 128 + off0);
    sum += (v1.x + v1.y) + (v1.z + v1.w);
    sumsq += (v1.x * v1.x + v1.y * v1.y) + (v1.z * v1.z + v1.w * v1.w);
  }
  if constexpr (D >= 384) {
    v2 = *(const float4*)(x + base + 256 + off0);
    sum += (v2.x + v2.y) + (v2.z + v2.w);
    sumsq += (v2.x * v2.x + v2.y * v2.y) + (v2.z * v2.z + v2.w * v2.w);
  }
  if constexpr (D >= 512) {
    v3 = *(const float4*)(x + base + 384 + off0);
    sum += (v3.x + v3.y) + (v3.z + v3.w);
    sumsq += (v3.x * v3.x + v3.y * v3.y) + (v3.z * v3.z + v3.w * v3.w);
  }
  if constexpr (D >= 640) {
    v4 = *(const float4*)(x + base + 512 + off0);
    sum += (v4.x + v4.y) + (v4.z + v4.w);
    sumsq += (v4.x * v4.x + v4.y * v4.y) + (v4.z * v4.z + v4.w * v4.w);
  }
  if constexpr (D >= 768) {
    v5 = *(const float4*)(x + base + 640 + off0);
    sum += (v5.x + v5.y) + (v5.z + v5.w);
    sumsq += (v5.x * v5.x + v5.y * v5.y) + (v5.z * v5.z + v5.w * v5.w);
  }
  if constexpr (D >= 896) {
    v6 = *(const float4*)(x + base + 768 + off0);
    sum += (v6.x + v6.y) + (v6.z + v6.w);
    sumsq += (v6.x * v6.x + v6.y * v6.y) + (v6.z * v6.z + v6.w * v6.w);
  }
  if constexpr (D >= 1024) {
    v7 = *(const float4*)(x + base + 896 + off0);
    sum += (v7.x + v7.y) + (v7.z + v7.w);
    sumsq += (v7.x * v7.x + v7.y * v7.y) + (v7.z * v7.z + v7.w * v7.w);
  }

  const float sum_r = _warp_reduce_sum(sum);
  const float sumsq_r = _warp_reduce_sum(sumsq);

  const float inv_d = 1.0f / (float)D;
  const float sum_t = __shfl_sync(0xffffffff, sum_r, 0);
  const float sumsq_t = __shfl_sync(0xffffffff, sumsq_r, 0);
  const float mean = sum_t * inv_d;
  const float var = sumsq_t * inv_d - mean * mean;
  const float inv = rsqrtf(var + 1.0e-5f);

  float4 w0 = *(const float4*)(w + off0);
  float4 b0 = *(const float4*)(b + off0);

  float4 o0;
  o0.x = (v0.x - mean) * inv * w0.x + b0.x;
  o0.y = (v0.y - mean) * inv * w0.y + b0.y;
  o0.z = (v0.z - mean) * inv * w0.z + b0.z;
  o0.w = (v0.w - mean) * inv * w0.w + b0.w;

  *(__half2*)(y + base + off0) = __floats2half2_rn(o0.x, o0.y);
  *(__half2*)(y + base + off0 + 2) = __floats2half2_rn(o0.z, o0.w);

  if constexpr (D >= 256) {
    float4 w1 = *(const float4*)(w + 128 + off0);
    float4 b1 = *(const float4*)(b + 128 + off0);
    float4 o1;
    o1.x = (v1.x - mean) * inv * w1.x + b1.x;
    o1.y = (v1.y - mean) * inv * w1.y + b1.y;
    o1.z = (v1.z - mean) * inv * w1.z + b1.z;
    o1.w = (v1.w - mean) * inv * w1.w + b1.w;
    *(__half2*)(y + base + 128 + off0) = __floats2half2_rn(o1.x, o1.y);
    *(__half2*)(y + base + 128 + off0 + 2) = __floats2half2_rn(o1.z, o1.w);
  }
  if constexpr (D >= 384) {
    float4 w2 = *(const float4*)(w + 256 + off0);
    float4 b2 = *(const float4*)(b + 256 + off0);
    float4 o2;
    o2.x = (v2.x - mean) * inv * w2.x + b2.x;
    o2.y = (v2.y - mean) * inv * w2.y + b2.y;
    o2.z = (v2.z - mean) * inv * w2.z + b2.z;
    o2.w = (v2.w - mean) * inv * w2.w + b2.w;
    *(__half2*)(y + base + 256 + off0) = __floats2half2_rn(o2.x, o2.y);
    *(__half2*)(y + base + 256 + off0 + 2) = __floats2half2_rn(o2.z, o2.w);
  }
  if constexpr (D >= 512) {
    float4 w3 = *(const float4*)(w + 384 + off0);
    float4 b3 = *(const float4*)(b + 384 + off0);
    float4 o3;
    o3.x = (v3.x - mean) * inv * w3.x + b3.x;
    o3.y = (v3.y - mean) * inv * w3.y + b3.y;
    o3.z = (v3.z - mean) * inv * w3.z + b3.z;
    o3.w = (v3.w - mean) * inv * w3.w + b3.w;
    *(__half2*)(y + base + 384 + off0) = __floats2half2_rn(o3.x, o3.y);
    *(__half2*)(y + base + 384 + off0 + 2) = __floats2half2_rn(o3.z, o3.w);
  }
  if constexpr (D >= 640) {
    float4 w4 = *(const float4*)(w + 512 + off0);
    float4 b4 = *(const float4*)(b + 512 + off0);
    float4 o4;
    o4.x = (v4.x - mean) * inv * w4.x + b4.x;
    o4.y = (v4.y - mean) * inv * w4.y + b4.y;
    o4.z = (v4.z - mean) * inv * w4.z + b4.z;
    o4.w = (v4.w - mean) * inv * w4.w + b4.w;
    *(__half2*)(y + base + 512 + off0) = __floats2half2_rn(o4.x, o4.y);
    *(__half2*)(y + base + 512 + off0 + 2) = __floats2half2_rn(o4.z, o4.w);
  }
  if constexpr (D >= 768) {
    float4 w5 = *(const float4*)(w + 640 + off0);
    float4 b5 = *(const float4*)(b + 640 + off0);
    float4 o5;
    o5.x = (v5.x - mean) * inv * w5.x + b5.x;
    o5.y = (v5.y - mean) * inv * w5.y + b5.y;
    o5.z = (v5.z - mean) * inv * w5.z + b5.z;
    o5.w = (v5.w - mean) * inv * w5.w + b5.w;
    *(__half2*)(y + base + 640 + off0) = __floats2half2_rn(o5.x, o5.y);
    *(__half2*)(y + base + 640 + off0 + 2) = __floats2half2_rn(o5.z, o5.w);
  }
  if constexpr (D >= 896) {
    float4 w6 = *(const float4*)(w + 768 + off0);
    float4 b6 = *(const float4*)(b + 768 + off0);
    float4 o6;
    o6.x = (v6.x - mean) * inv * w6.x + b6.x;
    o6.y = (v6.y - mean) * inv * w6.y + b6.y;
    o6.z = (v6.z - mean) * inv * w6.z + b6.z;
    o6.w = (v6.w - mean) * inv * w6.w + b6.w;
    *(__half2*)(y + base + 768 + off0) = __floats2half2_rn(o6.x, o6.y);
    *(__half2*)(y + base + 768 + off0 + 2) = __floats2half2_rn(o6.z, o6.w);
  }
  if constexpr (D >= 1024) {
    float4 w7 = *(const float4*)(w + 896 + off0);
    float4 b7 = *(const float4*)(b + 896 + off0);
    float4 o7;
    o7.x = (v7.x - mean) * inv * w7.x + b7.x;
    o7.y = (v7.y - mean) * inv * w7.y + b7.y;
    o7.z = (v7.z - mean) * inv * w7.z + b7.z;
    o7.w = (v7.w - mean) * inv * w7.w + b7.w;
    *(__half2*)(y + base + 896 + off0) = __floats2half2_rn(o7.x, o7.y);
    *(__half2*)(y + base + 896 + off0 + 2) = __floats2half2_rn(o7.z, o7.w);
  }
}

__global__ void _ln_fwd_f16_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    __half* __restrict__ y,
    int rows,
    int d) {
  const int row = (int)blockIdx.x;
  if (row >= rows) return;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  const int base = row * d;

  float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
  const int i0 = tid;
  const int i1 = tid + 128;
  const int i2 = tid + 256;
  const int i3 = tid + 384;
  const bool p0 = (i0 < d);
  const bool p1 = (i1 < d);
  const bool p2 = (i2 < d);
  const bool p3 = (i3 < d);
  if (p0) v0 = x[base + i0];
  if (p1) v1 = x[base + i1];
  if (p2) v2 = x[base + i2];
  if (p3) v3 = x[base + i3];

  float sum = 0.0f;
  float sumsq = 0.0f;
  if (p0) { sum += v0; sumsq += v0 * v0; }
  if (p1) { sum += v1; sumsq += v1 * v1; }
  if (p2) { sum += v2; sumsq += v2 * v2; }
  if (p3) { sum += v3; sumsq += v3 * v3; }

  for (int k = tid + 512; k < d; k += 128) {
    const float v = x[base + k];
    sum += v;
    sumsq += v * v;
  }

  sum = _warp_reduce_sum(sum);
  sumsq = _warp_reduce_sum(sumsq);

  __shared__ float warp_sum[4];
  __shared__ float warp_sumsq[4];
  __shared__ float mean_s;
  __shared__ float inv_s;

  if (lane == 0) {
    warp_sum[warp] = sum;
    warp_sumsq[warp] = sumsq;
  }
  __syncthreads();

  if (warp == 0) {
    float s0 = (lane < 4) ? warp_sum[lane] : 0.0f;
    float s1 = (lane < 4) ? warp_sumsq[lane] : 0.0f;
    s0 = _warp_reduce_sum(s0);
    s1 = _warp_reduce_sum(s1);
    if (lane == 0) {
      const float inv_d = 1.0f / (float)d;
      const float mean = s0 * inv_d;
      const float var = s1 * inv_d - mean * mean;
      mean_s = mean;
      inv_s = rsqrtf(var + 1.0e-5f);
    }
  }
  __syncthreads();

  const float mean = mean_s;
  const float inv = inv_s;

  if (p0) {
    const float o = (v0 - mean) * inv * w[i0] + b[i0];
    y[base + i0] = __float2half_rn(o);
  }
  if (p1) {
    const float o = (v1 - mean) * inv * w[i1] + b[i1];
    y[base + i1] = __float2half_rn(o);
  }
  if (p2) {
    const float o = (v2 - mean) * inv * w[i2] + b[i2];
    y[base + i2] = __float2half_rn(o);
  }
  if (p3) {
    const float o = (v3 - mean) * inv * w[i3] + b[i3];
    y[base + i3] = __float2half_rn(o);
  }
  for (int k = tid + 512; k < d; k += 128) {
    const float v = x[base + k];
    const float o = (v - mean) * inv * w[k] + b[k];
    y[base + k] = __float2half_rn(o);
  }
}

torch::Tensor ln_fwd_f16(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
  _ck_tensor_cuda_contig(x);
  _ck_tensor_cuda_contig(w);
  _ck_tensor_cuda_contig(b);
  _ck(x.dtype() == torch::kFloat32, "x must be float32");
  _ck(w.dtype() == torch::kFloat32, "w must be float32");
  _ck(b.dtype() == torch::kFloat32, "b must be float32");
  _ck(w.dim() == 1, "w must be 1D");
  _ck(b.dim() == 1, "b must be 1D");

  const int64_t d64 = w.numel();
  _ck(d64 == b.numel(), "w/b mismatch");
  _ck(d64 > 0 && d64 <= INT_MAX, "bad d");
  const int d = (int)d64;
  _ck(x.size(-1) == d64, "x last dim mismatch");

  auto y = torch::empty_like(x, x.options().dtype(torch::kFloat16));
  const int64_t rows64 = x.numel() / d64;
  _ck(rows64 > 0 && rows64 <= INT_MAX, "bad rows");
  const int rows = (int)rows64;

  if (d == 128 || d == 256 || d == 384 || d == 512 || d == 640 || d == 768 || d == 896 || d == 1024) {
    const dim3 block(256, 1, 1);
    const int warps = (int)block.x >> 5;
    const dim3 grid((rows + warps - 1) / warps, 1, 1);
    if (d == 128) {
      _ln_fwd_f16_warp4_kernel<128><<<grid, block>>>(
          x.data_ptr<float>(),
          w.data_ptr<float>(),
          b.data_ptr<float>(),
          (__half*)y.data_ptr<at::Half>(),
          rows);
    } else if (d == 256) {
      _ln_fwd_f16_warp4_kernel<256><<<grid, block>>>(
          x.data_ptr<float>(),
          w.data_ptr<float>(),
          b.data_ptr<float>(),
          (__half*)y.data_ptr<at::Half>(),
          rows);
    } else if (d == 384) {
      _ln_fwd_f16_warp4_kernel<384><<<grid, block>>>(
          x.data_ptr<float>(),
          w.data_ptr<float>(),
          b.data_ptr<float>(),
          (__half*)y.data_ptr<at::Half>(),
          rows);
    } else if (d == 512) {
      _ln_fwd_f16_warp4_kernel<512><<<grid, block>>>(
          x.data_ptr<float>(),
          w.data_ptr<float>(),
          b.data_ptr<float>(),
          (__half*)y.data_ptr<at::Half>(),
          rows);
    } else if (d == 640) {
      _ln_fwd_f16_warp4_kernel<640><<<grid, block>>>(
          x.data_ptr<float>(),
          w.data_ptr<float>(),
          b.data_ptr<float>(),
          (__half*)y.data_ptr<at::Half>(),
          rows);
    } else if (d == 768) {
      _ln_fwd_f16_warp4_kernel<768><<<grid, block>>>(
          x.data_ptr<float>(),
          w.data_ptr<float>(),
          b.data_ptr<float>(),
          (__half*)y.data_ptr<at::Half>(),
          rows);
    } else if (d == 896) {
      _ln_fwd_f16_warp4_kernel<896><<<grid, block>>>(
          x.data_ptr<float>(),
          w.data_ptr<float>(),
          b.data_ptr<float>(),
          (__half*)y.data_ptr<at::Half>(),
          rows);
    } else {
      _ln_fwd_f16_warp4_kernel<1024><<<grid, block>>>(
          x.data_ptr<float>(),
          w.data_ptr<float>(),
          b.data_ptr<float>(),
          (__half*)y.data_ptr<at::Half>(),
          rows);
    }
  } else {
    const dim3 block(128, 1, 1);
    const dim3 grid(rows, 1, 1);
    _ln_fwd_f16_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        (__half*)y.data_ptr<at::Half>(),
        rows,
        d);
  }
  _ck_cuda_last("ln_fwd_f16");
  return y;
}

__global__ void _pack5_f32_to_f16_vec2_kernel(
    const float* __restrict__ w0,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    const float* __restrict__ w3,
    const float* __restrict__ w4,
    __half* __restrict__ out,
    int elems_per_mat) {
  const int g = (int)blockIdx.y;
  const int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  const int i = t << 1;
  if (i >= elems_per_mat) return;
  const float* src = nullptr;
  if (g == 0) src = w0;
  else if (g == 1) src = w1;
  else if (g == 2) src = w2;
  else if (g == 3) src = w3;
  else src = w4;

  const int o = g * elems_per_mat + i;
  if (i + 1 < elems_per_mat) {
    const float2 v = *(const float2*)(src + i);
    *(__half2*)(out + o) = __floats2half2_rn(v.x, v.y);
  } else {
    out[o] = __float2half_rn(src[i]);
  }
}

torch::Tensor pack_w5_f16(torch::Tensor w0,
                          torch::Tensor w1,
                          torch::Tensor w2,
                          torch::Tensor w3,
                          torch::Tensor w4) {
  _ck_tensor_cuda_contig(w0);
  _ck_tensor_cuda_contig(w1);
  _ck_tensor_cuda_contig(w2);
  _ck_tensor_cuda_contig(w3);
  _ck_tensor_cuda_contig(w4);
  _ck(w0.dtype() == torch::kFloat32, "w0 must be float32");
  _ck(w1.dtype() == torch::kFloat32, "w1 must be float32");
  _ck(w2.dtype() == torch::kFloat32, "w2 must be float32");
  _ck(w3.dtype() == torch::kFloat32, "w3 must be float32");
  _ck(w4.dtype() == torch::kFloat32, "w4 must be float32");
  _ck(w0.dim() == 2, "w0 must be 2D");
  _ck(w1.dim() == 2, "w1 must be 2D");
  _ck(w2.dim() == 2, "w2 must be 2D");
  _ck(w3.dim() == 2, "w3 must be 2D");
  _ck(w4.dim() == 2, "w4 must be 2D");

  const int64_t h64 = w0.size(0);
  const int64_t d64 = w0.size(1);
  _ck(h64 == 128, "hidden_dim must be 128");
  _ck(w1.sizes() == w0.sizes(), "w1 shape mismatch");
  _ck(w2.sizes() == w0.sizes(), "w2 shape mismatch");
  _ck(w3.sizes() == w0.sizes(), "w3 shape mismatch");
  _ck(w4.sizes() == w0.sizes(), "w4 shape mismatch");

  const int64_t elems64 = h64 * d64;
  _ck(elems64 > 0 && elems64 <= INT_MAX, "weight too large");
  const int elems = (int)elems64;

  auto out = torch::empty({5 * h64, d64}, w0.options().dtype(torch::kFloat16));

  const int pairs = (elems + 1) >> 1;
  const dim3 block(256, 1, 1);
  const dim3 grid((pairs + (int)block.x - 1) / (int)block.x, 5, 1);
  _pack5_f32_to_f16_vec2_kernel<<<grid, block>>>(
      w0.data_ptr<float>(),
      w1.data_ptr<float>(),
      w2.data_ptr<float>(),
      w3.data_ptr<float>(),
      w4.data_ptr<float>(),
      (__half*)out.data_ptr<at::Half>(),
      elems);
  _ck_cuda_last("pack_w5_f16");
  return out;
}

__global__ void _cast_f32_to_f16_vec2_kernel(
    const float* __restrict__ src,
    __half* __restrict__ dst,
    int n) {
  const int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  const int i = t << 1;
  if (i >= n) return;
  if (i + 1 < n) {
    const float2 v = *(const float2*)(src + i);
    *(__half2*)(dst + i) = __floats2half2_rn(v.x, v.y);
  } else {
    dst[i] = __float2half_rn(src[i]);
  }
}

torch::Tensor cast_f32_to_f16(torch::Tensor x) {
  _ck_tensor_cuda_contig(x);
  _ck(x.dtype() == torch::kFloat32, "x must be float32");
  const int64_t n64 = x.numel();
  _ck(n64 > 0 && n64 <= INT_MAX, "x too large");
  const int n = (int)n64;
  auto y = torch::empty_like(x, x.options().dtype(torch::kFloat16));
  const int pairs = (n + 1) >> 1;
  const dim3 block(256, 1, 1);
  const dim3 grid((pairs + (int)block.x - 1) / (int)block.x, 1, 1);
  _cast_f32_to_f16_vec2_kernel<<<grid, block>>>(
      x.data_ptr<float>(),
      (__half*)y.data_ptr<at::Half>(),
      n);
  _ck_cuda_last("cast_f32_to_f16");
  return y;
}

__global__ void _pack5_and_cast_to_out_f32_to_f16_vec4_kernel(
    const float* __restrict__ w0,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    const float* __restrict__ w3,
    const float* __restrict__ w4,
    const float* __restrict__ w_to_out,
    __half* __restrict__ out_pack,
    __half* __restrict__ out_to_out,
    int elems_per_mat) {
  const int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  const int i = t << 2;
  if (i >= elems_per_mat) return;

  if (i + 3 < elems_per_mat) {
    const float4 v0 = *(const float4*)(w0 + i);
    const float4 v1 = *(const float4*)(w1 + i);
    const float4 v2 = *(const float4*)(w2 + i);
    const float4 v3 = *(const float4*)(w3 + i);
    const float4 v4 = *(const float4*)(w4 + i);
    const float4 vt = *(const float4*)(w_to_out + i);

    const int o0 = 0 * elems_per_mat + i;
    const int o1 = 1 * elems_per_mat + i;
    const int o2 = 2 * elems_per_mat + i;
    const int o3 = 3 * elems_per_mat + i;
    const int o4 = 4 * elems_per_mat + i;

    *(__half2*)(out_pack + o0) = __floats2half2_rn(v0.x, v0.y);
    *(__half2*)(out_pack + o0 + 2) = __floats2half2_rn(v0.z, v0.w);
    *(__half2*)(out_pack + o1) = __floats2half2_rn(v1.x, v1.y);
    *(__half2*)(out_pack + o1 + 2) = __floats2half2_rn(v1.z, v1.w);
    *(__half2*)(out_pack + o2) = __floats2half2_rn(v2.x, v2.y);
    *(__half2*)(out_pack + o2 + 2) = __floats2half2_rn(v2.z, v2.w);
    *(__half2*)(out_pack + o3) = __floats2half2_rn(v3.x, v3.y);
    *(__half2*)(out_pack + o3 + 2) = __floats2half2_rn(v3.z, v3.w);
    *(__half2*)(out_pack + o4) = __floats2half2_rn(v4.x, v4.y);
    *(__half2*)(out_pack + o4 + 2) = __floats2half2_rn(v4.z, v4.w);

    *(__half2*)(out_to_out + i) = __floats2half2_rn(vt.x, vt.y);
    *(__half2*)(out_to_out + i + 2) = __floats2half2_rn(vt.z, vt.w);
  } else {
    #pragma unroll
    for (int off = 0; off < 4; ++off) {
      const int j = i + off;
      if (j < elems_per_mat) {
        const float a0 = w0[j];
        const float a1 = w1[j];
        const float a2 = w2[j];
        const float a3 = w3[j];
        const float a4 = w4[j];
        const float at = w_to_out[j];

        out_pack[0 * elems_per_mat + j] = __float2half_rn(a0);
        out_pack[1 * elems_per_mat + j] = __float2half_rn(a1);
        out_pack[2 * elems_per_mat + j] = __float2half_rn(a2);
        out_pack[3 * elems_per_mat + j] = __float2half_rn(a3);
        out_pack[4 * elems_per_mat + j] = __float2half_rn(a4);
        out_to_out[j] = __float2half_rn(at);
      }
    }
  }
}

void pack_w5_to_out_f16_out(torch::Tensor w0,
                            torch::Tensor w1,
                            torch::Tensor w2,
                            torch::Tensor w3,
                            torch::Tensor w4,
                            torch::Tensor w_to_out,
                            torch::Tensor out_pack,
                            torch::Tensor out_to_out) {
  _ck_tensor_cuda_contig(w0);
  _ck_tensor_cuda_contig(w1);
  _ck_tensor_cuda_contig(w2);
  _ck_tensor_cuda_contig(w3);
  _ck_tensor_cuda_contig(w4);
  _ck_tensor_cuda_contig(w_to_out);
  _ck_tensor_cuda_contig(out_pack);
  _ck_tensor_cuda_contig(out_to_out);

  _ck(w0.dtype() == torch::kFloat32, "w0 must be float32");
  _ck(w1.dtype() == torch::kFloat32, "w1 must be float32");
  _ck(w2.dtype() == torch::kFloat32, "w2 must be float32");
  _ck(w3.dtype() == torch::kFloat32, "w3 must be float32");
  _ck(w4.dtype() == torch::kFloat32, "w4 must be float32");
  _ck(w_to_out.dtype() == torch::kFloat32, "w_to_out must be float32");
  _ck(out_pack.dtype() == torch::kFloat16, "out_pack must be float16");
  _ck(out_to_out.dtype() == torch::kFloat16, "out_to_out must be float16");

  _ck(w0.dim() == 2, "w0 must be 2D");
  _ck(w1.dim() == 2, "w1 must be 2D");
  _ck(w2.dim() == 2, "w2 must be 2D");
  _ck(w3.dim() == 2, "w3 must be 2D");
  _ck(w4.dim() == 2, "w4 must be 2D");
  _ck(w_to_out.dim() == 2, "w_to_out must be 2D");

  const int64_t hidden64 = w0.size(0);
  const int64_t dim64 = w0.size(1);
  _ck(hidden64 == 128, "hidden_dim must be 128");
  _ck(w1.sizes() == w0.sizes(), "w1 shape mismatch");
  _ck(w2.sizes() == w0.sizes(), "w2 shape mismatch");
  _ck(w3.sizes() == w0.sizes(), "w3 shape mismatch");
  _ck(w4.sizes() == w0.sizes(), "w4 shape mismatch");
  _ck(w_to_out.size(0) == dim64 && w_to_out.size(1) == hidden64, "w_to_out shape mismatch");

  _ck(out_pack.dim() == 2, "out_pack must be 2D");
  _ck(out_to_out.dim() == 2, "out_to_out must be 2D");
  _ck(out_pack.size(0) == 5 * hidden64 && out_pack.size(1) == dim64, "out_pack shape mismatch");
  _ck(out_to_out.size(0) == dim64 && out_to_out.size(1) == hidden64, "out_to_out shape mismatch");

  const int64_t elems64 = hidden64 * dim64;
  _ck(elems64 > 0 && elems64 <= INT_MAX, "weight too large");
  const int elems = (int)elems64;

  const int quads = (elems + 3) >> 2;
  const dim3 block(256, 1, 1);
  const dim3 grid((quads + (int)block.x - 1) / (int)block.x, 1, 1);
  _pack5_and_cast_to_out_f32_to_f16_vec4_kernel<<<grid, block>>>(
      w0.data_ptr<float>(),
      w1.data_ptr<float>(),
      w2.data_ptr<float>(),
      w3.data_ptr<float>(),
      w4.data_ptr<float>(),
      w_to_out.data_ptr<float>(),
      (__half*)out_pack.data_ptr<at::Half>(),
      (__half*)out_to_out.data_ptr<at::Half>(),
      elems);
  _ck_cuda_last("pack_w5_to_out_f16_out");
}

__global__ void _ln_gate_transpose_f16_new_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ g,
    const float* __restrict__ w,
    const float* __restrict__ b,
    __half* __restrict__ y,
    int inner) {
  const int tx = (int)threadIdx.x;
  const int ty = (int)threadIdx.y;
  const int col0 = (int)blockIdx.x * 32;
  const int col = col0 + tx;

  const int tid = ty * 32 + tx;

  __shared__ float sw[128];
  __shared__ float sb[128];
  if (tid < 128) {
    sw[tid] = w[tid];
    sb[tid] = b[tid];
  }

  __shared__ __half sx[128][33];
  __shared__ __half sg[128][33];

  float psum = 0.0f;
  float psumsq = 0.0f;

  #pragma unroll
  for (int k = 0; k < 32; ++k) {
    const int d = ty + (k << 2);
    __half xh = __float2half_rn(0.0f);
    __half gh = __float2half_rn(0.0f);
    float xv = 0.0f;
    if (col < inner) {
      xh = x[d * inner + col];
      gh = g[d * inner + col];
      xv = __half2float(xh);
    }
    sx[d][tx] = xh;
    sg[d][tx] = gh;
    psum += xv;
    psumsq += xv * xv;
  }

  __shared__ float ssum[4][32];
  __shared__ float ssumsq[4][32];
  ssum[ty][tx] = psum;
  ssumsq[ty][tx] = psumsq;
  __syncthreads();

  __shared__ float smean[32];
  __shared__ float sinv[32];
  if (ty == 0) {
    const float sum = ssum[0][tx] + ssum[1][tx] + ssum[2][tx] + ssum[3][tx];
    const float sumsq = ssumsq[0][tx] + ssumsq[1][tx] + ssumsq[2][tx] + ssumsq[3][tx];
    const float inv_d = 1.0f / 128.0f;
    const float mean = sum * inv_d;
    const float var = sumsq * inv_d - mean * mean;
    smean[tx] = mean;
    sinv[tx] = rsqrtf(var + 1.0e-5f);
  }
  __syncthreads();

  const int d0 = tid << 1;
  if (d0 + 1 < 128) {
    #pragma unroll
    for (int c = 0; c < 32; ++c) {
      const int cc = col0 + c;
      if (cc < inner) {
        const float mean = smean[c];
        const float inv = sinv[c];
        const float xv0 = __half2float(sx[d0][c]);
        const float gv0 = __half2float(sg[d0][c]);
        const float go0 = _sigmoid_f(gv0);
        const float o0 = ((xv0 - mean) * inv * sw[d0] + sb[d0]) * go0;

        const int d1 = d0 + 1;
        const float xv1 = __half2float(sx[d1][c]);
        const float gv1 = __half2float(sg[d1][c]);
        const float go1 = _sigmoid_f(gv1);
        const float o1 = ((xv1 - mean) * inv * sw[d1] + sb[d1]) * go1;

        *(__half2*)(y + cc * 128 + d0) = __floats2half2_rn(o0, o1);
      }
    }
  }
}

void ln_gate_transpose_f16_out(torch::Tensor x,
                               torch::Tensor w,
                               torch::Tensor b,
                               torch::Tensor g,
                               torch::Tensor y) {
  _ck_tensor_cuda_contig(x);
  _ck_tensor_cuda_contig(w);
  _ck_tensor_cuda_contig(b);
  _ck_tensor_cuda_contig(g);
  _ck_tensor_cuda_contig(y);
  _ck(x.dtype() == torch::kFloat16, "x must be float16");
  _ck(g.dtype() == torch::kFloat16, "g must be float16");
  _ck(y.dtype() == torch::kFloat16, "y must be float16");
  _ck(w.dtype() == torch::kFloat32, "w must be float32");
  _ck(b.dtype() == torch::kFloat32, "b must be float32");
  _ck(x.dim() == 2, "x must be 2D");
  _ck(g.dim() == 2, "g must be 2D");
  _ck(y.dim() == 2, "y must be 2D");
  _ck(w.dim() == 1, "w must be 1D");
  _ck(b.dim() == 1, "b must be 1D");

  const int64_t h64 = x.size(0);
  const int64_t inner64 = x.size(1);
  _ck(h64 == 128, "hidden_dim must be 128");
  _ck(g.sizes() == x.sizes(), "g shape mismatch");
  _ck(w.numel() == h64 && b.numel() == h64, "w/b mismatch");
  _ck(inner64 > 0 && inner64 <= INT_MAX, "inner too large");
  _ck(y.size(0) == inner64 && y.size(1) == h64, "y shape mismatch");
  const int inner = (int)inner64;

  const dim3 block(32, 4, 1);
  const dim3 grid((inner + 31) / 32, 1, 1);
  _ln_gate_transpose_f16_new_kernel<<<grid, block>>>(
      (const __half*)x.data_ptr<at::Half>(),
      (const __half*)g.data_ptr<at::Half>(),
      w.data_ptr<float>(),
      b.data_ptr<float>(),
      (__half*)y.data_ptr<at::Half>(),
      inner);
  _ck_cuda_last("ln_gate_transpose_f16_out_new");
}

torch::Tensor trimul_fwd_f16(torch::Tensor x,
                             torch::Tensor mask,
                             torch::Tensor w_norm,
                             torch::Tensor b_norm,
                             torch::Tensor w_out_norm,
                             torch::Tensor b_out_norm,
                             torch::Tensor w0,
                             torch::Tensor w1,
                             torch::Tensor w2,
                             torch::Tensor w3,
                             torch::Tensor w4,
                             torch::Tensor w_to_out,
                             int64_t hidden_cfg) {
  _ck_tensor_cuda_contig(x);
  _ck_tensor_cuda_contig(mask);
  _ck_tensor_cuda_contig(w_norm);
  _ck_tensor_cuda_contig(b_norm);
  _ck_tensor_cuda_contig(w_out_norm);
  _ck_tensor_cuda_contig(b_out_norm);
  _ck_tensor_cuda_contig(w0);
  _ck_tensor_cuda_contig(w1);
  _ck_tensor_cuda_contig(w2);
  _ck_tensor_cuda_contig(w3);
  _ck_tensor_cuda_contig(w4);
  _ck_tensor_cuda_contig(w_to_out);

  _ck(x.dtype() == torch::kFloat32, "x must be float32");
  _ck(mask.dim() == 3, "mask must be 3D");

  _ck(w_norm.dtype() == torch::kFloat32 && b_norm.dtype() == torch::kFloat32, "norm must be f32");
  _ck(w_out_norm.dtype() == torch::kFloat32 && b_out_norm.dtype() == torch::kFloat32, "out norm must be f32");
  _ck(w0.dtype() == torch::kFloat32, "w0 must be float32");
  _ck(w1.dtype() == torch::kFloat32, "w1 must be float32");
  _ck(w2.dtype() == torch::kFloat32, "w2 must be float32");
  _ck(w3.dtype() == torch::kFloat32, "w3 must be float32");
  _ck(w4.dtype() == torch::kFloat32, "w4 must be float32");
  _ck(w_to_out.dtype() == torch::kFloat32, "w_to_out must be float32");

  _ck(x.dim() == 4, "x must be 4D");
  const int64_t bs = x.size(0);
  const int64_t n = x.size(1);
  _ck(x.size(2) == n, "x must be square");
  const int64_t dim = x.size(3);
  _ck(dim > 0 && dim <= INT_MAX, "bad dim");
  _ck(bs > 0 && bs <= INT_MAX, "bad bs");
  _ck(n > 0 && n <= INT_MAX, "bad n");

  _ck(mask.size(0) == bs && mask.size(1) == n && mask.size(2) == n, "mask shape mismatch");
  _ck(w_norm.numel() == dim && b_norm.numel() == dim, "norm param mismatch");

  _ck(hidden_cfg > 0 && hidden_cfg <= INT_MAX, "bad hidden_dim");
  const int64_t hidden = hidden_cfg;
  _ck(hidden == 128, "hidden_dim must be 128");
  _ck(w_out_norm.numel() == hidden && b_out_norm.numel() == hidden, "out norm param mismatch");

  _ck(w0.dim() == 2 && w0.size(0) == hidden && w0.size(1) == dim, "w0 shape mismatch");
  _ck(w1.dim() == 2 && w1.size(0) == hidden && w1.size(1) == dim, "w1 shape mismatch");
  _ck(w2.dim() == 2 && w2.size(0) == hidden && w2.size(1) == dim, "w2 shape mismatch");
  _ck(w3.dim() == 2 && w3.size(0) == hidden && w3.size(1) == dim, "w3 shape mismatch");
  _ck(w4.dim() == 2 && w4.size(0) == hidden && w4.size(1) == dim, "w4 shape mismatch");
  _ck(w_to_out.dim() == 2 && w_to_out.size(0) == dim && w_to_out.size(1) == hidden, "w_to_out shape mismatch");

#if TRIMUL_STAGE_TIMING
  _stage_timer tm;
#endif

#if TRIMUL_STAGE_TIMING
  tm.tic();
#endif
  auto x16 = ln_fwd_f16(x, w_norm, b_norm);
#if TRIMUL_STAGE_TIMING
  tm.toc("ln");
#endif
  const int64_t m = bs * n * n;
  auto x2 = x16.view({m, dim});

#if TRIMUL_STAGE_TIMING
  tm.tic();
#endif
  auto w_cat16 = torch::empty({5 * hidden, dim}, w0.options().dtype(torch::kFloat16));
  auto w_to_out16 = torch::empty({dim, hidden}, w0.options().dtype(torch::kFloat16));
  pack_w5_to_out_f16_out(w0, w1, w2, w3, w4, w_to_out, w_cat16, w_to_out16);
  auto proj_all = gemm_f16(w_cat16, x2);
  proj_all = proj_all.view({5, hidden, bs, n, n});
#if TRIMUL_STAGE_TIMING
  tm.toc("proj");
#endif

  auto left = proj_all.select(0, 0);
  auto right = proj_all.select(0, 1);
  auto left_gate = proj_all.select(0, 2);
  auto right_gate = proj_all.select(0, 3);
  auto out_gate = proj_all.select(0, 4);

#if TRIMUL_STAGE_TIMING
  tm.tic();
#endif
  apply_mask_gate_lr_f16(left, right, left_gate, right_gate, mask);
#if TRIMUL_STAGE_TIMING
  tm.toc("mask_gate");
#endif

  const int64_t batch = bs * hidden;
  auto a = left.reshape({batch, n, n});
  auto bb = right.reshape({batch, n, n});
  auto c_buf = left_gate.reshape({batch, n, n});
  _ck(a.is_contiguous() && bb.is_contiguous() && c_buf.is_contiguous(), "bmm buffers must be contiguous");
#if TRIMUL_STAGE_TIMING
  tm.tic();
#endif
  gemm_sb_f16_out(a, bb, c_buf);
#if TRIMUL_STAGE_TIMING
  tm.toc("bmm");
#endif

  auto out_flat = c_buf.view({hidden, m});
  auto gate_flat = out_gate.view({hidden, m});

  auto out2 = right_gate.view({m, hidden});
#if TRIMUL_STAGE_TIMING
  tm.tic();
#endif
  ln_gate_transpose_f16_out(out_flat, w_out_norm, b_out_norm, gate_flat, out2);
#if TRIMUL_STAGE_TIMING
  tm.toc("ln_gate");
#endif

#if TRIMUL_STAGE_TIMING
  tm.tic();
#endif
  auto y16 = gemm_f16(out2, w_to_out16);
#if TRIMUL_STAGE_TIMING
  tm.toc("out_gemm");
#endif
  return y16.view({bs, n, n, dim});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_f16", &gemm_f16, "矩阵乘（f16 输出）");
  m.def("gemm_sb_f16_out", &gemm_sb_f16_out, "批量矩阵乘（写入输出）");
  m.def("apply_mask_gate_lr_f16", &apply_mask_gate_lr_f16, "mask+gate 融合（不处理 out_gate）");
  m.def("ln_fwd_f16", &ln_fwd_f16, "LayerNorm 前向（f16 输出）");
  m.def("pack_w5_f16", &pack_w5_f16, "5 组权重打包与转换（f16）");
  m.def("cast_f32_to_f16", &cast_f32_to_f16, "f32->f16 转换");
  m.def("ln_gate_transpose_f16_out", &ln_gate_transpose_f16_out, "LN+gate+转置（写入输出）");
  m.def("trimul_fwd_f16", &trimul_fwd_f16, "TriMul Outgoing 前向（f16 输出）");
}
"""

    stage_timing_define = f"-DTRIMUL_STAGE_TIMING={1 if _ENABLE_STAGE_TIMING else 0}"
    _EXT = load_inline(
        name=f"trimul_ext_f16_v13_t{1 if _ENABLE_STAGE_TIMING else 0}",
        cpp_sources="",
        cuda_sources=cuda_src,
        functions=None,
        with_cuda=True,
        extra_cuda_cflags=["-O3", "--use_fast_math", stage_timing_define],
        extra_cflags=["-O3"],
        verbose=False,
    )
    return _EXT


def _t_contig_f32(t: torch.Tensor) -> torch.Tensor:
    if t.dtype != torch.float32:
        raise RuntimeError("weight must be float32")
    if not t.is_cuda:
        raise RuntimeError("weight must be CUDA")
    return t.contiguous() if not t.is_contiguous() else t


def _prepare_mask_f32(mask: torch.Tensor) -> torch.Tensor:
    if not mask.is_cuda:
        raise RuntimeError("mask must be CUDA")
    if mask.dtype != torch.float32:
        mask = mask.to(dtype=torch.float32)
    return mask.contiguous() if not mask.is_contiguous() else mask


def _shape_eq(t: torch.Tensor, shape: Tuple[int, ...]) -> bool:
    if t.dim() != len(shape):
        return False
    for idx, expected in enumerate(shape):
        if int(t.size(idx)) != int(expected):
            return False
    return True


def _static_input_gate(
    x: torch.Tensor,
    mask: torch.Tensor,
    weights: Dict[str, torch.Tensor],
    config: Dict[str, Any],
) -> Tuple[int, int, int, int]:
    if not isinstance(config, dict):
        raise RuntimeError("config must be dict")
    if "dim" not in config or "hidden_dim" not in config:
        raise RuntimeError("config must contain dim and hidden_dim")

    dim_cfg = int(config["dim"])
    hidden_cfg = int(config["hidden_dim"])

    if not x.is_cuda:
        raise RuntimeError("CUDA only")
    if x.dtype != torch.float32:
        raise RuntimeError("x must be float32")
    if x.dim() != 4:
        raise RuntimeError("x must be 4D")

    bs = int(x.size(0))
    n = int(x.size(1))
    if int(x.size(2)) != n:
        raise RuntimeError("x must be square")
    dim = int(x.size(3))
    if dim_cfg != dim:
        raise RuntimeError("config dim mismatch")
    if hidden_cfg != 128:
        raise RuntimeError("hidden_dim must be 128")

    if mask.dim() != 3:
        raise RuntimeError("mask must be 3D")
    if int(mask.size(0)) != bs or int(mask.size(1)) != n or int(mask.size(2)) != n:
        raise RuntimeError("mask shape mismatch")

    for key in _REQUIRED_WEIGHT_KEYS:
        if key not in weights:
            raise RuntimeError(f"missing weight: {key}")

    if not _shape_eq(weights["norm.weight"], (dim,)):
        raise RuntimeError("norm.weight shape mismatch")
    if not _shape_eq(weights["norm.bias"], (dim,)):
        raise RuntimeError("norm.bias shape mismatch")
    if not _shape_eq(weights["left_proj.weight"], (hidden_cfg, dim)):
        raise RuntimeError("left_proj.weight shape mismatch")
    if not _shape_eq(weights["right_proj.weight"], (hidden_cfg, dim)):
        raise RuntimeError("right_proj.weight shape mismatch")
    if not _shape_eq(weights["left_gate.weight"], (hidden_cfg, dim)):
        raise RuntimeError("left_gate.weight shape mismatch")
    if not _shape_eq(weights["right_gate.weight"], (hidden_cfg, dim)):
        raise RuntimeError("right_gate.weight shape mismatch")
    if not _shape_eq(weights["out_gate.weight"], (hidden_cfg, dim)):
        raise RuntimeError("out_gate.weight shape mismatch")
    if not _shape_eq(weights["to_out_norm.weight"], (hidden_cfg,)):
        raise RuntimeError("to_out_norm.weight shape mismatch")
    if not _shape_eq(weights["to_out_norm.bias"], (hidden_cfg,)):
        raise RuntimeError("to_out_norm.bias shape mismatch")
    if not _shape_eq(weights["to_out.weight"], (dim, hidden_cfg)):
        raise RuntimeError("to_out.weight shape mismatch")

    return bs, n, dim_cfg, hidden_cfg


@torch.inference_mode()
def custom_kernel(
    data: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]],
) -> torch.Tensor:
    x, mask, weights, config = data
    bs, n, dim_cfg, hidden_cfg = _static_input_gate(x, mask, weights, config)
    _ = bs
    _ = n
    _ = dim_cfg

    if not x.is_cuda:
        raise RuntimeError("CUDA only")
    if x.dtype != torch.float32:
        raise RuntimeError("x must be float32")
    if not x.is_contiguous():
        x = x.contiguous()

    mask = _prepare_mask_f32(mask)

    w_norm = _t_contig_f32(weights["norm.weight"])
    b_norm = _t_contig_f32(weights["norm.bias"])
    w_out_norm = _t_contig_f32(weights["to_out_norm.weight"])
    b_out_norm = _t_contig_f32(weights["to_out_norm.bias"])
    w0 = _t_contig_f32(weights["left_proj.weight"])
    w1 = _t_contig_f32(weights["right_proj.weight"])
    w2 = _t_contig_f32(weights["left_gate.weight"])
    w3 = _t_contig_f32(weights["right_gate.weight"])
    w4 = _t_contig_f32(weights["out_gate.weight"])
    w_to_out = _t_contig_f32(weights["to_out.weight"])

    ext = _get_ext()

    return ext.trimul_fwd_f16(
        x,
        mask,
        w_norm,
        b_norm,
        w_out_norm,
        b_out_norm,
        w0,
        w1,
        w2,
        w3,
        w4,
        w_to_out,
        hidden_cfg,
    )
