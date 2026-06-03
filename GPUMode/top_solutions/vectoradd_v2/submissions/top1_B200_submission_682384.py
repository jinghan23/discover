#!POPCORN leaderboard vectoradd_v2
#!POPCORN gpu B200

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ __launch_bounds__(512, 4)
void vectoradd_kernel(const float4* __restrict__ A, const float4* __restrict__ B,
                      float4* __restrict__ C, const int n4) {
    const int idx = blockIdx.x * 512 + threadIdx.x;
    if (idx >= n4) return;

    if (idx + 512 < n4) {
        const void* prefetch_a = reinterpret_cast<const void*>(&A[idx + 512]);
        const void* prefetch_b = reinterpret_cast<const void*>(&B[idx + 512]);
        asm volatile("prefetch.global.L2 [%0];" :: "l"(prefetch_a));
        asm volatile("prefetch.global.L2 [%0];" :: "l"(prefetch_b));
    }

    float4 a, b;
    const float4* addr_a = &A[idx];
    const float4* addr_b = &B[idx];

    asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(reinterpret_cast<unsigned int*>(&a)[0]),
                   "=r"(reinterpret_cast<unsigned int*>(&a)[1]),
                   "=r"(reinterpret_cast<unsigned int*>(&a)[2]),
                   "=r"(reinterpret_cast<unsigned int*>(&a)[3])
                 : "l"(addr_a));

    asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(reinterpret_cast<unsigned int*>(&b)[0]),
                   "=r"(reinterpret_cast<unsigned int*>(&b)[1]),
                   "=r"(reinterpret_cast<unsigned int*>(&b)[2]),
                   "=r"(reinterpret_cast<unsigned int*>(&b)[3])
                 : "l"(addr_b));

    half2* ah = reinterpret_cast<half2*>(&a);
    half2* bh = reinterpret_cast<half2*>(&b);
    float4 c;
    half2* ch = reinterpret_cast<half2*>(&c);
    ch[0] = __hadd2(ah[0], bh[0]);
    ch[1] = __hadd2(ah[1], bh[1]);
    ch[2] = __hadd2(ah[2], bh[2]);
    ch[3] = __hadd2(ah[3], bh[3]);

    float4* addr_c = &C[idx];
    asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "l"(addr_c),
                    "r"(reinterpret_cast<unsigned int*>(&c)[0]),
                    "r"(reinterpret_cast<unsigned int*>(&c)[1]),
                    "r"(reinterpret_cast<unsigned int*>(&c)[2]),
                    "r"(reinterpret_cast<unsigned int*>(&c)[3]));
}

__global__ void vectoradd_tail(const __half* __restrict__ A, const __half* __restrict__ B,
                               __half* __restrict__ C, const int start, const int n) {
    const int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = __hadd(A[idx], B[idx]);
    }
}

void vectoradd_raw(int64_t a_ptr, int64_t b_ptr, int64_t c_ptr, int N) {
    const int n8 = N / 8;
    const int remainder = N - n8 * 8;

    if (n8 > 0) {
        const int blocks = (n8 + 511) / 512;
        vectoradd_kernel<<<blocks, 512>>>(
            reinterpret_cast<const float4*>(a_ptr),
            reinterpret_cast<const float4*>(b_ptr),
            reinterpret_cast<float4*>(c_ptr),
            n8);
    }

    if (remainder > 0) {
        const int start = n8 * 8;
        const int blocks = (remainder + 255) / 256;
        vectoradd_tail<<<blocks, 256>>>(
            reinterpret_cast<const __half*>(a_ptr),
            reinterpret_cast<const __half*>(b_ptr),
            reinterpret_cast<__half*>(c_ptr),
            start, N);
    }
}
"""

cpp_src = r"""
void vectoradd_raw(int64_t a_ptr, int64_t b_ptr, int64_t c_ptr, int N);
"""

_ext = load_inline(
    name="vectoradd_ptx512f",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["vectoradd_raw"],
    with_cuda=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_100a", "-maxrregcount=32"],
    verbose=False,
)

# 10 full-size warmup iterations to fully prime TLB, icache, memory controllers
_wa = torch.randn(16384, 16384, device="cuda", dtype=torch.float16)
_wb = torch.randn(16384, 16384, device="cuda", dtype=torch.float16)
_wc = torch.empty(16384, 16384, device="cuda", dtype=torch.float16)
for _ in range(10):
    _ext.vectoradd_raw(_wa.data_ptr(), _wb.data_ptr(), _wc.data_ptr(), _wa.numel())
torch.cuda.synchronize()
del _wa, _wb, _wc
torch.cuda.empty_cache()


def custom_kernel(data: input_t) -> output_t:
    A, B, output = data
    _ext.vectoradd_raw(A.data_ptr(), B.data_ptr(), output.data_ptr(), A.numel())
    return output
