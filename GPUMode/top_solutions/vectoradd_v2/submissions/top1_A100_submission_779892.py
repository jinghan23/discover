from utils import make_match_reference, DeterministicContext
import torch
from task import input_t, output_t
import sys

from torch.utils.cpp_extension import load_inline

N_ELEMENTS = 16384

_CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor cuda_add(std::vector<torch::Tensor> data);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_add", &cuda_add, "Sum reduction with custom CUDA kernel");
}
"""


_CUDA_SOURCE = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// 常量硬编码：总元素数 16384*16384 = 2^28
constexpr size_t N_SIZE = 16384;
constexpr size_t TOTAL_ELEMENTS = N_SIZE * N_SIZE;

// 针对 N=16384 硬编码，消除动态计算与分支开销
__global__ void __launch_bounds__(256)
add_fp16_n16384_kernel(const half* __restrict__ A,
                       const half* __restrict__ B,
                       half* __restrict__ C) {
    
    constexpr size_t VEC_SIZE       = TOTAL_ELEMENTS / 8; // uint4 一次处理 8 个 half
    constexpr size_t BLOCKS         = 16384;
    constexpr size_t THREADS        = 256;
    constexpr size_t STRIDE         = BLOCKS * THREADS;   // 4,194,304
    constexpr size_t ITERS          = VEC_SIZE / STRIDE;  // 8

    size_t idx = blockIdx.x * THREADS + threadIdx.x;

    // 128-bit 向量化指针 (要求 16B 对齐，PyTorch 默认满足)
    const uint4* __restrict__ A_vec = reinterpret_cast<const uint4*>(A);
    const uint4* __restrict__ B_vec = reinterpret_cast<const uint4*>(B);
    uint4* __restrict__ C_vec       = reinterpret_cast<uint4*>(C);

    // 固定次数循环，编译器将完全展开，消除分支与循环计数器开销
    #pragma unroll
    for (int i = 0; i < ITERS; ++i) {
        size_t offset = idx + i * STRIDE;
        
        // 向量化加载 (16B/thread)
        uint4 a = A_vec[offset];
        uint4 b = B_vec[offset];
        uint4 c;

        // 重解释为 half2[4] 利用 A100 的 SIMD FP16 单元
        half2* a2 = reinterpret_cast<half2*>(&a);
        half2* b2 = reinterpret_cast<half2*>(&b);
        half2* c2 = reinterpret_cast<half2*>(&c);

        // 4路 half2 并行加法 (每线程每次迭代处理 8 个 FP16)
        c2[0] = __hadd2(a2[0], b2[0]);
        c2[1] = __hadd2(a2[1], b2[1]);
        c2[2] = __hadd2(a2[2], b2[2]);
        c2[3] = __hadd2(a2[3], b2[3]);

        // 向量化写回
        C_vec[offset] = c;
    }
}


torch::Tensor cuda_add(std::vector<torch::Tensor> data) {

    if (data[0].size(0) == N_SIZE) {
        // 启动针对 N=16384 的专用内核，消除动态计算与分支开销
        add_fp16_n16384_kernel<<<N_SIZE, 256>>>(reinterpret_cast<const half*>(data[0].data_ptr<at::Half>()),
                                                     reinterpret_cast<const half*>(data[1].data_ptr<at::Half>()),
                                                     reinterpret_cast<half*>(data[2].data_ptr<at::Half>()));
        return data[2];
        
    } else { 
        // 退化到 PyTorch 内置实现，保证正确性
        data[2] = data[0] + data[1];
        return data[2];
    }
}
"""

_EXT = load_inline(
        name="cuda_add_extension_001",
        cpp_sources=[_CPP_SOURCE],
        cuda_sources=[_CUDA_SOURCE],
        functions=None,
        extra_cflags=["-O3 -use_fast_math"],
        extra_cuda_cflags=["-O3 -use_fast_math -Xptxas=-v -maxrregcount=32"],
        with_cuda=True,
        verbose=False,
    )

custom_kernel = _EXT.cuda_add

def ref_kernel(data: input_t) -> output_t:
    """
    Reference implementation of vector addition using PyTorch.
    Args:
        data: Tuple of tensors [A, B] to be added.
    Returns:
        Tensor containing element-wise sums.
    """
    with DeterministicContext():
        A, B, output = data
        output[...] = A + B
        return output


def generate_input(size: int, seed: int) -> input_t:
    """
    Generates random input tensors of specified shapes.
    Returns:
        Tuple of tensors [A, B] to be added.
    """
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    A = torch.randn(
        size, size, device="cuda", dtype=torch.float16, generator=gen
    ).contiguous()
    B = torch.randn(
        size, size, device="cuda", dtype=torch.float16, generator=gen
    ).contiguous()
    C = torch.empty(size, size, device="cuda", dtype=torch.float16).contiguous()
    return A, B, C


check_implementation = make_match_reference(ref_kernel)


def warmup(fn, args, n_warmup=5):
    for _ in range(n_warmup):
        _ = fn(args)
        torch.cuda.synchronize()


# warmup(custom_kernel, generate_input(N_ELEMENTS, 42))
