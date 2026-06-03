# kernel.py
# Fused pipeline: (1) load A + load B -> (2) elementwise add -> (3) store C
# Everything is done in a single Triton kernel (no unfused stages needed).
#
# Fix for timeout: remove @triton.autotune keyed on n_elements, which can trigger
# multiple compilations/benchmarks for each different input size and exceed the
# test timeout. Use a single compiled configuration instead.

import torch
import triton
import triton.language as tl


@triton.jit
def _add_f16_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Contiguous 1D flatten => naturally coalesced.
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(c_ptr + offs, a + b, mask=mask)


def kernel_function(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
    """
    Elementwise float16 add for two (N, N) CUDA tensors: C = A + B.

    Wrapper responsibilities only: validate/allocate/launch. No PyTorch math ops.
    """
    assert isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)
    assert A.is_cuda and B.is_cuda and C.is_cuda
    assert A.dtype == torch.float16 and B.dtype == torch.float16 and C.dtype == torch.float16
    assert A.shape == B.shape == C.shape
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous()

    n_elements = A.numel()

    # Single stable configuration (fast compile, avoids autotune timeouts).
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _add_f16_kernel[grid](
        A, B, C,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    return C

import inspect
def custom_kernel(input):
    sig = inspect.signature(kernel_function)
    num_params = len(sig.parameters)
    if len(input) == num_params:
        return kernel_function(*input)
    return kernel_function(input)

import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG", "") not in (":4096:8", ":16:8"):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
