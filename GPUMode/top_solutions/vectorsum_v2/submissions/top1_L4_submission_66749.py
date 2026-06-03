#!POPCORN leaderboard vectorsum_v2

import torch
import triton
import triton.language as tl
from task import input_t, output_t


@triton.jit
def sum_kernel_optimized(
    x_ptr,
    partial_sums_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first")
    block_sum = tl.sum(x, axis=0)
    tl.store(partial_sums_ptr + pid, block_sum)


@triton.jit
def sum_kernel_atomic(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first")
    block_sum = tl.sum(x, axis=0)

    tl.atomic_add(output_ptr, block_sum)


def custom_kernel(data: input_t) -> output_t:
    input, output = data
    n_elements = input.numel()

    BLOCK_SIZE = 2**10

    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    partial_sums = torch.empty(n_blocks, device=input.device, dtype=input.dtype)

    grid = (n_blocks,)
    sum_kernel_optimized[grid](input, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    while partial_sums.numel() > BLOCK_SIZE:
        n_elements = partial_sums.numel()
        n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
        next_level = torch.empty(n_blocks, device=input.device, dtype=input.dtype)

        grid = (n_blocks,)
        sum_kernel_optimized[grid](partial_sums, next_level, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        partial_sums = next_level

    if partial_sums.numel() <= 128:
        final_output = torch.zeros(1, device=input.device, dtype=input.dtype)
        n_elements = partial_sums.numel()
        sum_kernel_atomic[(1,)](partial_sums, final_output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return final_output[0]
    else:
        n_elements = partial_sums.numel()
        n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
        next_level = torch.empty(n_blocks, device=input.device, dtype=input.dtype)

        grid = (n_blocks,)
        sum_kernel_optimized[grid](partial_sums, next_level, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return next_level.sum()
