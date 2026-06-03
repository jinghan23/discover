#!POPCORN leaderboard vectorsum_v2

import torch
import triton
import triton.language as tl
from task import input_t, output_t


@triton.jit
def sum_kernel_single(
	x_ptr,
	partial_ptr,
	counter_ptr,
	output_ptr,
	n_elements,
	n_blocks,
	BLOCK_SIZE: tl.constexpr,
	FINAL_BLOCK: tl.constexpr,
):
	pid = tl.program_id(0)
	block_start = pid * BLOCK_SIZE
	offsets = block_start + tl.arange(0, BLOCK_SIZE)
	mask = offsets < n_elements
	x = tl.load(x_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first")
	block_sum = tl.sum(x, axis=0)
	tl.store(partial_ptr + pid, block_sum)
	tl.debug_barrier()
	old = tl.atomic_add(counter_ptr, 1, sem="relaxed")
	is_last = old == (n_blocks - 1)
	if is_last:
		offsets2 = tl.arange(0, FINAL_BLOCK)
		mask2 = offsets2 < n_blocks
		vals = tl.load(partial_ptr + offsets2, mask=mask2, other=0.0)
		total = tl.sum(vals, axis=0)
		tl.store(output_ptr, total)
		tl.store(counter_ptr, 0)


_COUNTER = torch.zeros(1, device="cuda", dtype=torch.int32)
_PARTS = torch.empty(12800, device="cuda", dtype=torch.float32)

_TABLE = {}
_RAW = {
	1638400: (16384, 16),
	3276800: (8192, 8),
	6553600: (8192, 4),
	13107200: (8192, 4),
	26214400: (8192, 8),
	52428800: (8192, 8),
}

for _n, (_bs, _nw) in _RAW.items():
	_nb = (_n + _bs - 1) // _bs
	_b2 = max(32, 1 << (_nb - 1).bit_length())
	_TABLE[_n] = (_bs, _nb, _b2, _nw)


def _get_config(n):
	if n in _TABLE:
		return _TABLE[n]
	bs = 8192
	nw = 8
	nb = (n + bs - 1) // bs
	b2 = max(32, 1 << (nb - 1).bit_length())
	return (bs, nb, b2, nw)


def custom_kernel(data: input_t) -> output_t:
	data, output = data
	n = data.numel()
	bs, nb, b2, nw = _get_config(n)
	sum_kernel_single[(nb,)](
		data, _PARTS, _COUNTER, output, n, nb,
		BLOCK_SIZE=bs, FINAL_BLOCK=b2, num_warps=nw,
	)
	return output[0]
