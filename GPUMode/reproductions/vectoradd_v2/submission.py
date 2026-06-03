import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


_BLOCK_SIZE = 1024


if _HAS_TRITON:

    @triton.jit
    def _vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        tl.store(c_ptr + offsets, a + b, mask=mask)


def _unpack_data(data):
    if len(data) == 3:
        return data
    if len(data) == 2:
        a, b = data
        return a, b, torch.empty_like(a)
    raise ValueError("vectoradd_v2 expects (A, B, C) or (A, B)")


def _can_use_triton(a, b, out):
    return (
        _HAS_TRITON
        and a.is_cuda
        and b.is_cuda
        and out.is_cuda
        and a.is_contiguous()
        and b.is_contiguous()
        and out.is_contiguous()
        and a.dtype == torch.float16
        and b.dtype == torch.float16
        and out.dtype == torch.float16
        and a.numel() == b.numel() == out.numel()
    )


def custom_kernel(data):
    a, b, out = _unpack_data(data)
    n_elements = out.numel()

    if n_elements == 0:
        return out

    if _can_use_triton(a, b, out):
        grid = (triton.cdiv(n_elements, _BLOCK_SIZE),)
        _vector_add_kernel[grid](
            a,
            b,
            out,
            n_elements,
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=8,
        )
        return out

    torch.add(a, b, out=out)
    return out
