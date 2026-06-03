"""H100-oriented reproduction for GPUMode matmul_v2."""

import os

# The H100 winning idea leaves GEMM selection to PyTorch/cuBLAS, but pins the
# workspace config so the deterministic reference path can use cuBLAS safely.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

try:
    from task import input_t, output_t
except Exception:  # Allows local smoke tests outside the official harness.
    input_t = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    output_t = torch.Tensor


_FORCE_OUT_FALLBACK = os.environ.get("GPUMODE_MATMUL_V2_FORCE_OUT", "0") == "1"


def _mm_out_fallback(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    if (
        isinstance(c, torch.Tensor)
        and a.dim() == 2
        and b.dim() == 2
        and c.shape == (a.shape[0], b.shape[1])
        and c.device == a.device
        and c.dtype == torch.result_type(a, b)
    ):
        torch.mm(a, b, out=c)
        return c

    return a @ b


def custom_kernel(data: input_t) -> output_t:
    a, b, c = data

    if _FORCE_OUT_FALLBACK:
        return _mm_out_fallback(a, b, c)

    return a @ b
