"""A100-oriented algorithmic reproduction for GPUMode TriMul."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _work_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.float16 if x.is_cuda else torch.float32


@torch.no_grad()
def custom_kernel(data: tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]) -> torch.Tensor:
    x, mask, weights, config = data
    batch, seqlen, seqlen_2, dim = x.shape
    if seqlen != seqlen_2:
        raise ValueError("TriMul expects square pair dimensions")

    hidden = int(config["hidden_dim"])
    total = batch * seqlen * seqlen
    dtype = _work_dtype(x)
    device = x.device

    x_norm = F.layer_norm(
        x,
        (dim,),
        weights["norm.weight"].to(device=device, dtype=torch.float32),
        weights["norm.bias"].to(device=device, dtype=torch.float32),
        eps=1e-5,
    )
    flat = x_norm.reshape(total, dim).to(dtype=dtype)
    packed_w = torch.cat(
        [
            weights["left_proj.weight"],
            weights["right_proj.weight"],
            weights["left_gate.weight"],
            weights["right_gate.weight"],
            weights["out_gate.weight"],
        ],
        dim=0,
    ).to(device=device, dtype=dtype)

    # A100 top1 idea: produce hidden-major projection buffers and use bmm for contraction.
    proj_t = packed_w @ flat.t()
    left = proj_t[0:hidden]
    right = proj_t[hidden : 2 * hidden]
    left_gate = proj_t[2 * hidden : 3 * hidden]
    right_gate = proj_t[3 * hidden : 4 * hidden]
    out_gate = proj_t[4 * hidden : 5 * hidden]

    mask_flat = mask.reshape(1, total).to(device=device, dtype=dtype)
    left = left * torch.sigmoid(left_gate) * mask_flat
    right = right * torch.sigmoid(right_gate) * mask_flat

    left_bh = left.view(hidden, batch, seqlen, seqlen).permute(1, 0, 2, 3).reshape(batch * hidden, seqlen, seqlen)
    right_bh = right.view(hidden, batch, seqlen, seqlen).permute(1, 0, 2, 3).reshape(batch * hidden, seqlen, seqlen)
    contracted = torch.bmm(left_bh, right_bh.transpose(1, 2))

    hidden_major = contracted.view(batch, hidden, seqlen, seqlen).permute(1, 0, 2, 3).reshape(hidden, total).float()
    mean = hidden_major.mean(dim=0, keepdim=True)
    centered = hidden_major - mean
    var = centered.square().mean(dim=0, keepdim=True)
    normed = centered * torch.rsqrt(var + 1e-5)
    normed = normed * weights["to_out_norm.weight"].to(device=device, dtype=torch.float32).view(hidden, 1)
    normed = normed + weights["to_out_norm.bias"].to(device=device, dtype=torch.float32).view(hidden, 1)
    normed = normed * torch.sigmoid(out_gate.float())

    final = weights["to_out.weight"].to(device=device, dtype=dtype) @ normed.to(dtype=dtype)
    return final.view(dim, batch, seqlen, seqlen).permute(1, 2, 3, 0)
