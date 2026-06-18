"""
Functional TriMul forward: layer-normalize the pair representation, form left/right
projected gated factors, contract over the shared sequence axis, normalize the
hidden output, apply the output gate, and project back to the input channel size.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")

_proj_weight_cache = {}


@triton.jit
def _gate_pack_kernel(
    proj_ptr,
    mask_ptr,
    left_out_ptr,
    right_out_ptr,
    og_out_ptr,
    total: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offs < total
    row = offs // 128
    h = offs - row * 128
    base = row * 640 + h
    l = tl.load(proj_ptr + base, mask=valid, other=0.0).to(tl.float32)
    r = tl.load(proj_ptr + base + 128, mask=valid, other=0.0).to(tl.float32)
    lg = tl.load(proj_ptr + base + 256, mask=valid, other=0.0).to(tl.float32)
    rg = tl.load(proj_ptr + base + 384, mask=valid, other=0.0).to(tl.float32)
    og = tl.load(proj_ptr + base + 512, mask=valid, other=0.0).to(tl.float32)
    lsg = tl.sigmoid(lg)
    rsg = tl.sigmoid(rg)
    osg = tl.sigmoid(og)
    if HAS_MASK:
        m = tl.load(mask_ptr + row, mask=valid, other=0.0).to(tl.float32)
        l = l * m
        r = r * m
    tl.store(left_out_ptr + offs, l * lsg, mask=valid)
    tl.store(right_out_ptr + offs, r * rsg, mask=valid)
    tl.store(og_out_ptr + offs, osg, mask=valid)


@triton.jit
def _ln_gate_kernel(
    x_ptr,
    gate_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    rows,
    n: tl.constexpr,
    xs0: tl.constexpr,
    xs1: tl.constexpr,
    xs2: tl.constexpr,
    xs3: tl.constexpr,
    gs0: tl.constexpr,
    gs1: tl.constexpr,
    gs2: tl.constexpr,
    gs3: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = tl.arange(0, BLOCK_H)
    n2: tl.constexpr = n * n
    bidx = offs_m // n2
    rem = offs_m - bidx * n2
    iidx = rem // n
    jidx = rem - iidx * n
    mask = offs_m[:, None] < rows
    x_ptrs = bidx[:, None] * xs0 + iidx[:, None] * xs1 + jidx[:, None] * xs2 + offs_h[None, :] * xs3
    g_ptrs = bidx[:, None] * gs0 + iidx[:, None] * gs1 + jidx[:, None] * gs2 + offs_h[None, :] * gs3
    y_ptrs = offs_m[:, None] * BLOCK_H + offs_h[None, :]
    x = tl.load(x_ptr + x_ptrs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / BLOCK_H
    mean2 = tl.sum(x * x, axis=1) / BLOCK_H
    xc = x - mean[:, None]
    var = tl.maximum(mean2 - mean * mean, 0.0)
    rstd = tl.rsqrt(var + 1.0e-5)
    w = tl.load(w_ptr + offs_h).to(tl.float32)
    b = tl.load(b_ptr + offs_h).to(tl.float32)
    g = tl.load(gate_ptr + g_ptrs, mask=mask, other=0.0).to(tl.float32)
    y = (xc * rstd[:, None] * w[None, :] + b[None, :]) * g
    tl.store(y_ptr + y_ptrs, y, mask=mask)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    x0 = input_tensor.to(torch.float32)
    batch, seq_len = input_tensor.shape[0], input_tensor.shape[1]
    x = F.layer_norm(
        x0,
        (dim,),
        weights["norm.weight"].to(torch.float32),
        weights["norm.bias"].to(torch.float32),
        1.0e-5,
    )

    proj_weights = (
        weights["left_proj.weight"],
        weights["right_proj.weight"],
        weights["left_gate.weight"],
        weights["right_gate.weight"],
        weights["out_gate.weight"],
    )
    cache_key = tuple((id(w), w.data_ptr(), tuple(w.shape), w.dtype) for w in proj_weights)
    cached = _proj_weight_cache.get(cache_key)
    if cached is None:
        proj_w = torch.cat(tuple(w.to(torch.float32) for w in proj_weights), dim=0)
        _proj_weight_cache[cache_key] = (proj_w, proj_weights)
    else:
        proj_w = cached[0]
    proj = F.linear(x, proj_w)

    shape_h = (input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2], hidden_dim)
    left_b = torch.empty(shape_h, device=input_tensor.device, dtype=torch.bfloat16)
    right_b = torch.empty_like(left_b)
    out_gate = torch.empty(shape_h, device=input_tensor.device, dtype=torch.float32)
    total = left_b.numel()
    has_mask = mask is not None and not bool(config.get("nomask", False))
    _gate_pack_kernel[(triton.cdiv(total, 512),)](
        proj,
        mask if has_mask else proj,
        left_b,
        right_b,
        out_gate,
        total,
        HAS_MASK=has_mask,
        BLOCK=512,
        num_warps=4,
    )

    out = torch.einsum("bikd,bjkd->bijd", left_b, right_b)
    if hidden_dim == 128:
        y = torch.empty((batch, seq_len, seq_len, hidden_dim), device=out.device, dtype=torch.float32)
        rows = out.numel() // hidden_dim
        _ln_gate_kernel[(triton.cdiv(rows, 8),)](
            out,
            out_gate,
            weights["to_out_norm.weight"].to(torch.float32),
            weights["to_out_norm.bias"].to(torch.float32),
            y,
            rows,
            input_tensor.shape[1],
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            out_gate.stride(0),
            out_gate.stride(1),
            out_gate.stride(2),
            out_gate.stride(3),
            BLOCK_M=8,
            BLOCK_H=128,
            num_warps=4,
        )
        out = y
    else:
        out = F.layer_norm(
            out,
            (hidden_dim,),
            weights["to_out_norm.weight"].to(torch.float32),
            weights["to_out_norm.bias"].to(torch.float32),
            1.0e-5,
        )
        out = out * out_gate
    out = F.linear(out, weights["to_out.weight"].to(torch.float32))
    return out.to(torch.float32)
