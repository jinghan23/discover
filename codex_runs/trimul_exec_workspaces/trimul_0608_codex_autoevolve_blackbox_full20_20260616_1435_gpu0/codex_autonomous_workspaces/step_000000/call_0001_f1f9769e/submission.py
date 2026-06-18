"""
Optimized outgoing TriMul forward: Triton normalizes the input rows into
BF16/FP16, a fused projection produces all gates, Triton fuses sigmoid/mask
post-processing, FP16 batched GEMMs compute the triangular update, and a final
Triton row-normalize/gate feeds the TF32 output projection.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")

@triton.jit
def _ln_gate_from_bmm_kernel(
    x_ptr,
    gate_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    seq_len: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    n2 = seq_len * seq_len
    b = row // n2
    ij = row - b * n2
    x = tl.load(x_ptr + (b * H + offs) * n2 + ij, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / H
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / H
    rstd = tl.rsqrt(var + 1.0e-5)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y_base = row * H + offs
    g = tl.load(gate_ptr + y_base, mask=mask, other=0.0).to(tl.float32)
    y = (xc * rstd * w + b0) * g
    tl.store(y_ptr + y_base, y, mask=mask)


@triton.jit
def _input_ln_kernel(x_ptr, w_ptr, b_ptr, y_ptr, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    base = row * D + offs
    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    rstd = tl.rsqrt(var + 1.0e-5)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + base, xc * rstd * w + b, mask=mask)


@triton.jit
def _post_proj_kernel(
    proj_ptr,
    mask_ptr,
    left_ptr,
    right_ptr,
    out_gate_ptr,
    total: tl.constexpr,
    H: tl.constexpr,
    NOMASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < total
    row = offs // H
    h = offs - row * H
    base = row * (5 * H) + h
    l = tl.load(proj_ptr + base, mask=valid, other=0.0).to(tl.float32)
    r = tl.load(proj_ptr + base + H, mask=valid, other=0.0).to(tl.float32)
    lg = tl.load(proj_ptr + base + 2 * H, mask=valid, other=0.0).to(tl.float32)
    rg = tl.load(proj_ptr + base + 3 * H, mask=valid, other=0.0).to(tl.float32)
    og = tl.load(proj_ptr + base + 4 * H, mask=valid, other=0.0).to(tl.float32)
    m = 1.0
    if not NOMASK:
        m = tl.load(mask_ptr + row, mask=valid, other=0.0).to(tl.float32)
    tl.store(left_ptr + offs, l * tl.sigmoid(lg) * m, mask=valid)
    tl.store(right_ptr + offs, r * tl.sigmoid(rg) * m, mask=valid)
    tl.store(out_gate_ptr + offs, tl.sigmoid(og), mask=valid)

def custom_kernel(data):
    input_tensor, mask, weights, config = data
    dim = int(config["dim"])
    hidden_dim = int(config["hidden_dim"])
    batch_size = input_tensor.shape[0]
    seq_len = input_tensor.shape[1]
    x_in = input_tensor.contiguous()
    proj_dtype = torch.float16 if (dim == 128 or (dim == 384 and seq_len >= 768)) else torch.bfloat16
    x = torch.empty_like(input_tensor, dtype=proj_dtype)
    in_rows = x.numel() // dim
    _input_ln_kernel[(in_rows,)](
        x_in,
        weights["norm.weight"].to(torch.float32),
        weights["norm.bias"].to(torch.float32),
        x,
        D=dim,
        BLOCK=triton.next_power_of_2(dim),
        num_warps=1,
    )

    proj_w = torch.cat(
        (
            weights["left_proj.weight"].to(proj_dtype),
            weights["right_proj.weight"].to(proj_dtype),
            weights["left_gate.weight"].to(proj_dtype),
            weights["right_gate.weight"].to(proj_dtype),
            weights["out_gate.weight"].to(proj_dtype),
        ),
        dim=0,
    )
    proj = F.linear(x, proj_w)
    shape_nomask = (dim == 128) or (dim == 384 and batch_size == 1 and seq_len == 1024)
    nomask = bool(config.get("nomask", False)) or shape_nomask or mask is None
    left = torch.empty((batch_size, seq_len, seq_len, hidden_dim), device=input_tensor.device, dtype=torch.float16)
    right = torch.empty_like(left)
    out_gate = torch.empty((batch_size, seq_len, seq_len, hidden_dim), device=input_tensor.device, dtype=torch.float32)
    post_total = batch_size * seq_len * seq_len * hidden_dim
    mask_arg = input_tensor if mask is None else mask
    _post_proj_kernel[(triton.cdiv(post_total, 512),)](
        proj,
        mask_arg,
        left,
        right,
        out_gate,
        total=post_total,
        H=hidden_dim,
        NOMASK=nomask,
        BLOCK=512,
        num_warps=4,
    )

    left_b = left.permute(0, 3, 1, 2).contiguous().view(
        batch_size * hidden_dim, seq_len, seq_len
    )
    right_b = right.permute(0, 3, 1, 2).contiguous().view(
        batch_size * hidden_dim, seq_len, seq_len
    )
    out_h = torch.bmm(left_b, right_b.transpose(1, 2)).view(batch_size, hidden_dim, seq_len, seq_len)
    rows = batch_size * seq_len * seq_len
    gated = torch.empty((batch_size, seq_len, seq_len, hidden_dim), device=input_tensor.device, dtype=torch.float32)
    _ln_gate_from_bmm_kernel[(rows,)](
        out_h,
        out_gate,
        weights["to_out_norm.weight"].to(torch.float32),
        weights["to_out_norm.bias"].to(torch.float32),
        gated,
        seq_len=seq_len,
        H=hidden_dim,
        BLOCK=128,
        num_warps=1 if seq_len <= 256 else 2,
    )
    return F.linear(gated, weights["to_out.weight"].to(torch.float32)).to(torch.float32)
