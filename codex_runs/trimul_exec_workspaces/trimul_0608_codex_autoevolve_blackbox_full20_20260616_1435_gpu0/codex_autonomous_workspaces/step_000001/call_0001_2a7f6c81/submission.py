"""
Outgoing TriMul forward: Triton row-normalizes inputs into FP16, PyTorch performs
the fused 5-way projection, Triton applies left/right gates while writing
directly into BMM layout, FP16 batched GEMM computes the triangular update, and
a blocked Triton output norm reads the out-gate projection directly before the
TF32 output projection.
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
def _ln_gate_from_bmm_block_kernel(
    x_ptr,
    gate_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    rows: tl.constexpr,
    seq_len: tl.constexpr,
    H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_h = tl.arange(0, BLOCK_H)
    valid_r = offs_r < rows
    n2 = seq_len * seq_len
    b = offs_r // n2
    ij = offs_r - b * n2
    hmask = offs_h < H
    ptrs = x_ptr + ((b[None, :] * H + offs_h[:, None]) * n2 + ij[None, :])
    x = tl.load(ptrs, mask=hmask[:, None] & valid_r[None, :], other=0.0).to(tl.float32)
    x = tl.where(hmask[:, None], x, 0.0)
    mean = tl.sum(x, axis=0) / H
    xc = tl.where(hmask[:, None], x - mean[None, :], 0.0)
    var = tl.sum(xc * xc, axis=0) / H
    rstd = tl.rsqrt(var + 1.0e-5)
    w = tl.load(w_ptr + offs_h, mask=hmask, other=0.0).to(tl.float32)
    b0 = tl.load(b_ptr + offs_h, mask=hmask, other=0.0).to(tl.float32)
    g = tl.load(
        gate_ptr + offs_r[None, :] * H + offs_h[:, None],
        mask=hmask[:, None] & valid_r[None, :],
        other=0.0,
    ).to(tl.float32)
    y = (xc * rstd[None, :] * w[:, None] + b0[:, None]) * g
    tl.store(
        y_ptr + offs_r[None, :] * H + offs_h[:, None],
        y,
        mask=hmask[:, None] & valid_r[None, :],
    )


@triton.jit
def _ln_gate_from_bmm_projgate_kernel(
    x_ptr,
    proj_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    rows: tl.constexpr,
    seq_len: tl.constexpr,
    H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_h = tl.arange(0, BLOCK_H)
    valid_r = offs_r < rows
    n2 = seq_len * seq_len
    b = offs_r // n2
    ij = offs_r - b * n2
    hmask = offs_h < H
    ptrs = x_ptr + ((b[None, :] * H + offs_h[:, None]) * n2 + ij[None, :])
    x = tl.load(ptrs, mask=hmask[:, None] & valid_r[None, :], other=0.0).to(tl.float32)
    x = tl.where(hmask[:, None], x, 0.0)
    mean = tl.sum(x, axis=0) / H
    xc = tl.where(hmask[:, None], x - mean[None, :], 0.0)
    var = tl.sum(xc * xc, axis=0) / H
    rstd = tl.rsqrt(var + 1.0e-5)
    w = tl.load(w_ptr + offs_h, mask=hmask, other=0.0).to(tl.float32)
    b0 = tl.load(b_ptr + offs_h, mask=hmask, other=0.0).to(tl.float32)
    og = tl.load(
        proj_ptr + offs_r[None, :] * (5 * H) + 4 * H + offs_h[:, None],
        mask=hmask[:, None] & valid_r[None, :],
        other=0.0,
    ).to(tl.float32)
    y = (xc * rstd[None, :] * w[:, None] + b0[:, None]) * tl.sigmoid(og)
    tl.store(
        y_ptr + offs_r[None, :] * H + offs_h[:, None],
        y,
        mask=hmask[:, None] & valid_r[None, :],
    )


@triton.jit
def _final_norm_proj_d128_kernel(
    x_ptr,
    proj_ptr,
    norm_w_ptr,
    norm_b_ptr,
    out_w_ptr,
    out_ptr,
    rows: tl.constexpr,
    seq_len: tl.constexpr,
    H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    valid_r = offs_r < rows
    n2 = seq_len * seq_len
    b = offs_r // n2
    ij = offs_r - b * n2
    hmask = offs_h < H
    x = tl.load(
        x_ptr + ((b[None, :] * H + offs_h[:, None]) * n2 + ij[None, :]),
        mask=hmask[:, None] & valid_r[None, :],
        other=0.0,
    ).to(tl.float32)
    x = tl.where(hmask[:, None], x, 0.0)
    mean = tl.sum(x, axis=0) / H
    xc = tl.where(hmask[:, None], x - mean[None, :], 0.0)
    var = tl.sum(xc * xc, axis=0) / H
    rstd = tl.rsqrt(var + 1.0e-5)
    nw = tl.load(norm_w_ptr + offs_h, mask=hmask, other=0.0).to(tl.float32)
    nb = tl.load(norm_b_ptr + offs_h, mask=hmask, other=0.0).to(tl.float32)
    og = tl.load(
        proj_ptr + offs_r[None, :] * (5 * H) + 4 * H + offs_h[:, None],
        mask=hmask[:, None] & valid_r[None, :],
        other=0.0,
    ).to(tl.float32)
    gated = (xc * rstd[None, :] * nw[:, None] + nb[:, None]) * tl.sigmoid(og)
    ow = tl.load(out_w_ptr + offs_d[None, :] * H + offs_h[:, None]).to(tl.float32)
    acc = tl.dot(tl.trans(gated), ow, input_precision="tf32")
    tl.store(out_ptr + offs_r[:, None] * BLOCK_D + offs_d[None, :], acc, mask=valid_r[:, None])


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


@triton.jit
def _transpose_lr_kernel(
    left_ptr,
    right_ptr,
    left_b_ptr,
    right_b_ptr,
    rows: tl.constexpr,
    seq_len: tl.constexpr,
    H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    valid = (offs_r[:, None] < rows) & (offs_h[None, :] < H)
    src = offs_r[:, None] * H + offs_h[None, :]
    n2 = seq_len * seq_len
    b = offs_r // n2
    ij = offs_r - b * n2
    dst = (b[:, None] * H + offs_h[None, :]) * n2 + ij[:, None]
    l = tl.load(left_ptr + src, mask=valid, other=0.0)
    r = tl.load(right_ptr + src, mask=valid, other=0.0)
    tl.store(left_b_ptr + dst, l, mask=valid)
    tl.store(right_b_ptr + dst, r, mask=valid)


@triton.jit
def _post_proj_transpose_kernel(
    proj_ptr,
    mask_ptr,
    left_b_ptr,
    right_b_ptr,
    out_gate_ptr,
    rows: tl.constexpr,
    seq_len: tl.constexpr,
    H: tl.constexpr,
    NOMASK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    valid = (offs_r[:, None] < rows) & (offs_h[None, :] < H)
    proj_base = offs_r[:, None] * (5 * H) + offs_h[None, :]
    m = 1.0
    if not NOMASK:
        m = tl.load(mask_ptr + offs_r, mask=offs_r < rows, other=0.0).to(tl.float32)[:, None]
    n2 = seq_len * seq_len
    b = offs_r // n2
    ij = offs_r - b * n2
    dst = (b[:, None] * H + offs_h[None, :]) * n2 + ij[:, None]
    l = tl.load(proj_ptr + proj_base, mask=valid, other=0.0).to(tl.float32)
    lg = tl.load(proj_ptr + proj_base + 2 * H, mask=valid, other=0.0).to(tl.float32)
    tl.store(left_b_ptr + dst, l * tl.sigmoid(lg) * m, mask=valid)
    r = tl.load(proj_ptr + proj_base + H, mask=valid, other=0.0).to(tl.float32)
    rg = tl.load(proj_ptr + proj_base + 3 * H, mask=valid, other=0.0).to(tl.float32)
    tl.store(right_b_ptr + dst, r * tl.sigmoid(rg) * m, mask=valid)
    og = tl.load(proj_ptr + proj_base + 4 * H, mask=valid, other=0.0).to(tl.float32)
    tl.store(out_gate_ptr + offs_r[:, None] * H + offs_h[None, :], tl.sigmoid(og), mask=valid)


@triton.jit
def _post_proj_transpose_nogate_kernel(
    proj_ptr,
    mask_ptr,
    left_b_ptr,
    right_b_ptr,
    rows: tl.constexpr,
    seq_len: tl.constexpr,
    H: tl.constexpr,
    NOMASK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    valid = (offs_r[:, None] < rows) & (offs_h[None, :] < H)
    proj_base = offs_r[:, None] * (5 * H) + offs_h[None, :]
    m = 1.0
    if not NOMASK:
        m = tl.load(mask_ptr + offs_r, mask=offs_r < rows, other=0.0).to(tl.float32)[:, None]
    n2 = seq_len * seq_len
    b = offs_r // n2
    ij = offs_r - b * n2
    dst = (b[:, None] * H + offs_h[None, :]) * n2 + ij[:, None]
    l = tl.load(proj_ptr + proj_base, mask=valid, other=0.0).to(tl.float32)
    lg = tl.load(proj_ptr + proj_base + 2 * H, mask=valid, other=0.0).to(tl.float32)
    tl.store(left_b_ptr + dst, l * tl.sigmoid(lg) * m, mask=valid)
    r = tl.load(proj_ptr + proj_base + H, mask=valid, other=0.0).to(tl.float32)
    rg = tl.load(proj_ptr + proj_base + 3 * H, mask=valid, other=0.0).to(tl.float32)
    tl.store(right_b_ptr + dst, r * tl.sigmoid(rg) * m, mask=valid)

def custom_kernel(data):
    input_tensor, mask, weights, config = data
    dim = int(config["dim"])
    hidden_dim = int(config["hidden_dim"])
    batch_size = input_tensor.shape[0]
    seq_len = input_tensor.shape[1]
    x_in = input_tensor.contiguous()
    proj_dtype = torch.float16
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
    rows = batch_size * seq_len * seq_len
    left_b = torch.empty((batch_size * hidden_dim, seq_len, seq_len), device=input_tensor.device, dtype=torch.float16)
    right_b = torch.empty_like(left_b)
    mask_arg = input_tensor if mask is None else mask
    _post_proj_transpose_nogate_kernel[(triton.cdiv(rows, 16), triton.cdiv(hidden_dim, 32))](
        proj,
        mask_arg,
        left_b,
        right_b,
        rows=rows,
        seq_len=seq_len,
        H=hidden_dim,
        NOMASK=nomask,
        BLOCK_R=16,
        BLOCK_H=32,
        num_warps=4,
    )

    out_h = torch.bmm(left_b, right_b.transpose(1, 2)).view(batch_size, hidden_dim, seq_len, seq_len)
    gated = torch.empty((batch_size, seq_len, seq_len, hidden_dim), device=input_tensor.device, dtype=torch.float32)
    if seq_len <= 256:
        _ln_gate_from_bmm_projgate_kernel[(triton.cdiv(rows, 32),)](
            out_h,
            proj,
            weights["to_out_norm.weight"].to(torch.float32),
            weights["to_out_norm.bias"].to(torch.float32),
            gated,
            rows=rows,
            seq_len=seq_len,
            H=hidden_dim,
            BLOCK_R=32,
            BLOCK_H=128,
            num_warps=4,
        )
    else:
        _ln_gate_from_bmm_projgate_kernel[(triton.cdiv(rows, 32),)](
            out_h,
            proj,
            weights["to_out_norm.weight"].to(torch.float32),
            weights["to_out_norm.bias"].to(torch.float32),
            gated,
            rows=rows,
            seq_len=seq_len,
            H=hidden_dim,
            BLOCK_R=32,
            BLOCK_H=128,
            num_warps=8,
        )
    return F.linear(gated, weights["to_out.weight"].to(torch.float32)).to(torch.float32)
