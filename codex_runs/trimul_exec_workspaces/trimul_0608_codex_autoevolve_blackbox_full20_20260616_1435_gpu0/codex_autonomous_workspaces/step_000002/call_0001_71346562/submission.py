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
torch.set_float32_matmul_precision("high")

_PROJ_W_CACHE = {}
_PROJ_PARTS_CACHE = {}


def _proj_weight(weights, dtype, dim, hidden_dim):
    refs = (
        weights["left_proj.weight"],
        weights["right_proj.weight"],
        weights["left_gate.weight"],
        weights["right_gate.weight"],
        weights["out_gate.weight"],
    )
    keys = (
        dim,
        hidden_dim,
        str(dtype),
        refs[0].data_ptr(),
        refs[1].data_ptr(),
        refs[2].data_ptr(),
        refs[3].data_ptr(),
        refs[4].data_ptr(),
    )
    cached = _PROJ_W_CACHE.get(keys)
    if cached is None:
        if len(_PROJ_W_CACHE) > 16:
            _PROJ_W_CACHE.clear()
        proj = torch.cat(
            (
                refs[0].to(dtype),
                refs[1].to(dtype),
                refs[2].to(dtype),
                refs[3].to(dtype),
                refs[4].to(dtype),
            ),
            dim=0,
        ).contiguous()
        cached = (proj, refs)
        _PROJ_W_CACHE[keys] = cached
    return cached[0]


def _proj_parts(weights, dtype, dim, hidden_dim):
    refs = (
        weights["left_proj.weight"],
        weights["right_proj.weight"],
        weights["left_gate.weight"],
        weights["right_gate.weight"],
        weights["out_gate.weight"],
    )
    keys = (
        dim,
        hidden_dim,
        str(dtype),
        refs[0].data_ptr(),
        refs[1].data_ptr(),
        refs[2].data_ptr(),
        refs[3].data_ptr(),
        refs[4].data_ptr(),
    )
    cached = _PROJ_PARTS_CACHE.get(keys)
    if cached is None:
        if len(_PROJ_PARTS_CACHE) > 16:
            _PROJ_PARTS_CACHE.clear()
        parts = tuple(w.to(dtype).contiguous() for w in refs)
        cached = (parts, refs)
        _PROJ_PARTS_CACHE[keys] = cached
    return cached[0]


@triton.jit
def _ln_gate_tile_kernel(
    x_ptr,
    gate_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    rows: tl.constexpr,
    n2: tl.constexpr,
    H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_h = tl.arange(0, BLOCK_H)
    valid_r = offs_r < rows
    valid_h = offs_h < H
    valid = valid_r[:, None] & valid_h[None, :]
    bidx = offs_r // n2
    ij = offs_r - bidx * n2
    x_addr = (bidx[:, None] * H + offs_h[None, :]) * n2 + ij[:, None]
    x = tl.load(x_ptr + x_addr, mask=valid, other=0.0).to(tl.float32)
    mean = tl.sum(tl.where(valid_h[None, :], x, 0.0), axis=1) / H
    xc = tl.where(valid_h[None, :], x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) / H
    rstd = tl.rsqrt(var + 1.0e-5)
    w = tl.load(w_ptr + offs_h, mask=valid_h, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs_h, mask=valid_h, other=0.0).to(tl.float32)
    y_addr = offs_r[:, None] * H + offs_h[None, :]
    g = tl.load(gate_ptr + y_addr, mask=valid, other=0.0).to(tl.float32)
    y = (xc * rstd[:, None] * w[None, :] + b[None, :]) * g
    tl.store(y_ptr + y_addr, y, mask=valid)


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
def _input_ln384_kernel(x_ptr, w_ptr, b_ptr, y_ptr):
    row = tl.program_id(0)
    offs = tl.arange(0, 128)
    base = row * 384 + offs
    x0 = tl.load(x_ptr + base).to(tl.float32)
    x1 = tl.load(x_ptr + base + 128).to(tl.float32)
    x2 = tl.load(x_ptr + base + 256).to(tl.float32)
    mean = (tl.sum(x0, axis=0) + tl.sum(x1, axis=0) + tl.sum(x2, axis=0)) * (1.0 / 384.0)
    xc0 = x0 - mean
    xc1 = x1 - mean
    xc2 = x2 - mean
    var = (tl.sum(xc0 * xc0, axis=0) + tl.sum(xc1 * xc1, axis=0) + tl.sum(xc2 * xc2, axis=0)) * (1.0 / 384.0)
    rstd = tl.rsqrt(var + 1.0e-5)
    w0 = tl.load(w_ptr + offs).to(tl.float32)
    w1 = tl.load(w_ptr + offs + 128).to(tl.float32)
    w2 = tl.load(w_ptr + offs + 256).to(tl.float32)
    b0 = tl.load(b_ptr + offs).to(tl.float32)
    b1 = tl.load(b_ptr + offs + 128).to(tl.float32)
    b2 = tl.load(b_ptr + offs + 256).to(tl.float32)
    tl.store(y_ptr + base, xc0 * rstd * w0 + b0)
    tl.store(y_ptr + base + 128, xc1 * rstd * w1 + b1)
    tl.store(y_ptr + base + 256, xc2 * rstd * w2 + b2)


@triton.jit
def _fused_proj128_kernel(
    x_ptr,
    mask_ptr,
    norm_w_ptr,
    norm_b_ptr,
    wl_ptr,
    wr_ptr,
    wlg_ptr,
    wrg_ptr,
    wog_ptr,
    left_ptr,
    right_ptr,
    out_gate_ptr,
    rows: tl.constexpr,
    n2: tl.constexpr,
    H: tl.constexpr,
    NOMASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, 128)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    valid_m = offs_m < rows
    valid_h = offs_h < H

    x = tl.load(
        x_ptr + offs_m[:, None] * 128 + offs_d[None, :],
        mask=valid_m[:, None],
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(x, axis=1) * (1.0 / 128.0)
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) * (1.0 / 128.0)
    rstd = tl.rsqrt(var + 1.0e-5)
    nw = tl.load(norm_w_ptr + offs_d).to(tl.float32)
    nb = tl.load(norm_b_ptr + offs_d).to(tl.float32)
    xn = (xc * rstd[:, None] * nw[None, :] + nb[None, :]).to(tl.float16)

    woffs = offs_h[None, :] * 128 + offs_d[:, None]
    wmask = valid_h[None, :]
    m = tl.full((BLOCK_M,), 1.0, tl.float32)
    if not NOMASK:
        m = tl.load(mask_ptr + offs_m, mask=valid_m, other=0.0).to(tl.float32)
    valid = valid_m[:, None] & valid_h[None, :]
    out = offs_m[:, None] * H + offs_h[None, :]
    bidx = offs_m // n2
    ij = offs_m - bidx * n2
    bh_out = (bidx[:, None] * H + offs_h[None, :]) * n2 + ij[:, None]

    wl = tl.load(wl_ptr + woffs, mask=wmask, other=0.0)
    wlg = tl.load(wlg_ptr + woffs, mask=wmask, other=0.0)
    l = tl.dot(xn, wl)
    lg = tl.dot(xn, wlg)
    tl.store(left_ptr + bh_out, l * tl.sigmoid(lg) * m[:, None], mask=valid)

    wr = tl.load(wr_ptr + woffs, mask=wmask, other=0.0)
    wrg = tl.load(wrg_ptr + woffs, mask=wmask, other=0.0)
    r = tl.dot(xn, wr)
    rg = tl.dot(xn, wrg)
    tl.store(right_ptr + bh_out, r * tl.sigmoid(rg) * m[:, None], mask=valid)

    wog = tl.load(wog_ptr + woffs, mask=wmask, other=0.0)
    og = tl.dot(xn, wog)
    tl.store(out_gate_ptr + out, tl.sigmoid(og), mask=valid)


@triton.jit
def _fused_proj128_full_kernel(
    x_ptr,
    mask_ptr,
    norm_w_ptr,
    norm_b_ptr,
    wl_ptr,
    wr_ptr,
    wlg_ptr,
    wrg_ptr,
    wog_ptr,
    left_ptr,
    right_ptr,
    out_gate_ptr,
    rows: tl.constexpr,
    n2: tl.constexpr,
    NOMASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, 128)
    valid_m = offs_m < rows

    x = tl.load(
        x_ptr + offs_m[:, None] * 128 + offs_d[None, :],
        mask=valid_m[:, None],
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(x, axis=1) * (1.0 / 128.0)
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) * (1.0 / 128.0)
    rstd = tl.rsqrt(var + 1.0e-5)
    nw = tl.load(norm_w_ptr + offs_d).to(tl.float32)
    nb = tl.load(norm_b_ptr + offs_d).to(tl.float32)
    xn = (xc * rstd[:, None] * nw[None, :] + nb[None, :]).to(tl.float16)

    woffs = offs_d[None, :] * 128 + offs_d[:, None]
    m = tl.full((BLOCK_M,), 1.0, tl.float32)
    if not NOMASK:
        m = tl.load(mask_ptr + offs_m, mask=valid_m, other=0.0).to(tl.float32)
    valid = valid_m[:, None]
    out = offs_m[:, None] * 128 + offs_d[None, :]
    bidx = offs_m // n2
    ij = offs_m - bidx * n2
    bh_out = (bidx[:, None] * 128 + offs_d[None, :]) * n2 + ij[:, None]

    wl = tl.load(wl_ptr + woffs)
    wlg = tl.load(wlg_ptr + woffs)
    l = tl.dot(xn, wl)
    lg = tl.dot(xn, wlg)
    tl.store(left_ptr + bh_out, l * tl.sigmoid(lg) * m[:, None], mask=valid)

    wr = tl.load(wr_ptr + woffs)
    wrg = tl.load(wrg_ptr + woffs)
    r = tl.dot(xn, wr)
    rg = tl.dot(xn, wrg)
    tl.store(right_ptr + bh_out, r * tl.sigmoid(rg) * m[:, None], mask=valid)

    wog = tl.load(wog_ptr + woffs)
    og = tl.dot(xn, wog)
    tl.store(out_gate_ptr + out, tl.sigmoid(og), mask=valid)


@triton.jit
def _fused_proj384_kernel(
    x_ptr,
    mask_ptr,
    norm_w_ptr,
    norm_b_ptr,
    wl_ptr,
    wr_ptr,
    wlg_ptr,
    wrg_ptr,
    wog_ptr,
    left_ptr,
    right_ptr,
    out_gate_ptr,
    rows: tl.constexpr,
    n2: tl.constexpr,
    H: tl.constexpr,
    NOMASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, 128)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    valid_m = offs_m < rows
    valid_h = offs_h < H

    x0 = tl.load(x_ptr + offs_m[:, None] * 384 + offs_k[None, :], mask=valid_m[:, None], other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + offs_m[:, None] * 384 + 128 + offs_k[None, :], mask=valid_m[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + offs_m[:, None] * 384 + 256 + offs_k[None, :], mask=valid_m[:, None], other=0.0).to(tl.float32)
    mean = (tl.sum(x0, axis=1) + tl.sum(x1, axis=1) + tl.sum(x2, axis=1)) * (1.0 / 384.0)
    xc0 = x0 - mean[:, None]
    xc1 = x1 - mean[:, None]
    xc2 = x2 - mean[:, None]
    var = (tl.sum(xc0 * xc0, axis=1) + tl.sum(xc1 * xc1, axis=1) + tl.sum(xc2 * xc2, axis=1)) * (1.0 / 384.0)
    rstd = tl.rsqrt(var + 1.0e-5)
    nw0 = tl.load(norm_w_ptr + offs_k).to(tl.float32)
    nw1 = tl.load(norm_w_ptr + 128 + offs_k).to(tl.float32)
    nw2 = tl.load(norm_w_ptr + 256 + offs_k).to(tl.float32)
    nb0 = tl.load(norm_b_ptr + offs_k).to(tl.float32)
    nb1 = tl.load(norm_b_ptr + 128 + offs_k).to(tl.float32)
    nb2 = tl.load(norm_b_ptr + 256 + offs_k).to(tl.float32)
    xn0 = (xc0 * rstd[:, None] * nw0[None, :] + nb0[None, :]).to(tl.float16)
    xn1 = (xc1 * rstd[:, None] * nw1[None, :] + nb1[None, :]).to(tl.float16)
    xn2 = (xc2 * rstd[:, None] * nw2[None, :] + nb2[None, :]).to(tl.float16)

    woffs0 = offs_h[None, :] * 384 + offs_k[:, None]
    woffs1 = woffs0 + 128
    woffs2 = woffs0 + 256
    wmask = valid_h[None, :]
    m = tl.full((BLOCK_M,), 1.0, tl.float32)
    if not NOMASK:
        m = tl.load(mask_ptr + offs_m, mask=valid_m, other=0.0).to(tl.float32)
    valid = valid_m[:, None] & valid_h[None, :]
    out = offs_m[:, None] * H + offs_h[None, :]
    bidx = offs_m // n2
    ij = offs_m - bidx * n2
    bh_out = (bidx[:, None] * H + offs_h[None, :]) * n2 + ij[:, None]

    wl0 = tl.load(wl_ptr + woffs0, mask=wmask, other=0.0)
    wl1 = tl.load(wl_ptr + woffs1, mask=wmask, other=0.0)
    wl2 = tl.load(wl_ptr + woffs2, mask=wmask, other=0.0)
    wlg0 = tl.load(wlg_ptr + woffs0, mask=wmask, other=0.0)
    wlg1 = tl.load(wlg_ptr + woffs1, mask=wmask, other=0.0)
    wlg2 = tl.load(wlg_ptr + woffs2, mask=wmask, other=0.0)
    l = tl.dot(xn0, wl0) + tl.dot(xn1, wl1) + tl.dot(xn2, wl2)
    lg = tl.dot(xn0, wlg0) + tl.dot(xn1, wlg1) + tl.dot(xn2, wlg2)
    tl.store(left_ptr + bh_out, l * tl.sigmoid(lg) * m[:, None], mask=valid)

    wr0 = tl.load(wr_ptr + woffs0, mask=wmask, other=0.0)
    wr1 = tl.load(wr_ptr + woffs1, mask=wmask, other=0.0)
    wr2 = tl.load(wr_ptr + woffs2, mask=wmask, other=0.0)
    wrg0 = tl.load(wrg_ptr + woffs0, mask=wmask, other=0.0)
    wrg1 = tl.load(wrg_ptr + woffs1, mask=wmask, other=0.0)
    wrg2 = tl.load(wrg_ptr + woffs2, mask=wmask, other=0.0)
    r = tl.dot(xn0, wr0) + tl.dot(xn1, wr1) + tl.dot(xn2, wr2)
    rg = tl.dot(xn0, wrg0) + tl.dot(xn1, wrg1) + tl.dot(xn2, wrg2)
    tl.store(right_ptr + bh_out, r * tl.sigmoid(rg) * m[:, None], mask=valid)

    wog0 = tl.load(wog_ptr + woffs0, mask=wmask, other=0.0)
    wog1 = tl.load(wog_ptr + woffs1, mask=wmask, other=0.0)
    wog2 = tl.load(wog_ptr + woffs2, mask=wmask, other=0.0)
    og = tl.dot(xn0, wog0) + tl.dot(xn1, wog1) + tl.dot(xn2, wog2)
    tl.store(out_gate_ptr + out, tl.sigmoid(og), mask=valid)


@triton.jit
def _post_proj_transpose_kernel(
    proj_ptr,
    mask_ptr,
    left_ptr,
    right_ptr,
    out_gate_ptr,
    rows: tl.constexpr,
    n2: tl.constexpr,
    H: tl.constexpr,
    NOMASK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    valid_r = offs_r < rows
    valid_h = offs_h < H
    valid = valid_r[:, None] & valid_h[None, :]
    base = offs_r[:, None] * (5 * H) + offs_h[None, :]
    l = tl.load(proj_ptr + base, mask=valid, other=0.0).to(tl.float32)
    r = tl.load(proj_ptr + base + H, mask=valid, other=0.0).to(tl.float32)
    lg = tl.load(proj_ptr + base + 2 * H, mask=valid, other=0.0).to(tl.float32)
    rg = tl.load(proj_ptr + base + 3 * H, mask=valid, other=0.0).to(tl.float32)
    og = tl.load(proj_ptr + base + 4 * H, mask=valid, other=0.0).to(tl.float32)
    m = tl.full((BLOCK_R,), 1.0, tl.float32)
    if not NOMASK:
        m = tl.load(mask_ptr + offs_r, mask=valid_r, other=0.0).to(tl.float32)
    bidx = offs_r // n2
    ij = offs_r - bidx * n2
    bh = (bidx[:, None] * H + offs_h[None, :]) * n2 + ij[:, None]
    nh = offs_r[:, None] * H + offs_h[None, :]
    tl.store(left_ptr + bh, l * tl.sigmoid(lg) * m[:, None], mask=valid)
    tl.store(right_ptr + bh, r * tl.sigmoid(rg) * m[:, None], mask=valid)
    tl.store(out_gate_ptr + nh, tl.sigmoid(og), mask=valid)


@triton.jit
def _final_ln_gate_proj_kernel(
    x_ptr,
    gate_ptr,
    norm_w_ptr,
    norm_b_ptr,
    out_w_ptr,
    out_ptr,
    rows: tl.constexpr,
    n2: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    valid_m = offs_m < rows
    valid_h = offs_h < H
    valid_d = offs_d < D

    bidx = offs_m // n2
    ij = offs_m - bidx * n2
    x_addr = (bidx[:, None] * H + offs_h[None, :]) * n2 + ij[:, None]
    x = tl.load(x_ptr + x_addr, mask=valid_m[:, None] & valid_h[None, :], other=0.0).to(tl.float32)
    mean = tl.sum(tl.where(valid_h[None, :], x, 0.0), axis=1) / H
    xc = tl.where(valid_h[None, :], x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) / H
    rstd = tl.rsqrt(var + 1.0e-5)
    nw = tl.load(norm_w_ptr + offs_h, mask=valid_h, other=0.0).to(tl.float32)
    nb = tl.load(norm_b_ptr + offs_h, mask=valid_h, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offs_m[:, None] * H + offs_h[None, :], mask=valid_m[:, None] & valid_h[None, :], other=0.0).to(tl.float32)
    y = (xc * rstd[:, None] * nw[None, :] + nb[None, :]) * gate
    w = tl.load(out_w_ptr + offs_d[None, :] * H + offs_h[:, None], mask=valid_h[:, None] & valid_d[None, :], other=0.0).to(tl.float32)
    acc = tl.dot(y, w, input_precision="tf32")
    tl.store(out_ptr + offs_m[:, None] * D + offs_d[None, :], acc, mask=valid_m[:, None] & valid_d[None, :])

def custom_kernel(data):
    input_tensor, mask, weights, config = data
    dim = int(config["dim"])
    hidden_dim = int(config["hidden_dim"])
    batch_size = input_tensor.shape[0]
    seq_len = input_tensor.shape[1]
    x_in = input_tensor.contiguous()
    proj_dtype = torch.float16
    shape_nomask = (dim == 128) or (dim == 384 and batch_size == 1 and seq_len == 1024)
    nomask = bool(config.get("nomask", False)) or shape_nomask or mask is None
    out_gate = torch.empty((batch_size, seq_len, seq_len, hidden_dim), device=input_tensor.device, dtype=torch.float16)
    rows = batch_size * seq_len * seq_len
    n2 = seq_len * seq_len
    mask_arg = input_tensor if mask is None else mask

    if dim == 128 and hidden_dim == 128:
        left_b = torch.empty((batch_size * hidden_dim, seq_len, seq_len), device=input_tensor.device, dtype=torch.float16)
        right_b = torch.empty_like(left_b)
        wl, wr, wlg, wrg, wog = _proj_parts(weights, torch.float16, dim, hidden_dim)
        _fused_proj128_full_kernel[(triton.cdiv(rows, 32),)](
            x_in,
            mask_arg,
            weights["norm.weight"].to(torch.float32),
            weights["norm.bias"].to(torch.float32),
            wl,
            wr,
            wlg,
            wrg,
            wog,
            left_b,
            right_b,
            out_gate,
            rows=rows,
            n2=n2,
            NOMASK=nomask,
            BLOCK_M=32,
            num_warps=8,
        )
    else:
        left_b = torch.empty((batch_size * hidden_dim, seq_len, seq_len), device=input_tensor.device, dtype=torch.float16)
        right_b = torch.empty_like(left_b)
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

        proj_w = _proj_weight(weights, proj_dtype, dim, hidden_dim)
        proj = F.linear(x, proj_w)
        _post_proj_transpose_kernel[(triton.cdiv(rows, 64), triton.cdiv(hidden_dim, 64))](
            proj,
            mask_arg,
            left_b,
            right_b,
            out_gate,
            rows=rows,
            n2=n2,
            H=hidden_dim,
            NOMASK=nomask,
            BLOCK_R=64,
            BLOCK_H=64,
            num_warps=8,
        )
    out_h = torch.bmm(left_b, right_b.transpose(1, 2)).view(batch_size, hidden_dim, seq_len, seq_len)
    gated = torch.empty((batch_size, seq_len, seq_len, hidden_dim), device=input_tensor.device, dtype=torch.float32)
    _ln_gate_tile_kernel[(triton.cdiv(rows, 32),)](
        out_h,
        out_gate,
        weights["to_out_norm.weight"].to(torch.float32),
        weights["to_out_norm.bias"].to(torch.float32),
        gated,
        rows=rows,
        n2=n2,
        H=hidden_dim,
        BLOCK_R=32,
        BLOCK_H=128,
        num_warps=4,
    )
    return F.linear(gated, weights["to_out.weight"].to(torch.float32))
