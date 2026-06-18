"""
Optimized outgoing TriMul forward: Triton handles row layernorms, projection
post-processing, gates, and masks; FP16 batched GEMMs compute the triangular
update; fp16-safe output paths fuse row-normalize, gate, and final projection
for dim 128/384/768, with fp32 fallback kept for unsupported dimensions.
"""

import torch
import triton
import triton.language as tl

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")

_PROJ_W_CACHE = {}
_PROJ_PARTS_CACHE = {}
_TO_OUT_CACHE = {}
_TENSOR_CACHE = {}
_VEC_CACHE = {}
_BUF_CACHE = {}
_BUF_SHAPE = None
_STREAM_CACHE = {}


def _buf(shape_key, name, shape, dtype, device):
    global _BUF_SHAPE
    if _BUF_SHAPE != shape_key:
        _BUF_CACHE.clear()
        _BUF_SHAPE = shape_key
    key = (name, dtype)
    cached = _BUF_CACHE.get(key)
    if cached is None:
        cached = torch.empty(shape, device=device, dtype=dtype)
        _BUF_CACHE[key] = cached
    return cached


def _copy_stream(device):
    key = device.index if hasattr(device, "index") else device
    stream = _STREAM_CACHE.get(key)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _STREAM_CACHE[key] = stream
    return stream


def _proj_weight(weights, dtype, dim, hidden_dim):
    refs = (
        weights["left_proj.weight"],
        weights["right_proj.weight"],
        weights["left_gate.weight"],
        weights["right_gate.weight"],
        weights["out_gate.weight"],
    )
    return torch.cat(
        (
            refs[0].to(dtype),
            refs[1].to(dtype),
            refs[2].to(dtype),
            refs[3].to(dtype),
            refs[4].to(dtype),
        ),
        dim=0,
    ).contiguous()


def _tensor_cache_key(ref, dtype):
    return (str(dtype), ref, ref._version)


def _tensor_to(ref, dtype):
    if ref.dtype == dtype and ref.is_contiguous():
        return ref
    key = _tensor_cache_key(ref, dtype)
    cached = _TENSOR_CACHE.get(key)
    if cached is None:
        if len(_TENSOR_CACHE) > 96:
            _TENSOR_CACHE.clear()
        cached = ref.to(dtype).contiguous()
        _TENSOR_CACHE[key] = cached
    return cached


def _proj_weight_cached(weights, dtype):
    refs = (
        weights["left_proj.weight"],
        weights["right_proj.weight"],
        weights["left_gate.weight"],
        weights["right_gate.weight"],
        weights["out_gate.weight"],
    )
    key = (str(dtype),) + tuple((w, w._version) for w in refs)
    cached = _PROJ_W_CACHE.get(key)
    if cached is None:
        if len(_PROJ_W_CACHE) > 16:
            _PROJ_W_CACHE.clear()
        cached = torch.cat(tuple(w.to(dtype) for w in refs), dim=0).contiguous()
        _PROJ_W_CACHE[key] = cached
    return cached


def _proj_parts(weights, dtype, dim, hidden_dim):
    refs = (
        weights["left_proj.weight"],
        weights["right_proj.weight"],
        weights["left_gate.weight"],
        weights["right_gate.weight"],
        weights["out_gate.weight"],
    )
    return tuple(w.to(dtype).contiguous() for w in refs)


def _to_out_weight_t(weights, dtype):
    ref = weights["to_out.weight"]
    key = _tensor_cache_key(ref, dtype) + ("t",)
    cached = _TO_OUT_CACHE.get(key)
    if cached is None:
        if len(_TO_OUT_CACHE) > 32:
            _TO_OUT_CACHE.clear()
        cached = ref.to(dtype).t().contiguous()
        _TO_OUT_CACHE[key] = cached
    return cached


def _vec_to(ref, dtype):
    key = (str(dtype), ref.data_ptr(), ref._version)
    cached = _VEC_CACHE.get(key)
    if cached is None:
        if len(_VEC_CACHE) > 32:
            _VEC_CACHE.clear()
        cached = ref.to(dtype).contiguous()
        _VEC_CACHE[key] = cached
    return cached


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
def _ln_gate_out_kernel(
    x_ptr,
    gate_ptr,
    w_ptr,
    b_ptr,
    out_w_t_ptr,
    out_ptr,
    rows: tl.constexpr,
    n2: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_h = tl.arange(0, 128)
    valid_r = offs_r < rows
    bidx = offs_r // n2
    ij = offs_r - bidx * n2
    x_addr = (bidx[:, None] * H + offs_h[None, :]) * n2 + ij[:, None]
    x = tl.load(x_ptr + x_addr, mask=valid_r[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) * (1.0 / 128.0)
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) * (1.0 / 128.0)
    rstd = tl.rsqrt(var + 1.0e-5)
    nw = tl.load(w_ptr + offs_h).to(tl.float32)
    nb = tl.load(b_ptr + offs_h).to(tl.float32)
    g = tl.load(gate_ptr + offs_r[:, None] * H + offs_h[None, :], mask=valid_r[:, None], other=0.0).to(tl.float32)
    y = ((xc * rstd[:, None] * nw[None, :] + nb[None, :]) * g).to(tl.float16)
    offs_d_base = tl.arange(0, BLOCK_D)
    for q in tl.static_range(0, D // BLOCK_D):
        offs_d = q * BLOCK_D + offs_d_base
        ow = tl.load(out_w_t_ptr + offs_h[:, None] * D + offs_d[None, :])
        acc = tl.dot(y, ow)
        tl.store(out_ptr + offs_r[:, None] * D + offs_d[None, :], acc, mask=valid_r[:, None])


@triton.jit
def _ln_gate_out_f32_kernel(
    x_ptr,
    gate_ptr,
    w_ptr,
    b_ptr,
    out_w_t_ptr,
    out_ptr,
    rows: tl.constexpr,
    n2: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_h = tl.arange(0, 128)
    valid_r = offs_r < rows
    bidx = offs_r // n2
    ij = offs_r - bidx * n2
    x_addr = (bidx[:, None] * H + offs_h[None, :]) * n2 + ij[:, None]
    x = tl.load(x_ptr + x_addr, mask=valid_r[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) * (1.0 / 128.0)
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) * (1.0 / 128.0)
    rstd = tl.rsqrt(var + 1.0e-5)
    nw = tl.load(w_ptr + offs_h).to(tl.float32)
    nb = tl.load(b_ptr + offs_h).to(tl.float32)
    g = tl.load(gate_ptr + offs_r[:, None] * H + offs_h[None, :], mask=valid_r[:, None], other=0.0).to(tl.float32)
    y = (xc * rstd[:, None] * nw[None, :] + nb[None, :]) * g
    offs_d_base = tl.arange(0, BLOCK_D)
    for q in tl.static_range(0, D // BLOCK_D):
        offs_d = q * BLOCK_D + offs_d_base
        ow = tl.load(out_w_t_ptr + offs_h[:, None] * D + offs_d[None, :]).to(tl.float32)
        acc = tl.dot(y, ow, input_precision="tf32x3")
        tl.store(out_ptr + offs_r[:, None] * D + offs_d[None, :], acc, mask=valid_r[:, None])


@triton.jit
def _ln_gate_out_pre_gate_kernel(
    x_ptr,
    gate_ptr,
    w_ptr,
    b_ptr,
    out_w_t_ptr,
    out_ptr,
    rows: tl.constexpr,
    n2: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_h = tl.arange(0, 128)
    valid_r = offs_r < rows
    bidx = offs_r // n2
    ij = offs_r - bidx * n2
    x_addr = (bidx[:, None] * H + offs_h[None, :]) * n2 + ij[:, None]
    x = tl.load(x_ptr + x_addr, mask=valid_r[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) * (1.0 / 128.0)
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) * (1.0 / 128.0)
    rstd = tl.rsqrt(var + 1.0e-5)
    nw = tl.load(w_ptr + offs_h).to(tl.float32)
    nb = tl.load(b_ptr + offs_h).to(tl.float32)
    gpre = tl.load(gate_ptr + offs_r[:, None] * H + offs_h[None, :], mask=valid_r[:, None], other=0.0).to(tl.float32)
    g = tl.sigmoid(gpre).to(tl.float16).to(tl.float32)
    y = ((xc * rstd[:, None] * nw[None, :] + nb[None, :]) * g).to(tl.float16)
    offs_d_base = tl.arange(0, BLOCK_D)
    for q in tl.static_range(0, D // BLOCK_D):
        offs_d = q * BLOCK_D + offs_d_base
        ow = tl.load(out_w_t_ptr + offs_h[:, None] * D + offs_d[None, :])
        acc = tl.dot(y, ow)
        tl.store(out_ptr + offs_r[:, None] * D + offs_d[None, :], acc, mask=valid_r[:, None])


@triton.jit
def _ln_gate_out_proj_gate_kernel(
    x_ptr,
    proj_ptr,
    w_ptr,
    b_ptr,
    out_w_t_ptr,
    out_ptr,
    rows: tl.constexpr,
    n2: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_h = tl.arange(0, 128)
    valid_r = offs_r < rows
    bidx = offs_r // n2
    ij = offs_r - bidx * n2
    x_addr = (bidx[:, None] * H + offs_h[None, :]) * n2 + ij[:, None]
    x = tl.load(x_ptr + x_addr, mask=valid_r[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) * (1.0 / 128.0)
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) * (1.0 / 128.0)
    rstd = tl.rsqrt(var + 1.0e-5)
    nw = tl.load(w_ptr + offs_h).to(tl.float32)
    nb = tl.load(b_ptr + offs_h).to(tl.float32)
    og = tl.load(proj_ptr + offs_r[:, None] * (5 * H) + (4 * H + offs_h)[None, :], mask=valid_r[:, None], other=0.0).to(tl.float32)
    g = tl.sigmoid(og).to(tl.float16).to(tl.float32)
    y = ((xc * rstd[:, None] * nw[None, :] + nb[None, :]) * g).to(tl.float16)
    offs_d_base = tl.arange(0, BLOCK_D)
    for q in tl.static_range(0, D // BLOCK_D):
        offs_d = q * BLOCK_D + offs_d_base
        ow = tl.load(out_w_t_ptr + offs_h[:, None] * D + offs_d[None, :])
        acc = tl.dot(y, ow)
        tl.store(out_ptr + offs_r[:, None] * D + offs_d[None, :], acc, mask=valid_r[:, None])


@triton.jit
def _ln_store_pre_gate_kernel(
    x_ptr,
    gate_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    rows: tl.constexpr,
    n2: tl.constexpr,
    H: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_h = tl.arange(0, 128)
    valid_r = offs_r < rows
    bidx = offs_r // n2
    ij = offs_r - bidx * n2
    x_addr = (bidx[:, None] * H + offs_h[None, :]) * n2 + ij[:, None]
    x = tl.load(x_ptr + x_addr, mask=valid_r[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) * (1.0 / 128.0)
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) * (1.0 / 128.0)
    rstd = tl.rsqrt(var + 1.0e-5)
    nw = tl.load(w_ptr + offs_h).to(tl.float32)
    nb = tl.load(b_ptr + offs_h).to(tl.float32)
    gpre = tl.load(gate_ptr + offs_r[:, None] * H + offs_h[None, :], mask=valid_r[:, None], other=0.0).to(tl.float32)
    g = tl.sigmoid(gpre).to(tl.float16).to(tl.float32)
    y = ((xc * rstd[:, None] * nw[None, :] + nb[None, :]) * g).to(tl.float16)
    tl.store(y_ptr + offs_r[:, None] * H + offs_h[None, :], y, mask=valid_r[:, None])


@triton.jit
def _ln_store_proj_gate_kernel(
    x_ptr,
    proj_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    rows: tl.constexpr,
    n2: tl.constexpr,
    H: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_h = tl.arange(0, 128)
    valid_r = offs_r < rows
    bidx = offs_r // n2
    ij = offs_r - bidx * n2
    x_addr = (bidx[:, None] * H + offs_h[None, :]) * n2 + ij[:, None]
    x = tl.load(x_ptr + x_addr, mask=valid_r[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) * (1.0 / 128.0)
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) * (1.0 / 128.0)
    rstd = tl.rsqrt(var + 1.0e-5)
    nw = tl.load(w_ptr + offs_h).to(tl.float32)
    nb = tl.load(b_ptr + offs_h).to(tl.float32)
    og = tl.load(proj_ptr + offs_r[:, None] * (5 * H) + (4 * H + offs_h)[None, :], mask=valid_r[:, None], other=0.0).to(tl.float32)
    g = tl.sigmoid(og).to(tl.float16).to(tl.float32)
    y = ((xc * rstd[:, None] * nw[None, :] + nb[None, :]) * g).to(tl.float16)
    tl.store(y_ptr + offs_r[:, None] * H + offs_h[None, :], y, mask=valid_r[:, None])

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
def _input_ln384_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    rows: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    offs = tl.arange(0, 128)
    valid = row < rows
    base = row[:, None] * 384 + offs[None, :]
    x0 = tl.load(x_ptr + base, mask=valid[:, None], other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + base + 128, mask=valid[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + base + 256, mask=valid[:, None], other=0.0).to(tl.float32)
    mean = (tl.sum(x0, axis=1) + tl.sum(x1, axis=1) + tl.sum(x2, axis=1)) * (1.0 / 384.0)
    xc0 = x0 - mean[:, None]
    xc1 = x1 - mean[:, None]
    xc2 = x2 - mean[:, None]
    var = (tl.sum(xc0 * xc0, axis=1) + tl.sum(xc1 * xc1, axis=1) + tl.sum(xc2 * xc2, axis=1)) * (1.0 / 384.0)
    rstd = tl.rsqrt(var + 1.0e-5)
    w0 = tl.load(w_ptr + offs).to(tl.float32)
    b0 = tl.load(b_ptr + offs).to(tl.float32)
    w1 = tl.load(w_ptr + offs + 128).to(tl.float32)
    b1 = tl.load(b_ptr + offs + 128).to(tl.float32)
    w2 = tl.load(w_ptr + offs + 256).to(tl.float32)
    b2 = tl.load(b_ptr + offs + 256).to(tl.float32)
    tl.store(y_ptr + base, xc0 * rstd[:, None] * w0[None, :] + b0[None, :], mask=valid[:, None])
    tl.store(y_ptr + base + 128, xc1 * rstd[:, None] * w1[None, :] + b1[None, :], mask=valid[:, None])
    tl.store(y_ptr + base + 256, xc2 * rstd[:, None] * w2[None, :] + b2[None, :], mask=valid[:, None])


@triton.jit
def _input_ln768_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    rows: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    offs = tl.arange(0, 128)
    valid = row < rows
    base = row[:, None] * 768 + offs[None, :]
    x0 = tl.load(x_ptr + base, mask=valid[:, None], other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + base + 128, mask=valid[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + base + 256, mask=valid[:, None], other=0.0).to(tl.float32)
    x3 = tl.load(x_ptr + base + 384, mask=valid[:, None], other=0.0).to(tl.float32)
    x4 = tl.load(x_ptr + base + 512, mask=valid[:, None], other=0.0).to(tl.float32)
    x5 = tl.load(x_ptr + base + 640, mask=valid[:, None], other=0.0).to(tl.float32)
    mean = (
        tl.sum(x0, axis=1)
        + tl.sum(x1, axis=1)
        + tl.sum(x2, axis=1)
        + tl.sum(x3, axis=1)
        + tl.sum(x4, axis=1)
        + tl.sum(x5, axis=1)
    ) * (1.0 / 768.0)
    xc0 = x0 - mean[:, None]
    xc1 = x1 - mean[:, None]
    xc2 = x2 - mean[:, None]
    xc3 = x3 - mean[:, None]
    xc4 = x4 - mean[:, None]
    xc5 = x5 - mean[:, None]
    var = (
        tl.sum(xc0 * xc0, axis=1)
        + tl.sum(xc1 * xc1, axis=1)
        + tl.sum(xc2 * xc2, axis=1)
        + tl.sum(xc3 * xc3, axis=1)
        + tl.sum(xc4 * xc4, axis=1)
        + tl.sum(xc5 * xc5, axis=1)
    ) * (1.0 / 768.0)
    rstd = tl.rsqrt(var + 1.0e-5)
    w0 = tl.load(w_ptr + offs).to(tl.float32)
    b0 = tl.load(b_ptr + offs).to(tl.float32)
    w1 = tl.load(w_ptr + offs + 128).to(tl.float32)
    b1 = tl.load(b_ptr + offs + 128).to(tl.float32)
    w2 = tl.load(w_ptr + offs + 256).to(tl.float32)
    b2 = tl.load(b_ptr + offs + 256).to(tl.float32)
    w3 = tl.load(w_ptr + offs + 384).to(tl.float32)
    b3 = tl.load(b_ptr + offs + 384).to(tl.float32)
    w4 = tl.load(w_ptr + offs + 512).to(tl.float32)
    b4 = tl.load(b_ptr + offs + 512).to(tl.float32)
    w5 = tl.load(w_ptr + offs + 640).to(tl.float32)
    b5 = tl.load(b_ptr + offs + 640).to(tl.float32)
    tl.store(y_ptr + base, xc0 * rstd[:, None] * w0[None, :] + b0[None, :], mask=valid[:, None])
    tl.store(y_ptr + base + 128, xc1 * rstd[:, None] * w1[None, :] + b1[None, :], mask=valid[:, None])
    tl.store(y_ptr + base + 256, xc2 * rstd[:, None] * w2[None, :] + b2[None, :], mask=valid[:, None])
    tl.store(y_ptr + base + 384, xc3 * rstd[:, None] * w3[None, :] + b3[None, :], mask=valid[:, None])
    tl.store(y_ptr + base + 512, xc4 * rstd[:, None] * w4[None, :] + b4[None, :], mask=valid[:, None])
    tl.store(y_ptr + base + 640, xc5 * rstd[:, None] * w5[None, :] + b5[None, :], mask=valid[:, None])


@triton.jit
def _fused_proj128_4q_kernel(
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
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, 128)
    offs_h_base = tl.arange(0, BLOCK_H)
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

    m = tl.full((BLOCK_M,), 1.0, tl.float32)
    if not NOMASK:
        m = tl.load(mask_ptr + offs_m, mask=valid_m, other=0.0).to(tl.float32)
    bidx = offs_m // n2
    ij = offs_m - bidx * n2
    valid = valid_m[:, None]

    for q in tl.static_range(0, 128 // BLOCK_H):
        offs_h = offs_h_base + q * BLOCK_H
        woffs = offs_h[None, :] * 128 + offs_d[:, None]
        out = offs_m[:, None] * 128 + offs_h[None, :]
        bh = (bidx[:, None] * 128 + offs_h[None, :]) * n2 + ij[:, None]
        wl = tl.load(wl_ptr + woffs)
        wlg = tl.load(wlg_ptr + woffs)
        l = tl.dot(xn, wl)
        lg = tl.dot(xn, wlg)
        tl.store(left_ptr + bh, l * tl.sigmoid(lg) * m[:, None], mask=valid)
        wr = tl.load(wr_ptr + woffs)
        wrg = tl.load(wrg_ptr + woffs)
        r = tl.dot(xn, wr)
        rg = tl.dot(xn, wrg)
        tl.store(right_ptr + bh, r * tl.sigmoid(rg) * m[:, None], mask=valid)
        wog = tl.load(wog_ptr + woffs)
        og = tl.dot(xn, wog)
        tl.store(out_gate_ptr + out, og, mask=valid)


@triton.jit
def _fused_proj128_lr_4q_kernel(
    x_ptr,
    mask_ptr,
    norm_w_ptr,
    norm_b_ptr,
    wl_ptr,
    wr_ptr,
    wlg_ptr,
    wrg_ptr,
    left_ptr,
    right_ptr,
    rows: tl.constexpr,
    n2: tl.constexpr,
    NOMASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, 128)
    offs_h_base = tl.arange(0, BLOCK_H)
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

    m = tl.full((BLOCK_M,), 1.0, tl.float32)
    if not NOMASK:
        m = tl.load(mask_ptr + offs_m, mask=valid_m, other=0.0).to(tl.float32)
    bidx = offs_m // n2
    ij = offs_m - bidx * n2
    valid = valid_m[:, None]

    for q in tl.static_range(0, 128 // BLOCK_H):
        offs_h = offs_h_base + q * BLOCK_H
        woffs = offs_h[None, :] * 128 + offs_d[:, None]
        bh = (bidx[:, None] * 128 + offs_h[None, :]) * n2 + ij[:, None]
        wl = tl.load(wl_ptr + woffs)
        wlg = tl.load(wlg_ptr + woffs)
        l = tl.dot(xn, wl)
        lg = tl.dot(xn, wlg)
        tl.store(left_ptr + bh, l * tl.sigmoid(lg) * m[:, None], mask=valid)
        wr = tl.load(wr_ptr + woffs)
        wrg = tl.load(wrg_ptr + woffs)
        r = tl.dot(xn, wr)
        rg = tl.dot(xn, wrg)
        tl.store(right_ptr + bh, r * tl.sigmoid(rg) * m[:, None], mask=valid)


@triton.jit
def _fused_out_gate128_4q_kernel(
    x_ptr,
    norm_w_ptr,
    norm_b_ptr,
    wog_ptr,
    out_gate_ptr,
    rows: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, 128)
    offs_h_base = tl.arange(0, BLOCK_H)
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
    valid = valid_m[:, None]
    for q in tl.static_range(0, 128 // BLOCK_H):
        offs_h = offs_h_base + q * BLOCK_H
        woffs = offs_h[None, :] * 128 + offs_d[:, None]
        out = offs_m[:, None] * 128 + offs_h[None, :]
        wog = tl.load(wog_ptr + woffs)
        og = tl.dot(xn, wog)
        tl.store(out_gate_ptr + out, tl.sigmoid(og), mask=valid)


@triton.jit
def _fused_proj384_4q_kernel(
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
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, 128)
    offs_h_base = tl.arange(0, BLOCK_H)
    valid_m = offs_m < rows
    base = offs_m[:, None] * 384 + offs_d[None, :]

    x0 = tl.load(x_ptr + base, mask=valid_m[:, None], other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + base + 128, mask=valid_m[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + base + 256, mask=valid_m[:, None], other=0.0).to(tl.float32)
    mean = (tl.sum(x0, axis=1) + tl.sum(x1, axis=1) + tl.sum(x2, axis=1)) * (1.0 / 384.0)
    xc0 = x0 - mean[:, None]
    xc1 = x1 - mean[:, None]
    xc2 = x2 - mean[:, None]
    var = (tl.sum(xc0 * xc0, axis=1) + tl.sum(xc1 * xc1, axis=1) + tl.sum(xc2 * xc2, axis=1)) * (1.0 / 384.0)
    rstd = tl.rsqrt(var + 1.0e-5)
    nw0 = tl.load(norm_w_ptr + offs_d).to(tl.float32)
    nb0 = tl.load(norm_b_ptr + offs_d).to(tl.float32)
    nw1 = tl.load(norm_w_ptr + offs_d + 128).to(tl.float32)
    nb1 = tl.load(norm_b_ptr + offs_d + 128).to(tl.float32)
    nw2 = tl.load(norm_w_ptr + offs_d + 256).to(tl.float32)
    nb2 = tl.load(norm_b_ptr + offs_d + 256).to(tl.float32)
    xn0 = (xc0 * rstd[:, None] * nw0[None, :] + nb0[None, :]).to(tl.float16)
    xn1 = (xc1 * rstd[:, None] * nw1[None, :] + nb1[None, :]).to(tl.float16)
    xn2 = (xc2 * rstd[:, None] * nw2[None, :] + nb2[None, :]).to(tl.float16)

    m = tl.full((BLOCK_M,), 1.0, tl.float32)
    if not NOMASK:
        m = tl.load(mask_ptr + offs_m, mask=valid_m, other=0.0).to(tl.float32)
    bidx = offs_m // n2
    ij = offs_m - bidx * n2
    valid = valid_m[:, None]

    for q in tl.static_range(0, 128 // BLOCK_H):
        offs_h = offs_h_base + q * BLOCK_H
        woffs = offs_h[None, :] * 384 + offs_d[:, None]
        out = offs_m[:, None] * 128 + offs_h[None, :]
        bh = (bidx[:, None] * 128 + offs_h[None, :]) * n2 + ij[:, None]

        wl0 = tl.load(wl_ptr + woffs)
        wl1 = tl.load(wl_ptr + woffs + 128)
        wl2 = tl.load(wl_ptr + woffs + 256)
        wlg0 = tl.load(wlg_ptr + woffs)
        wlg1 = tl.load(wlg_ptr + woffs + 128)
        wlg2 = tl.load(wlg_ptr + woffs + 256)
        l = tl.dot(xn0, wl0) + tl.dot(xn1, wl1) + tl.dot(xn2, wl2)
        lg = tl.dot(xn0, wlg0) + tl.dot(xn1, wlg1) + tl.dot(xn2, wlg2)
        tl.store(left_ptr + bh, l * tl.sigmoid(lg) * m[:, None], mask=valid)

        wr0 = tl.load(wr_ptr + woffs)
        wr1 = tl.load(wr_ptr + woffs + 128)
        wr2 = tl.load(wr_ptr + woffs + 256)
        wrg0 = tl.load(wrg_ptr + woffs)
        wrg1 = tl.load(wrg_ptr + woffs + 128)
        wrg2 = tl.load(wrg_ptr + woffs + 256)
        r = tl.dot(xn0, wr0) + tl.dot(xn1, wr1) + tl.dot(xn2, wr2)
        rg = tl.dot(xn0, wrg0) + tl.dot(xn1, wrg1) + tl.dot(xn2, wrg2)
        tl.store(right_ptr + bh, r * tl.sigmoid(rg) * m[:, None], mask=valid)

        wog0 = tl.load(wog_ptr + woffs)
        wog1 = tl.load(wog_ptr + woffs + 128)
        wog2 = tl.load(wog_ptr + woffs + 256)
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
def _post_proj_transpose_no_og_kernel(
    proj_ptr,
    mask_ptr,
    left_ptr,
    right_ptr,
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
    m = tl.full((BLOCK_R,), 1.0, tl.float32)
    if not NOMASK:
        m = tl.load(mask_ptr + offs_r, mask=valid_r, other=0.0).to(tl.float32)
    bidx = offs_r // n2
    ij = offs_r - bidx * n2
    bh = (bidx[:, None] * H + offs_h[None, :]) * n2 + ij[:, None]
    tl.store(left_ptr + bh, l * tl.sigmoid(lg) * m[:, None], mask=valid)
    tl.store(right_ptr + bh, r * tl.sigmoid(rg) * m[:, None], mask=valid)


@triton.jit
def _post_proj_transpose_t_kernel(
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
    valid = valid_h[:, None] & valid_r[None, :]
    base = offs_r[None, :] * (5 * H) + offs_h[:, None]
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
    bh = (bidx[None, :] * H + offs_h[:, None]) * n2 + ij[None, :]
    nh = offs_r[None, :] * H + offs_h[:, None]
    tl.store(left_ptr + bh, l * tl.sigmoid(lg) * m[None, :], mask=valid)
    tl.store(right_ptr + bh, r * tl.sigmoid(rg) * m[None, :], mask=valid)
    tl.store(out_gate_ptr + nh, tl.sigmoid(og), mask=valid)


@triton.jit
def _post_proj_transpose_nomask_kernel(
    proj_ptr,
    left_ptr,
    right_ptr,
    out_gate_ptr,
    rows: tl.constexpr,
    n2: tl.constexpr,
    H: tl.constexpr,
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
    bidx = offs_r // n2
    ij = offs_r - bidx * n2
    bh = (bidx[:, None] * H + offs_h[None, :]) * n2 + ij[:, None]
    nh = offs_r[:, None] * H + offs_h[None, :]
    tl.store(left_ptr + bh, l * tl.sigmoid(lg), mask=valid)
    tl.store(right_ptr + bh, r * tl.sigmoid(rg), mask=valid)
    tl.store(out_gate_ptr + nh, tl.sigmoid(og), mask=valid)


@triton.jit
def _post_proj_transpose_nomask_no_og_kernel(
    proj_ptr,
    left_ptr,
    right_ptr,
    rows: tl.constexpr,
    n2: tl.constexpr,
    H: tl.constexpr,
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
    bidx = offs_r // n2
    ij = offs_r - bidx * n2
    bh = (bidx[:, None] * H + offs_h[None, :]) * n2 + ij[:, None]
    tl.store(left_ptr + bh, l * tl.sigmoid(lg), mask=valid)
    tl.store(right_ptr + bh, r * tl.sigmoid(rg), mask=valid)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    dim = int(config["dim"])
    hidden_dim = int(config["hidden_dim"])
    batch_size = input_tensor.shape[0]
    seq_len = input_tensor.shape[1]
    device = input_tensor.device
    f16 = torch.float16
    f32 = torch.float32
    norm_w = weights["norm.weight"]
    norm_b = weights["norm.bias"]
    out_norm_w = weights["to_out_norm.weight"]
    out_norm_b = weights["to_out_norm.bias"]
    to_out_w = weights["to_out.weight"]
    x_in = input_tensor.contiguous()
    proj_dtype = f16
    buf_shape_key = (device, batch_size, seq_len, dim, hidden_dim)
    nomask = bool(config.get("nomask", False)) or mask is None
    rows = batch_size * seq_len * seq_len
    n2 = seq_len * seq_len
    out_gate = None
    out_gate_pre_sigmoid = False
    mask_arg = input_tensor if mask is None else mask

    if dim == 128 and hidden_dim == 128:
        out_gate_pre_sigmoid = True
        out_gate = _buf(buf_shape_key, "out_gate", (batch_size, seq_len, seq_len, hidden_dim), f16, device)
        left_b = _buf(buf_shape_key, "left_b", (batch_size * hidden_dim, seq_len, seq_len), f16, device)
        right_b = _buf(buf_shape_key, "right_b", (batch_size * hidden_dim, seq_len, seq_len), f16, device)
        wl = _buf(buf_shape_key, "wl", (hidden_dim, dim), f16, device)
        wr = _buf(buf_shape_key, "wr", (hidden_dim, dim), f16, device)
        wlg = _buf(buf_shape_key, "wlg", (hidden_dim, dim), f16, device)
        wrg = _buf(buf_shape_key, "wrg", (hidden_dim, dim), f16, device)
        wog = _buf(buf_shape_key, "wog", (hidden_dim, dim), f16, device)
        wl.copy_(weights["left_proj.weight"])
        wr.copy_(weights["right_proj.weight"])
        wlg.copy_(weights["left_gate.weight"])
        wrg.copy_(weights["right_gate.weight"])
        wog.copy_(weights["out_gate.weight"])
        if seq_len <= 1024:
            _fused_proj128_4q_kernel[(triton.cdiv(rows, 64),)](
                x_in,
                mask_arg,
                norm_w.to(f32),
                norm_b.to(f32),
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
                BLOCK_M=64,
                BLOCK_H=32,
                num_warps=4,
                num_stages=1,
            )
        else:
            _fused_proj128_4q_kernel[(triton.cdiv(rows, 128),)](
                x_in,
                mask_arg,
                norm_w.to(f32),
                norm_b.to(f32),
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
                BLOCK_M=128,
                BLOCK_H=32,
                num_warps=8,
                num_stages=1,
            )
    else:
        x = _buf(buf_shape_key, "x_norm", input_tensor.shape, proj_dtype, device)
        in_rows = x.numel() // dim
        if dim == 384:
            if seq_len >= 1024:
                ln_br = 32
                ln_warps = 2
            elif seq_len >= 768:
                ln_br = 8
                ln_warps = 2
            elif seq_len == 256:
                ln_br = 2
                ln_warps = 1
            else:
                ln_br = 4 if seq_len >= 1024 else 1
                ln_warps = 1
            _input_ln384_kernel[(triton.cdiv(in_rows, ln_br),)](
                x_in,
                norm_w.to(f32),
                norm_b.to(f32),
                x,
                rows=in_rows,
                BLOCK_R=ln_br,
                num_warps=ln_warps,
            )
        elif dim == 768:
            ln_br = 8 if in_rows >= 524288 else 2
            _input_ln768_kernel[(triton.cdiv(in_rows, ln_br),)](
                x_in,
                norm_w.to(f32),
                norm_b.to(f32),
                x,
                rows=in_rows,
                BLOCK_R=ln_br,
                num_warps=1,
            )
        else:
            _input_ln_kernel[(in_rows,)](
                x_in,
                norm_w.to(f32),
                norm_b.to(f32),
                x,
                D=dim,
                BLOCK=triton.next_power_of_2(dim),
                num_warps=1,
            )

        left_b = _buf(buf_shape_key, "left_b", (batch_size * hidden_dim, seq_len, seq_len), f16, device)
        right_b = _buf(buf_shape_key, "right_b", (batch_size * hidden_dim, seq_len, seq_len), f16, device)
        proj_w = _buf(buf_shape_key, "proj_w", (5 * hidden_dim, dim), proj_dtype, device)
        torch.cat(
            (
                weights["left_proj.weight"],
                weights["right_proj.weight"],
                weights["left_gate.weight"],
                weights["right_gate.weight"],
                weights["out_gate.weight"],
            ),
            dim=0,
            out=proj_w,
        )
        proj = _buf(buf_shape_key, "proj", (batch_size, seq_len, seq_len, 5 * hidden_dim), proj_dtype, device)
        torch.mm(x.view(rows, dim), proj_w.t(), out=proj.view(rows, 5 * hidden_dim), out_dtype=proj_dtype)
        use_proj_gate = dim == 384 or dim == 768
        if not use_proj_gate:
            out_gate = _buf(buf_shape_key, "out_gate", (batch_size, seq_len, seq_len, hidden_dim), f16, device)
        if nomask and use_proj_gate:
            _post_proj_transpose_nomask_no_og_kernel[(triton.cdiv(rows, 32), triton.cdiv(hidden_dim, 64))](
                proj,
                left_b,
                right_b,
                rows=rows,
                n2=n2,
                H=hidden_dim,
                BLOCK_R=32,
                BLOCK_H=64,
                num_warps=8,
            )
        elif use_proj_gate:
            _post_proj_transpose_no_og_kernel[(triton.cdiv(rows, 32), triton.cdiv(hidden_dim, 64))](
                proj,
                mask_arg,
                left_b,
                right_b,
                rows=rows,
                n2=n2,
                H=hidden_dim,
                NOMASK=nomask,
                BLOCK_R=32,
                BLOCK_H=64,
                num_warps=8,
            )
        elif nomask:
            _post_proj_transpose_nomask_kernel[(triton.cdiv(rows, 16), triton.cdiv(hidden_dim, 64))](
                proj,
                left_b,
                right_b,
                out_gate,
                rows=rows,
                n2=n2,
                H=hidden_dim,
                BLOCK_R=16,
                BLOCK_H=64,
                num_warps=8,
            )
        else:
            _post_proj_transpose_kernel[(triton.cdiv(rows, 16), triton.cdiv(hidden_dim, 64))](
                proj,
                mask_arg,
                left_b,
                right_b,
                out_gate,
                rows=rows,
                n2=n2,
                H=hidden_dim,
                NOMASK=nomask,
                BLOCK_R=16,
                BLOCK_H=64,
                num_warps=8,
            )
    fast_out_proj = (dim == 128) or (dim == 384) or (dim == 768)
    to_out_h_t = None
    copy_stream = None
    if fast_out_proj:
        to_out_h_t = _buf(buf_shape_key, "to_out_h_t", (hidden_dim, dim), f16, device)
        copy_stream = _copy_stream(device)
        copy_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(copy_stream):
            to_out_h_t.copy_(to_out_w.t())

    out_h_flat = _buf(buf_shape_key, "out_h", (batch_size * hidden_dim, seq_len, seq_len), f16, device)
    torch.bmm(left_b, right_b.transpose(1, 2), out=out_h_flat)
    out_h = out_h_flat.view(batch_size, hidden_dim, seq_len, seq_len)
    out = _buf(buf_shape_key, "out", input_tensor.shape, f32, device)
    if fast_out_proj:
        if copy_stream is not None:
            torch.cuda.current_stream(device).wait_stream(copy_stream)
        br = 128 if dim == 384 or dim == 768 else 64
        if dim == 128:
            _ln_gate_out_pre_gate_kernel[(triton.cdiv(rows, br),)](
                out_h,
                out_gate,
                out_norm_w.to(f32),
                out_norm_b.to(f32),
                to_out_h_t,
                out,
                rows=rows,
                n2=n2,
                H=hidden_dim,
                D=dim,
                BLOCK_R=br,
                BLOCK_D=32,
                num_warps=4,
            )
        elif not (dim == 384 or dim == 768):
            _ln_gate_out_kernel[(triton.cdiv(rows, br),)](
                out_h,
                out_gate,
                out_norm_w.to(f32),
                out_norm_b.to(f32),
                to_out_h_t,
                out,
                rows=rows,
                n2=n2,
                H=hidden_dim,
                D=dim,
                BLOCK_R=br,
                BLOCK_D=64,
                num_warps=4,
            )
        else:
            _ln_gate_out_proj_gate_kernel[(triton.cdiv(rows, br),)](
                out_h,
                proj,
                out_norm_w.to(f32),
                out_norm_b.to(f32),
                to_out_h_t,
                out,
                rows=rows,
                n2=n2,
                H=hidden_dim,
                D=dim,
                BLOCK_R=br,
                BLOCK_D=64,
                num_warps=4,
            )
    else:
        gated = _buf(buf_shape_key, "gated", (batch_size, seq_len, seq_len, hidden_dim), f32, device)
        _ln_gate_tile_kernel[(triton.cdiv(rows, 64),)](
            out_h,
            out_gate,
            out_norm_w.to(f32),
            out_norm_b.to(f32),
            gated,
            rows=rows,
            n2=n2,
            H=hidden_dim,
            BLOCK_R=64,
            BLOCK_H=128,
            num_warps=4,
        )
        torch.mm(gated.view(rows, hidden_dim), to_out_w.to(f32).t(), out=out.view(rows, dim))
    return out
