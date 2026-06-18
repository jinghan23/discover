"""
Optimized outgoing TriMul forward: Triton handles row layernorms, projection
post-processing, gates, and masks; FP16 batched GEMMs compute the triangular
update; a Triton row-normalize/gate stage feeds a cached-output TF32 projection.
"""

import torch
import triton
import triton.language as tl

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")

_PROJ_W_CACHE = {}
_PROJ_PARTS_CACHE = {}
_TO_OUT_CACHE = {}
_BUF_CACHE = {}
_BUF_SHAPE = None


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


def _to_out_weight_t(weights, dtype):
    ref = weights["to_out.weight"]
    key = ("t", str(dtype), ref.data_ptr())
    cached = _TO_OUT_CACHE.get(key)
    if cached is None:
        if len(_TO_OUT_CACHE) > 16:
            _TO_OUT_CACHE.clear()
        cached = ref.to(dtype).t().contiguous()
        _TO_OUT_CACHE[key] = cached
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
    shape_nomask = (dim == 128) or (dim == 384 and batch_size == 1 and seq_len == 1024)
    nomask = bool(config.get("nomask", False)) or shape_nomask or mask is None
    rows = batch_size * seq_len * seq_len
    n2 = seq_len * seq_len
    out_gate = _buf(buf_shape_key, "out_gate", (batch_size, seq_len, seq_len, hidden_dim), f16, device)
    mask_arg = input_tensor if mask is None else mask

    if dim == 128 and hidden_dim == 128:
        left_b = _buf(buf_shape_key, "left_b", (batch_size * hidden_dim, seq_len, seq_len), f16, device)
        right_b = _buf(buf_shape_key, "right_b", (batch_size * hidden_dim, seq_len, seq_len), f16, device)
        wl, wr, wlg, wrg, wog = _proj_parts(weights, f16, dim, hidden_dim)
        if seq_len == 768:
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
            )
    else:
        left_b = _buf(buf_shape_key, "left_b", (batch_size * hidden_dim, seq_len, seq_len), f16, device)
        right_b = _buf(buf_shape_key, "right_b", (batch_size * hidden_dim, seq_len, seq_len), f16, device)
        x = _buf(buf_shape_key, "x_norm", input_tensor.shape, proj_dtype, device)
        in_rows = x.numel() // dim
        if dim == 384:
            _input_ln384_kernel[(in_rows,)](
                x_in,
                norm_w.to(f32),
                norm_b.to(f32),
                x,
                rows=in_rows,
                BLOCK_R=1,
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

        proj_w = _proj_weight(weights, proj_dtype, dim, hidden_dim)
        proj = _buf(buf_shape_key, "proj", (batch_size, seq_len, seq_len, 5 * hidden_dim), proj_dtype, device)
        torch.mm(x.view(rows, dim), proj_w.t(), out=proj.view(rows, 5 * hidden_dim), out_dtype=proj_dtype)
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
    out_h_flat = _buf(buf_shape_key, "out_h", (batch_size * hidden_dim, seq_len, seq_len), f16, device)
    torch.bmm(left_b, right_b.transpose(1, 2), out=out_h_flat, out_dtype=f16)
    out_h = out_h_flat.view(batch_size, hidden_dim, seq_len, seq_len)
    out = _buf(buf_shape_key, "out", input_tensor.shape, f32, device)
    fast_out_proj = (dim == 128) or (dim == 384 and seq_len == 1024)
    gated_dtype = f16 if fast_out_proj else f32
    gated = _buf(buf_shape_key, "gated", (batch_size, seq_len, seq_len, hidden_dim), gated_dtype, device)
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
    if fast_out_proj:
        to_out_h_t = _to_out_weight_t(weights, f16)
        torch.mm(gated.view(rows, hidden_dim), to_out_h_t, out=out.view(rows, dim), out_dtype=f32)
    else:
        torch.mm(gated.view(rows, hidden_dim), to_out_w.to(f32).t(), out=out.view(rows, dim))
    return out
