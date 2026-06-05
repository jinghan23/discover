"""TriMul forward — fp16 fused-Triton implementation.

Architecture:
  1. LN1 (fp32 reduction, fp16 output) over the last dim of x.
  2. Fused projection: 5 GEMMs (left/right/lg/rg/og) + sigmoid + mask + permute
     into (B, H, N, N) fp16 left/right and (B, N, N, H) fp16 og — single kernel.
  3. cuBLAS bmm in fp16 (fp32 accumulate) for the contraction.
  4. Fused back: LN over H + out-gate mul + final linear (B,N,N,dim) — single kernel.

fp16 has a 10-bit mantissa (vs bf16's 7) which is enough to pass tolerance even on
heavy-tailed distributions, unlike bf16 which forced the slower fp32 lr-projection
split for precision.
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t


# -----------------------------------------------------------------------------
# 1) Row-wise LayerNorm over last dim → fp16 output.
# -----------------------------------------------------------------------------
@triton.jit
def _ln1_kernel(
    X_ptr, Y_ptr,
    w_ptr, b_ptr,
    M, C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M

    sm = tl.zeros([BLOCK_M], dtype=tl.float32)
    sm2 = tl.zeros([BLOCK_M], dtype=tl.float32)
    for c0 in range(0, C, BLOCK_C):
        cur_c = c0 + tl.arange(0, BLOCK_C)
        cmask = cur_c < C
        x = tl.load(X_ptr + rows[:, None] * C + cur_c[None, :],
                    mask=row_mask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        sm += tl.sum(x, axis=1)
        sm2 += tl.sum(x * x, axis=1)
    mean = sm / C
    var = sm2 / C - mean * mean
    inv_std = 1.0 / tl.sqrt(var + 1e-5)

    for c0 in range(0, C, BLOCK_C):
        cur_c = c0 + tl.arange(0, BLOCK_C)
        cmask = cur_c < C
        x = tl.load(X_ptr + rows[:, None] * C + cur_c[None, :],
                    mask=row_mask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cur_c, mask=cmask, other=0.0)
        b = tl.load(b_ptr + cur_c, mask=cmask, other=0.0)
        y = (x - mean[:, None]) * inv_std[:, None] * w[None, :] + b[None, :]
        tl.store(Y_ptr + rows[:, None] * C + cur_c[None, :],
                 y.to(tl.float16),
                 mask=row_mask[:, None] & cmask[None, :])


# -----------------------------------------------------------------------------
# 2) Fused projection: 5 fp16 GEMMs + sigmoid + mask + permute writes.
#    Inputs:  x_norm (M, C) fp16, w_l/r/lg/rg/og (H, C) fp16 row-major (PyTorch native)
#    Outputs: left, right (B, H, N, N) fp16 ; out_gate (B, N, N, H) fp16
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_H": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_H": 64, "BLOCK_K": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_H": 64, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_H": 64, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_H": 128, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_H": 128, "BLOCK_K": 32}, num_warps=8),
    ],
    key=["C", "H"],
)
@triton.jit
def _proj_kernel(
    x_ptr, mask_ptr,
    w_l_ptr, w_r_ptr, w_lg_ptr, w_rg_ptr, w_og_ptr,
    left_ptr, right_ptr, og_ptr,
    M, N,
    C: tl.constexpr, H: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    hids = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    row_mask = rows < M
    hid_mask = hids < H

    if HAS_MASK:
        mval = tl.load(mask_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    else:
        mval = tl.full([BLOCK_M], 1.0, dtype=tl.float32)

    acc_l  = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    acc_r  = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    acc_lg = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    acc_rg = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    acc_og = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

    for k in range(0, C, BLOCK_K):
        cur_k = k + tl.arange(0, BLOCK_K)
        kmask = cur_k < C
        a = tl.load(x_ptr + rows[:, None] * C + cur_k[None, :],
                    mask=row_mask[:, None] & kmask[None, :], other=0.0)
        # Weights come in PyTorch row-major (H, C) layout — no python-level transpose.
        # Read tile (BLOCK_H, BLOCK_K) and use tl.trans inside the dot.
        w_off = hids[:, None] * C + cur_k[None, :]
        w_kmask = hid_mask[:, None] & kmask[None, :]
        wl  = tl.load(w_l_ptr  + w_off, mask=w_kmask, other=0.0)
        wr  = tl.load(w_r_ptr  + w_off, mask=w_kmask, other=0.0)
        wlg = tl.load(w_lg_ptr + w_off, mask=w_kmask, other=0.0)
        wrg = tl.load(w_rg_ptr + w_off, mask=w_kmask, other=0.0)
        wog = tl.load(w_og_ptr + w_off, mask=w_kmask, other=0.0)
        acc_l  += tl.dot(a, tl.trans(wl))
        acc_r  += tl.dot(a, tl.trans(wr))
        acc_lg += tl.dot(a, tl.trans(wlg))
        acc_rg += tl.dot(a, tl.trans(wrg))
        acc_og += tl.dot(a, tl.trans(wog))

    lg = tl.sigmoid(acc_lg)
    rg = tl.sigmoid(acc_rg)
    og = tl.sigmoid(acc_og)

    left_out  = (acc_l * lg) * mval[:, None]
    right_out = (acc_r * rg) * mval[:, None]

    # rows is a flat index into (B, N, N). Compute (b, i, j).
    NN = N * N
    b_idx = rows // NN
    rem = rows - b_idx * NN
    i_idx = rem // N
    j_idx = rem - i_idx * N

    # left[b, h, i, j] addr = ((b*H+h)*N + i)*N + j   →  b*H*NN + h*NN + i*N + j
    lr_off = (b_idx[:, None] * H + hids[None, :]) * NN + i_idx[:, None] * N + j_idx[:, None]
    lr_mask = row_mask[:, None] & hid_mask[None, :]
    tl.store(left_ptr  + lr_off, left_out.to(tl.float16),  mask=lr_mask)
    tl.store(right_ptr + lr_off, right_out.to(tl.float16), mask=lr_mask)

    # og[b, i, j, h] is contiguous (B, N, N, H) — flat index = rows * H + h.
    og_off = rows[:, None] * H + hids[None, :]
    tl.store(og_ptr + og_off, og.to(tl.float16), mask=lr_mask)


# -----------------------------------------------------------------------------
# 3) Fused back: read bmm output (B,H,N,N) fp16 → permute + LN over H +
#    multiply by og (sigmoided) + final linear (H, dim) → (B, N, N, dim) fp32.
# -----------------------------------------------------------------------------
@triton.jit
def _back_kernel(
    bmm_ptr,     # (B*H*N*N,) fp16 — (B, H, N, N) layout
    og_ptr,      # (B*N*N*H,) fp16 — (B, N, N, H) layout
    ln_w_ptr, ln_b_ptr,    # (H,) fp32
    w_out_ptr,             # (H, D) fp16
    out_ptr,               # (B, N, N, D) fp32
    B, N,
    H: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < (B * N * N)

    NN = N * N
    b_idx = rows // NN
    rem = rows - b_idx * NN
    i_idx = rem // N
    j_idx = rem - i_idx * N

    # H is the full hidden dim (always 128 in this competition). Load all H per row.
    hids = tl.arange(0, H)

    # Read bmm[b, h, i, j] - strided in h (stride NN).
    h_off = (b_idx[:, None] * H + hids[None, :]) * NN + i_idx[:, None] * N + j_idx[:, None]
    htile = tl.load(bmm_ptr + h_off, mask=row_mask[:, None], other=0.0).to(tl.float32)

    # LN over H.
    s = tl.sum(htile, axis=1)
    s2 = tl.sum(htile * htile, axis=1)
    mean = s / H
    var = s2 / H - mean * mean
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    w = tl.load(ln_w_ptr + hids)
    b = tl.load(ln_b_ptr + hids)
    normed = (htile - mean[:, None]) * inv_std[:, None] * w[None, :] + b[None, :]

    # Read og (B, N, N, H) contiguous.
    og = tl.load(og_ptr + rows[:, None] * H + hids[None, :],
                 mask=row_mask[:, None], other=0.0).to(tl.float32)
    gated = (normed * og).to(tl.float16)

    # Final linear: (BLOCK_M, H) @ (H, D) → (BLOCK_M, D), looped over D in BLOCK_D chunks.
    for d0 in range(0, D, BLOCK_D):
        cols = d0 + tl.arange(0, BLOCK_D)
        col_mask = cols < D
        wo = tl.load(w_out_ptr + hids[:, None] * D + cols[None, :],
                     mask=col_mask[None, :], other=0.0)
        out = tl.dot(gated, wo)  # fp16 inputs, fp32 acc
        tl.store(out_ptr + rows[:, None] * D + cols[None, :], out,
                 mask=row_mask[:, None] & col_mask[None, :])


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def custom_kernel(data: input_t) -> output_t:
    inp, mask, weights, cfg = data
    dim = int(cfg["dim"])
    H = int(cfg["hidden_dim"])

    device = inp.device
    B, N, _, _ = inp.shape
    M = B * N * N

    # 1) LN1 → fp16
    x_flat = inp.reshape(M, dim).contiguous()
    x_norm = torch.empty((M, dim), dtype=torch.float16, device=device)

    BLOCK_M_LN = 128
    BLOCK_C_LN = 128
    grid_ln = (triton.cdiv(M, BLOCK_M_LN),)
    _ln1_kernel[grid_ln](
        x_flat, x_norm,
        weights["norm.weight"], weights["norm.bias"],
        M, dim,
        BLOCK_M=BLOCK_M_LN, BLOCK_C=BLOCK_C_LN, num_warps=8,
    )

    # 2) Project weights → fp16, row-major (H, C) — no transpose needed (kernel uses tl.trans).
    wl  = weights["left_proj.weight"].to(torch.float16)
    wr  = weights["right_proj.weight"].to(torch.float16)
    wlg = weights["left_gate.weight"].to(torch.float16)
    wrg = weights["right_gate.weight"].to(torch.float16)
    wog = weights["out_gate.weight"].to(torch.float16)

    # Always pass mask in fp16. nomask cases just have mask=ones — multiply by 1
    # is free; skipping the read saves only a small amount of bandwidth and a
    # CUDA sync would cost more than that.
    if mask.dtype != torch.float16:
        mask_flat = mask.reshape(M).to(torch.float16).contiguous()
    else:
        mask_flat = mask.reshape(M).contiguous()

    left = torch.empty((B, H, N, N), dtype=torch.float16, device=device)
    right = torch.empty_like(left)
    og = torch.empty((B, N, N, H), dtype=torch.float16, device=device)

    grid_pj = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(H, meta["BLOCK_H"]))
    _proj_kernel[grid_pj](
        x_norm, mask_flat,
        wl, wr, wlg, wrg, wog,
        left, right, og,
        M, N, dim, H,
        HAS_MASK=True,
    )

    # 3) bmm in fp16 (cuBLAS, fp32 accum)
    L = left.view(B * H, N, N)
    R = right.view(B * H, N, N).transpose(1, 2)
    bmm_out = torch.bmm(L, R)  # (B*H, N, N) fp16
    bmm_out = bmm_out.view(B, H, N, N)

    # 4) Fused back + final linear.
    # to_out.weight is (dim, H) row-major. We want to compute gated @ W_out where
    # gated is (M, H) and we want output (M, dim). Need W_out as (H, dim).
    w_out_T = weights["to_out.weight"].t().contiguous().to(torch.float16)  # (H, dim) fp16
    out = torch.empty((B, N, N, dim), dtype=torch.float32, device=device)

    BLOCK_M_BK = 64
    BLOCK_D_BK = 64
    grid_bk = (triton.cdiv(M, BLOCK_M_BK),)
    _back_kernel[grid_bk](
        bmm_out.view(-1), og.view(-1),
        weights["to_out_norm.weight"], weights["to_out_norm.bias"],
        w_out_T, out,
        B, N, H, dim,
        BLOCK_M=BLOCK_M_BK, BLOCK_D=BLOCK_D_BK, num_warps=4,
    )
    return out
