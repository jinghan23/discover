#!/usr/bin/env python3
#!POPCORN leaderboard trimul
#!POPCORN gpu H100

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.libdevice import rsqrt as tl_rsqrt
from task import input_t, output_t

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ── Kernel 1: Fused LayerNorm + cast to fp16 ─────────────────────────────
@triton.jit
def ln_fused_cast(in_ptr, out_ptr, w_ptr, b_ptr, T, stride_dim: tl.constexpr,
                  dim: tl.constexpr, BD: tl.constexpr, BR: tl.constexpr = 1):
    pid = tl.program_id(0)
    c = tl.arange(0, BD); m = c < dim
    w = tl.load(w_ptr + c, mask=m, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + c, mask=m, other=0.0).to(tl.float32)
    for i in range(BR):
        row = pid * BR + i
        if row < T:
            x = tl.load(in_ptr + row * stride_dim + c, mask=m, other=0.0).to(tl.float32)
            mean = tl.sum(x, axis=0) / dim; xc = x - mean
            var = tl.sum(xc * xc, axis=0) / dim; rstd = tl_rsqrt(var + 1e-5)
            tl.store(out_ptr + row * stride_dim + c, ((xc * rstd) * w + b).to(tl.float16), mask=m)


# ── Kernel 2: Fused dual GEMM + sigmoid gate + mask + permuted store ─────
# Inspired by cuEquivariance's fused_sigmoid_gated_dual_gemm pattern.
# Computes: sigmoid(x @ W_gate.T) * (x @ W_proj.T) * mask
# Both GEMMs share input x tiles (loaded once, used for 2 accumulators).
# Writes output directly in permuted (B*hd, N, N) layout.
@triton.jit
def fused_dual_gemm_gate_permute(
    x_ptr, w_gate_ptr, w_proj_ptr, mask_ptr, out_ptr,
    T, N2, K, hd: tl.constexpr,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
    K_ITERS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    start_m = pid_m * TILE_M
    offs_m = start_m + tl.arange(0, TILE_M)
    offs_n = tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    mask_m = offs_m < T

    x_ptrs = x_ptr + (offs_m[:, None] * K + offs_k[None, :])
    w_tile_offs = offs_n[None, :] * K + offs_k[:, None]

    wg_base = w_gate_ptr + pid_n * TILE_N * K
    wp_base = w_proj_ptr + pid_n * TILE_N * K

    acc_gate = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    acc_proj = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    for ki in range(0, K_ITERS):
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
        wg = tl.load(wg_base + w_tile_offs)
        wp = tl.load(wp_base + w_tile_offs)
        acc_gate = tl.dot(x, wg, acc_gate)
        acc_proj = tl.dot(x, wp, acc_proj)
        x_ptrs += TILE_K
        wg_base += TILE_K
        wp_base += TILE_K

    gated = (1.0 / (1.0 + tl.exp(-acc_gate))) * acc_proj
    m_vals = tl.load(mask_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    gated = gated * m_vals[:, None]

    ys = offs_m % N2
    yb = offs_m // N2
    n_off = pid_n * TILE_N + tl.arange(0, TILE_N)
    perm = ys[:, None] + N2 * n_off[None, :] + (hd * N2) * yb[:, None]

    out_mask = mask_m[:, None] & (n_off[None, :] < hd)
    tl.store(out_ptr + perm, gated.to(tl.bfloat16), mask=out_mask)


# ── Kernel 3: Fused right dual GEMM + out_gate (sequential phases) ───────
# Phase 1: right dual GEMM → R in permuted (B*hd, N, N) layout
# Phase 2: out_gate single GEMM → sigmoid → og (reuses x tiles from L2)
@triton.jit
def fused_right_and_outgate(
    x_ptr, w_rgate_ptr, w_rproj_ptr, w_og_ptr, mask_ptr,
    R_ptr, og_ptr,
    T, N2, K, hd: tl.constexpr,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
    K_ITERS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    start_m = pid_m * TILE_M
    offs_m = start_m + tl.arange(0, TILE_M)
    offs_n = tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)
    mask_m = offs_m < T
    x_start = x_ptr + (offs_m[:, None] * K)
    w_tile_offs = offs_n[None, :] * K + offs_k[:, None]
    n_off = pid_n * TILE_N + tl.arange(0, TILE_N)
    out_mask = mask_m[:, None] & (n_off[None, :] < hd)

    # Phase 1: right dual GEMM
    x_ptrs = x_start + offs_k[None, :]
    wg_base = w_rgate_ptr + pid_n * TILE_N * K
    wp_base = w_rproj_ptr + pid_n * TILE_N * K
    acc_gate = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    acc_proj = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for ki in range(0, K_ITERS):
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
        wg = tl.load(wg_base + w_tile_offs)
        wp = tl.load(wp_base + w_tile_offs)
        acc_gate = tl.dot(x, wg, acc_gate)
        acc_proj = tl.dot(x, wp, acc_proj)
        x_ptrs += TILE_K
        wg_base += TILE_K
        wp_base += TILE_K
    gated = (1.0 / (1.0 + tl.exp(-acc_gate))) * acc_proj
    m_vals = tl.load(mask_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    gated = gated * m_vals[:, None]
    ys = offs_m % N2
    yb = offs_m // N2
    perm = ys[:, None] + N2 * n_off[None, :] + (hd * N2) * yb[:, None]
    tl.store(R_ptr + perm, gated.to(tl.bfloat16), mask=out_mask)

    # Phase 2: out_gate GEMM + sigmoid (x tiles reused from L2)
    x_ptrs2 = x_start + offs_k[None, :]
    wo_base = w_og_ptr + pid_n * TILE_N * K
    acc_og = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for ki in range(0, K_ITERS):
        x2 = tl.load(x_ptrs2, mask=mask_m[:, None], other=0.0)
        wo = tl.load(wo_base + w_tile_offs)
        acc_og = tl.dot(x2, wo, acc_og)
        x_ptrs2 += TILE_K
        wo_base += TILE_K
    og_result = tl.sigmoid(acc_og)
    tl.store(og_ptr + offs_m[:, None] * hd + n_off[None, :],
             og_result.to(tl.float16), mask=out_mask)


# ── Kernel 4: Fused post-LN + gate + output projection ──────────────────
@triton.jit
def fused_postln_gate_proj(bmm_ptr, og_ptr, ln_w_ptr, ln_b_ptr, proj_w_ptr, out_ptr,
                           B, N2, T, hd: tl.constexpr, dim, eps: tl.constexpr,
                           TI: tl.constexpr, DB: tl.constexpr):
    pb = tl.program_id(0); pi = tl.program_id(1)
    ij = pi*TI + tl.arange(0, TI); d = tl.arange(0, hd); ij_ok = ij < N2
    x = tl.load(bmm_ptr + (pb*hd + d[None,:])*N2 + ij[:,None], mask=ij_ok[:,None]).to(tl.float32)
    mu = tl.sum(x, axis=1)/hd; xc = x - mu[:,None]
    xn = xc * tl_rsqrt(tl.sum(xc*xc, axis=1)[:,None]/hd + eps)
    x_ln = xn * tl.load(ln_w_ptr+d)[None,:] + tl.load(ln_b_ptr+d)[None,:]
    t_off = pb*N2 + ij
    og_val = tl.load(og_ptr + t_off[:,None]*hd + d[None,:], mask=ij_ok[:,None]).to(tl.float32)
    gated = (x_ln * og_val).to(tl.float16)
    d_off = 0
    while d_off < dim:
        od = d_off + tl.arange(0, DB); od_mask = od < dim
        w = tl.load(proj_w_ptr + od[None,:]*hd + d[:,None], mask=od_mask[None,:]).to(tl.float16)
        res = tl.dot(gated, w)
        tl.store(out_ptr + t_off[:,None]*dim + od[None,:], res.to(tl.float16), mask=ij_ok[:,None] & od_mask[None,:])
        d_off += DB


# ── Main pipeline ─────────────────────────────────────────────────────────
def custom_kernel(data: input_t) -> output_t:
    x_in, mask, W, cfg = data
    dim, hd = cfg['dim'], cfg['hidden_dim']
    B, N = x_in.shape[0], x_in.shape[1]
    T = B * N * N; N2 = N * N; BD = triton.next_power_of_2(dim)
    x_flat = x_in.reshape(T, dim)

    # 1) Fused LN → fp16
    x_ln = torch.empty(T, dim, device=x_in.device, dtype=torch.float16)
    if dim <= 128:
        ln_BR, ln_warps = 4, 1
    elif dim <= 384:
        ln_BR, ln_warps = 1, 2
    else:
        ln_BR, ln_warps = 1, 2
    ln_fused_cast[(triton.cdiv(T, ln_BR),)](x_flat, x_ln, W['norm.weight'], W['norm.bias'],
                        T, stride_dim=dim, dim=dim, BD=BD, BR=ln_BR, num_warps=ln_warps)

    # Shape-adaptive GEMM configs (autotuned per dim)
    if dim <= 128:
        TILE_M, TILE_K, gemm_warps, gemm_stages = 64, 32, 4, 3
        rog_TM, rog_TK, rog_warps, rog_stages = 64, 64, 8, 2
    else:
        TILE_M, TILE_K, gemm_warps, gemm_stages = 128, 64, 8, 3
        rog_TM, rog_TK, rog_warps, rog_stages = 128, 64, 8, 3
    TILE_N = 128
    DB = 128

    # 2) Fused dual GEMM: sigmoid(x @ W_gate.T) * (x @ W_proj.T) * mask → permuted output
    mask_flat = mask.reshape(T).float()
    L = torch.empty(B*hd, N, N, device=x_in.device, dtype=torch.bfloat16)
    R = torch.empty(B*hd, N, N, device=x_in.device, dtype=torch.bfloat16)

    K_ITERS = triton.cdiv(dim, TILE_K)
    grid = (triton.cdiv(T, TILE_M), triton.cdiv(hd, TILE_N))

    left_gate_w = W['left_gate.weight'].contiguous().half()
    left_proj_w = W['left_proj.weight'].contiguous().half()
    right_gate_w = W['right_gate.weight'].contiguous().half()
    right_proj_w = W['right_proj.weight'].contiguous().half()
    out_gate_w = W['out_gate.weight'].contiguous().half()
    out_gate = torch.empty(T, hd, device=x_in.device, dtype=torch.float16)

    fused_dual_gemm_gate_permute[grid](
        x_ln, left_gate_w, left_proj_w, mask_flat, L,
        T, N2, dim, hd=hd, TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K,
        K_ITERS=K_ITERS, num_warps=gemm_warps, num_stages=gemm_stages)

    # 3) Fused right dual GEMM + out_gate (sequential phases, L2 x reuse)
    rog_K_ITERS = triton.cdiv(dim, rog_TK)
    rog_grid = (triton.cdiv(T, rog_TM), triton.cdiv(hd, TILE_N))
    fused_right_and_outgate[rog_grid](
        x_ln, right_gate_w, right_proj_w, out_gate_w, mask_flat, R, out_gate,
        T, N2, dim, hd=hd, TILE_M=rog_TM, TILE_N=TILE_N, TILE_K=rog_TK,
        K_ITERS=rog_K_ITERS, num_warps=rog_warps, num_stages=rog_stages)
    del x_ln

    # 4) BMM (cuBLAS bf16)
    out_bmm = torch.bmm(L, R.transpose(-1, -2))
    del L, R

    # 5+6) Fused post-LN + gate + output projection
    out_w = W['to_out.weight'].contiguous().half()
    result = torch.empty(T, dim, device=x_in.device, dtype=torch.float16)
    fused_postln_gate_proj[(B, triton.cdiv(N2, 64))](
        out_bmm.reshape(B*hd, N2), out_gate,
        W['to_out_norm.weight'], W['to_out_norm.bias'], out_w, result,
        B, N2, T, hd=hd, dim=dim, eps=1e-5, TI=64, DB=DB, num_warps=4)
    del out_bmm, out_gate
    return result.view(B, N, N, dim)
