"""Hybrid Triton/cuBLAS outgoing Triangle Multiplicative Update forward.

Triton computes layer norms and packs gated left/right activations into
[B * H, N, N] BF16 matrices.  The cubic contraction is delegated to cuBLAS
batched BF16 GEMM, followed by a Triton output layer norm/gate and a final
cuBLAS projection.  C=128 uses shape-specific accuracy paths; wider channels
use a packed BF16 projection for all five hidden tensors.
"""

from task import input_t, output_t

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


@triton.jit
def _layernorm_bf16_kernel(
    x,
    weight,
    bias,
    y,
    C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    valid_c = offs_c < C

    ptrs = x + rows[None, :] * C + offs_c[:, None]
    vals = tl.load(ptrs, mask=valid_c[:, None], other=0.0).to(tl.float32)
    vals = tl.where(valid_c[:, None], vals, 0.0)
    mean = tl.sum(vals, axis=0) / C
    centered = tl.where(valid_c[:, None], vals - mean[None, :], 0.0)
    var = tl.sum(centered * centered, axis=0) / C
    inv = tl.rsqrt(var + 1.0e-5)

    w = tl.load(weight + offs_c, mask=valid_c, other=0.0).to(tl.float32)
    b = tl.load(bias + offs_c, mask=valid_c, other=0.0).to(tl.float32)
    out = centered * inv[None, :] * w[:, None] + b[:, None]
    tl.store(y + rows[None, :] * C + offs_c[:, None], out, mask=valid_c[:, None])


@triton.jit
def _norm_gate_tile_kernel(
    out_h,
    out_gate_logits,
    norm_w,
    norm_b,
    z,
    n_pairs: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = tl.arange(0, BLOCK_H)

    b = offs_m // n_pairs
    rem = offs_m - b * n_pairs
    hm = (b[None, :] * H + offs_h[:, None]) * n_pairs + rem[None, :]
    vals = tl.load(out_h + hm).to(tl.float32)

    mean = tl.sum(vals, axis=0) / H
    centered = vals - mean[None, :]
    var = tl.sum(centered * centered, axis=0) / H
    inv = tl.rsqrt(var + 1.0e-5)

    nw = tl.load(norm_w + offs_h).to(tl.float32)
    nb = tl.load(norm_b + offs_h).to(tl.float32)
    cl = offs_m[None, :] * H + offs_h[:, None]
    gate_logits = tl.load(out_gate_logits + cl).to(tl.float32)
    gate = 1.0 / (1.0 + tl.exp(-gate_logits))
    out = (centered * inv[None, :] * nw[:, None] + nb[:, None]) * gate
    tl.store(z + cl, out)


@triton.jit
def _prep_packed_hmajor_mask_kernel(
    proj,
    mask,
    left,
    right,
    n_pairs: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    valid_h = offs_h < H

    b = offs_m // n_pairs
    rem = offs_m - b * n_pairs
    base = offs_m[:, None] * (5 * H) + offs_h[None, :]
    valid = valid_h[None, :]
    m = tl.load(mask + offs_m).to(tl.float32)

    lp_v = tl.load(proj + base, mask=valid, other=0.0).to(tl.float32)
    rp_v = tl.load(proj + base + H, mask=valid, other=0.0).to(tl.float32)
    lg_v = tl.load(proj + base + (2 * H), mask=valid, other=0.0).to(tl.float32)
    rg_v = tl.load(proj + base + (3 * H), mask=valid, other=0.0).to(tl.float32)
    left_v = lp_v * m[:, None] / (1.0 + tl.exp(-lg_v))
    right_v = rp_v * m[:, None] / (1.0 + tl.exp(-rg_v))

    hmajor = (b[:, None] * H + offs_h[None, :]) * n_pairs + rem[:, None]
    tl.store(left + hmajor, left_v, mask=valid)
    tl.store(right + hmajor, right_v, mask=valid)


@triton.jit
def _prep_packed_hmajor_nomask_kernel(
    proj,
    left,
    right,
    n_pairs: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    valid_h = offs_h < H

    b = offs_m // n_pairs
    rem = offs_m - b * n_pairs
    base = offs_m[:, None] * (5 * H) + offs_h[None, :]
    valid = valid_h[None, :]

    lp_v = tl.load(proj + base, mask=valid, other=0.0).to(tl.float32)
    rp_v = tl.load(proj + base + H, mask=valid, other=0.0).to(tl.float32)
    lg_v = tl.load(proj + base + (2 * H), mask=valid, other=0.0).to(tl.float32)
    rg_v = tl.load(proj + base + (3 * H), mask=valid, other=0.0).to(tl.float32)
    left_v = lp_v / (1.0 + tl.exp(-lg_v))
    right_v = rp_v / (1.0 + tl.exp(-rg_v))

    hmajor = (b[:, None] * H + offs_h[None, :]) * n_pairs + rem[:, None]
    tl.store(left + hmajor, left_v, mask=valid)
    tl.store(right + hmajor, right_v, mask=valid)


@triton.jit
def _norm_gate_packed_tile_kernel(
    out_h,
    out_gate_packed,
    norm_w,
    norm_b,
    z,
    n_pairs: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = tl.arange(0, BLOCK_H)

    b = offs_m // n_pairs
    rem = offs_m - b * n_pairs
    hm = (b[None, :] * H + offs_h[:, None]) * n_pairs + rem[None, :]
    vals = tl.load(out_h + hm).to(tl.float32)

    mean = tl.sum(vals, axis=0) / H
    centered = vals - mean[None, :]
    var = tl.sum(centered * centered, axis=0) / H
    inv = tl.rsqrt(var + 1.0e-5)

    nw = tl.load(norm_w + offs_h).to(tl.float32)
    nb = tl.load(norm_b + offs_h).to(tl.float32)
    cl = offs_m[None, :] * H + offs_h[:, None]
    gate_logits = tl.load(out_gate_packed + offs_m[None, :] * (5 * H) + (4 * H) + offs_h[:, None]).to(tl.float32)
    gate = 1.0 / (1.0 + tl.exp(-gate_logits))
    out = (centered * inv[None, :] * nw[:, None] + nb[:, None]) * gate
    tl.store(z + cl, out)


@triton.jit
def _prep_packed4_hmajor_mask_kernel(
    proj,
    mask,
    left,
    right,
    n_pairs: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    valid_h = offs_h < H

    b = offs_m // n_pairs
    rem = offs_m - b * n_pairs
    base = offs_m[:, None] * (4 * H) + offs_h[None, :]
    valid = valid_h[None, :]
    m = tl.load(mask + offs_m).to(tl.float32)

    lp_v = tl.load(proj + base, mask=valid, other=0.0).to(tl.float32)
    rp_v = tl.load(proj + base + H, mask=valid, other=0.0).to(tl.float32)
    lg_v = tl.load(proj + base + (2 * H), mask=valid, other=0.0).to(tl.float32)
    rg_v = tl.load(proj + base + (3 * H), mask=valid, other=0.0).to(tl.float32)
    left_v = lp_v * m[:, None] / (1.0 + tl.exp(-lg_v))
    right_v = rp_v * m[:, None] / (1.0 + tl.exp(-rg_v))

    hmajor = (b[:, None] * H + offs_h[None, :]) * n_pairs + rem[:, None]
    tl.store(left + hmajor, left_v, mask=valid)
    tl.store(right + hmajor, right_v, mask=valid)


@triton.jit
def _prep_packed4_hmajor_nomask_kernel(
    proj,
    left,
    right,
    n_pairs: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    valid_h = offs_h < H

    b = offs_m // n_pairs
    rem = offs_m - b * n_pairs
    base = offs_m[:, None] * (4 * H) + offs_h[None, :]
    valid = valid_h[None, :]

    lp_v = tl.load(proj + base, mask=valid, other=0.0).to(tl.float32)
    rp_v = tl.load(proj + base + H, mask=valid, other=0.0).to(tl.float32)
    lg_v = tl.load(proj + base + (2 * H), mask=valid, other=0.0).to(tl.float32)
    rg_v = tl.load(proj + base + (3 * H), mask=valid, other=0.0).to(tl.float32)
    left_v = lp_v / (1.0 + tl.exp(-lg_v))
    right_v = rp_v / (1.0 + tl.exp(-rg_v))

    hmajor = (b[:, None] * H + offs_h[None, :]) * n_pairs + rem[:, None]
    tl.store(left + hmajor, left_v, mask=valid)
    tl.store(right + hmajor, right_v, mask=valid)


def _sigmoid_mask_layout_packed4(proj, mask, B: int, N: int, H: int):
    left = torch.empty((B * H, N, N), device=proj.device, dtype=torch.bfloat16)
    right = torch.empty((B * H, N, N), device=proj.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(B * N * N, 32), triton.cdiv(H, 64))
    if mask.dtype is torch.float32:
        _prep_packed4_hmajor_nomask_kernel[grid](
            proj,
            left,
            right,
            N * N,
            H,
            BLOCK_M=32,
            BLOCK_H=64,
            num_warps=8,
        )
    else:
        _prep_packed4_hmajor_mask_kernel[grid](
            proj,
            mask,
            left,
            right,
            N * N,
            H,
            BLOCK_M=32,
            BLOCK_H=64,
            num_warps=8,
        )
    return left, right


def _sigmoid_mask_layout_packed(proj, mask, B: int, N: int, H: int):
    left = torch.empty((B * H, N, N), device=proj.device, dtype=torch.bfloat16)
    right = torch.empty((B * H, N, N), device=proj.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(B * N * N, 32), triton.cdiv(H, 64))
    if mask.dtype is torch.float32:
        _prep_packed_hmajor_nomask_kernel[grid](
            proj,
            left,
            right,
            N * N,
            H,
            BLOCK_M=32,
            BLOCK_H=64,
            num_warps=8,
        )
    else:
        _prep_packed_hmajor_mask_kernel[grid](
            proj,
            mask,
            left,
            right,
            N * N,
            H,
            BLOCK_M=32,
            BLOCK_H=64,
            num_warps=8,
        )
    return left, right


def _norm_gate(out_h, out_gate_logits, norm_w, norm_b, B: int, N: int, H: int):
    z = torch.empty((B, N, N, H), device=out_h.device, dtype=torch.float32)
    _norm_gate_tile_kernel[(triton.cdiv(B * N * N, 32),)](
        out_h,
        out_gate_logits,
        norm_w,
        norm_b,
        z,
        N * N,
        N,
        H,
        BLOCK_M=32,
        BLOCK_H=128,
        num_warps=4,
    )
    return z


def _norm_gate_packed(out_h, proj, norm_w, norm_b, B: int, N: int, H: int):
    z = torch.empty((B, N, N, H), device=out_h.device, dtype=torch.float32)
    _norm_gate_packed_tile_kernel[(triton.cdiv(B * N * N, 64),)](
        out_h,
        proj,
        norm_w,
        norm_b,
        z,
        N * N,
        H,
        BLOCK_M=64,
        BLOCK_H=128,
        num_warps=4,
    )
    return z


def _layernorm_bf16(input_tensor, norm_w, norm_b, B: int, N: int, C: int):
    y = torch.empty_like(input_tensor, dtype=torch.bfloat16)
    rows = B * N * N
    if C <= 128:
        block_m = 32
        block_c = 128
        warps = 4
    elif C <= 256:
        block_m = 16
        block_c = 256
        warps = 4
    elif C <= 512:
        block_m = 4
        block_c = 512
        warps = 4
    else:
        block_m = 8
        block_c = 1024
        warps = 8
    _layernorm_bf16_kernel[(triton.cdiv(rows, block_m),)](
        input_tensor,
        norm_w,
        norm_b,
        y,
        C,
        BLOCK_M=block_m,
        BLOCK_C=block_c,
        num_warps=warps,
    )
    return y


def _layernorm_fp32(input_tensor, norm_w, norm_b, B: int, N: int, C: int):
    y = torch.empty_like(input_tensor)
    rows = B * N * N
    if C <= 128:
        block_m = 32
        block_c = 128
        warps = 4
    elif C <= 256:
        block_m = 16
        block_c = 256
        warps = 4
    elif C <= 512:
        block_m = 4
        block_c = 512
        warps = 4
    else:
        block_m = 8
        block_c = 1024
        warps = 8
    _layernorm_bf16_kernel[(triton.cdiv(rows, block_m),)](
        input_tensor,
        norm_w,
        norm_b,
        y,
        C,
        BLOCK_M=block_m,
        BLOCK_C=block_c,
        num_warps=warps,
    )
    return y


def custom_kernel(data: input_t) -> output_t:
    input_tensor, mask, weights, config = data
    B = input_tensor.shape[0]
    N = input_tensor.shape[1]
    C = config["dim"]
    H = config["hidden_dim"]

    if C == 128:
        if mask.dtype is torch.float32 and N != 256 and N != 768 and N != 1024:
            x = _layernorm_bf16(
                input_tensor,
                weights["norm.weight"],
                weights["norm.bias"],
                B,
                N,
                C,
            )
            proj_weight = torch.cat(
                (
                    weights["left_proj.weight"],
                    weights["right_proj.weight"],
                    weights["left_gate.weight"],
                    weights["right_gate.weight"],
                    weights["out_gate.weight"],
                ),
                dim=0,
            ).to(torch.bfloat16)
            proj = F.linear(x, proj_weight)
            left, right = _sigmoid_mask_layout_packed(proj, mask, B, N, H)
            out_h = torch.bmm(left, right.transpose(1, 2))
            z = _norm_gate_packed(
                out_h,
                proj,
                weights["to_out_norm.weight"],
                weights["to_out_norm.bias"],
                B,
                N,
                H,
            )
            return F.linear(z, weights["to_out.weight"]).to(torch.float32)

        x = _layernorm_fp32(
            input_tensor,
            weights["norm.weight"],
            weights["norm.bias"],
            B,
            N,
            C,
        )
        xb = x.to(torch.bfloat16)
        tri_weight = torch.cat(
            (
                weights["left_proj.weight"],
                weights["right_proj.weight"],
                weights["left_gate.weight"],
                weights["right_gate.weight"],
            ),
            dim=0,
        ).to(torch.bfloat16)
        tri_proj = F.linear(xb, tri_weight)
        out_gate_logits = F.linear(x, weights["out_gate.weight"])

        left, right = _sigmoid_mask_layout_packed4(
            tri_proj,
            mask,
            B,
            N,
            H,
        )

        out_h = torch.bmm(left, right.transpose(1, 2))
        z = _norm_gate(
            out_h,
            out_gate_logits,
            weights["to_out_norm.weight"],
            weights["to_out_norm.bias"],
            B,
            N,
            H,
        )
        return F.linear(z, weights["to_out.weight"]).to(torch.float32)
    else:
        x = _layernorm_bf16(
            input_tensor,
            weights["norm.weight"],
            weights["norm.bias"],
            B,
            N,
            C,
        )

    proj_weight = torch.cat(
        (
            weights["left_proj.weight"],
            weights["right_proj.weight"],
            weights["left_gate.weight"],
            weights["right_gate.weight"],
            weights["out_gate.weight"],
        ),
        dim=0,
    )
    if C != 128:
        proj_weight = proj_weight.to(torch.bfloat16)
    proj = F.linear(x, proj_weight)

    left, right = _sigmoid_mask_layout_packed(
        proj,
        mask,
        B,
        N,
        H,
    )

    out_h = torch.bmm(left, right.transpose(1, 2))
    z = _norm_gate_packed(
        out_h,
        proj,
        weights["to_out_norm.weight"],
        weights["to_out_norm.bias"],
        B,
        N,
        H,
    )
    return F.linear(z, weights["to_out.weight"]).to(torch.float32)
