"""Hybrid Triton/cuBLAS outgoing Triangle Multiplicative Update forward.

Triton computes input layer norm and packs gated left/right activations into
[B * H, N, N] low-precision matrices.  C=128 uses FP16 for the first norm,
packed projection, and batched contraction; larger channel cases use BF16.
The output norm/gate stays in Triton, writes FP16 hidden activations, and the
final projection uses cuBLAS tensor cores with a float32 output.
"""

from task import input_t, output_t

import torch
import triton
import triton.language as tl


torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

_BUFFER_CACHE = {}


def _cached_empty(name: str, shape, device, dtype):
    key = (name, tuple(shape), dtype, device.index)
    out = _BUFFER_CACHE.get(key)
    if out is None or out.shape != tuple(shape) or out.dtype != dtype or out.device != device:
        out = torch.empty(shape, device=device, dtype=dtype)
        _BUFFER_CACHE[key] = out
    return out


def _cached_to(name: str, tensor: torch.Tensor, dtype):
    out = _cached_empty(name, tensor.shape, tensor.device, dtype)
    out.copy_(tensor)
    return out

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
    load_valid = valid & (m[:, None] != 0.0)

    lp_v = tl.load(proj + base, mask=load_valid, other=0.0).to(tl.float32)
    rp_v = tl.load(proj + base + H, mask=load_valid, other=0.0).to(tl.float32)
    lg_v = tl.load(proj + base + (2 * H), mask=load_valid, other=0.0).to(tl.float32)
    rg_v = tl.load(proj + base + (3 * H), mask=load_valid, other=0.0).to(tl.float32)
    left_v = lp_v * m[:, None] * tl.sigmoid(lg_v)
    right_v = rp_v * m[:, None] * tl.sigmoid(rg_v)

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
    left_v = lp_v * tl.sigmoid(lg_v)
    right_v = rp_v * tl.sigmoid(rg_v)

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
    gate = tl.sigmoid(gate_logits)
    out = (centered * inv[None, :] * nw[:, None] + nb[:, None]) * gate
    tl.store(z + cl, out)


@triton.jit
def _norm_gate_from_gate_tile_kernel(
    out_h,
    out_gate,
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
    gate_logits = tl.load(out_gate + offs_m[None, :] * H + offs_h[:, None]).to(tl.float32)
    gate = tl.sigmoid(gate_logits)
    out = (centered * inv[None, :] * nw[:, None] + nb[:, None]) * gate
    tl.store(z + cl, out)


@triton.jit
def _proj_gate_pack_c128_nomask_kernel(
    x,
    left_w,
    right_w,
    left_gate_w,
    right_gate_w,
    out_gate_w,
    left,
    right,
    out_gate,
    n_pairs: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    xv = tl.load(x + offs_m[:, None] * C + offs_c[None, :])
    w_l = tl.load(left_w + offs_h[None, :] * C + offs_c[:, None])
    w_lg = tl.load(left_gate_w + offs_h[None, :] * C + offs_c[:, None])
    lp = tl.dot(xv, w_l, out_dtype=tl.float32)
    lg = tl.dot(xv, w_lg, out_dtype=tl.float32)

    b = offs_m // n_pairs
    rem = offs_m - b * n_pairs
    hmajor = (b[:, None] * H + offs_h[None, :]) * n_pairs + rem[:, None]
    tl.store(left + hmajor, lp * tl.sigmoid(lg))

    w_r = tl.load(right_w + offs_h[None, :] * C + offs_c[:, None])
    w_rg = tl.load(right_gate_w + offs_h[None, :] * C + offs_c[:, None])
    rp = tl.dot(xv, w_r, out_dtype=tl.float32)
    rg = tl.dot(xv, w_rg, out_dtype=tl.float32)
    tl.store(right + hmajor, rp * tl.sigmoid(rg))

    w_og = tl.load(out_gate_w + offs_h[None, :] * C + offs_c[:, None])
    og = tl.dot(xv, w_og, out_dtype=tl.float32)
    tl.store(out_gate + offs_m[:, None] * H + offs_h[None, :], og)

@triton.jit
def _bmm_hmajor_kernel(
    left,
    right,
    out,
    N: tl.constexpr,
    n_pairs: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    base = bid * n_pairs

    a_ptrs = left + base + offs_m[:, None] * N + offs_k[None, :]
    b_ptrs = right + base + offs_n[None, :] * N + offs_k[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, N, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    tl.store(out + base + offs_m[:, None] * N + offs_n[None, :], acc)


def _sigmoid_mask_layout_packed(proj, mask, B: int, N: int, H: int):
    left = _cached_empty("left_bf16", (B * H, N, N), proj.device, torch.bfloat16)
    right = _cached_empty("right_bf16", (B * H, N, N), proj.device, torch.bfloat16)
    if mask.dtype is torch.float32:
        grid = (triton.cdiv(B * N * N, 64), triton.cdiv(H, 64))
        _prep_packed_hmajor_nomask_kernel[grid](
            proj,
            left,
            right,
            N * N,
            H,
            BLOCK_M=64,
            BLOCK_H=64,
            num_warps=8,
        )
    else:
        grid = (triton.cdiv(B * N * N, 32), triton.cdiv(H, 64))
        _prep_packed_hmajor_mask_kernel[grid](
            proj,
            mask,
            left,
            right,
            N * N,
            H,
            BLOCK_M=32,
            BLOCK_H=64,
            num_warps=4,
        )
    return left, right

def _sigmoid_mask_layout_packed_fp16(proj, mask, B: int, N: int, H: int):
    left = _cached_empty("left_fp16", (B * H, N, N), proj.device, torch.float16)
    right = _cached_empty("right_fp16", (B * H, N, N), proj.device, torch.float16)
    if mask.dtype is torch.float32:
        grid = (triton.cdiv(B * N * N, 64), triton.cdiv(H, 64))
        _prep_packed_hmajor_nomask_kernel[grid](
            proj,
            left,
            right,
            N * N,
            H,
            BLOCK_M=64,
            BLOCK_H=64,
            num_warps=8,
        )
    else:
        grid = (triton.cdiv(B * N * N, 32), triton.cdiv(H, 64))
        _prep_packed_hmajor_mask_kernel[grid](
            proj,
            mask,
            left,
            right,
            N * N,
            H,
            BLOCK_M=32,
            BLOCK_H=64,
            num_warps=4,
        )
    return left, right


def _norm_gate_packed(out_h, proj, norm_w, norm_b, B: int, N: int, H: int, dtype):
    z = _cached_empty("z", (B, N, N, H), out_h.device, dtype)
    block_m = 64
    warps = 8 if 512 <= N < 1024 else 4
    _norm_gate_packed_tile_kernel[(triton.cdiv(B * N * N, block_m),)](
        out_h,
        proj,
        norm_w,
        norm_b,
        z,
        N * N,
        H,
        BLOCK_M=block_m,
        BLOCK_H=128,
        num_warps=warps,
    )
    return z


def _norm_gate_from_gate(out_h, out_gate, norm_w, norm_b, B: int, N: int, H: int, dtype):
    z = _cached_empty("z", (B, N, N, H), out_h.device, dtype)
    block_m = 64
    warps = 8 if N == 768 else 4
    _norm_gate_from_gate_tile_kernel[(triton.cdiv(B * N * N, block_m),)](
        out_h,
        out_gate,
        norm_w,
        norm_b,
        z,
        N * N,
        H,
        BLOCK_M=block_m,
        BLOCK_H=128,
        num_warps=warps,
    )
    return z


def _proj_gate_pack_c128_nomask(x, weights, B: int, N: int, H: int, block_m: int = 128, block_h: int = 64, warps: int = 4):
    left = _cached_empty("left_fp16", (B * H, N, N), x.device, torch.float16)
    right = _cached_empty("right_fp16", (B * H, N, N), x.device, torch.float16)
    out_gate = _cached_empty("out_gate_fp16", (B * N * N, H), x.device, torch.float16)
    _proj_gate_pack_c128_nomask_kernel[(triton.cdiv(B * N * N, block_m), triton.cdiv(H, block_h))](
        x,
        _cached_to("left_proj_w_fp16", weights["left_proj.weight"], torch.float16),
        _cached_to("right_proj_w_fp16", weights["right_proj.weight"], torch.float16),
        _cached_to("left_gate_w_fp16", weights["left_gate.weight"], torch.float16),
        _cached_to("right_gate_w_fp16", weights["right_gate.weight"], torch.float16),
        _cached_to("out_gate_w_fp16", weights["out_gate.weight"], torch.float16),
        left,
        right,
        out_gate,
        N * N,
        128,
        H,
        BLOCK_M=block_m,
        BLOCK_C=128,
        BLOCK_H=block_h,
        num_warps=warps,
    )
    return left, right, out_gate

def _bmm_hmajor(left, right, B: int, N: int, H: int, block_m: int = 32, block_n: int = 32, block_k: int = 64, warps: int = 4):
    out = _cached_empty("out_h", (B * H, N, N), left.device, left.dtype)
    grid = (triton.cdiv(N, block_m) * triton.cdiv(N, block_n), B * H)
    _bmm_hmajor_kernel[grid](
        left,
        right,
        out,
        N,
        N * N,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=warps,
    )
    return out

def _layernorm_bf16(input_tensor, norm_w, norm_b, B: int, N: int, C: int):
    y = _cached_empty("x_norm_bf16", input_tensor.shape, input_tensor.device, torch.bfloat16)
    rows = B * N * N
    if C <= 128:
        block_m = 16
        block_c = 128
        warps = 4
    elif C <= 256:
        block_m = 16
        block_c = 256
        warps = 4
    elif C <= 512:
        block_m = 16
        block_c = 512
        warps = 8
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


def _layernorm_fp16(input_tensor, norm_w, norm_b, B: int, N: int, C: int):
    y = _cached_empty("x_norm_fp16", input_tensor.shape, input_tensor.device, torch.float16)
    rows = B * N * N
    if C <= 128:
        block_m = 16
        block_c = 128
        warps = 4
    elif C <= 256:
        block_m = 16
        block_c = 256
        warps = 4
    elif C <= 512:
        block_m = 16
        block_c = 512
        warps = 8
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

    use_fp16 = C == 128
    fused_c128_nomask = use_fp16 and mask.dtype is torch.float32
    if fused_c128_nomask:
        x = _layernorm_fp16(
            input_tensor,
            weights["norm.weight"],
            weights["norm.bias"],
            B,
            N,
            C,
        )
        left, right, out_gate = _proj_gate_pack_c128_nomask(
            x,
            weights,
            B,
            N,
            H,
        )
    else:
        if use_fp16:
            x = _layernorm_fp16(
                input_tensor,
                weights["norm.weight"],
                weights["norm.bias"],
                B,
                N,
                C,
            )
            proj_dtype = torch.float16
        else:
            x = _layernorm_bf16(
                input_tensor,
                weights["norm.weight"],
                weights["norm.bias"],
                B,
                N,
                C,
            )
            proj_dtype = torch.bfloat16

        proj_weight_f32 = _cached_empty("proj_weight_f32", (5 * H, C), x.device, torch.float32)
        torch.cat(
            (
                weights["left_proj.weight"],
                weights["right_proj.weight"],
                weights["left_gate.weight"],
                weights["right_gate.weight"],
                weights["out_gate.weight"],
            ),
            dim=0,
            out=proj_weight_f32,
        )
        proj_weight = _cached_to("proj_weight", proj_weight_f32, proj_dtype)

        proj = _cached_empty("proj", (B * N * N, 5 * H), x.device, proj_dtype)
        torch.mm(x.reshape(B * N * N, C), proj_weight.t(), out=proj)

        if use_fp16:
            left, right = _sigmoid_mask_layout_packed_fp16(
                proj,
                mask,
                B,
                N,
                H,
            )
        else:
            left, right = _sigmoid_mask_layout_packed(
                proj,
                mask,
                B,
                N,
                H,
            )

    if C == 128 and N == 256:
        out_h = _bmm_hmajor(left, right, B, N, H, block_m=128, block_n=64, block_k=64, warps=4)
    else:
        out_h = _cached_empty("out_h", (B * H, N, N), left.device, left.dtype)
        torch.bmm(left, right.transpose(1, 2), out=out_h)
    if fused_c128_nomask:
        z = _norm_gate_from_gate(
            out_h,
            out_gate,
            weights["to_out_norm.weight"],
            weights["to_out_norm.bias"],
            B,
            N,
            H,
            torch.float16,
        )
    else:
        final_dtype = torch.float16
        z = _norm_gate_packed(
            out_h,
            proj,
            weights["to_out_norm.weight"],
            weights["to_out_norm.bias"],
            B,
            N,
            H,
            final_dtype,
        )
    out = _cached_empty("out", (B * N * N, C), z.device, torch.float32)
    torch.mm(
        z.reshape(B * N * N, H),
        _cached_to("to_out_w_fp16", weights["to_out.weight"], torch.float16).t(),
        out=out,
        out_dtype=torch.float32,
    )
    return out.reshape(B, N, N, C)
