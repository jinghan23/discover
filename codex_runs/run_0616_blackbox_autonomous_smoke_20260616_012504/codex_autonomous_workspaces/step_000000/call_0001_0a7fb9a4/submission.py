"""
Optimized outgoing TriMul forward pass.

The code normalizes input rows in Triton directly to fp16, computes all five
input projections with one fp16 GEMM, packs/gates left and right tensors into
channel-major bf16 matrices for batched GEMM contraction, then fuses output
normalization with the output gate before a final fp16 GEMM with float32 output.
"""

import torch
import triton
import triton.language as tl

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

@triton.jit
def _layer_norm_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
                       dim: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < dim
    vals = tl.load(x_ptr + row * dim + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(vals, axis=0) / dim
    centered = tl.where(mask, vals - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / dim
    normed = centered * tl.rsqrt(var + 1.0e-5)
    weight = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * dim + offs, normed * weight + bias, mask=mask)


def _triton_layer_norm(x, weight, bias, dim):
    y = torch.empty_like(x, dtype=torch.float16)
    n_rows = x.numel() // dim
    block = triton.next_power_of_2(dim)
    warps = 1 if dim == 128 else 2
    _layer_norm_kernel[(n_rows,)](x, weight, bias, y, dim, block, num_warps=warps)
    return y


@triton.jit
def _prep_lr_tiled_kernel(left_ptr, right_ptr, lg_ptr, rg_ptr, mask_ptr, left_o_ptr, right_o_ptr,
                          n_rows: tl.constexpr, plane: tl.constexpr,
                          left_rs: tl.constexpr, right_rs: tl.constexpr,
                          lg_rs: tl.constexpr, rg_rs: tl.constexpr,
                          hidden: tl.constexpr, NOMASK: tl.constexpr,
                          BLOCK_R: tl.constexpr, BLOCK_H: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    h = tl.arange(0, BLOCK_H)
    row_mask = rows < n_rows
    b = rows // plane
    rem = rows - b * plane

    l = tl.load(left_ptr + rows[:, None] * left_rs + h[None, :],
                mask=row_mask[:, None], other=0.0).to(tl.float32)
    r = tl.load(right_ptr + rows[:, None] * right_rs + h[None, :],
                mask=row_mask[:, None], other=0.0).to(tl.float32)
    lg = tl.sigmoid(tl.load(lg_ptr + rows[:, None] * lg_rs + h[None, :],
                            mask=row_mask[:, None], other=0.0).to(tl.float32))
    rg = tl.sigmoid(tl.load(rg_ptr + rows[:, None] * rg_rs + h[None, :],
                            mask=row_mask[:, None], other=0.0).to(tl.float32))
    if not NOMASK:
        m = tl.load(mask_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        l = l * m[:, None]
        r = r * m[:, None]

    out_off = (b[None, :] * hidden + h[:, None]) * plane + rem[None, :]
    mask_o = row_mask[None, :]
    tl.store(left_o_ptr + out_off, tl.trans(l * lg), mask=mask_o)
    tl.store(right_o_ptr + out_off, tl.trans(r * rg), mask=mask_o)


def _prep_lr_tiled(left, right, left_gate, right_gate, mask, nomask, bs, seq_len, hidden_dim):
    left_o = torch.empty((bs * hidden_dim, seq_len, seq_len), device=left.device, dtype=torch.bfloat16)
    right_o = torch.empty_like(left_o)
    plane = seq_len * seq_len
    n_rows = bs * plane
    _prep_lr_tiled_kernel[(triton.cdiv(n_rows, 16),)](
        left, right, left_gate, right_gate, mask, left_o, right_o,
        n_rows, plane,
        left.stride(2), right.stride(2), left_gate.stride(2), right_gate.stride(2),
        hidden_dim, nomask,
        BLOCK_R=16, BLOCK_H=128, num_warps=8,
    )
    return left_o, right_o


@triton.jit
def _out_norm_gate_kernel(out_ptr, gate_ptr, norm_w_ptr, norm_b_ptr, y_ptr,
                          plane: tl.constexpr,
                          gate_rs: tl.constexpr,
                          BLOCK_H: tl.constexpr):
    row = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    batch = row // plane
    rem = row - batch * plane
    out_base = batch * BLOCK_H * plane + rem
    vals = tl.load(out_ptr + out_base + h * plane).to(tl.float32)
    mean = tl.sum(vals, axis=0) / BLOCK_H
    centered = vals - mean
    var = tl.sum(centered * centered, axis=0) / BLOCK_H
    nw = tl.load(norm_w_ptr + h).to(tl.float32)
    nb = tl.load(norm_b_ptr + h).to(tl.float32)
    gate = tl.sigmoid(tl.load(gate_ptr + row * gate_rs + h).to(tl.float32))
    y = (centered * tl.rsqrt(var + 1.0e-5) * nw + nb) * gate
    tl.store(y_ptr + row * BLOCK_H + h, y)


def _out_norm_gate(out, gate, norm_w, norm_b):
    hidden = 128
    n_rows = out.numel() // hidden
    y = torch.empty((n_rows, hidden), device=out.device, dtype=torch.float16)
    plane = out.shape[1] * out.shape[2]
    _out_norm_gate_kernel[(n_rows,)](
        out, gate, norm_w, norm_b, y,
        plane,
        gate.stride(2), BLOCK_H=128, num_warps=4,
    )
    return y.reshape(*out.shape[:-1], hidden)


@torch.no_grad()
def custom_kernel(data):
    input_tensor, mask, weights, config = data

    dim = config["dim"]
    x = _triton_layer_norm(input_tensor, weights["norm.weight"], weights["norm.bias"], dim)

    hidden_dim = config["hidden_dim"]
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
    x_2d = x.reshape(-1, dim)
    proj = torch.mm(x_2d, proj_weight.to(torch.float16).t()).reshape(
        input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2], hidden_dim * 5
    )
    left, right, left_gate, right_gate, out_gate = proj.split(hidden_dim, dim=-1)

    bs, seq_len = input_tensor.shape[0], input_tensor.shape[1]
    nomask = config.get("nomask", False)
    left_h, right_h = _prep_lr_tiled(left, right, left_gate, right_gate, mask, nomask, bs, seq_len, hidden_dim)
    out = torch.bmm(left_h, right_h.transpose(1, 2))
    out = out.reshape(bs, hidden_dim, seq_len, seq_len).permute(0, 2, 3, 1)
    out = _out_norm_gate(
        out,
        out_gate,
        weights["to_out_norm.weight"],
        weights["to_out_norm.bias"],
    )
    out = torch.mm(
        out.reshape(-1, hidden_dim),
        weights["to_out.weight"].to(torch.float16).t(),
        out_dtype=torch.float32,
    )
    return out.reshape(bs, seq_len, seq_len, dim)
