from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


def _work_dtype(x: torch.Tensor) -> torch.dtype:
    if x.is_cuda:
        return torch.float16
    return torch.float32


@torch.no_grad()
def custom_kernel(
    data: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]
) -> torch.Tensor:
    """TriMul outgoing forward pass with an H100-top1-style matrix layout."""

    input_tensor, mask, weights, config = data
    batch, seqlen, seqlen_2, dim = input_tensor.shape
    if seqlen != seqlen_2:
        raise ValueError("TriMul input must have square pair dimensions")

    hidden_dim = int(config["hidden_dim"])
    total_pairs = batch * seqlen * seqlen
    device = input_tensor.device
    work_dtype = _work_dtype(input_tensor)

    x_norm = F.layer_norm(
        input_tensor,
        (dim,),
        weights["norm.weight"].to(device=device, dtype=torch.float32),
        weights["norm.bias"].to(device=device, dtype=torch.float32),
        eps=1e-5,
    )

    flat_x = x_norm.reshape(total_pairs, dim).to(dtype=work_dtype)
    del x_norm
    packed_proj_weight = torch.cat(
        (
            weights["left_proj.weight"],
            weights["right_proj.weight"],
            weights["left_gate.weight"],
            weights["right_gate.weight"],
            weights["out_gate.weight"],
        ),
        dim=0,
    ).to(device=device, dtype=work_dtype)

    # H100 top submissions keep the projection output hidden-major: [5H, M].
    proj_t = packed_proj_weight @ flat_x.t()
    del flat_x, packed_proj_weight

    h = hidden_dim
    left_hm = proj_t[0:h]
    right_hm = proj_t[h : 2 * h]
    left_gate_hm = proj_t[2 * h : 3 * h]
    right_gate_hm = proj_t[3 * h : 4 * h]
    out_gate_hm = proj_t[4 * h : 5 * h]

    left_gate_hm.sigmoid_()
    right_gate_hm.sigmoid_()
    left_hm.mul_(left_gate_hm)
    right_hm.mul_(right_gate_hm)

    mask_flat = mask.reshape(1, total_pairs).to(device=device, dtype=work_dtype)
    left_hm.mul_(mask_flat)
    right_hm.mul_(mask_flat)
    del left_gate_hm, right_gate_hm, mask_flat

    left_bmm = left_hm.view(h * batch, seqlen, seqlen)
    right_bmm = right_hm.view(h * batch, seqlen, seqlen)
    contracted = torch.bmm(left_bmm, right_bmm.transpose(1, 2))
    del left_bmm, right_bmm, left_hm, right_hm

    contracted_hm = contracted.view(h, total_pairs)
    stats = contracted_hm.float()
    del contracted, contracted_hm

    mean = stats.mean(dim=0, keepdim=True)
    stats.sub_(mean)
    var = stats.square().mean(dim=0, keepdim=True)
    stats.mul_(torch.rsqrt(var.add_(1e-5)))
    del mean, var

    stats.mul_(weights["to_out_norm.weight"].to(device=device, dtype=torch.float32).view(h, 1))
    stats.add_(weights["to_out_norm.bias"].to(device=device, dtype=torch.float32).view(h, 1))
    out_gate_hm.sigmoid_()
    stats.mul_(out_gate_hm)
    del out_gate_hm, proj_t

    final_in = stats.to(dtype=work_dtype)
    del stats
    final_weight = weights["to_out.weight"].to(device=device, dtype=work_dtype)
    output_t = final_weight @ final_in

    return output_t.view(dim, batch, seqlen, seqlen).permute(1, 2, 3, 0)
