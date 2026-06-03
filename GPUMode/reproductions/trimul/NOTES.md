# TriMul reproduction notes

## Reproduced ideas

- Rewrites the forward pass as the same high-level stages described for the H100 top1 path: first LayerNorm and packed projection, hidden-channel batched triangle GEMM, then hidden LayerNorm, output gate, and final projection.
- Packs the five projection/gate weights into one `[5H, C]` matrix and computes the projection as `[5H, C] @ [C, M] -> [5H, M]`, where `M = B * N * N`.
- Keeps projection and contraction buffers hidden-major (`[H, M]` and `[H * B, N, N]`) so the outgoing contraction becomes one `torch.bmm(left, right.transpose(1, 2))`.
- Uses fp32 LayerNorm statistics and CUDA fp16 working buffers for the Tensor Core-friendly portions, matching the tolerance-based approximation strategy in the top solutions.
- Applies the mask for both masked and unmasked cases rather than relying on a `nomask` flag in `config`, because the provided problem config only contains `dim` and `hidden_dim`.

## Simplifications vs. top1

- This is a pure PyTorch reproduction. It does not build a CUDA/C++ extension, does not call cuBLASLt directly, and does not cache or benchmark Lt algorithms per shape.
- LayerNorm, sigmoid, gate, and final projection are not fused into custom kernels. They are expressed as PyTorch tensor operations, so launch count and temporary memory are higher.
- Projection weights are repacked every call. There is no cross-call weight cache, which keeps behavior correct if the evaluator passes new weight tensors.
- The final `[C, M]` result is returned as a view/permute to `[B, N, N, C]`, following the top1 layout idea, but without a specialized output kernel.

## Validation requirements

- Intended target: NVIDIA H100 or another CUDA GPU with fast fp16 Tensor Cores. The official benchmark shapes can require many GB of memory, especially for `N=1024, C=768`.
- Runtime dependencies: PyTorch with CUDA support. No Triton or custom CUDA build is required for this reproduction.
- This local machine reports `torch.cuda.is_available() == False`, so I could not run the official GPUMode CUDA evaluator here. I validated the math path with a small CPU reference check instead.
