# GPUMode Problem Index

Source repo: <https://github.com/gpu-mode/reference-kernels>

Local checkout:

- Path: `GPUMode/reference-kernels`
- Commit: `76d3011`
- Size: about `1.7M`

Each problem directory is usually organized as:

- `task.yml`: problem statement, tests, benchmarks, timeout, ranking metric.
- `reference.py`: reference implementation.
- `task.py`: input/output schema and data generation helpers.
- `eval.py`: evaluator, usually shared at the competition-group level.
- `README.md`: extra notes where present.

## Mapping From The Kernel Taxonomy

| Category | Problem names | Local directories |
| --- | --- | --- |
| Dense GEMM / Tensor Core | `matmul_v2` | `reference-kernels/problems/pmpp_v2/matmul_py` |
| Dense GEMM / Tensor Core | `amd-fp8-mm` | `reference-kernels/problems/amd/fp8-mm` |
| Dense GEMM / Tensor Core | `nvfp4_gemm` | `reference-kernels/problems/nvidia/nvfp4_gemm` |
| Dense GEMM / Tensor Core | `nvfp4_dual_gemm` | `reference-kernels/problems/nvidia/nvfp4_dual_gemm` |
| Dense GEMM / Tensor Core | `modal_nvfp4_dual_gemm` | `reference-kernels/problems/nvidia/modal_nvfp4_dual_gemm` |
| Dense GEMM / Tensor Core | `nvfp4_group_gemm` | `reference-kernels/problems/nvidia/nvfp4_group_gemm` |
| Dense GEMM / Tensor Core | `amd-mxfp4-mm` | `reference-kernels/problems/amd_202602/mxfp4-mm` |
| GEMV / skinny GEMM / decode-like GEMM | `nvfp4_gemv` | `reference-kernels/problems/nvidia/nvfp4_gemv` |
| GEMV / skinny GEMM / decode-like GEMM | `amd-mla-decode` | `reference-kernels/problems/amd/mla-decode` |
| Quantized / low-precision | `fp8_quant` | `reference-kernels/problems/helion/fp8_quant_py` |
| Quantized / low-precision | `amd-fp8-mm`, `amd-mxfp4-mm`, `nvfp4_*` | See GEMM/NVIDIA rows above |
| Elementwise / streaming memory | `vectoradd_v2` | `reference-kernels/problems/pmpp_v2/vectoradd_py` |
| Elementwise / streaming memory | `grayscale_v2` | `reference-kernels/problems/pmpp_v2/grayscale_py` |
| Reduction / histogram / softmax-like | `vectorsum_v2` | `reference-kernels/problems/pmpp_v2/vectorsum_py` |
| Reduction / histogram / softmax-like | `histogram_v2` | `reference-kernels/problems/pmpp_v2/histogram_py` |
| Reduction / histogram / softmax-like | `princeton_cross_entropy` | `reference-kernels/problems/princeton/cross_entropy_py` |
| Scan / prefix / sort | `prefixsum_v2` | `reference-kernels/problems/pmpp_v2/prefixsum_py` |
| Scan / prefix / sort | `sort_v2` | `reference-kernels/problems/pmpp_v2/sort_py` |
| Convolution / local-window | `conv2d_v2` | `reference-kernels/problems/pmpp_v2/conv2d_py` |
| Convolution / local-window | `causal_conv1d` | `reference-kernels/problems/helion/causal_conv1d_py` |
| Attention decode / KV-cache | `amd-mla-decode` | `reference-kernels/problems/amd/mla-decode` |
| Attention decode / KV-cache | `amd-mixed-mla` | `reference-kernels/problems/amd_202602/mixed-mla` |
| MoE / routing / grouped GEMM | `amd-mixture-of-experts` | `reference-kernels/problems/amd/moe` |
| MoE / routing / grouped GEMM | `amd-moe-mxfp4` | `reference-kernels/problems/amd_202602/moe-mxfp4` |
| Distributed / communication-bound | `amd-all2all` | `reference-kernels/problems/amd_distributed/all2all` |
| Distributed / communication-bound | `amd-gemm-rs` | `reference-kernels/problems/amd_distributed/gemm-rs` |
| Distributed / communication-bound | `amd-ag-gemm` | `reference-kernels/problems/amd_distributed/ag-gemm` |
| Model-specific fused | `trimul` | `reference-kernels/problems/bioml/trimul` |
| Model-specific fused | `gated_deltanet_chunk_fwd_h` | `reference-kernels/problems/helion/gated_deltanet_chunk_fwd_h_py` |
| Model-specific fused | `gated_deltanet_chunk_fwd_o` | `reference-kernels/problems/helion/gated_deltanet_chunk_fwd_o_py` |
| Model-specific fused | `gated_deltanet_recompute_w_u` | `reference-kernels/problems/helion/gated_deltanet_recompute_w_u_py` |
| Warmup / correctness | `amd-identity` | `reference-kernels/problems/amd/identity` |

## First-pass Notes

- PMPP v2 problems are compact practice kernels and support `B200`, `H100`, `A100`, and `L4`.
- NVIDIA NVFP4 problems target Blackwell/B200-style FP4 layouts and mostly rank by geometric mean.
- AMD 2025 problems target `MI300`; AMD February 2026 problems target `MI355X`.
- Distributed AMD problems are for `MI300x8` and include communication plus compute patterns.
- The existing project already had local integrations for `trimul` and `mla_decode_nvidia` under `examples/gpu_mode`.
- I did not download the full KernelBot submission datasets from Hugging Face; they are multi-GB. This checkout contains the problem specs and reference kernels.
