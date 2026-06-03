# Reproduction Comparison Report

All reproduced submissions were generated from the problem files plus idea notes, not by copying the top1 sources directly. The current machine has no CUDA device, so official GPUMode evaluator correctness and timing remain pending on matching GPUs.

| Problem | Reproduction | Insight | Local correctness | Closest top1 similarity | Notes |
| --- | --- | --- | --- | ---: | --- |
| `matmul_v2` | `GPUMode/reproductions/matmul_v2/submission.py` | `GPUMode/insights/matmul_v2.md` | CPU smoke pass; max abs err 0 | 27.6% | GPU official evaluator pending; cuda_available=false |
| `trimul` | `GPUMode/reproductions/trimul/submission.py` | `GPUMode/insights/trimul.md` | CPU smoke pass; max abs err 1.79e-07 | 12.5% | GPU official evaluator pending; cuda_available=false |
| `vectoradd_v2` | `GPUMode/reproductions/vectoradd_v2/submission.py` | `GPUMode/insights/vectoradd_v2.md` | CPU smoke pass; max abs err 0 | 43.9% | GPU official evaluator pending; cuda_available=false |
| `vectorsum_v2` | `GPUMode/reproductions/vectorsum_v2/submission.py` | `GPUMode/insights/vectorsum_v2.md` | CPU smoke pass; max abs err 0 | 30.6% | GPU official evaluator pending; cuda_available=false |

## Interpretation

- `matmul_v2`: Reproduces the H100 top1 idea directly: rely on PyTorch/cuBLAS `a @ b` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`. CPU smoke is exact for a small case.
- `vectoradd_v2`: Reproduces the H100 top1 as a single Triton flatten-and-add kernel with `BLOCK_SIZE=1024`, plus a PyTorch fallback. CPU fallback passes odd/even size cases.
- `vectorsum_v2`: Reproduces the H100 top1 direction more closely for the maximum benchmark size using nvcc/ctypes, vector loads, block reduction, scratch slots, and a completion counter. CPU fallback uses float64 reference-style sum.
- `trimul`: Reproduces the H100 top1 idea at a higher level: packed `[5H, M]` projection, hidden-major layout, `torch.bmm` triangle contraction, fp32 normalization stats, and fp16 CUDA intermediates. It is intentionally not a cuBLASLt/C++ extension clone.

## Remaining GPU Checks

- Run official evaluator for `trimul` on H100, ideally with the benchmark shapes in `problem_files/task.yml`.
- Run official evaluator for `matmul_v2`, `vectoradd_v2`, and `vectorsum_v2` on H100 first, since these reproductions target H100 top1 ideas.
- For non-H100 runners, use the insight docs rather than these H100-oriented reproductions; A100/B200/L4/MI300 top1 ideas differ materially.
