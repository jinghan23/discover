# A100-only GPUMode Reproduction

This directory contains the redone A100-focused pass.

- `insights/`: one short A100-only idea per problem, each kept intentionally under 64 tokens.
- `reproductions/`: A100-targeted reproduction code.
- `reports/verification_results.json`: local verification output.
- `reports/verify_a100_only.py`: verification script.

Local environment note: this machine has no CUDA device, so official GPUMode A100 evaluator correctness and timing are still pending on an A100 runner.

## Results

| Problem | A100 focus | Local check |
| --- | --- | --- |
| `matmul_v2` | cuBLASLt max-shape fast path, `torch.mm` fallback | CPU smoke pass |
| `trimul` | A100 algorithmic layout, PyTorch fallback-style implementation | CPU smoke pass |
| `vectoradd_v2` | `N=16384` `uint4/half2` CUDA fast path | CPU smoke pass |
| `vectorsum_v2` | 52M `float4` reduction + `atomicAdd` fast path | CPU smoke pass |
