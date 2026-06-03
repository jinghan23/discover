# GPUMode Reproduction Reports

This directory stores local verification notes for reproduced submissions.

Important constraint: the current machine has no CUDA device, so official
KernelBot correctness and timing cannot be run here. Local checks are limited to:

- Python syntax parsing.
- `custom_kernel` presence and importability when possible.
- CPU smoke tests for reproductions that use PyTorch operations compatible with CPU.
- Static comparison against the original top1 submission and the problem reference.

Official correctness still requires running each reproduction with the matching
KernelBot evaluator and GPU runner.
