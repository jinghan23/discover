# TriMul blocked-pair single-idea N=3 audit

Scope:
- Prompt/manifest: `/opt/tiger/discover-gpu-kernel-experiments/trimul_subagent_prompts_blocked_pairs.md`
- Runs: `/opt/tiger/discover-gpu-kernel-experiments/codex_runs/trimul_blocked_pair_p*/`
- Checked: starter, target snapshot, extracted single idea, execute_01/02/03 submissions, agent outputs, and `latest_eval_summary.json`.

Note: every extracted "single idea" is far longer than the prompt's `<=64 words` rule. This audit treats that as prompt-format noncompliance, but evaluates content fidelity and execution fidelity separately.

## Summary matrix

| pair | idea vs starter->target gap | execution fidelity | iteration interpretation |
| --- | --- | --- | --- |
| p01_proxy2319_to_direct1869 | High, with one small mask/config drift | execute_02 is best fidelity | r1 missed generic Triton LN fallback; r2 fixed it and sped up; r3 no code change |
| p02_local2412_to_autoevolve2024 | Very high | execute_02 is best fidelity | r1 guard was too broad; r2 fixed exact `mask.dtype is torch.float32`; r3 no code change |
| p03_discover2460_to_blackbox1921 | High but incomplete on overlap/branch details | execute_03 is best among attempts, still not full target fidelity | r2/r3 make cache/copy refinements; remaining target overlap and some kernel details are absent |
| p04_reward2569_to_blackbox1921 | High and more detailed than p03 | execute_02 is best score/fidelity tradeoff | r2 fixes real layout/cat fidelity issues; r3 follows a minor detail but slows |
| p05_restart3483_to_autoevolve2024 | High | execute_01 already complete | execute_02/03 are byte-identical to earlier rounds; score changes are re-eval noise |
| p06_local2423_to_autoevolve2024 | High | execute_02 slightly closer | r2 only aligns final `to_out` copy helper; effect is negligible; r3 no code change |
| p07_discover2468_to_blackbox1926 | High enough to reproduce target if followed | Not executed correctly | execute_01 public tests fail; no valid N=3 conclusion |
| p08_proxy2319_to_a100rank3 | Very high | execute_01 already complete | execute_02/03 are byte-identical; later slowdown is re-eval noise |

## Per-pair notes

### p01_proxy2319_to_direct1869

Idea strictness: high. It accurately captures the target's direct pipeline: removal of starter `_WEIGHT_CACHE/_WORKSPACE_CACHE/_MM_MODE/_MM_OUT_MODE`, inline CUDA layernorm extension `trimul_ln384_ext`, scratch cache keyed by `(bsz, n, dim, hidden_dim, device_index)`, per-call `_pack_weights_kernel`, `_proj_gate_mask_kernel`, BMM, and `_ln_gate_out_kernel`.

Small drift: the idea says `config["nomask"]` is treated as no-mask. The target only checks `mask is not None and mask.dtype != torch.float32`; it does not read `config["nomask"]` in this branch.

Execution:
- execute_01 implements the main direct pipeline but used a PyTorch fallback for `dim > 512` input LN.
- execute_02 replaces that fallback with the target-style generic Triton row LN. This is a real fidelity fix and matches the small speedup: `2013.46 -> 2010.13 us`.
- execute_03 has no code diff from execute_02. The slower `2011.86 us` is not an iteration/fidelity signal.

### p02_local2412_to_autoevolve2024

Idea strictness: very high. It matches the autoevolve target structure: `_BUFFER_CACHE`, `_cached_empty`, `_cached_to`, BF16/FP16 Triton LN, H-major left/right packing, C=128 float32-mask fused projection path, C=128,N=256 Triton BMM, packed norm/gate kernels, and final FP16 weight to FP32 output matmul.

Execution:
- execute_01 implements the target structure, but its fused C=128 guard was too broad: it also fused `mask is None`.
- execute_02 fixes that to the idea/target guard: `C == 128 and mask.dtype is torch.float32`. This is a real fidelity fix and gives the best score: `2071.86 -> 2057.21 us`.
- execute_03 has no code diff from execute_02, so `2061.92 us` is measurement/runtime variance, not a worse idea implementation.

### p03_discover2460_to_blackbox1921

Idea strictness: high but not fully strict. It captures the active blackbox custom-kernel dataflow: shape-keyed buffers, dim128 fused LN/proj/gate path, dim384/768 LN + `proj` + channel-major transpose, raw gate from `proj`, BMM, and fused output projection.

Important omissions/looseness:
- The target starts async `to_out.weight.t()` copy before `torch.bmm`, then waits after BMM, overlapping copy with contraction. The idea does not make this overlap clear, and execute_03 copies after BMM.
- The target has separate nomask/no-out-gate post-projection kernels and exact row-block/warps in branches. The execution collapses some of this into one `_post_proj_transpose_kernel`.
- execute output LN kernels clamp variance with `tl.maximum(..., 0.0)` while the target output kernels use centered variance without that clamp.

Execution:
- execute_01 implements the main idea and passes.
- execute_02 improves cache-key fidelity by adding actual shape and normalized device index.
- execute_03 adds source metadata to avoid redundant async `to_out` copies. It improves score to `2107.95 us`, but it still is not full target fidelity because copy/BMM overlap and some branch exactness are missing.

### p04_reward2569_to_blackbox1921

Idea strictness: high. This is the strongest blackbox-family idea: it includes buffer shape clearing, dim-specialized LN, dim128 fused projection, post-projection no-out-gate paths, raw gate use for 384/768, async `to_out` copy overlapped with BMM, and fused output kernels.

Execution:
- execute_01 implements the broad structure but has fidelity issues in projection packing and left/right store layout.
- execute_02 fixes `torch.cat(..., out=proj_w)` fidelity and the H-major transpose stores. That is a real idea-following fix and gives the best score: `3266.97 -> 3257.17 us`.
- execute_03 derives `dim` from `config` and uses direct `torch.mm(..., out_dtype=torch.float16)`. This is closer to the text but slower: `3266.00 us`.

Interpretation: this pair remains far from the target score even when the idea is mostly followed. That points more to an insufficient/hard-to-transplant idea than to only sloppy execution.

### p05_restart3483_to_autoevolve2024

Idea strictness: high. It reflects the autoevolve target well: cached buffers, Triton BF16/FP16 LN, C=128 fused path, H-major pack kernels, C=128,N=256 Triton BMM, packed/separate gate norm kernels, and FP32 final output.

Execution:
- execute_01 already implements the idea.
- execute_02 is byte-identical to execute_01.
- execute_03 is byte-identical to execute_02.

Interpretation: the score movement `2506.52 -> 2501.53 -> 2506.65 us` is re-evaluation variance, not iteration benefit. This is a case where the idea was already executed to the available fidelity in round 1.

### p06_local2423_to_autoevolve2024

Idea strictness: high. It is another accurate autoevolve-target diff, similar to p02/p05 but scoped to the simpler local starter.

Execution:
- execute_01 implements the hybrid target structure.
- execute_02 changes only the final `to_out` copy path from manual `_cached_empty` + `copy_` to `_cached_to(...)`, which is slightly closer to the idea's helper contract.
- execute_03 is byte-identical to execute_02.

Interpretation: r2's `2511.73 -> 2510.43 us` improvement is too small to treat as strong evidence. The main idea was already executed in r1; r3's `2514.85 us` is not a fidelity failure.

### p07_discover2468_to_blackbox1926

Idea strictness: high enough. The idea matches the earlier blackbox target family: shape-keyed buffers, dim128 fused path, 384/768 branches, post-proj raw-gate path, async `to_out`, and fast output projection.

Execution is not to spec:
- execute_01 fails public tests, so there is no usable performance score or N=3 sequence.
- It changes the idea's dim128 fast path to `N < 768`, while the target handles dim128 with fused projection for larger N too.
- It uses `dim384, N>=1024 -> BLOCK_R=8` instead of the idea/target's `BLOCK_R=32`.
- It uses `dim768 -> BLOCK_R=2` even for large rows where the idea/target uses `8`.
- It forces `N>=768` through a `gated16 + mm` fallback instead of the target's fused output path for dim128/384/768.

Correctness failures are on large `N=1024` dim384/dim768 tests, exactly where those deviations matter. This pair should be classified as "execution failed", not as evidence that the idea had no value.

### p08_proxy2319_to_a100rank3

Idea strictness: very high. It accurately describes the rank3 target's uniform Triton pipeline: no caches/workspaces/shape branches, row LN to FP16, separate transposed FP16 projection weights, mask handling, hidden-major left/right, BMM, and fused hidden LN/gate/output projection.

Execution:
- execute_01 implements the idea. It adds backend flags, but otherwise preserves the uniform pipeline.
- execute_02 is byte-identical to execute_01.
- execute_03 is byte-identical to execute_02.

Interpretation: `2187.87 -> 2191.58 -> 2199.13 us` is not an iteration signal. This is a clean case where round 1 already followed the idea, and later rounds only re-ran equivalent code.

## Overall conclusion

The N=3 result should not be summarized as "iteration usually helps." It is more precise to split the cases:

- Real fidelity fixes with speedup: p01 execute_02, p02 execute_02, p04 execute_02.
- Mostly/no-op later rounds: p01 execute_03, p02 execute_03, p05 execute_02/03, p06 execute_03, p08 execute_02/03.
- Small implementation refinements with weak performance evidence: p03 execute_02/03, p06 execute_02.
- Invalid execution: p07 execute_01 fails correctness.

For p05 and p08, the right interpretation is not "following the idea better has no gain"; the implementation did not change. For p04, following the idea better helped a little, but the idea itself did not contain enough target-specific performance content to recover the target. For p03, some target-critical scheduling/overlap details were not fully captured or implemented, so it is only partial evidence.
