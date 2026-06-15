# Dependency Tree / Graph

Generated: 2026-06-16

This is a dependency DAG, not a strict tree. Most non-blocked bases are previous generated submissions. Reference-derived rows also have a dotted dependency on a known-better source.

Files:
- Graphviz DOT: `DEPENDENCY_GRAPH.dot`
- Source index: `IDEA_RESULTS_INDEX.csv`

## Answer To Base Question

- Yes: for ordinary implementation rows, if `start_file` is not `blocked_file/submission.py`, it is a prior generated result such as `top_split_03`, `iter02_09_workspace_only_no_mask_patch`, or `iter03_02_left_right_only`.
- Exceptions are controls/reference rows: known-better direct evals, invalid direct-copy rows, baseline/rerun rows, and comparison-only rows. They are not normal child implementations.
- The graph should be read as a DAG because reference-derived candidates can have two conceptual inputs: blocked as execution base and known-better as reference source.

## Main Spine

```text
blocked_file/submission.py
├─ top_split_03_mask_plus_workspace/submission.py
│  └─ iter02_09_workspace_only_no_mask_patch/submission.py
│     └─ iter03_02_left_right_only/submission.py
│        ├─ iter04_*
│        ├─ iter05_*
│        ├─ iter06_*
│        ├─ iter07_*
│        └─ manual LR-base experiments
├─ random_* / top_split_* / known_better_repro_*
├─ unrun_worker blocked-base implementations
└─ blocked_base_candidates_20260612/*

known_better/web_a100_rank1_josusanmartin.py
└─ dotted reference edge into rank1 C++ hybrid/self-contained blocked-base candidates
```

## Children By Parent

### `blocked_file/submission.py`

| child/output | name | score_us | validity | base_source |
| --- | --- | --- | --- | --- |
| `blocked_base_candidates_20260612/blocked_rank1_cxx_refhybrid/submission.py` | blocked_rank1_cxx_refhybrid_gpu3 | 2508.575 | reference-derived hybrid evidence | blocked_base summary: copied blocked and added rank1 target branch |
| `blocked_base_candidates_20260612/blocked_rank1_cxx_refhybrid/submission.py` | blocked_rank1_cxx_refhybrid_rerun_gpu3 | 2517.699 | reference-derived hybrid evidence | blocked_base summary: copied blocked and added rank1 target branch |
| `subagent_2/rank1_cxx_branch_worker_20260612/submission.py` | hybrid_rank1_cxx_branch_rerun | 2532.275 | reference-derived hybrid evidence | worker summary: blocked for non-target branch; rank1 reference for target branch |
| `subagent_2/rank1_cxx_branch_worker_20260612/submission.py` | hybrid_rank1_cxx_branch | 2532.829 | reference-derived hybrid evidence | worker summary: blocked for non-target branch; rank1 reference for target branch |
| `subagent_2/results/top_split_03_mask_plus_workspace/submission.py` | agent2_top_split_03_mask_plus_workspace | 2533.983 | valid agent implementation | handoff Direct Base Mapping: blocked_file/submission.py |
| `blocked_base_candidates_20260612/blocked_rank1_cxx_selfcontained/submission.py` | blocked_rank1_cxx_selfcontained_gpu3 | 2534.902 | self-contained blocked-base candidate | blocked_base summary: copied blocked and embedded self-contained rank1 target branch |
| `blocked_base_candidates_20260612/blocked_lr_rank1_cxx_refhybrid/submission.py` | blocked_lr_rank1_cxx_refhybrid_gpu3 | 2537.230 | reference-derived hybrid evidence | blocked_base summary: copied blocked, added LR cache and rank1 target branch |
| `subagent_2/unrun_ideas_worker_20260612/results/top_split_05_c384_shape_cache__blocked_base/submission.py` | top_split_05_c384_shape_cache__blocked_base | 2538.959 | valid agent implementation | worker summary.csv base column |
| `manual_experiments_20260612/blocked_out_cache/submission.py` | blocked_out_cache | 2539.886 | valid but not better than LR cache | manual_experiments summary: copied blocked |
| `subagent_2/unrun_ideas_worker_20260612/results/iter03_04_og_tri_z_only__blocked_base/submission.py` | iter03_04_og_tri_z_only__blocked_base | 2545.380 | valid agent implementation | worker summary.csv base column |
| `subagent_2/unrun_ideas_worker_20260612/results/iter02_12_fixed_c128_shape_cache__blocked_base/submission.py` | iter02_12_fixed_c128_shape_cache__blocked_base | 2545.653 | valid agent implementation | worker summary.csv base column |
| `blocked_base_candidates_20260612/blocked_lr_cache/submission.py` | blocked_lr_cache_gpu3 | 2546.226 | valid blocked-base candidate | blocked_base summary: copied blocked and added LR cache |
| `subagent_2/results/top_split_08_c128_pack_block_tune/submission.py` | agent2_top_split_08_c128_pack_block_tune | 2548.705 | valid agent implementation | handoff Direct Base Mapping: blocked_file/submission.py |
| `subagent_2/results/random_07/submission.py` | agent2_random_07 | 2549.145 | valid agent implementation | handoff Direct Base Mapping: blocked_file/submission.py |
| `subagent_2/results/random_01/submission.py` | agent2_random_01 | 2553.579 | valid agent implementation | handoff Direct Base Mapping: blocked_file/submission.py |
| `subagent_2/unrun_ideas_worker_20260612/results/top_split_02_workspace_cache__blocked_base/submission.py` | top_split_02_workspace_cache__blocked_base | 2555.048 | valid agent implementation | worker summary.csv base column |
| `subagent_2/unrun_ideas_worker_20260612/results/iter03_03_left_right_tri_z_no_og__blocked_base/submission.py` | iter03_03_left_right_tri_z_no_og__blocked_base | 2565.035 | valid agent implementation | worker summary.csv base column |
| `subagent_2/results/top_split_04_c128_large_shape_cache/submission.py` | agent2_top_split_04_c128_large_shape_cache | 2572.296 | valid agent implementation | handoff Direct Base Mapping: blocked_file/submission.py |
| `subagent_2/results/top_split_07_intmask_u8_cache/submission.py` | agent2_top_split_07_intmask_u8_cache | 2577.896 | valid agent implementation | handoff Direct Base Mapping: blocked_file/submission.py |
| `manual_experiments_20260612/blocked_full_scratch/submission.py` | blocked_full_scratch | 2583.152 | negative | manual_experiments summary: copied blocked |
| `subagent_2/results/random_04/submission.py` | agent2_random_04 | 2592.634 | valid agent implementation | handoff Direct Base Mapping: blocked_file/submission.py |
| `subagent_2/unrun_ideas_worker_20260612/results/top_split_01_mask_dtype__blocked_base/submission.py` | top_split_01_mask_dtype__blocked_base | 2595.751 | valid agent implementation | worker summary.csv base column |
| `subagent_2/results/random_05/submission.py` | agent2_random_05 | 3009.356 | valid agent implementation | handoff Direct Base Mapping: blocked_file/submission.py |
| `subagent_2/results/top_split_06_mask_plus_final128/submission.py` | agent2_top_split_06_mask_plus_final128 | 3041.000 | valid agent implementation | handoff Direct Base Mapping: blocked_file/submission.py |
| `subagent_2/results/web_a100_rank1_josusanmartin/submission.py` | agent2_web_a100_rank1_josusanmartin | 3076.469 | valid agent implementation | handoff Direct Base Mapping: blocked_file/submission.py |
| `subagent_2/results/random_08/submission.py` | agent2_random_08 | 3562.273 | valid agent implementation | handoff Direct Base Mapping: blocked_file/submission.py |
| `subagent_2/results/random_02/submission.py` | agent2_random_02 | 3821.111 | valid agent implementation | handoff Direct Base Mapping: blocked_file/submission.py |
| `subagent_2/results/web_a100_rank3_ttt/submission.py` | agent2_web_a100_rank3_ttt | 3918.034 | valid agent implementation | handoff Direct Base Mapping: blocked_file/submission.py |
| `subagent_2/results/random_03/submission.py` | agent2_random_03 | 4268.109 | valid agent implementation | handoff Direct Base Mapping: blocked_file/submission.py |
| `subagent_2/results/web_a100_rank2_rd9000/submission.py` | agent2_web_a100_rank2_rd9000 | 7213.795 | valid agent implementation | handoff Direct Base Mapping: blocked_file/submission.py |
| `subagent_2/results/uuq_best_1785/submission.py` | agent2_uuq_best_1785 | 7389.063 | valid agent implementation | handoff Direct Base Mapping: blocked_file/submission.py |
| `subagent_2/unrun_ideas_worker_20260612/results/uuq_best_1998_noncopy_template__blocked_base/submission.py` | uuq_best_1998_noncopy_template__blocked_base | 7399.961 | valid agent implementation | worker summary.csv base column |
| `subagent_2/unrun_ideas_worker_20260612/results/uuq_best_1792_noncopy_template__blocked_base/submission.py` | uuq_best_1792_noncopy_template__blocked_base | 7400.136 | valid agent implementation | worker summary.csv base column |
| `subagent_2/results/random_06/submission.py` | agent2_random_06 | 8663.091 | valid agent implementation | handoff Direct Base Mapping: blocked_file/submission.py |
| `subagent_2/unrun_ideas_worker_20260612/results/uuq_best_1777_noncopy_template__blocked_base/submission.py` | uuq_best_1777_noncopy_template__blocked_base |  | not scored; see note | worker summary.csv base column |
| `subagent_2/unrun_ideas_worker_20260612/results/uuq_best_1783_noncopy_template__blocked_base/submission.py` | uuq_best_1783_noncopy_template__blocked_base |  | not scored; see note | worker summary.csv base column |

### `subagent_2/iter_02_base_top_split_03/results/iter02_09_workspace_only_no_mask_patch/submission.py`

| child/output | name | score_us | validity | base_source |
| --- | --- | --- | --- | --- |
| `subagent_2/iter_03_base_workspace_only/results/iter03_02_left_right_only/submission.py` | agent2_iter03_iter03_02_left_right_only | 2521.688 | valid agent implementation | handoff Direct Base Mapping: iter03 base workspace_only |
| `subagent_2/iter_03_base_workspace_only/results/iter03_08_left_right_plus_tri/submission.py` | agent2_iter03_iter03_08_left_right_plus_tri | 2525.011 | valid agent implementation | handoff Direct Base Mapping: iter03 base workspace_only |
| `subagent_2/unrun_ideas_worker_20260612/results/iter03_05_workspace_plus_mask_patch_exact__workspace_only_base/submission.py` | iter03_05_workspace_plus_mask_patch_exact__workspace_only_base | 2530.228 | valid agent implementation | worker summary.csv base column |
| `subagent_2/iter_03_base_workspace_only/results/iter03_09_left_right_plus_z/submission.py` | agent2_iter03_iter03_09_left_right_plus_z | 2542.444 | valid agent implementation | handoff Direct Base Mapping: iter03 base workspace_only |
| `subagent_2/iter_03_base_workspace_only/results/iter03_07_left_right_plus_mask_patch/submission.py` | agent2_iter03_iter03_07_left_right_plus_mask_patch | 2551.598 | valid agent implementation | handoff Direct Base Mapping: iter03 base workspace_only |
| `subagent_2/unrun_ideas_worker_20260612/results/iter03_06_workspace_device_index_key__workspace_only_base/submission.py` | iter03_06_workspace_device_index_key__workspace_only_base | 2551.916 | valid agent implementation | worker summary.csv base column |
| `subagent_2/iter_03_base_workspace_only/results/iter03_01_left_right_og_only/submission.py` | agent2_iter03_iter03_01_left_right_og_only | 2591.196 | valid agent implementation | handoff Direct Base Mapping: iter03 base workspace_only |

### `subagent_2/iter_03_base_workspace_only/results/iter03_02_left_right_only/submission.py`

| child/output | name | score_us | validity | base_source |
| --- | --- | --- | --- | --- |
| `subagent_2/iter_04_base_lr_only/results/iter04_01_shape_selective_extra_workspace/submission.py` | agent2_iter04_iter04_01_shape_selective_extra_workspace | 2530.313 | valid agent implementation | handoff Direct Base Mapping: iter04 base iter03_02_left_right_only |
| `subagent_2/iter_05_shape_specific_big/results/iter05_01_c384_mask_pack_r16_w4/submission.py` | agent2_iter05_iter05_01_c384_mask_pack_r16_w4 | 2530.432 | valid agent implementation | handoff Direct Base Mapping: iter05 base iter03_02_left_right_only |
| `manual_experiments_20260612/lr_smallln_n768_block32/submission.py` | lr_smallln_n768_block32 | 2532.374 | GPU2 looked better; GPU1 tied LR cache | manual_experiments summary: copied iter03_02 LR base |
| `manual_experiments_20260612/lr_rank1_ln_ext_smallshape/submission.py` | lr_rank1_ln_ext_smallshape | 2534.796 | mixed; cross-GPU score not enough | manual_experiments summary: copied iter03_02 LR base |
| `subagent_2/unrun_ideas_worker_20260612/results/iter06_05_c128_n1024_split_h32_four_launches__iter03_02_base/submission.py` | iter06_05_c128_n1024_split_h32_four_launches__iter03_02_base | 2538.044 | valid agent implementation | worker summary.csv base column |
| `subagent_2/iter_07_shape_selective_and_knownbetter/results/iter07_04_c384_n1024_pack_r16_w4/submission.py` | agent2_iter07_iter07_04_c384_n1024_pack_r16_w4 | 2540.095 | valid agent implementation | handoff Direct Base Mapping: iter07 base iter03_02_left_right_only |
| `subagent_2/iter_07_shape_selective_and_knownbetter/results/iter07_06_c384_n1024_bmm_split64/submission.py` | agent2_iter07_iter07_06_c384_n1024_bmm_split64 | 2540.781 | valid agent implementation | handoff Direct Base Mapping: iter07 base iter03_02_left_right_only |
| `subagent_2/iter_07_shape_selective_and_knownbetter/results/iter07_05_c384_n1024_pack_h32_w4/submission.py` | agent2_iter07_iter07_05_c384_n1024_pack_h32_w4 | 2541.863 | valid agent implementation | handoff Direct Base Mapping: iter07 base iter03_02_left_right_only |
| `manual_experiments_20260612/lr_rank1_ln_ext/submission.py` | lr_rank1_ln_ext | 2543.075 | mixed/negative overall | manual_experiments summary: copied iter03_02 LR base |
| `subagent_2/iter_05_shape_specific_big/results/iter05_03_c128_large_nomask_warps8/submission.py` | agent2_iter05_iter05_03_c128_large_nomask_warps8 | 2543.415 | valid agent implementation | handoff Direct Base Mapping: iter05 base iter03_02_left_right_only |
| `manual_experiments_20260612/lr_full_scratch_with_out/submission.py` | lr_full_scratch_with_out | 2545.986 | negative | manual_experiments summary: copied iter03_02 LR base |
| `subagent_2/iter_07_shape_selective_and_knownbetter/results/iter07_08_c384_n1024_pack_h32_w4_correct_branch/submission.py` | agent2_iter07_iter07_08_c384_n1024_pack_h32_w4_correct_branch | 2548.911 | valid agent implementation | handoff Direct Base Mapping: iter07 base iter03_02_left_right_only |
| `subagent_2/iter_07_shape_selective_and_knownbetter/results/iter07_02_shape1_pack_block32h32_only/submission.py` | agent2_iter07_iter07_02_shape1_pack_block32h32_only | 2548.999 | valid agent implementation | handoff Direct Base Mapping: iter07 base iter03_02_left_right_only |
| `subagent_2/iter_05_shape_specific_big/results/iter05_04_c384_n1024_final_fusion/submission.py` | agent2_iter05_iter05_04_c384_n1024_final_fusion | 2551.272 | valid agent implementation | handoff Direct Base Mapping: iter05 base iter03_02_left_right_only |
| `subagent_2/iter_06_projection_bmm/results/iter06_02_c384_n768_cuda_pack_only/submission.py` | agent2_iter06_iter06_02_c384_n768_cuda_pack_only | 2551.511 | valid agent implementation | handoff Direct Base Mapping: iter06 base iter03_02_left_right_only |
| `subagent_2/unrun_ideas_worker_20260612/results/iter06_03_c384_n1024_cuda_ln_only__iter03_02_base/submission.py` | iter06_03_c384_n1024_cuda_ln_only__iter03_02_base | 2552.354 | valid agent implementation | worker summary.csv base column |
| `manual_experiments_20260612/lr_shape_selective_fast/submission.py` | lr_shape_selective_fast | 2556.213 | negative | manual_experiments summary: copied iter03_02 LR base |
| `subagent_2/iter_07_shape_selective_and_knownbetter/results/iter07_01_shape1_workspace_only/submission.py` | agent2_iter07_iter07_01_shape1_workspace_only | 2559.884 | valid agent implementation | handoff Direct Base Mapping: iter07 base iter03_02_left_right_only |
| `subagent_2/iter_04_base_lr_only/results/iter04_02_lr_cache_fast_lookup/submission.py` | agent2_iter04_iter04_02_lr_cache_fast_lookup | 2562.024 | valid agent implementation | handoff Direct Base Mapping: iter04 base iter03_02_left_right_only |
| `subagent_2/iter_06_projection_bmm/results/iter06_04_c128_n1024_cache_proj5_weight_split/submission.py` | agent2_iter06_iter06_04_c128_n1024_cache_proj5_weight_split | 2568.123 | valid agent implementation | handoff Direct Base Mapping: iter06 base iter03_02_left_right_only |
| `manual_experiments_20260612/lr_full_scratch_no_out/submission.py` | lr_full_scratch_no_out | 2568.326 | negative | manual_experiments summary: copied iter03_02 LR base |
| `subagent_2/unrun_ideas_worker_20260612/results/iter07_03_shape0_tri_cache_only__iter03_02_base/submission.py` | iter07_03_shape0_tri_cache_only__iter03_02_base | 2570.584 | valid agent implementation | worker summary.csv base column |
| `subagent_2/iter_05_shape_specific_big/results/iter05_02_c384_mask_pack_h32/submission.py` | agent2_iter05_iter05_02_c384_mask_pack_h32 | 2585.371 | valid agent implementation | handoff Direct Base Mapping: iter05 base iter03_02_left_right_only |
| `manual_experiments_20260612/lr_z_out_cache/submission.py` | lr_z_out_cache | 2586.073 | negative | manual_experiments summary: copied iter03_02 LR base |
| `subagent_2/unrun_ideas_worker_20260612/results/iter03_10_left_right_device_index_key__iter03_02_base/submission.py` | iter03_10_left_right_device_index_key__iter03_02_base | 2589.069 | valid agent implementation | worker summary.csv base column |
| `subagent_2/iter_04_base_lr_only/results/iter04_03_lr_device_index_key/submission.py` | agent2_iter04_iter04_03_lr_device_index_key | 2590.935 | valid agent implementation | handoff Direct Base Mapping: iter04 base iter03_02_left_right_only |
| `subagent_2/iter_06_projection_bmm/results/iter06_01_c384_n768_cuda_ln_only/submission.py` | agent2_iter06_iter06_01_c384_n768_cuda_ln_only | 2593.173 | valid agent implementation | handoff Direct Base Mapping: iter06 base iter03_02_left_right_only |
| `manual_experiments_20260612/lr_out_cache/submission.py` | lr_out_cache | 2595.935 | negative | manual_experiments summary: copied iter03_02 LR base |
| `subagent_2/iter_07_shape_selective_and_knownbetter/results/iter07_07_c384_n1024_pack_r16_w4_correct_branch/submission.py` | agent2_iter07_iter07_07_c384_n1024_pack_r16_w4_correct_branch | 2602.620 | valid agent implementation | handoff Direct Base Mapping: iter07 base iter03_02_left_right_only |
| `subagent_2/unrun_ideas_worker_20260612/results/iter06_06_c384_n1024_cache_xn_only__iter03_02_base/submission.py` | iter06_06_c384_n1024_cache_xn_only__iter03_02_base | 2611.667 | valid agent implementation | worker summary.csv base column |
| `subagent_2/iter_05_shape_specific_big/results/iter05_06_c384_n768_cuda_ln_pack/submission.py` | agent2_iter05_iter05_06_c384_n768_cuda_ln_pack | 2865.568 | valid agent implementation | handoff Direct Base Mapping: iter05 base iter03_02_left_right_only |
| `subagent_2/iter_04_base_lr_only/results/iter04_04_fix_rank1_floatmask_nomask/submission.py` | agent2_iter04_iter04_04_fix_rank1_floatmask_nomask | 3064.690 | valid agent implementation | handoff Direct Base Mapping: iter04 base iter03_02_left_right_only |
| `manual_experiments_20260612/lr_shape_selective_out_cache/submission.py` | lr_shape_selective_out_cache | 3354.858 | strongly negative | manual_experiments summary: copied iter03_02 LR base |

### `subagent_2/results/top_split_03_mask_plus_workspace/submission.py`

| child/output | name | score_us | validity | base_source |
| --- | --- | --- | --- | --- |
| `subagent_2/iter_02_base_top_split_03/results/iter02_09_workspace_only_no_mask_patch/submission.py` | agent2_iter02_iter02_09_workspace_only_no_mask_patch | 2527.583 | valid agent implementation | handoff Direct Base Mapping: iter02 base top_split_03 |
| `subagent_2/iter_02_base_top_split_03/results/iter02_03_cache_only_tri_z/submission.py` | agent2_iter02_iter02_03_cache_only_tri_z | 2552.993 | valid agent implementation | handoff Direct Base Mapping: iter02 base top_split_03 |
| `subagent_2/iter_02_base_top_split_03/results/iter02_01_workspace_plus_c128_block_tune/submission.py` | agent2_iter02_iter02_01_workspace_plus_c128_block_tune | 2559.802 | valid agent implementation | handoff Direct Base Mapping: iter02 base top_split_03 |
| `subagent_2/iter_02_base_top_split_03/results/iter02_04_no_workspace_mask_only_plus_block_tune/submission.py` | agent2_iter02_iter02_04_no_workspace_mask_only_plus_block_tune | 2574.087 | valid agent implementation | handoff Direct Base Mapping: iter02 base top_split_03 |
| `subagent_2/iter_02_base_top_split_03/results/iter02_02_cache_xn_proj_for_mm_paths/submission.py` | agent2_iter02_iter02_02_cache_xn_proj_for_mm_paths | 2576.363 | valid agent implementation | handoff Direct Base Mapping: iter02 base top_split_03 |
| `subagent_2/unrun_ideas_worker_20260612/results/iter02_11_cache_left_right_only__top_split_03_base/submission.py` | iter02_11_cache_left_right_only__top_split_03_base | 2578.508 | valid agent implementation | worker summary.csv base column |
| `subagent_2/iter_02_base_top_split_03/results/iter02_10_cache_left_right_og_only/submission.py` | agent2_iter02_iter02_10_cache_left_right_og_only | 2615.664 | valid agent implementation | handoff Direct Base Mapping: iter02 base top_split_03 |
| `subagent_2/unrun_ideas_worker_20260612/results/iter02_07_c384_masked_u8_apply__top_split_03_base/submission.py` | iter02_07_c384_masked_u8_apply__top_split_03_base | 3228.355 | valid agent implementation | worker summary.csv base column |

## Backlog / Idea-Only Nodes

| start_file | name | idea | base_source |
| --- | --- | --- | --- |
| `subagent_2/results/top_split_03_mask_plus_workspace/submission.py` | iter02_05_c384_n1024_extension_path | `subagent_2/iter_02_base_top_split_03/ideas/iter02_05_c384_n1024_extension_path.txt` | handoff Saved-But-Not-Run path: iter02 idea folder base |
| `subagent_2/results/top_split_03_mask_plus_workspace/submission.py` | iter02_06_c128_large_extension_path | `subagent_2/iter_02_base_top_split_03/ideas/iter02_06_c128_large_extension_path.txt` | handoff Saved-But-Not-Run path: iter02 idea folder base |
| `subagent_2/results/top_split_03_mask_plus_workspace/submission.py` | iter02_07_c384_masked_u8_apply | `subagent_2/iter_02_base_top_split_03/ideas/iter02_07_c384_masked_u8_apply.txt` | handoff Saved-But-Not-Run path: iter02 idea folder base |
| `subagent_2/results/top_split_03_mask_plus_workspace/submission.py` | iter02_08_benchmark_shape_dispatch_table | `subagent_2/iter_02_base_top_split_03/ideas/iter02_08_benchmark_shape_dispatch_table.txt` | handoff Saved-But-Not-Run path: iter02 idea folder base |
| `subagent_2/results/top_split_03_mask_plus_workspace/submission.py` | iter02_11_cache_left_right_only | `subagent_2/iter_02_base_top_split_03/ideas/iter02_11_cache_left_right_only.txt` | handoff Saved-But-Not-Run path: iter02 idea folder base |
| `subagent_2/results/top_split_03_mask_plus_workspace/submission.py` | iter02_12_fixed_c128_shape_cache | `subagent_2/iter_02_base_top_split_03/ideas/iter02_12_fixed_c128_shape_cache.txt` | handoff Saved-But-Not-Run path: iter02 idea folder base |
| `subagent_2/iter_02_base_top_split_03/results/iter02_09_workspace_only_no_mask_patch/submission.py` | iter03_03_left_right_tri_z_no_og | `subagent_2/iter_03_base_workspace_only/ideas/iter03_03_left_right_tri_z_no_og.txt` | handoff Saved-But-Not-Run path: iter03 idea folder base |
| `subagent_2/iter_02_base_top_split_03/results/iter02_09_workspace_only_no_mask_patch/submission.py` | iter03_04_og_tri_z_only | `subagent_2/iter_03_base_workspace_only/ideas/iter03_04_og_tri_z_only.txt` | handoff Saved-But-Not-Run path: iter03 idea folder base |
| `subagent_2/iter_02_base_top_split_03/results/iter02_09_workspace_only_no_mask_patch/submission.py` | iter03_05_workspace_plus_mask_patch_exact | `subagent_2/iter_03_base_workspace_only/ideas/iter03_05_workspace_plus_mask_patch_exact.txt` | handoff Saved-But-Not-Run path: iter03 idea folder base |
| `subagent_2/iter_02_base_top_split_03/results/iter02_09_workspace_only_no_mask_patch/submission.py` | iter03_06_workspace_device_index_key | `subagent_2/iter_03_base_workspace_only/ideas/iter03_06_workspace_device_index_key.txt` | handoff Saved-But-Not-Run path: iter03 idea folder base |
| `subagent_2/iter_02_base_top_split_03/results/iter02_09_workspace_only_no_mask_patch/submission.py` | iter03_10_left_right_device_index_key | `subagent_2/iter_03_base_workspace_only/ideas/iter03_10_left_right_device_index_key.txt` | handoff Saved-But-Not-Run path: iter03 idea folder base |
| `subagent_2/iter_03_base_workspace_only/results/iter03_02_left_right_only/submission.py` | iter05_05_c384_n768_final_fusion | `subagent_2/iter_05_shape_specific_big/ideas/iter05_05_c384_n768_final_fusion.txt` | handoff Saved-But-Not-Run path: later iteration idea folder base |
| `subagent_2/iter_03_base_workspace_only/results/iter03_02_left_right_only/submission.py` | iter06_03_c384_n1024_cuda_ln_only | `subagent_2/iter_06_projection_bmm/ideas/iter06_03_c384_n1024_cuda_ln_only.txt` | handoff Saved-But-Not-Run path: later iteration idea folder base |
| `subagent_2/iter_03_base_workspace_only/results/iter03_02_left_right_only/submission.py` | iter06_05_c128_n1024_split_h32_four_launches | `subagent_2/iter_06_projection_bmm/ideas/iter06_05_c128_n1024_split_h32_four_launches.txt` | handoff Saved-But-Not-Run path: later iteration idea folder base |
| `subagent_2/iter_03_base_workspace_only/results/iter03_02_left_right_only/submission.py` | iter06_06_c384_n1024_cache_xn_only | `subagent_2/iter_06_projection_bmm/ideas/iter06_06_c384_n1024_cache_xn_only.txt` | handoff Saved-But-Not-Run path: later iteration idea folder base |
| `subagent_2/iter_03_base_workspace_only/results/iter03_02_left_right_only/submission.py` | iter07_03_shape0_tri_cache_only | `subagent_2/iter_07_shape_selective_and_knownbetter/ideas/iter07_03_shape0_tri_cache_only.txt` | handoff Saved-But-Not-Run path: later iteration idea folder base |
| `blocked_file/submission.py` | top_split_01_mask_dtype | `subagent_2/top_split_ideas/top_split_01_mask_dtype.txt` | handoff Saved-But-Not-Run path: initial implementation base |
| `blocked_file/submission.py` | top_split_02_workspace_cache | `subagent_2/top_split_ideas/top_split_02_workspace_cache.txt` | handoff Saved-But-Not-Run path: initial implementation base |
| `blocked_file/submission.py` | top_split_05_c384_shape_cache | `subagent_2/top_split_ideas/top_split_05_c384_shape_cache.txt` | handoff Saved-But-Not-Run path: initial implementation base |
| `blocked_file/submission.py` | uuq_best_1777 | `subagent_1/ideas/uuq_best_1777.txt` | handoff Saved-But-Not-Run path: initial implementation base |
| `blocked_file/submission.py` | uuq_best_1783 | `subagent_1/ideas/uuq_best_1783.txt` | handoff Saved-But-Not-Run path: initial implementation base |
| `blocked_file/submission.py` | uuq_best_1792 | `subagent_1/ideas/uuq_best_1792.txt` | handoff Saved-But-Not-Run path: initial implementation base |
| `blocked_file/submission.py` | uuq_best_1998 | `subagent_1/ideas/uuq_best_1998.txt` | handoff Saved-But-Not-Run path: initial implementation base |

## Excluded Controls

These are not implementation children in the tree: baselines, reruns, direct known-better evals, invalid copy rows, and comparison-only evals. They remain in `IDEA_RESULTS_INDEX.csv`.
