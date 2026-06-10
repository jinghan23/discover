# main

## Summary

主线基准分支；相对当前基准 2234282 是更早历史，不是一个独立实现分支。本文档记录它与 2234282 的关系，避免把主线历史差异误认为新的采样方法。

Note: main 没有 checkout 到独立 worktree；这里记录的是 ref 与 2234282 的历史差异。

## Branch State

- Worktree: `(none)`
- HEAD: `6c40e82`
- Base used for comparison: `2234282`
- Commits ahead of base: `0`
- Commits behind base: `16`
- Group: `other`
- Implementation location: `no implementation diff found`

## Implemented Behavior

- The summary above is based on reading the changed Python code, added config fields, and sampler/prompt hooks in this branch.

## Added Markers

### Config fields

- `gpu_type`
- `mode`
- `task_name`
- `app_name`
- `env_type`

### Constants

- None

### Classes

- `SimpleReporter`

### Functions

- `_update_message`
- `display_report`
- `load_task`
- `run_on_modal`
- `get_gpu_mode_error`
- `custom_kernel`

## Diff Summary

- Worktree tracked shortstat: `(empty)`
- Committed diff stat tail: `278 files changed, 181 insertions(+), 131586 deletions(-)`
- Untracked files: `0`

### Worktree Status

````text
(no worktree checked out)
````
### Committed Branch Files Compared To Base

````text
M	.gitignore
D	.gitmodules
D	GPUMode/a100_only/README.md
D	GPUMode/a100_only/insights/matmul_v2.md
D	GPUMode/a100_only/insights/trimul.md
D	GPUMode/a100_only/insights/vectoradd_v2.md
D	GPUMode/a100_only/insights/vectorsum_v2.md
D	GPUMode/a100_only/reports/reproduction_quality_review.md
D	GPUMode/a100_only/reports/summary.md
D	GPUMode/a100_only/reports/verification_results.json
D	GPUMode/a100_only/reports/verify_a100_only.py
D	GPUMode/a100_only/reproductions/matmul_v2/submission.py
D	GPUMode/a100_only/reproductions/trimul/submission.py
D	GPUMode/a100_only/reproductions/vectoradd_v2/submission.py
D	GPUMode/a100_only/reproductions/vectorsum_v2/submission.py
D	GPUMode/insights/matmul_v2.md
D	GPUMode/insights/trimul.md
D	GPUMode/insights/vectoradd_v2.md
D	GPUMode/insights/vectorsum_v2.md
D	GPUMode/problem_index.md
D	GPUMode/reference-kernels
D	GPUMode/reproduction_reports/README.md
D	GPUMode/reproduction_reports/final_comparison.md
D	GPUMode/reproduction_reports/semantic_equivalence.md
D	GPUMode/reproduction_reports/verification_results.json
D	GPUMode/reproduction_reports/verify_reproductions.py
D	GPUMode/reproductions/matmul_v2/NOTES.md
D	GPUMode/reproductions/matmul_v2/submission.py
D	GPUMode/reproductions/trimul/NOTES.md
D	GPUMode/reproductions/trimul/submission.py
D	GPUMode/reproductions/vectoradd_v2/NOTES.md
D	GPUMode/reproductions/vectoradd_v2/submission.py
D	GPUMode/reproductions/vectorsum_v2/NOTES.md
D	GPUMode/reproductions/vectorsum_v2/submission.py
D	GPUMode/top_solutions/README.md
D	GPUMode/top_solutions/matmul_v2/metadata.json
D	GPUMode/top_solutions/matmul_v2/problem_files/reference.py
D	GPUMode/top_solutions/matmul_v2/problem_files/shared_eval.py
D	GPUMode/top_solutions/matmul_v2/problem_files/shared_template.py
D	GPUMode/top_solutions/matmul_v2/problem_files/shared_utils.py
D	GPUMode/top_solutions/matmul_v2/problem_files/submission.py
D	GPUMode/top_solutions/matmul_v2/problem_files/task.py
D	GPUMode/top_solutions/matmul_v2/problem_files/task.yml
D	GPUMode/top_solutions/matmul_v2/submissions/top1_A100_submission_780718.py
D	GPUMode/top_solutions/matmul_v2/submissions/top1_B200_submission_773912.py
D	GPUMode/top_solutions/matmul_v2/submissions/top1_H100_submission_512472.py
D	GPUMode/top_solutions/matmul_v2/submissions/top1_L4_submission_780611.py
D	GPUMode/top_solutions/trimul/metadata.json
D	GPUMode/top_solutions/trimul/problem_files/eval.py
D	GPUMode/top_solutions/trimul/problem_files/reference.py
D	GPUMode/top_solutions/trimul/problem_files/submission.py
D	GPUMode/top_solutions/trimul/problem_files/task.py
D	GPUMode/top_solutions/trimul/problem_files/task.yml
D	GPUMode/top_solutions/trimul/problem_files/utils.py
D	GPUMode/top_solutions/trimul/submissions/top1_A100_submission_380716.py
D	GPUMode/top_solutions/trimul/submissions/top1_B200_submission_480316.py
D	GPUMode/top_solutions/trimul/submissions/top1_H100_submission_450489.py
D	GPUMode/top_solutions/trimul/submissions/top1_MI300_submission_34649.py
D	GPUMode/top_solutions/trimul/web_top5/README.md
D	GPUMode/top_solutions/trimul/web_top5/browser_fetch_codes.js
D	GPUMode/top_solutions/trimul/web_top5/metadata.json
D	GPUMode/top_solutions/trimul/web_top5/raw/gpumode496_a100_h100_b200_selected_codes.json
D	GPUMode/top_solutions/trimul/web_top5/submissions/A100/rank1_A100_submission_782275_josusanmartin.py
D	GPUMode/top_solutions/trimul/web_top5/submissions/A100/rank2_A100_submission_781115_rd9000.py
D	GPUMode/top_solutions/trimul/web_top5/submissions/A100/rank3_A100_submission_380716_ttt.py
D	GPUMode/top_solutions/trimul/web_top5/submissions/A100/rank4_A100_submission_781360_brianyu.py
D	GPUMode/top_solutions/trimul/web_top5/submissions/A100/rank5_A100_submission_483089_shiyegao.py
D	GPUMode/top_solutions/trimul/web_top5/submissions/B200/rank1_B200_submission_782380_josusanmartin.py
D	GPUMode/top_solutions/trimul/web_top5/submissions/B200/rank2_B200_submission_480316_shiyegao.py
D	GPUMode/top_solutions/trimul/web_top5/submissions/H100/rank1_H100_submission_782080_josusanmartin.py
D	GPUMode/top_solutions/trimul/web_top5/submissions/H100/rank2_H100_submission_781381_stashuk_olek.py
D	GPUMode/top_solutions/trimul/web_top5/submissions/H100/rank3_H100_submission_450489_shiyegao.py
D	GPUMode/top_solutions/trimul/web_top5/submissions/H100/rank4_H100_submission_408928_zeyu_shen.py
D	GPUMode/top_solutions/trimul/web_top5/submissions/H100/rank5_H100_submission_781100_rd9000.py
D	GPUMode/top_solutions/vectoradd_v2/metadata.json
D	GPUMode/top_solutions/vectoradd_v2/problem_files/reference.py
D	GPUMode/top_solutions/vectoradd_v2/problem_files/shared_eval.py
D	GPUMode/top_solutions/vectoradd_v2/problem_files/shared_template.py
D	GPUMode/top_solutions/vectoradd_v2/problem_files/shared_utils.py
D	GPUMode/top_solutions/vectoradd_v2/problem_files/submission.py
D	GPUMode/top_solutions/vectoradd_v2/problem_files/task.py
D	GPUMode/top_solutions/vectoradd_v2/problem_files/task.yml
D	GPUMode/top_solutions/vectoradd_v2/submissions/top1_A100_submission_779892.py
D	GPUMode/top_solutions/vectoradd_v2/submissions/top1_B200_submission_682384.py
D	GPUMode/top_solutions/vectoradd_v2/submissions/top1_H100_submission_639715.py
D	GPUMode/top_solutions/vectoradd_v2/submissions/top1_L4_submission_607485.py
D	GPUMode/top_solutions/vectorsum_v2/metadata.json
D	GPUMode/top_solutions/vectorsum_v2/problem_files/reference.py
D	GPUMode/top_solutions/vectorsum_v2/problem_files/shared_eval.py
D	GPUMode/top_solutions/vectorsum_v2/problem_files/shared_template.py
D	GPUMode/top_solutions/vectorsum_v2/problem_files/shared_utils.py
D	GPUMode/top_solutions/vectorsum_v2/problem_files/submission.py
D	GPUMode/top_solutions/vectorsum_v2/problem_files/task.py
D	GPUMode/top_solutions/vectorsum_v2/problem_files/task.yml
D	GPUMode/top_solutions/vectorsum_v2/submissions/top1_A100_submission_779823.py
D	GPUMode/top_solutions/vectorsum_v2/submissions/top1_B200_submission_755317.py
D	GPUMode/top_solutions/vectorsum_v2/submissions/top1_H100_submission_612491.py
D	GPUMode/top_solutions/vectorsum_v2/submissions/top1_L4_submission_66749.py
M	examples/ac_inequalities/env.py
D	examples/cap_set_priority/__init__.py
D	examples/cap_set_priority/env.py
M	examples/erdos_min_overlap/env.py
M	examples/gpu_mode/env.py
D	examples/gpu_mode/env_modal.py
M	examples/gpu_mode/lib/libkernelbot/submission.py
M	examples/gpu_mode/prompt.py
D	examples/kakeya/__init__.py
D	examples/kakeya/env.py
M	pyproject.toml
D	repro/__init__.py
D	repro/cap_set/EXPERIMENTS.md
D	repro/cap_set/README.md
D	repro/cap_set/__init__.py
D	repro/cap_set/best_priority_400_codex_20260601.py
D	repro/cap_set/best_priority_431_codex_20260602.py
D	repro/cap_set/best_priority_from_baseline_live.py
D	repro/cap_set/build_initial_pool.py
D	repro/cap_set/initial_pool_400_to_512.json
D	repro/cap_set/initial_priority_296.py
D	repro/cap_set/run_cap_set_codex_experiment.py
D	repro/cap_set/run_cap_set_discovery.py
D	repro/cap_set/self_loop_eval.py
D	repro/erdos/.env.example
D	repro/erdos/README.md
D	repro/erdos/__init__.py
D	repro/erdos/build_initial_pool.py
D	repro/erdos/check_baseline.py
D	repro/erdos/erdos_constructions_comparison.png
D	repro/erdos/initial_pool_reference_plus_codex_20260603.json
D	repro/erdos/reference_dense_final_constructions_overlay.png
D	repro/erdos/reference_final_constructions_from_source_summary.png
D	repro/erdos/run_erdos_codex_experiment.py
D	repro/erdos/run_erdos_discovery.py
D	repro/kakeya/README.md
D	repro/kakeya/run_kakeya_discovery.py
D	repro/kakeya/seeds/initial_cell_with_full_space_fallback.py
D	repro_external/README.md
D	repro_external/alpharesearchcomp/agent_outputs/README.md
D	repro_external/alpharesearchcomp/agent_outputs/run_public_agent_outputs.py
D	repro_external/alpharesearchcomp/agent_outputs/run_results.json
D	repro_external/alpharesearchcomp/agent_outputs/third_autocorrelation_4f4c7847/agent_code.py
D	repro_external/alpharesearchcomp/agent_outputs/third_autocorrelation_e436c26a/agent_code.py
D	repro_external/alpharesearchcomp/runs/MSTD/improved_eval.json
D	repro_external/alpharesearchcomp/runs/MSTD/improvement_notes.md
D	repro_external/alpharesearchcomp/runs/MSTD/initial_eval.json
D	repro_external/alpharesearchcomp/runs/MSTD/source/evaluator.py
D	repro_external/alpharesearchcomp/runs/MSTD/source/initial_program.py
D	repro_external/alpharesearchcomp/runs/MSTD/source/initial_proposal.txt
D	repro_external/alpharesearchcomp/runs/MSTD/work/evaluator.py
D	repro_external/alpharesearchcomp/runs/MSTD/work/improved_program.py
D	repro_external/alpharesearchcomp/runs/MSTD/work/initial_program.py
D	repro_external/alpharesearchcomp/runs/MSTD/work/initial_proposal.txt
D	repro_external/alpharesearchcomp/runs/_common/README.md
D	repro_external/alpharesearchcomp/runs/_common/config_full.yaml
D	repro_external/alpharesearchcomp/runs/_common/config_quick.yaml
D	repro_external/alpharesearchcomp/runs/_common/run_evolve.py
D	repro_external/alpharesearchcomp/runs/_common/run_initial_eval.py
D	repro_external/alpharesearchcomp/runs/_common/run_program_eval.py
D	repro_external/alpharesearchcomp/runs/autoconvolution_peak_minimization/improved_eval.json
D	repro_external/alpharesearchcomp/runs/autoconvolution_peak_minimization/improvement_notes.md
D	repro_external/alpharesearchcomp/runs/autoconvolution_peak_minimization/initial_eval.json
D	repro_external/alpharesearchcomp/runs/autoconvolution_peak_minimization/source/evaluator.py
D	repro_external/alpharesearchcomp/runs/autoconvolution_peak_minimization/source/initial_program.py
D	repro_external/alpharesearchcomp/runs/autoconvolution_peak_minimization/source/initial_proposal.txt
D	repro_external/alpharesearchcomp/runs/autoconvolution_peak_minimization/work/evaluator.py
D	repro_external/alpharesearchcomp/runs/autoconvolution_peak_minimization/work/improved_program.py
D	repro_external/alpharesearchcomp/runs/autoconvolution_peak_minimization/work/initial_program.py
D	repro_external/alpharesearchcomp/runs/autoconvolution_peak_minimization/work/initial_proposal.txt
D	repro_external/alpharesearchcomp/runs/heilbronn_in_the_unit_square/improved_eval.json
D	repro_external/alpharesearchcomp/runs/heilbronn_in_the_unit_square/improvement_notes.md
D	repro_external/alpharesearchcomp/runs/heilbronn_in_the_unit_square/initial_eval.json
D	repro_external/alpharesearchcomp/runs/heilbronn_in_the_unit_square/source/evaluator.py
D	repro_external/alpharesearchcomp/runs/heilbronn_in_the_unit_square/source/initial_program.py
D	repro_external/alpharesearchcomp/runs/heilbronn_in_the_unit_square/source/initial_proposal.txt
D	repro_external/alpharesearchcomp/runs/heilbronn_in_the_unit_square/source/points.npy
D	repro_external/alpharesearchcomp/runs/heilbronn_in_the_unit_square/source/visualization.py
D	repro_external/alpharesearchcomp/runs/heilbronn_in_the_unit_square/work/evaluator.py
D	repro_external/alpharesearchcomp/runs/heilbronn_in_the_unit_square/work/improved_program.py
D	repro_external/alpharesearchcomp/runs/heilbronn_in_the_unit_square/work/initial_program.py
D	repro_external/alpharesearchcomp/runs/heilbronn_in_the_unit_square/work/initial_proposal.txt
D	repro_external/alpharesearchcomp/runs/heilbronn_in_the_unit_square/work/points.npy
D	repro_external/alpharesearchcomp/runs/heilbronn_in_the_unit_square/work/visualization.py
D	repro_external/alpharesearchcomp/runs/improved_eval_summary.json
D	repro_external/alpharesearchcomp/runs/improved_eval_summary.md
D	repro_external/alpharesearchcomp/runs/initial_eval_summary.json
D	repro_external/alpharesearchcomp/runs/kissing_number/improved_eval.json
D	repro_external/alpharesearchcomp/runs/kissing_number/improvement_notes.md
D	repro_external/alpharesearchcomp/runs/kissing_number/initial_eval.json
D	repro_external/alpharesearchcomp/runs/kissing_number/source/evaluator.py
D	repro_external/alpharesearchcomp/runs/kissing_number/source/initial_program.py
D	repro_external/alpharesearchcomp/runs/kissing_number/source/initial_proposal.txt
D	repro_external/alpharesearchcomp/runs/kissing_number/work/evaluator.py
D	repro_external/alpharesearchcomp/runs/kissing_number/work/improved_program.py
D	repro_external/alpharesearchcomp/runs/kissing_number/work/initial_program.py
D	repro_external/alpharesearchcomp/runs/kissing_number/work/initial_proposal.txt
D	repro_external/alpharesearchcomp/runs/littlewood_polynomials/improvement_notes.md
D	repro_external/alpharesearchcomp/runs/littlewood_polynomials/initial_eval.json
D	repro_external/alpharesearchcomp/runs/littlewood_polynomials/source/evaluator.py
D	repro_external/alpharesearchcomp/runs/littlewood_polynomials/source/initial_program.py
D	repro_external/alpharesearchcomp/runs/littlewood_polynomials/source/initial_proposal.txt
D	repro_external/alpharesearchcomp/runs/littlewood_polynomials/work/evaluator.py
D	repro_external/alpharesearchcomp/runs/littlewood_polynomials/work/initial_program.py
D	repro_external/alpharesearchcomp/runs/littlewood_polynomials/work/initial_proposal.txt
D	repro_external/alpharesearchcomp/runs/minizing_raio_max_min_distance/improved_eval.json
D	repro_external/alpharesearchcomp/runs/minizing_raio_max_min_distance/improvement_notes.md
D	repro_external/alpharesearchcomp/runs/minizing_raio_max_min_distance/initial_eval.json
D	repro_external/alpharesearchcomp/runs/minizing_raio_max_min_distance/source/evaluator.py
D	repro_external/alpharesearchcomp/runs/minizing_raio_max_min_distance/source/initial_program.py
D	repro_external/alpharesearchcomp/runs/minizing_raio_max_min_distance/source/initial_proposal.txt
D	repro_external/alpharesearchcomp/runs/minizing_raio_max_min_distance/work/evaluator.py
D	repro_external/alpharesearchcomp/runs/minizing_raio_max_min_distance/work/improved_program.py
D	repro_external/alpharesearchcomp/runs/minizing_raio_max_min_distance/work/initial_program.py
D	repro_external/alpharesearchcomp/runs/minizing_raio_max_min_distance/work/initial_proposal.txt
D	repro_external/alpharesearchcomp/runs/packing_circles/improved_eval.json
D	repro_external/alpharesearchcomp/runs/packing_circles/improved_eval_rerun.json
D	repro_external/alpharesearchcomp/runs/packing_circles/improvement_notes.md
D	repro_external/alpharesearchcomp/runs/packing_circles/initial_eval.json
D	repro_external/alpharesearchcomp/runs/packing_circles/source/evaluator.py
D	repro_external/alpharesearchcomp/runs/packing_circles/source/initial_program.py
D	repro_external/alpharesearchcomp/runs/packing_circles/source/initial_proposal.txt
D	repro_external/alpharesearchcomp/runs/packing_circles/work/evaluator.py
D	repro_external/alpharesearchcomp/runs/packing_circles/work/improved_eval_check.json
D	repro_external/alpharesearchcomp/runs/packing_circles/work/improved_program.py
D	repro_external/alpharesearchcomp/runs/packing_circles/work/initial_program.py
D	repro_external/alpharesearchcomp/runs/packing_circles/work/initial_proposal.txt
D	repro_external/alpharesearchcomp/runs/spherical_code/improved_eval.json
D	repro_external/alpharesearchcomp/runs/spherical_code/improvement_notes.md
D	repro_external/alpharesearchcomp/runs/spherical_code/initial_eval.json
D	repro_external/alpharesearchcomp/runs/spherical_code/source/evaluator.py
D	repro_external/alpharesearchcomp/runs/spherical_code/source/initial_program.py
D	repro_external/alpharesearchcomp/runs/spherical_code/source/initial_proposal.txt
D	repro_external/alpharesearchcomp/runs/spherical_code/source/visualization.py
D	repro_external/alpharesearchcomp/runs/spherical_code/work/evaluator.py
D	repro_external/alpharesearchcomp/runs/spherical_code/work/improved_program.py
D	repro_external/alpharesearchcomp/runs/spherical_code/work/initial_program.py
D	repro_external/alpharesearchcomp/runs/spherical_code/work/initial_proposal.txt
D	repro_external/alpharesearchcomp/runs/spherical_code/work/visualization.py
D	repro_external/alpharesearchcomp/runs/third_autocorrelation_inequality/improved_eval.json
D	repro_external/alpharesearchcomp/runs/third_autocorrelation_inequality/improvement_notes.md
D	repro_external/alpharesearchcomp/runs/third_autocorrelation_inequality/initial_eval.json
D	repro_external/alpharesearchcomp/runs/third_autocorrelation_inequality/source/evaluator.py
D	repro_external/alpharesearchcomp/runs/third_autocorrelation_inequality/source/initial_program.py
D	repro_external/alpharesearchcomp/runs/third_autocorrelation_inequality/source/initial_proposal.txt
D	repro_external/alpharesearchcomp/runs/third_autocorrelation_inequality/work/evaluator.py
D	repro_external/alpharesearchcomp/runs/third_autocorrelation_inequality/work/improved_program.py
D	repro_external/alpharesearchcomp/runs/third_autocorrelation_inequality/work/initial_program.py
D	repro_external/alpharesearchcomp/runs/third_autocorrelation_inequality/work/initial_proposal.txt
D	repro_external/autocorrelation_autoconvolution/README.md
D	repro_external/autocorrelation_autoconvolution/ac_results.json
D	repro_external/autocorrelation_autoconvolution/official_ac1_evolved_search_result.json
D	repro_external/autocorrelation_autoconvolution/reproduce_ac.py
D	repro_external/autocorrelation_autoconvolution/run_official_ac1_evolved_search.py
D	repro_external/cap_set_priority_function/README.md
D	repro_external/cap_set_priority_function/cap_set_results.json
D	repro_external/cap_set_priority_function/reproduce_cap_set.py
D	repro_external/kakeya_construction/README.md
D	repro_external/kakeya_construction/alphaevolve_notebook_program_results.json
D	repro_external/kakeya_construction/finite_field_kakeya_results.json
D	repro_external/kakeya_construction/idea_reconstruction/README.md
D	repro_external/kakeya_construction/idea_reconstruction/compare_reconstructions.py
D	repro_external/kakeya_construction/idea_reconstruction/comparison_results.json
D	repro_external/kakeya_construction/idea_reconstruction/exp1_reconstructed.py
D	repro_external/kakeya_construction/idea_reconstruction/exp2_reconstructed.py
D	repro_external/kakeya_construction/idea_reconstruction/exp3_reconstructed.py
D	repro_external/kakeya_construction/reproduce_alphaevolve_notebook_programs.py
D	repro_external/kakeya_construction/reproduce_finite_field_kakeya.py
M	ttt_discover/__init__.py
D	ttt_discover/codex_utils/__init__.py
D	ttt_discover/codex_utils/completers.py
D	ttt_discover/codex_utils/discovery.py
D	ttt_discover/codex_utils/environment.py
D	ttt_discover/codex_utils/runtime.py
D	ttt_discover/codex_utils/sampler.py
M	ttt_discover/discovery.py
D	ttt_discover/rl/codex_no_finetune.py
M	ttt_discover/rl/train.py
M	ttt_discover/tinker_utils/completers.py
M	ttt_discover/tinker_utils/sampler.py
````
## Diff Stat

### Committed Diff Stat Compared To Base

````text
 .gitignore                                         |    16 -
 .gitmodules                                        |     3 -
 GPUMode/a100_only/README.md                        |    19 -
 GPUMode/a100_only/insights/matmul_v2.md            |     1 -
 GPUMode/a100_only/insights/trimul.md               |     1 -
 GPUMode/a100_only/insights/vectoradd_v2.md         |     1 -
 GPUMode/a100_only/insights/vectorsum_v2.md         |     1 -
 .../reports/reproduction_quality_review.md         |    83 -
 GPUMode/a100_only/reports/summary.md               |    16 -
 .../a100_only/reports/verification_results.json    |    94 -
 GPUMode/a100_only/reports/verify_a100_only.py      |   206 -
 .../reproductions/matmul_v2/submission.py          |   151 -
 .../a100_only/reproductions/trimul/submission.py   |    72 -
 .../reproductions/vectoradd_v2/submission.py       |   123 -
 .../reproductions/vectorsum_v2/submission.py       |   121 -
 GPUMode/insights/matmul_v2.md                      |    84 -
 GPUMode/insights/trimul.md                         |   100 -
 GPUMode/insights/vectoradd_v2.md                   |    63 -
 GPUMode/insights/vectorsum_v2.md                   |    70 -
 GPUMode/problem_index.md                           |    63 -
 GPUMode/reference-kernels                          |     1 -
 GPUMode/reproduction_reports/README.md             |    14 -
 GPUMode/reproduction_reports/final_comparison.md   |    23 -
 .../reproduction_reports/semantic_equivalence.md   |   128 -
 .../reproduction_reports/verification_results.json |   140 -
 .../reproduction_reports/verify_reproductions.py   |   263 -
 GPUMode/reproductions/matmul_v2/NOTES.md           |    20 -
 GPUMode/reproductions/matmul_v2/submission.py      |    42 -
 GPUMode/reproductions/trimul/NOTES.md              |    22 -
 GPUMode/reproductions/trimul/submission.py         |    97 -
 GPUMode/reproductions/vectoradd_v2/NOTES.md        |    21 -
 GPUMode/reproductions/vectoradd_v2/submission.py   |    75 -
 GPUMode/reproductions/vectorsum_v2/NOTES.md        |    25 -
 GPUMode/reproductions/vectorsum_v2/submission.py   |   300 -
 GPUMode/top_solutions/README.md                    |    26 -
 GPUMode/top_solutions/matmul_v2/metadata.json      |    53 -
 .../matmul_v2/problem_files/reference.py           |    23 -
 .../matmul_v2/problem_files/shared_eval.py         |   375 -
 .../matmul_v2/problem_files/shared_template.py     |     5 -
 .../matmul_v2/problem_files/shared_utils.py        |   176 -
 .../matmul_v2/problem_files/submission.py          |     6 -
 .../top_solutions/matmul_v2/problem_files/task.py  |    11 -
 .../top_solutions/matmul_v2/problem_files/task.yml |    44 -
 .../submissions/top1_A100_submission_780718.py     |   155 -
 .../submissions/top1_B200_submission_773912.py     |   264 -
 .../submissions/top1_H100_submission_512472.py     |     9 -
 .../submissions/top1_L4_submission_780611.py       |   132 -
 GPUMode/top_solutions/trimul/metadata.json         |    52 -
 GPUMode/top_solutions/trimul/problem_files/eval.py |   384 -
 .../trimul/problem_files/reference.py              |   168 -
 .../trimul/problem_files/submission.py             |    98 -
 GPUMode/top_solutions/trimul/problem_files/task.py |    14 -
 .../top_solutions/trimul/problem_files/task.yml    |    72 -
 .../top_solutions/trimul/problem_files/utils.py    |   168 -
 .../submissions/top1_A100_submission_380716.py     |   462 -
 .../submissions/top1_B200_submission_480316.py     |  1413 -
 .../submissions/top1_H100_submission_450489.py     |  2245 -
 .../submissions/top1_MI300_submission_34649.py     |   514 -
 GPUMode/top_solutions/trimul/web_top5/README.md    |    47 -
 .../trimul/web_top5/browser_fetch_codes.js         |    26 -
 .../top_solutions/trimul/web_top5/metadata.json    |   258 -
 .../gpumode496_a100_h100_b200_selected_codes.json  |    80 -
 .../rank1_A100_submission_782275_josusanmartin.py  |  2751 -
 .../A100/rank2_A100_submission_781115_rd9000.py    |   287 -
 .../A100/rank3_A100_submission_380716_ttt.py       |   462 -
 .../A100/rank4_A100_submission_781360_brianyu.py   |  2188 -
 .../A100/rank5_A100_submission_483089_shiyegao.py  |  5127 --
 .../rank1_B200_submission_782380_josusanmartin.py  |  2296 -
 .../B200/rank2_B200_submission_480316_shiyegao.py  |  1413 -
 .../rank1_H100_submission_782080_josusanmartin.py  |  2754 -
 .../rank2_H100_submission_781381_stashuk_olek.py   |   241 -
 .../H100/rank3_H100_submission_450489_shiyegao.py  |  2245 -
 .../H100/rank4_H100_submission_408928_zeyu_shen.py |   212 -
 .../H100/rank5_H100_submission_781100_rd9000.py    |   278 -
 GPUMode/top_solutions/vectoradd_v2/metadata.json   |    53 -
 .../vectoradd_v2/problem_files/reference.py        |    38 -
 .../vectoradd_v2/problem_files/shared_eval.py      |   375 -
 .../vectoradd_v2/problem_files/shared_template.py  |     5 -
 .../vectoradd_v2/problem_files/shared_utils.py     |   176 -
 .../vectoradd_v2/problem_files/submission.py       |     7 -
 .../vectoradd_v2/problem_files/task.py             |    11 -
 .../vectoradd_v2/problem_files/task.yml            |    41 -
 .../submissions/top1_A100_submission_779892.py     |   147 -
 .../submissions/top1_B200_submission_682384.py     |   124 -
 .../submissions/top1_H100_submission_639715.py     |    68 -
 .../submissions/top1_L4_submission_607485.py       |    69 -
 GPUMode/top_solutions/vectorsum_v2/metadata.json   |    53 -
 .../vectorsum_v2/problem_files/reference.py        |    55 -
 .../vectorsum_v2/problem_files/shared_eval.py      |   375 -
 .../vectorsum_v2/problem_files/shared_template.py  |     5 -
 .../vectorsum_v2/problem_files/shared_utils.py     |   176 -
 .../vectorsum_v2/problem_files/submission.py       |    62 -
 .../vectorsum_v2/problem_files/task.py             |     9 -
 .../vectorsum_v2/problem_files/task.yml            |    41 -
 .../submissions/top1_A100_submission_779823.py     |   166 -
 .../submissions/top1_B200_submission_755317.py     |    76 -
 .../submissions/top1_H100_submission_612491.py     |   234 -
 .../submissions/top1_L4_submission_66749.py        |    76 -
 examples/ac_inequalities/env.py                    |    12 -
 examples/cap_set_priority/__init__.py              |     2 -
 examples/cap_set_priority/env.py                   |   292 -
 examples/erdos_min_overlap/env.py                  |    84 +-
 examples/gpu_mode/env.py                           |   513 +-
 examples/gpu_mode/env_modal.py                     |   237 -
 examples/gpu_mode/lib/libkernelbot/submission.py   |    10 +-
 examples/gpu_mode/prompt.py                        |     8 +-
 examples/kakeya/__init__.py                        |     2 -
 examples/kakeya/env.py                             |   402 -
 pyproject.toml                                     |     1 -
 repro/__init__.py                                  |     1 -
 repro/cap_set/EXPERIMENTS.md                       |    37 -
 repro/cap_set/README.md                            |    23 -
 repro/cap_set/__init__.py                          |     2 -
 repro/cap_set/best_priority_400_codex_20260601.py  |    82 -
 repro/cap_set/best_priority_431_codex_20260602.py  |   370 -
 repro/cap_set/best_priority_from_baseline_live.py  |   333 -
 repro/cap_set/build_initial_pool.py                |   237 -
 repro/cap_set/initial_pool_400_to_512.json         | 59432 -------------------
 repro/cap_set/initial_priority_296.py              |    46 -
 repro/cap_set/run_cap_set_codex_experiment.py      |    98 -
 repro/cap_set/run_cap_set_discovery.py             |   125 -
 repro/cap_set/self_loop_eval.py                    |    52 -
 repro/erdos/.env.example                           |     8 -
 repro/erdos/README.md                              |   149 -
 repro/erdos/__init__.py                            |     1 -
 repro/erdos/build_initial_pool.py                  |   287 -
 repro/erdos/check_baseline.py                      |    77 -
 repro/erdos/erdos_constructions_comparison.png     |   Bin 177349 -> 0 bytes
 ...initial_pool_reference_plus_codex_20260603.json | 20963 -------
 ...reference_dense_final_constructions_overlay.png |   Bin 164719 -> 0 bytes
 ...nce_final_constructions_from_source_summary.png |   Bin 313351 -> 0 bytes
 repro/erdos/run_erdos_codex_experiment.py          |   130 -
 repro/erdos/run_erdos_discovery.py                 |   121 -
 repro/kakeya/README.md                             |    47 -
 repro/kakeya/run_kakeya_discovery.py               |   134 -
 .../seeds/initial_cell_with_full_space_fallback.py |    32 -
 repro_external/README.md                           |     7 -
 .../alpharesearchcomp/agent_outputs/README.md      |    16 -
 .../agent_outputs/run_public_agent_outputs.py      |    56 -
 .../agent_outputs/run_results.json                 |    28 -
 .../third_autocorrelation_4f4c7847/agent_code.py   |   109 -
 .../third_autocorrelation_e436c26a/agent_code.py   |    92 -
 .../alpharesearchcomp/runs/MSTD/improved_eval.json |    13 -
 .../runs/MSTD/improvement_notes.md                 |    32 -
 .../alpharesearchcomp/runs/MSTD/initial_eval.json  |    13 -
 .../runs/MSTD/source/evaluator.py                  |    77 -
 .../runs/MSTD/source/initial_program.py            |    20 -
 .../runs/MSTD/source/initial_proposal.txt          |    21 -
 .../alpharesearchcomp/runs/MSTD/work/evaluator.py  |    77 -
 .../runs/MSTD/work/improved_program.py             |    23 -
 .../runs/MSTD/work/initial_program.py              |    20 -
 .../runs/MSTD/work/initial_proposal.txt            |    21 -
 .../alpharesearchcomp/runs/_common/README.md       |    26 -
 .../runs/_common/config_full.yaml                  |    62 -
 .../runs/_common/config_quick.yaml                 |    62 -
 .../alpharesearchcomp/runs/_common/run_evolve.py   |   104 -
 .../runs/_common/run_initial_eval.py               |   136 -
 .../runs/_common/run_program_eval.py               |    95 -
 .../improved_eval.json                             |    12 -
 .../improvement_notes.md                           |    36 -
 .../initial_eval.json                              |    12 -
 .../source/evaluator.py                            |    66 -
 .../source/initial_program.py                      |    80 -
 .../source/initial_proposal.txt                    |    34 -
 .../work/evaluator.py                              |    66 -
 .../work/improved_program.py                       |    83 -
 .../work/initial_program.py                        |    80 -
 .../work/initial_proposal.txt                      |    34 -
 .../improved_eval.json                             |    15 -
 .../improvement_notes.md                           |    40 -
 .../heilbronn_in_the_unit_square/initial_eval.json |    15 -
 .../source/evaluator.py                            |    88 -
 .../source/initial_program.py                      |   346 -
 .../source/initial_proposal.txt                    |    17 -
 .../heilbronn_in_the_unit_square/source/points.npy |   Bin 384 -> 0 bytes
 .../source/visualization.py                        |   198 -
 .../heilbronn_in_the_unit_square/work/evaluator.py |    88 -
 .../work/improved_program.py                       |    40 -
 .../work/initial_program.py                        |   346 -
 .../work/initial_proposal.txt                      |    17 -
 .../heilbronn_in_the_unit_square/work/points.npy   |   Bin 384 -> 0 bytes
 .../work/visualization.py                          |   198 -
 .../runs/improved_eval_summary.json                |   139 -
 .../runs/improved_eval_summary.md                  |    13 -
 .../runs/initial_eval_summary.json                 |    96 -
 .../runs/kissing_number/improved_eval.json         |    14 -
 .../runs/kissing_number/improvement_notes.md       |    23 -
 .../runs/kissing_number/initial_eval.json          |    14 -
 .../runs/kissing_number/source/evaluator.py        |    99 -
 .../runs/kissing_number/source/initial_program.py  |   479 -
 .../kissing_number/source/initial_proposal.txt     |    11 -
 .../runs/kissing_number/work/evaluator.py          |    99 -
 .../runs/kissing_number/work/improved_program.py   |  1816 -
 .../runs/kissing_number/work/initial_program.py    |   479 -
 .../runs/kissing_number/work/initial_proposal.txt  |    11 -
 .../littlewood_polynomials/improvement_notes.md    |   105 -
 .../runs/littlewood_polynomials/initial_eval.json  |    12 -
 .../littlewood_polynomials/source/evaluator.py     |    79 -
 .../source/initial_program.py                      |    34 -
 .../source/initial_proposal.txt                    |    16 -
 .../runs/littlewood_polynomials/work/evaluator.py  |    85 -
 .../littlewood_polynomials/work/initial_program.py |    34 -
 .../work/initial_proposal.txt                      |    16 -
 .../improved_eval.json                             |    14 -
 .../improvement_notes.md                           |    35 -
 .../initial_eval.json                              |    14 -
 .../source/evaluator.py                            |    53 -
 .../source/initial_program.py                      |   188 -
 .../source/initial_proposal.txt                    |     6 -
 .../work/evaluator.py                              |    63 -
 .../work/improved_program.py                       |    78 -
 .../work/initial_program.py                        |   188 -
 .../work/initial_proposal.txt                      |     6 -
 .../runs/packing_circles/improved_eval.json        |    14 -
 .../runs/packing_circles/improved_eval_rerun.json  |    14 -
 .../runs/packing_circles/improvement_notes.md      |    31 -
 .../runs/packing_circles/initial_eval.json         |    14 -
 .../runs/packing_circles/source/evaluator.py       |    73 -
 .../runs/packing_circles/source/initial_program.py |   366 -
 .../packing_circles/source/initial_proposal.txt    |    13 -
 .../runs/packing_circles/work/evaluator.py         |    82 -
 .../packing_circles/work/improved_eval_check.json  |    12 -
 .../runs/packing_circles/work/improved_program.py  |   133 -
 .../runs/packing_circles/work/initial_program.py   |   366 -
 .../runs/packing_circles/work/initial_proposal.txt |    13 -
 .../runs/spherical_code/improved_eval.json         |    15 -
 .../runs/spherical_code/improvement_notes.md       |    19 -
 .../runs/spherical_code/initial_eval.json          |    15 -
 .../runs/spherical_code/source/evaluator.py        |    57 -
 .../runs/spherical_code/source/initial_program.py  |    68 -
 .../spherical_code/source/initial_proposal.txt     |    14 -
 .../runs/spherical_code/source/visualization.py    |   122 -
 .../runs/spherical_code/work/evaluator.py          |    57 -
 .../runs/spherical_code/work/improved_program.py   |    58 -
 .../runs/spherical_code/work/initial_program.py    |    68 -
 .../runs/spherical_code/work/initial_proposal.txt  |    14 -
 .../runs/spherical_code/work/visualization.py      |   122 -
 .../improved_eval.json                             |    12 -
 .../improvement_notes.md                           |    28 -
 .../initial_eval.json                              |    12 -
 .../source/evaluator.py                            |    26 -
 .../source/initial_program.py                      |    99 -
 .../source/initial_proposal.txt                    |     9 -
 .../work/evaluator.py                              |    34 -
 .../work/improved_program.py                       |   273 -
 .../work/initial_program.py                        |    99 -
 .../work/initial_proposal.txt                      |     9 -
 .../autocorrelation_autoconvolution/README.md      |    33 -
 .../ac_results.json                                |    33 -
 .../official_ac1_evolved_search_result.json        |   307 -
 .../reproduce_ac.py                                |   188 -
 .../run_official_ac1_evolved_search.py             |   104 -
 repro_external/cap_set_priority_function/README.md |    21 -
 .../cap_set_priority_function/cap_set_results.json |    36 -
 .../cap_set_priority_function/reproduce_cap_set.py |   232 -
 repro_external/kakeya_construction/README.md       |    36 -
 .../alphaevolve_notebook_program_results.json      |   114 -
 .../finite_field_kakeya_results.json               |    20 -
 .../idea_reconstruction/README.md                  |   103 -
 .../idea_reconstruction/compare_reconstructions.py |   187 -
 .../idea_reconstruction/comparison_results.json    |   212 -
 .../idea_reconstruction/exp1_reconstructed.py      |   100 -
 .../idea_reconstruction/exp2_reconstructed.py      |   162 -
 .../idea_reconstruction/exp3_reconstructed.py      |   186 -
 .../reproduce_alphaevolve_notebook_programs.py     |   188 -
 .../reproduce_finite_field_kakeya.py               |   128 -
 ttt_discover/__init__.py                           |    40 +-
 ttt_discover/codex_utils/__init__.py               |    31 -
 ttt_discover/codex_utils/completers.py             |   270 -
 ttt_discover/codex_utils/discovery.py              |   105 -
 ttt_discover/codex_utils/environment.py            |   123 -
 ttt_discover/codex_utils/runtime.py                |   130 -
 ttt_discover/codex_utils/sampler.py                |   790 -
 ttt_discover/discovery.py                          |   103 +-
 ttt_discover/rl/codex_no_finetune.py               |  1020 -
 ttt_discover/rl/train.py                           |    23 -
 ttt_discover/tinker_utils/completers.py            |   243 +-
 ttt_discover/tinker_utils/sampler.py               |    13 +-
 278 files changed, 181 insertions(+), 131586 deletions(-)
````
## Raw Diff

Committed patch omitted because it is very large or is baseline history rather than this branch implementation. Use the file/status sections above for the recorded diff shape.
