# GPUMode Top Solutions

Source data: `GPUMODE/kernelbot-data` Hugging Face dataset, local parquet files under `../data/kernelbot-data`.

Selection rule: `mode == leaderboard`, `passed == True`, non-null `score` and `code`, then lowest score per `runner`. Scores are seconds; lower is better. Do not compare scores across different GPU runners as if they were one leaderboard.

| Problem | Runner | Submission | User | Score | File |
| --- | --- | ---: | --- | ---: | --- |
| `matmul_v2` | `A100` | `780718` | brianyu | 627.712 us | `matmul_v2/submissions/top1_A100_submission_780718.py` |
| `matmul_v2` | `B200` | `773912` | olezhka_007 | 106.933 us | `matmul_v2/submissions/top1_B200_submission_773912.py` |
| `matmul_v2` | `H100` | `512472` | iharryli | 219.053 us | `matmul_v2/submissions/top1_H100_submission_512472.py` |
| `matmul_v2` | `L4` | `780611` | brianyu | 2076.331 us | `matmul_v2/submissions/top1_L4_submission_780611.py` |
| `trimul` | `A100` | `380716` | TTT | 2198.190 us | `trimul/submissions/top1_A100_submission_380716.py` |
| `trimul` | `B200` | `480316` | shiyegao | 554.395 us | `trimul/submissions/top1_B200_submission_480316.py` |
| `trimul` | `H100` | `450489` | shiyegao | 1074.354 us | `trimul/submissions/top1_H100_submission_450489.py` |
| `trimul` | `MI300` | `34649` | Arseni Ivanov | 2657.275 us | `trimul/submissions/top1_MI300_submission_34649.py` |
| `vectoradd_v2` | `A100` | `779892` | Kernel-Zhang | 891.904 us | `vectoradd_v2/submissions/top1_A100_submission_779892.py` |
| `vectoradd_v2` | `B200` | `682384` | ngolhn | 232.459 us | `vectoradd_v2/submissions/top1_B200_submission_682384.py` |
| `vectoradd_v2` | `H100` | `639715` | KernelAgent | 523.056 us | `vectoradd_v2/submissions/top1_H100_submission_639715.py` |
| `vectoradd_v2` | `L4` | `607485` | bigpeach | 6266.539 us | `vectoradd_v2/submissions/top1_L4_submission_607485.py` |
| `vectorsum_v2` | `A100` | `779823` | Kernel-Zhang | 135.339 us | `vectorsum_v2/submissions/top1_A100_submission_779823.py` |
| `vectorsum_v2` | `B200` | `755317` | dannywillowliu-uchi | 40.768 us | `vectorsum_v2/submissions/top1_B200_submission_755317.py` |
| `vectorsum_v2` | `H100` | `612491` | Pouya Hamadanian | 77.316 us | `vectorsum_v2/submissions/top1_H100_submission_612491.py` |
| `vectorsum_v2` | `L4` | `66749` | Saint of the Famished | 864.768 us | `vectorsum_v2/submissions/top1_L4_submission_66749.py` |

Problem scaffolds are copied into each `problem_files/` directory from `../reference-kernels/problems` for offline inspection.
