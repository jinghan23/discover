# GPUMode TriMul Current Web Top5

Source: `https://www.gpumode.com/api/leaderboard/496`, current web rankings, not the Hugging Face offline export.

Raw authenticated code payload: `raw/gpumode496_a100_h100_b200_selected_codes.json`.

Extracted code is available for A100 top5, H100 top5, and B200 top2 under `submissions/`.

## A100

| Rank | Submission | User | Score | Time | File name | Code |
| ---: | ---: | --- | ---: | --- | --- | --- |
| 1 | 782275 | josusanmartin | 2113.832 us | 2026-05-10T12:18:43.139979+00:00 | `submission_a100_hybrid_cublas_dim384n1024_nodecor2.py` | `submissions/A100/rank1_A100_submission_782275_josusanmartin.py` |
| 2 | 781115 | rd9000 | 2172.734 us | 2026-05-07T19:37:22.262872+00:00 | `submission_a100.py` | `submissions/A100/rank2_A100_submission_781115_rd9000.py` |
| 3 | 380716 | TTT | 2198.190 us | 2026-01-19T09:13:03.617804+00:00 | `TTT_A100.py` | `submissions/A100/rank3_A100_submission_380716_ttt.py` |
| 4 | 781360 | brianyu | 2267.563 us | 2026-05-08T10:19:04.873921+00:00 | `submission.py` | `submissions/A100/rank4_A100_submission_781360_brianyu.py` |
| 5 | 483089 | shiyegao | 2272.951 us | 2026-02-07T07:16:28.988999+00:00 | `submission.py` | `submissions/A100/rank5_A100_submission_483089_shiyegao.py` |

## H100

| Rank | Submission | User | Score | Time | File name | Code |
| ---: | ---: | --- | ---: | --- | --- | --- |
| 1 | 782080 | josusanmartin | 1042.440 us | 2026-05-10T07:54:09.844807+00:00 | `submission_h100_hybrid_a100mathbuf_select.py` | `submissions/H100/rank1_H100_submission_782080_josusanmartin.py` |
| 2 | 781381 | stashuk-olek | 1048.881 us | 2026-05-08T16:57:07.735959+00:00 | `submission_reworked.py` | `submissions/H100/rank2_H100_submission_781381_stashuk_olek.py` |
| 3 | 450489 | shiyegao | 1074.354 us | 2026-02-04T10:50:27.701308+00:00 | `submission.py` | `submissions/H100/rank3_H100_submission_450489_shiyegao.py` |
| 4 | 408928 | Zeyu Shen | 1139.970 us | 2026-01-29T08:15:51.646259+00:00 | `triton_kernels_v44.py` | `submissions/H100/rank4_H100_submission_408928_zeyu_shen.py` |
| 5 | 781100 | rd9000 | 1147.349 us | 2026-05-07T19:27:47.643531+00:00 | `submission.py` | `submissions/H100/rank5_H100_submission_781100_rd9000.py` |

## B200

| Rank | Submission | User | Score | Time | File name | Code |
| ---: | ---: | --- | ---: | --- | --- | --- |
| 1 | 782380 | josusanmartin | 553.982 us | 2026-05-10T16:42:19.812299+00:00 | `submission_b200_mixedgate_ltcontract_fp16acc_ltproj16_h1d128_staticext_intmask_pythin_floatfastall_ltfinal_h0_sbalgo0.py` | `submissions/B200/rank1_B200_submission_782380_josusanmartin.py` |
| 2 | 480316 | shiyegao | 554.395 us | 2026-02-06T19:09:53.294899+00:00 | `submission.py` | `submissions/B200/rank2_B200_submission_480316_shiyegao.py` |
| 3 | 415223 | novo_force | 589.386 us | 2026-01-30T16:26:58.33618+00:00 | `submission.py` |  |
| 4 | 781147 | ajay_a | 598.881 us | 2026-05-07T20:04:48.043921+00:00 | `submission.py` |  |
| 5 | 781740 | rd9000 | 800.624 us | 2026-05-09T20:58:30.222965+00:00 | `submission_b200.py` |  |

## MI300

| Rank | Submission | User | Score | Time | File name | Code |
| ---: | ---: | --- | ---: | --- | --- | --- |
| 1 | 34649 | Arseni Ivanov | 2657.275 us | 2025-09-02T06:04:33.205639+00:00 | `triton_fully_fused_branched_pt.py` |  |
| 2 | 42921 | msuiche | 5364.367 us | 2025-09-23T12:23:57.309757+00:00 | `submission_baseline_tuned.py` |  |
| 3 | 34219 | Apeirogon | 5647.626 us | 2025-08-24T05:18:13.869166+00:00 | `no_compile.py` |  |
| 4 | 33826 | Waqar | 7180.072 us | 2025-08-16T10:57:12.874014+00:00 | `submission.py` |  |
| 5 | 32750 | Dante | 7827.614 us | 2025-06-27T09:37:52.246389+00:00 | `submission.py` |  |
