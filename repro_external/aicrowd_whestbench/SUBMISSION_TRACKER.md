# WhestBench Submission Tracker

Generated at `2026-07-16T16:20:00+08:00` from the local registry and the AIcrowd API.
Lower adjusted score is better. This file is generated; do not edit it by hand.

## Summary

- registered submissions: **25**; graded with an official score: **24**
- best official result among registered submissions: [#316192](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316192) German reproduction, 13% budget at `2.787552892e-07`
- official score is measured on the hosted public-50 suite; local rows use either mini first-10 or mini full-100 and are not interchangeable
- protocol and column glossary: [`docs/PROTOCOLS.md`](docs/PROTOCOLS.md)

## Local-to-official drift

`Delta` is `(official / local - 1) * 100%`; negative means the official public score was lower (better). Correlations measure ordering, not absolute agreement.

| Local protocol | Submissions | Mean delta | Median abs delta | Pearson r | Spearman rho |
|---|---:|---:|---:|---:|---:|
| mini n=100 | 11 | -9.922% | 8.019% | 0.892 | 0.609 |
| mini n=10 | 13 | 9.783% | 12.240% | 0.887 | 0.571 |

### Drift by protocol and family

| Local protocol | Family | Submissions | Mean delta | Median abs delta | Spearman rho |
|---|---|---:|---:|---:|---:|
| mini n=100 | autoevolve | 8 | -6.675% | 7.628% | 0.905 |
| mini n=100 | german | 1 | -24.056% | 24.056% | - |
| mini n=100 | hybrid_mc | 1 | -5.310% | 5.310% | - |
| mini n=100 | rqmc | 1 | -26.372% | 26.372% | - |
| mini n=10 | german | 10 | 16.179% | 13.619% | 0.055 |
| mini n=10 | rqmc | 3 | -11.538% | 10.173% | 1.000 |

## All submissions

Official `MSE` is AIcrowd's `score_secondary` for these Phase 1 submissions. `S/MSE` is the ratio of the two hosted aggregates; treat it as an implied compute multiplier, not an exact mean C/B.

| ID | Method | Local n | Local adjusted | Local MSE | Local C/B | Fail | Official adjusted | Official MSE | S/MSE | Delta | Status |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| [#316181](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316181) | RQMC Rao-Blackwell, original budget | 100 | `4.885414e-07` | `1.153299e-06` | 0.4236 | 0 | `3.597021e-07` | `8.558566e-07` | 0.4203 | -26.37% | graded |
| [#316182](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316182) | AutoEvolve smoke candidate | 100 | `5.896640e-07` | `5.881288e-06` | 0.1003 | 0 | `4.346721e-07` | `4.231416e-06` | 0.1027 | -26.28% | graded |
| [#316183](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316183) | German whitened-antithetic reproduction | 100 | `3.834192e-07` | `2.465925e-06` | 0.1555 | 0 | `2.911839e-07` | `1.859657e-06` | 0.1566 | -24.06% | graded |
| [#316184](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316184) | RQMC Rao-Blackwell, 30% budget | 10 | `3.997607e-07` | `1.309560e-06` | 0.3052 | 0 | `3.262642e-07` | `1.079888e-06` | 0.3021 | -18.39% | graded |
| [#316185](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316185) | RQMC Rao-Blackwell, 55% budget | 10 | `4.147758e-07` | `7.464828e-07` | 0.5558 | 0 | `3.896613e-07` | `7.063287e-07` | 0.5517 | -6.05% | graded |
| [#316186](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316186) | RQMC Rao-Blackwell, 70% budget | 10 | `4.060269e-07` | `5.740656e-07` | 0.7073 | 0 | `3.647210e-07` | `5.202637e-07` | 0.7010 | -10.17% | graded |
| [#316188](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316188) | German reproduction, 10% budget | 10 | `2.314163e-07` | `2.303840e-06` | 0.1004 | 0 | `3.058597e-07` | `2.996417e-06` | 0.1021 | 32.17% | graded |
| [#316189](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316189) | German reproduction, 12% budget | 10 | `2.511563e-07` | `2.084153e-06` | 0.1205 | 0 | `2.818968e-07` | `2.312305e-06` | 0.1219 | 12.24% | graded |
| [#316190](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316190) | German reproduction, 20% budget | 10 | `2.540605e-07` | `1.267042e-06` | 0.2004 | 0 | `2.831611e-07` | `1.402579e-06` | 0.2019 | 11.45% | graded |
| [#316191](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316191) | German reproduction, 25% budget | 10 | `2.865727e-07` | `1.143270e-06` | 0.2506 | 0 | `3.197294e-07` | `1.268281e-06` | 0.2521 | 11.57% | graded |
| [#316192](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316192) | German reproduction, 13% budget | 10 | `2.492188e-07` | `1.909508e-06` | 0.1306 | 0 | `2.787553e-07` | `2.113107e-06` | 0.1319 | 11.85% | graded |
| [#316193](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316193) | German reproduction, 14% budget | 10 | `2.500576e-07` | `1.777524e-06` | 0.1407 | 0 | `2.875612e-07` | `2.022678e-06` | 0.1422 | 15.00% | graded |
| [#316194](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316194) | German reproduction, 16% budget | 10 | `2.675257e-07` | `1.665033e-06` | 0.1607 | 0 | `2.923943e-07` | `1.803904e-06` | 0.1621 | 9.30% | graded |
| [#316195](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316195) | German reproduction, 18% budget | 10 | `2.496365e-07` | `1.382409e-06` | 0.1805 | 0 | `2.920203e-07` | `1.606325e-06` | 0.1818 | 16.98% | graded |
| [#316196](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316196) | German reproduction, 19% budget | 10 | `2.516721e-07` | `1.320551e-06` | 0.1905 | 0 | `2.929196e-07` | `1.524329e-06` | 0.1922 | 16.39% | graded |
| [#316197](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316197) | German reproduction, 21% budget | 10 | `2.418113e-07` | `1.148469e-06` | 0.2105 | 0 | `3.018792e-07` | `1.425589e-06` | 0.2118 | 24.84% | graded |
| [#316298](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316298) | AutoEvolve Edgeworth c02503e1 | 100 | `3.629318e-07` | `1.484736e-06` | 0.2445 | 0 | `3.167075e-07` | `1.280396e-06` | 0.2474 | -12.74% | graded |
| [#316299](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316299) | AutoEvolve Gaussian CV 25d94716 | 100 | `3.621376e-07` | `1.479585e-06` | 0.2448 | 0 | `3.088166e-07` | `1.254226e-06` | 0.2462 | -14.72% | graded |
| [#316300](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316300) | AutoEvolve layer rescaling f4443aad | 100 | `3.633336e-07` | `3.633272e-06` | 0.0991 | 0 | `3.472519e-07` | `3.445321e-06` | 0.1008 | -4.43% | graded |
| [#316559](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316559) | AutoEvolve n100 setup-cached 41d53487 | 100 | `3.004490e-07` | - | - | 0 | `2.960862e-07` | `2.044541e-06` | 0.1448 | -1.45% | graded |
| [#316560](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316560) | AutoEvolve n100 analytic blend 57a2629d | 100 | `3.711656e-07` | - | - | 0 | `3.443012e-07` | `2.938379e-06` | 0.1172 | -7.24% | graded |
| [#316561](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316561) | AutoEvolve n100 mean-only bff13d51 | 100 | `2.748615e-07` | - | - | 0 | `2.969021e-07` | `2.049496e-06` | 0.1449 | 8.02% | graded |
| [#316562](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316562) | AutoEvolve n100 affine-f64 1cb0eb3a | 100 | `2.789888e-07` | - | - | 0 | `2.941713e-07` | `2.017920e-06` | 0.1458 | 5.44% | graded |
| [#316625](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316625) | Whitened-antithetic spherical RQMC + radial RB, 10% | 100 | `3.543378e-07` | `3.502966e-06` | 0.1012 | 0 | `3.355234e-07` | `3.262751e-06` | 0.1028 | -5.31% | graded |
| [#316628](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316628) | Whitened-antithetic spherical RQMC + radial RB, 10% (repackaged duplicate) | 100 | `3.543378e-07` | `3.502966e-06` | 0.1012 | 0 | - | - | - | - | failed |

## Provenance

| ID | Family | Local source | Local evaluation | Artifact | Submitted SHA-256 |
|---:|---|---|---|---|---|
| #316181 | rqmc | `repro_external/aicrowd_whestbench/public_reproductions/evaaaz_rqmc_rao_blackwell.py` | `repro_external/aicrowd_whestbench/public_reproductions/evaaaz_rqmc_rao_blackwell_official_mini_eval_20260714.json` | `repro_external/aicrowd_whestbench/submissions/packages/evaaaz_rqmc_rao_blackwell_phase1_20260714.tar.gz` | `c1a358958b954590cfa71c95c3203c0a3e6b8cc4cbdd223347d63ff4b11fedfd` |
| #316182 | autoevolve | `codex_runs/aicrowd_whestbench/whestbench_mini_autoevolve_blackbox_smoke_20260713_r5/autoevolve_workspaces/step_000000/call_0001_661899f5/submission.py` | `repro_external/aicrowd_whestbench/discovered/autoevolve_smoke_candidate_official_mini_eval_20260713.json` | `repro_external/aicrowd_whestbench/submissions/packages/autoevolve_blackbox_phase1_20260714.tar.gz` | `e0488a3775804d94c0d36942e4fbe402f9ca564aa119e90260325ca65ee0379e` |
| #316183 | german | `repro_external/aicrowd_whestbench/public_reproductions/german_alfaro_311690_whitened_antithetic.py` | `repro_external/aicrowd_whestbench/public_reproductions/german_alfaro_311690_official_mini_eval_20260713.json` | `repro_external/aicrowd_whestbench/submissions/packages/german_whitened_antithetic_reimplementation_phase1_20260714.tar.gz` | `644f6da862c57b4b29106ba53431c25f3ef9ef3368e31e774b180b8a5455c38f` |
| #316184 | rqmc | `repro_external/aicrowd_whestbench/submissions/candidates/rqmc_budget030.py` | `repro_external/aicrowd_whestbench/submissions/candidates/rqmc_budget030_first10_eval_20260714.json` | `repro_external/aicrowd_whestbench/submissions/packages/rqmc_budget030_phase1_20260714.tar.gz` | `6288a7fb59c7ae0db30d5f9fa75176c87fcd4c08694149660630a966d39b3a09` |
| #316185 | rqmc | `repro_external/aicrowd_whestbench/submissions/candidates/rqmc_budget055.py` | `repro_external/aicrowd_whestbench/submissions/candidates/rqmc_budget055_first10_eval_20260714.json` | `repro_external/aicrowd_whestbench/submissions/packages/rqmc_budget055_phase1_20260714.tar.gz` | `1f470f4883dba64aae5b2f804690688539aaca4693a9cd40cfce28e04b79a808` |
| #316186 | rqmc | `repro_external/aicrowd_whestbench/submissions/candidates/rqmc_budget070.py` | `repro_external/aicrowd_whestbench/submissions/candidates/rqmc_budget070_first10_eval_20260714.json` | `repro_external/aicrowd_whestbench/submissions/packages/rqmc_budget070_phase1_20260714.tar.gz` | `c1fef7995d14581e0eda05961e0acf7ed2d57798f49cf00501976d610af5a375` |
| #316188 | german | `repro_external/aicrowd_whestbench/submissions/candidates/german_budget010.py` | `repro_external/aicrowd_whestbench/submissions/candidates/german_budget010_first10_eval_20260714.json` | `repro_external/aicrowd_whestbench/submissions/packages/german_budget010_phase1_20260714.tar.gz` | `048ed3a10a01cf3cb7b0f131a4c040241bf6462ee2925206f4ee1487eb5ff8c4` |
| #316189 | german | `repro_external/aicrowd_whestbench/submissions/candidates/german_budget012.py` | `repro_external/aicrowd_whestbench/submissions/candidates/german_budget012_first10_eval_20260714.json` | `repro_external/aicrowd_whestbench/submissions/packages/german_budget012_phase1_20260714.tar.gz` | `1c6d0b1a81b908722f49496a8f67d5654062a231a432a7668a721247bf4f47c8` |
| #316190 | german | `repro_external/aicrowd_whestbench/submissions/candidates/german_budget020.py` | `repro_external/aicrowd_whestbench/submissions/candidates/german_budget020_first10_eval_20260714.json` | `repro_external/aicrowd_whestbench/submissions/packages/german_budget020_phase1_20260714.tar.gz` | `194235814ad2279b31edc9fdb08babeefd5e11b30ddeb4230749effc95e7385c` |
| #316191 | german | `repro_external/aicrowd_whestbench/submissions/candidates/german_budget025.py` | `repro_external/aicrowd_whestbench/submissions/candidates/german_budget025_first10_eval_20260714.json` | `repro_external/aicrowd_whestbench/submissions/packages/german_budget025_phase1_20260714.tar.gz` | `42c01e687472bbd0c96ec6221fa2bf0e18bd1e10f99eb57a79f3d3737891df5c` |
| #316192 | german | `repro_external/aicrowd_whestbench/submissions/candidates/german_budget013.py` | `repro_external/aicrowd_whestbench/submissions/candidates/german_budget013_first10_eval_20260714.json` | `repro_external/aicrowd_whestbench/submissions/packages/german_budget013_phase1_20260714.tar.gz` | `a1395b694c3bbbeb717ee2f8430f3677d90983c4fa7e8e86a17669da1b453a4f` |
| #316193 | german | `repro_external/aicrowd_whestbench/submissions/candidates/german_budget014.py` | `repro_external/aicrowd_whestbench/submissions/candidates/german_budget014_first10_eval_20260714.json` | `repro_external/aicrowd_whestbench/submissions/packages/german_budget014_phase1_20260714.tar.gz` | `2daa561948fa069d2e6c1f18b2cbf735d1f9791ed1ee0d410a451ec91bb4accc` |
| #316194 | german | `repro_external/aicrowd_whestbench/submissions/candidates/german_budget016.py` | `repro_external/aicrowd_whestbench/submissions/candidates/german_budget016_first10_eval_20260714.json` | `repro_external/aicrowd_whestbench/submissions/packages/german_budget016_phase1_20260714.tar.gz` | `2cfbc29538a893d01777c28988507a090ca44ef9527dab6fdba180874fb2b2fd` |
| #316195 | german | `repro_external/aicrowd_whestbench/submissions/candidates/german_budget018.py` | `repro_external/aicrowd_whestbench/submissions/candidates/german_budget018_first10_eval_20260714.json` | `repro_external/aicrowd_whestbench/submissions/packages/german_budget018_phase1_20260714.tar.gz` | `59454db9c4901bf196048c237b96aaba4bff960bc25bfe5520865717fee18977` |
| #316196 | german | `repro_external/aicrowd_whestbench/submissions/candidates/german_budget019.py` | `repro_external/aicrowd_whestbench/submissions/candidates/german_budget019_first10_eval_20260714.json` | `repro_external/aicrowd_whestbench/submissions/packages/german_budget019_phase1_20260714.tar.gz` | `7e6483e2498c65063fa708fefdb483143ea9fc542f983350de0da747bc75890a` |
| #316197 | german | `repro_external/aicrowd_whestbench/submissions/candidates/german_budget021.py` | `repro_external/aicrowd_whestbench/submissions/candidates/german_budget021_first10_eval_20260714.json` | `repro_external/aicrowd_whestbench/submissions/packages/german_budget021_phase1_20260714.tar.gz` | `0741d309cf02c4b9e0671c0fb51da8da3f3a71b1d9443c72ce8b3793aab863e2` |
| #316298 | autoevolve | `codex_runs/aicrowd_whestbench/whestbench_mini_autoevolve_blackbox_r5seed_n10_b500_20260713/autoevolve_workspaces/step_000017/call_0001_c02503e1/submission.py` | `repro_external/aicrowd_whestbench/submissions/candidates/autoevolve_edgeworth_c02503e1_full100_eval_20260714.json` | `repro_external/aicrowd_whestbench/submissions/packages/autoevolve_edgeworth_c02503e1_phase1_20260714.tar.gz` | `0e9fca4fb3faa9dc1a8a9c179b88c7344054c33464a1ec04376417b9695817c2` |
| #316299 | autoevolve | `codex_runs/aicrowd_whestbench/whestbench_mini_autoevolve_blackbox_r5seed_n10_b500_20260713/autoevolve_workspaces/step_000010/call_0001_25d94716/submission.py` | `repro_external/aicrowd_whestbench/submissions/candidates/autoevolve_gaussian_cv_25d94716_full100_eval_20260714.json` | `repro_external/aicrowd_whestbench/submissions/packages/autoevolve_gaussian_cv_25d94716_phase1_20260714.tar.gz` | `515139e490d642b6350d3fc81171c3c2942304a7434f5137f2ac0a149177c809` |
| #316300 | autoevolve | `codex_runs/aicrowd_whestbench/whestbench_mini_autoevolve_blackbox_r5seed_n10_b500_20260713/autoevolve_workspaces/step_000000/call_0001_f4443aad/submission.py` | `repro_external/aicrowd_whestbench/submissions/candidates/autoevolve_layer_rescale_f4443aad_full100_eval_20260714.json` | `repro_external/aicrowd_whestbench/submissions/packages/autoevolve_layer_rescale_f4443aad_phase1_20260714.tar.gz` | `ff4437d67e38527b37435360f0c4ecdbd778f1d28a09398ebeb91903abe0ada5` |
| #316559 | autoevolve | `codex_runs/aicrowd_whestbench/whestbench_mini_autoevolve_blackbox_german013_n100_b500_20260715/autoevolve_workspaces/step_000000/call_0001_117b42af/submission.py` | `repro_external/aicrowd_whestbench/submissions/candidates/autoevolve_n100_setup_cached_41d53487_full100_eval_20260716.json` | `repro_external/aicrowd_whestbench/submissions/packages/autoevolve_n100_setup_cached_41d53487_phase1_20260716.tar.gz` | `a6a56db3f90dd069e4f1975c6ddabc6db8c09f81c99b4bdcb5de0a2b0fb8a024` |
| #316560 | autoevolve | `codex_runs/aicrowd_whestbench/whestbench_mini_autoevolve_blackbox_german013_n100_b500_20260715/autoevolve_workspaces/step_000000/call_0001_8d3beaf7/submission.py` | `repro_external/aicrowd_whestbench/submissions/candidates/autoevolve_n100_analytic_blend_57a2629d_full100_eval_20260716.json` | `repro_external/aicrowd_whestbench/submissions/packages/autoevolve_n100_analytic_blend_57a2629d_phase1_20260716.tar.gz` | `63761169b1569fe9dbc9007019eaa1579a914d39fbbc3952f6af6e9839c31920` |
| #316561 | autoevolve | `codex_runs/aicrowd_whestbench/whestbench_mini_autoevolve_blackbox_german013_n100_b500_20260715/autoevolve_workspaces/step_000001/call_0001_b30c3f35/submission.py` | `repro_external/aicrowd_whestbench/submissions/candidates/autoevolve_n100_mean_only_bff13d51_full100_eval_20260716.json` | `repro_external/aicrowd_whestbench/submissions/packages/autoevolve_n100_mean_only_bff13d51_phase1_20260716.tar.gz` | `badd395c7264afe62873e9fd61c5a271acc6bb2cbca5322ca7dedc9bcd80c7a0` |
| #316562 | autoevolve | `codex_runs/aicrowd_whestbench/whestbench_mini_autoevolve_blackbox_german013_n100_b500_20260715/autoevolve_workspaces/step_000001/call_0001_4b4c2ecd/submission.py` | `repro_external/aicrowd_whestbench/submissions/candidates/autoevolve_n100_affine_f64_1cb0eb3a_full100_eval_20260716.json` | `repro_external/aicrowd_whestbench/submissions/packages/autoevolve_n100_affine_f64_1cb0eb3a_phase1_20260716.tar.gz` | `1cc9ebd9dc10f1330c149c162c9e1c869f189da20d70dc48af587caed5eead70` |
| #316625 | hybrid_mc | `repro_external/aicrowd_whestbench/baselines/monte_carlo_whitened_antithetic_rb_budget010.py` | `repro_external/aicrowd_whestbench/baselines/monte_carlo_whitened_antithetic_rb_budget010_full100_eval_20260716.json` | `repro_external/aicrowd_whestbench/submissions/packages/whitened_antithetic_spherical_rqmc_rb_budget010_phase1_20260716.tar.gz` | `dae36b30f44616518526eaee8e7aec639cf0b9b05e1187416b363b921c7fd11b` |
| #316628 | hybrid_mc | `repro_external/aicrowd_whestbench/baselines/monte_carlo_whitened_antithetic_rb_budget010.py` | `repro_external/aicrowd_whestbench/baselines/monte_carlo_whitened_antithetic_rb_budget010_full100_eval_20260716.json` | `repro_external/aicrowd_whestbench/submissions/packages/whitened_antithetic_spherical_rqmc_rb_budget010_phase1_20260716.tar.gz` | `309f97f132bc87c9bd4ec771ecf19b122cda5be03b0a27551e346bfffb1dc778` |

### Artifact integrity notes

The registry pins the SHA-256 of the archive that was actually submitted. A mismatch means the same local path was rebuilt later; it must not silently replace the submitted provenance.

- #316625: submitted `dae36b30f44616518526eaee8e7aec639cf0b9b05e1187416b363b921c7fd11b`, current file `309f97f132bc87c9bd4ec771ecf19b122cda5be03b0a27551e346bfffb1dc778` at `repro_external/aicrowd_whestbench/submissions/packages/whitened_antithetic_spherical_rqmc_rb_budget010_phase1_20260716.tar.gz`.

## Interpretation

- Compare absolute local and official values only within the same local protocol. A first-10 estimate has high subset variance and was also used during candidate selection, so it is optimistically biased.
- The local full-100 suite and hosted public-50 suite have different aggregation sets. A systematic score shift does not by itself demonstrate an evaluator bug.
- Prefer full-100 results for promotion decisions. Use first-10 only for failure, runtime, and rough budget screening.

## Refresh

```bash
PYTHONPATH=/tmp/whest-official-deps:$PWD \
  python repro/aicrowd_whestbench/update_submission_tracker.py
```

Use `--offline` to regenerate from the last cached official API response. Add new submission mappings to `repro_external/aicrowd_whestbench/submissions/registry.json` before refreshing.
