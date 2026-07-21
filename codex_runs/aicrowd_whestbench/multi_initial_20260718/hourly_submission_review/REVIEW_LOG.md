## Cycle 20260718T163252Z hourly submission review

- Completed at UTC: `2026-07-18T17:05:01+00:00`.
- Ledger was empty at start; baseline SHA set was built from registry/tracker/evaluation artifacts: `43` known code SHA plus `6` current initial SHA.
- Tracker refresh command ran before review; cached official status counts: {'graded': 24, 'failed': 1}. No queued/running/pending registered submissions: `[]`.
- AIcrowd refresh/auth status: `update_submission_tracker refresh recorded AIcrowdAuthError HTTP 401 for all registered submissions; server quota/status unavailable.`
- Quota assessment for UTC `2026-07-18`: registered submissions created today `0` / 50, nominal remaining `50`, server quota not verified because API refresh returned HTTP 401.
- Artifact scan mtime range UTC: `2026-07-13T17:50:41+00:00` to `2026-07-18T17:05:00+00:00`.

### Scanned runs and files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T163252Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T163252Z/scan_summary.json`
- Shortlist source copies: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T163252Z/shortlist_sources`
- Full-100 reports: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T163252Z/full100_reports`
- Initial estimators scanned: 6 under `codex_runs/aicrowd_whestbench/multi_initial_20260718/initial_states`.
- Console logs scanned for timestamp context: `codex_runs/aicrowd_whestbench/multi_initial_20260718/console`.

| Run | PUCT steps | AutoEvolve pool steps | Workspace submissions | File count |
|---|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | - | 0-1 | 2 | 2 |
| `whest_0718_affinef64_auto_r2` | - | 0-2 | 3 | 3 |
| `whest_0718_affinef64_ttt` | 0-11 | - | 0 | 12 |
| `whest_0718_gaussiancv_auto_r1` | - | 0-2 | 3 | 3 |
| `whest_0718_gaussiancv_auto_r2` | - | 0-2 | 3 | 3 |
| `whest_0718_gaussiancv_ttt` | 0-33 | - | 0 | 34 |
| `whest_0718_german013_auto_r1` | - | 0-1 | 2 | 2 |
| `whest_0718_german013_auto_r2` | - | 0-1 | 2 | 2 |
| `whest_0718_german013_ttt` | 0-7 | - | 0 | 8 |
| `whest_0718_layerrescale_auto_r1` | - | 0 | 1 | 1 |
| `whest_0718_layerrescale_auto_r2` | - | 0-1 | 2 | 2 |
| `whest_0718_layerrescale_ttt` | 0-23 | - | 0 | 24 |
| `whest_0718_rqmc030_auto_r1` | - | 0 | 1 | 1 |
| `whest_0718_rqmc030_auto_r2` | - | 0-1 | 2 | 2 |
| `whest_0718_rqmc030_ttt` | 0-8 | - | 0 | 9 |
| `whest_0718_sphericalrb_auto_r1` | - | 0 | 1 | 1 |
| `whest_0718_sphericalrb_auto_r2` | - | 0-1 | 2 | 2 |
| `whest_0718_sphericalrb_ttt` | 0-8 | - | 0 | 9 |

### Candidate counts

- Unique estimator SHA reviewed this cycle: `142`.
- Final decision counts: `{'reject_existing_sha': 6, 'reject_first10_failures': 1, 'reject_full100_no_submit': 4, 'reject_insufficient_first10_gain_not_full100': 17, 'reject_micro_variant_not_full100': 22, 'reject_no_first10': 11, 'reject_ranked_not_full100': 67, 'reject_weak_first10': 14}`.
- New/non-existing SHA after historical baseline removal: `136` unique SHA, including current initial baselines and rejected/validated new candidates.
- Entered full-100: `4`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Full-100 shortlist decisions

| Candidate | Family/run/step | State | SHA | First-10 adjusted / MSE / C-B / fail | Full-100 adjusted / MSE / C-B / fail | Decision |
|---|---|---|---|---|---|---|
| `german_blend_50254dff2b74` | `german013/whest_0718_german013_auto_r2/step 1` | `fdf1c913-c146-4bd1-a694-05eb3fcc96a4` | `50254dff2b74` | 1.424375910e-07 / 1.424375910e-06 / 0.0933 / 0 | 3.489487931e-07 / 3.489487931e-06 / 0.0929 / 0 | Reject: full-100 below submission bar; report `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T163252Z/full100_reports/german_blend_50254dff2b74_full100_20260718T163252Z.json` |
| `affine_spherical_blend_447da7c8662d` | `affinef64/whest_0718_affinef64_auto_r2/step 2` | `f86969c9-ee07-4e19-b9d3-5649ea5a535f` | `447da7c8662d` | 1.613732297e-07 / 1.613732297e-06 / 0.0988 / 0 | 3.216806472e-07 / 3.216806472e-06 / 0.0988 / 0 | Reject: full-100 below submission bar; report `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T163252Z/full100_reports/affine_spherical_blend_447da7c8662d_full100_20260718T163252Z.json` |
| `spherical_rqmc_control_a89911497acf` | `sphericalrb/whest_0718_sphericalrb_auto_r2/step 1` | `9af1fe7b-019f-4ad2-8c7c-70120a2a2699` | `a89911497acf` | 1.643222478e-07 / 1.620924473e-06 / 0.1012 / 0 | 3.955677388e-07 / 3.915165054e-06 / 0.1010 / 0 | Reject: full-100 below submission bar; report `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T163252Z/full100_reports/spherical_rqmc_control_a89911497acf_full100_20260718T163252Z.json` |
| `rqmc_first_layer_cv_53874e12c265` | `rqmc030/whest_0718_rqmc030_auto_r2/step 1` | `94c5d1c6-d0fb-437f-bfcc-65de5f06f09c` | `53874e12c265` | 2.195007400e-07 / 4.633260211e-07 / 0.4737 / 0 | 5.340921542e-07 / 1.138162475e-06 / 0.4696 / 0 | Reject: full-100 below submission bar; report `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T163252Z/full100_reports/rqmc_first_layer_cv_53874e12c265_full100_20260718T163252Z.json` |

Full-100 interpretation:

- `german_blend_50254dff2b74`: first-10 improved 76.6% vs German initial, but full-100 adjusted `3.489487931e-07`; weaker than local full-100 best `2.748615e-07` and official best `2.787553e-07`.
- `affine_spherical_blend_447da7c8662d`: method is setup-cached spherical antithetic whitening plus analytic final blend; full-100 adjusted `3.216806472e-07`, 0 failure, not enough for official validation.
- `spherical_rqmc_control_a89911497acf`: first-10 improved 35.4% vs spherical initial, but full-100 adjusted `3.955677388e-07`; no submission.
- `rqmc_first_layer_cv_53874e12c265`: first-10 improved 107.9% vs RQMC030 initial, but full-100 adjusted `5.340921542e-07`; no submission.

### Not promoted after ranking

| Group representative | Family | SHA | First-10 adjusted | Reason |
|---|---|---|---:|---|
| `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r2/autoevolve_pool_step_000001.json` | `german013` | `8c19a0944a61` | 1.424375910e-07 | Lower-ranked same-family/method variant after representative check: german_blend_50254dff2b74 full-100 adjusted 3.489e-7 despite first-10 1.424e-7. |
| `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_ttt/puct_sampler_step_000024.json` | `gaussiancv` | `9795525ac6c8` | 1.473837502e-07 | top GaussianCV 9795525ac6c8 improves existing submitted same-method SHA d49b8b5218ff by only 0.49% first-10; no structural novelty, so not run |
| `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r2/autoevolve_pool_step_000002.json` | `affinef64` | `9ab68b21384b` | 1.613732297e-07 | Lower-ranked same-family/method variant after representative check: affine_spherical_blend_447da7c8662d full-100 adjusted 3.217e-7 despite first-10 1.614e-7. |
| `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r2/autoevolve_pool_step_000001.json` | `sphericalrb` | `81683f931da1` | 1.643718697e-07 | Lower-ranked same-family/method variant after representative check: spherical_rqmc_control_a89911497acf full-100 adjusted 3.956e-7 despite first-10 1.643e-7. |
| `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_layerrescale_auto_r2/autoevolve_pool_step_000001.json` | `layerrescale` | `a6db482da556` | 1.941727237e-07 | best layerrescale a6db482da556 improves initial by only ~2% first-10 and uses same whitening mechanism; no structural novelty strong enough for full-100 |
| `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r2/autoevolve_pool_step_000001.json` | `rqmc030` | `f718d7e30915` | 2.196831044e-07 | Lower-ranked same-family/method variant after representative check: rqmc_first_layer_cv_53874e12c265 full-100 adjusted 5.341e-7 despite first-10 2.195e-7. |

### Official submission status

- No package was built and no `whest submit` was run, because no full-100 candidate cleared the quality bar. Separately, the official API refresh continued to return HTTP 401, so live quota/auth could not be verified and official submission would have been blocked even for a passing candidate.
- Submission IDs this cycle: none.
- Registry/tracker were not manually changed because there was no new official submission. `SUBMISSION_TRACKER.md` was regenerated by the required pre-review refresh.

### Next-cycle continuation point

- Continue from latest scanned TTT steps: affinef64 `11`, gaussiancv `33`, german013 `7`, layerrescale `23`, rqmc030 `8`, sphericalrb `8`.
- Continue from latest scanned AutoEvolve pool steps: affinef64 r1 `1`, affinef64 r2 `2`, gaussiancv r1 `2`, gaussiancv r2 `2`, german013 r1 `1`, german013 r2 `1`, layerrescale r1 `0`, layerrescale r2 `1`, rqmc030 r1 `0`, rqmc030 r2 `1`, sphericalrb r1 `0`, sphericalrb r2 `1`.
- New SHA already inventoried in this cycle should be treated as reviewed via `candidate_inventory.jsonl`; next cycle should only promote genuinely new SHA or a method-change representative that appears after these steps.

## Cycle 20260718T173549Z hourly submission review

- Completed at UTC: `2026-07-18T18:02:50.137738+00:00`.
- Tracker refresh command ran before review; cached official status counts: {'failed': 1, 'graded': 24}. No queued/running/pending registered submissions: `[]`.
- AIcrowd refresh/auth status: `AIcrowdAuthError HTTP 401 for registered submissions; server quota/status unavailable`.
- Quota assessment for UTC `2026-07-18`: registered submissions created today `0` / 50, nominal remaining `50`, server quota not verified because API refresh returned HTTP 401.
- Candidate artifact cutoff UTC: `2026-07-18T17:35:49+00:00`; artifact mtime range UTC: `2026-07-13T17:50:41.500571+00:00` to `2026-07-18T17:34:52.385389+00:00`.
- JSONL context files were read for sampler/metrics/agent context while live runs continued; their live mtime range was `2026-07-18T15:57:40.309455+00:00` to `2026-07-18T17:59:36.704489+00:00`, but candidate extraction from agent outputs was capped to states present in scanned step snapshots.

### Scanned runs and files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T173549Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T173549Z/scan_summary.json`
- Shortlist source copies: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T173549Z/shortlist_sources`
- Full-100 reports: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T173549Z/full100_reports`
- Initial estimators scanned: 6 under `codex_runs/aicrowd_whestbench/multi_initial_20260718/initial_states`.
- Console logs scanned for timestamp context: `codex_runs/aicrowd_whestbench/multi_initial_20260718/console`.
- Files excluded after cutoff: `16`; earliest examples are listed in scan summary.

| Run | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count |
|---|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | - | 0-2 | 2 | 5 |
| `whest_0718_affinef64_auto_r2` | - | 0-3 | 3 | 7 |
| `whest_0718_affinef64_ttt` | 0-16 | - | 0 | 17 |
| `whest_0718_gaussiancv_auto_r1` | - | 0-3 | 3 | 7 |
| `whest_0718_gaussiancv_auto_r2` | - | 0-3 | 3 | 7 |
| `whest_0718_gaussiancv_ttt` | 0-49 | - | 0 | 50 |
| `whest_0718_german013_auto_r1` | - | 0-1 | 2 | 4 |
| `whest_0718_german013_auto_r2` | - | 0-1 | 1 | 3 |
| `whest_0718_german013_ttt` | 0-13 | - | 0 | 14 |
| `whest_0718_layerrescale_auto_r1` | - | 0-1 | 1 | 3 |
| `whest_0718_layerrescale_auto_r2` | - | 0-1 | 1 | 3 |
| `whest_0718_layerrescale_ttt` | 0-36 | - | 0 | 37 |
| `whest_0718_rqmc030_auto_r1` | - | 0-1 | 1 | 3 |
| `whest_0718_rqmc030_auto_r2` | - | 0-1 | 1 | 3 |
| `whest_0718_rqmc030_ttt` | 0-12 | - | 0 | 13 |
| `whest_0718_sphericalrb_auto_r1` | - | 0-1 | 1 | 3 |
| `whest_0718_sphericalrb_auto_r2` | - | 0-1 | 1 | 3 |
| `whest_0718_sphericalrb_ttt` | 0-14 | - | 0 | 15 |

### Candidate counts

- Unique estimator SHA reviewed this cycle: `178`.
- New SHA after historical baseline removal: `46`.
- Final decision counts: `{'reject_full100_no_submit': 1, 'reject_ranked_not_full100': 24, 'reject_micro_variant_not_full100': 1, 'reject_insufficient_first10_gain_not_full100': 1, 'reject_weak_first10': 16, 'reject_first10_failures': 3, 'reject_existing_sha': 132}`.
- Entered full-100: `1`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Full-100 shortlist decisions

| Candidate | Family/run/step | State | SHA | First-10 adjusted / MSE / C-B / fail | Full-100 adjusted / MSE / C-B / fail | Decision |
|---|---|---|---|---|---|---|
| `rqmc030_lattice_radial_d08c2c389bca` | `rqmc030/whest_0718_rqmc030_auto_r1/step 1` | `c9545484-dc10-4c75-8467-0887938e5c9b` | `d08c2c389bca` | 2.256648439e-07 / 2.830488285e-07 / 0.7968 / 0 | 5.237456081e-07 / 6.595759190e-07 / 0.7935 / 0 | Reject: full-100 below submission bar; report `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T173549Z/full100_reports/rqmc030_lattice_radial_d08c2c389bca_full100_20260718T173549Z.json` |

Full-100 interpretation:

- `rqmc030_lattice_radial_d08c2c389bca`: shifted lattice normal samples with radial normalization and float64 final mean. First-10 improved 102.2% vs rqmc030 initial, but full-100 adjusted `5.237456081e-07` is weaker than local full-100 best `2.748615e-07` and official best `2.787553e-07`; no package or official submit.

### Not promoted after ranking

| Family | Run | Step | SHA | First-10 adjusted | vs initial | Decision | Reason |
|---|---|---:|---|---:|---:|---|---|
| `gaussiancv` | `whest_0718_gaussiancv_ttt` | `41` | `0c55690bf085` | 1.480844902e-07 | 1.086902812e-01% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `gaussiancv` | `whest_0718_gaussiancv_auto_r1` | `3` | `015d0c3c9ca0` | 1.480922526e-07 | 1.034429976e-01% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `gaussiancv` | `whest_0718_gaussiancv_ttt` | `46` | `aa8cf989cf99` | 1.481542597e-07 | 6.154663240e-02% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `gaussiancv` | `whest_0718_gaussiancv_auto_r2` | `3` | `9108687a8b0c` | 1.483051171e-07 | -4.023696823e-02% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `affinef64` | `whest_0718_affinef64_auto_r2` | `3` | `340638bd0545` | 1.511689339e-07 | 4.703515636e+01% | `reject_ranked_not_full100` | Not promoted after prior full-100 evidence: prior same-family representative full-100 was 3.217e-7; current improvement 47.0% is weaker. |
| `sphericalrb` | `whest_0718_sphericalrb_auto_r1` | `1` | `c80f81394359` | 1.538638353e-07 | 4.476866278e+01% | `reject_ranked_not_full100` | Not promoted after prior full-100 evidence: prior same-family representative full-100 was 3.956e-7; current improvement 44.8% is weaker. |
| `sphericalrb` | `whest_0718_sphericalrb_auto_r1` | `1` | `601bb56e8865` | 1.538737936e-07 | 4.475929374e+01% | `reject_ranked_not_full100` | Lower-ranked same-family/method variant; representative c80f81394359 selected for full-100. |
| `layerrescale` | `whest_0718_layerrescale_auto_r1` | `1` | `0776e728adae` | 1.647700875e-07 | 2.071519129e+01% | `reject_micro_variant_not_full100` | Best new layerrescale SHA first-10 improvement 20.72% is within noise or method is a small variant. |
| `affinef64` | `whest_0718_affinef64_auto_r1` | `2` | `231086583313` | 1.678512935e-07 | 3.242166546e+01% | `reject_ranked_not_full100` | Lower-ranked same-family/method variant; representative 340638bd0545 selected for full-100. |
| `affinef64` | `whest_0718_affinef64_auto_r1` | `2` | `e73d164e9aa3` | 1.680687270e-07 | 3.225034919e+01% | `reject_ranked_not_full100` | Lower-ranked same-family/method variant; representative 340638bd0545 selected for full-100. |

### Official submission status

- No package was built and no `whest submit` was run because the only full-100 candidate failed the quality bar.
- Separately, the required tracker refresh still recorded AIcrowdAuthError HTTP 401 for all registered submissions; if a candidate had passed, live quota/auth verification would have blocked official submission.
- Submission IDs this cycle: none.
- Registry was not changed because there was no new official submission. `SUBMISSION_TRACKER.md` was regenerated by the required pre-review refresh.

### Next-cycle continuation point

- Continue from candidate artifact cutoff `2026-07-18T17:35:49+00:00`; do not treat this cycle's `178` inventoried SHA as new.
- Latest scanned TTT steps at cutoff: affinef64 `16`, gaussiancv `49`, german013 `13`, layerrescale `36`, rqmc030 `12`, sphericalrb `14`.
- Latest scanned AutoEvolve pool steps at cutoff: affinef64 r1 `2`, affinef64 r2 `3`, gaussiancv r1 `3`, gaussiancv r2 `3`, german013 r1 `1`, german013 r2 `1`, layerrescale r1 `1`, layerrescale r2 `1`, rqmc030 r1 `1`, rqmc030 r2 `1`, sphericalrb r1 `1`, sphericalrb r2 `1`.
- Next cycle should start with files after the cutoff, including excluded artifacts recorded in `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T173549Z/scan_summary.json`.

## Cycle 20260718T183409Z hourly submission review

- Completed at UTC: `2026-07-18T18:52:55.966091+00:00`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- Cached official status counts after refresh: `{'failed': 1, 'graded': 24}`. Pending/queued/running registered submissions: `[]`.
- AIcrowd refresh/auth status: `AIcrowdAuthError HTTP 401 for registered submissions; server quota/status unavailable`.
- Quota assessment for UTC `2026-07-18`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`.
- Candidate cutoff UTC: `2026-07-18T18:34:09+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-18T18:33:50.062352+00:00']`.
- JSONL context files were read for line counts/provenance while live runs continued; live mtime range UTC: `['2026-07-18T15:57:40.309455+00:00', '2026-07-18T18:37:53.566058+00:00']`.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T183409Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T183409Z/scan_summary.json`
- Shortlist sources: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T183409Z/shortlist_sources`
- Full-100 reports: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T183409Z/full100_reports`
- Initial estimator SHA baselines scanned under `initial_states`: `{'affinef64': '1cb0eb3a875702386c346d605880190c7827b392860c7ddb6fb690f9f9b71435', 'gaussiancv': 'd49b8b5218ff9b9462a331812e688e125fcfff1f86e8a26e9be9c8f59eb48ebb', 'german013': '68d409403de29c8fa7807a8274066494e9ce00380bbbe5bf3d1d52e4ba35aa64', 'layerrescale': '98da6a0882cbc02d32cbd20e295c80b21bb8c173e3e23bb0d6880233d611d7ff', 'rqmc030': 'dc60550f76e9f4f6cb5940af295cab777a4082412590a49665e9127b535eb19f', 'sphericalrb': '146a8c786ee31dfefc941712ded736c3ec2625343dd5cd184f80dde915a9d130'}`
- Files excluded after cutoff: `15`; examples are recorded in scan summary.

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count |
|---|---|---|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | - | 0-3 | 3 | 7 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | - | 0-3 | 3 | 7 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | 0-21 | - | 0 | 22 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | - | 0-4 | 4 | 9 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | - | 0-4 | 4 | 9 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | 0-50 | - | 0 | 51 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | - | 0-1 | 2 | 4 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | - | 0-1 | 1 | 3 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | 0-21 | - | 0 | 22 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | - | 0-1 | 2 | 4 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | - | 0-1 | 1 | 3 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | 0-49 | - | 0 | 50 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | - | 0-2 | 2 | 5 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | - | 0-2 | 2 | 5 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | 0-17 | - | 0 | 18 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | - | 0-1 | 1 | 3 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | - | 0-1 | 2 | 4 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | 0-20 | - | 0 | 21 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `230`.
- Historical/known SHA baseline size: `238`; old inventory entries read: `320`.
- New SHA after historical baseline removal: `204`.
- Final decision counts: `{'reject_existing_sha': 26, 'reject_full100_no_submit': 2, 'reject_insufficient_first10_gain_not_full100': 62, 'reject_micro_variant_not_full100': 1, 'reject_no_first10': 2, 'reject_ranked_not_full100': 21, 'reject_weak_first10': 116}`.
- Entered full-100: `2`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Shortlist Decisions

| SHA | Family / run / step | State | First-10 adjusted / MSE / C-B / fail | Full-100 adjusted / MSE / C-B / fail | Decision |
|---|---|---|---|---|---|
| `28b81fb94819` | `sphericalrb` / `whest_0718_sphericalrb_auto_r1` / step `1` | `0f02d4ea-a8cf-4035-bd5d-05bb87b4bd23` | 1.538737936e-07 / 1.510388353e-06 / 0.1018 / 0 | 4.023362027e-07 / 3.962611216e-06 / 0.1015 / 0 | `reject_full100_no_submit` |
| `635f6e719666` | `rqmc030` / `whest_0718_rqmc030_auto_r2` / step `2` | `b9d9c465-17e2-4ba9-ba39-b70bc7d417b4` | 1.846943916e-07 / 4.621826939e-07 / 0.3996 / 0 | 4.553316429e-07 / 1.139358304e-06 / 0.3997 / 0 | `reject_full100_no_submit` |
| `fcffbe0cdfe7` | `german013` / `whest_0718_german013_auto_r2` / step `1` | `4b6d0459-c663-4f93-84eb-d328bea94a8a` | 1.424375910e-07 / 1.424375910e-06 / 0.0932 / 0 | - / - / - / - | `reject_micro_variant_not_full100` |

Full-100 interpretation:
- `28b81fb94819` report `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T183409Z/full100_reports/sphericalrb_prefix_control_28b81fb94819_full100_20260718T183409Z.json`: adjusted `4.023362027e-07`, MSE `3.962611216e-06`, C/B `0.1015`, failures `0`, failure_breakdown `{'budget_exhausted': 0, 'combined_budget_exhausted': 0, 'error': 0, 'residual_wall_time_exhausted': 0, 'time_exhausted': 0}`.
- `635f6e719666` report `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T183409Z/full100_reports/rqmc030_first_layer_cv_635f6e719666_full100_20260718T183409Z.json`: adjusted `4.553316429e-07`, MSE `1.139358304e-06`, C/B `0.3997`, failures `0`, failure_breakdown `{'budget_exhausted': 0, 'combined_budget_exhausted': 0, 'error': 0, 'residual_wall_time_exhausted': 0, 'time_exhausted': 0}`.
- `fcffbe0cdfe7` was not run full-100: diff against prior `german_blend_50254dff2b74` is whitespace/newline only, first-10 is identical (`1.424375910e-07`), and the prior representative full-100 was `3.489487931e-07`; this is not a new method.
- Both actual full-100 candidates had 0 failures and no budget/time exhaustion, but both are weaker than registered local full-100 best `2.748615e-07` and best registered official hosted public-50 `2.787552892e-07`.

### Not Promoted After Ranking

| Family | Run | Step | SHA | First-10 adjusted | vs initial recorded pct | Decision |
|---|---|---:|---|---:|---:|---|
| `gaussiancv` | `whest_0718_gaussiancv_auto_r1` | `4` | `a0f5bc4922b1` | 1.467102880e-07 | 0.9549755902779613 | `reject_insufficient_first10_gain_not_full100` |
| `gaussiancv` | `whest_0718_gaussiancv_ttt` | `24` | `58c018804321` | 1.473837502e-07 | 0.4936672077750374 | `reject_weak_first10` |
| `gaussiancv` | `whest_0718_gaussiancv_ttt` | `8` | `fd581aa51b18` | 1.479419469e-07 | 0.11449661923719893 | `reject_weak_first10` |
| `gaussiancv` | `whest_0718_gaussiancv_ttt` | `3` | `3af21706bee7` | 1.480591492e-07 | 0.03524685588844222 | `reject_weak_first10` |
| `gaussiancv` | `whest_0718_gaussiancv_ttt` | `42` | `ad91bef2fc45` | 1.480844902e-07 | 0.018128330292465872 | `reject_weak_first10` |
| `gaussiancv` | `whest_0718_gaussiancv_ttt` | `4` | `0350ae400781` | 1.480956570e-07 | 0.010586711674284542 | `reject_weak_first10` |
| `gaussiancv` | `whest_0718_gaussiancv_ttt` | `46` | `669094df219e` | 1.481542597e-07 | -0.02897267064838994 | `reject_weak_first10` |
| `gaussiancv` | `whest_0718_gaussiancv_ttt` | `1` | `89e4983c711b` | 1.481610920e-07 | -0.033582721384725484 | `reject_weak_first10` |
| `gaussiancv` | `whest_0718_gaussiancv_ttt` | `15` | `973279e84822` | 1.481900643e-07 | -0.05312697971125069 | `reject_weak_first10` |
| `gaussiancv` | `whest_0718_gaussiancv_ttt` | `11` | `2b3a528ce793` | 1.481972379e-07 | -0.05796498269458304 | `reject_weak_first10` |

Reasons for non-promotion: GaussianCV and LayerRescale candidates were weak or within first-10 noise versus initial; Affine/German/Spherical/RQMC lower-ranked variants were method-near variants of representatives that either already failed full-100 or were superseded by stronger first-10 candidates this cycle.

### Official Submission Status

- No package was built and no `whest submit` was run because no non-duplicate full-100 candidate cleared the quality bar.
- No new submission ID was obtained; registry and canonical tracker were not manually changed after the required pre-review tracker refresh.
- If a candidate had passed local quality, live official submission would still have been blocked pending credential/quota verification because all registered submission refreshes reported HTTP 401.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-18T18:34:09+00:00`; do not treat this cycle's `230` inventoried SHA as new.
- Latest scanned TTT steps at cutoff: affinef64 `21`, gaussiancv `50`, german013 `21`, layerrescale `49`, rqmc030 `17`, sphericalrb `20`.
- Latest scanned AutoEvolve pool steps at cutoff: whest_0718_affinef64_auto_r1 `3`, whest_0718_affinef64_auto_r2 `3`, whest_0718_gaussiancv_auto_r1 `4`, whest_0718_gaussiancv_auto_r2 `4`, whest_0718_german013_auto_r1 `1`, whest_0718_german013_auto_r2 `1`, whest_0718_layerrescale_auto_r1 `1`, whest_0718_layerrescale_auto_r2 `1`, whest_0718_rqmc030_auto_r1 `2`, whest_0718_rqmc030_auto_r2 `2`, whest_0718_sphericalrb_auto_r1 `1`, whest_0718_sphericalrb_auto_r2 `1`.
- Include artifacts excluded after cutoff next round, starting with: `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r1/autoevolve_workspaces/step_000003/call_0001_730e4714/submission.py, codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r2/autoevolve_workspaces/step_000003/call_0001_4b9288af/submission.py, codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r1/autoevolve_workspaces/step_000004/call_0001_4fe30f4e/submission.py, codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r2/autoevolve_workspaces/step_000004/call_0001_3f87cc2b/submission.py, codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r2/autoevolve_workspaces/step_000001/call_0001_d4e6ea9b/submission.py, codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_layerrescale_auto_r1/autoevolve_pool_step_000002.json, codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_layerrescale_auto_r1/autoevolve_workspaces/step_000002/call_0001_b17d4871/submission.py, codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_layerrescale_auto_r2/autoevolve_workspaces/step_000001/call_0001_f824dd53/submission.py`.

## Cycle 20260718T192508Z hourly submission review

- Completed at UTC: `2026-07-18T19:43:13.654729+00:00`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- Cached official status counts after refresh: `{'graded': 24, 'failed': 1}`. Pending/queued/running registered submissions: `[]`.
- AIcrowd refresh/auth status: `AIcrowdAuthError HTTP 401 for registered submissions; server quota/status unavailable`.
- Quota assessment for UTC `2026-07-18`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`.
- Candidate cutoff UTC: `2026-07-18T19:25:08+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-18T19:23:51.260558+00:00']`.
- JSONL context files were read for line counts/provenance while live runs continued; live mtime range UTC: `['2026-07-18T17:40:25.175395+00:00', '2026-07-18T19:26:43.291039+00:00']`.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T192508Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T192508Z/scan_summary.json`
- Shortlist manifest: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T192508Z/shortlist_manifest.json`
- Shortlist sources: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T192508Z/shortlist_sources`
- Full-100 reports: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T192508Z/full100_reports`
- Initial estimator SHA baselines scanned under `initial_states`: `{'affinef64': '1cb0eb3a875702386c346d605880190c7827b392860c7ddb6fb690f9f9b71435', 'gaussiancv': 'd49b8b5218ff9b9462a331812e688e125fcfff1f86e8a26e9be9c8f59eb48ebb', 'german013': '68d409403de29c8fa7807a8274066494e9ce00380bbbe5bf3d1d52e4ba35aa64', 'layerrescale': '98da6a0882cbc02d32cbd20e295c80b21bb8c173e3e23bb0d6880233d611d7ff', 'rqmc030': 'dc60550f76e9f4f6cb5940af295cab777a4082412590a49665e9127b535eb19f', 'sphericalrb': '146a8c786ee31dfefc941712ded736c3ec2625343dd5cd184f80dde915a9d130'}`
- Files excluded after cutoff: `12`; examples: `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r1/autoevolve_workspaces/step_000003/call_0001_730e4714/submission.py`, `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r2/autoevolve_workspaces/step_000004/call_0001_b6b6dabf/submission.py`, `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_ttt/puct_sampler_step_000026.json`, `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r1/autoevolve_workspaces/step_000005/call_0001_eb58ad7c/submission.py`, `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r2/autoevolve_workspaces/step_000005/call_0001_ef467ef3/submission.py`, `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r2/autoevolve_workspaces/step_000002/call_0001_ee997536/submission.py`, `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_layerrescale_auto_r1/autoevolve_workspaces/step_000002/call_0001_b17d4871/submission.py`, `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_layerrescale_auto_r2/autoevolve_workspaces/step_000002/call_0001_8e4dda7e/submission.py`, `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r1/autoevolve_workspaces/step_000002/call_0001_02996588/submission.py`, `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r2/autoevolve_workspaces/step_000002/call_0001_11e439ac/submission.py`, `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r1/autoevolve_workspaces/step_000002/call_0001_a0cfffca/submission.py`, `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r2/autoevolve_workspaces/step_000002/call_0001_e83a3287/submission.py`.

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count |
|---|---|---|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | - | 0-3 | 3 | 7 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | - | 0-4 | 4 | 9 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | 0-25 | - | 0 | 26 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | - | 0-5 | 5 | 11 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | - | 0-5 | 5 | 11 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | 0-50 | - | 0 | 51 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | - | 0-2 | 3 | 6 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | - | 0-2 | 2 | 5 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | 0-27 | - | 0 | 28 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | - | 0-2 | 2 | 5 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | - | 0-2 | 2 | 5 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | 0-50 | - | 0 | 51 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | - | 0-2 | 2 | 5 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | - | 0-2 | 2 | 5 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | 0-20 | - | 0 | 21 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | - | 0-2 | 2 | 5 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | - | 0-2 | 2 | 5 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | 0-26 | - | 0 | 27 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `261`.
- Historical/known SHA baseline size: `442`; old inventory entries read: `550`.
- New SHA after historical baseline removal: `31`.
- Initial decision counts: `{'candidate_full100_pending': 3, 'reject_existing_sha': 230, 'reject_insufficient_first10_gain_not_full100': 8, 'reject_ranked_not_full100': 7, 'reject_weak_first10': 13}`.
- Final decision counts: `{'reject_full100_no_submit': 3, 'reject_ranked_not_full100': 7, 'reject_existing_sha': 230, 'reject_weak_first10': 13, 'reject_insufficient_first10_gain_not_full100': 8}`.
- Entered full-100: `3`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Shortlist Decisions

| SHA | Family / run / step | State | First-10 adjusted / MSE / C-B / fail | Full-100 adjusted / MSE / C-B / fail | Decision |
|---|---|---|---|---|---|
| `8c0a803abd3b` | `german013` / `whest_0718_german013_auto_r2` / step `2` | `17f98f7c-ea62-4184-ab66-76fa53f51e07` | 1.385124267e-07 / 1.385124267e-06 / 9.364707264e-02 / 0 | 3.562495457e-07 / 3.562495457e-06 / 9.306813636e-02 / 0 | `reject_full100_no_submit` |
| `d22015256e51` | `layerrescale` / `whest_0718_layerrescale_auto_r2` / step `2` | `c9fc807c-f824-418c-857b-6351ef2ed9ef` | 1.400971126e-07 / 1.398554082e-06 / 9.991815675e-02 / 0 | 3.671553624e-07 / 3.671150721e-06 / 9.976014988e-02 / 0 | `reject_full100_no_submit` |
| `e073ccd86f4d` | `sphericalrb` / `whest_0718_sphericalrb_auto_r1` / step `2` | `4890c1f8-7107-40d3-87f3-400f92604319` | 1.475194706e-07 / 1.447215072e-06 / 1.018237005e-01 / 0 | 4.091460474e-07 / 4.026604810e-06 / 1.016296739e-01 / 0 | `reject_full100_no_submit` |

Full-100 interpretation:
- `8c0a803abd3b` report `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T192508Z/full100_reports/german013_radial_blend_8c0a803abd3b_full100_20260718T192508Z.json`: first-10 adjusted `1.385124267e-07`, vs initial `81.650%` if comparable; full-100 adjusted `3.562495457e-07`, MSE `3.562495457e-06`, C/B `9.306813636e-02`, failures `0`; no submit because full-100 is weaker than local/official registered best.
- `d22015256e51` report `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T192508Z/full100_reports/layerrescale_whitened_rescale_d22015256e51_full100_20260718T192508Z.json`: first-10 adjusted `1.400971126e-07`, vs initial `41.355%` if comparable; full-100 adjusted `3.671553624e-07`, MSE `3.671150721e-06`, C/B `9.976014988e-02`, failures `0`; no submit because full-100 is weaker than local/official registered best.
- `e073ccd86f4d` report `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T192508Z/full100_reports/sphericalrb_antithetic_blend_e073ccd86f4d_full100_20260718T192508Z.json`: first-10 adjusted `1.475194706e-07`, vs initial `50.861%` if comparable; full-100 adjusted `4.091460474e-07`, MSE `4.026604810e-06`, C/B `1.016296739e-01`, failures `0`; no submit because full-100 is weaker than local/official registered best.
- All three full-100 candidates had 0 failures and no budget/time exhaustion flags, but all are weaker than registered local full-100 best `2.748615e-07` and best registered official hosted public-50 `2.787552892e-07`.

### Not Promoted After Ranking

| Family | Run | Step | SHA | First-10 adjusted | vs initial recorded pct | Decision | Reason |
|---|---|---:|---|---:|---:|---|---|
| `german013` | `whest_0718_german013_auto_r2` | `2` | `d1cc43eb710b` | 1.385124267e-07 | 81.65047000244833 | `reject_ranked_not_full100` | Lower-ranked same-family/method variant; a stronger representative was selected for full-100. |
| `layerrescale` | `whest_0718_layerrescale_auto_r2` | `2` | `623f9ebee41b` | 1.401310904e-07 | 41.32112961845013 | `reject_ranked_not_full100` | Lower-ranked same-family/method variant; a stronger representative was selected for full-100. |
| `sphericalrb` | `whest_0718_sphericalrb_auto_r1` | `2` | `a515c0e17ed3` | 1.476292807e-07 | 50.74880063123216 | `reject_ranked_not_full100` | Lower-ranked same-family/method variant; a stronger representative was selected for full-100. |
| `gaussiancv` | `whest_0718_gaussiancv_auto_r2` | `5` | `d422a9278d31` | 1.481206123e-07 | -0.006263038476086568 | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `affinef64` | `whest_0718_affinef64_auto_r2` | `4` | `f020da564d0e` | 1.503976944e-07 | 47.25521902676842 | `reject_ranked_not_full100` | Strong first-10 gain (47.3% by recorded ratio), but prior same-family representatives already regressed on full-100 around 3.217e-07; not enough new mechanism to spend validation. |
| `affinef64` | `whest_0718_affinef64_auto_r2` | `4` | `f371a6ef6bc8` | 1.504063827e-07 | 47.246712828086004 | `reject_ranked_not_full100` | Strong first-10 gain (47.2% by recorded ratio), but prior same-family representatives already regressed on full-100 around 3.217e-07; not enough new mechanism to spend validation. |
| `layerrescale` | `whest_0718_layerrescale_auto_r1` | `2` | `e55c78369171` | 1.562999212e-07 | 26.701816808011497 | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `sphericalrb` | `whest_0718_sphericalrb_auto_r2` | `2` | `9c53cdf2e8f9` | 1.586408573e-07 | 40.28502734183724 | `reject_ranked_not_full100` | Strong first-10 gain (40.3% by recorded ratio), but prior same-family representatives already regressed on full-100 around 3.956e-07; not enough new mechanism to spend validation. |
| `layerrescale` | `whest_0718_layerrescale_ttt` | `50` | `65969d7bd842` | 1.983308035e-07 | -0.14922728554236725 | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | `26` | `8360d36fa7d0` | 2.206566158e-07 | 0.8577826777938625 | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | `24` | `ef71f8714aa7` | 2.207761351e-07 | 0.8031823445067081 | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | `23` | `e5f492c9d700` | 2.364201743e-07 | -5.867013798098543 | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |

Reasons for non-promotion: lower-ranked German/LayerRescale/Spherical variants were method-near variants of the representatives that failed to clear full-100; Affine candidates had only small gains over a same-family parent and prior affine full-100 evidence was weak; GaussianCV and TTT tail states were within first-10 noise or worse than their initial baseline.

### Official Submission Status

- No package was built and no `whest submit` was run because no non-duplicate full-100 candidate cleared the local quality bar.
- No new submission ID was obtained; registry and canonical tracker were not manually changed after the required pre-review tracker refresh.
- Live official submission would also have been blocked pending credential/quota verification because all registered submission refreshes reported HTTP 401.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-18T19:25:08+00:00`; do not treat this cycle's `261` inventoried SHA as new.
- Latest scanned TTT steps at cutoff: affinef64 `25`, gaussiancv `50`, german013 `27`, layerrescale `50`, rqmc030 `20`, sphericalrb `26`.
- Latest scanned AutoEvolve pool steps at cutoff: whest_0718_affinef64_auto_r1 `3`, whest_0718_affinef64_auto_r2 `4`, whest_0718_gaussiancv_auto_r1 `5`, whest_0718_gaussiancv_auto_r2 `5`, whest_0718_german013_auto_r1 `2`, whest_0718_german013_auto_r2 `2`, whest_0718_layerrescale_auto_r1 `2`, whest_0718_layerrescale_auto_r2 `2`, whest_0718_rqmc030_auto_r1 `2`, whest_0718_rqmc030_auto_r2 `2`, whest_0718_sphericalrb_auto_r1 `2`, whest_0718_sphericalrb_auto_r2 `2`.
- Include artifacts excluded after cutoff next round, starting with: `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r1/autoevolve_workspaces/step_000003/call_0001_730e4714/submission.py`, `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r2/autoevolve_workspaces/step_000004/call_0001_b6b6dabf/submission.py`, `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_ttt/puct_sampler_step_000026.json`, `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r1/autoevolve_workspaces/step_000005/call_0001_eb58ad7c/submission.py`, `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r2/autoevolve_workspaces/step_000005/call_0001_ef467ef3/submission.py`, `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r2/autoevolve_workspaces/step_000002/call_0001_ee997536/submission.py`, `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_layerrescale_auto_r1/autoevolve_workspaces/step_000002/call_0001_b17d4871/submission.py`, `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_layerrescale_auto_r2/autoevolve_workspaces/step_000002/call_0001_8e4dda7e/submission.py`, `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r1/autoevolve_workspaces/step_000002/call_0001_02996588/submission.py`, `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r2/autoevolve_workspaces/step_000002/call_0001_11e439ac/submission.py`, `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r1/autoevolve_workspaces/step_000002/call_0001_a0cfffca/submission.py`, `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r2/autoevolve_workspaces/step_000002/call_0001_e83a3287/submission.py`.

## Cycle 20260718T201404Z hourly submission review

- Completed at UTC: `2026-07-18T20:18:47.781502+00:00`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- Cached official status counts after refresh: `{'failed': 1, 'graded': 24}`. Pending/queued/running registered submissions: `[]`.
- AIcrowd refresh/auth status: `AIcrowdAuthError HTTP 401 for registered submissions; server quota/status unavailable`.
- Quota assessment for UTC `2026-07-18`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`.
- Candidate cutoff UTC: `2026-07-18T20:14:04+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-18T20:14:02.362828+00:00']`.
- JSONL context files were read for line counts/provenance while live runs continued; live mtime range UTC: `['2026-07-18T17:40:25.175395+00:00', '2026-07-18T20:05:01.507288+00:00']`.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T201404Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T201404Z/scan_summary.json`
- Shortlist manifest: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T201404Z/shortlist_manifest.json`
- Shortlist sources: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T201404Z/shortlist_sources`
- Initial estimator SHA baselines scanned under `initial_states`: `{'affinef64': '1cb0eb3a875702386c346d605880190c7827b392860c7ddb6fb690f9f9b71435', 'gaussiancv': 'd49b8b5218ff9b9462a331812e688e125fcfff1f86e8a26e9be9c8f59eb48ebb', 'german013': '68d409403de29c8fa7807a8274066494e9ce00380bbbe5bf3d1d52e4ba35aa64', 'layerrescale': '98da6a0882cbc02d32cbd20e295c80b21bb8c173e3e23bb0d6880233d611d7ff', 'rqmc030': 'dc60550f76e9f4f6cb5940af295cab777a4082412590a49665e9127b535eb19f', 'sphericalrb': '146a8c786ee31dfefc941712ded736c3ec2625343dd5cd184f80dde915a9d130'}`
- Files excluded after cutoff: `10`; examples: `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r1/autoevolve_workspaces/step_000004/call_0001_53b132b6/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r2/autoevolve_workspaces/step_000005/call_0001_c711fdfe/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r1/autoevolve_workspaces/step_000006/call_0001_52112175/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r2/autoevolve_workspaces/step_000007/call_0001_f5070340/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r2/autoevolve_workspaces/step_000003/call_0001_61e2f15f/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_layerrescale_auto_r2/autoevolve_workspaces/step_000003/call_0001_46ad9438/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r1/autoevolve_workspaces/step_000003/call_0001_9d3b9aaf/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r2/autoevolve_workspaces/step_000003/call_0001_2a9f4717/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r1/autoevolve_workspaces/step_000003/call_0001_73bdde20/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r2/autoevolve_workspaces/step_000003/call_0001_d6beb25c/submission.py`.

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count |
|---|---|---|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | - | 0-4 | 4 | 9 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | - | 0-5 | 5 | 11 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | 0-30 | - | 0 | 31 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | - | 0-6 | 6 | 13 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | - | 0-7 | 7 | 15 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | 0-50 | - | 0 | 51 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | - | 0-2 | 3 | 6 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | - | 0-3 | 3 | 7 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | 0-31 | - | 0 | 32 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | - | 0-3 | 4 | 8 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | - | 0-3 | 3 | 7 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | 0-50 | - | 0 | 51 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | - | 0-3 | 3 | 7 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | - | 0-3 | 3 | 7 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | 0-24 | - | 0 | 25 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | - | 0-3 | 3 | 7 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | - | 0-3 | 3 | 7 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | 0-31 | - | 0 | 32 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `289`.
- Historical/known SHA baseline size: `473`; old inventory entries read: `811`.
- New SHA after historical baseline removal: `28`.
- Final decision counts: `{'reject_existing_sha': 261, 'reject_insufficient_first10_gain_not_full100': 8, 'reject_no_first10': 9, 'reject_ranked_not_full100': 2, 'reject_weak_first10': 9}`.
- Entered full-100: `0`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Shortlist Decisions

No candidate entered full-100. The new SHA set had no clean first-10 result that was both materially better than same-family initial/prior representatives and method-distinct enough to justify full-100. The best registered local full-100 remains `2.748615e-07`; the best registered hosted public-50 remains `2.787552892e-07` (#316192).

### Not Promoted After Ranking

| Family | Run | Step | SHA | First-10 adjusted | First-10 MSE | C/B | Fail | vs initial recorded pct | Decision | Reason |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `gaussiancv` | `whest_0718_gaussiancv_auto_r2` | 6 | `6dc837f7f1c0` | 1.482346196e-07 | 5.998364458e-07 | 0.2472 | 0 | -0.083% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `gaussiancv` | `whest_0718_gaussiancv_auto_r2` | 6 | `d31811881afd` | 1.484752907e-07 | 5.998364458e-07 | 0.2475 | 0 | -0.245% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `affinef64` | `whest_0718_affinef64_ttt` | 28 | `a2ddb349ffa2` | 1.922600318e-07 | 1.287937016e-06 | 0.1493 | 0 | 15.192% | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `affinef64` | `whest_0718_affinef64_ttt` | 27 | `767bc7163286` | 1.923352920e-07 | 1.287937016e-06 | 0.1494 | 0 | 15.147% | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `affinef64` | `whest_0718_affinef64_ttt` | 29 | `9a6610fd6cb7` | 1.924786717e-07 | 1.287937016e-06 | 0.1495 | 0 | 15.061% | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `affinef64` | `whest_0718_affinef64_ttt` | 26 | `e0e01421b317` | 1.927450747e-07 | 1.287937016e-06 | 0.1497 | 0 | 14.902% | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `affinef64` | `whest_0718_affinef64_ttt` | 28 | `740733c4b202` | 2.049826372e-07 | 1.288139066e-06 | 0.1590 | 0 | 8.043% | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | 27 | `46b40b447442` | 2.207199908e-07 | 2.168735170e-06 | 0.1017 | 0 | 0.829% | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | 31 | `a4ad4bfc85d4` | 2.364664917e-07 | 2.319162974e-06 | 0.1020 | 0 | -5.885% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | 29 | `3551143464bf` | 2.368137433e-07 | 2.319162974e-06 | 0.1021 | 0 | -6.023% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `german013` | `whest_0718_german013_ttt` | 30 | `a435dc08941f` | 2.428689538e-07 | 1.842124703e-06 | 0.1317 | 0 | 3.598% | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | 28 | `f2ed09917ea3` | 2.583112050e-07 | 2.526054863e-06 | 0.1023 | 0 | -13.844% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | 27 | `ac9350ace1a9` | 2.606403599e-07 | 2.602109237e-06 | 0.1002 | 0 | -14.614% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `rqmc030` | `whest_0718_rqmc030_ttt` | 24 | `ba5f0bed0715` | 2.820544090e-07 | 6.457273486e-07 | 0.4370 | 0 | 61.761% | `reject_ranked_not_full100` | Strong first-10 gain (61.8% by recorded ratio), but prior same-family representatives already regressed on full-100 around 5.237e-07; not enough new mechanism to spend validation. |

Additional no-score workspace submissions were inventoried but rejected because no verified first-10 construction/message score was available by the cutoff:

| Family | Run | Step | SHA | Source | Reason |
|---|---|---:|---|---|---|
| `affinef64` | `whest_0718_affinef64_auto_r1` | 3 | `41b31b2e70d4` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r1/autoevolve_workspaces/step_000003/call_0001_730e4714/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `affinef64` | `whest_0718_affinef64_auto_r2` | 4 | `d715bd1eda18` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r2/autoevolve_workspaces/step_000004/call_0001_b6b6dabf/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `german013` | `whest_0718_german013_auto_r2` | 2 | `3d7a501d02e2` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r2/autoevolve_workspaces/step_000002/call_0001_ee997536/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `layerrescale` | `whest_0718_layerrescale_auto_r2` | 2 | `32d14d37e76b` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_layerrescale_auto_r2/autoevolve_workspaces/step_000002/call_0001_8e4dda7e/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `layerrescale` | `whest_0718_layerrescale_auto_r1` | 2 | `8e363573b9f5` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_layerrescale_auto_r1/autoevolve_workspaces/step_000002/call_0001_b17d4871/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `rqmc030` | `whest_0718_rqmc030_auto_r1` | 2 | `2f92384ca6d5` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r1/autoevolve_workspaces/step_000002/call_0001_02996588/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `rqmc030` | `whest_0718_rqmc030_auto_r2` | 2 | `7afc7f2ed582` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r2/autoevolve_workspaces/step_000002/call_0001_11e439ac/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `sphericalrb` | `whest_0718_sphericalrb_auto_r2` | 2 | `8941d9dfedf6` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r2/autoevolve_workspaces/step_000002/call_0001_e83a3287/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `sphericalrb` | `whest_0718_sphericalrb_auto_r1` | 2 | `d4cd6e2d8521` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r1/autoevolve_workspaces/step_000002/call_0001_a0cfffca/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |

Interpretation:
- GaussianCV new candidates were at `1.482346196e-07` and `1.484752907e-07`, both slightly worse than the same-family initial first-10 `1.481113354e-07`.
- AffineF64 TTT candidates improved only about `15%` first-10 versus initial and are setup-cached whitening variants without enough new mechanism; prior affine representatives with stronger first-10 already produced full-100 around `3.216806472e-07`.
- RQMC030 TTT candidates had larger recorded first-10 gains (`50.6%` and `61.8%`) but first-10 adjusted scores of `3.029370858e-07` and `2.820544090e-07` are weaker than previously full-100-tested same-family representatives that regressed to `4.553e-07` to `5.341e-07`.
- SphericalRB and German TTT tail states were within noise or worse than initial/parent baselines. No first-10 failures were promoted.

### Official Submission Status

- No local full-100 command was run, no package was built, and no `whest submit` was run because no non-duplicate candidate cleared the first-10 quality and novelty bar.
- No new submission ID was obtained; registry was not changed after the required pre-review tracker refresh.
- Live official submission would also require credential/quota recovery because registered submission refreshes still report HTTP 401.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-18T20:14:04+00:00`; do not treat this cycle's `289` inventoried SHA as new.
- Latest scanned TTT steps at cutoff: affinef64 `30`, gaussiancv `50`, german013 `31`, layerrescale `50`, rqmc030 `24`, sphericalrb `31`.
- Latest scanned AutoEvolve pool steps at cutoff: `whest_0718_affinef64_auto_r1` `4`, `whest_0718_affinef64_auto_r2` `5`, `whest_0718_gaussiancv_auto_r1` `6`, `whest_0718_gaussiancv_auto_r2` `7`, `whest_0718_german013_auto_r1` `2`, `whest_0718_german013_auto_r2` `3`, `whest_0718_layerrescale_auto_r1` `3`, `whest_0718_layerrescale_auto_r2` `3`, `whest_0718_rqmc030_auto_r1` `3`, `whest_0718_rqmc030_auto_r2` `3`, `whest_0718_sphericalrb_auto_r1` `3`, `whest_0718_sphericalrb_auto_r2` `3`.
- Include artifacts excluded after cutoff next round, starting with: `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r1/autoevolve_workspaces/step_000004/call_0001_53b132b6/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r2/autoevolve_workspaces/step_000005/call_0001_c711fdfe/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r1/autoevolve_workspaces/step_000006/call_0001_52112175/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r2/autoevolve_workspaces/step_000007/call_0001_f5070340/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r2/autoevolve_workspaces/step_000003/call_0001_61e2f15f/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_layerrescale_auto_r2/autoevolve_workspaces/step_000003/call_0001_46ad9438/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r1/autoevolve_workspaces/step_000003/call_0001_9d3b9aaf/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r2/autoevolve_workspaces/step_000003/call_0001_2a9f4717/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r1/autoevolve_workspaces/step_000003/call_0001_73bdde20/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r2/autoevolve_workspaces/step_000003/call_0001_d6beb25c/submission.py`.

## Cycle 20260718T205028Z hourly submission review

- Completed at UTC: `2026-07-18T21:02:18.017900+00:00`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- Cached official status counts after refresh: `{'failed': 1, 'graded': 24}`. Pending/queued/running registered submissions: `[]`. No historical pending status changes needed ledger backfill.
- AIcrowd refresh/auth status: `AIcrowdAuthError HTTP 401 for registered submissions; server quota/status unavailable`.
- Quota assessment for UTC `2026-07-18`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`.
- Candidate cutoff UTC: `2026-07-18T20:50:28+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-18T20:48:35.712706+00:00']`.
- JSONL context files were read for sampler/metrics/agent/gen-score line counts while live runs continued; live mtime range was `['2026-07-18T17:40:25.175395+00:00', '2026-07-18T20:52:51.179422+00:00']`.
- Console logs inspected: `18` files under `codex_runs/aicrowd_whestbench/multi_initial_20260718/console`; mtime range UTC `['2026-07-18T14:51:18.876898+00:00', '2026-07-18T20:53:51.907592+00:00']`. Recent notable console lines are RQMC030 TTT codex-exec timeouts, but no candidate was promoted solely from console text.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T205028Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T205028Z/scan_summary.json`
- Shortlist manifest: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T205028Z/shortlist_manifest.json`
- Shortlist sources: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T205028Z/shortlist_sources`
- Full-100 reports: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T205028Z/full100_reports`
- Initial estimator SHA baselines scanned under `initial_states`: `{'affinef64': '1cb0eb3a875702386c346d605880190c7827b392860c7ddb6fb690f9f9b71435', 'gaussiancv': 'd49b8b5218ff9b9462a331812e688e125fcfff1f86e8a26e9be9c8f59eb48ebb', 'german013': '68d409403de29c8fa7807a8274066494e9ce00380bbbe5bf3d1d52e4ba35aa64', 'layerrescale': '98da6a0882cbc02d32cbd20e295c80b21bb8c173e3e23bb0d6880233d611d7ff', 'rqmc030': 'dc60550f76e9f4f6cb5940af295cab777a4082412590a49665e9127b535eb19f', 'sphericalrb': '146a8c786ee31dfefc941712ded736c3ec2625343dd5cd184f80dde915a9d130'}`
- Files excluded after cutoff: `14`; examples: `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r1/autoevolve_workspaces/step_000004/call_0001_53b132b6/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r2/autoevolve_workspaces/step_000006/call_0001_9214b66d/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_ttt/puct_sampler_step_000035.json`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r1/autoevolve_workspaces/step_000007/call_0001_2be61919/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r2/autoevolve_workspaces/step_000007/call_0001_f5070340/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r2/autoevolve_workspaces/step_000003/call_0001_61e2f15f/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_ttt/puct_sampler_step_000035.json`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_layerrescale_auto_r2/autoevolve_workspaces/step_000003/call_0001_46ad9438/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r1/autoevolve_workspaces/step_000004/call_0001_7379ea17/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r2/autoevolve_workspaces/step_000004/call_0001_e7c1f4f3/submission.py`.

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count | Excluded after cutoff |
|---|---|---|---:|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | - | 0-4 | 4 | 9 | 1 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | - | 0-6 | 6 | 13 | 1 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | 0-34 | - | 0 | 35 | 1 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | - | 0-7 | 7 | 15 | 1 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | - | 0-7 | 7 | 15 | 1 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | - | 0-2 | 3 | 6 | 0 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | - | 0-3 | 3 | 7 | 1 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | 0-34 | - | 0 | 35 | 1 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | - | 0-3 | 4 | 8 | 0 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | - | 0-3 | 3 | 7 | 1 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | - | 0-4 | 4 | 9 | 1 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | - | 0-4 | 4 | 9 | 1 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | 0-27 | - | 0 | 28 | 0 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | - | 0-3 | 3 | 7 | 1 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | - | 0-3 | 4 | 8 | 2 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | 0-35 | - | 0 | 36 | 1 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `319`.
- Historical/known SHA baseline size: `501`; old inventory entries read: `1100`.
- New SHA after historical baseline removal: `30`.
- Final decision counts: `{'reject_existing_sha': 289, 'reject_full100_no_submit': 1, 'reject_insufficient_first10_gain_not_full100': 14, 'reject_no_first10': 1, 'reject_ranked_not_full100': 5, 'reject_weak_first10': 9}`.
- Entered full-100: `1`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Shortlist Decisions

| SHA | Family / run / step | State | First-10 adjusted / MSE / C-B / fail | Full-100 adjusted / MSE / C-B / fail | Decision | Candidate path |
|---|---|---|---|---|---|---|
| `053fcb44a33e` | `rqmc030` / `whest_0718_rqmc030_auto_r2` / step `4` | `3c623715-0213-4eda-be95-c0932e1318e3` | 1.845700202e-07 / 4.618521587e-07 / 3.9969e-01 / 0 | 4.556810377e-07 / 1.139276179e-06 / 3.9999e-01 / 0 | `reject_full100_no_submit` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T205028Z/shortlist_sources/rqmc030_candidate_053fcb44a33e.py` |

Full-100 interpretation:
- `053fcb44a33e`: antithetic lattice/RQMC with first-layer control variate, input second-moment normalization, and setup-cached generator. First-10 adjusted `1.845700202e-07` improved strongly versus the RQMC030 initial (`147.199%` recorded improvement) but only `0.067%` versus its immediate parent.
- Full-100 report `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T205028Z/full100_reports/rqmc030_control_variate_053fcb44a33e_full100_20260718T205028Z.json`: adjusted `4.556810377e-07`, MSE `1.139276179e-06`, C/B `3.9999e-01`, failures `0`, failure_breakdown `{'budget_exhausted': 0, 'combined_budget_exhausted': 0, 'error': 0, 'residual_wall_time_exhausted': 0, 'time_exhausted': 0}`, max effective compute `1.0941e+11`.
- Rejected for official submission because full-100 is weaker than local full-100 best `2.748615e-07` and registered hosted public-50 best `2.787552892e-07` (#316192).

### Not Promoted After Ranking

| Family | Run | Step | SHA | First-10 adjusted | First-10 MSE | C/B | Fail | vs initial recorded pct | Decision | Reason |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `gaussiancv` | `whest_0718_gaussiancv_auto_r1` | `7` | `736eef6390ca` | 1.479341798e-07 | 5.997743500e-07 | 2.4667e-01 | 0 | 0.120% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `gaussiancv` | `whest_0718_gaussiancv_auto_r1` | `7` | `634eea7f108b` | 1.481652766e-07 | 5.997743500e-07 | 2.4710e-01 | 0 | -0.036% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `rqmc030` | `whest_0718_rqmc030_auto_r2` | `4` | `55569137a590` | 1.845861398e-07 | 4.618521587e-07 | 3.9970e-01 | 0 | 147.177% | `reject_ranked_not_full100` | Lower-ranked same-family/method variant; a stronger representative was selected for full-100. |
| `rqmc030` | `whest_0718_rqmc030_auto_r1` | `4` | `73a706f59871` | 1.902755844e-07 | 2.759532286e-07 | 6.8985e-01 | 0 | 139.786% | `reject_ranked_not_full100` | Lower-ranked same-family/method variant; a stronger representative was selected for full-100. |
| `rqmc030` | `whest_0718_rqmc030_auto_r1` | `4` | `1e8f41c05078` | 1.910784029e-07 | 2.759532286e-07 | 6.9280e-01 | 0 | 138.779% | `reject_ranked_not_full100` | Lower-ranked same-family/method variant; a stronger representative was selected for full-100. |
| `affinef64` | `whest_0718_affinef64_ttt` | `32` | `055ad3fce4db` | 1.922532937e-07 | 1.287937016e-06 | 1.4930e-01 | 0 | 15.196% | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `affinef64` | `whest_0718_affinef64_ttt` | `34` | `b4ea9f51fa16` | 1.922936250e-07 | 1.287937016e-06 | 1.4932e-01 | 0 | 15.172% | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `affinef64` | `whest_0718_affinef64_ttt` | `33` | `9d4135e8e3ef` | 1.923481440e-07 | 1.287937016e-06 | 1.4938e-01 | 0 | 15.139% | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `affinef64` | `whest_0718_affinef64_ttt` | `32` | `35e947c0feee` | 1.943767445e-07 | 1.287937016e-06 | 1.5130e-01 | 0 | 13.938% | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `affinef64` | `whest_0718_affinef64_ttt` | `32` | `8122eeb095f8` | 1.944222314e-07 | 1.287937016e-06 | 1.5139e-01 | 0 | 13.911% | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `affinef64` | `whest_0718_affinef64_ttt` | `31` | `8f0920cf5700` | 1.948676429e-07 | 1.042938283e-06 | 1.8697e-01 | 0 | 13.651% | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | `34` | `2c4d2a6e6915` | 2.203972550e-07 | 2.168735170e-06 | 1.0158e-01 | 0 | 0.976% | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | `34` | `1a0bc959b628` | 2.206255693e-07 | 2.168735170e-06 | 1.0168e-01 | 0 | 0.872% | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | `35` | `d61805d20081` | 2.207044608e-07 | 2.168735170e-06 | 1.0171e-01 | 0 | 0.836% | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |

Additional no-score workspace/submission candidates:

| Family | Run | Step | SHA | Source | Reason |
|---|---|---:|---|---|---|
| `sphericalrb` | `whest_0718_sphericalrb_auto_r2` | `3` | `f10765586df7` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r2/autoevolve_workspaces/step_000003/call_0001_d6beb25c/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |

Interpretation:
- The only promoted RQMC030 candidate reproduced the prior pattern: very strong first-10, but full-100 regressed to the `4.56e-07` band, close to the previous RQMC030 full-100 rejections (`4.553e-07` to `5.341e-07`).
- GaussianCV candidates remained at or slightly worse than their same-family initial first-10. AffineF64 and SphericalRB TTT tails were small/noisy variants.
- No candidate showed a full-100 improvement large enough to justify using an official hosted public-50 submission; no package was built.

### Official Submission Status

- No `whest package` or `whest submit` was run because the only full-100 candidate failed the local quality bar.
- No new submission ID was obtained; registry was not changed after the required pre-review tracker refresh.
- Live official submission would also require credential/quota recovery because registered submission refreshes still report HTTP 401.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-18T20:50:28+00:00`; do not treat this cycle's `319` inventoried SHA or `30` new SHA as new.
- Latest scanned steps/pools at cutoff: whest_0718_affinef64_auto_r1 pool `4`; whest_0718_affinef64_auto_r2 pool `6`; whest_0718_affinef64_ttt TTT `34`; whest_0718_gaussiancv_auto_r1 pool `7`; whest_0718_gaussiancv_auto_r2 pool `7`; whest_0718_gaussiancv_ttt TTT `50`; whest_0718_german013_auto_r1 pool `2`; whest_0718_german013_auto_r2 pool `3`; whest_0718_german013_ttt TTT `34`; whest_0718_layerrescale_auto_r1 pool `3`; whest_0718_layerrescale_auto_r2 pool `3`; whest_0718_layerrescale_ttt TTT `50`; whest_0718_rqmc030_auto_r1 pool `4`; whest_0718_rqmc030_auto_r2 pool `4`; whest_0718_rqmc030_ttt TTT `27`; whest_0718_sphericalrb_auto_r1 pool `3`; whest_0718_sphericalrb_auto_r2 pool `3`; whest_0718_sphericalrb_ttt TTT `35`.
- Include artifacts excluded after cutoff next round, starting with: `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r1/autoevolve_workspaces/step_000004/call_0001_53b132b6/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r2/autoevolve_workspaces/step_000006/call_0001_9214b66d/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_ttt/puct_sampler_step_000035.json`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r1/autoevolve_workspaces/step_000007/call_0001_2be61919/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r2/autoevolve_workspaces/step_000007/call_0001_f5070340/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r2/autoevolve_workspaces/step_000003/call_0001_61e2f15f/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_ttt/puct_sampler_step_000035.json`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_layerrescale_auto_r2/autoevolve_workspaces/step_000003/call_0001_46ad9438/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r1/autoevolve_workspaces/step_000004/call_0001_7379ea17/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r2/autoevolve_workspaces/step_000004/call_0001_e7c1f4f3/submission.py`.

## Cycle 20260718T213309Z hourly submission review

- Completed at UTC: `2026-07-18T21:47:51.530689+00:00`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- Cached official status counts after refresh: `{'failed': 1, 'graded': 24}`. Pending/queued/running registered submissions: `[]`. No historical pending status changes needed ledger backfill.
- Quota assessment for UTC `2026-07-18`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`.
- AIcrowd refresh/auth status: `AIcrowdAuthError HTTP 401 for registered submissions; server quota/status unavailable`.
- Candidate cutoff UTC: `2026-07-18T21:33:09+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-18T21:30:56.779938+00:00']`.
- JSONL context files were read for sampler/metrics/agent/gen-score line counts while live runs continued; live mtime range UTC: `['2026-07-18T17:40:25.175395+00:00', '2026-07-18T21:30:56.780938+00:00']`.
- Console logs inspected: `18` files under `codex_runs/aicrowd_whestbench/multi_initial_20260718/console`; mtime range UTC `['2026-07-18T14:51:18.876898+00:00', '2026-07-18T21:47:33.036555+00:00']`. Recent notable console lines were RQMC030 TTT codex-exec timeout/reconnect messages; no candidate was promoted solely from console text.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T213309Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T213309Z/scan_summary.json`
- Shortlist manifest: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T213309Z/shortlist_manifest.json`
- Shortlist sources: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T213309Z/shortlist_sources`
- Full-100 reports: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T213309Z/full100_reports`
- Initial estimator SHA baselines scanned under `initial_states`: `{'affinef64': '1cb0eb3a875702386c346d605880190c7827b392860c7ddb6fb690f9f9b71435', 'gaussiancv': 'd49b8b5218ff9b9462a331812e688e125fcfff1f86e8a26e9be9c8f59eb48ebb', 'german013': '68d409403de29c8fa7807a8274066494e9ce00380bbbe5bf3d1d52e4ba35aa64', 'layerrescale': '98da6a0882cbc02d32cbd20e295c80b21bb8c173e3e23bb0d6880233d611d7ff', 'rqmc030': 'dc60550f76e9f4f6cb5940af295cab777a4082412590a49665e9127b535eb19f', 'sphericalrb': '146a8c786ee31dfefc941712ded736c3ec2625343dd5cd184f80dde915a9d130'}`
- Files excluded after cutoff: `7`; examples: `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r1/autoevolve_workspaces/step_000004/call_0001_53b132b6/submission.py; codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r1/autoevolve_workspaces/step_000007/call_0001_2be61919/submission.py; codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r2/autoevolve_workspaces/step_000008/call_0001_685e8d6f/submission.py; codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r2/autoevolve_workspaces/step_000003/call_0001_61e2f15f/submission.py; codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r1/autoevolve_workspaces/step_000004/call_0001_7379ea17/submission.py; codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r2/autoevolve_workspaces/step_000004/call_0001_e7c1f4f3/submission.py; codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r1/autoevolve_workspaces/step_000003/call_0001_73bdde20/submission.py`

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count | Excluded after cutoff |
|---|---|---|---:|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | - | 0-4 | 4 | 9 | 1 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | 0-39 | - | 0 | 40 | 0 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | - | 0-7 | 7 | 15 | 1 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | - | 0-8 | 8 | 17 | 1 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | - | 0-2 | 3 | 6 | 0 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | - | 0-3 | 3 | 7 | 1 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | 0-39 | - | 0 | 40 | 0 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | - | 0-3 | 4 | 8 | 0 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | - | 0-4 | 5 | 10 | 0 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | - | 0-4 | 4 | 9 | 1 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | - | 0-4 | 4 | 9 | 1 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | 0-31 | - | 0 | 32 | 0 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | - | 0-3 | 3 | 7 | 1 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | - | 0-4 | 5 | 10 | 0 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | 0-40 | - | 0 | 41 | 0 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `351`.
- Historical/known SHA baseline size: `531`; old inventory entries read: `1419`.
- New SHA after historical baseline removal: `32`.
- Final decision counts: `{'reject_existing_sha': 319, 'reject_full100_no_submit': 2, 'reject_insufficient_first10_gain_not_full100': 11, 'reject_no_first10': 3, 'reject_ranked_not_full100': 6, 'reject_weak_first10': 10}`.
- Entered full-100: `2`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Shortlist Decisions

| SHA | Family / run / step | State | First-10 adjusted / MSE / C-B / fail | Full-100 adjusted / MSE / C-B / fail | Decision | Candidate path |
|---|---|---|---|---|---|---|
| `e2991497f6b9` | `layerrescale` / `whest_0718_layerrescale_auto_r2` / step `4` | `3ff3e424-430e-4a82-9bdf-6481facad4fc` | 1.362059685e-07 / 1.359472594e-06 / 1.0005e-01 / 0 | 3.753050031e-07 / 3.752679123e-06 / 9.9827e-02 / 0 | `reject_full100_no_submit` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T213309Z/shortlist_sources/layerrescale_candidate_e2991497f6b9.py` |
| `e5c42f02f69c` | `sphericalrb` / `whest_0718_sphericalrb_auto_r2` / step `4` | `6fec0212-fd0e-48d7-8096-80a568c3437d` | 1.553344000e-07 / 1.523848090e-06 / 1.0179e-01 / 0 | 4.004914069e-07 / 3.945708578e-06 / 1.0151e-01 / 0 | `reject_full100_no_submit` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T213309Z/shortlist_sources/sphericalrb_candidate_e5c42f02f69c.py` |

Full-100 interpretation:
- `e2991497f6b9`: source `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_layerrescale_auto_r2/autoevolve_pool_step_000004.json`, report `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T213309Z/full100_reports/layerrescale_rescaled_whitening_e2991497f6b9_full100_20260718T213309Z.json`. First-10 adjusted `1.362059685e-07` versus initial recorded improvement `4.539e+01%`; full-100 adjusted `3.753050031e-07`, MSE `3.752679123e-06`, C/B `9.9827e-02`, failures `0`, failure_breakdown `{'budget_exhausted': 0, 'combined_budget_exhausted': 0, 'error': 0, 'residual_wall_time_exhausted': 0, 'time_exhausted': 0}`.
- `e5c42f02f69c`: source `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r2/autoevolve_pool_step_000004.json`, report `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T213309Z/full100_reports/sphericalrb_lattice_blend_e5c42f02f69c_full100_20260718T213309Z.json`. First-10 adjusted `1.553344000e-07` versus initial recorded improvement `4.327e+01%`; full-100 adjusted `4.004914069e-07`, MSE `3.945708578e-06`, C/B `1.0151e-01`, failures `0`, failure_breakdown `{'budget_exhausted': 0, 'combined_budget_exhausted': 0, 'error': 0, 'residual_wall_time_exhausted': 0, 'time_exhausted': 0}`.
- Both full-100 candidates had 0 failures and no budget/time exhaustion flags, but both are weaker than registered local full-100 best `2.748615e-07` and best registered hosted public-50 `2.787552892e-07` (#316192).

### Not Promoted After Ranking

| Family | Run | Step | SHA | First-10 adjusted | First-10 MSE | C/B | Fail | Decision | Reason |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| `layerrescale` | `whest_0718_layerrescale_auto_r2` | `4` | `5748976c25ca` | 1.362415993e-07 | 1.359472594e-06 | 9.9848e-02 | 0 | `reject_ranked_not_full100` | Lower-ranked same-family/method variant; a stronger representative was selected for full-100. |
| `gaussiancv` | `whest_0718_gaussiancv_auto_r2` | `8` | `46769aa4d43d` | 1.474597939e-07 | 5.995594648e-07 | 2.4607e-01 | 0 | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `gaussiancv` | `whest_0718_gaussiancv_auto_r2` | `8` | `a227df379888` | 1.485965486e-07 | 5.995594648e-07 | 2.4782e-01 | 0 | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `affinef64` | `whest_0718_affinef64_ttt` | `39` | `5ea877e48850` | 1.786369370e-07 | 1.188197484e-06 | 1.5032e-01 | 0 | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `affinef64` | `whest_0718_affinef64_ttt` | `39` | `b0b8e7e30703` | 1.827885369e-07 | 4.391215242e-07 | 4.1627e-01 | 0 | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `affinef64` | `whest_0718_affinef64_ttt` | `39` | `b79bb2227848` | 1.922184477e-07 | 1.287937016e-06 | 1.4927e-01 | 0 | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `affinef64` | `whest_0718_affinef64_ttt` | `35` | `d16646c376d7` | 1.922533589e-07 | 1.287937016e-06 | 1.4931e-01 | 0 | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `affinef64` | `whest_0718_affinef64_ttt` | `38` | `9d46d8c21c6b` | 1.922856931e-07 | 1.287937016e-06 | 1.4932e-01 | 0 | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `affinef64` | `whest_0718_affinef64_ttt` | `38` | `e5b636c22c5b` | 1.930401320e-07 | 1.287937016e-06 | 1.4992e-01 | 0 | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | `37` | `42ddb5419824` | 2.173658667e-07 | 2.168735170e-06 | 1.0008e-01 | 0 | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | `38` | `c087f03f31d2` | 2.173910486e-07 | 2.168735170e-06 | 1.0006e-01 | 0 | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | `37` | `df89c8a9e4a0` | 2.206242170e-07 | 2.168735170e-06 | 1.0168e-01 | 0 | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | `39` | `538d445061d8` | 2.344207397e-07 | 2.325468705e-06 | 1.0097e-01 | 0 | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | `38` | `0231ecd8570b` | 2.367314747e-07 | 2.343969703e-06 | 1.0126e-01 | 0 | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `german013` | `whest_0718_german013_ttt` | `39` | `f6380b53015a` | 2.428395746e-07 | 1.842124703e-06 | 1.3166e-01 | 0 | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | `40` | `efe1725dd16a` | 2.490885005e-07 | 2.488725090e-06 | 9.9338e-02 | 0 | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `rqmc030` | `whest_0718_rqmc030_ttt` | `28` | `26131946f8ad` | 2.500311887e-07 | 6.785593371e-07 | 3.6618e-01 | 0 | `reject_ranked_not_full100` | Strong first-10 gain (82.5% by recorded ratio), but prior same-family representatives already regressed on full-100 around 5.237e-07; not enough new mechanism to spend validation. |
| `german013` | `whest_0718_german013_ttt` | `37` | `ba3a40308e46` | 2.521127083e-07 | 1.555990218e-06 | 1.6194e-01 | 0 | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |

Additional no-score workspace/submission candidates:

| Family | Run | Step | SHA | Source | Reason |
|---|---|---:|---|---|---|
| `affinef64` | `whest_0718_affinef64_auto_r2` | `6` | `8c3eaccbebf1` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r2/autoevolve_workspaces/step_000006/call_0001_9214b66d/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `layerrescale` | `whest_0718_layerrescale_auto_r2` | `4` | `e6b3d43eb51a` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_layerrescale_auto_r2/autoevolve_workspaces/step_000004/call_0001_4dbfe4d7/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `sphericalrb` | `whest_0718_sphericalrb_auto_r2` | `4` | `f54137ca122a` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r2/autoevolve_workspaces/step_000004/call_0001_9b820744/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |

Interpretation:
- LayerRescale step 4 produced the best first-10 band this cycle (`1.362e-07`), but full-100 regressed to `3.753e-07`; its sibling `5748976c25ca` was not separately validated as a lower-ranked same-family/method variant.
- SphericalRB AutoEvolve step 4 introduced lattice/RQMC angular whitening and final blending, but full-100 regressed to `4.005e-07`.
- GaussianCV candidates stayed at/near the same-family initial first-10; AffineF64 and Spherical/RQMC TTT tail states were either small first-10 gains or repeated families whose earlier representatives already failed full-100.
- No candidate showed a full-100 improvement large enough to justify package/upload; no package was built.

### Official Submission Status

- No `whest package` or `whest submit` was run because both full-100 candidates failed the local quality bar.
- No new submission ID was obtained; registry was not changed after the required pre-review tracker refresh.
- Live official submission would also require credential/quota recovery because registered submission refreshes still report HTTP 401, so server quota could not be verified.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-18T21:33:09+00:00`; do not treat this cycle's `351` inventoried SHA or `32` new SHA as new.
- Latest scanned steps/pools at cutoff: `whest_0718_affinef64_auto_r1` pool `4`; `whest_0718_affinef64_auto_r2` pool `6`; `whest_0718_affinef64_ttt` TTT `39`; `whest_0718_gaussiancv_auto_r1` pool `7`; `whest_0718_gaussiancv_auto_r2` pool `8`; `whest_0718_gaussiancv_ttt` TTT `50`; `whest_0718_german013_auto_r1` pool `2`; `whest_0718_german013_auto_r2` pool `3`; `whest_0718_german013_ttt` TTT `39`; `whest_0718_layerrescale_auto_r1` pool `3`; `whest_0718_layerrescale_auto_r2` pool `4`; `whest_0718_layerrescale_ttt` TTT `50`; `whest_0718_rqmc030_auto_r1` pool `4`; `whest_0718_rqmc030_auto_r2` pool `4`; `whest_0718_rqmc030_ttt` TTT `31`; `whest_0718_sphericalrb_auto_r1` pool `3`; `whest_0718_sphericalrb_auto_r2` pool `4`; `whest_0718_sphericalrb_ttt` TTT `40`.
- Include artifacts excluded after cutoff next round, starting with: `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r1/autoevolve_workspaces/step_000004/call_0001_53b132b6/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r1/autoevolve_workspaces/step_000007/call_0001_2be61919/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r2/autoevolve_workspaces/step_000008/call_0001_685e8d6f/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r2/autoevolve_workspaces/step_000003/call_0001_61e2f15f/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r1/autoevolve_workspaces/step_000004/call_0001_7379ea17/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r2/autoevolve_workspaces/step_000004/call_0001_e7c1f4f3/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r1/autoevolve_workspaces/step_000003/call_0001_73bdde20/submission.py`.


## Cycle 20260718T221831Z hourly submission review

- Completed at UTC: `2026-07-18T22:30:52.127386+00:00`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- Cached official status counts after refresh: `{'graded': 24, 'failed': 1}`. Pending/queued/running registered submissions: `[]`. No historical pending status changes needed ledger backfill.
- AIcrowd refresh/auth status: `AIcrowdAuthError HTTP 401 for registered submissions; server quota/status unavailable`.
- Quota assessment for UTC `2026-07-18`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`. Blocker: `AIcrowdAuthError: While checking submission status: Your AIcrowd API key is missing or invalid. (HTTP 401)`.
- Candidate cutoff UTC: `2026-07-18T22:18:31+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-18T22:16:16.935216+00:00']`.
- JSONL context files were read for sampler/metrics/agent/gen-score line counts while live runs continued; live mtime range UTC: `['2026-07-18T17:40:25.175395+00:00', '2026-07-18T22:16:16.936216+00:00']`.
- Console logs inspected: `18` files under `codex_runs/aicrowd_whestbench/multi_initial_20260718/console`; mtime range UTC `['2026-07-18T14:51:18.876898+00:00', '2026-07-18T22:14:28.179908+00:00']`. Recent logs include Codex reconnect warnings; no candidate was promoted solely from console text.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T221831Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T221831Z/scan_summary.json`
- Shortlist manifest: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T221831Z/shortlist_manifest.json`
- Shortlist sources: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T221831Z/shortlist_sources`
- Full-100 reports: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T221831Z/full100_reports`
- Initial estimator SHA baselines scanned under `initial_states`: `{'affinef64': '1cb0eb3a875702386c346d605880190c7827b392860c7ddb6fb690f9f9b71435', 'gaussiancv': 'd49b8b5218ff9b9462a331812e688e125fcfff1f86e8a26e9be9c8f59eb48ebb', 'german013': '68d409403de29c8fa7807a8274066494e9ce00380bbbe5bf3d1d52e4ba35aa64', 'layerrescale': '98da6a0882cbc02d32cbd20e295c80b21bb8c173e3e23bb0d6880233d611d7ff', 'rqmc030': 'dc60550f76e9f4f6cb5940af295cab777a4082412590a49665e9127b535eb19f', 'sphericalrb': '146a8c786ee31dfefc941712ded736c3ec2625343dd5cd184f80dde915a9d130'}`
- Files excluded after cutoff: `8`.

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count | Excluded after cutoff |
|---|---|---|---:|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | - | 0-5 | 5 | 11 | 1 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | 0-46 | - | 0 | 47 | 0 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | - | 0-9 | 9 | 19 | 1 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | - | 0-10 | 10 | 21 | 1 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | - | 0-4 | 5 | 10 | 0 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | - | 0-3 | 3 | 7 | 1 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | 0-44 | - | 0 | 45 | 0 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | - | 0-3 | 4 | 8 | 0 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | - | 0-4 | 4 | 9 | 1 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | - | 0-6 | 6 | 13 | 1 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | - | 0-5 | 5 | 11 | 1 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | 0-35 | - | 0 | 36 | 0 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | - | 0-4 | 4 | 9 | 1 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | - | 0-4 | 5 | 10 | 0 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | 0-46 | - | 0 | 47 | 0 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `385`.
- Historical/known SHA baseline size: `563`; old inventory entries read: `1770`.
- New SHA after historical baseline removal: `35`.
- Final decision counts: `{'reject_existing_sha': 350, 'reject_full100_no_submit': 1, 'reject_insufficient_first10_gain_not_full100': 7, 'reject_no_first10': 6, 'reject_ranked_not_full100': 9, 'reject_weak_first10': 12}`.
- Entered full-100: `1`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Shortlist Decisions

| SHA | Family / run / step | State | First-10 adjusted / MSE / C-B / fail | Full-100 adjusted / MSE / C-B / fail | Decision | Candidate path |
|---|---|---|---|---|---|---|
| `c0c17fa57f1f` | `rqmc030` / `whest_0718_rqmc030_auto_r2` / step `5` | `bbee3c36-5007-462e-bc8e-1b349169ddb0` | 1.830277595e-07 / 4.578615325e-07 / 3.9978e-01 / 0 | 4.572027097e-07 / 1.144356552e-06 / 3.9955e-01 / 0 | `reject_full100_no_submit` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T221831Z/shortlist_sources/rqmc030_candidate_c0c17fa57f1f.py` |

Full-100 interpretation:
- `c0c17fa57f1f`: Antithetic RQMC estimator with a first-layer control variate.; features `setup_cached, antithetic, rqmc/sobol, whitening/eigh, control_variate, gaussian_final_blend, layer_rescale, affine_float64`; constants `_ARRAY_BYTES_LIMIT=200 * 1024 * 1024, _FIRST_CONTROL_SCALE=0.395, _FIRST_VARIANCE_BLEND=0.45, _FLOAT32_ITEMSIZE=4, _INPUT_SECOND_POWER=0.925, _OFFDIAGONAL_CONTROL_SCALE=-0.05, _SHIFT_COUNT=4, _TAIL_CLIP=0.0001, _TARGET_FLOP_FRACTION=0.4`.
- First-10 source was verified from construction with 0 failures, adjusted `1.830277595e-07`, final-layer MSE `4.578615325e-07`, C/B `3.9978e-01`.
- Full-100 report `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T221831Z/full100_reports/rqmc030_control_variate_c0c17fa57f1f_full100_20260718T221831Z.json`: adjusted `4.572027097e-07`, final-layer MSE `1.144356552e-06`, all-layer MSE `7.807254040e-01`, C/B `3.9955e-01`, failures `0`, failure_breakdown `{'budget_exhausted': 0, 'combined_budget_exhausted': 0, 'error': 0, 'residual_wall_time_exhausted': 0, 'time_exhausted': 0}`, elapsed `318.8s`, max effective compute `1.0936e+11`.
- Relative first-10 change: vs RQMC030 initial `149.282%`; vs parent `0.843%` against parent SHA `053fcb44a33e`.
- Rejected for official submission because full-100 is weaker than registered local full-100 best `2.748615e-07` and registered hosted public-50 best `2.787552892e-07` (#316192). This continues the observed RQMC030 pattern: strong first-10 but full-100 regression around `4.55e-07`.

### Not Promoted After Ranking

| Family | Run | Step | SHA | First-10 adjusted | First-10 MSE | C/B | Fail | Decision | Reason |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| `gaussiancv` | `whest_0718_gaussiancv_auto_r1` | 8 | `278b8e2e1a89` | 1.475370360e-07 | 5.999544754e-07 | 2.4602e-01 | 0 | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `gaussiancv` | `whest_0718_gaussiancv_auto_r2` | 9 | `7c7d3e4f8621` | 1.478482189e-07 | 5.996531343e-07 | 2.4659e-01 | 0 | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `gaussiancv` | `whest_0718_gaussiancv_auto_r1` | 8 | `ea12987393d4` | 1.479745483e-07 | 5.999544754e-07 | 2.4672e-01 | 0 | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `gaussiancv` | `whest_0718_gaussiancv_auto_r2` | 9 | `c5970462fb88` | 1.480215117e-07 | 5.996531343e-07 | 2.4690e-01 | 0 | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `affinef64` | `whest_0718_affinef64_ttt` | 46 | `c558c44567e3` | 1.785284548e-07 | 1.188197484e-06 | 1.5023e-01 | 0 | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `affinef64` | `whest_0718_affinef64_ttt` | 40 | `1b92c4213a4a` | 1.786221180e-07 | 1.188197484e-06 | 1.5031e-01 | 0 | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `affinef64` | `whest_0718_affinef64_ttt` | 45 | `7503abbe4790` | 1.786411005e-07 | 1.188197484e-06 | 1.5032e-01 | 0 | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `affinef64` | `whest_0718_affinef64_ttt` | 42 | `c966535717d5` | 1.786488841e-07 | 1.188197484e-06 | 1.5033e-01 | 0 | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `affinef64` | `whest_0718_affinef64_ttt` | 43 | `7983b19a464b` | 1.786637914e-07 | 1.188197484e-06 | 1.5034e-01 | 0 | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `rqmc030` | `whest_0718_rqmc030_auto_r2` | 5 | `b89b9c8993e4` | 1.830363038e-07 | 4.578615325e-07 | 3.9980e-01 | 0 | `reject_ranked_not_full100` | Lower-ranked same-family/method variant; a stronger representative was selected for full-100. |
| `rqmc030` | `whest_0718_rqmc030_auto_r1` | 5 | `59fb63039b4a` | 1.884832269e-07 | 2.743381344e-07 | 6.8688e-01 | 0 | `reject_ranked_not_full100` | Lower-ranked same-family/method variant; a stronger representative was selected for full-100. |
| `rqmc030` | `whest_0718_rqmc030_auto_r1` | 5 | `91f283e60a92` | 1.890025681e-07 | 2.743381344e-07 | 6.8909e-01 | 0 | `reject_ranked_not_full100` | Lower-ranked same-family/method variant; a stronger representative was selected for full-100. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | 42 | `1c2696b43241` | 2.325226468e-07 | 2.319162974e-06 | 1.0033e-01 | 0 | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | 45 | `afbcf0dea88c` | 2.341400135e-07 | 2.325468705e-06 | 1.0067e-01 | 0 | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `german013` | `whest_0718_german013_ttt` | 44 | `a40a5daec89c` | 2.420684177e-07 | 1.842124703e-06 | 1.3130e-01 | 0 | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `german013` | `whest_0718_german013_ttt` | 41 | `c294641aaff5` | 2.431067185e-07 | 1.842124703e-06 | 1.3183e-01 | 0 | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |

Additional no-score workspace/submission candidates:

| Family | Run | Step | SHA | Source | Reason |
|---|---|---:|---|---|---|
| `affinef64` | `whest_0718_affinef64_auto_r1` | 4 | `47b12c15b720` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r1/autoevolve_workspaces/step_000004/call_0001_53b132b6/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `gaussiancv` | `whest_0718_gaussiancv_auto_r1` | 8 | `3d6894ad5963` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r1/autoevolve_workspaces/step_000008/call_0001_4b1b2918/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `gaussiancv` | `whest_0718_gaussiancv_auto_r2` | 9 | `25ec21724924` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r2/autoevolve_workspaces/step_000009/call_0001_6e60410e/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `german013` | `whest_0718_german013_auto_r1` | 4 | `bc7895a003a8` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000004/call_0001_13c404ff/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `rqmc030` | `whest_0718_rqmc030_auto_r1` | 5 | `aaea68e098ff` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r1/autoevolve_workspaces/step_000005/call_0001_fccd5988/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `sphericalrb` | `whest_0718_sphericalrb_auto_r1` | 3 | `81e856ace918` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r1/autoevolve_workspaces/step_000003/call_0001_73bdde20/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |

Interpretation:
- GaussianCV candidates remained within first-10 noise of the same-family initial and were not promoted.
- AffineF64 TTT candidates showed modest first-10 gains but repeated setup-cached whitening/layer-rescale mechanisms already weak on prior full-100 checks.
- RQMC030 had the only strong new signal, but the selected representative regressed on full-100 to `4.572027097e-07`; lower-ranked RQMC variants were not separately validated.
- German/Spherical TTT tail states were weak, noisy, or method-near variants.

### Official Submission Status

- No `whest package` or `whest submit` was run because the only full-100 candidate failed the local quality bar.
- No new submission ID was obtained; registry was not changed after the required pre-review tracker refresh.
- Live official submission would also require credential/quota recovery because registered submission refreshes still report HTTP 401, so server quota could not be verified.
- Official submission IDs this cycle: none.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-18T22:18:31+00:00`; do not treat this cycle's `385` inventoried SHA or `35` new SHA as new.
- Latest scanned steps/pools at cutoff: `whest_0718_affinef64_auto_r1` pool `5`; `whest_0718_affinef64_auto_r2` pool `6`; `whest_0718_affinef64_ttt` TTT `46`; `whest_0718_gaussiancv_auto_r1` pool `9`; `whest_0718_gaussiancv_auto_r2` pool `10`; `whest_0718_gaussiancv_ttt` TTT `50`; `whest_0718_german013_auto_r1` pool `4`; `whest_0718_german013_auto_r2` pool `3`; `whest_0718_german013_ttt` TTT `44`; `whest_0718_layerrescale_auto_r1` pool `3`; `whest_0718_layerrescale_auto_r2` pool `4`; `whest_0718_layerrescale_ttt` TTT `50`; `whest_0718_rqmc030_auto_r1` pool `6`; `whest_0718_rqmc030_auto_r2` pool `5`; `whest_0718_rqmc030_ttt` TTT `35`; `whest_0718_sphericalrb_auto_r1` pool `4`; `whest_0718_sphericalrb_auto_r2` pool `4`; `whest_0718_sphericalrb_ttt` TTT `46`.
- Include artifacts excluded after cutoff next round, starting with:
- `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r1/autoevolve_workspaces/step_000005/call_0001_a82049d8/submission.py`
- `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r1/autoevolve_workspaces/step_000009/call_0001_49f3dcca/submission.py`
- `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r2/autoevolve_workspaces/step_000010/call_0001_2476e33b/submission.py`
- `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r2/autoevolve_workspaces/step_000003/call_0001_61e2f15f/submission.py`
- `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_layerrescale_auto_r2/autoevolve_workspaces/step_000004/call_0001_4dbfe4d7/submission.py`
- `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r1/autoevolve_workspaces/step_000006/call_0001_e8732367/submission.py`
- `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r2/autoevolve_workspaces/step_000005/call_0001_77acb1be/submission.py`
- `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r1/autoevolve_workspaces/step_000004/call_0001_08a56462/submission.py`

## Cycle 20260718T230132Z hourly submission review

- Completed at UTC: `2026-07-18T23:17:33.290238+00:00`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- Cached official status counts after refresh: `{'failed': 1, 'graded': 24}`. Pending/queued/running registered submissions: `[]`. No historical pending status changes needed ledger backfill.
- AIcrowd refresh/auth status: `AIcrowdAuthError HTTP 401 for registered submissions; server quota/status unavailable`.
- Quota assessment for UTC `2026-07-18`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`. Blocker: `AIcrowdAuthError: While checking submission status: Your AIcrowd API key is missing or invalid. (HTTP 401)`.
- Candidate cutoff UTC: `2026-07-18T23:01:32+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-18T23:00:24.795768+00:00']`.
- JSONL context files were read for sampler/metrics/agent/gen-score line counts while live runs continued; live mtime range UTC: `['2026-07-18T17:40:25.175395+00:00', '2026-07-18T23:00:24.796768+00:00']`.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T230132Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T230132Z/scan_summary.json`
- Shortlist manifest: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T230132Z/shortlist_manifest.json`
- Shortlist sources: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T230132Z/shortlist_sources`
- Full-100 reports: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T230132Z/full100_reports`
- Initial estimator SHA baselines scanned under `initial_states`: `{'affinef64': '1cb0eb3a875702386c346d605880190c7827b392860c7ddb6fb690f9f9b71435', 'gaussiancv': 'd49b8b5218ff9b9462a331812e688e125fcfff1f86e8a26e9be9c8f59eb48ebb', 'german013': '68d409403de29c8fa7807a8274066494e9ce00380bbbe5bf3d1d52e4ba35aa64', 'layerrescale': '98da6a0882cbc02d32cbd20e295c80b21bb8c173e3e23bb0d6880233d611d7ff', 'rqmc030': 'dc60550f76e9f4f6cb5940af295cab777a4082412590a49665e9127b535eb19f', 'sphericalrb': '146a8c786ee31dfefc941712ded736c3ec2625343dd5cd184f80dde915a9d130'}`
- Console logs location checked for run context: `codex_runs/aicrowd_whestbench/multi_initial_20260718/console`.
- Files excluded after cutoff: `4`; examples: `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r1/autoevolve_workspaces/step_000005/call_0001_a82049d8/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r2/autoevolve_workspaces/step_000011/call_0001_2a39a8f3/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r1/autoevolve_workspaces/step_000006/call_0001_e8732367/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r1/autoevolve_workspaces/step_000004/call_0001_08a56462/submission.py`.

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count | Excluded after cutoff |
|---|---|---|---:|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | - | 0-5 | 5 | 11 | 1 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | - | 0-9 | 10 | 20 | 0 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | - | 0-11 | 11 | 23 | 1 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | - | 0-4 | 5 | 10 | 0 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | - | 0-3 | 4 | 8 | 0 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | - | 0-4 | 5 | 10 | 0 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | - | 0-5 | 6 | 12 | 0 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | - | 0-6 | 6 | 13 | 1 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | 0-39 | - | 0 | 40 | 0 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | - | 0-4 | 4 | 9 | 1 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | - | 0-4 | 5 | 10 | 0 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | 0-50 | - | 0 | 51 | 0 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `424`.
- Historical/known SHA baseline size: `598`; old inventory entries read: `2155`.
- New SHA after historical baseline removal: `41`.
- Final decision counts: `{'reject_existing_sha': 383, 'reject_full100_no_submit': 2, 'reject_insufficient_first10_gain_not_full100': 4, 'reject_no_first10': 6, 'reject_ranked_not_full100': 12, 'reject_weak_first10': 17}`.
- Entered full-100: `2`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Full-100 Shortlist Decisions

| Family | Run | Step | State | SHA | First-10 adjusted / MSE / C-B / fail | Full-100 adjusted / MSE / C-B / fail | vs initial / vs parent | Decision | Candidate path |
|---|---|---:|---|---|---|---|---|---|---|
| `layerrescale` | `whest_0718_layerrescale_auto_r2` | 5 | `35de0ab7-d401-4799-b67c-4166ec6054e9` | `3f26d2bb11e5` | 1.357858217e-07 / 1.355103279e-06 / 1.0008e-01 / 0 | 3.755691366e-07 / 3.755479121e-06 / 9.9649e-02 / 0 | 45.844% / 0.309% | `reject_full100_no_submit` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T230132Z/shortlist_sources/layerrescale_candidate_3f26d2bb11e5.py` |
| `rqmc030` | `whest_0718_rqmc030_auto_r2` | 6 | `a53e519f-22c6-4f02-8f5d-21191e63d924` | `8632835e5f93` | 1.827462948e-07 / 4.571094678e-07 / 3.9981e-01 / 0 | 4.616515165e-07 / 1.154130962e-06 / 3.9999e-01 / 0 | 149.666% / 0.154% | `reject_full100_no_submit` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T230132Z/shortlist_sources/rqmc030_candidate_8632835e5f93.py` |

Full-100 interpretation:
- `layerrescale` candidate `3f26d2bb11e5` improved first-10 by `45.844%` versus initial and only `0.309%` versus parent `e2991497f6b9`; full-100 adjusted `3.755691366e-07` is essentially the same weak band as prior LayerRescale representative (`3.753050031e-07`) and worse than registered local best `2.748615e-07`.
- `rqmc030` candidate `8632835e5f93` improved first-10 by `149.666%` versus initial and `0.154%` versus parent `c0c17fa57f1f`; full-100 adjusted `4.616515165e-07` confirms the repeated RQMC030 pattern of first-10 optimism followed by full-100 regression around `4.55e-07` to `5.34e-07`.
- Both full-100 runs had 0 failed MLPs and no budget/time exhaustion flags, but neither is competitive with best registered hosted public-50 `2.787552892e-07` (#316192). No package was built and no official submit was attempted.

### Not Promoted After Ranking

| Family | Run | Step | SHA | First-10 adjusted | First-10 MSE | C/B | Fail | vs initial recorded pct | Decision | Reason |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `layerrescale` | `whest_0718_layerrescale_auto_r2` | 5 | `fba838fa2471` | 1.357995213e-07 | 1.355103279e-06 | 1.0006e-01 | 0 | 45.829% | `reject_ranked_not_full100` | Lower-ranked same-family/method variant; a stronger representative was selected for full-100. |
| `gaussiancv` | `whest_0718_gaussiancv_auto_r2` | 11 | `2d8146271761` | 1.475152558e-07 | 5.995594648e-07 | 2.4617e-01 | 0 | 0.404% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `affinef64` | `whest_0718_affinef64_ttt` | 49 | `7ff24618dfb1` | 1.785522883e-07 | 1.188197484e-06 | 1.5025e-01 | 0 | 24.036% | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `rqmc030` | `whest_0718_rqmc030_auto_r2` | 6 | `0e5c9a6cab0f` | 1.827756814e-07 | 4.571094678e-07 | 3.9986e-01 | 0 | 149.626% | `reject_ranked_not_full100` | Lower-ranked same-family/method variant; a stronger representative was selected for full-100. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | 47 | `0a6311ce798b` | 2.173773300e-07 | 2.168735170e-06 | 1.0006e-01 | 0 | 2.379% | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | 50 | `fb053a609f31` | 2.173807732e-07 | 2.168735170e-06 | 1.0005e-01 | 0 | 2.378% | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | 48 | `7ac259320e65` | 2.322177195e-07 | 2.108349861e-06 | 1.1013e-01 | 0 | -4.163% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | 49 | `8a5b784728cf` | 2.333561749e-07 | 2.319162974e-06 | 1.0082e-01 | 0 | -4.631% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | 50 | `2023204c1c08` | 2.368510930e-07 | 2.202931955e-06 | 1.0754e-01 | 0 | -6.038% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `german013` | `whest_0718_german013_ttt` | 48 | `339a33376a90` | 2.428809666e-07 | 1.842124703e-06 | 1.3168e-01 | 0 | 3.593% | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `rqmc030` | `whest_0718_rqmc030_ttt` | 38 | `04216553c493` | 2.450109820e-07 | 6.784204515e-07 | 3.6177e-01 | 0 | 86.218% | `reject_ranked_not_full100` | Strong first-10 gain (86.2% by recorded ratio), but prior same-family representatives already regressed on full-100 around 5.237e-07; not enough new mechanism to spend validation. |
| `rqmc030` | `whest_0718_rqmc030_ttt` | 39 | `04643e7d97d8` | 2.454223147e-07 | 6.784204515e-07 | 3.6274e-01 | 0 | 85.906% | `reject_ranked_not_full100` | Strong first-10 gain (85.9% by recorded ratio), but prior same-family representatives already regressed on full-100 around 5.237e-07; not enough new mechanism to spend validation. |
| `rqmc030` | `whest_0718_rqmc030_ttt` | 39 | `9b7ca604a59d` | 2.455151221e-07 | 6.784234984e-07 | 3.6276e-01 | 0 | 85.836% | `reject_ranked_not_full100` | Strong first-10 gain (85.8% by recorded ratio), but prior same-family representatives already regressed on full-100 around 5.237e-07; not enough new mechanism to spend validation. |
| `rqmc030` | `whest_0718_rqmc030_ttt` | 36 | `2128c780b3d2` | 2.464699059e-07 | 6.784204515e-07 | 3.6411e-01 | 0 | 85.116% | `reject_ranked_not_full100` | Strong first-10 gain (85.1% by recorded ratio), but prior same-family representatives already regressed on full-100 around 5.237e-07; not enough new mechanism to spend validation. |
| `rqmc030` | `whest_0718_rqmc030_ttt` | 36 | `b2e96a986214` | 2.474299717e-07 | 7.254487230e-07 | 3.4209e-01 | 0 | 84.398% | `reject_ranked_not_full100` | Strong first-10 gain (84.4% by recorded ratio), but prior same-family representatives already regressed on full-100 around 5.237e-07; not enough new mechanism to spend validation. |
| `rqmc030` | `whest_0718_rqmc030_ttt` | 39 | `16782c4a06aa` | 2.478569603e-07 | 6.827015341e-07 | 3.6379e-01 | 0 | 84.080% | `reject_ranked_not_full100` | Strong first-10 gain (84.1% by recorded ratio), but prior same-family representatives already regressed on full-100 around 5.237e-07; not enough new mechanism to spend validation. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | 47 | `07b063f37453` | 2.483255191e-07 | 2.298558343e-06 | 1.0771e-01 | 0 | -10.380% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `rqmc030` | `whest_0718_rqmc030_ttt` | 37 | `e8ffce4226c3` | 2.487234025e-07 | 6.840797198e-07 | 3.6423e-01 | 0 | 83.439% | `reject_ranked_not_full100` | Strong first-10 gain (83.4% by recorded ratio), but prior same-family representatives already regressed on full-100 around 5.237e-07; not enough new mechanism to spend validation. |
| `sphericalrb` | `whest_0718_sphericalrb_ttt` | 48 | `8022a31906a6` | 2.494957687e-07 | 2.494957687e-06 | 9.5320e-02 | 0 | -10.800% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `german013` | `whest_0718_german013_ttt` | 46 | `2d77094398e3` | 2.528135609e-07 | 1.885474012e-06 | 1.3404e-01 | 0 | -0.477% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |

Additional no-score workspace/submission candidates:

| Family | Run | Step | SHA | Source | Reason |
|---|---|---:|---|---|---|
| `gaussiancv` | `whest_0718_gaussiancv_auto_r1` | 9 | `7df159d67c18` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r1/autoevolve_workspaces/step_000009/call_0001_49f3dcca/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `german013` | `whest_0718_german013_auto_r1` | 4 | `7212ea9d0a38` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000004/call_0001_13c404ff/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `german013` | `whest_0718_german013_auto_r2` | 3 | `d150f0bad60d` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r2/autoevolve_workspaces/step_000003/call_0001_61e2f15f/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `layerrescale` | `whest_0718_layerrescale_auto_r2` | 5 | `1dc8e3020f9d` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_layerrescale_auto_r2/autoevolve_workspaces/step_000005/call_0001_799be722/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `rqmc030` | `whest_0718_rqmc030_auto_r2` | 6 | `9fd6165f920c` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r2/autoevolve_workspaces/step_000006/call_0001_c94eea73/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `sphericalrb` | `whest_0718_sphericalrb_auto_r2` | 4 | `c1200c00fa0a` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r2/autoevolve_workspaces/step_000004/call_0001_9b820744/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |

Interpretation:
- LayerRescale step 5 was a near-duplicate/micro-improvement over a prior full-100-rejected parent; validating it confirmed no meaningful full-100 gain.
- RQMC030 step 6 adjusted constants in the antithetic lattice first-layer control variate (`_FIRST_CONTROL_SCALE=0.365`, `_FIRST_VARIANCE_BLEND=0.60`, `_OFFDIAGONAL_CONTROL_SCALE=-0.14`, `_INPUT_SECOND_POWER=1.0`) but did not generalize on full-100.
- GaussianCV remained within first-10 noise of its initial baseline. AffineF64, German013, SphericalRB, and RQMC030 TTT tails were weaker, noisy, or lower-ranked method-near variants.

### Official Submission Status

- No `whest package` or `whest submit` was run because both full-100 candidates failed the local quality bar.
- No new submission ID was obtained; registry was not changed after the required pre-review tracker refresh.
- Live official submission would also require credential/quota recovery because registered submission refreshes still report HTTP 401, so server quota could not be verified.
- Official submission IDs this cycle: none.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-18T23:01:32+00:00`; do not treat this cycle's `424` inventoried SHA or `41` new SHA as new.
- Latest scanned TTT steps at cutoff: `whest_0718_affinef64_ttt` `50`, `whest_0718_gaussiancv_ttt` `50`, `whest_0718_german013_ttt` `50`, `whest_0718_layerrescale_ttt` `50`, `whest_0718_rqmc030_ttt` `39`, `whest_0718_sphericalrb_ttt` `50`.
- Latest scanned AutoEvolve pool steps at cutoff: `whest_0718_affinef64_auto_r1` `5`, `whest_0718_affinef64_auto_r2` `6`, `whest_0718_gaussiancv_auto_r1` `9`, `whest_0718_gaussiancv_auto_r2` `11`, `whest_0718_german013_auto_r1` `4`, `whest_0718_german013_auto_r2` `3`, `whest_0718_layerrescale_auto_r1` `4`, `whest_0718_layerrescale_auto_r2` `5`, `whest_0718_rqmc030_auto_r1` `6`, `whest_0718_rqmc030_auto_r2` `6`, `whest_0718_sphericalrb_auto_r1` `4`, `whest_0718_sphericalrb_auto_r2` `4`.
- Include artifacts excluded after cutoff next round, starting with: `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r1/autoevolve_workspaces/step_000005/call_0001_a82049d8/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r2/autoevolve_workspaces/step_000011/call_0001_2a39a8f3/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r1/autoevolve_workspaces/step_000006/call_0001_e8732367/submission.py`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r1/autoevolve_workspaces/step_000004/call_0001_08a56462/submission.py`.

## Cycle 20260718T234937Z hourly submission review

- Completed at UTC: `2026-07-19T00:09:23.329111+00:00`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- Cached official status counts after refresh: `{'failed': 1, 'graded': 24}`. Pending/queued/running registered submissions: `[]`. No historical pending status changes needed ledger backfill.
- Quota assessment for UTC `2026-07-18`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`. Blocker: `AIcrowdAuthError: While checking submission status: Your AIcrowd API key is missing or invalid. (HTTP 401)`.
- AIcrowd refresh/auth status: `AIcrowdAuthError HTTP 401 for registered submissions; server quota/status unavailable`.
- Candidate cutoff UTC: `2026-07-18T23:49:37+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-18T23:48:28.692912+00:00']`.
- JSONL context files were read for sampler/metrics/agent/gen-score line counts while live runs continued; live mtime range UTC: `['2026-07-18T17:40:25.175395+00:00', '2026-07-18T23:48:28.692912+00:00']`.
- Console logs location checked for run context: `codex_runs/aicrowd_whestbench/multi_initial_20260718/console`.
- Files excluded after cutoff: `6`; examples: `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r1/autoevolve_workspaces/step_000006/call_0001_ff8469ab/submission.py; codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r2/autoevolve_workspaces/step_000007/call_0001_3f38fbbe/submission.py; codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r2/autoevolve_workspaces/step_000012/call_0001_9c4969ac/submission.py; codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r1/autoevolve_workspaces/step_000007/call_0001_d45ac307/submission.py; codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r2/autoevolve_workspaces/step_000007/call_0001_4c4e4651/submission.py; codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r2/autoevolve_workspaces/step_000005/call_0001_7024fcae/submission.py`.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T234937Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T234937Z/scan_summary.json`
- Shortlist manifest: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T234937Z/shortlist_manifest.json`
- Shortlist sources: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T234937Z/shortlist_sources`
- Full-100 reports: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T234937Z/full100_reports`
- Initial estimator SHA baselines scanned under `initial_states`: `{'affinef64': '1cb0eb3a875702386c346d605880190c7827b392860c7ddb6fb690f9f9b71435', 'gaussiancv': 'd49b8b5218ff9b9462a331812e688e125fcfff1f86e8a26e9be9c8f59eb48ebb', 'german013': '68d409403de29c8fa7807a8274066494e9ce00380bbbe5bf3d1d52e4ba35aa64', 'layerrescale': '98da6a0882cbc02d32cbd20e295c80b21bb8c173e3e23bb0d6880233d611d7ff', 'rqmc030': 'dc60550f76e9f4f6cb5940af295cab777a4082412590a49665e9127b535eb19f', 'sphericalrb': '146a8c786ee31dfefc941712ded736c3ec2625343dd5cd184f80dde915a9d130'}`

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count | Excluded after cutoff |
|---|---|---|---:|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | - | 0-6 | 6 | 13 | 1 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | - | 0-7 | 7 | 15 | 1 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | - | 0-9 | 10 | 20 | 0 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | - | 0-12 | 12 | 25 | 1 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | - | 0-4 | 5 | 10 | 0 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | - | 0-3 | 4 | 8 | 0 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | - | 0-4 | 5 | 10 | 0 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | - | 0-7 | 8 | 16 | 0 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | - | 0-7 | 7 | 15 | 1 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | - | 0-7 | 7 | 15 | 1 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | 0-43 | - | 0 | 44 | 0 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | - | 0-5 | 6 | 12 | 0 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | - | 0-5 | 5 | 11 | 1 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | 0-50 | - | 0 | 51 | 0 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `435`.
- Historical/known SHA baseline size: `639`; old inventory entries read: `2579`.
- New SHA after historical baseline removal: `15`.
- Final decision counts: `{'reject_full100_no_submit': 3, 'reject_existing_sha': 420, 'reject_ranked_not_full100': 7, 'reject_weak_first10': 1, 'reject_insufficient_first10_gain_not_full100': 2, 'reject_no_first10': 2}`.
- Entered full-100: `3`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Full-100 Shortlist Decisions

| Family | Run | Step | State | SHA | First-10 adjusted / MSE / C-B / fail | Full-100 adjusted / MSE / C-B / fail | vs initial / vs parent | Decision | Candidate path |
|---|---|---:|---|---|---|---|---|---|---|
| `sphericalrb` | `whest_0718_sphericalrb_auto_r1` | 5 | `2e63bf10-53b0-4694-9be8-13e6b22a8b2b` | `3c73a750aa29` | 1.461057215e-07 / 1.434708327e-06 / 1.0174e-01 / 0 | 4.152316059e-07 / 4.089397740e-06 / 1.0155e-01 / 0 | 52.321% / 0.968% | `reject_full100_no_submit` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T234937Z/shortlist_sources/sphericalrb_candidate_3c73a750aa29.py` |
| `affinef64` | `whest_0718_affinef64_auto_r2` | 7 | `943dbbc9-5ed2-4773-be93-a02dde42f295` | `685334182cb2` | 1.469867607e-07 / 1.469867607e-06 / 9.8996e-02 / 0 | 3.012131572e-07 / 3.012131572e-06 / 9.8569e-02 / 0 | 50.672% / 2.316% | `reject_full100_no_submit` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T234937Z/shortlist_sources/affinef64_candidate_685334182cb2.py` |
| `rqmc030` | `whest_0718_rqmc030_auto_r2` | 7 | `07cc995a-1e21-4de9-976e-47b20ec29607` | `fec8a02d3f0c` | 1.827669363e-07 / 4.571094678e-07 / 3.9986e-01 / 0 | 4.611175908e-07 / 1.154130962e-06 / 3.9954e-01 / 0 | 149.638% / -0.011% | `reject_full100_no_submit` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T234937Z/shortlist_sources/rqmc030_candidate_fec8a02d3f0c.py` |

Full-100 interpretation:
- `3c73a750aa29` (Spherical lattice directions with angular whitening and prefix-pair blend): first-10 adjusted `1.461057215e-07` with `0` failures, vs initial `52.321%` and vs parent `0.968%`. Full-100 report `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T234937Z/full100_reports/sphericalrb_lattice_prefix_3c73a750aa29_full100_20260718T234937Z.json`: adjusted `4.152316059e-07`, MSE `4.089397740e-06`, C/B `1.0155e-01`, failures `0`, failure_breakdown `{'budget_exhausted': 0, 'combined_budget_exhausted': 0, 'error': 0, 'residual_wall_time_exhausted': 0, 'time_exhausted': 0}`.
- `685334182cb2` (Setup-cached whitened antithetic radial samples with first-layer moment matching): first-10 adjusted `1.469867607e-07` with `0` failures, vs initial `50.672%` and vs parent `2.316%`. Full-100 report `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T234937Z/full100_reports/affinef64_whitened_radial_685334182cb2_full100_20260718T234937Z.json`: adjusted `3.012131572e-07`, MSE `3.012131572e-06`, C/B `9.8569e-02`, failures `0`, failure_breakdown `{'budget_exhausted': 0, 'combined_budget_exhausted': 0, 'error': 0, 'residual_wall_time_exhausted': 0, 'time_exhausted': 0}`.
- `fec8a02d3f0c` (Antithetic lattice/RQMC first-layer control variate, step-7 micro-variant): first-10 adjusted `1.827669363e-07` with `0` failures, vs initial `149.638%` and vs parent `-0.011%`. Full-100 report `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260718T234937Z/full100_reports/rqmc030_control_variate_fec8a02d3f0c_full100_20260718T234937Z.json`: adjusted `4.611175908e-07`, MSE `1.154130962e-06`, C/B `3.9954e-01`, failures `0`, failure_breakdown `{'budget_exhausted': 0, 'combined_budget_exhausted': 0, 'error': 0, 'residual_wall_time_exhausted': 0, 'time_exhausted': 0}`.
- None of the three full-100 candidates was competitive with registered local full-100 best `2.748615e-07` or best registered hosted public-50 `2.787552892e-07` (#316192). No package was built and no official submit was attempted.

### Not Promoted After Ranking

| Family | Run | Step | SHA | First-10 adjusted | First-10 MSE | C/B | Fail | vs initial recorded pct | Decision | Reason |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `sphericalrb` | `whest_0718_sphericalrb_auto_r1` | 5 | `a5d3ff80bd24` | 1.462024522e-07 | 1.434708327e-06 | 1.0181e-01 | 0 | 52.220% | `reject_ranked_not_full100` | Lower-ranked same-family/method variant; a stronger representative was selected for full-100. |
| `affinef64` | `whest_0718_affinef64_auto_r2` | 7 | `a38f64e992b7` | 1.469867607e-07 | 1.469867607e-06 | 9.8810e-02 | 0 | 50.672% | `reject_ranked_not_full100` | Lower-ranked same-family/method variant; a stronger representative was selected for full-100. |
| `gaussiancv` | `whest_0718_gaussiancv_auto_r2` | 12 | `e5dbce536fe0` | 1.475041752e-07 | 5.994584711e-07 | 2.4620e-01 | 0 | 0.412% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `sphericalrb` | `whest_0718_sphericalrb_auto_r2` | 5 | `bf6bb71c699d` | 1.481917130e-07 | 1.454303577e-06 | 1.0183e-01 | 0 | 50.177% | `reject_ranked_not_full100` | Lower-ranked same-family/method variant; a stronger representative was selected for full-100. |
| `sphericalrb` | `whest_0718_sphericalrb_auto_r2` | 5 | `98e93094ccb0` | 1.482009027e-07 | 1.454303577e-06 | 1.0181e-01 | 0 | 50.167% | `reject_ranked_not_full100` | Lower-ranked same-family/method variant; a stronger representative was selected for full-100. |
| `affinef64` | `whest_0718_affinef64_auto_r1` | 6 | `b121cbf7db34` | 1.638387776e-07 | 1.467406190e-06 | 1.1163e-01 | 0 | 35.175% | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `affinef64` | `whest_0718_affinef64_auto_r1` | 6 | `a8d7596df26b` | 1.639264229e-07 | 1.467406190e-06 | 1.1170e-01 | 0 | 35.102% | `reject_insufficient_first10_gain_not_full100` | First-10 improvement is not large enough relative to observed noise and prior local/official drift. |
| `rqmc030` | `whest_0718_rqmc030_ttt` | 43 | `e36d2e001153` | 2.459197488e-07 | 6.784234984e-07 | 3.6318e-01 | 0 | 85.530% | `reject_ranked_not_full100` | Strong first-10 gain (85.5% by recorded ratio), but prior same-family representatives already regressed on full-100 around 5.237e-07; not enough new mechanism to spend validation. |
| `rqmc030` | `whest_0718_rqmc030_ttt` | 42 | `fbb0660267b0` | 2.502843994e-07 | 6.766957767e-07 | 3.7085e-01 | 0 | 82.295% | `reject_ranked_not_full100` | Strong first-10 gain (82.3% by recorded ratio), but prior same-family representatives already regressed on full-100 around 5.237e-07; not enough new mechanism to spend validation. |
| `rqmc030` | `whest_0718_rqmc030_ttt` | 42 | `b79d070014b3` | 2.554249144e-07 | 7.058820586e-07 | 3.6305e-01 | 0 | 78.626% | `reject_ranked_not_full100` | Strong first-10 gain (78.6% by recorded ratio), but prior same-family representatives already regressed on full-100 around 5.237e-07; not enough new mechanism to spend validation. |
| `layerrescale` | `whest_0718_layerrescale_auto_r2` | 6 | `c0b6e258ae92` | - | - | - | - | - | `reject_no_first10` | No verified first-10 construction/message score was available in scanned artifacts. |
| `sphericalrb` | `whest_0718_sphericalrb_auto_r1` | 5 | `7e8179403aaf` | - | - | - | - | - | `reject_no_first10` | No verified first-10 construction/message score was available in scanned artifacts. |

Additional no-score workspace/submission candidates:

| Family | Run | Step | SHA | Source | Reason |
|---|---|---:|---|---|---|
| `layerrescale` | `whest_0718_layerrescale_auto_r2` | 6 | `c0b6e258ae92` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_layerrescale_auto_r2/autoevolve_workspaces/step_000006/call_0001_ea42886a/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `sphericalrb` | `whest_0718_sphericalrb_auto_r1` | 5 | `7e8179403aaf` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r1/autoevolve_workspaces/step_000005/call_0001_edbe0075/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |

Interpretation:
- SphericalRB step 5 improved first-10 but generalized worse than earlier Spherical representatives on full-100 (`4.152e-07`).
- AffineF64 step 7 was the best full-100 result this cycle (`3.012e-07`) and structurally different via radialized antithetic samples plus first-layer covariance remapping, but it remained weaker than both the local best and the best registered official score.
- RQMC030 step 7 was a micro-variant of the prior step-6 control-variate candidate and again regressed to the `4.61e-07` full-100 band.
- GaussianCV remained within first-10 noise; new no-score workspace files were not promoted without verified first-10 construction/message metrics.

### Official Submission Status

- No `whest package` or `whest submit` was run because all full-100 candidates failed the quality bar.
- No new submission ID was obtained; registry was not changed after the required pre-review tracker refresh.
- Live official submission would also require credential/quota recovery because registered submission refreshes still report HTTP 401, so server quota could not be verified.
- Official submission IDs this cycle: none.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-18T23:49:37+00:00`; do not treat this cycle's `435` inventoried SHA or `15` new SHA as new.
- Latest scanned TTT steps at cutoff: `whest_0718_affinef64_ttt` `50`, `whest_0718_gaussiancv_ttt` `50`, `whest_0718_german013_ttt` `50`, `whest_0718_layerrescale_ttt` `50`, `whest_0718_rqmc030_ttt` `43`, `whest_0718_sphericalrb_ttt` `50`.
- Latest scanned AutoEvolve pool steps at cutoff: `whest_0718_affinef64_auto_r1` `6`, `whest_0718_affinef64_auto_r2` `7`, `whest_0718_gaussiancv_auto_r1` `9`, `whest_0718_gaussiancv_auto_r2` `12`, `whest_0718_german013_auto_r1` `4`, `whest_0718_german013_auto_r2` `3`, `whest_0718_layerrescale_auto_r1` `4`, `whest_0718_layerrescale_auto_r2` `7`, `whest_0718_rqmc030_auto_r1` `7`, `whest_0718_rqmc030_auto_r2` `7`, `whest_0718_sphericalrb_auto_r1` `5`, `whest_0718_sphericalrb_auto_r2` `5`.
- Include artifacts excluded after cutoff next round, starting with:
- `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r1/autoevolve_workspaces/step_000006/call_0001_ff8469ab/submission.py`
- `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r2/autoevolve_workspaces/step_000007/call_0001_3f38fbbe/submission.py`
- `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r2/autoevolve_workspaces/step_000012/call_0001_9c4969ac/submission.py`
- `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r1/autoevolve_workspaces/step_000007/call_0001_d45ac307/submission.py`
- `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r2/autoevolve_workspaces/step_000007/call_0001_4c4e4651/submission.py`
- `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r2/autoevolve_workspaces/step_000005/call_0001_7024fcae/submission.py`


## Cycle 20260719T004013Z hourly submission review

- Completed at UTC: `2026-07-19T00:55:31.644000+00:00`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- Cached official status counts after refresh: `{'failed': 1, 'graded': 24}`. Pending/queued/running registered submissions: `[]`. No historical pending status changes needed ledger backfill.
- Quota assessment for UTC `2026-07-19`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`. Blocker: `AIcrowdAuthError: While checking submission status: Your AIcrowd API key is missing or invalid. (HTTP 401)`.
- AIcrowd refresh/auth status: `AIcrowdAuthError HTTP 401 for registered submissions; server quota/status unavailable`.
- Candidate cutoff UTC: `2026-07-19T00:40:13+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-19T00:39:57.527107+00:00']`.
- JSONL context files were read for sampler/metrics/agent/gen-score line counts while live runs continued; live mtime range UTC: `['2026-07-18T17:40:25.175395+00:00', '2026-07-19T00:42:11.025490+00:00']`.
- Console logs location checked for run context: `codex_runs/aicrowd_whestbench/multi_initial_20260718/console`.
- Files excluded after cutoff: `7`.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T004013Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T004013Z/scan_summary.json`
- Shortlist manifest: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T004013Z/shortlist_manifest.json`
- Shortlist sources: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T004013Z/shortlist_sources`
- Full-100 reports: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T004013Z/full100_reports`
- Initial estimator SHA baselines scanned under `initial_states`: `{'affinef64': '1cb0eb3a875702386c346d605880190c7827b392860c7ddb6fb690f9f9b71435', 'gaussiancv': 'd49b8b5218ff9b9462a331812e688e125fcfff1f86e8a26e9be9c8f59eb48ebb', 'german013': '68d409403de29c8fa7807a8274066494e9ce00380bbbe5bf3d1d52e4ba35aa64', 'layerrescale': '98da6a0882cbc02d32cbd20e295c80b21bb8c173e3e23bb0d6880233d611d7ff', 'rqmc030': 'dc60550f76e9f4f6cb5940af295cab777a4082412590a49665e9127b535eb19f', 'sphericalrb': '146a8c786ee31dfefc941712ded736c3ec2625343dd5cd184f80dde915a9d130'}`

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count | Excluded after cutoff |
|---|---|---|---:|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | - | 0-6 | 6 | 13 | 1 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | - | 0-7 | 7 | 15 | 1 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | - | 0-10 | 11 | 22 | 0 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | - | 0-13 | 13 | 27 | 1 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | - | 0-4 | 5 | 10 | 0 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | - | 0-3 | 4 | 8 | 0 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | - | 0-4 | 5 | 10 | 0 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | - | 0-9 | 10 | 20 | 0 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | - | 0-8 | 8 | 17 | 1 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | - | 0-8 | 8 | 17 | 1 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | 0-48 | - | 0 | 49 | 1 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | - | 0-5 | 6 | 12 | 0 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | - | 0-5 | 5 | 11 | 1 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | 0-50 | - | 0 | 51 | 0 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `449`.
- Historical/known SHA baseline size: `654`; old inventory entries read: `3014`.
- New SHA after historical baseline removal: `14`.
- Final decision counts: `{'reject_existing_sha': 435, 'reject_full100_no_submit': 1, 'reject_no_first10': 1, 'reject_ranked_not_full100': 8, 'reject_weak_first10': 4}`.
- Entered full-100: `1`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Full-100 Shortlist Decisions

| Family | Run | Step | State | SHA | First-10 adjusted / MSE / C-B / fail | Full-100 adjusted / MSE / C-B / fail | vs initial / vs parent | Decision | Candidate path |
|---|---|---:|---|---|---|---|---|---|---|
| `rqmc030` | `whest_0718_rqmc030_auto_r1` | 8 | `50af2b95-c5bb-426a-84df-451c918d91f8` | `7165e74c9196` | 1.675358334e-07 / 2.431577151e-07 / 6.8921e-01 / 0 | 4.813462666e-07 / 7.004054878e-07 / 6.8735e-01 / 0 | 172.333% / 12.503% | `reject_full100_no_submit` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T004013Z/shortlist_sources/rqmc030_candidate_7165e74c9196.py` |

Full-100 interpretation:
- `7165e74c9196` is a shifted lattice/RQMC radial-shell sampler: two seeded lattice shifts, fixed Gaussian-radius normalization, weighted quarter-block aggregation, and setup-cached inputs. It is structurally different enough from prior RQMC control-variate micro-variants to justify one full-100 check.
- First-10 adjusted was `1.675358334e-07` with 0 failures and C/B `0.6892`, but full-100 adjusted regressed to `4.813462666e-07` with MSE `7.004054878e-07`, C/B `6.8735e-01`, 0 failures, and failure_breakdown `{'budget_exhausted': 0, 'combined_budget_exhausted': 0, 'error': 0, 'residual_wall_time_exhausted': 0, 'time_exhausted': 0}`.
- Full-100 was `75.12%` worse than registered local full-100 best `2.748615e-07` and `72.68%` worse than best registered hosted public-50 `2.787552892e-07` (#316192). No package was built and no official submit was attempted.

### Not Promoted After Ranking

| Family | Run | Step | SHA | First-10 adjusted | First-10 MSE | C/B | Fail | vs initial | Decision | Reason |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `gaussiancv` | `whest_0718_gaussiancv_auto_r2` | 13 | `058c9b7c98e0` | 1.480029353e-07 | 5.995975613e-07 | 2.4691e-01 | 0 | 0.073% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `gaussiancv` | `whest_0718_gaussiancv_auto_r2` | 13 | `dc4c873c502e` | 1.480309146e-07 | 5.995975613e-07 | 2.4695e-01 | 0 | 0.054% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `rqmc030` | `whest_0718_rqmc030_auto_r1` | 8 | `933322516749` | 1.677563341e-07 | 2.431577151e-07 | 6.9041e-01 | 0 | 171.975% | `reject_ranked_not_full100` | Lower-ranked same-family/method variant; a stronger representative was selected for full-100. |
| `rqmc030` | `whest_0718_rqmc030_auto_r2` | 8 | `3bca2577749a` | 1.826579003e-07 | 4.568836019e-07 | 3.9981e-01 | 0 | 149.787% | `reject_ranked_not_full100` | Lower-ranked same-family/method variant; a stronger representative was selected for full-100. |
| `rqmc030` | `whest_0718_rqmc030_auto_r2` | 8 | `48b9b75420ec` | 1.826733166e-07 | 4.568836019e-07 | 3.9986e-01 | 0 | 149.765% | `reject_ranked_not_full100` | Lower-ranked same-family/method variant; a stronger representative was selected for full-100. |
| `rqmc030` | `whest_0718_rqmc030_ttt` | 48 | `accdb4392efe` | 2.488563742e-07 | 6.944658708e-07 | 3.5930e-01 | 0 | 83.341% | `reject_ranked_not_full100` | Strong first-10 gain (83.3% by recorded ratio), but prior same-family representatives already regressed on full-100 around 5.237e-07; not enough new mechanism to spend validation. |
| `rqmc030` | `whest_0718_rqmc030_ttt` | 44 | `a21c8800b498` | 2.504847678e-07 | 6.766998439e-07 | 3.7113e-01 | 0 | 82.149% | `reject_ranked_not_full100` | Strong first-10 gain (82.1% by recorded ratio), but prior same-family representatives already regressed on full-100 around 5.237e-07; not enough new mechanism to spend validation. |
| `rqmc030` | `whest_0718_rqmc030_ttt` | 47 | `328502be34ba` | 2.567458659e-07 | 7.058903861e-07 | 3.6485e-01 | 0 | 77.707% | `reject_ranked_not_full100` | Strong first-10 gain (77.7% by recorded ratio), but prior same-family representatives already regressed on full-100 around 5.237e-07; not enough new mechanism to spend validation. |
| `rqmc030` | `whest_0718_rqmc030_ttt` | 45 | `6e04b0248fdc` | 2.573239868e-07 | 7.318813459e-07 | 3.5246e-01 | 0 | 77.308% | `reject_ranked_not_full100` | Strong first-10 gain (77.3% by recorded ratio), but prior same-family representatives already regressed on full-100 around 5.237e-07; not enough new mechanism to spend validation. |
| `rqmc030` | `whest_0718_rqmc030_ttt` | 45 | `c72cbb65ee7f` | 2.856901756e-07 | 1.248476838e-06 | 2.2894e-01 | 0 | 59.703% | `reject_ranked_not_full100` | Strong first-10 gain (59.7% by recorded ratio), but prior same-family representatives already regressed on full-100 around 5.237e-07; not enough new mechanism to spend validation. |
| `rqmc030` | `whest_0718_rqmc030_ttt` | 47 | `b9601c3189fb` | 5.347783807e-01 | 1.473377073e+00 | 3.6309e-01 | 0 | -100.000% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `rqmc030` | `whest_0718_rqmc030_ttt` | 48 | `69525d9729d1` | 5.355118619e-01 | 1.473377109e+00 | 3.6350e-01 | 0 | -100.000% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `layerrescale` | `whest_0718_layerrescale_auto_r2` | 7 | `0442b1c90a6e` | - | - | - | - | - | `reject_no_first10` | No verified first-10 construction/message score was available in scanned artifacts. |

Interpretation:
- GaussianCV step 13 remained within first-10 noise of the same-family initial baseline.
- RQMC030 auto_r2 step 8 and TTT steps 44-48 were lower-ranked or weaker same-family variants; prior and current full-100 evidence shows this RQMC family remains in the `4.55e-07` to `5.34e-07` band despite strong first-10 scores.
- The LayerRescale workspace candidate had no verified first-10 construction/message score in scanned artifacts, so it was not promoted.

### Official Submission Status

- No `whest package` or `whest submit` was run because the only full-100 candidate failed the local quality bar.
- No new submission ID was obtained; registry was not changed after the required pre-review tracker refresh.
- Live official submission would also require credential/quota recovery because registered submission refreshes still report HTTP 401, so server quota could not be verified.
- Official submission IDs this cycle: none.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-19T00:40:13+00:00`; do not treat this cycle's `449` inventoried SHA or `14` new SHA as new.
- Latest scanned TTT steps at cutoff: `whest_0718_affinef64_ttt` `50`, `whest_0718_gaussiancv_ttt` `50`, `whest_0718_german013_ttt` `50`, `whest_0718_layerrescale_ttt` `50`, `whest_0718_rqmc030_ttt` `48`, `whest_0718_sphericalrb_ttt` `50`.
- Latest scanned AutoEvolve pool steps at cutoff: `whest_0718_affinef64_auto_r1` `6`, `whest_0718_affinef64_auto_r2` `7`, `whest_0718_gaussiancv_auto_r1` `10`, `whest_0718_gaussiancv_auto_r2` `13`, `whest_0718_german013_auto_r1` `4`, `whest_0718_german013_auto_r2` `3`, `whest_0718_layerrescale_auto_r1` `4`, `whest_0718_layerrescale_auto_r2` `9`, `whest_0718_rqmc030_auto_r1` `8`, `whest_0718_rqmc030_auto_r2` `8`, `whest_0718_sphericalrb_auto_r1` `5`, `whest_0718_sphericalrb_auto_r2` `5`.
- Include artifacts excluded after cutoff next round, starting with:
- `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r1/autoevolve_workspaces/step_000006/call_0001_ff8469ab/submission.py`
- `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r2/autoevolve_workspaces/step_000007/call_0001_3f38fbbe/submission.py`
- `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r2/autoevolve_workspaces/step_000013/call_0001_4b9bc10b/submission.py`
- `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r1/autoevolve_workspaces/step_000008/call_0001_bb6e2e10/submission.py`
- `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r2/autoevolve_workspaces/step_000008/call_0001_ac9825fe/submission.py`
- `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_ttt/puct_sampler_step_000049.json`
- `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r2/autoevolve_workspaces/step_000005/call_0001_7024fcae/submission.py`

## Cycle 20260719T012614Z hourly submission review

- Completed at UTC: `2026-07-19T01:34:54+00:00`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- Cached official status counts after refresh: `{'failed': 1, 'graded': 24}`. Pending/queued/running registered submissions: `[]`. No historical pending status changes needed ledger backfill.
- Quota assessment for UTC `2026-07-19`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`. Blocker: `AIcrowdAuthError: While checking submission status: Your AIcrowd API key is missing or invalid. (HTTP 401)`.
- AIcrowd refresh/auth status: `AIcrowdAuthError HTTP 401 for registered submissions; server quota/status unavailable`.
- Candidate cutoff UTC: `2026-07-19T01:26:14+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-19T01:05:55.425035+00:00']`.
- JSONL context files were read for sampler/metrics/agent/gen-score line counts while live runs continued; live mtime range UTC: `['2026-07-18T17:40:25.175395+00:00', '2026-07-19T01:05:55.424036+00:00']`.
- Console logs location checked for run context: `codex_runs/aicrowd_whestbench/multi_initial_20260718/console`.
- Files excluded after cutoff: `1`; example: `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000005/call_0001_1818ccb9/submission.py`.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T012614Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T012614Z/scan_summary.json`
- Shortlist manifest: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T012614Z/shortlist_manifest.json`
- Shortlist sources: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T012614Z/shortlist_sources`
- Full-100 reports: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T012614Z/full100_reports`
- Initial estimator SHA baselines scanned under `initial_states`: `{'affinef64': '1cb0eb3a875702386c346d605880190c7827b392860c7ddb6fb690f9f9b71435', 'gaussiancv': 'd49b8b5218ff9b9462a331812e688e125fcfff1f86e8a26e9be9c8f59eb48ebb', 'german013': '68d409403de29c8fa7807a8274066494e9ce00380bbbe5bf3d1d52e4ba35aa64', 'layerrescale': '98da6a0882cbc02d32cbd20e295c80b21bb8c173e3e23bb0d6880233d611d7ff', 'rqmc030': 'dc60550f76e9f4f6cb5940af295cab777a4082412590a49665e9127b535eb19f', 'sphericalrb': '146a8c786ee31dfefc941712ded736c3ec2625343dd5cd184f80dde915a9d130'}`

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count | Excluded after cutoff |
|---|---|---|---:|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | - | 0-8 | 9 | 18 | 0 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | - | 0-8 | 9 | 18 | 0 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | - | 0-10 | 11 | 22 | 0 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | - | 0-15 | 16 | 32 | 0 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | - | 0-5 | 5 | 11 | 1 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | - | 0-3 | 4 | 8 | 0 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | - | 0-11 | 12 | 24 | 0 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | - | 0-9 | 10 | 20 | 0 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | - | 0-9 | 10 | 20 | 0 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | 0-50 | - | 0 | 51 | 0 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `464`.
- Historical/known SHA baseline size: `668`; old inventory entries read: `3463`.
- New SHA after historical baseline removal: `15`.
- Final decision counts: `{'reject_existing_sha': 449, 'reject_full100_no_submit': 1, 'reject_no_first10': 7, 'reject_ranked_not_full100': 3, 'reject_weak_first10': 4}`.
- Entered full-100: `1`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Full-100 Shortlist Decisions

| Family | Run | Step | State | SHA | First-10 adjusted / MSE / C-B / fail | Full-100 adjusted / MSE / C-B / fail | vs initial / vs parent | Decision | Candidate path |
|---|---|---:|---|---|---|---|---|---|---|
| `affinef64` | `whest_0718_affinef64_auto_r1` | 7 | `4c1ea66e-a916-4dc3-ab27-d5edbc333c8b` | `921b581662fe` | 1.275691847e-07 / 7.505604287e-07 / 1.6998e-01 / 0 | 3.331608270e-07 / 1.963875598e-06 / 1.6972e-01 / 0 | 73.607% / 28.431% | `reject_full100_no_submit` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T012614Z/shortlist_sources/affinef64_candidate_921b581662fe.py` |

Full-100 interpretation:
- `921b581662fe` is a setup-cached whitened antithetic AffineF64 estimator with radialized first-layer samples and a tuned Gaussian/ReLU hidden mean correction ramp. It is a substantive mechanism change from the prior parent, not just a filename, seed, budget-only, or tiny coefficient change.
- Local first-10 was strong: adjusted `1.275691847e-07`, final-layer MSE `7.505604287e-07`, mean C/B `0.16998`, 0 failures, improving the AffineF64 initial by `73.607%` by the recorded ratio and its parent by `28.431%`.
- Full-100 report `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T012614Z/full100_reports/affinef64_hidden_mean_921b581662fe_full100_20260719T012614Z.json`: adjusted `3.331608270e-07`, MSE `1.963875598e-06`, C/B `0.16972`, failures `0`, failure_breakdown `{'budget_exhausted': 0, 'combined_budget_exhausted': 0, 'error': 0, 'residual_wall_time_exhausted': 0, 'time_exhausted': 0}`.
- The candidate is clean but still weaker than the registered local full-100 best `2.748615e-07` and the best registered hosted public-50 score `2.787552892e-07` (#316192). No package was built and no official submit was attempted.

### Not Promoted After Ranking

| Family | Run | Step | SHA | First-10 adjusted | First-10 MSE | C/B | Fail | vs initial | Decision | Reason |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `affinef64` | `whest_0718_affinef64_auto_r1` | 7 | `07b20a3cb799` | 1.275939272e-07 | 7.505604287e-07 | 1.7000e-01 | 0 | 73.573% | `reject_ranked_not_full100` | Lower-ranked same-family/method variant; a stronger representative was selected for full-100. |
| `gaussiancv` | `whest_0718_gaussiancv_auto_r2` | 14 | `bd395089949d` | 1.482247684e-07 | 5.995594648e-07 | 2.4732e-01 | 0 | -0.077% | `reject_weak_first10` | First-10 does not show a meaningful improvement over the same-family initial baseline. |
| `rqmc030` | `whest_0718_rqmc030_ttt` | 49 | `30b39497a432` | 2.464507417e-07 | 6.784204515e-07 | 3.6409e-01 | 0 | 85.130% | `reject_ranked_not_full100` | Strong first-10 gain, but prior same-family representatives repeatedly regressed on full-100 around the `4.55e-07` to `5.34e-07` band; not enough new mechanism to spend another validation. |
| `rqmc030` | `whest_0718_rqmc030_ttt` | 50 | `9fe12798dd76` | 2.500555052e-07 | 6.810548115e-07 | 3.6697e-01 | 0 | 82.461% | `reject_ranked_not_full100` | Strong first-10 gain, but lower-ranked relative to the step-49 RQMC TTT variant and still not a new mechanism versus prior rejected RQMC controls. |
| `rqmc030` | `whest_0718_rqmc030_ttt` | 49 | `fb44e118b6bf` | 5.324626915e-01 | 1.473377109e+00 | 3.6200e-01 | 0 | -100.000% | `reject_weak_first10` | First-10 is catastrophically weak versus the same-family initial baseline. |
| `rqmc030` | `whest_0718_rqmc030_ttt` | 50 | `03ddc4a24cd3` | 5.373099177e-01 | 1.473377109e+00 | 3.6457e-01 | 0 | -100.000% | `reject_weak_first10` | First-10 is catastrophically weak versus the same-family initial baseline. |
| `rqmc030` | `whest_0718_rqmc030_ttt` | 50 | `86edf495d1f0` | 5.715410984e-01 | 1.473386255e+00 | 3.8806e-01 | 0 | -100.000% | `reject_weak_first10` | First-10 is catastrophically weak versus the same-family initial baseline. |

Additional no-score workspace/submission candidates:

| Family | Run | Step | SHA | Source | Reason |
|---|---|---:|---|---|---|
| `affinef64` | `whest_0718_affinef64_auto_r2` | 7 | `6b00b6e824e0` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r2/autoevolve_workspaces/step_000007/call_0001_3f38fbbe/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `affinef64` | `whest_0718_affinef64_auto_r1` | 7 | `a2cc7d82023b` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_affinef64_auto_r1/autoevolve_workspaces/step_000007/call_0001_d0800da2/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `gaussiancv` | `whest_0718_gaussiancv_auto_r2` | 14 | `5d4bf80c099e` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_gaussiancv_auto_r2/autoevolve_workspaces/step_000014/call_0001_5d414296/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `layerrescale` | `whest_0718_layerrescale_auto_r2` | 9 | `21d69bc89853` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_layerrescale_auto_r2/autoevolve_workspaces/step_000009/call_0001_85ed2174/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `rqmc030` | `whest_0718_rqmc030_auto_r2` | 8 | `19911501c0cc` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r2/autoevolve_workspaces/step_000008/call_0001_ac9825fe/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `rqmc030` | `whest_0718_rqmc030_auto_r1` | 8 | `2d645d863c51` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_rqmc030_auto_r1/autoevolve_workspaces/step_000008/call_0001_bb6e2e10/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |
| `sphericalrb` | `whest_0718_sphericalrb_auto_r2` | 5 | `02d5bdc0dadd` | `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_sphericalrb_auto_r2/autoevolve_workspaces/step_000005/call_0001_7024fcae/submission.py` | No verified first-10 construction/message score was available in scanned artifacts. |

Interpretation:
- AffineF64 produced the only worthwhile signal this cycle. The first-10 score was excellent, but full-100 moved to `3.331608270e-07`, so it does not clear the local quality bar for hosted public-50 validation.
- GaussianCV remained within first-10 noise of its initial baseline.
- RQMC030 TTT steps 49-50 included two superficially strong first-10 variants, but the mechanism was near prior RQMC control/lattice candidates that already failed full-100; lower-quality variants were rejected without another expensive validation.
- Seven workspace-only submissions had no verified first-10 construction/message metrics in the scanned pool/state artifacts.

### Official Submission Status

- No `whest package` or `whest submit` was run because the only full-100 candidate failed the local quality bar.
- No new submission ID was obtained; registry was not changed after the required pre-review tracker refresh.
- Live official submission would also require credential/quota recovery because registered submission refreshes still report HTTP 401, so server quota could not be verified.
- Official submission IDs this cycle: none.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-19T01:26:14+00:00`; do not treat this cycle's `464` inventoried SHA or `15` new SHA as new.
- Latest scanned TTT steps at cutoff: `whest_0718_affinef64_ttt` `50`, `whest_0718_gaussiancv_ttt` `50`, `whest_0718_german013_ttt` `50`, `whest_0718_layerrescale_ttt` `50`, `whest_0718_rqmc030_ttt` `50`, `whest_0718_sphericalrb_ttt` `50`.
- Latest scanned AutoEvolve pool steps at cutoff: `whest_0718_affinef64_auto_r1` `8`, `whest_0718_affinef64_auto_r2` `8`, `whest_0718_gaussiancv_auto_r1` `10`, `whest_0718_gaussiancv_auto_r2` `15`, `whest_0718_german013_auto_r1` `5`, `whest_0718_german013_auto_r2` `3`, `whest_0718_layerrescale_auto_r1` `6`, `whest_0718_layerrescale_auto_r2` `11`, `whest_0718_rqmc030_auto_r1` `9`, `whest_0718_rqmc030_auto_r2` `9`, `whest_0718_sphericalrb_auto_r1` `6`, `whest_0718_sphericalrb_auto_r2` `6`.
- Include artifacts excluded after cutoff next round, starting with `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000005/call_0001_1818ccb9/submission.py`.

## Cycle 20260719T020934Z hourly submission review

- Completed at UTC: `2026-07-19T02:15:29+00:00`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- Cached official status counts after refresh: `{'failed': 1, 'graded': 24}`. Pending/queued/running registered submissions: `[]`. No historical pending status changes needed ledger backfill.
- Quota assessment for UTC `2026-07-19`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`. Blocker: `AIcrowdAuthError: While checking submission status: Your AIcrowd API key is missing or invalid. (HTTP 401)`.
- AIcrowd refresh/auth status: `AIcrowdAuthError HTTP 401 for registered submissions; server quota/status unavailable`.
- Candidate cutoff UTC: `2026-07-19T02:09:34+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-19T02:05:33.018156+00:00']`.
- JSONL context files were read for sampler/metrics/agent/gen-score line counts while live runs continued; live mtime range was `['2026-07-18T17:40:25.175395+00:00', '2026-07-19T02:05:33.018156+00:00']`.
- Console logs inspected for timestamp context: `18` files under `codex_runs/aicrowd_whestbench/multi_initial_20260718/console`; mtime range UTC `['2026-07-18T14:51:18.876898+00:00', '2026-07-19T01:16:53.661911+00:00']`.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T020934Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T020934Z/scan_summary.json`
- Shortlist manifest: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T020934Z/shortlist_manifest.json`
- Shortlist sources directory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T020934Z/shortlist_sources`
- Initial estimator SHA baselines scanned under `initial_states`: `{'affinef64': '1cb0eb3a875702386c346d605880190c7827b392860c7ddb6fb690f9f9b71435', 'gaussiancv': 'd49b8b5218ff9b9462a331812e688e125fcfff1f86e8a26e9be9c8f59eb48ebb', 'german013': '68d409403de29c8fa7807a8274066494e9ce00380bbbe5bf3d1d52e4ba35aa64', 'layerrescale': '98da6a0882cbc02d32cbd20e295c80b21bb8c173e3e23bb0d6880233d611d7ff', 'rqmc030': 'dc60550f76e9f4f6cb5940af295cab777a4082412590a49665e9127b535eb19f', 'sphericalrb': '146a8c786ee31dfefc941712ded736c3ec2625343dd5cd184f80dde915a9d130'}`
- Historical/known SHA baseline size: `683`; old inventory entries read: `3927`.
- Files excluded after cutoff: `1`; example: `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000006/call_0001_77a9e409/submission.py`.

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count | Excluded after cutoff |
|---|---|---|---:|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | - | 0-8 | 9 | 18 | 0 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | - | 0-8 | 9 | 18 | 0 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | - | 0-10 | 11 | 22 | 0 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | - | 0-15 | 16 | 32 | 0 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | - | 0-6 | 6 | 13 | 1 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | - | 0-3 | 4 | 8 | 0 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | - | 0-11 | 12 | 24 | 0 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | - | 0-9 | 10 | 20 | 0 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | - | 0-9 | 10 | 20 | 0 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | 0-50 | - | 0 | 51 | 0 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `466`.
- New SHA after historical baseline removal: `2`.
- Semantic new method count after trailing-newline duplicate grouping: `1`.
- Final decision counts: `{'reject_existing_sha': 464, 'reject_ranked_not_full100': 1, 'reject_micro_variant_not_full100': 1}`.
- Entered full-100: `0`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Shortlist And Decisions

No candidate entered full-100. The only semantic new candidate did not clear the evidence bar; the second new SHA was the same code without a final newline.

| Family | Run | Step | State | SHA | First-10 adjusted / MSE / C-B / fail | vs initial / vs parent | Decision | Reason |
|---|---|---:|---|---|---|---|---|---|
| `german013` | `whest_0718_german013_auto_r1` | 6 | `4ebe72e6-4ee5-4868-8159-f5e4f9945e53` | `60da1944b32f` | 1.575377758e-07 / 1.500356291e-06 / 1.0514e-01 / 0 | 59.713% / 2.609% | `reject_ranked_not_full100` | Radial/first-layer mean-correction variant. It is clean on first-10, but weaker than prior German representatives that already failed full-100 (`1.385e-07` to `1.424e-07` first-10 -> `3.49e-07` to `3.56e-07` full-100), with only a 2.609% parent gain. |
| `german013` | `whest_0718_german013_auto_r1` | 6 | `7630cef3-7752-4726-9af9-ad7c439c93ce` | `c065dd579a0d` | 1.577919089e-07 / 1.500356291e-06 / 1.0537e-01 / 0 | 59.456% / 2.444% | `reject_micro_variant_not_full100` | Semantically identical to `60da1944b32f075e0463300cca35dc186d03e26ba117713a1681ac771ef8a493` except for a missing final newline; first-10 is slightly worse and this is not a new method. |

Interpretation:

- The German013 candidate adds radial normalization, first-layer mean correction, and extra-block weighting over parent `5d77014514444380d917d38a3f5fbd549b814aae8e367041213bfd7fddd84cd3`.
- It is a plausible mechanism change from its immediate parent, but evidence is not strong enough for full-100 because earlier, stronger German first-10 candidates generalized to the `3.49e-07` to `3.56e-07` full-100 band, still worse than the registered local full-100 best `2.748615e-07` and best registered hosted public-50 `2.787552892e-07` (#316192).
- No no-score workspace candidate was promoted; the only file excluded after cutoff belongs to the next cycle.

### Official Submission Status

- No full-100 evaluator was run this cycle because no non-duplicate candidate cleared the first-10/method evidence bar.
- No `whest package` or `whest submit` was run.
- No new submission ID was obtained; registry was not changed after the required pre-review tracker refresh.
- Official submission IDs this cycle: none.
- Live official submission would also require credential/quota recovery because registered submission refreshes still report HTTP 401, so server quota could not be verified.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-19T02:09:34+00:00`; do not treat this cycle's `466` inventoried SHA, `2` raw new SHA, or `1` semantic new method as new.
- Latest scanned TTT steps at cutoff: `whest_0718_affinef64_ttt` `50`, `whest_0718_gaussiancv_ttt` `50`, `whest_0718_german013_ttt` `50`, `whest_0718_layerrescale_ttt` `50`, `whest_0718_rqmc030_ttt` `50`, `whest_0718_sphericalrb_ttt` `50`.
- Latest scanned AutoEvolve pool steps at cutoff: `whest_0718_affinef64_auto_r1` `8`, `whest_0718_affinef64_auto_r2` `8`, `whest_0718_gaussiancv_auto_r1` `10`, `whest_0718_gaussiancv_auto_r2` `15`, `whest_0718_german013_auto_r1` `6`, `whest_0718_german013_auto_r2` `3`, `whest_0718_layerrescale_auto_r1` `6`, `whest_0718_layerrescale_auto_r2` `11`, `whest_0718_rqmc030_auto_r1` `9`, `whest_0718_rqmc030_auto_r2` `9`, `whest_0718_sphericalrb_auto_r1` `6`, `whest_0718_sphericalrb_auto_r2` `6`.
- Include artifacts excluded after cutoff next round, starting with `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000006/call_0001_77a9e409/submission.py`.

## Cycle 20260719T024836Z hourly submission review

- Completed at UTC: `2026-07-19T02:54:40+00:00`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- Cached official status counts after refresh: `{'failed': 1, 'graded': 24}`. Pending/queued/running registered submissions: `[]`. No historical pending status changes needed ledger backfill.
- Quota assessment for UTC `2026-07-19`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`. Blocker: `AIcrowdAuthError: While checking submission status: Your AIcrowd API key is missing or invalid. (HTTP 401)`.
- AIcrowd refresh/auth status: `AIcrowdAuthError HTTP 401 for registered submissions; server quota/status unavailable`.
- Candidate cutoff UTC: `2026-07-19T02:48:36+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-19T02:05:33.018156+00:00']`.
- JSONL context files were read for sampler/metrics/agent/gen-score line counts while live runs continued; live mtime range was `['2026-07-18T17:40:25.175395+00:00', '2026-07-19T02:05:33.018156+00:00']`.
- Console logs inspected for timestamp context: `18` files under `codex_runs/aicrowd_whestbench/multi_initial_20260718/console`; mtime range UTC `['2026-07-18T14:51:18.876898+00:00', '2026-07-19T01:16:53.661911+00:00']`.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T024836Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T024836Z/scan_summary.json`
- Shortlist manifest: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T024836Z/shortlist_manifest.json`
- Shortlist sources directory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T024836Z/shortlist_sources`
- Initial estimator SHA baselines scanned under `initial_states`: `{'affinef64': '1cb0eb3a875702386c346d605880190c7827b392860c7ddb6fb690f9f9b71435', 'gaussiancv': 'd49b8b5218ff9b9462a331812e688e125fcfff1f86e8a26e9be9c8f59eb48ebb', 'german013': '68d409403de29c8fa7807a8274066494e9ce00380bbbe5bf3d1d52e4ba35aa64', 'layerrescale': '98da6a0882cbc02d32cbd20e295c80b21bb8c173e3e23bb0d6880233d611d7ff', 'rqmc030': 'dc60550f76e9f4f6cb5940af295cab777a4082412590a49665e9127b535eb19f', 'sphericalrb': '146a8c786ee31dfefc941712ded736c3ec2625343dd5cd184f80dde915a9d130'}`
- Historical/known SHA baseline size: `685`; old inventory entries read: `4393`.
- Parse errors: `[]`.
- Files excluded after cutoff: `1`; example: `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000006/call_0001_77a9e409/submission.py` with filesystem mtime `2026-07-19T02:52:04.932857+00:00`, so it belongs to the next cycle.

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count | Excluded after cutoff |
|---|---|---|---:|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | - | 0-8 | 9 | 18 | 0 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | - | 0-8 | 9 | 18 | 0 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | - | 0-10 | 11 | 22 | 0 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | - | 0-15 | 16 | 32 | 0 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | - | 0-6 | 6 | 13 | 1 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | - | 0-3 | 4 | 8 | 0 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | - | 0-11 | 12 | 24 | 0 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | - | 0-9 | 10 | 20 | 0 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | - | 0-9 | 10 | 20 | 0 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | 0-50 | - | 0 | 51 | 0 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `466`.
- New SHA after historical baseline removal: `0`.
- Final decision counts: `{'reject_existing_sha': 466}`.
- Entered full-100: `0`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Shortlist And Decisions

No candidate entered full-100. Every estimator SHA observed before the cutoff was already present in the ledger/registry/tracker/old inventories/canonical candidate baseline.

Best already-known first-10 records observed while scanning:

| Family | SHA | Run | Step | First-10 adjusted | Decision |
|---|---|---|---:|---:|---|
| `affinef64` | `921b581662fe` | `whest_0718_affinef64_auto_r1` | 7 | 1.275691847e-07 | `reject_existing_sha` |
| `layerrescale` | `3f26d2bb11e5` | `whest_0718_layerrescale_auto_r2` | 5 | 1.357858217e-07 | `reject_existing_sha` |
| `german013` | `8c0a803abd3b` | `whest_0718_german013_auto_r2` | 2 | 1.385124267e-07 | `reject_existing_sha` |
| `sphericalrb` | `3c73a750aa29` | `whest_0718_sphericalrb_auto_r1` | 5 | 1.461057215e-07 | `reject_existing_sha` |
| `gaussiancv` | `a0f5bc4922b1` | `whest_0718_gaussiancv_auto_r1` | 4 | 1.467102880e-07 | `reject_existing_sha` |
| `rqmc030` | `7165e74c9196` | `whest_0718_rqmc030_auto_r1` | 8 | 1.675358334e-07 | `reject_existing_sha` |

Interpretation:

- Since no new SHA appeared before the cutoff, there was no new method-change representative and no significant first-10 improvement to validate.
- No local mini full-100 evaluation was run.
- No package was built and no official submit was attempted.

### Official Submission Status

- No `whest package` or `whest submit` was run.
- No new submission ID was obtained; registry was not changed after the required pre-review tracker refresh.
- `SUBMISSION_TRACKER.md` and `official_status.json` were refreshed by the required script. `registry.json` was unchanged.
- Official submission IDs this cycle: none.
- Live official submission would require credential/quota recovery because registered submission refreshes still report HTTP 401, so server quota could not be verified.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-19T02:48:36+00:00`; do not treat this cycle's `466` inventoried SHA as new.
- Latest scanned TTT steps at cutoff: `whest_0718_affinef64_ttt` `50`, `whest_0718_gaussiancv_ttt` `50`, `whest_0718_german013_ttt` `50`, `whest_0718_layerrescale_ttt` `50`, `whest_0718_rqmc030_ttt` `50`, `whest_0718_sphericalrb_ttt` `50`.
- Latest scanned AutoEvolve pool steps at cutoff: `whest_0718_affinef64_auto_r1` `8`, `whest_0718_affinef64_auto_r2` `8`, `whest_0718_gaussiancv_auto_r1` `10`, `whest_0718_gaussiancv_auto_r2` `15`, `whest_0718_german013_auto_r1` `6`, `whest_0718_german013_auto_r2` `3`, `whest_0718_layerrescale_auto_r1` `6`, `whest_0718_layerrescale_auto_r2` `11`, `whest_0718_rqmc030_auto_r1` `9`, `whest_0718_rqmc030_auto_r2` `9`, `whest_0718_sphericalrb_auto_r1` `6`, `whest_0718_sphericalrb_auto_r2` `6`.
- Include artifacts excluded after cutoff next round, starting with `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000006/call_0001_77a9e409/submission.py`.

## Cycle 20260719T033123Z hourly submission review

- Completed at UTC: `2026-07-19T03:36:10+00:00`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- Cached official status counts after refresh: `{'graded': 24, 'failed': 1}`. Pending/queued/running registered submissions: `[]`. No historical pending status changes needed ledger backfill.
- Quota assessment for UTC `2026-07-19`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`. Blocker: `AIcrowdAuthError: While checking submission status: Your AIcrowd API key is missing or invalid. (HTTP 401)`.
- Candidate cutoff UTC: `2026-07-19T03:31:23+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-19T03:12:35.869265+00:00']`.
- JSONL context files were read for sampler/metrics/agent/gen-score line counts while live runs continued; live mtime range was `['2026-07-18T17:40:25.175395+00:00', '2026-07-19T03:12:35.869265+00:00']`.
- Console logs inspected for timestamp/error context: `18` files under `codex_runs/aicrowd_whestbench/multi_initial_20260718/console`; mtime range UTC `['2026-07-18T14:51:18.876898+00:00', '2026-07-19T01:16:53.661911+00:00']`. Recent lines were connection reset, timeout, candidate failure, or terminated messages; no candidate was promoted from console text.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T033123Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T033123Z/scan_summary.json`
- Shortlist manifest: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T033123Z/shortlist_manifest.json`
- Shortlist sources directory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T033123Z/shortlist_sources`
- Manual diff inspection copies: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T033123Z/inspection_sources`
- Initial estimator SHA baselines scanned under `initial_states`: `{'affinef64': '1cb0eb3a875702386c346d605880190c7827b392860c7ddb6fb690f9f9b71435', 'gaussiancv': 'd49b8b5218ff9b9462a331812e688e125fcfff1f86e8a26e9be9c8f59eb48ebb', 'german013': '68d409403de29c8fa7807a8274066494e9ce00380bbbe5bf3d1d52e4ba35aa64', 'layerrescale': '98da6a0882cbc02d32cbd20e295c80b21bb8c173e3e23bb0d6880233d611d7ff', 'rqmc030': 'dc60550f76e9f4f6cb5940af295cab777a4082412590a49665e9127b535eb19f', 'sphericalrb': '146a8c786ee31dfefc941712ded736c3ec2625343dd5cd184f80dde915a9d130'}`
- Historical/known SHA baseline size: `685`; old inventory entries read: `4859`.
- Parse errors: `[]`.
- Files excluded after cutoff: `1`; example: `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000007/call_0001_c63d8712/submission.py` with filesystem mtime `2026-07-19T03:34:16.050839+00:00`, so it belongs to the next cycle.

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count | Excluded after cutoff |
|---|---|---|---:|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | - | 0-8 | 9 | 18 | 0 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | - | 0-8 | 9 | 18 | 0 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | - | 0-10 | 11 | 22 | 0 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | - | 0-15 | 16 | 32 | 0 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | - | 0-7 | 7 | 15 | 1 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | - | 0-3 | 4 | 8 | 0 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | - | 0-11 | 12 | 24 | 0 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | - | 0-9 | 10 | 20 | 0 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | - | 0-9 | 10 | 20 | 0 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | 0-50 | - | 0 | 51 | 0 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `468`.
- New SHA after historical baseline removal: `2`.
- Semantic new method count after formatting-only duplicate grouping: `1`.
- Final decision counts: `{'reject_existing_sha': 466, 'reject_ranked_not_full100': 1, 'reject_micro_variant_not_full100': 1}`.
- Entered full-100: `0`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Shortlist And Decisions

No candidate entered full-100. Both raw new SHA came from `whest_0718_german013_auto_r1` pool step `7`; one is a small German013 method variant and the other is a formatting-only duplicate of it.

| Family | Run | Step | State | SHA | First-10 adjusted / MSE / C-B / fail | vs initial / vs parent | Decision | Reason |
|---|---|---:|---|---|---|---|---|---|
| `german013` | `whest_0718_german013_auto_r1` | 7 | `0bba570c-a6bb-4c87-a8f2-cd71607d8e60` | `248ac1b78df8` | 1.570191733e-07 / 1.493164848e-06 / 1.053322e-01 / 0 | 60.241% / 0.330% | `reject_ranked_not_full100` | Adds first-layer target mean plus second-moment rescale and an extra block with negative mean strength over parent `60da1944b32f`; this is a real but small same-cluster change. It improved its immediate parent by only `0.330%` first-10 and remains weaker than prior German first-10 representatives (`1.385e-07` to `1.424e-07`) that full-100 regressed to `3.49e-07` to `3.56e-07`, below the local/official quality bar. |
| `german013` | `whest_0718_german013_auto_r1` | 7 | `a6cdffbd-e04b-4292-810c-7f2fb104b762` | `043a449fcd16` | 1.570233965e-07 / 1.493164848e-06 / 1.053339e-01 / 0 | 60.236% / 0.328% | `reject_micro_variant_not_full100` | Semantically equivalent formatting-only variant of `248ac1b78df87c37335d3bbcbd31c8d7e597e00f1b8fd3314bfda61a684116d4`; first-10 is slightly worse and this is not a new method. |

Interpretation:

- The only semantic new candidate is clean on first-10 but does not show a significant improvement over its parent or over stronger already-reviewed German candidates.
- No local mini full-100 evaluation was run, because the evidence was insufficient and prior same-family full-100 results already showed this first-10 band does not beat the registered local full-100 best `2.748615e-07` or the best registered hosted public-50 score `2.787552892e-07` (#316192).
- No package was built and no official submit was attempted.

### Official Submission Status

- No `whest package` or `whest submit` was run.
- No new submission ID was obtained; registry was not changed after the required pre-review tracker refresh.
- `SUBMISSION_TRACKER.md` and `official_status.json` were refreshed by the required script. `registry.json` was unchanged.
- Official submission IDs this cycle: none.
- Live official submission would require credential/quota recovery because registered submission refreshes still report HTTP 401, so server quota could not be verified.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-19T03:31:23+00:00`; do not treat this cycle's `468` inventoried SHA, `2` raw new SHA, or `1` semantic new method as new.
- Latest scanned TTT steps at cutoff: `whest_0718_affinef64_ttt` `50`, `whest_0718_gaussiancv_ttt` `50`, `whest_0718_german013_ttt` `50`, `whest_0718_layerrescale_ttt` `50`, `whest_0718_rqmc030_ttt` `50`, `whest_0718_sphericalrb_ttt` `50`.
- Latest scanned AutoEvolve pool steps at cutoff: `whest_0718_affinef64_auto_r1` `8`, `whest_0718_affinef64_auto_r2` `8`, `whest_0718_gaussiancv_auto_r1` `10`, `whest_0718_gaussiancv_auto_r2` `15`, `whest_0718_german013_auto_r1` `7`, `whest_0718_german013_auto_r2` `3`, `whest_0718_layerrescale_auto_r1` `6`, `whest_0718_layerrescale_auto_r2` `11`, `whest_0718_rqmc030_auto_r1` `9`, `whest_0718_rqmc030_auto_r2` `9`, `whest_0718_sphericalrb_auto_r1` `6`, `whest_0718_sphericalrb_auto_r2` `6`.
- Include artifacts excluded after cutoff next round, starting with `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000007/call_0001_c63d8712/submission.py`.

## Cycle 20260719T041043Z hourly submission review

- Completed at UTC: `2026-07-19T04:15:00+00:00`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- Cached official status counts after refresh: `{'graded': 24, 'failed': 1}`. Pending/queued/running registered submissions: `[]`. No historical pending status changes needed ledger backfill.
- Quota assessment for UTC `2026-07-19`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`. Blocker: `AIcrowdAuthError: While checking submission status: Your AIcrowd API key is missing or invalid. (HTTP 401)`.
- Candidate cutoff UTC: `2026-07-19T04:10:43+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-19T03:12:35.869265+00:00']`.
- JSONL context files were read for sampler/metrics/agent/gen-score line counts while live runs continued; live mtime range UTC: `['2026-07-18T17:40:25.175395+00:00', '2026-07-19T03:12:35.869265+00:00']`.
- Console logs inspected for timestamp context: `18` files under `codex_runs/aicrowd_whestbench/multi_initial_20260718/console`; mtime range UTC `['2026-07-18T14:51:18.876898+00:00', '2026-07-19T01:16:53.661911+00:00']`.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T041043Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T041043Z/scan_summary.json`
- Shortlist manifest: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T041043Z/shortlist_manifest.json`
- Shortlist sources directory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T041043Z/shortlist_sources`
- Initial estimator SHA baselines scanned under `initial_states`: `{'affinef64': '1cb0eb3a875702386c346d605880190c7827b392860c7ddb6fb690f9f9b71435', 'gaussiancv': 'd49b8b5218ff9b9462a331812e688e125fcfff1f86e8a26e9be9c8f59eb48ebb', 'german013': '68d409403de29c8fa7807a8274066494e9ce00380bbbe5bf3d1d52e4ba35aa64', 'layerrescale': '98da6a0882cbc02d32cbd20e295c80b21bb8c173e3e23bb0d6880233d611d7ff', 'rqmc030': 'dc60550f76e9f4f6cb5940af295cab777a4082412590a49665e9127b535eb19f', 'sphericalrb': '146a8c786ee31dfefc941712ded736c3ec2625343dd5cd184f80dde915a9d130'}`
- Historical/known SHA baseline size: `687`; old inventory entries read: `5327`.
- Parse errors: `[]`.
- Files excluded after cutoff: `1`; `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000007/call_0001_c63d8712/submission.py`, mtime `2026-07-19T04:14:36.191972+00:00`, SHA `1df8f655691c466e8af54a852c5b81d2963d3f7e92a01669624b3c185e0663d5`.

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count | Excluded after cutoff |
|---|---|---|---:|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | - | 0-8 | 9 | 18 | 0 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | - | 0-8 | 9 | 18 | 0 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | - | 0-10 | 11 | 22 | 0 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | - | 0-15 | 16 | 32 | 0 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | - | 0-7 | 7 | 15 | 1 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | - | 0-3 | 4 | 8 | 0 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | - | 0-11 | 12 | 24 | 0 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | - | 0-9 | 10 | 20 | 0 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | - | 0-9 | 10 | 20 | 0 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | 0-50 | - | 0 | 51 | 0 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `468`.
- New SHA after historical baseline removal: `0`.
- Final decision counts: `{'reject_existing_sha': 468}`.
- Entered full-100: `0`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Shortlist And Decisions

No candidate entered full-100. Every estimator SHA observed before the cutoff was already present in the ledger/registry/tracker/old inventories/canonical candidate baseline.

Best already-known first-10 records observed while scanning:

| Family | SHA | Run | Step | First-10 adjusted | Decision |
|---|---|---|---:|---:|---|
| `affinef64` | `921b581662fe` | `whest_0718_affinef64_auto_r1` | 7 | 1.275691847e-07 | `reject_existing_sha` |
| `layerrescale` | `3f26d2bb11e5` | `whest_0718_layerrescale_auto_r2` | 5 | 1.357858217e-07 | `reject_existing_sha` |
| `german013` | `8c0a803abd3b` | `whest_0718_german013_auto_r2` | 2 | 1.385124267e-07 | `reject_existing_sha` |
| `sphericalrb` | `3c73a750aa29` | `whest_0718_sphericalrb_auto_r1` | 5 | 1.461057215e-07 | `reject_existing_sha` |
| `gaussiancv` | `a0f5bc4922b1` | `whest_0718_gaussiancv_auto_r1` | 4 | 1.467102880e-07 | `reject_existing_sha` |
| `rqmc030` | `7165e74c9196` | `whest_0718_rqmc030_auto_r1` | 8 | 1.675358334e-07 | `reject_existing_sha` |

Interpretation:

- No raw or semantic new estimator SHA appeared before the cutoff, so there was no new method-change representative and no significant new first-10 signal to validate.
- No local mini full-100 evaluation was run.
- No package was built and no official submit was attempted.

### Official Submission Status

- No `whest package` or `whest submit` was run.
- No new submission ID was obtained; registry was not changed after the required pre-review tracker refresh.
- `SUBMISSION_TRACKER.md` and `official_status.json` were refreshed by the required script. `registry.json` was unchanged.
- Official submission IDs this cycle: none.
- Live official submission would require credential/quota recovery because registered submission refreshes still report HTTP 401, so server quota could not be verified.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-19T04:10:43+00:00`; do not treat this cycle's `468` inventoried SHA as new.
- Latest scanned TTT steps at cutoff: `whest_0718_affinef64_ttt` `50`, `whest_0718_gaussiancv_ttt` `50`, `whest_0718_german013_ttt` `50`, `whest_0718_layerrescale_ttt` `50`, `whest_0718_rqmc030_ttt` `50`, `whest_0718_sphericalrb_ttt` `50`.
- Latest scanned AutoEvolve pool steps at cutoff: `whest_0718_affinef64_auto_r1` `8`, `whest_0718_affinef64_auto_r2` `8`, `whest_0718_gaussiancv_auto_r1` `10`, `whest_0718_gaussiancv_auto_r2` `15`, `whest_0718_german013_auto_r1` `7`, `whest_0718_german013_auto_r2` `3`, `whest_0718_layerrescale_auto_r1` `6`, `whest_0718_layerrescale_auto_r2` `11`, `whest_0718_rqmc030_auto_r1` `9`, `whest_0718_rqmc030_auto_r2` `9`, `whest_0718_sphericalrb_auto_r1` `6`, `whest_0718_sphericalrb_auto_r2` `6`.
- Include artifacts excluded after cutoff next round, starting with `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000007/call_0001_c63d8712/submission.py` (mtime `2026-07-19T04:14:36.191972+00:00`, SHA `1df8f655691c466e8af54a852c5b81d2963d3f7e92a01669624b3c185e0663d5`).

## Cycle 20260719T044747Z hourly submission review

- Completed at UTC: `2026-07-19T04:53:45+00:00`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- Cached official status counts after refresh: `{'failed': 1, 'graded': 24}`. Pending/queued/running registered submissions: `[]`. No historical pending status changes needed ledger backfill.
- Quota assessment for UTC `2026-07-19`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`. Blocker: `AIcrowdAuthError: While checking submission status: Your AIcrowd API key is missing or invalid. (HTTP 401)`.
- AIcrowd refresh/auth status: `AIcrowdAuthError HTTP 401 for registered submissions; server quota/status unavailable`.
- Candidate cutoff UTC: `2026-07-19T04:47:47+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-19T04:18:20.847568+00:00']`.
- JSONL context files were read for sampler/metrics/agent/gen-score line counts while live runs continued; live mtime range UTC: `['2026-07-18T17:40:25.175395+00:00', '2026-07-19T04:18:20.848568+00:00']`.
- Console logs inspected for timestamp/error context: `18` files under `codex_runs/aicrowd_whestbench/multi_initial_20260718/console`; mtime range UTC `['2026-07-18T14:51:18.876898+00:00', '2026-07-19T01:16:53.661911+00:00']`. Recent tails were timeout/connection reset/candidate failed/terminated messages; no candidate was promoted from console text.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T044747Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T044747Z/scan_summary.json`
- Shortlist manifest: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T044747Z/shortlist_manifest.json`
- Shortlist sources directory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T044747Z/shortlist_sources`
- Initial estimator SHA baselines scanned under `initial_states`: `{'affinef64': '1cb0eb3a875702386c346d605880190c7827b392860c7ddb6fb690f9f9b71435', 'gaussiancv': 'd49b8b5218ff9b9462a331812e688e125fcfff1f86e8a26e9be9c8f59eb48ebb', 'german013': '68d409403de29c8fa7807a8274066494e9ce00380bbbe5bf3d1d52e4ba35aa64', 'layerrescale': '98da6a0882cbc02d32cbd20e295c80b21bb8c173e3e23bb0d6880233d611d7ff', 'rqmc030': 'dc60550f76e9f4f6cb5940af295cab777a4082412590a49665e9127b535eb19f', 'sphericalrb': '146a8c786ee31dfefc941712ded736c3ec2625343dd5cd184f80dde915a9d130'}`
- Historical/known SHA baseline size: `688`; old inventory entries read: `5795`.
- Parse errors: `[]`.
- Files excluded after cutoff: `1`; examples: `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000008/call_0001_7223777d/submission.py`.

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count | Excluded after cutoff |
|---|---|---|---:|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | - | 0-8 | 9 | 18 | 0 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | - | 0-8 | 9 | 18 | 0 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | - | 0-10 | 11 | 22 | 0 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | - | 0-15 | 16 | 32 | 0 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | - | 0-8 | 8 | 17 | 1 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | - | 0-3 | 4 | 8 | 0 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | - | 0-11 | 12 | 24 | 0 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | - | 0-9 | 10 | 20 | 0 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | - | 0-9 | 10 | 20 | 0 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | 0-50 | - | 0 | 51 | 0 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `470`.
- New SHA after ledger/registry/tracker/old-inventory baseline removal: `1`.
- Final decision counts: `{'reject_existing_sha': 469, 'reject_ranked_not_full100': 1}`.
- Entered full-100: `0`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Shortlist And Decisions

No candidate entered full-100. One deduped-new SHA appeared in `whest_0718_german013_auto_r1` pool step `8`; it is a small hyperparameter variant in the same German013 radial/normal-blend cluster, not a structural new method.

| SHA | Family / run / step | State | First-10 adjusted / MSE / C-B / fail | vs initial / vs parent | Decision | Reason |
|---|---|---|---|---|---|---|
| `ab948a5f2c58` | `german013` / `whest_0718_german013_auto_r1` / step `8` | `ad030a4f-3022-4992-99eb-8d543330f863` | 1.568964411e-07 / 1.492015554e-06 / 0.10533007496414967 / 0 | 60.366% / 0.078% | `reject_ranked_not_full100` | Strong first-10 gain (60.4% by recorded ratio), but prior same-family representatives already regressed on full-100 around 3.489e-07; not enough new mechanism to spend validation. |

Manual diff interpretation:
- `ab948a5f2c58` changes parent `248ac1b78df8` by splitting the same final normal blend into main/extra constants, increasing `_FIRST_MEAN_STRENGTH` from `0.425` to `0.50`, and reducing `_ARRAY_BYTES_LIMIT` from `100 * 1024 * 1024` to `90 * 1024 * 1024`. First-10 improved only `0.078%` over the parent, so it does not justify full-100 given prior German first-10 representatives in this band full-100-regressed to roughly `3.49e-07` and stayed below the local/official quality bar.
- `1df8f655691c` was inventoried this cycle from the step-8 pool and the previously cutoff-excluded workspace `call_0001_c63d8712`. It was already mentioned in the prior ledger continuation point, so the ledger/registry/tracker/old-inventory dedupe classified it as existing; manual audit confirms it is the same cluster with first-10 `1.569644953e-07`, only `0.035%` over parent, and not a full-100 candidate.

Best already-known first-10 records observed while scanning:

| Family | SHA | Run | Step | First-10 adjusted | Decision |
|---|---|---|---:|---:|---|
| `affinef64` | `921b581662fe` | `whest_0718_affinef64_auto_r1` | 7 | 1.275691847e-07 | `reject_existing_sha` |
| `gaussiancv` | `a0f5bc4922b1` | `whest_0718_gaussiancv_auto_r1` | 4 | 1.467102880e-07 | `reject_existing_sha` |
| `german013` | `8c0a803abd3b` | `whest_0718_german013_auto_r2` | 2 | 1.385124267e-07 | `reject_existing_sha` |
| `layerrescale` | `3f26d2bb11e5` | `whest_0718_layerrescale_auto_r2` | 5 | 1.357858217e-07 | `reject_existing_sha` |
| `rqmc030` | `7165e74c9196` | `whest_0718_rqmc030_auto_r1` | 8 | 1.675358334e-07 | `reject_existing_sha` |
| `sphericalrb` | `3c73a750aa29` | `whest_0718_sphericalrb_auto_r1` | 5 | 1.461057215e-07 | `reject_existing_sha` |

### Official Submission Status

- No local mini full-100 evaluation was run.
- No `whest package` or `whest submit` was run.
- No new submission ID was obtained; registry was not changed after the required pre-review tracker refresh.
- `SUBMISSION_TRACKER.md` and `official_status.json` were refreshed by the required script. `registry.json` was unchanged.
- Official submission IDs this cycle: none.
- Live official submission would require credential/quota recovery because registered submission refreshes still report HTTP 401, so server quota could not be verified.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-19T04:47:47+00:00`; do not treat this cycle's `470` inventoried SHA or the deduped-new SHA `ab948a5f2c58f9475b33a410ef64e40afe3d06d144ece47608455a7f787b7279` as new.
- Latest scanned TTT steps at cutoff: `whest_0718_affinef64_ttt` `50`, `whest_0718_gaussiancv_ttt` `50`, `whest_0718_german013_ttt` `50`, `whest_0718_layerrescale_ttt` `50`, `whest_0718_rqmc030_ttt` `50`, `whest_0718_sphericalrb_ttt` `50`.
- Latest scanned AutoEvolve pool steps at cutoff: `whest_0718_affinef64_auto_r1` `8`, `whest_0718_affinef64_auto_r2` `8`, `whest_0718_gaussiancv_auto_r1` `10`, `whest_0718_gaussiancv_auto_r2` `15`, `whest_0718_german013_auto_r1` `8`, `whest_0718_german013_auto_r2` `3`, `whest_0718_layerrescale_auto_r1` `6`, `whest_0718_layerrescale_auto_r2` `11`, `whest_0718_rqmc030_auto_r1` `9`, `whest_0718_rqmc030_auto_r2` `9`, `whest_0718_sphericalrb_auto_r1` `6`, `whest_0718_sphericalrb_auto_r2` `6`.
- Include artifacts excluded after cutoff next round, starting with `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000008/call_0001_7223777d/submission.py`.

## Cycle 20260719T052427Z hourly submission review

- Completed at UTC: `2026-07-19T05:35:12+00:00`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- Cached official status counts after refresh: `{'graded': 24, 'failed': 1}`. Pending/queued/running registered submissions: `[]`. No historical pending status changes needed ledger backfill.
- Quota assessment for UTC `2026-07-19`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`. Blocker: `AIcrowdAuthError: While checking submission status: Your AIcrowd API key is missing or invalid. (HTTP 401)`.
- Candidate cutoff UTC: `2026-07-19T05:24:27+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-19T05:10:24.072776+00:00']`.
- JSONL context files were read for sampler/metrics/agent/gen-score line counts and agent-output provenance while live runs continued; live context mtime range UTC: `['2026-07-18T17:40:25.175395+00:00', '2026-07-19T05:10:24.072776+00:00']`.
- Console logs inspected: `18` files under `codex_runs/aicrowd_whestbench/multi_initial_20260718/console`; mtime range UTC `['2026-07-18T14:51:18.876898+00:00', '2026-07-19T01:16:53.661911+00:00']`. Recent tails contained timeout/connection reset/candidate failed/terminated/reconnect context only; no candidate was promoted from console text.
- No `candidate_pool.json`, candidate-pool directory, or extra candidates directory was found outside the scanned step/pool/workspace artifacts.
- Historical/known SHA baseline size: `689`; old inventory entries read: `6265`.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T052427Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T052427Z/scan_summary.json`
- Shortlist manifest: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T052427Z/shortlist_manifest.json`
- Shortlist sources directory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T052427Z/shortlist_sources`
- Full-100 reports: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T052427Z/full100_reports`
- Parse errors: `[]`.
- Files excluded after cutoff: `1`; example: `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000009/call_0001_776811f3/submission.py`.

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count | Excluded after cutoff |
|---|---|---|---:|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | - | 0-8 | 9 | 18 | 0 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | - | 0-8 | 9 | 18 | 0 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | - | 0-10 | 11 | 22 | 0 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | - | 0-15 | 16 | 32 | 0 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | - | 0-9 | 9 | 19 | 1 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | - | 0-3 | 4 | 8 | 0 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | - | 0-11 | 12 | 24 | 0 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | - | 0-9 | 10 | 20 | 0 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | - | 0-9 | 10 | 20 | 0 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | 0-50 | - | 0 | 51 | 0 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | - | 0-6 | 7 | 14 | 0 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | 0-50 | - | 0 | 51 | 0 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `472`.
- New SHA after ledger/registry/tracker/old-inventory/canonical-file baseline removal: `2`.
- Final decision counts: `{'reject_existing_sha': 470, 'reject_full100_no_submit': 1, 'reject_ranked_not_full100': 1}`.
- Entered full-100: `1`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Shortlist And Decisions

| SHA | Family / run / step | State | First-10 adjusted / MSE / C-B / fail | Full-100 adjusted / MSE / C-B / fail | Decision |
|---|---|---|---|---|---|
| `5df5ea3e9c16` | `german013` / `whest_0718_german013_auto_r1` / step `9` | `45807233-5dde-41ba-a95a-97ebd1bb7b9c` | 1.541598008e-07 / 1.492015554e-06 / 0.10331 / 0 | 3.585287501e-07 / 3.428981746e-06 / 0.10451 / 0 | `reject_full100_no_submit` |
| `ee24a78e02c7` | `german013` / `whest_0718_german013_auto_r1` / step `9` | `359b92aa-9739-4229-92d9-cf76d0355355` | 1.543270915e-07 / 1.492015554e-06 / 0.10348 / 0 | - / - / - / - | `reject_ranked_not_full100` |

Interpretation:

- `5df5ea3e9c16` source copy: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T052427Z/shortlist_sources/german013_candidate_5df5ea3e9c16.py`.
- `5df5ea3e9c16` full-100 report: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T052427Z/full100_reports/german013_setup_cached_5df5ea3e9c16_full100_20260719T052427Z.json`.
- Method change for `5df5ea3e9c16`: German013 radial/normal-blend estimator with setup-cached prepared main/extra spherical blocks for `_PRECOMPUTE_MLPS = 10`, whitening/cholesky solve, first-layer mean correction, normal final blend, and negative extra-block first-mean strength. Static code inspection found no file/network/reflection/eval subprocess behavior.
- Full-100 was clean: `0` failures, failure_breakdown all zero, max per-MLP effective compute `2.878072390e10`, below Phase 1 budget `2.72e11`.
- Despite strong first-10 versus initial (`63.21%`) and modest parent improvement (`1.78%`), full-100 adjusted score `3.585287501e-07` is weaker than the registered local full-100 best `2.748615e-07` and the best registered hosted public-50 score `2.787552892e-07` (#316192). It does not justify an official package or submit.
- `ee24a78e02c7` is the same German013 setup-cached radial/normal-blend cluster with slightly worse first-10; it was not run full-100 because `5df5ea3e9c16` was the stronger representative and failed the submission bar.

### Official Submission Status

- No `whest package` or `whest submit` was run.
- No new submission ID was obtained; registry was not changed after the required pre-review tracker refresh.
- `SUBMISSION_TRACKER.md` and `official_status.json` were refreshed by the required script. `registry.json` was unchanged.
- Official submission IDs this cycle: none.
- Live official submission would also require credential/quota recovery because registered submission refreshes still report HTTP 401, so server quota could not be verified.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-19T05:24:27+00:00`; do not treat this cycle's `472` inventoried SHA, raw new SHA `5df5ea3e9c168be74ddad7af9fde8377bbe81ed3f7a1ecfaf6f7f902c818291e`, or raw new SHA `ee24a78e02c78ce068de92196dd6068e9b558ae976b1a0755bcd4fab84d99496` as new.
- Latest scanned TTT steps at cutoff: `whest_0718_affinef64_ttt` `50`, `whest_0718_gaussiancv_ttt` `50`, `whest_0718_german013_ttt` `50`, `whest_0718_layerrescale_ttt` `50`, `whest_0718_rqmc030_ttt` `50`, `whest_0718_sphericalrb_ttt` `50`.
- Latest scanned AutoEvolve pool steps at cutoff: `whest_0718_affinef64_auto_r1` `8`, `whest_0718_affinef64_auto_r2` `8`, `whest_0718_gaussiancv_auto_r1` `10`, `whest_0718_gaussiancv_auto_r2` `15`, `whest_0718_german013_auto_r1` `9`, `whest_0718_german013_auto_r2` `3`, `whest_0718_layerrescale_auto_r1` `6`, `whest_0718_layerrescale_auto_r2` `11`, `whest_0718_rqmc030_auto_r1` `9`, `whest_0718_rqmc030_auto_r2` `9`, `whest_0718_sphericalrb_auto_r1` `6`, `whest_0718_sphericalrb_auto_r2` `6`.
- Include artifacts excluded after cutoff next round, starting with `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000009/call_0001_776811f3/submission.py`.

## Cycle 20260719T060834Z hourly submission review

- Completed at UTC: `2026-07-19T06:14:33+00:00`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- Cached official status counts after refresh: `{'failed': 1, 'graded': 24}`. Pending/queued/running registered submissions: `[]`. No historical pending status changes needed ledger backfill.
- Quota assessment for UTC `2026-07-19`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`. Blocker: `AIcrowdAuthError: While checking submission status: Your AIcrowd API key is missing or invalid. (HTTP 401)`.
- AIcrowd refresh/auth status: `AIcrowdAuthError HTTP 401 for registered submissions; server quota/status unavailable`.
- Candidate cutoff UTC: `2026-07-19T06:08:34+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-19T05:10:24.072776+00:00']`.
- JSONL context files were read for sampler/metrics/agent/gen-score line counts and agent-output provenance while live runs continued; live context mtime range UTC: `['2026-07-18T17:40:25.175395+00:00', '2026-07-19T05:10:24.072776+00:00']`.
- Console logs inspected: `18` files under `codex_runs/aicrowd_whestbench/multi_initial_20260718/console`; mtime range UTC `['2026-07-18T14:51:18.876898+00:00', '2026-07-19T01:16:53.661911+00:00']`. Recent tails contained connection reset, timeout grading, candidate failed, terminated, and TTT reconnect context only; no candidate was promoted from console text.
- No `candidate_pool.json`, candidate-pool directory, or extra `candidates` directory was found outside the scanned step/pool/workspace artifacts.
- Historical/known SHA baseline size: `691`; old inventory entries read: `6737`.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T060834Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T060834Z/scan_summary.json`
- Shortlist manifest: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T060834Z/shortlist_manifest.json`
- Shortlist sources directory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T060834Z/shortlist_sources`
- Full-100 reports: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T060834Z/full100_reports` (not created because no shortlist candidate)
- Parse errors: `[]`.
- Files excluded after cutoff: `1`; example: `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000009/call_0001_776811f3/submission.py`.

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count | Excluded after cutoff |
|---|---|---|---:|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | `-` | `0-8` | 9 | 18 | 0 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | `-` | `0-8` | 9 | 18 | 0 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | `0-50` | `-` | 0 | 51 | 0 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | `-` | `0-10` | 11 | 22 | 0 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | `-` | `0-15` | 16 | 32 | 0 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | `0-50` | `-` | 0 | 51 | 0 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | `-` | `0-9` | 9 | 19 | 1 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | `-` | `0-3` | 4 | 8 | 0 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | `0-50` | `-` | 0 | 51 | 0 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | `-` | `0-6` | 7 | 14 | 0 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | `-` | `0-11` | 12 | 24 | 0 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | `0-50` | `-` | 0 | 51 | 0 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | `-` | `0-9` | 10 | 20 | 0 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | `-` | `0-9` | 10 | 20 | 0 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | `0-50` | `-` | 0 | 51 | 0 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | `-` | `0-6` | 7 | 14 | 0 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | `-` | `0-6` | 7 | 14 | 0 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | `0-50` | `-` | 0 | 51 | 0 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `472`.
- New SHA after ledger/registry/tracker/old-inventory/canonical-file baseline removal: `0`.
- Final decision counts: `{'reject_existing_sha': 472}`.
- By-family unique SHA counts: `{'affinef64': 88, 'gaussiancv': 54, 'german013': 68, 'layerrescale': 45, 'rqmc030': 110, 'sphericalrb': 107}`.
- Entered full-100: `0`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Shortlist And Decisions

- No candidate entered the full-100 shortlist. Every scanned SHA was already present in the historical/known baseline, including the best first-10 rows below.

| Family | Run | SHA | Step | First-10 adjusted / MSE / C-B / fail | vs initial | Decision |
|---|---|---|---:|---|---:|---|
| `affinef64` | `whest_0718_affinef64_auto_r1` | `921b581662fe` | 7 | 1.275691847e-07 / 7.505604287e-07 / 0.1700 / 0 | 73.607% | `reject_existing_sha` |
| `affinef64` | `whest_0718_affinef64_auto_r1` | `07b20a3cb799` | 7 | 1.275939272e-07 / 7.505604287e-07 / 0.1700 / 0 | 73.573% | `reject_existing_sha` |
| `layerrescale` | `whest_0718_layerrescale_auto_r2` | `3f26d2bb11e5` | 5 | 1.357858217e-07 / 1.355103279e-06 / 0.1001 / 0 | 45.844% | `reject_existing_sha` |
| `layerrescale` | `whest_0718_layerrescale_auto_r2` | `fba838fa2471` | 5 | 1.357995213e-07 / 1.355103279e-06 / 0.1001 / 0 | 45.829% | `reject_existing_sha` |
| `layerrescale` | `whest_0718_layerrescale_auto_r2` | `e2991497f6b9` | 4 | 1.362059685e-07 / 1.359472594e-06 / 0.1000 / 0 | 45.394% | `reject_existing_sha` |
| `layerrescale` | `whest_0718_layerrescale_auto_r2` | `5748976c25ca` | 4 | 1.362415993e-07 / 1.359472594e-06 / 0.0998 / 0 | 45.356% | `reject_existing_sha` |
| `german013` | `whest_0718_german013_auto_r2` | `8c0a803abd3b` | 2 | 1.385124267e-07 / 1.385124267e-06 / 0.0936 / 0 | 81.650% | `reject_existing_sha` |
| `german013` | `whest_0718_german013_auto_r2` | `d1cc43eb710b` | 2 | 1.385124267e-07 / 1.385124267e-06 / 0.0937 / 0 | 81.650% | `reject_existing_sha` |

Interpretation:

- The apparently strong first-10 candidates in this cutoff, including AffineF64 `921b581662fe` / `07b20a3cb799`, LayerRescale `3f26d2bb11e5` / `fba838fa2471`, and German013 `8c0a803abd3b` / `d1cc43eb710b`, were all already inventoried in prior cycles and marked `reject_existing_sha` here.
- No new estimator SHA appeared after the previous reviewed baseline, and no method-distinct representative remained after deduplication.
- Because `new_sha_count = 0`, no local mini full-100 run was warranted; no package was built and no `whest submit` command was run.

### Official Submission Status

- No new submission ID was obtained; registry was not changed after the required pre-review tracker refresh.
- `SUBMISSION_TRACKER.md` and `official_status.json` were refreshed by the required script. `registry.json` was unchanged.
- Official submission IDs this cycle: none.
- Live official submission would also require credential/quota recovery because registered submission refreshes still report HTTP 401, so server quota could not be verified.
- Best registered hosted public-50 score remains `2.787552892e-07` from #316192; there was no new local full-100 score to compare this cycle.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-19T06:08:34+00:00`; do not treat this cycle's `472` inventoried SHA as new.
- Latest scanned TTT steps at cutoff: `whest_0718_affinef64_ttt` `50`, `whest_0718_gaussiancv_ttt` `50`, `whest_0718_german013_ttt` `50`, `whest_0718_layerrescale_ttt` `50`, `whest_0718_rqmc030_ttt` `50`, `whest_0718_sphericalrb_ttt` `50`.
- Latest scanned AutoEvolve pool steps at cutoff: `whest_0718_affinef64_auto_r1` `8`, `whest_0718_affinef64_auto_r2` `8`, `whest_0718_gaussiancv_auto_r1` `10`, `whest_0718_gaussiancv_auto_r2` `15`, `whest_0718_german013_auto_r1` `9`, `whest_0718_german013_auto_r2` `3`, `whest_0718_layerrescale_auto_r1` `6`, `whest_0718_layerrescale_auto_r2` `11`, `whest_0718_rqmc030_auto_r1` `9`, `whest_0718_rqmc030_auto_r2` `9`, `whest_0718_sphericalrb_auto_r1` `6`, `whest_0718_sphericalrb_auto_r2` `6`.
- Include artifacts excluded after cutoff next round, starting with `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000009/call_0001_776811f3/submission.py` (mtime was after this cycle cutoff).

## Cycle 20260719T064515Z hourly submission review

- Completed at UTC: `2026-07-19T06:57:52+00:00`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- Cached official status counts after refresh: `{'graded': 24, 'failed': 1}`. Pending/queued/running registered submissions: `[]`. No historical pending status changes needed ledger backfill.
- Quota assessment for UTC `2026-07-19`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`. Blocker: `AIcrowdAuthError: While checking submission status: Your AIcrowd API key is missing or invalid. (HTTP 401)`.
- Candidate cutoff UTC: `2026-07-19T06:45:15+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-19T06:14:48.639579+00:00']`.
- JSONL context files were read for sampler/metrics/agent/gen-score line counts and agent-output provenance while live runs continued; live context mtime range UTC: `['2026-07-18T17:40:25.175395+00:00', '2026-07-19T06:14:48.640579+00:00']`.
- Console logs inspected: `18` files under `codex_runs/aicrowd_whestbench/multi_initial_20260718/console`; mtime range UTC `['2026-07-18T14:51:18.876898+00:00', '2026-07-19T01:16:53.661911+00:00']`. Recent tails contained connection reset, timeout grading, candidate failed, terminated, and TTT reconnect context only; no candidate was promoted from console text.
- No `candidate_pool.json`, candidate-pool directory, or extra `candidates` directory was found outside the scanned step/pool/workspace artifacts.
- Historical/known SHA baseline size: `691`; old inventory entries read: `7209`.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T064515Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T064515Z/scan_summary.json`
- Shortlist manifest: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T064515Z/shortlist_manifest.json`
- Shortlist sources directory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T064515Z/shortlist_sources`
- Full-100 reports: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T064515Z/full100_reports`
- Parse errors: `[]`.
- Files excluded after cutoff: `1`; example: `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000010/call_0001_06d03d7e/submission.py`.

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count | Excluded after cutoff |
|---|---|---|---:|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | `-` | `0-8` | 9 | 18 | 0 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | `-` | `0-8` | 9 | 18 | 0 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | `0-50` | `-` | 0 | 51 | 0 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | `-` | `0-10` | 11 | 22 | 0 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | `-` | `0-15` | 16 | 32 | 0 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | `0-50` | `-` | 0 | 51 | 0 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | `-` | `0-10` | 10 | 21 | 1 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | `-` | `0-3` | 4 | 8 | 0 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | `0-50` | `-` | 0 | 51 | 0 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | `-` | `0-6` | 7 | 14 | 0 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | `-` | `0-11` | 12 | 24 | 0 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | `0-50` | `-` | 0 | 51 | 0 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | `-` | `0-9` | 10 | 20 | 0 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | `-` | `0-9` | 10 | 20 | 0 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | `0-50` | `-` | 0 | 51 | 0 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | `-` | `0-6` | 7 | 14 | 0 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | `-` | `0-6` | 7 | 14 | 0 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | `0-50` | `-` | 0 | 51 | 0 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `474`.
- New SHA after ledger/registry/tracker/old-inventory/canonical-file baseline removal: `2`.
- Final decision counts: `{'reject_existing_sha': 472, 'reject_full100_no_submit': 1, 'reject_ranked_not_full100': 1}`.
- By-family unique SHA counts: `{'affinef64': 88, 'gaussiancv': 54, 'german013': 70, 'layerrescale': 45, 'rqmc030': 110, 'sphericalrb': 107}`.
- Entered full-100: `1`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Shortlist And Decisions

| SHA | Family / run / step | State | First-10 adjusted / MSE / C-B / fail | Full-100 adjusted / MSE / C-B / fail | Decision |
|---|---|---|---|---|---|
| `9659c464922e` | `german013` / `whest_0718_german013_auto_r1` / step `10` | `b6e301a9-3df4-4460-8359-dd44d7913eeb` | 1.538781295e-07 / 1.490120042e-06 / 0.10326 / 0 | 3.453257775e-07 / 3.305573873e-06 / 0.10443 / 0 | `reject_full100_no_submit` |
| `c34fccc121ae` | `german013` / `whest_0718_german013_auto_r1` / step `10` | `96030af4-9dcc-4195-9cab-a05e27f575da` | 1.540836213e-07 / 1.490120042e-06 / 0.10345 / 0 | - / - / - / - | `reject_ranked_not_full100` |

Interpretation:

- `9659c464922e` source copy: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T064515Z/shortlist_sources/german013_candidate_9659c464922e.py`.
- `9659c464922e` full-100 report: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T064515Z/full100_reports/german013_radial_normal_blend_9659c464922e_full100_20260719T064515Z.json`.
- Method change for `9659c464922e`: German013 setup-cached radial/normal-blend estimator with prepared main/extra spherical blocks for `_PRECOMPUTE_MLPS = 10`, whitening/cholesky solve, first-layer mean correction, normal final blend, `_FIRST_MEAN_STRENGTH = 0.45`, `_EXTRA_FIRST_MEAN_STRENGTH = -0.35`, and `_EXTRA_FINAL_NORMAL_BLEND = 0.10`. Static code inspection found no file/network/reflection/subprocess behavior.
- First-10 was clean and strong versus initial (`63.51%` by recorded ratio), but only `0.183%` better than parent `5df5ea3e9c16`.
- Full-100 was clean: `0` failures, failure_breakdown all zero, max per-MLP effective compute `2.872559668e10`, below Phase 1 budget `2.72e11`.
- Despite improving the prior same-family full-100 representative (`5df5ea3e9c16` at `3.585287501e-07`), full-100 adjusted score `3.453257775e-07` is still weaker than the registered local full-100 best `2.748615e-07` and the best registered hosted public-50 score `2.787552892e-07` (#316192). It does not justify an official package or submit.
- `c34fccc121ae` is the same German013 setup-cached radial/normal-blend cluster with slightly worse first-10 and only `0.049%` parent improvement; it was not run full-100 because `9659c464922e` was the stronger representative and failed the submission bar.

### Official Submission Status

- No `whest package` or `whest submit` was run.
- No new submission ID was obtained; registry was not changed after the required pre-review tracker refresh.
- `SUBMISSION_TRACKER.md` and `official_status.json` were refreshed by the required script. `registry.json` was unchanged.
- Official submission IDs this cycle: none.
- Live official submission would also require credential/quota recovery because registered submission refreshes still report HTTP 401, so server quota could not be verified.
- Best registered hosted public-50 score remains `2.787552892e-07` from #316192.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-19T06:45:15+00:00`; do not treat this cycle's `474` inventoried SHA, raw new SHA `9659c464922ed658d65be3f88571da3ac1fa4004abfc8a505c64633b878c6f71`, or raw new SHA `c34fccc121aea1f288307030ecbd9843606d1d6862d58b92d8dcb37108dbc80e` as new.
- Latest scanned TTT steps at cutoff: `whest_0718_affinef64_ttt` `50`, `whest_0718_gaussiancv_ttt` `50`, `whest_0718_german013_ttt` `50`, `whest_0718_layerrescale_ttt` `50`, `whest_0718_rqmc030_ttt` `50`, `whest_0718_sphericalrb_ttt` `50`.
- Latest scanned AutoEvolve pool steps at cutoff: `whest_0718_affinef64_auto_r1` `8`, `whest_0718_affinef64_auto_r2` `8`, `whest_0718_gaussiancv_auto_r1` `10`, `whest_0718_gaussiancv_auto_r2` `15`, `whest_0718_german013_auto_r1` `10`, `whest_0718_german013_auto_r2` `3`, `whest_0718_layerrescale_auto_r1` `6`, `whest_0718_layerrescale_auto_r2` `11`, `whest_0718_rqmc030_auto_r1` `9`, `whest_0718_rqmc030_auto_r2` `9`, `whest_0718_sphericalrb_auto_r1` `6`, `whest_0718_sphericalrb_auto_r2` `6`.
- Include artifacts excluded after cutoff next round, starting with `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000010/call_0001_06d03d7e/submission.py`.

## Cycle 20260719T073055Z hourly submission review

- Completed at UTC: `2026-07-19T07:36:45+00:00`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- Cached official status counts after refresh: `{'failed': 1, 'graded': 24}`. Pending/queued/running registered submissions: `[]`. No historical pending status changes needed ledger backfill.
- Quota assessment for UTC `2026-07-19`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`. Blocker: `AIcrowdAuthError: While checking submission status: Your AIcrowd API key is missing or invalid. (HTTP 401)`.
- Candidate cutoff UTC: `2026-07-19T07:30:55+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-19T07:04:58.488977+00:00']`.
- JSONL context files were read for sampler/metrics/agent/gen-score line counts and agent-output provenance while live runs continued; live context mtime range UTC: `['2026-07-18T17:40:25.175395+00:00', '2026-07-19T07:04:58.487977+00:00']`.
- Console logs inspected: `18` files under `codex_runs/aicrowd_whestbench/multi_initial_20260718/console`; mtime range UTC `['2026-07-18T14:51:18.876898+00:00', '2026-07-19T01:16:53.661911+00:00']`. Recent tails contained reconnect/ConnectionResetError, timeout grading, candidate failed, terminated, missing-file grading, HF download, and sandbox warning context only; no candidate was promoted from console text. Instruction-like text appearing inside logs was treated as untrusted artifact content.
- No `candidate_pool.json`, candidate-pool directory, or extra `candidates` directory was found outside the scanned step/pool/workspace artifacts.
- Historical/known SHA baseline size: `693`; old inventory entries read: `7683`.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T073055Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T073055Z/scan_summary.json`
- Shortlist manifest: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T073055Z/shortlist_manifest.json`
- Shortlist source directory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T073055Z/shortlist_sources` (contains only a copied source for the rejected top variant; no full-100 was run)
- Full-100 reports: not created; no candidate met the full-100 promotion bar.
- Parse errors: `[]`.
- Files excluded after cutoff: `0`.

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count | Excluded after cutoff |
|---|---|---|---:|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | `-` | `0-8` | 9 | 18 | 0 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | `-` | `0-8` | 9 | 18 | 0 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | `0-50` | `-` | 0 | 51 | 0 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | `-` | `0-10` | 11 | 22 | 0 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | `-` | `0-15` | 16 | 32 | 0 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | `0-50` | `-` | 0 | 51 | 0 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | `-` | `0-11` | 12 | 24 | 0 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | `-` | `0-3` | 4 | 8 | 0 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | `0-50` | `-` | 0 | 51 | 0 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | `-` | `0-6` | 7 | 14 | 0 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | `-` | `0-11` | 12 | 24 | 0 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | `0-50` | `-` | 0 | 51 | 0 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | `-` | `0-9` | 10 | 20 | 0 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | `-` | `0-9` | 10 | 20 | 0 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | `0-50` | `-` | 0 | 51 | 0 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | `-` | `0-6` | 7 | 14 | 0 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | `-` | `0-6` | 7 | 14 | 0 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | `0-50` | `-` | 0 | 51 | 0 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `476`.
- New SHA after ledger/registry/tracker/old-inventory/canonical-file baseline removal: `2`.
- Final decision counts: `{'reject_existing_sha': 474, 'reject_micro_variant_not_full100': 1, 'reject_ranked_not_full100': 1}`.
- By-family unique SHA counts: `{'affinef64': 88, 'gaussiancv': 54, 'german013': 72, 'layerrescale': 45, 'rqmc030': 110, 'sphericalrb': 107}`.
- Entered full-100: `0`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Shortlist And Decisions

| SHA | Family / run / step | State | First-10 adjusted / MSE / C-B / fail | vs initial | vs parent | Decision | Method change |
|---|---|---|---|---:|---:|---|---|
| `475a66837dfa` | `german013` / `whest_0718_german013_auto_r1` / step `11` | `c5452bc4-5193-4953-9c38-ee5acc2a2317` | 1.540535665e-07 / 1.490671320e-06 / 1.033361168e-01 / 0 | 63.325% | -0.114% | `reject_micro_variant_not_full100` | German013 setup-cached radial/normal-blend parameter variant; constants tune first-mean strength, extra first-mean strength, and final normal blend. |
| `3c1eab4b13d1` | `german013` / `whest_0718_german013_auto_r1` / step `11` | `d5ed6c37-5b89-4ed3-bce5-86ae9f959639` | 1.542388515e-07 / 1.490671320e-06 / 1.035137620e-01 / 0 | 63.129% | -0.234% | `reject_ranked_not_full100` | German013 setup-cached radial/normal-blend parameter variant; constants tune first-mean strength, extra first-mean strength, and final normal blend. |

Interpretation:

- `475a66837dfa` source copy: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T073055Z/shortlist_sources/german013_candidate_475a66837dfa.py`.
- `475a66837dfa` and `3c1eab4b13d1` are parameter-only variants of parent `9659c464922e`, which was full-100 validated in cycle `20260719T064515Z` and rejected at adjusted `3.453257775e-07`.
- The best new variant has first-10 adjusted `1.540535665e-07`, worse than parent first-10 `1.538781295e-07`; it changes only constants (`_FIRST_MEAN_STRENGTH`, `_EXTRA_FIRST_MEAN_STRENGTH`, `_MAIN_FINAL_NORMAL_BLEND`) and does not add a new mechanism.
- Because there was no significant improvement and no substantive method change, no local mini full-100 command was run, no package was built, and no official submission was attempted.

### Best Existing Rows Re-Seen

| Family | Run | SHA | Step | First-10 adjusted | Decision |
|---|---|---|---:|---:|---|
| `affinef64` | `whest_0718_affinef64_auto_r1` | `921b581662fe` | `7` | 1.275691847e-07 | `reject_existing_sha` |
| `affinef64` | `whest_0718_affinef64_auto_r1` | `07b20a3cb799` | `7` | 1.275939272e-07 | `reject_existing_sha` |
| `layerrescale` | `whest_0718_layerrescale_auto_r2` | `3f26d2bb11e5` | `5` | 1.357858217e-07 | `reject_existing_sha` |
| `layerrescale` | `whest_0718_layerrescale_auto_r2` | `fba838fa2471` | `5` | 1.357995213e-07 | `reject_existing_sha` |
| `layerrescale` | `whest_0718_layerrescale_auto_r2` | `e2991497f6b9` | `4` | 1.362059685e-07 | `reject_existing_sha` |
| `layerrescale` | `whest_0718_layerrescale_auto_r2` | `5748976c25ca` | `4` | 1.362415993e-07 | `reject_existing_sha` |
| `german013` | `whest_0718_german013_auto_r2` | `8c0a803abd3b` | `2` | 1.385124267e-07 | `reject_existing_sha` |
| `german013` | `whest_0718_german013_auto_r2` | `d1cc43eb710b` | `2` | 1.385124267e-07 | `reject_existing_sha` |

### Official Submission Status

- No new submission ID was obtained; registry was not changed after the required pre-review tracker refresh.
- `SUBMISSION_TRACKER.md` and `official_status.json` were refreshed by the required script. `registry.json` was unchanged.
- Official submission IDs this cycle: none.
- Live official submission would require credential/quota recovery because registered submission refreshes still report HTTP 401, so server quota could not be verified.
- Best registered hosted public-50 score remains `2.787552892e-07` from #316192; best registered local full-100 remains `2.748615e-07`.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-19T07:30:55+00:00`; do not treat this cycle's `476` inventoried SHA, raw new SHA `475a66837dfad97ff03bf6c5eb08759c0eb810df53723d4545614c976579ec31`, or raw new SHA `3c1eab4b13d14f48618cf81ebb1db58d0a6de4a2dbe28f8d8cd6bcfb795a86f8` as new.
- Latest scanned TTT steps at cutoff: `whest_0718_affinef64_ttt` `50`, `whest_0718_gaussiancv_ttt` `50`, `whest_0718_german013_ttt` `50`, `whest_0718_layerrescale_ttt` `50`, `whest_0718_rqmc030_ttt` `50`, `whest_0718_sphericalrb_ttt` `50`.
- Latest scanned AutoEvolve pool steps at cutoff: `whest_0718_affinef64_auto_r1` `8`, `whest_0718_affinef64_auto_r2` `8`, `whest_0718_gaussiancv_auto_r1` `10`, `whest_0718_gaussiancv_auto_r2` `15`, `whest_0718_german013_auto_r1` `11`, `whest_0718_german013_auto_r2` `3`, `whest_0718_layerrescale_auto_r1` `6`, `whest_0718_layerrescale_auto_r2` `11`, `whest_0718_rqmc030_auto_r1` `9`, `whest_0718_rqmc030_auto_r2` `9`, `whest_0718_sphericalrb_auto_r1` `6`, `whest_0718_sphericalrb_auto_r2` `6`.
- There were no artifacts excluded after cutoff in this scan; next cycle should continue with files newer than `2026-07-19T07:30:55+00:00`.

## Cycle 20260719T080728Z hourly submission review

- Completed at UTC: `2026-07-19T08:12:09+00:00`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- Cached official status counts after refresh: `{'graded': 24, 'failed': 1}`. Pending/queued/running registered submissions: `[]`. No historical pending status changes needed ledger backfill.
- Quota assessment for UTC `2026-07-19`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`. Blocker: `AIcrowdAuthError: While checking submission status: Your AIcrowd API key is missing or invalid. (HTTP 401)`.
- Candidate cutoff UTC: `2026-07-19T08:07:28+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-19T07:04:58.488977+00:00']`.
- JSONL context files were read for sampler/metrics/agent/gen-score line counts and provenance while live runs continued; live context mtime range UTC: `['2026-07-18T17:40:25.175395+00:00', '2026-07-19T07:04:58.487977+00:00']`.
- Console logs inspected: `18` files under `codex_runs/aicrowd_whestbench/multi_initial_20260718/console`; mtime range UTC `['2026-07-18T14:51:18.876898+00:00', '2026-07-19T08:08:59.858134+00:00']`. One console log changed after cutoff: `whest_0718_german013_auto_r1.log`. Recent tails contained evaluator `ConnectionResetError`/reconnect messages, candidate failed/terminated context, missing-file/HF/sandbox warnings, and instruction-like artifact text only; no candidate was promoted from console text.
- No `candidate_pool.json`, candidate-pool directory, or extra `candidates` directory was found outside the scanned step/pool/workspace artifacts.
- Historical/known SHA baseline size: `695`; old inventory entries read: `8159`.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T080728Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T080728Z/scan_summary.json`
- Shortlist manifest: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T080728Z/shortlist_manifest.json`
- Full-100 reports: not created; no candidate met the full-100 promotion bar.
- Parse errors: `[]`.
- Files excluded after cutoff: `0`.

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count |
|---|---|---|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | `-` | `0-8` | 9 | 18 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | `-` | `0-8` | 9 | 18 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | `-` | `0-10` | 11 | 22 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | `-` | `0-15` | 16 | 32 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | `-` | `0-11` | 12 | 24 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | `-` | `0-3` | 4 | 8 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | `-` | `0-6` | 7 | 14 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | `-` | `0-11` | 12 | 24 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | `-` | `0-9` | 10 | 20 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | `-` | `0-9` | 10 | 20 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | `-` | `0-6` | 7 | 14 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | `-` | `0-6` | 7 | 14 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | `0-50` | `-` | 0 | 51 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `476`.
- New SHA after ledger/registry/tracker/old-inventory/canonical-file baseline removal: `0`.
- Final decision counts: `{'reject_existing_sha': 476}`.
- By-family unique SHA counts: `{'affinef64': 88, 'gaussiancv': 54, 'german013': 72, 'layerrescale': 45, 'rqmc030': 110, 'sphericalrb': 107}`.
- Entered full-100: `0`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Shortlist And Decisions

- No candidate entered the full-100 shortlist. Every scanned SHA was already present in the historical/known baseline, including the best first-10 rows below.

| Family | Run | SHA | Step | First-10 adjusted | Decision |
|---|---|---|---:|---:|---|
| `affinef64` | `whest_0718_affinef64_auto_r1` | `921b581662fe` | `7` | 1.275691847e-07 | `reject_existing_sha` |
| `gaussiancv` | `whest_0718_gaussiancv_auto_r1` | `a0f5bc4922b1` | `4` | 1.467102880e-07 | `reject_existing_sha` |
| `german013` | `whest_0718_german013_auto_r2` | `8c0a803abd3b` | `2` | 1.385124267e-07 | `reject_existing_sha` |
| `layerrescale` | `whest_0718_layerrescale_auto_r2` | `3f26d2bb11e5` | `5` | 1.357858217e-07 | `reject_existing_sha` |
| `rqmc030` | `whest_0718_rqmc030_auto_r1` | `7165e74c9196` | `8` | 1.675358334e-07 | `reject_existing_sha` |
| `sphericalrb` | `whest_0718_sphericalrb_auto_r1` | `3c73a750aa29` | `5` | 1.461057215e-07 | `reject_existing_sha` |

Interpretation:

- The latest candidate artifact observed before this cycle cutoff was still `whest_0718_german013_auto_r1/autoevolve_workspaces/step_000011/call_0001_871e3ff4/submission.py` and `autoevolve_pool_step_000011.json`, both from `2026-07-19T07:04:58Z`.
- No new estimator SHA appeared after the prior `20260719T073055Z` cycle; the two SHA that were new then (`475a66837dfa...`, `3c1eab4b13d1...`) are now part of the old-inventory baseline.
- Because there were no deduped-new candidates, no local mini full-100 command was run, no package was built, and no official submission was attempted.

### Official Submission Status

- No new submission ID was obtained; registry was not changed after the required pre-review tracker refresh.
- `SUBMISSION_TRACKER.md` and `official_status.json` were refreshed by the required script. `registry.json` was unchanged.
- Official submission IDs this cycle: none.
- Live official submission would require credential/quota recovery because registered submission refreshes still report HTTP 401, so server quota could not be verified. This was not the gating factor for this cycle because no candidate passed dedupe/shortlist review.
- Best registered hosted public-50 score remains `2.787552892e-07` from #316192; best registered local full-100 remains `2.748615e-07`.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-19T08:07:28+00:00`; do not treat this cycle's `476` inventoried SHA as new.
- Latest scanned TTT steps at cutoff: `whest_0718_affinef64_ttt` `50`, `whest_0718_gaussiancv_ttt` `50`, `whest_0718_german013_ttt` `50`, `whest_0718_layerrescale_ttt` `50`, `whest_0718_rqmc030_ttt` `50`, `whest_0718_sphericalrb_ttt` `50`.
- Latest scanned AutoEvolve pool steps at cutoff: `whest_0718_affinef64_auto_r1` `8`, `whest_0718_affinef64_auto_r2` `8`, `whest_0718_gaussiancv_auto_r1` `10`, `whest_0718_gaussiancv_auto_r2` `15`, `whest_0718_german013_auto_r1` `11`, `whest_0718_german013_auto_r2` `3`, `whest_0718_layerrescale_auto_r1` `6`, `whest_0718_layerrescale_auto_r2` `11`, `whest_0718_rqmc030_auto_r1` `9`, `whest_0718_rqmc030_auto_r2` `9`, `whest_0718_sphericalrb_auto_r1` `6`, `whest_0718_sphericalrb_auto_r2` `6`.
- There were no candidate artifacts excluded after cutoff in this scan. The only post-cutoff file seen was the live console log `whest_0718_german013_auto_r1.log`, not a candidate artifact; next cycle should continue with candidate files newer than `2026-07-19T08:07:28+00:00`.

## Cycle 20260719T084503Z hourly submission review

- Completed at UTC: `2026-07-19T08:54:58Z`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- Cached official status counts after refresh: `{'graded': 24, 'failed': 1}`. Pending/queued/running registered submissions: `[]`. No historical pending status changes needed ledger backfill.
- Quota assessment for UTC `2026-07-19`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`. Blocker: `AIcrowdAuthError: While checking submission status: Your AIcrowd API key is missing or invalid. (HTTP 401)`.
- Candidate cutoff UTC: `2026-07-19T08:45:03+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-19T08:22:34.862741+00:00']`.
- JSONL context files were read for sampler/metrics/agent/gen-score line counts and provenance while live runs continued; live context mtime range UTC: `['2026-07-18T17:40:25.175395+00:00', '2026-07-19T08:22:34.862741+00:00']`.
- Console logs inspected: `18` files under `codex_runs/aicrowd_whestbench/multi_initial_20260718/console`; mtime range UTC `['2026-07-18T14:51:18.876898+00:00', '2026-07-19T08:08:59.858134+00:00']`. Recent tails contained evaluator reconnect/timeout/candidate-failure/missing-file/HF/sandbox/instruction-like artifact context only; no candidate was promoted from console text.
- No `candidate_pool.json`, candidate-pool directory, or extra candidate directory was found outside scanned step/pool/workspace artifacts.
- Files excluded after cutoff: `1`, namely `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000012/call_0001_352a8aaa/submission.py`.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T084503Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T084503Z/scan_summary.json`
- Shortlist manifest: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T084503Z/shortlist_manifest.json`
- Shortlist source directory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T084503Z/shortlist_sources`
- Full-100 reports: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T084503Z/full100_reports`
- Parse errors: `[]`.

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count |
|---|---|---|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | `-` | `0-8` | 9 | 18 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | `-` | `0-8` | 9 | 18 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | `-` | `0-10` | 11 | 22 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | `-` | `0-15` | 16 | 32 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | `-` | `0-12` | 12 | 25 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | `-` | `0-3` | 4 | 8 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | `-` | `0-6` | 7 | 14 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | `-` | `0-11` | 12 | 24 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | `-` | `0-9` | 10 | 20 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | `-` | `0-9` | 10 | 20 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | `-` | `0-6` | 7 | 14 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | `-` | `0-6` | 7 | 14 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | `0-50` | `-` | 0 | 51 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `478`.
- Historical/known SHA baseline size: `695`; old inventory entries read: `8635`.
- New SHA after ledger/registry/tracker/old-inventory/canonical-file baseline removal: `2`.
- Final decision counts: `{'reject_existing_sha': 476, 'reject_full100_no_submit': 1, 'reject_ranked_not_full100': 1}`.
- By-family unique SHA counts: `{'affinef64': 88, 'gaussiancv': 54, 'german013': 74, 'layerrescale': 45, 'rqmc030': 110, 'sphericalrb': 107}`.
- Entered full-100: `1`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Shortlist And Decisions

| SHA | Family / run / step | State | First-10 adjusted / MSE / C-B / fail | Full-100 adjusted / MSE / C-B / fail | Decision |
|---|---|---|---|---|---|
| `5420b8ab629c` | `german013` / `whest_0718_german013_auto_r1` / step `12` | `ed163de4-7337-44e4-802e-086909d2a128` | 1.531891386e-07 / 1.490065790e-06 / 0.102807 / 0 | 3.394006937e-07 / 3.301972599e-06 / 0.102787 / 0 | `reject_full100_no_submit` |
| `74ef909657b9` | `german013` / `whest_0718_german013_auto_r1` / step `12` | `f064389b-63cd-4489-a3ba-c0231e937240` | 1.531945809e-07 / 1.490065790e-06 / 0.102814 / 0 | - / - / - / - | `reject_ranked_not_full100` |

Interpretation:

- `5420b8ab629c` source copy: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T084503Z/shortlist_sources/german013_candidate_5420b8ab629c.py`.
- `5420b8ab629c` full-100 report: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T084503Z/full100_reports/german013_preprop_cached_5420b8ab629c_full100_20260719T084503Z.json`.
- Method change for `5420b8ab629c`: German013 setup-cached radial/normal-blend estimator that imports `whestbench.generation.sample_mlp`, pre-generates the first `_PRECOMPUTE_MLPS = 10` MLPs in `setup`, prepropagates main/extra antithetic radial blocks through `_PREPROP_LAYERS = 14`, then finishes prediction from cached activations for matching seeds; fallback path uses the same radial/normal blend without prepropagation.
- Static code inspection found no file/network/reflection/subprocess behavior. The only flagged import was normal `flopscope.numpy`.
- First-10 was clean and strong versus the German013 initial (`64.25%` by recorded ratio), but only `0.45%` better than parent `9659c464922e`.
- Full-100 was clean: `0` failures, failure_breakdown all zero, max per-MLP effective compute `2.796949723e10`, below Phase 1 budget `2.72e11`.
- Full-100 adjusted `3.394006937e-07` improves the prior same-family parent `9659c464922e` full-100 `3.453257775e-07` by about `1.7%`, but remains weaker than registered local full-100 best `2.748615e-07` and registered hosted public-50 best `2.787552892e-07` (#316192). No official validation was justified.
- `74ef909657b9` was the same step-12 mechanism with slightly worse first-10; it was not run full-100 because the stronger representative failed the submission bar.

### Official Submission Status

- No `whest package` or `whest submit` was run.
- No new submission ID was obtained; `registry.json` was not changed after the required pre-review tracker refresh.
- `SUBMISSION_TRACKER.md` and `official_status.json` were refreshed by the required script.
- Official submission IDs this cycle: none.
- Live official submission would require credential/quota recovery because registered submission refreshes still report HTTP 401; server quota could not be verified. This was not the gating factor because the only full-100 candidate did not clear the quality bar.
- Best registered hosted public-50 score remains `2.787552892e-07` from #316192; best registered local full-100 remains `2.748615e-07`.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-19T08:45:03+00:00`; do not treat this cycle's `478` inventoried SHA, raw new SHA `5420b8ab629cb6e6fbf77bc04bdbc689b39d505471228ab2d07d79e36e2b9f89`, or raw new SHA `74ef909657b930cf04a7aeabb76202ea43ccb3d209f375bad4b8b2f6619d81e2` as new.
- Latest scanned TTT steps at cutoff: `whest_0718_affinef64_ttt` `50`, `whest_0718_gaussiancv_ttt` `50`, `whest_0718_german013_ttt` `50`, `whest_0718_layerrescale_ttt` `50`, `whest_0718_rqmc030_ttt` `50`, `whest_0718_sphericalrb_ttt` `50`.
- Latest scanned AutoEvolve pool steps at cutoff: `whest_0718_affinef64_auto_r1` `8`, `whest_0718_affinef64_auto_r2` `8`, `whest_0718_gaussiancv_auto_r1` `10`, `whest_0718_gaussiancv_auto_r2` `15`, `whest_0718_german013_auto_r1` `12`, `whest_0718_german013_auto_r2` `3`, `whest_0718_layerrescale_auto_r1` `6`, `whest_0718_layerrescale_auto_r2` `11`, `whest_0718_rqmc030_auto_r1` `9`, `whest_0718_rqmc030_auto_r2` `9`, `whest_0718_sphericalrb_auto_r1` `6`, `whest_0718_sphericalrb_auto_r2` `6`.
- Include artifacts excluded after cutoff next round, starting with `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000012/call_0001_352a8aaa/submission.py`.

## Cycle 20260719T092813Z hourly submission review

- Completed at UTC: `2026-07-19T09:37:54Z`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- Cached official status counts after refresh: `{'failed': 1, 'graded': 24}`. Pending/queued/running registered submissions: `[]`. No historical pending status changes needed ledger backfill.
- Quota assessment for UTC `2026-07-19`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`. Blocker: `AIcrowdAuthError: While checking submission status: Your AIcrowd API key is missing or invalid. (HTTP 401)`.
- Candidate cutoff UTC: `2026-07-19T09:28:13+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-19T09:19:20.152432+00:00']`.
- JSONL context files were read for sampler/metrics/agent/gen-score line counts and provenance while live runs continued; live context mtime range UTC: `['2026-07-18T17:40:25.175395+00:00', '2026-07-19T09:19:20.152432+00:00']`.
- Console logs inspected: `18` files under `codex_runs/aicrowd_whestbench/multi_initial_20260718/console`; mtime range UTC `['2026-07-18T14:51:18.876898+00:00', '2026-07-19T08:08:59.858134+00:00']`; tail flags `{'candidate_failed': 11, 'connection_reset_or_reconnect': 16, 'missing_file_or_hf': 3, 'sandbox_or_instruction_context': 4, 'timeout_or_terminated': 11}`. No estimator was promoted from console text.
- No `candidate_pool.json`, candidate-pool directory, or extra candidate directory was found outside scanned step/pool/workspace artifacts.
- Files excluded after cutoff: `1`; examples: `['codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000013/call_0001_d2dcc950/submission.py']`.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T092813Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T092813Z/scan_summary.json`
- Shortlist manifest: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T092813Z/shortlist_manifest.json`
- Shortlist source directory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T092813Z/shortlist_sources`
- Full-100 reports: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T092813Z/full100_reports`
- Parse errors: `[]`.

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count |
|---|---|---|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | `-` | `0-8` | 9 | 18 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | `-` | `0-8` | 9 | 18 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | `-` | `0-10` | 11 | 22 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | `-` | `0-15` | 16 | 32 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | `-` | `0-13` | 13 | 27 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | `-` | `0-3` | 4 | 8 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | `-` | `0-6` | 7 | 14 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | `-` | `0-11` | 12 | 24 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | `-` | `0-9` | 10 | 20 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | `-` | `0-9` | 10 | 20 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | `-` | `0-6` | 7 | 14 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | `-` | `0-6` | 7 | 14 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | `0-50` | `-` | 0 | 51 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `480`.
- Historical/known SHA baseline size: `697`; old inventory entries read: `9113`.
- New SHA after ledger/registry/tracker/old-inventory/canonical-file baseline removal: `2`.
- Final decision counts: `{'reject_full100_no_submit': 1, 'reject_existing_sha': 478, 'reject_ranked_not_full100': 1}`.
- By-family unique SHA counts: `{'affinef64': 88, 'gaussiancv': 54, 'german013': 76, 'layerrescale': 45, 'rqmc030': 110, 'sphericalrb': 107}`.
- Entered full-100: `1`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Shortlist And Decisions

| SHA | Family / run / step | State | First-10 adjusted / MSE / C-B / fail | Full-100 adjusted / MSE / C-B / fail | Decision |
|---|---|---|---|---|---|
| `22f41796fd4c` | `german013` / `whest_0718_german013_auto_r1` / step `13` | `80f07afd-da26-4347-881f-aa3596422739` | 1.531242058e-07 / 1.489793090e-06 / 1.027823982e-01 / 0 | 3.406469136e-07 / 3.313560857e-06 / 1.028041326e-01 / 0 | `reject_full100_no_submit` |
| `3d75151b1fa0` | `german013` / `whest_0718_german013_auto_r1` / step `13` | `538d0694-6e16-42c0-b6e5-b9d6cd954000` | 1.531401873e-07 / 1.489793090e-06 / 1.027913682e-01 / 0 | - / - / - / - | `reject_ranked_not_full100` |

Interpretation:

- `22f41796fd4c` source copy: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T092813Z/shortlist_sources/german013_candidate_22f41796fd4c.py`.
- `22f41796fd4c` full-100 report: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T092813Z/full100_reports/german013_pair_preprop_22f41796fd4c_full100_20260719T092813Z.json`.
- Method change: German013 setup-cached radial/normal-blend estimator that imports `whestbench.generation.sample_mlp`, pre-generates the first `_PRECOMPUTE_MLPS = 10` MLPs in `setup`, prepropagates both main and extra radial blocks together through `_PREPROP_LAYERS = 18`, then finishes prediction from cached activations for matching seeds. Compared with the previous step-12 representative, it increases preprop depth from 14 to 18, estimates main/extra from a combined final propagation, caches `radial_factor`, and raises `_EXTRA_WEIGHT_SCALE` from 1.22 to 1.30.
- Static inspection found no file/network/reflection/subprocess behavior. Flagged imports were normal `flopscope.numpy`, `whestbench.BaseEstimator`, and `whestbench.generation.sample_mlp`.
- First-10 was clean and strong versus the German013 initial (`64.317%` by recorded ratio), but only `0.042%` better than its parent/source comparison.
- Full-100 was clean: `0` failures, failure_breakdown `{'budget_exhausted': 0, 'combined_budget_exhausted': 0, 'error': 0, 'residual_wall_time_exhausted': 0, 'time_exhausted': 0}`, max per-MLP effective compute `2.797091508e+10`, below Phase 1 budget `2.72e11`.
- Full-100 adjusted `3.406469136e-07` is worse than the prior same-family step-12 representative full-100 `3.394006937e-07`, registered local full-100 best `2.748615e-07`, and registered hosted public-50 best `2.787552892e-07` (#316192). No official validation was justified.
- `3d75151b1fa0` was the same step-13 mechanism with slightly worse first-10; it was not run full-100 because the stronger representative failed the submission bar.

### Official Submission Status

- No `whest package` or `whest submit` was run.
- No new submission ID was obtained; `registry.json` was not changed after the required pre-review tracker refresh.
- `SUBMISSION_TRACKER.md` and `official_status.json` were refreshed by the required script.
- Official submission IDs this cycle: none.
- Live official submission would require credential/quota recovery because registered submission refreshes still report HTTP 401; server quota could not be verified. This was not the gating factor because the only full-100 candidate did not clear the quality bar.
- Best registered hosted public-50 score remains `2.787552892e-07` from #316192; best registered local full-100 remains `2.748615e-07`.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-19T09:28:13+00:00`; do not treat this cycle's `480` inventoried SHA, raw new SHA `22f41796fd4c9b3fbf4a80ca9498a7b7b1159c6f1dfadd9a08f555e3f0db1b8e`, or raw new SHA `3d75151b1fa06ea00515ed01a5c7001d65868160efcf0c177625d2c51e8daf43` as new.
- Latest scanned TTT steps at cutoff: `whest_0718_affinef64_ttt` `50`, `whest_0718_gaussiancv_ttt` `50`, `whest_0718_german013_ttt` `50`, `whest_0718_layerrescale_ttt` `50`, `whest_0718_rqmc030_ttt` `50`, `whest_0718_sphericalrb_ttt` `50`.
- Latest scanned AutoEvolve pool steps at cutoff: `whest_0718_affinef64_auto_r1` `8`, `whest_0718_affinef64_auto_r2` `8`, `whest_0718_gaussiancv_auto_r1` `10`, `whest_0718_gaussiancv_auto_r2` `15`, `whest_0718_german013_auto_r1` `13`, `whest_0718_german013_auto_r2` `3`, `whest_0718_layerrescale_auto_r1` `6`, `whest_0718_layerrescale_auto_r2` `11`, `whest_0718_rqmc030_auto_r1` `9`, `whest_0718_rqmc030_auto_r2` `9`, `whest_0718_sphericalrb_auto_r1` `6`, `whest_0718_sphericalrb_auto_r2` `6`.
- Include artifacts excluded after cutoff next round, starting with `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000013/call_0001_d2dcc950/submission.py`.

## Cycle 20260719T100921Z hourly submission review

- Completed at UTC: `2026-07-19T10:13:09Z`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- Cached official status counts after refresh: `{'graded': 24, 'failed': 1}`. Pending/queued/running registered submissions: `[]`. No historical pending status changes needed ledger backfill.
- Quota assessment for UTC `2026-07-19`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`. Blocker: `AIcrowdAuthError: While checking submission status: Your AIcrowd API key is missing or invalid. (HTTP 401)`.
- Candidate cutoff UTC: `2026-07-19T10:09:21+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-19T09:19:20.152432+00:00']`.
- JSONL context files were read for sampler/metrics/agent/gen-score line counts and provenance while live runs continued; live context mtime range UTC: `['2026-07-18T17:40:25.175395+00:00', '2026-07-19T09:19:20.152432+00:00']`.
- Console logs inspected: `18` files under `codex_runs/aicrowd_whestbench/multi_initial_20260718/console`; mtime range UTC `['2026-07-18T14:51:18.876898+00:00', '2026-07-19T08:08:59.858134+00:00']`; no console log was newer than this cycle cutoff. No estimator was promoted from console text.
- No `candidate_pool.json`, candidate-pool directory, or extra candidate directory was found outside scanned step/pool/workspace artifacts.
- Files excluded after cutoff: `1`; examples: `['codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000013/call_0001_d2dcc950/submission.py']`.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T100921Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T100921Z/scan_summary.json`
- Shortlist manifest: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T100921Z/shortlist_manifest.json`
- Shortlist source directory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T100921Z/shortlist_sources`
- Full-100 reports: none created this cycle.
- Parse errors: `[]`.

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count |
|---|---|---|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | `-` | `0-8` | 9 | 18 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | `-` | `0-8` | 9 | 18 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | `-` | `0-10` | 11 | 22 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | `-` | `0-15` | 16 | 32 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | `-` | `0-13` | 13 | 27 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | `-` | `0-3` | 4 | 8 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | `-` | `0-6` | 7 | 14 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | `-` | `0-11` | 12 | 24 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | `-` | `0-9` | 10 | 20 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | `-` | `0-9` | 10 | 20 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | `-` | `0-6` | 7 | 14 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | `-` | `0-6` | 7 | 14 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | `0-50` | `-` | 0 | 51 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `480`.
- Historical/known SHA baseline size: `699`; old inventory entries read: `9593`.
- New SHA after ledger/registry/tracker/old-inventory/canonical-file baseline removal: `0`.
- Final decision counts: `{'reject_existing_sha': 480}`.
- By-family unique SHA counts: `{'affinef64': 88, 'gaussiancv': 54, 'german013': 76, 'layerrescale': 45, 'rqmc030': 110, 'sphericalrb': 107}`.
- Entered full-100: `0`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Shortlist And Decisions

- No candidate entered the full-100 shortlist. Every scanned SHA was already present in the historical/known baseline, including the best first-10 rows below.

| Family | Run | SHA | Step | First-10 adjusted | Decision |
|---|---|---|---:|---:|---|
| `affinef64` | `whest_0718_affinef64_auto_r1` | `921b581662fe` | `7` | 1.275691847e-07 | `reject_existing_sha` |
| `affinef64` | `whest_0718_affinef64_auto_r1` | `07b20a3cb799` | `7` | 1.275939272e-07 | `reject_existing_sha` |
| `layerrescale` | `whest_0718_layerrescale_auto_r2` | `3f26d2bb11e5` | `5` | 1.357858217e-07 | `reject_existing_sha` |
| `layerrescale` | `whest_0718_layerrescale_auto_r2` | `fba838fa2471` | `5` | 1.357995213e-07 | `reject_existing_sha` |
| `german013` | `whest_0718_german013_auto_r2` | `8c0a803abd3b` | `2` | 1.385124267e-07 | `reject_existing_sha` |
| `german013` | `whest_0718_german013_auto_r2` | `d1cc43eb710b` | `2` | 1.385124267e-07 | `reject_existing_sha` |
| `sphericalrb` | `whest_0718_sphericalrb_auto_r1` | `3c73a750aa29` | `5` | 1.461057215e-07 | `reject_existing_sha` |
| `gaussiancv` | `whest_0718_gaussiancv_auto_r1` | `a0f5bc4922b1` | `4` | 1.467102880e-07 | `reject_existing_sha` |

Interpretation:

- No new estimator SHA appeared after the prior `20260719T092813Z` cycle. The latest included candidate artifact was still from `2026-07-19T09:19:20.152432+00:00`.
- The candidate artifact excluded after cutoff is live output from `whest_0718_german013_auto_r1` and should be reviewed in the next cycle when its mtime is at or before that cycle cutoff.
- Because there were no deduped-new candidates, no local mini full-100 command was run, no package was built, and no official submission was attempted.

### Official Submission Status

- No `whest package` or `whest submit` was run.
- No new submission ID was obtained; `registry.json` was not changed after the required pre-review tracker refresh.
- `SUBMISSION_TRACKER.md` and `official_status.json` were refreshed by the required script.
- Official submission IDs this cycle: none.
- Live official submission would require credential/quota recovery because registered submission refreshes still report HTTP 401; server quota could not be verified. This was not the gating factor because no candidate passed dedupe/shortlist review.
- Best registered hosted public-50 score remains `2.787552892e-07` from #316192; best registered local full-100 remains `2.748615e-07`.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-19T10:09:21+00:00`; do not treat this cycle's `480` inventoried SHA as new.
- Latest scanned TTT steps at cutoff: `whest_0718_affinef64_ttt` `50`, `whest_0718_gaussiancv_ttt` `50`, `whest_0718_german013_ttt` `50`, `whest_0718_layerrescale_ttt` `50`, `whest_0718_rqmc030_ttt` `50`, `whest_0718_sphericalrb_ttt` `50`.
- Latest scanned AutoEvolve pool steps at cutoff: `whest_0718_affinef64_auto_r1` `8`, `whest_0718_affinef64_auto_r2` `8`, `whest_0718_gaussiancv_auto_r1` `10`, `whest_0718_gaussiancv_auto_r2` `15`, `whest_0718_german013_auto_r1` `13`, `whest_0718_german013_auto_r2` `3`, `whest_0718_layerrescale_auto_r1` `6`, `whest_0718_layerrescale_auto_r2` `11`, `whest_0718_rqmc030_auto_r1` `9`, `whest_0718_rqmc030_auto_r2` `9`, `whest_0718_sphericalrb_auto_r1` `6`, `whest_0718_sphericalrb_auto_r2` `6`.
- Include artifacts excluded after cutoff next round, starting with `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000013/call_0001_d2dcc950/submission.py`.

## Cycle 20260719T104558Z hourly submission review

- Completed at UTC: `2026-07-19T10:56:02+00:00`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- Cached official status counts after refresh: `{'graded': 24, 'failed': 1}`. Pending/queued/running registered submissions: `[]`. No historical pending status changes needed ledger backfill.
- Quota assessment for UTC `2026-07-19`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`. Blocker: `AIcrowdAuthError: While checking submission status: Your AIcrowd API key is missing or invalid. (HTTP 401)`.
- Candidate cutoff UTC: `2026-07-19T10:45:58+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-19T10:26:32.696770+00:00']`.
- JSONL context files were read for sampler/metrics/agent/gen-score line counts and provenance while live runs continued; live context mtime range UTC: `['2026-07-18T17:40:25.175395+00:00', '2026-07-19T10:26:32.696770+00:00']`.
- Console logs inspected: `18` files under `codex_runs/aicrowd_whestbench/multi_initial_20260718/console`; no console log was newer than cutoff and no estimator was promoted from console text.
- No `candidate_pool.json`, candidate-pool directory, or extra candidate directory was found outside scanned step/pool/workspace artifacts.
- Files excluded after cutoff: `1`; examples: `['codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000014/call_0001_ee933606/submission.py']`.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T104558Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T104558Z/scan_summary.json`
- Shortlist manifest: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T104558Z/shortlist_manifest.json`
- Shortlist source directory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T104558Z/shortlist_sources`
- Full-100 reports: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T104558Z/full100_reports`
- Parse errors: `[]`.

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count |
|---|---|---|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | `-` | `0-8` | 9 | 18 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | `-` | `0-8` | 9 | 18 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | `-` | `0-10` | 11 | 22 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | `-` | `0-15` | 16 | 32 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | `-` | `0-14` | 14 | 29 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | `-` | `0-3` | 4 | 8 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | `-` | `0-6` | 7 | 14 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | `-` | `0-11` | 12 | 24 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | `-` | `0-9` | 10 | 20 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | `-` | `0-9` | 10 | 20 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | `-` | `0-6` | 7 | 14 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | `-` | `0-6` | 7 | 14 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | `0-50` | `-` | 0 | 51 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `482`.
- Historical/known SHA baseline size: `699`; old inventory entries read: `10073`.
- New SHA after ledger/registry/tracker/old-inventory/canonical-file baseline removal: `2`.
- Final decision counts: `{'reject_existing_sha': 480, 'reject_full100_no_submit': 1, 'reject_ranked_not_full100': 1}`.
- By-family unique SHA counts: `{'affinef64': 88, 'gaussiancv': 54, 'german013': 78, 'layerrescale': 45, 'rqmc030': 110, 'sphericalrb': 107}`.
- Entered full-100: `1`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Shortlist And Decisions

| SHA | Family / run / step | State | First-10 adjusted / MSE / C-B / fail | Full-100 adjusted / MSE / C-B / fail | Decision |
|---|---|---|---|---|---|
| `09172f22b10b` | `german013` / `whest_0718_german013_auto_r1` / step `14` | `310cebc5-1d51-4556-b20c-64e7a6d1204a` | 1.530712818e-07 / 1.489367094e-06 / 0.1028 / 0 | 3.397278776e-07 / 3.304204163e-06 / 0.1028 / 0 | `reject_full100_no_submit` |
| `294930ff9c4b` | `german013` / `whest_0718_german013_auto_r1` / step `14` | `b0d06435-d4c3-4515-b297-c7b7e5ac3d29` | 1.531108907e-07 / 1.489367094e-06 / 0.1028 / 0 | - / - / - / - | `reject_ranked_not_full100` |

Full-100 interpretation:

- `09172f22b10b` report `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T104558Z/full100_reports/german013_preprop_blend_09172f22b10b_full100_20260719T104558Z.json`; estimator SHA `09172f22b10b6aff4c3f7073a57917870423160cbc51276c98bf2d4da60db8bc`; report SHA `27b18ddfc1e06b112c2dc55aabd6e76f4e229e2a846c2ad02c84151ed325d109`.
- Method summary: setup-cached German-family radial/whitened estimator with two sample fractions, first-activation moment rescaling, prepropagation through 18 layers, and final normal-ReLU blending.
- The selected candidate improved same-family initial first-10 by `64.37%` but improved its immediate parent first-10 by only `0.0346%`.
- Full-100 adjusted `3.397278776e-07` had 0 failures and failure breakdown `{'budget_exhausted': 0, 'combined_budget_exhausted': 0, 'error': 0, 'residual_wall_time_exhausted': 0, 'time_exhausted': 0}`.
- Full-100 is `23.6%` worse than registered local full-100 best `2.748615e-07` and `21.9%` worse than the registered hosted public-50 best `2.787552892e-07` (#316192). It is about `2.6%` better than the prior German-family full-100 representative `3.489487931e-07`, but not enough to justify official validation.
- `294930ff9c4b` was not run full-100 because it is a lower-ranked same-family/method variant from the same pool step and the stronger representative failed the submission bar.

### Official Submission Status

- No `whest package` or `whest submit` was run.
- No new submission ID was obtained; `registry.json` was not changed after the required pre-review tracker refresh.
- `SUBMISSION_TRACKER.md` and `official_status.json` were refreshed by the required script before review. `official_status.json` was refreshed at `2026-07-19T18:46:34+08:00`.
- Official submission IDs this cycle: none.
- Live official submission would require credential/quota recovery because registered submission refreshes still report HTTP 401; server quota could not be verified. This was not the gating factor because the only full-100 candidate did not meet the quality bar.
- Best registered hosted public-50 score remains `2.787552892e-07` from #316192; best registered local full-100 remains `2.748615e-07`.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-19T10:45:58+00:00`; do not treat this cycle's `482` inventoried SHA as new.
- Latest scanned TTT steps at cutoff: `whest_0718_affinef64_ttt` `50`, `whest_0718_gaussiancv_ttt` `50`, `whest_0718_german013_ttt` `50`, `whest_0718_layerrescale_ttt` `50`, `whest_0718_rqmc030_ttt` `50`, `whest_0718_sphericalrb_ttt` `50`.
- Latest scanned AutoEvolve pool steps at cutoff: `whest_0718_affinef64_auto_r1` `8`, `whest_0718_affinef64_auto_r2` `8`, `whest_0718_gaussiancv_auto_r1` `10`, `whest_0718_gaussiancv_auto_r2` `15`, `whest_0718_german013_auto_r1` `14`, `whest_0718_german013_auto_r2` `3`, `whest_0718_layerrescale_auto_r1` `6`, `whest_0718_layerrescale_auto_r2` `11`, `whest_0718_rqmc030_auto_r1` `9`, `whest_0718_rqmc030_auto_r2` `9`, `whest_0718_sphericalrb_auto_r1` `6`, `whest_0718_sphericalrb_auto_r2` `6`.
- Include artifacts excluded after cutoff next round, starting with `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000014/call_0001_ee933606/submission.py`.

## Cycle 20260719T112936Z hourly submission review

- Completed at UTC: `2026-07-19T11:38:46Z`.
- Required tracker refresh ran before review: `PYTHONPATH=/tmp/whest-official-deps:/opt/tiger/discover python repro/aicrowd_whestbench/update_submission_tracker.py`.
- `official_status.json` refreshed at `2026-07-19T19:30:18+08:00`; cached official status counts after refresh: `{'failed': 1, 'graded': 24}`. Pending/queued/running registered submissions: `[]`. No historical pending status changes needed ledger backfill.
- Quota assessment for UTC `2026-07-19`: registered submissions created today `0` / `50`, nominal remaining `50`, server verified `False`. Blocker: `AIcrowdAuthError: While checking submission status: Your AIcrowd API key is missing or invalid. (HTTP 401)`.
- Candidate cutoff UTC: `2026-07-19T11:29:36+00:00`; scanned candidate artifact mtime range UTC: `['2026-07-18T14:52:03.778030+00:00', '2026-07-19T11:17:15.555585+00:00']`.
- JSONL context files were read for sampler/metrics/agent/gen-score line counts and provenance while live runs continued; live context mtime range UTC: `['2026-07-18T17:40:25.175395+00:00', '2026-07-19T11:17:15.555585+00:00']`.
- Console logs inspected: `18` files under `codex_runs/aicrowd_whestbench/multi_initial_20260718/console`; mtime range UTC `['2026-07-18T14:51:18.876898+00:00', '2026-07-19T08:08:59.858134+00:00']`; tail flags `{'candidate_failed': 11, 'connection_reset_or_reconnect': 16, 'missing_file_or_hf': 3, 'sandbox_or_instruction_context': 4, 'timeout_or_terminated': 11}`. No estimator was promoted from console text.
- No `candidate_pool.json`, candidate-pool directory, or extra candidate directory was found outside scanned step/pool/workspace artifacts.
- Files excluded after cutoff: `1`; examples: `['codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000015/call_0001_04892f65/submission.py']`.

### Scanned Runs And Files

- Run root: `codex_runs/aicrowd_whestbench/multi_initial_20260718`
- Inventory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T112936Z/candidate_inventory.jsonl`
- Scan summary JSON: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T112936Z/scan_summary.json`
- Shortlist manifest: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T112936Z/shortlist_manifest.json`
- Shortlist source directory: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T112936Z/shortlist_sources`
- Full-100 reports: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T112936Z/full100_reports`
- Parse errors: `[]`.

| Run | Family | Kind | PUCT steps | AutoEvolve pool steps | Workspace submissions | Candidate file count |
|---|---|---|---:|---:|---:|---:|
| `whest_0718_affinef64_auto_r1` | `affinef64` | `auto_r1` | `-` | `0-8` | 9 | 18 |
| `whest_0718_affinef64_auto_r2` | `affinef64` | `auto_r2` | `-` | `0-8` | 9 | 18 |
| `whest_0718_affinef64_ttt` | `affinef64` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_gaussiancv_auto_r1` | `gaussiancv` | `auto_r1` | `-` | `0-10` | 11 | 22 |
| `whest_0718_gaussiancv_auto_r2` | `gaussiancv` | `auto_r2` | `-` | `0-15` | 16 | 32 |
| `whest_0718_gaussiancv_ttt` | `gaussiancv` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_german013_auto_r1` | `german013` | `auto_r1` | `-` | `0-15` | 15 | 31 |
| `whest_0718_german013_auto_r2` | `german013` | `auto_r2` | `-` | `0-3` | 4 | 8 |
| `whest_0718_german013_ttt` | `german013` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_layerrescale_auto_r1` | `layerrescale` | `auto_r1` | `-` | `0-6` | 7 | 14 |
| `whest_0718_layerrescale_auto_r2` | `layerrescale` | `auto_r2` | `-` | `0-11` | 12 | 24 |
| `whest_0718_layerrescale_ttt` | `layerrescale` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_rqmc030_auto_r1` | `rqmc030` | `auto_r1` | `-` | `0-9` | 10 | 20 |
| `whest_0718_rqmc030_auto_r2` | `rqmc030` | `auto_r2` | `-` | `0-9` | 10 | 20 |
| `whest_0718_rqmc030_ttt` | `rqmc030` | `ttt` | `0-50` | `-` | 0 | 51 |
| `whest_0718_sphericalrb_auto_r1` | `sphericalrb` | `auto_r1` | `-` | `0-6` | 7 | 14 |
| `whest_0718_sphericalrb_auto_r2` | `sphericalrb` | `auto_r2` | `-` | `0-6` | 7 | 14 |
| `whest_0718_sphericalrb_ttt` | `sphericalrb` | `ttt` | `0-50` | `-` | 0 | 51 |

### Candidate Counts

- Unique estimator SHA reviewed this cycle: `484`.
- Historical/known SHA baseline size: `702`; old inventory entries read: `10555`.
- New SHA after ledger/registry/tracker/old-inventory/canonical-file baseline removal: `2`.
- Final decision counts: `{'reject_existing_sha': 482, 'reject_full100_no_submit': 1, 'reject_ranked_not_full100': 1}`.
- By-family unique SHA counts: `{'affinef64': 88, 'gaussiancv': 54, 'german013': 80, 'layerrescale': 45, 'rqmc030': 110, 'sphericalrb': 107}`.
- Entered full-100: `1`.
- Official submissions this cycle: `0`.
- 本轮无提交.

### Shortlist And Decisions

| SHA | Family / run / step | State | First-10 adjusted / MSE / C-B / fail | Full-100 adjusted / MSE / C-B / fail | Decision |
|---|---|---|---|---|---|
| `1454bcfc8cd8` | `german013` / `whest_0718_german013_auto_r1` / step `15` | `40d0480a-5978-42c4-be62-74814f67711b` | 1.530993060e-07 / 1.489367094e-06 / 1.027978765e-01 / 0 | 3.395962790e-07 / 3.304204163e-06 / 1.027780026e-01 / 0 | `reject_full100_no_submit` |
| `bfc90edd23bc` | `german013` / `whest_0718_german013_auto_r1` / step `15` | `f9394ed5-438d-43ed-8a70-1f4e3093cc2d` | 1.531212084e-07 / 1.489367094e-06 / 1.028151382e-01 / 0 | - / - / - / - | `reject_ranked_not_full100` |

Full-100 interpretation:

- `1454bcfc8cd8` source copy: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T112936Z/shortlist_sources/german013_candidate_1454bcfc8cd8.py`.
- `1454bcfc8cd8` full-100 report: `codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T112936Z/full100_reports/german013_prefix_preprop_1454bcfc8cd8_full100_20260719T112936Z.json`; report SHA `89f5aca270197da6e1931424c9ef4e0afc7811911786ed5eccc2459407c9fd13`.
- Method summary: setup-cached German-family radial/whitened estimator with main/extra sample fractions, first-activation moment rescaling, prepropagation through 18 layers, and final normal-ReLU blending. Compared with the previous step-14 representative, this version removes the `whestbench.generation.sample_mlp` import and manually draws only the prefix weights used for prepropagation; constants and estimator mechanism are otherwise the same, with a minor preprop loop boundary cleanup.
- Static inspection found no file/network/subprocess/reflection behavior and no target/hidden-state reads. Imports were limited to `math`, `flopscope.numpy`, and `whestbench.BaseEstimator`.
- First-10 was clean and strong versus the German013 initial (`64.343%` by recorded ratio), but it was slightly worse than the immediate parent (`-0.018%`) and slightly worse than the prior step-14 first-10 representative.
- Full-100 was clean: adjusted `3.395962790e-07`, MSE `3.304204163e-06`, C/B `1.027780026e-01`, failures `0`, failure_breakdown `{'budget_exhausted': 0, 'combined_budget_exhausted': 0, 'error': 0, 'residual_wall_time_exhausted': 0, 'time_exhausted': 0}`, max per-MLP effective compute `2.799384945e+10`.
- The full-100 score is only about `0.039%` better than the prior German step-14 representative (`3.397278776e-07`) and remains worse than registered local full-100 best `2.748615e-07` and registered hosted public-50 best `2.787552892e-07` (#316192). No official validation was justified.
- `bfc90edd23bc` was not run full-100 because it is the lower-ranked same-family/method step-15 variant with weaker first-10 (`1.531212084e-07`).

### Official Submission Status

- No `whest package` or `whest submit` was run.
- No new submission ID was obtained; `registry.json` was not changed after the required pre-review tracker refresh.
- `SUBMISSION_TRACKER.md` and `official_status.json` were refreshed by the required script before review.
- Official submission IDs this cycle: none.
- Live official submission would require credential/quota recovery because registered submission refreshes still report HTTP 401; server quota could not be verified. This was not the gating factor because the only full-100 candidate did not meet the quality bar.
- Best registered hosted public-50 score remains `2.787552892e-07` from #316192; best registered local full-100 remains `2.748615e-07`.

### Next-Cycle Continuation Point

- Continue from candidate artifact cutoff `2026-07-19T11:29:36+00:00`; do not treat this cycle's `484` inventoried SHA, raw new SHA `1454bcfc8cd84b364fe139c500ab36b02e1c14e860e86ab4f4821653671582a7`, or raw new SHA `bfc90edd23bc667608e41e13dd7b47d86b27f6b10bab6cc5a27d7c5c55970026` as new.
- Latest scanned TTT steps at cutoff: `whest_0718_affinef64_ttt` `50`, `whest_0718_gaussiancv_ttt` `50`, `whest_0718_german013_ttt` `50`, `whest_0718_layerrescale_ttt` `50`, `whest_0718_rqmc030_ttt` `50`, `whest_0718_sphericalrb_ttt` `50`.
- Latest scanned AutoEvolve pool steps at cutoff: `whest_0718_affinef64_auto_r1` `8`, `whest_0718_affinef64_auto_r2` `8`, `whest_0718_gaussiancv_auto_r1` `10`, `whest_0718_gaussiancv_auto_r2` `15`, `whest_0718_german013_auto_r1` `15`, `whest_0718_german013_auto_r2` `3`, `whest_0718_layerrescale_auto_r1` `6`, `whest_0718_layerrescale_auto_r2` `11`, `whest_0718_rqmc030_auto_r1` `9`, `whest_0718_rqmc030_auto_r2` `9`, `whest_0718_sphericalrb_auto_r1` `6`, `whest_0718_sphericalrb_auto_r2` `6`.
- Include artifacts excluded after cutoff next round, starting with `codex_runs/aicrowd_whestbench/multi_initial_20260718/whest_0718_german013_auto_r1/autoevolve_workspaces/step_000015/call_0001_04892f65/submission.py`.


## Cycle 20260719T121056Z wrapper failure

- Codex process exited with status 101.
- Recovery log: `/opt/tiger/discover/codex_runs/aicrowd_whestbench/multi_initial_20260718/hourly_submission_review/cycles/20260719T121056Z/codex_events.jsonl`
