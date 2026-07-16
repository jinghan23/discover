# WhestBench Submission History

> The canonical, API-refreshed local-versus-official table is
> [`../SUBMISSION_TRACKER.md`](../SUBMISSION_TRACKER.md). This file retains the
> original narrative and command history and is not the live score source.

> Archive note: this document is a dated narrative from 2026-07-14. For current
> results use [`../SUBMISSION_TRACKER.md`](../SUBMISSION_TRACKER.md); for field
> definitions use [`PROTOCOLS.md`](PROTOCOLS.md).

Checked on 2026-07-14 with `whestbench 0.12.0rc5`, `flopscope 0.8.0rc5`,
`aicrowd/arc-whestbench-public-2026@v1-phase1`, mini split, 100 MLPs,
official subprocess isolation, a `2.72e11` per-MLP budget, and
`lambda=1e11` FLOPs/s. Lower adjusted score is better.

## Phase 1 full-mini results

| Estimator | Adjusted score | Final MSE | Mean effective compute | C/B | Failures | Submission status |
|---|---:|---:|---:|---:|---:|---|
| AutoEvolve Gaussian CV `25d94716` | `3.621376e-7` | `1.479585e-6` | `6.658414e10` | `0.244795` | 0/100 | Submitted as AIcrowd #316299 |
| AutoEvolve Edgeworth `c02503e1` | `3.629318e-7` | `1.484736e-6` | `6.649210e10` | `0.244456` | 0/100 | Submitted as AIcrowd #316298 |
| AutoEvolve layer rescaling `f4443aad` | `3.633336e-7` | `3.633272e-6` | `2.696342e10` | `0.099130` | 0/100 | Submitted as AIcrowd #316300 |
| German Alfaro #311690 independent reproduction | `3.834192e-7` | `2.465925e-6` | `4.228989e10` | `0.155478` | 0/100 | Submitted as AIcrowd #316183 |
| evaaaz RQMC + Rao-Blackwell reproduction | `4.885414e-7` | `1.153299e-6` | `1.152261e11` | `0.423625` | 0/100 | Submitted as AIcrowd #316181 |
| AutoEvolve discovered candidate | `5.896640e-7` | `5.881288e-6` | `2.727423e10` | `0.100273` | 0/100 | Submitted as AIcrowd #316182 |
| pscamillo #314331 reproduction | `2.574242e-6` | `2.574242e-5` | `2.026487e9` | `0.007450` | 0/100 | Runnable, not packaged |
| TTT-Discover smoke candidate | `7.822846e-6` | `7.822846e-5` | `5.172857e9` | `0.019018` | 0/100 | Runnable, not packaged |
| Official diagonal baseline | `9.482215e-5` | `9.482215e-4` | `2.441924e8` | `0.000898` | 0/100 | Reference baseline |

The first three rows beat the official Monte Carlo reference band of roughly
`6.5e-7` on this local suite. The German entry is an independent implementation
of the publicly described method, not a copy of the upstream estimator.

## Warm-up-only method

The ascender1729 cumulant k3 + k4 estimator completes the old depth-8 warm-up
mini-100 at `4.975772e-7`, with `1.750546e10` tracked FLOPs per MLP and no
failures. It is not a valid Phase 1 candidate: the real depth-32 path exceeds
the official 30-second prediction timeout on the first network.

## Ready artifacts

All artifacts contain only `estimator.py` and `manifest.json` and pass the
official `whest validate` command.

| Artifact | SHA-256 |
|---|---|
| `../submissions/packages/evaaaz_rqmc_rao_blackwell_phase1_20260714.tar.gz` | `c1a358958b954590cfa71c95c3203c0a3e6b8cc4cbdd223347d63ff4b11fedfd` |
| `../submissions/packages/autoevolve_blackbox_phase1_20260714.tar.gz` | `e0488a3775804d94c0d36942e4fbe402f9ca564aa119e90260325ca65ee0379e` |
| `../submissions/packages/german_whitened_antithetic_reimplementation_phase1_20260714.tar.gz` | `644f6da862c57b4b29106ba53431c25f3ef9ef3368e31e774b180b8a5455c38f` |
| `../submissions/packages/rqmc_budget030_phase1_20260714.tar.gz` | `6288a7fb59c7ae0db30d5f9fa75176c87fcd4c08694149660630a966d39b3a09` |
| `../submissions/packages/rqmc_budget055_phase1_20260714.tar.gz` | `1f470f4883dba64aae5b2f804690688539aaca4693a9cd40cfce28e04b79a808` |
| `../submissions/packages/rqmc_budget070_phase1_20260714.tar.gz` | `c1fef7995d14581e0eda05961e0acf7ed2d57798f49cf00501976d610af5a375` |
| `../submissions/packages/german_budget010_phase1_20260714.tar.gz` | `048ed3a10a01cf3cb7b0f131a4c040241bf6462ee2925206f4ee1487eb5ff8c4` |
| `../submissions/packages/german_budget012_phase1_20260714.tar.gz` | `1c6d0b1a81b908722f49496a8f67d5654062a231a432a7668a721247bf4f47c8` |
| `../submissions/packages/german_budget020_phase1_20260714.tar.gz` | `194235814ad2279b31edc9fdb08babeefd5e11b30ddeb4230749effc95e7385c` |
| `../submissions/packages/german_budget025_phase1_20260714.tar.gz` | `42c01e687472bbd0c96ec6221fa2bf0e18bd1e10f99eb57a79f3d3737891df5c` |
| `../submissions/packages/german_budget013_phase1_20260714.tar.gz` | `a1395b694c3bbbeb717ee2f8430f3677d90983c4fa7e8e86a17669da1b453a4f` |
| `../submissions/packages/german_budget014_phase1_20260714.tar.gz` | `2daa561948fa069d2e6c1f18b2cbf735d1f9791ed1ee0d410a451ec91bb4accc` |
| `../submissions/packages/german_budget016_phase1_20260714.tar.gz` | `2cfbc29538a893d01777c28988507a090ca44ef9527dab6fdba180874fb2b2fd` |
| `../submissions/packages/german_budget018_phase1_20260714.tar.gz` | `59454db9c4901bf196048c237b96aaba4bff960bc25bfe5520865717fee18977` |
| `../submissions/packages/german_budget019_phase1_20260714.tar.gz` | `7e6483e2498c65063fa708fefdb483143ea9fc542f983350de0da747bc75890a` |
| `../submissions/packages/german_budget021_phase1_20260714.tar.gz` | `0741d309cf02c4b9e0671c0fb51da8da3f3a71b1d9443c72ce8b3793aab863e2` |
| `../submissions/packages/autoevolve_edgeworth_c02503e1_phase1_20260714.tar.gz` | `0e9fca4fb3faa9dc1a8a9c179b88c7344054c33464a1ec04376417b9695817c2` |
| `../submissions/packages/autoevolve_gaussian_cv_25d94716_phase1_20260714.tar.gz` | `515139e490d642b6350d3fc81171c3c2942304a7434f5137f2ac0a149177c809` |
| `../submissions/packages/autoevolve_layer_rescale_f4443aad_phase1_20260714.tar.gz` | `ff4437d67e38527b37435360f0c4ecdbd778f1d28a09398ebeb91903abe0ada5` |

## Official upload

This competition is hosted on AIcrowd, not Kaggle. Phase 1 remains open until
2026-07-31. The machine is authenticated with `whest login`.

### Hosted results

| Submission | Method | Official public adjusted | Secondary | Status |
|---|---|---:|---:|---|
| [#316181](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316181) | RQMC, original budget | `3.597020870802097e-7` | `8.558566429428537e-7` | graded |
| [#316182](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316182) | AutoEvolve | `4.346720784348491e-7` | `4.2314156712564e-6` | graded |
| [#316183](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316183) | German independent reproduction | `2.911839404774997e-7` | `1.859657245404378e-6` | graded |
| [#316184](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316184) | RQMC, 30% budget | `3.2626420687756247e-7` | `1.0798875160844545e-6` | graded |
| [#316185](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316185) | RQMC, 55% budget | `3.8966132953064667e-7` | `7.063287182518252e-7` | graded |
| [#316186](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316186) | RQMC, 70% budget | `3.647209562568389e-7` | `5.202637382240027e-7` | graded |
| [#316188](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316188) | German, 10% budget | `3.0585965126896845e-7` | `2.9964170641960664e-6` | graded |
| [#316189](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316189) | German, 12% budget | `2.818968207515024e-7` | `2.312304783345098e-6` | graded |
| [#316190](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316190) | German, 20% budget | `2.831611051104007e-7` | `1.4025789710103708e-6` | graded |
| [#316191](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316191) | German, 25% budget | `3.1972938035903804e-7` | `1.268280875592609e-6` | graded |
| [#316192](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316192) | German, 13% budget | `2.787552892175721e-7` | `2.113107371997103e-6` | graded |
| [#316193](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316193) | German, 14% budget | `2.8756119521941855e-7` | `2.0226783260568483e-6` | graded |
| [#316194](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316194) | German, 16% budget | `2.92394293697902e-7` | `1.8039042140571837e-6` | graded |
| [#316195](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316195) | German, 18% budget | `2.9202032364489314e-7` | `1.6063251047171434e-6` | graded |
| [#316196](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316196) | German, 19% budget | `2.929195575774832e-7` | `1.5243288032706913e-6` | graded |
| [#316197](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316197) | German, 21% budget | `3.0187919498472164e-7` | `1.4255890619097045e-6` | graded |
| [#316298](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316298) | AutoEvolve Edgeworth `c02503e1` | `3.167074694511809e-7` | `1.280396007246054e-6` | graded |
| [#316299](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316299) | AutoEvolve Gaussian CV `25d94716` | `3.088166465834724e-7` | `1.2542263675641153e-6` | graded |
| [#316300](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/316300) | AutoEvolve layer rescaling `f4443aad` | `3.472519114368648e-7` | `3.445321449362382e-6` | graded |

At the 2026-07-14 check, the public leaderboard's rank-50 score was
`2.867e-7`. None of these three AutoEvolve candidates entered the top 50;
the team's German 13% submission `#316192` remained at about rank 47 with
`2.787552892175721e-7`.

Authentication, when needed on a fresh machine, is performed without committing
the key:

```bash
PYTHONPATH=/tmp/whest-official-deps:$PWD \
  /tmp/whest-official-deps/bin/whest login
```

The upload command used for submission `#316181` was:

```bash
PYTHONPATH=/tmp/whest-official-deps:$PWD \
  /tmp/whest-official-deps/bin/whest submit \
  repro_external/aicrowd_whestbench/submissions/packages/evaaaz_rqmc_rao_blackwell_phase1_20260714.tar.gz \
  --description "RQMC Rao-Blackwell Phase 1 local mini 4.885e-7" \
  --watch
```

The hosted submission page reveals the 50-MLP public score. The other 50 MLPs
remain sealed, so a submission does not immediately reveal the final private
score or final rank.
