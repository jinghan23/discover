# Validated submission packages

本目录中的 `.tar.gz` 是上传 AIcrowd 的验证后产物。文件名带构建日期，SHA-256 和
submission ID 的权威映射在 [`../registry.json`](../registry.json)，可读汇总在
[`../../SUBMISSION_TRACKER.md`](../../SUBMISSION_TRACKER.md)。

不要解压后原地修改 tarball；任何 estimator 变化都应重新构建、重新验证并生成新
文件名，然后更新 registry。

## 2026-07-21 retained review package

`autoevolve_quantile_secondmatch_4649711fe32a_phase1_20260721T094843Z.tar.gz`
通过了本地打包验证，但在三 seed 复验和相关方法的官方漂移复核后决定不上传。它保留
在仓库中用于固定该次审查候选的精确 estimator 字节，不应加入 `registry.json`，除非
未来确实取得新的官方 submission ID。

## 2026-07-16 same-name rebuild

`whitened_antithetic_spherical_rqmc_rb_budget010_phase1_20260716.tar.gz`
曾被两个并行会话使用：

- #316625 实际上传 archive SHA-256：
  `dae36b30f44616518526eaee8e7aec639cf0b9b05e1187416b363b921c7fd11b`。
- 同名文件随后被重新打包，当前 archive SHA-256：
  `309f97f132bc87c9bd4ec771ecf19b122cda5be03b0a27551e346bfffb1dc778`，
  并上传为 #316628。
- 两个 archive 内的 `estimator.py` 完全相同，SHA-256：
  `146a8c786ee31dfefc941712ded736c3ec2625343dd5cd184f80dde915a9d130`。

因此 #316625 的原始 archive 字节已不在当前路径，registry 使用
`artifact_sha256` 保留实际提交哈希；当前路径对应 #316628 的重新打包版本。

## #316628 failure classification

#316628 未产生分数，AIcrowd grading message 为
`Error : Evaluation could not complete; please retry`。平台没有报告 import、
contract、预算、超时或 estimator exception。由于包含相同 `estimator.py` 的
#316625 已成功完成评分，本次失败记录为平台/evaluation failure，不视为算法或
submission package 的可复现故障，也不再重复提交。
