# Submission candidates

这里保存提交候选的本地 estimator 和评测 JSON。当前官方成绩不要在本目录手工汇总，
统一查看 [`../../SUBMISSION_TRACKER.md`](../../SUBMISSION_TRACKER.md)。

## 文件分组

- `german_budget*.py`：German whitened-antithetic 方法的不同目标预算版本。
- `rqmc_budget*.py`：RQMC + Rao-Blackwell 方法的不同目标预算版本。
- `autoevolve_*_full100_eval_*.json`：AutoEvolve 候选的 full-100 报告；其源代码
  保存在报告 `source` 字段指向的 `codex_runs/` 工作区。
- 与 `.py` 同前缀的 `*_eval_*.json`：该 estimator 的不可变评测记录。

`codex_runs/` 是本地原始运行沙箱，默认不进入 Git。对于已提交候选，远端可从
`../packages/` 下对应 tarball 的 `estimator.py` 恢复实际上传源码；registry 和
tracker 保存 submission ID、评测报告、artifact 路径与 SHA-256。不要为了让报告
中的本地 `source` 路径可访问而强制提交整个 `codex_runs/`。

`budget013` 表示目标计算量约为总预算的 13%，与测试 MLP 数量无关。first-10 与
full-100 的定义见 [`../../docs/PROTOCOLS.md`](../../docs/PROTOCOLS.md)。

新候选优先采用 `*_first10_eval_YYYYMMDD.json` 或
`*_full100_eval_YYYYMMDD.json`，不要继续使用含糊的 `official_mini` 命名。
