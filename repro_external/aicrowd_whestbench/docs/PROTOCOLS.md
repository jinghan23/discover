# WhestBench 评测协议与命名

本文是本目录的统一词汇表。所有新记录应使用这里的名称，避免继续混用
`mini`、`official mini`、`full-mini` 和 `public`。

## 三种评测范围

| 规范名称 | 常见旧写法 | 数据范围 | 推荐用途 |
|---|---|---:|---|
| local mini first-10 | `first-10`, `mini n=10` | 本地 mini 前 10 个 MLP | 报错、超时和粗略预算筛选 |
| local mini full-100 | `full-100`, `full-mini`, `mini n=100`, `official mini` | 本地 mini 全部 100 个 MLP | 本地最终比较与提交晋级 |
| hosted public-50 | `official`, `public score` | AIcrowd 托管的 50 个公开评分 MLP | 官方公开成绩 |

`full-100` 中的 “full” 只表示完整跑完本地 mini 的 100 个 MLP，不表示完整官方
隐藏测试集。`first-10` 是 full-100 的固定前 10 个，不是随机抽样。hosted
public-50 使用不同的聚合集，不能假定它是本地 100 个中的前 50 个。

## 主分数

对每个 MLP：

```text
C = tracked FLOPs + lambda * residual wall time
per-MLP adjusted score = final-layer MSE * max(0.1, C / B)
suite adjusted score = 所有 per-MLP adjusted score 的平均值
```

Phase 1 中 `B = 2.72e11` FLOPs，`lambda = 1e11` FLOPs/s。分数越低越好。
如果 estimator 失败、超时或超预算，该 MLP 按官方失败规则处理。

## Tracker 字段

| 字段 | 含义 |
|---|---|
| `Local n` | 本地评测使用的 MLP 数量；10 是 first-10，100 是 full-100 |
| `Local adjusted` | 本地 suite adjusted score，越低越好 |
| `Local MSE` | 本地 final-layer MSE 的平均值 |
| `Local C/B` | 本地平均有效计算量占预算的比例 |
| `Fail` | 失败 MLP 数量 |
| `Official adjusted` | AIcrowd hosted public-50 主分数 |
| `Official MSE` | AIcrowd API 的 `score_secondary` |
| `S/MSE` | 两个官方聚合值之比，只是隐含计算乘数，不是精确平均 C/B |
| `Delta` | `(official / local - 1) * 100%`；负数表示官方分数更低、更好 |
| `Status` | AIcrowd 评分状态 |

adjusted score 是“每个 MLP 先乘再平均”，因此通常不能用表里的平均 MSE 与平均
C/B 简单相乘来精确复原。

## 文件命名

- `*_first10_eval_YYYYMMDD.json`：local mini first-10 报告。
- `*_full100_eval_YYYYMMDD.json`：local mini full-100 报告。
- 新文件不要再使用含糊的 `official_mini_eval`；现有旧文件保留原名以保存出处。
- `budget013`、`budget030` 分别表示目标计算预算约 13%、30%。
- JSON 评测报告是机器记录；结论与导航应写入相邻的 `README.md`。

## 比较规则

1. 只在相同本地协议内比较绝对本地分数。
2. first-10 只用于快速筛选，不用于细粒度参数排序。
3. 提交候选至少应补跑 full-100；没有 full-100 时必须明确标注证据较弱。
4. hosted public-50 是最终公开依据，但它与本地集合不同；漂移不自动表示 evaluator
   有错误。
5. “best official” 若没有特别说明，只指本地 registry 已登记的提交，不指全榜第一。

## 三 sampling-seed 评测协议

新的 stochastic sampling 候选不能再凭单次 local full-100 结果直接提交。使用固定的
100 个 MLP 和 ground truth，只平移 estimator 自己的 RNG stream，默认 offsets 为
`0`、`1000003`、`2000003`；German current 使用相同 offsets 作逐 seed、逐 MLP
配对基线。

`evaluate_public_reproduction.py` 是三 seed 门禁客户端。它通过已经运行的 blackbox
server 分别评测候选和 German current 的三个 offset，共六次 official suite 调用；
随后执行逐 seed/逐 MLP 配对、20,000 次 MLP-cluster bootstrap 和提交门禁。它不在
客户端加载数据，也不启动 evaluator。server 需要使用普通的
`WhestBenchRewardEvaluator` 并允许 request state（`--allow-request-state`）：

```bash
TTT_BLACKBOX_EVAL_SOCKET=/tmp/ttt_blackbox_eval_whestbench.sock \
  .venv/bin/python repro/aicrowd_whestbench/evaluate_public_reproduction.py \
  repro_external/aicrowd_whestbench/submissions/candidates/CANDIDATE.py \
  repro_external/aicrowd_whestbench/submissions/seed_sweeps/METHOD_three_seed_gate.json
```

默认 baseline 是
`public_reproductions/german_alfaro_current_41493b1.py`，可用 `--baseline` 显式覆盖。

blackbox 协议只传 submission 源码和 request state，**不能携带附属文件**。因此权重放在
单独文件里的提交（learned residual 需要同目录的 `learned_residual_weights.npz`）走不了
这条路，必须用 `evaluate_full100_direct.py`：它在进程内直接调官方 scorer，用可重复的
`--asset` 把附属文件拷到 `estimator.py` 旁边。learned residual 训练途中的 full-100
监控用的就是它。该脚本是单次打分，不做三 seed 门禁；正式提交前的验收仍以上面的
三 seed 协议为准。

若共享宿主在连续 suite 切换时出现全套 `flops_used=0` 的 worker
`SETUP_TIMEOUT`，未预声明恢复策略的该整轮作废，不能把失败行用于方法比较，也不能
事后只挑成功 seed 拼接晋级。允许用
`--interleave-baseline --inter-run-cooldown-s 15 --max-infra-retries 2
--infra-retry-cooldown-s 30` 预声明恢复策略并重新执行完整六套门禁：六个 nominal
请求前对候选和 baseline 对称等待，顺序为 `C0,B0,C1,B1,C2,B2`。这只调节 client
侧 subprocess teardown/startup 节奏；server 的 official setup timeout、数据、预算
和计分必须保持不变，报告也会记录请求顺序和 cooldown。

基础设施 retry 只在一个 suite 的 100/100 行都是 `SETUP_TIMEOUT`、`flops_used=0`、
`effective_compute=0` 且没有任何 exhaustion flag 时触发；candidate 与 baseline 使用
完全相同的最多两次 retry。客户端采用第一个非基础设施失败的 attempt，不按分数
择优，并在 JSON 中保存所有 attempt 的 source hash、failure summary 和是否进入聚合。
partial setup failure、预测/预算/时间失败或任何非零计算都不得 retry；retry 用尽时
整轮仍不通过。

只有同时满足以下条件才可打包并测试 hosted public-50：

1. 三次 local full-100 都是 0 failures；
2. 三个 sampling seed 的 suite adjusted score 都低于配对 German current；
3. 三 seed 平均 adjusted score 低于 German current；
4. 先在每个 MLP 内平均三 seed 配对差，再按 MLP 做 20,000 次 bootstrap，其 95%
   percentile CI 上界仍小于 0。

seed sweep 报告保存在 `submissions/seed_sweeps/`。未通过门禁的方案保留为本地负
结果，不生成官方提交。
