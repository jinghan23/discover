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
