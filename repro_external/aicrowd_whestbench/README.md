# ARC WhestBench 资料索引

这里保存 ARC White-Box Estimation Challenge 2026 的本地复现、评测报告和
AIcrowd 提交记录。**从本页开始阅读，不要直接从散落的 JSON 猜实验含义。**

## 先看什么

1. 看最新提交结果：[`SUBMISSION_TRACKER.md`](SUBMISSION_TRACKER.md)。这是由本地
   registry 和 AIcrowd API 生成的唯一权威提交表。
2. 看 `first-10`、`full-100`、`public-50` 和各分数字段的含义：
   [`docs/PROTOCOLS.md`](docs/PROTOCOLS.md)。
3. 看某种方法的算法与复现结论：
   [`docs/PUBLIC_METHODS.md`](docs/PUBLIC_METHODS.md) 和
   [`public_reproductions/`](public_reproductions/README.md)。
4. 看旧的实验过程、上传命令和历史叙述：
   [`docs/SUBMISSION_HISTORY.md`](docs/SUBMISSION_HISTORY.md)。它是历史档案，
   不是当前分数来源。

## 目录结构

| 路径 | 内容 | 是否权威/可编辑 |
|---|---|---|
| `SUBMISSION_TRACKER.md` | 所有已登记提交的本地与官方成绩 | 权威；生成文件，不手改 |
| `submissions/` | registry、API 缓存、候选代码和上传包 | registry 手工维护，其余多为产物 |
| `baselines/` | 官方 starter baseline 的代码与 full-100 报告 | 可复现参考 |
| `public_reproductions/` | 公开方法的独立实现与评测报告 | 可复现参考 |
| `discovered/` | TTT-Discover / AutoEvolve 发现的 estimator 与报告 | 实验记录 |
| `leaderboard/` | 某一日期的公开榜快照 | 历史快照，不是实时榜单 |
| `docs/` | 协议词汇、公开方法说明和历史记录 | 人工维护的说明 |

## 统一术语

- **local mini first-10**：本地 `mini` 数据集的前 10 个 MLP，仅用于快速筛选。
- **local mini full-100**：本地 `mini` 数据集全部 100 个 MLP，用于本地最终比较。
- **hosted public-50**：AIcrowd 服务器公开评分使用的 50 个 MLP。
- `budget013`：目标计算预算约 13%，不是 13 个 MLP。
- `adjusted score`：同时考虑预测误差和计算量的主分数，越低越好。

三套评测对象不能混成同一个排行榜。特别是 first-10 方差很大，不能用它的细微
分差判断最终优劣。

## 维护入口

新增提交时，先更新 [`submissions/registry.json`](submissions/registry.json)，再运行：

```bash
PYTHONPATH=/tmp/whest-official-deps:$PWD \
  python repro/aicrowd_whestbench/update_submission_tracker.py
```

离线重建 tracker 时加 `--offline`。详细字段和文件命名规则见
[`docs/PROTOCOLS.md`](docs/PROTOCOLS.md)。

## 外部资源

- [Challenge overview](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026)
- [Public leaderboard](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards)
- [Official starter kit](https://github.com/AIcrowd/whest-starterkit)
