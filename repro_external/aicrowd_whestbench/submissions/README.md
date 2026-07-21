# Submission records

本目录集中保存与团队提交直接相关的文件；当前成绩请始终查看根目录的
[`../SUBMISSION_TRACKER.md`](../SUBMISSION_TRACKER.md)。

| 路径 | 作用 | 维护方式 |
|---|---|---|
| `registry.json` | AIcrowd submission ID 到本地评测、上传包和提交时 SHA 的稳定映射 | 新提交后手工追加 |
| `official_status.json` | AIcrowd API 最近一次响应缓存 | tracker 脚本自动更新 |
| `candidates/` | 已考虑或已提交的 estimator 代码与本地报告 | 实验产物 |
| `packages/` | 通过 `whest validate` 的上传 tarball | 构建产物，内容不可随意改 |
| `reviews/` | 从本地搜索轨迹提取的候选快照、跨 seed 复验和提交决策 | 精简审计快照 |

## 新提交记录流程

1. 保存 estimator 与对应的 first-10 或 full-100 JSON 报告。
2. 构建并验证 `packages/` 下的 tarball。
3. 上传 AIcrowd，取得 submission ID。
4. 将 ID、method、family、local evaluation 和 artifact 路径加入 `registry.json`。
5. 从仓库根目录运行 `repro/aicrowd_whestbench/update_submission_tracker.py`。

如果同一路径可能被重新打包，必须在 registry 中写入 `artifact_sha256`，固定“实际
上传的 archive”哈希。tracker 会同时计算当前文件哈希，并在二者不一致时生成
integrity note。新包应始终使用新文件名，避免覆盖已有 submission 的 provenance。

`codex_runs/` 保存完整搜索沙箱，可能包含大量模型调用日志、临时 workspace 和运行时
配置，默认不进入 Git。需要长期保留的结果应提取到 `reviews/`：只保存候选源码、机器
评测 JSON、候选清单和人工结论，不复制 Codex home、stdout/stderr 或完整搜索轨迹。

`official_status.json` 只是缓存，不是人工记录；不要把它当作最终报告编辑。
