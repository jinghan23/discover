你在 repo root 工作。请生成用户所要求的 iterative trajectory 的 HTML 报告和 state tree 页面。

输入数据位置：
- 扫描 `codex_runs/**/metrics.jsonl`
- 只选文件 mtime 在最近 14 天内的轨迹目录
- 每个轨迹目录通常包含：
  - `metrics.jsonl`
  - `agent_outputs.jsonl`
  - `puct_sampler_step_*.json`
  - autoevolve 轨迹还可能包含 `codex_autonomous_workspaces/step_*/call_*`
  - 普通 Codex CLI 轨迹还可能包含 `codex_cli_calls/step_*/group_*`
- 邻近父目录可能有 launch `.sh`、terminal `.log`、`README.md`，用于推断运行方式。

输出目录：
- `reports/trajectory_analysis_<YYYYMMDD>/`

必须生成：
- `index.html`
- 每条轨迹一个 `<run_name>.html`
- `analysis_summary.json`
- 对 autoevolve 轨迹额外生成：
  - `<run_name>_autoevolve_calls.json`
  - `<run_name>_autoevolve_events.jsonl`
- 对 TTT Discover 轨迹生成：
  - `<run_name>_state_graph.svg`
  - `<run_name>_state_graph.mmd`
  - `<run_name>_state_graph.dot`

轨迹解析规则：
1. `metrics.jsonl` 是每 step 指标。
2. `agent_outputs.jsonl` 是候选输出。
3. 最终 pool 用最新的 `puct_sampler_step_*.json`。
4. 对 autoevolve，不要只看 `agent_outputs.jsonl` 的最终候选；还要解析每个 Codex 子进程调用目录里的内部过程。
5. autoevolve 调用目录优先扫描 `codex_autonomous_workspaces/step_*/call_*`，排除 `*_codex_home` 和 `removed_steps_*`；普通 CLI 调用可扫描 `codex_cli_calls/step_*/group_*`。
6. 调用目录内：
   - `prompt.txt` 是该 call 的输入。
   - `codex.stdout.log` / `final_response.txt` 通常是 Codex 最终回答。
   - `submission.py` 是 autoevolve call 最终留下的代码产物。
   - `codex.stderr.log` 是 Codex CLI transcript，通常包含中间 shell 命令、文件查看、evaluator 调用和 evaluator 输出；不要把它简单当作错误日志。
7. 正确 runtime 候选定义为：
   - `correctness == 1`
   - `raw_score` 是正数
   - `raw_score` 越低越好，单位 microseconds。
8. 若 agent outputs 缺失，可 fallback 到 metrics 里的 `gpu_mode/score_us/min` 或 `raw_score/min`。

Autoevolve 内部 evaluator 解析：
- 同时读取每个 call 的 `codex.stderr.log` 和 `codex.stdout.log`，但 evaluator 时间线主要来自 `codex.stderr.log`。
- 识别 evaluator 命令：
  - `/bin/bash -lc ... python eval_client.py`
  - `/bin/bash -lc ... python eval_candidate.py`
- 识别 evaluator 结果事件：
  - blackbox 成功：`pass score=...`
  - blackbox 失败：`FAIL stage=...`
  - local 成功：`score_us ...`
  - local 失败：`TEST FAILED` 或 `LEADERBOARD FAILED`
- 每个结果事件记录：`loop`, `call_name`, `source_log`, `line`, `kind`, `status`, `score`, `stage`, `message`, `raw_line`, `nearest_previous_eval_command`。
- Codex 可能执行 `tail codex.stderr.log` 导致旧输出回放；统计主口径用结果事件而不是命令文本。完全相同的成功结果行在同一 call 内去重，失败结果保留 raw 计数。
- 每个 call 汇总：`cmd_mentions`, `result_events_raw`, `success_raw`, `success_unique`, `fail_raw`, `cleaned_count = success_unique + fail_raw`, `best_internal_score`, `last_internal_score`, `last_event`。
- 每个 autoevolve run 汇总 per-loop / per-call evaluator counts，并在页面里展示内部 best score 曲线；TriMul runtime 分数仍按 lower-is-better。

计算指标：
- `best`: 所有正确候选中最低 `raw_score`。
- `best-updates`: running best 每次刷新的 step 列表，包含初始有效 best。
- `last-improve`: 最后一次 running best 改进的 step。
- `plateau`: `final_step - last_improve_step`，同时给 plateau share。
- `early-share`: 前 25% observed steps 已完成的总 improvement 占比：
  `(first_score - early_best) / (first_score - final_best)`，clamp 到 `[0,1]`；无改进则 1.0。
- 另外汇总：
  - total samples
  - correct samples
  - failed samples
  - correct rate
  - timeout count
  - top repeated messages
  - pool size trend
  - reward trend
  - buffer value trend

产出方式推断：
- 从 launch `.sh` / README / path 中解析或推断：
  - `--runner`
  - `--task`
  - `--num-epochs`
  - `--group-size`
  - `--codex-cli-sandbox`
  - `--codex-autonomous`
  - `--codex-autonomous-blackbox`
  - `--reward-shaping`
  - `--codex-model-name`
  - `--gpu`
- method label 规则：
  - 路径或 flags 含 `blackbox` / `--codex-autonomous-blackbox`: `codex autonomous blackbox`
  - flags 含 `--codex-autonomous` 或路径含 `autoevolve` / `autonomous`: `codex autonomous / autoevolve`
  - 路径含 `ttt_discover` 或 runner 是 `codex_no_finetune`: `ttt discover`
  - 有 reward shaping 加 `reward shaping`
  - 有 proxy 加 `proxy`
  - 加上 sandbox、group、epochs 信息。

HTML 页面要求：
- `index.html` 汇总所有轨迹，按 best runtime 排序。
- 每条轨迹页面包含：
  - run path
  - method label
  - command flags
  - best / best step
  - early-share
  - last-improve
  - best-updates
  - plateau
  - correct rate
  - runtime trend chart
  - reward / pool value chart
  - timeline sample table
  - 如果是 autoevolve，增加 autonomous call summary：
    - 每个 loop/call 的 evaluator cleaned count、raw count、best internal score、last event、stdout/final_response 摘要、`submission.py` 链接
    - 可折叠的 evaluator event timeline，默认显示 first / best-improvement / last events，避免页面过长
    - 将 evaluator event timeline 中的 score/result event 按 call 内发生顺序串成 evaluator score state tree；成功节点显示 score，失败节点显示 stage，lower-is-better 用绿色突出，全局 best-improvement 路径加粗；输出并链接 SVG/Mermaid/DOT
    - 在 evaluator score state tree 下方增加中文 node-change table，且必须使用净变化口径：为每个 node 关联 evaluator 前最近一次真实 `submission.py` 状态/hash，只比较相邻 node 对应状态之间的净 diff；过滤 transcript 中重复打印的完整 diff / tail 回放；表中显示 from_hash -> to_hash、变更类型（首次实现、净新增/调参、纯复测、回退/恢复）、净新增/删除行数和中文变化意义
  - best updates table
  - top candidates table
  - final pool top states
  - anomalies / repeated messages
  - 如果是 TTT Discover，嵌入 state tree SVG，并在 tree 下方增加中文节点间净变化说明表：对最终 pool state tree 的每条直接父边 `parents[0] -> child`，比较 parent/child `State.code` 的净 diff，不使用 transcript 回放；表中显示 score delta、from_hash -> to_hash、best-path 标记、变更类型、净新增/删除行数和中文变化意义。

State tree 规则：
- 只给 method label 以 `ttt discover` 开头的轨迹画 tree。
- 使用最终 `puct_sampler_step_*.json`。
- 每个最终 pool state 一个节点。
- 只画直接父边：`parents[0] -> child`。
- 节点间变化说明只使用最终 snapshot 中的 `State.code`，必须是相邻 state 的净变化；如果 score 变差，标为探索/回退，不要把重复候选或历史 transcript diff 当作新改动。
- 不要把 `agent_outputs.jsonl` 中的 invalid/timeout/failed/duplicate 候选额外画成节点，除非它已经在最终 snapshot 的 `states` 里。
- `parents` 如果是完整祖先链，只使用 `parents[0]` 作为 direct parent。
- 节点标签：
  - id 前 8 位
  - `ts=<timestep> n=<puct_n or usage>`
  - 对普通 runtime run，若 state value 是负数，显示 `score_us=-value`
  - 对 reward-shaping run 或正 value，显示 `value=<value>`
- 颜色：
  - green = better value / lower runtime
  - yellow/orange/red = worse
  - gray = root / sentinel / missing score
  - thick green path = highest-value direct-parent chain
- 布局：
  - left-to-right by ancestry depth
  - siblings 按 timestep，再按 value 排序
- 输出 SVG/Mermaid/DOT，并在 HTML 页面链接它们。

验证：
- 所有 HTML 包含 `<html>` 和 `</html>`。
- TTT Discover 轨迹数量等于 state graph 数量。
- 每个 graph 都有 `.svg/.mmd/.dot`。
- 每个 autoevolve run 都有 `_autoevolve_calls.json` 和 `_autoevolve_events.jsonl`。
- SVG 以 `<svg` 开头。
- Mermaid 以 `graph LR` 开头。
- DOT 以 `digraph state_graph` 开头。
- 最终回复 index.html、analysis_summary.json 路径，并列出每个 tree 的 node/edge/root/max-depth/best-state。

发布 HTML 报告：
- 如果需要通过 bdgrok 发布报告，先在报告输出目录启动本地静态服务，例如：
  ```bash
  cd reports/trajectory_analysis_<YYYYMMDD>/
  python -m http.server 3000 --bind 127.0.0.1
  ```
- 安装或升级 bdgrok：
  ```bash
  pip3 install byted_bdgrok -i https://bytedpypi.byted.org/simple --trusted-host bytedpypi.byted.org --upgrade
  ```
- 启用详细日志并发布 3000 端口：
  ```bash
  bdgrok http 3000 --verbose
  ```
