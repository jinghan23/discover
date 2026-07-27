#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

INTERVAL_SECONDS="${WHEST_REVIEW_INTERVAL_SECONDS:-1200}"
CODEX_MODEL="${WHEST_REVIEW_CODEX_MODEL:-gpt-5.6-sol}"
CODEX_REASONING_EFFORT="${WHEST_REVIEW_REASONING_EFFORT:-xhigh}"
RUN_ROOT="${WHEST_REVIEW_RUN_ROOT:-$REPO_ROOT/codex_runs/aicrowd_whestbench/multi_initial_20260718}"
STATE_ROOT="${WHEST_REVIEW_STATE_ROOT:-$RUN_ROOT/hourly_submission_review}"
LEDGER="$STATE_ROOT/REVIEW_LOG.md"
MONITOR_LOG="$STATE_ROOT/monitor.log"
DEPS_PATH="${WHEST_DEPS_PATH:-/tmp/whest-official-deps}"
BLACKBOX_SOCKET="${TTT_BLACKBOX_EVAL_SOCKET:-/tmp/ttt_blackbox_eval_whestbench.sock}"

mkdir -p "$STATE_ROOT"
touch "$LEDGER" "$MONITOR_LOG"

cycle_stamp=""
cycle_dir=""

render_prompt() {
    local prompt
    IFS= read -r -d '' prompt <<'PROMPT' || true
你是 ARC WhestBench Phase 1 的周期候选审查与官方提交代理。本轮时间戳是
`__CYCLE_STAMP__`。用户明确授权你：检查下面指定搜索产物；对达到严格门槛的候选
做本地官方验证、打包并提交 AIcrowd 官方 leaderboard；维护完整审计记录。

## 不可违反的边界

1. 不要停止、重启、修改或干扰正在运行的 TTT-Discover、AutoEvolve、tmux 或
   blackbox evaluator 进程。搜索目录原则上只读；仅在下述 review/submission 记录
   目录和既有 canonical submission 记录中写文件。
2. 保留工作树中所有既有改动，不要 reset、checkout、clean、commit、push 或删除文件。
3. 不得提交利用 evaluator/flopscope 漏洞、读取隐藏状态、依赖评测目标过拟合或明显违反
   比赛规则的候选。
4. 官方提交总配额是每天 50 次，不设每轮固定提交上限。先根据官方状态、registry 和
   ledger 核对今天已经使用的次数与剩余配额，再自行决定本轮提交多少；不得超过当天
   剩余配额，也不要为了用满配额而提交明显无价值的重复候选。相同 estimator SHA-256
   永远不得重复提交；仅改文件名、打包时间、随机种子、预算常数或微小系数不算新方法。
5. 在每一个官方提交完成或取得 submission ID 后，必须立刻把结果追加到总 ledger，
   然后才允许处理下一个官方提交。

## 首先读取的持久状态

- 上轮及历史审查 ledger：
  `__LEDGER__`
- 本轮专属审计目录：
  `__CYCLE_DIR__`
- 已有官方提交、分数、SHA 和 provenance：
  `__REPO_ROOT__/repro_external/aicrowd_whestbench/SUBMISSION_TRACKER.md`
  `__REPO_ROOT__/repro_external/aicrowd_whestbench/submissions/registry.json`
  `__REPO_ROOT__/repro_external/aicrowd_whestbench/submissions/official_status.json`
- 协议说明：
  `__REPO_ROOT__/repro_external/aicrowd_whestbench/docs/PROTOCOLS.md`
- submission 记录流程：
  `__REPO_ROOT__/repro_external/aicrowd_whestbench/submissions/README.md`

必须先读取 ledger，建立过去已经检查和提交过的 SHA 集合。若 ledger 为空，就从当前
registry、tracker 和所有现有产物建立第一次基线。不要把旧候选当作本小时新候选。
随后先运行
`PYTHONPATH=__DEPS_PATH__:__REPO_ROOT__ python repro/aicrowd_whestbench/update_submission_tracker.py`
刷新所有已登记 submission 的官方状态。若历史记录中有 queued/running/pending 项，必须
把本轮查到的最终成绩或状态变化补记到 ledger；它们不计作本轮新提交，也绝不能重提。

## 要检查的搜索产物

总根目录：
`__RUN_ROOT__`

它包含 6 个 initial-state 家族，每个家族一个 TTT run 和两个 AutoEvolve run，当前
full-100 实验名为：

- `whest_0720_full100_german013_{ttt,auto_r1,auto_r2}`
- `whest_0720_full100_affinef64_{ttt,auto_r1,auto_r2}`
- `whest_0720_full100_gaussiancv_{ttt,auto_r1,auto_r2}`
- `whest_0720_full100_rqmc030_{ttt,auto_r1,auto_r2}`
- `whest_0720_full100_sphericalrb_{ttt,auto_r1,auto_r2}`
- `whest_0720_full100_layerrescale_{ttt,auto_r1,auto_r2}`

TTT-Discover 重点检查：

- 每个 `*_ttt/puct_sampler_step_*.json`，尤其最新 step 中 `states` 的 code、value、
  construction、metadata；
- `*_ttt/sampler_states.jsonl`、`metrics.jsonl`、`agent_outputs.jsonl`、
  `gen_score_codex.jsonl`；
- 不要只看最后一个状态，要找自上轮之后出现的所有新 SHA，并保留分数最好的版本和
  方法明显不同的版本。

AutoEvolve 重点检查：

- 每个 `*_auto_r*/autoevolve_pool_step_*.json` 中的 states；
- `*_auto_r*/autoevolve_workspaces/step_*/call_*/submission.py`；
- 如有 `candidate_pool.json` 或候选子目录也一并检查；
- 忽略名称带 `_codex_home` 的镜像/缓存目录，它们不是候选产物。

initial estimator 位于：
`__RUN_ROOT__/initial_states/*/estimator.py`

控制台日志位于：
`__RUN_ROOT__/console/*.log`

## 本轮审查流程

1. 为每个发现的 estimator 规范化代码并计算 SHA-256。把本轮检查到的每个候选写入
   `__CYCLE_DIR__/candidate_inventory.jsonl`，每行至少包含：时间、family、run、step、
   source_path/state_id、sha256、full-100 adjusted score、失败数、相对 parent/initial 的
   变化、方法摘要、decision 和 reason。即使拒绝也要记。
2. 去掉已经在 ledger、registry、tracker 或旧 inventory 中出现的 SHA。对于相同代码的
   多个来源，合并 provenance，不重复验证。
3. 搜索本身已在 full-100 上运行。不要拿 state.value 的正负号猜分数；从
   construction、metrics、raw_score 或评分消息中核实 adjusted score 与 failure 数，并且
   必须确认候选的 construction 恰好包含 100 个 MLP。任何 n=10 或无法证明 n=100
   的结果都不得作为本轮提交证据。
4. 优先让以下两类进入独立 full-100 复验，但不要使用机械的固定百分比或固定分数线：
   - **显著分数改进**：相对相同 full-100 协议下 parent、initial 或已验证最佳候选，
     adjusted score 的改善明显超过评测噪声和已知 local/official 漂移；改善幅度
     越大，优先级越高；
   - **实质方法改变**：算法机制发生结构变化，而非预算/系数/seed/缓存/重构微调；其
     full-100 结果仍有值得独立复验的信号，或有合理的 hidden/private 泛化依据。
   探索阶段允许有判断空间；Codex 应结合候选数量、方法互补性和证据强弱决定本轮要验证
   多少个，而不是套用硬阈值。
5. full-100 串行执行且 BLAS/OMP 线程数固定为 1，避免影响正在运行的搜索。使用官方
   Phase 1 mini full-100、subprocess runner：

   `PYTHONPATH=__DEPS_PATH__:__REPO_ROOT__ TTT_BLACKBOX_EVAL_SOCKET=__BLACKBOX_SOCKET__ python __REPO_ROOT__/repro/aicrowd_whestbench/evaluate_public_reproduction.py <estimator.py> <unique-full100-report.json> --n-mlps 100`

6. full-100 必须 0 failure、每个 MLP 不超预算且结果可复现，才能考虑官方提交。综合判断：
   - 分数是否相对可比基线有足够大的真实改进；
   - 方法机制是否真正新颖，是否值得用官方 public-50 验证 local/official 漂移；
   - 与今天已经提交的方法是否重复，是否能提供新的信息。
   不要因为某个单一固定百分比或固定分数线自动接受或拒绝候选。
7. 若候选很多，按分数证据、方法新颖性和与现有提交的互补性排序，在当天剩余配额内
   自行决定提交数量。先确认提交 quota/认证可用；如果配额状态不明，保守地不提交并
   记录 blocker。

## 官方提交与记录流程

对于每个最终入选候选：

1. 把 exact estimator、full-100 JSON 和唯一命名的 tarball 保存下来；文件名包含 UTC
   时间、方法短名和代码 SHA 前 8--12 位，不得覆盖旧文件。
2. 使用 `PYTHONPATH=__DEPS_PATH__:__REPO_ROOT__ __DEPS_PATH__/bin/whest package --estimator <path> --output <unique.tar.gz> --yes`
   打包并验证，记录 estimator SHA 和 archive SHA。
3. 使用 `PYTHONPATH=__DEPS_PATH__:__REPO_ROOT__ __DEPS_PATH__/bin/whest submit <artifact> --description <清楚的方法/来源/本地分数> --watch --watch-timeout 1800 --json`
   提交并等待官方结果。不要打印或记录 API key。
4. 一旦得到 submission ID（即使仍 queued 或最后 failed），立即在
   `__LEDGER__` 追加一段记录，至少包括：
   - cycle 时间、submission ID/URL/status；
   - family、run、step、state ID、原始路径；
   - estimator SHA、artifact SHA、candidate/report/package 路径；
   - 搜索 full-100 与独立复验 full-100 adjusted/MSE/C-B/failure；
   - 为什么属于“显著改进”或“实质方法改变”；
   - 官方 adjusted score、secondary MSE、grading message（若已返回）；
   - 与已有最佳官方成绩的对比。
5. 按现有 workflow 将新 submission 映射追加进
   `repro_external/aicrowd_whestbench/submissions/registry.json`，必须固定实际上传包的
   `artifact_sha256`；然后运行
   `PYTHONPATH=__DEPS_PATH__:__REPO_ROOT__ python repro/aicrowd_whestbench/update_submission_tracker.py`
   刷新 canonical tracker。不要手改生成的 tracker。
6. 如果 package、submit、watch、registry 或 tracker 任一步失败，把完整错误与当前恢复
   点立即记入 ledger；不要静默重试同一 artifact，也不要因此重复提交。

## 每轮结束必须留下的笔记

无论是否提交，都必须在 `__LEDGER__` 追加本轮总结，并在
`__CYCLE_DIR__/cycle_summary.md` 写同样的详细版本：

- 实际扫描了哪些 run、step、pool/state 文件和时间范围；
- 新发现、去重后、进入 full-100、最终提交的候选数；
- 每个 shortlist 的路径、SHA、方法变化、搜索/独立复验 full-100 分数和决定；
- 所有官方 submission ID、结果或 pending 状态；
- 本轮未提交的明确原因；
- 下一轮应从哪些最新 step/SHA/时间继续。

若没有显著候选，明确写“本轮无提交”，不要为了完成任务而降低门槛。最终回复只需简洁
汇报本轮检查数、full-100 数、提交数、submission ID/成绩，以及 ledger 路径。
PROMPT
    prompt="${prompt//__CYCLE_STAMP__/$cycle_stamp}"
    prompt="${prompt//__LEDGER__/$LEDGER}"
    prompt="${prompt//__CYCLE_DIR__/$cycle_dir}"
    prompt="${prompt//__REPO_ROOT__/$REPO_ROOT}"
    prompt="${prompt//__RUN_ROOT__/$RUN_ROOT}"
    prompt="${prompt//__DEPS_PATH__/$DEPS_PATH}"
    prompt="${prompt//__BLACKBOX_SOCKET__/$BLACKBOX_SOCKET}"
    printf '%s\n' "$prompt"
}

run_cycle() {
    cycle_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    cycle_dir="$STATE_ROOT/cycles/$cycle_stamp"
    mkdir -p "$cycle_dir"

    printf '[%s] starting Codex review cycle\n' "$cycle_stamp" | tee -a "$MONITOR_LOG"
    if render_prompt | codex exec \
        --model "$CODEX_MODEL" \
        -c "model_reasoning_effort=\"$CODEX_REASONING_EFFORT\"" \
        --cd "$REPO_ROOT" \
        --dangerously-bypass-approvals-and-sandbox \
        --json \
        --output-last-message "$cycle_dir/last_message.md" \
        - 2>&1 | tee "$cycle_dir/codex_events.jsonl"; then
        printf '[%s] Codex review cycle completed\n' "$cycle_stamp" | tee -a "$MONITOR_LOG"
    else
        status=${PIPESTATUS[1]:-1}
        printf '[%s] Codex review cycle failed with status %s; inspect %s\n' \
            "$cycle_stamp" "$status" "$cycle_dir/codex_events.jsonl" | tee -a "$MONITOR_LOG"
        printf '\n## Cycle %s wrapper failure\n\n- Codex process exited with status %s.\n- Recovery log: `%s`\n' \
            "$cycle_stamp" "$status" "$cycle_dir/codex_events.jsonl" >> "$LEDGER"
    fi
}

sleep_then_run_once() {
    printf '[%s] sleeping %s seconds before next review\n' \
        "$(date -u +%Y%m%dT%H%M%SZ)" "$INTERVAL_SECONDS" | tee -a "$MONITOR_LOG"
    sleep "$INTERVAL_SECONDS"
    run_cycle
}

run_loop() {
    exec 9>"$STATE_ROOT/monitor.lock"
    if ! flock -n 9; then
        echo "another hourly review monitor already holds $STATE_ROOT/monitor.lock" >&2
        exit 1
    fi
    while true; do
        sleep_then_run_once
    done
}

case "${1:-loop}" in
    loop)
        run_loop
        ;;
    loop-now)
        exec 9>"$STATE_ROOT/monitor.lock"
        if ! flock -n 9; then
            echo "another review monitor already holds $STATE_ROOT/monitor.lock" >&2
            exit 1
        fi
        while true; do
            run_cycle
            printf '[%s] sleeping %s seconds before next review\n' \
                "$(date -u +%Y%m%dT%H%M%SZ)" "$INTERVAL_SECONDS" | tee -a "$MONITOR_LOG"
            sleep "$INTERVAL_SECONDS"
        done
        ;;
    cycle)
        exec 9>"$STATE_ROOT/monitor.lock"
        if ! flock -n 9; then
            echo "another review monitor already holds $STATE_ROOT/monitor.lock" >&2
            exit 1
        fi
        run_cycle
        ;;
    once)
        exec 9>"$STATE_ROOT/monitor.lock"
        if ! flock -n 9; then
            echo "another hourly review monitor already holds $STATE_ROOT/monitor.lock" >&2
            exit 1
        fi
        sleep_then_run_once
        ;;
    render-prompt)
        cycle_stamp="DRYRUN_$(date -u +%Y%m%dT%H%M%SZ)"
        cycle_dir="$STATE_ROOT/cycles/$cycle_stamp"
        render_prompt
        ;;
    *)
        echo "usage: $0 [loop|loop-now|cycle|once|render-prompt]" >&2
        exit 2
        ;;
esac
