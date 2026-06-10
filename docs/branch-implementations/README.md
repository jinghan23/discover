# Branch Implementation Notes

Generated from local branch/worktree state under `/opt/tiger/discover`. Base commit for comparison: `2234282`.

- Local branches documented: `53`
- Diversity branches documented: `50`
- Listed completed diversity branches: `19`
- Extra diversity branches: `31`

Important: most diversity branch refs point at the base commit; their implementations live in uncommitted worktree changes. Each per-branch file records that explicitly.

## Files

- [codex/ahc-discovery-script](codex/ahc-discovery-script.md): 新增 AHC 场景的 Codex discovery 运行脚本和评测辅助：扩展 AHC env/case runner/prompt，加入 g++ 包装、time wrapper、自循环评估脚本和 repro/run_discovery.py。
- [codex/diversity-age-balance](codex/diversity-age-balance.md): 在 PUCT 排序基础上按非 seed 状态的年龄/归档位置分桶 early/middle/recent，优先覆盖不同年龄段；seed 单独标记，缺口时回退到原 PUCT 顺序。
- [codex/diversity-ancestry-distance](codex/diversity-ancestry-distance.md): 为每个候选构造 ancestry set，并用 Jaccard overlap 衡量祖先重合度；采样时先选择祖先距离更远的 PUCT 候选，阈值不足时分阶段回退。
- [codex/diversity-api-surface-balance](codex/diversity-api-surface-balance.md): 从候选代码中识别 API surface（Triton、cuBLAS、CUDA/C++ extension、CuPy、Numba、torch.compile、torch ops、NumPy/SciPy 等），在 surface 间做父样本平衡，surface 内保持 PUCT 排序。
- [codex/diversity-archive-age-surface-balance](codex/diversity-archive-age-surface-balance.md): 新增 puct_surface_balance=archive_age，把 archive 中状态按 birth timestep 划成 seed/fresh/warm/cold，并用持久计数轮转选择低采样 surface。
- [codex/diversity-archive-attachment-balance](codex/diversity-archive-attachment-balance.md): 根据候选的直接 parent 是否仍在当前 archive/candidate 集合中分为 anchored/detached，平衡选择这两类结构位置。
- [codex/diversity-archive-slot-surface-balance](codex/diversity-archive-slot-surface-balance.md): 将 archive buffer index 映射到固定 slot surface，并在 parent_surface_mode=archive_slot_surface 下平衡不同 slot；保留 PUCT fallback 标记。
- [codex/diversity-birth-cohort-density-balance](codex/diversity-birth-cohort-density-balance.md): 按同一 birth cohort/parent cohort 的密度分类为 cohort_solo/cohort_small/cohort_dense/seed_root，优先采样历史上较少选择的 cohort density surface。
- [codex/diversity-birth-phase-surface-balance](codex/diversity-birth-phase-surface-balance.md): 按 timestep % 4 生成 phase_0..phase_3，并把 seed/unknown 单独成面；在 sampling_diversity=birth_phase_surface 时平衡出生相位。
- [codex/diversity-child-count-balance](codex/diversity-child-count-balance.md): 统计每个候选在 retained archive 中的直接子节点数量，并按 child-count bucket 做 PUCT 后处理，避免只采样同一种已扩展程度的父节点。
- [codex/diversity-complexity-balance](codex/diversity-complexity-balance.md): 为 state construction/code 计算复杂度分数并分为 simple/medium/complex；优先覆盖复杂度桶，缺口时回退 PUCT。
- [codex/diversity-delta-balance](codex/diversity-delta-balance.md): 计算候选 value 相对 parent value 的 delta，并按改进/持平/退化等 delta bucket 做平衡，避免只挑同一收益形态。
- [codex/diversity-depth-balance](codex/diversity-depth-balance.md): 根据 lineage parent 链深度划分 root/depth1/depth2_3/depth4_7/depth8_plus 等 bucket，在 PUCT 排序上增加深度覆盖。
- [codex/diversity-dissimilarity-gate](codex/diversity-dissimilarity-gate.md): 把 construction/code/observation 等内容 token 化，用 Jaccard similarity 对已选样本做 sequential gate；超过阈值的相似候选先跳过，最后回退。
- [codex/diversity-edit-distance-balance](codex/diversity-edit-distance-balance.md): 比较 child 与 parent 的 token edit（新增/删除 token）并按 edit distance bucket 采样，鼓励不同修改幅度的父节点。
- [codex/diversity-focus-slices](codex/diversity-focus-slices.md): 不是 sampler 平衡，而是 prompt 层多样性：为 GPUMode trimul 定义多个 evaluation focus slice，并按样本序号循环追加到 prompt。
- [codex/diversity-fork-cadence-balance](codex/diversity-fork-cadence-balance.md): 按同一 birth timestep/父系 fork 的节奏分类 singleton_birth/paired_birth/burst_birth/seed，平衡不同 fork cadence 的父节点。
- [codex/diversity-id-hash-lane-balance](codex/diversity-id-hash-lane-balance.md): 用 state id 的 hash 稳定映射到 id_lane_0..id_lane_3，作为与语义无关的随机 lane 平衡基线。
- [codex/diversity-island-pools](codex/diversity-island-pools.md): 引入多 island sampler 池，把状态、保存文件和父选择路由隔离到不同 island，支持按 island 采样、更新、恢复和统计。
- [codex/diversity-lineage-balance](codex/diversity-lineage-balance.md): 按 state 的 lineage family key（可配置 parent 深度）分组，在不同 lineage family 之间平衡选择，减少同一家族连续扩张。
- [codex/diversity-lineage-depth-surface-balance](codex/diversity-lineage-depth-surface-balance.md): 按 parent 链长度分类 root/shallow/middle/deep，并持久记录各 depth surface 的采样次数来做 PUCT 表面平衡。
- [codex/diversity-lineage-trend-surface-balance](codex/diversity-lineage-trend-surface-balance.md): 读取当前 value 与最近 parent_values 的趋势，分类为 no_history/flat/up_streak/down_streak/reversal/mixed，并平衡不同 lineage value trend。
- [codex/diversity-mode-forcing](codex/diversity-mode-forcing.md): prompt 层 diversity mode forcing：维护一组模式说明，按 sample index 轮转注入 prompt；同时扩展 Codex CLI 环境、日志和输出 token 相关配置。
- [codex/diversity-novelty-archive](codex/diversity-novelty-archive.md): 从 sampler archive 中选择 top/recent 候选摘要，提取函数名、kernel 名、markers、hash 等，作为 Novelty Archive 附加到 prompt 以避免近重复实现。
- [codex/diversity-observation-surface-balance](codex/diversity-observation-surface-balance.md): 根据父状态 observation 文本分类（如 traceback、timeout、empty、normal 等 profile），在 parent_balance=observation_surface 时按 observation surface 平衡。
- [codex/diversity-origin-request-slot-balance](codex/diversity-origin-request-slot-balance.md): 为每个 state 记录来源 request/sample slot（seed/unknown/slot_i），update_states 传入 metadata 后持久化，并按 origin slot 平衡父选择。
- [codex/diversity-outcome-surface-balance](codex/diversity-outcome-surface-balance.md): 从 CandidateResult 的 metrics/message/error 中抽取 outcome surface，记录到 state sidecar，并在 parent_balance=outcome_surface 时按结果类别平衡。
- [codex/diversity-parent-birth-gap-surface-balance](codex/diversity-parent-birth-gap-surface-balance.md): 按 child birth timestep 与直接 parent birth timestep 的 gap 分类 root_or_seed/gap_0/gap_1/gap_2_4/gap_5_plus/unknown，并做 surface 计数平衡。
- [codex/diversity-parent-family-width-balance](codex/diversity-parent-family-width-balance.md): 统计直接 parent 在 retained archive 中的 sibling/child 数量，将候选分为 only_child/paired_child/wide_family/root_or_seed 后平衡。
- [codex/diversity-parent-merge](codex/diversity-parent-merge.md): prompt 层多父上下文：除主 parent 外选择若干高分/近期/非近亲候选，把压缩代码块和指标合并到 prompt，供模型综合。
- [codex/diversity-parent-portfolio](codex/diversity-parent-portfolio.md): 提供 parent portfolio lanes（puct/top/recent/underexplored），按 lane 选择父节点并记录 parent_lane，让一个 batch 覆盖不同父选择策略。
- [codex/diversity-parent-selection-lag-balance](codex/diversity-parent-selection-lag-balance.md): 记录候选上次被选择的 timestep，分类 never_selected/stale/ready/cooldown，优先选择历史上采样较少或滞后的父节点。
- [codex/diversity-puct-backup-lift-surface-balance](codex/diversity-puct-backup-lift-surface-balance.md): 比较 PUCT backup 值 Q/m 与 state 当前 value 的 lift，分类 backup:unvisited/lifted/self/depressed，并平衡这些 PUCT 反馈表面。
- [codex/diversity-puct-component-dominance-balance](codex/diversity-puct-component-dominance-balance.md): 拆分 PUCT score 中 exploitation(Q/P/value) 与 exploration bonus 的相对主导性，分类 unvisited/explore/exploit/mixed 后平衡。
- [codex/diversity-puct-feedback-surface-balance](codex/diversity-puct-feedback-surface-balance.md): 按 PUCT 访问/备份反馈分类 feedback_untried/improved/flat/declined，持久记录 surface sample counts 并做平衡选择。
- [codex/diversity-puct-rank-pressure-balance](codex/diversity-puct-rank-pressure-balance.md): 新增独立 rank-pressure helper，根据候选在 PUCT 排序中的压力/相邻 rank 情况分类 surface；平衡 surface 时仍在每个 surface 内按 PUCT 顺序取样。
- [codex/diversity-puct-score-gap-surface-balance](codex/diversity-puct-score-gap-surface-balance.md): 根据候选 PUCT score 与 leader 的 gap 分类 gap:leader/near/middle/far，用 close/far frac 阈值做 surface 平衡。
- [codex/diversity-rank-volatility-balance](codex/diversity-rank-volatility-balance.md): 持久记录上一轮 PUCT rank，比较当前 rank 得到 rank:new/up/down/flat，平衡不同 rank volatility 的候选。
- [codex/diversity-recent-ancestry-trace-balance](codex/diversity-recent-ancestry-trace-balance.md): 检查最近若干 parent trace 是否仍完整保留在 archive 中，分类 root/trace_intact/prefix_gap/skip/lost，并按 trace surface 平衡。
- [codex/diversity-root-drift-balance](codex/diversity-root-drift-balance.md): 把候选与 lineage root 的 construction/code token 集合比较，按从 root 漂移程度分桶，鼓励不同 root-drift 水平。
- [codex/diversity-runtime-surface-balance](codex/diversity-runtime-surface-balance.md): 在状态 metadata 中记录 birth_eval_wall_s，按运行时分为 unknown/short/medium/long，并平衡不同 runtime surface；支持短/长分位阈值和 timeout。
- [codex/diversity-shape-profile-balance](codex/diversity-shape-profile-balance.md): 对 construction 的嵌套容器/ndarray/标量结构生成 shape profile，过滤退化 profile 后按结构形态平衡父选择。
- [codex/diversity-sibling-novelty-balance](codex/diversity-sibling-novelty-balance.md): 提取候选 token 并与同一直接 parent 的 siblings 比较，按 sibling novelty token pair/bucket 做平衡。
- [codex/diversity-sibling-rank-balance](codex/diversity-sibling-rank-balance.md): 在同一 sibling group 内按 value 排名，记录 group size、rank 和 rank bucket，并平衡不同 sibling rank 位置。
- [codex/diversity-signature-buckets](codex/diversity-signature-buckets.md): 为 state 生成结构签名：结合 construction 形态、代码 AST/regex 特征、imports/decorators/calls、env signature 等，再按 signature bucket 平衡。
- [codex/diversity-subtree-frontier-balance](codex/diversity-subtree-frontier-balance.md): 构造 retained archive 的 children map，计算每个候选子树 frontier leaf count，分类 leaf/chain/branch_2_3/branch_4_plus 并平衡。
- [codex/diversity-subtree-load-balance](codex/diversity-subtree-load-balance.md): 计算候选在 archive 内的 subtree descendant count，并按 subtree load bucket 选样，避免只扩展超大或超小子树。
- [codex/diversity-token-rarity-balance](codex/diversity-token-rarity-balance.md): 从 construction/code/observation 中抽 token，按 archive 文档频率计算 token rarity，分类 token_common/mixed/rare 后平衡。
- [codex/diversity-value-quantile-balance](codex/diversity-value-quantile-balance.md): 对有效 numeric value 做分位桶（另有 seed_or_invalid_value），在不同 value quantile 之间平衡父选择。
- [codex/diversity-visit-count-balance](codex/diversity-visit-count-balance.md): 基于 PUCT visit count n 为候选分桶，平衡 unvisited/low/medium/high 访问程度，避免只选同一访问次数层。
- [codex/diversity-whiteboard](codex/diversity-whiteboard.md): prompt/memory 层 whiteboard：在 log_path 下维护 codex_whiteboard.md，压缩保存历史候选结果摘要，并在后续 prompt 中注入。
- [codex/gpu-kernel-experiments](codex/gpu-kernel-experiments.md): GPU kernel 实验支持分支：调整 GPU Mode 环境、Codex completer 调用环境/日志/超时处理，并扩展 no-finetune 路径以跑 kernel 实验。
- [main](main.md): 主线基准分支；相对当前基准 2234282 是更早历史，不是一个独立实现分支。本文档记录它与 2234282 的关系，避免把主线历史差异误认为新的采样方法。
