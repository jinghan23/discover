# GPUMode vectorsum_v2 top1 submission idea 提炼

## 1. Kernel 要求摘要

- 任务是实现一维 `torch.float32` 向量求和：输入协议实际为 `(input_tensor, output_tensor)`，返回一个标量 tensor，数值等于 `sum(input_tensor)`。
- 题面写输入来自标准正态分布；`reference.py` 实际还会对随机数据乘随机 scale、加随机 offset，因此输出通常带有很大的均值项，不能假设和接近 0。
- 参考实现先把输入转成 `float64` 做 reduction，再转回 `float32`；checker 使用默认 `allclose(rtol=1e-5, atol=1e-8)`。这允许一定 float32 归约误差，但不是逐位任意误差。
- 公开 correctness tests 覆盖小尺寸 `1023/1024/1025/2048/4096`；leaderboard benchmarks 覆盖 `1,638,400` 到 `52,428,800` 六个固定尺寸。
- evaluator 计时前会清 L2 cache；leaderboard 模式还会每轮换 seed 重新生成输入并重新校验。因此优化重点是冷 cache 下的 streaming bandwidth、kernel launch 次数、框架/分配开销，以及对固定 size 集合的特化。

## 2. 每个 runner top1 的关键 idea / 优化策略

### A100: 最大 benchmark size 的 CUDA hard-code fast path

对应文件：`top1_A100_submission_779823.py`

- 只为最大 benchmark `N = 52,428,800` 写手工 CUDA fast path；其它 size 直接 fallback 到 `data[0].sum()`，用 PyTorch 保 correctness。
- fast path 硬编码 `2048` blocks、`256` threads/block、每 thread 读取 `25` 个 `float4`。刚好覆盖 `2048 * 256 * 25 * 4 = 52,428,800` 个 float，完全去掉动态循环边界和 tail 判断。
- 每个 thread 在寄存器内累加自己的 100 个元素；block 内先用 warp shuffle 做 warp-level reduction，再让 8 个 warp leader 写共享内存，最后由 warp 0 汇总 block sum。
- block 级结果直接 `atomicAdd` 到输出标量，总共只有 2048 次全局原子加。A100 上 float atomic 性能足够好，省掉第二个 reduction kernel 的 launch。
- 提交用 `torch.utils.cpp_extension.load_inline` 编译 CUDA extension，并用 `-O3/-use_fast_math`。这是典型 leaderboard overfit：把实现复杂度押在耗时最大的固定输入上。

### B200: Triton 单 kernel partial + last-block final reduction

对应文件：`top1_B200_submission_755317.py`

- 使用一个 Triton kernel 完成两阶段逻辑：每个 program 先对一段连续 block 做 `tl.sum`，把 partial sum 写入预分配全局 `_PARTS`；随后通过全局 `_COUNTER` 判断最后完成的 program，由它读取所有 partial 并写最终输出。
- 这个 last-block pattern 避免了常规“第一阶段 partial + 第二阶段 final reduce”的第二次 kernel launch，也避免每次调用动态分配 partial buffer。
- 针对六个 benchmark size 建了参数表：最小 benchmark 用 `BLOCK_SIZE=16384, num_warps=16`，其余多用 `BLOCK_SIZE=8192`，并按 size 调 `num_warps=4/8`。final reduce 的 `FINAL_BLOCK` 取 `n_blocks` 的 next power of two，方便单个 program 一次向量化加载所有 partial。
- load 使用 `eviction_policy="evict_first"`，符合冷 L2 streaming reduction 的访问模式：数据只读一次，不值得长期驻留 cache。
- 相比 A100/H100 的最大 size 特化，B200 版本对所有 benchmark size 都走同一个单 kernel fast path；小测试也能由 mask 覆盖。

### H100: ctypes 编译 CUDA .so，最大 size 单 kernel槽位归约

对应文件：`top1_H100_submission_612491.py`

- 只在 `N = 52,428,800`、float32、contiguous、输出为单元素且指针至少 8B 对齐时走 fast path；其它情况 fallback 到 `x.sum(dtype=torch.float32)`。
- Python 侧用 `ctypes` 调本地 nvcc 编译出的 shared library，按当前 GPU capability 生成 `sm_xx` 代码，并缓存到 `/tmp`。这比 PyTorch extension wrapper 更薄，但依赖本机 nvcc/CUDA 工具链。
- CUDA fast path 固定 `NUM_BLOCKS=4288`、`BLOCK_THREADS=1024`、每 thread 做 2 次向量 load。16B 对齐走 `float4` path，否则走 `float2` path。
- 每个 block 可通过 grid-stride 处理多个 tile，先在寄存器里累加多个 lane，再用 warp shuffle 和共享内存得到 block sum。
- 全局汇总不是所有 block 直接 atomic 到输出，而是 atomic 到 4 个 `float2` scratch slots；最后完成的 block 通过 `count` 判断后，把 4 个 slot 加总、写 output，并清空 scratch。这样把全局原子热点从单标量拆成 4 个槽，降低 contention。

### L4: 通用 Triton 多级 reduction，少冒险换稳定

对应文件：`top1_L4_submission_66749.py`

- 使用朴素而通用的 Triton 分层归约：第一层每个 program 对 `BLOCK_SIZE=1024` 个元素做 `tl.sum`，写入 `partial_sums`。
- 当 partial 数量仍大于 1024 时，继续用同一个 kernel 递归压缩；对本题 benchmark，通常是“原输入 -> partial -> 很小 partial”两层。
- 最后若 partial 数量小于等于 128，则发一个单 program atomic kernel 写到临时 `final_output`；否则再压一层后交给 `next_level.sum()`。
- 该实现没有硬编码某一个 benchmark size，也没有 CUDA extension、向量 reinterpret 或全局 counter trick；代价是多次 kernel launch 和中间 tensor 分配。
- 对 L4 这种较小 GPU，稳定的 coalesced streaming + 简单 Triton reduction 已经足够强；过度特化可能被编译、launch、分配或 correctness 风险抵消。

## 3. 共享的 high-level insight

- `vectorsum_v2` 是典型 memory-bound reduction：每个元素只参与一次加法，真正瓶颈是冷 cache 下从 HBM/显存把数据流过来，以及最后少量 partial 的汇总方式。
- Top1 解法都避免 PyTorch eager 的通用 reduction 路径在关键 case 上产生额外调度开销；区别在于 A100/H100 押最大 size 特化，B200 押单 kernel last-block 汇总，L4 押通用稳定多级归约。
- 固定 benchmark size 是核心信息。A100/H100 直接把最大 `52,428,800` 当成主战场；B200 虽覆盖所有 benchmark，也为每个公开 size 调了 block size/warp 数；L4 的 `1024` block 则是保守通用点。
- 单 kernel 汇总很有价值：A100 用 2048 次 `atomicAdd(out)`，B200 用 counter 让最后一个 program 做 final reduce，H100 用 4 个 scratch slots 降低原子热点，本质都是在减少第二次 launch。
- 归约顺序不追求和 float64 reference 完全一致，而是利用题目 `allclose` 容差、输入 offset 通常较大、输出为 float32 的条件，让 float32 tree/atomic reduction 通过校验。

## 4. 复现 correctness 风险和硬件依赖

- 数值风险：reference 是 float64 sum 后转 float32，而 top1 fast path 都是 float32 分块/原子归约。换 seed、offset 接近 0、scale 较大或容差收紧时，非结合加法的误差可能暴露。
- 原子顺序风险：A100 的全局 `atomicAdd`、H100 的 scratch slot atomic、B200 的 last-block 读取 partial，都依赖硬件上实际执行顺序能落在容差内；它们不提供 bit-exact deterministic sum。
- B200 的 global counter pattern 对内存可见性敏感：partial store 之后用 relaxed atomic 计数，最后一个 program 读取所有 partial。若 Triton/硬件内存语义变化、kernel 异常导致 counter 未复位，可能读到旧 partial 或死在错误状态。
- A100/H100 最大 size fast path 没有 tail 处理，安全性来自硬编码 `N == 52,428,800`。迁移到其它 size 必须重算 grid、每线程元素数和尾部 mask，不能直接放宽 guard。
- 向量化对齐风险：A100 直接 reinterpret 为 `float4`，隐含 PyTorch contiguous allocation 至少 16B 对齐；H100 显式检查 16B/8B 对齐并在 `float4/float2` 间切换。若输入来自切片、非 contiguous view 或特殊 allocator，这些假设要重审。
- 输出语义风险：题目传入了预分配 `output_tensor`，但 checker 主要看返回值。A100/H100/B200 fast path 写 output，L4 和 fallback 路径常返回新 tensor。若换成要求原地写 output 的 harness，需要统一返回/写入语义。
- 硬件依赖：A100 版本的 block/grid/atomic 取舍按 A100 调；H100 版本依赖 nvcc、CUDA shared library 加载、`float2`/`float4` atomicAdd 支持和 Hopper 原子性能；B200 版本依赖 Triton 在 Blackwell 上的 program 调度和 atomic/counter 成本；L4 版本最可移植但 launch/分配开销也最高。
- 编译/运行环境风险：A100 用 `load_inline`，H100 用 `ctypes + nvcc + /tmp` 缓存，二者在缺少编译器、CUDA_HOME 配错、只读临时目录或架构 flag 不匹配时会 fallback 或失败；复现前应先确认实际 fast path 被触发。
