# GPUMode vectoradd_v2 top1 idea 提炼

## 1. Kernel 要求摘要

- 任务是实现 FP16 矩阵/向量加法：对两个连续的 CUDA `torch.float16` 张量做逐元素 `C = A + B`，逻辑形状为 `(N, N)`，返回同形状 FP16 输出。
- 题面描述里写的是输入 `(A, B)`，但 `reference.py` 和 evaluator 实际生成并传入 `(A, B, C)`：`C` 是预分配输出，提交应写入/返回这个输出张量。
- 公开 correctness tests 覆盖 `N = 127, 128, 129, 256, 512`；leaderboard benchmarks 覆盖 `N = 1024, 2048, 4096, 8192, 16384`。
- 正确性由 PyTorch reference `output[...] = A + B` 检查，默认 `allclose` 容差；leaderboard 模式会在计时循环中反复换 seed 重新校验。
- 计时前 evaluator 会清 L2 cache，因此优化重点不是数据复用，而是单次 streaming bandwidth、coalescing、launch/包装开销和对固定 size 集合的特化。

## 2. 每个 runner top1 的关键 idea/优化策略

### A100: `top1_A100_submission_779892.py`

- 核心策略是强特化 `N = 16384` 这个最大 benchmark：硬编码总元素数 `16384 * 16384 = 2^28`，避免通用 shape 计算、尾部判断和动态循环。
- 对最大规模使用 CUDA inline extension，launch `16384` blocks、每 block `256` threads；每个 thread 通过 128-bit `uint4` 一次搬运 8 个 FP16，并用 4 个 `half2` 加法完成计算。
- 每个 thread 固定做 8 次 stride 访问，`#pragma unroll` 后形成完全展开的 streaming loop，降低循环控制和分支成本。
- 小尺寸/非 16384 输入直接退回 PyTorch `A + B`，用 correctness 保底；这是一种明显面向 leaderboard 权重的 overfit：把主要优化预算押在最大、最耗时的 case。
- 编译侧用 `-O3`、`use_fast_math`、`maxrregcount=32` 约束寄存器，目标是保持足够 occupancy 来打满 A100 的内存带宽。

### B200: `top1_B200_submission_682384.py`

- 核心策略是面向 Blackwell/B200 的通用高带宽 streaming kernel：把 `numel` 展平后按 8 个 FP16 为一组处理，不只特化单一 N。
- 主 kernel 使用 512 threads/block，每个 thread 处理一个 16B `float4` chunk；内部 reinterpret 成 4 个 `half2` 做向量化 FP16 加法，再以 16B 向量写回。
- 使用 inline PTX `ld.global.v4.b32` / `st.global.v4.b32` 明确生成 128-bit 全局内存 load/store，并通过 `prefetch.global.L2` 预取下一段，尽量贴近 B200 的 streaming bandwidth 上限。
- 对 `numel % 8 != 0` 的尾部单独发一个 scalar half tail kernel，保证 127/129 这类非 8 对齐测试也正确。
- Python 侧只传裸指针和 `numel` 给 extension，减少 Tensor wrapper 逻辑；同时在 import 时做 10 次 16384 全尺寸 warmup，意图是预热 TLB、指令 cache 和内存控制器状态。
- 编译显式指定 `-arch=sm_100a`，这是它能充分利用 B200 的关键，也让它强依赖 Blackwell 编译/运行环境。

### H100: `top1_H100_submission_639715.py`

- 核心策略是用一个稳定 Triton kernel 完成 flatten 后的一维逐元素 add：`tl.load(A + offs)`、`tl.load(B + offs)`、`tl.store(C + offs, a + b)`。
- 固定 `BLOCK_SIZE = 1024`、`num_warps = 8`，不使用 autotune；提交注释里明确指出 autotune keyed on `n_elements` 会导致多 size 编译/benchmark 超时。
- 通过 mask `offs < n_elements` 原生覆盖所有公开 test/benchmark size，不需要额外 tail kernel。
- 优化方向偏“稳定通用”：减少 PyTorch op 调度，保留单 kernel、连续 coalesced 访问，同时避免复杂 CUDA extension 和架构特化带来的编译/移植风险。
- Python wrapper 有一些 assert 和 `inspect.signature` 逻辑，说明这份 top1 并不是极致压 CPU 包装开销，而是靠 H100 上 Triton 生成的简单 bandwidth kernel 足够强。

### L4: `top1_L4_submission_607485.py`

- 核心策略非常朴素：CUDA inline extension 中一个 1D grid，每个 thread 处理一个 FP16 元素，`threads = 1024`，`blocks = ceil(numel / 1024)`。
- 使用 `AT_DISPATCH_FLOATING_TYPES_AND_HALF` 泛型发射，内部做 `C[idx] = A[idx] + B[idx]`，靠连续内存和 coalesced scalar load/store 获得基本带宽。
- 没有 half2、向量化 load/store、prefetch 或手写 PTX；优势是实现短、正确性风险低、对 size 完全通用。
- 对 L4 这类较小 GPU，单 launch 自定义 kernel 已经能避免 PyTorch eager `A + B` 的额外调度/分配开销；进一步 micro-optimization 的收益可能没有 A100/B200 上明显。
- 这份提交开启 `verbose=True`，更像“足够快且可调试”的 baseline kernel 取胜，而不是架构极限榨干。

## 3. 共享 high-level insight

- `vectoradd_v2` 本质是 memory-bound：每个元素只有一次 FP16 add，但至少要读 A、读 B、写 C；瓶颈主要是 DRAM streaming bandwidth 和 launch/框架开销，而不是算力。
- 所有 top1 都把 `(N, N)` 当成连续 1D buffer，避免二维索引开销，让 warp 内访问自然 coalesced。
- 最好的实现形态是单 kernel、无中间张量、直接写 output；大 GPU 上进一步把每个 thread 的工作扩成 8 个 half，用 16B 向量化访存和 `half2` 降低指令/索引开销。
- leaderboard 的固定 size 很重要：A100 直接为 `N=16384` 做硬编码，B200/H100/L4 虽更通用，也都围绕这些固定规模选择 block size、tail/mask 和编译策略。
- 稳定性本身也是优化：H100 放弃 autotune 来避免多 size 编译超时，B200 用 warmup 降低首轮抖动，L4 选择最少机制降低错误面。

## 4. 复现 correctness 风险和硬件依赖

- 输入协议风险：不要只按题面写 `(A, B)`；当前 evaluator 实际传 `(A, B, C)`。复现时应显式 unpack 三个张量，并返回输出。
- 尾部风险：公开 tests 包含 `127^2`、`129^2`，不是 8 的倍数。任何 `half2`/`float4`/`uint4` 向量化实现都必须有 mask 或 tail kernel；A100 的无边界特化只适合精确 `N=16384`。
- 对齐风险：`uint4`/`float4` reinterpret 假设 PyTorch contiguous CUDA allocation 至少 16B 对齐。若换成切片、非 contiguous tensor 或自定义 allocator，可能破坏性能甚至语义假设。
- 输出语义风险：当前 checker 主要检查返回的 tensor；但更稳妥的实现应写入传入的 `C` 并返回它。像 A100 fallback 那样把 `data[2]` 重新绑定为 `A + B`，在更严格的 harness 下可能不算原地写 output。
- 硬件/编译依赖：B200 版本写死 `sm_100a` 和 Blackwell PTX/prefetch 策略，不能直接拿到 A100/H100/L4 上跑；A100 版本虽然没显式 `sm_80`，但 block/stride/register 选择是按 A100 bandwidth/occupancy 调的。
- Triton 依赖：H100 版本需要 Triton 环境和兼容驱动；如果改回 autotune，可能因为多输入 size 编译和调参触发 timeout。
- 内存依赖：B200 import-time warmup 会分配 3 个 `16384 x 16384` FP16 tensor，约 1.5 GiB 显存，并运行 10 次全尺寸 kernel；在较小显存或不同评测沙箱中可能 OOM 或拖慢初始化。
- 整数范围：这些提交普遍用 `int` 保存 `numel`，当前最大 `16384^2 = 268,435,456` 安全；如果迁移到更大 size，需要检查 32-bit overflow。
