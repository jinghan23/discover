# GPUMode matmul_v2 top1 submission idea 提炼

## 1. Kernel 要求摘要

- 入口是 `custom_kernel(data)`，`data = (a, b, c)`；输入由题目生成，`a` 为 `(m, k)`、`b` 为 `(k, n)`、`c` 为预分配 `(m, n)`，均在 CUDA 上且 dtype 为 `float16`。
- 正确性目标是匹配参考实现 `a @ b`。参考并不要求必须写回传入的 `c`，只要返回 tensor 的 shape 和数值正确即可；因此有的 submission 写 `c` 并返回 `c`，H100 top1 则直接返回 `a @ b`。
- 公开测试/benchmark shape 都是 16 的倍数；benchmark 包含小方阵到大矩阵：
  - 方阵：`128/256/512/1024/2048`
  - 矩形/大 shape：`1024x1536x1024`、`2048x3072x2048`、`4096x5120x4096`
- leaderboard 计时前会清 L2 cache，并且在 leaderboard 模式下每轮都会用新 seed 重新生成输入、重新做 correctness check。只对某个固定输入“记答案”没有意义。
- checker 使用 `torch.allclose(rtol=1e-5, atol=1e-8)` 对比参考。由于输出是 fp16，大数区间一个 ulp 往往远大于该 tolerance，实际风险接近“必须和 PyTorch/cuBLAS 的 fp16 结果 bit-level 非常一致”。

## 2. 各 runner top1 的关键 idea / 优化策略

### A100: cuBLASLt 固定大 shape + 固定 heuristic algo

对应文件：`top1_A100_submission_780718.py`

- 只对最大 benchmark shape `(M, N, K) = (4096, 5120, 4096)` 做特化；其他 shape 直接 fallback 到 `torch.mm(a, b, out=c)`。
- 通过 `torch.utils.cpp_extension.load_inline` 编译一个很薄的 C++ extension，直接调用 cuBLASLt，而不是手写 CUDA matmul。
- cuBLASLt descriptor 固定为 row-major fp16 输入/输出、FP32 compute，`alpha=1, beta=0`。
- 给 cuBLASLt 32 MiB workspace，请求前 8 个 heuristic algorithm，并硬编码选择第 2 个 heuristic 结果。
- 本质是“用 cuBLASLt 的算法选择能力替代 PyTorch 默认路径”，再把选择范围收窄到得分最重的最大 shape，避免为所有 shape 建复杂调度。

### B200: Python ctypes 直连 cuBLASLt + import-time autotune

对应文件：`top1_B200_submission_773912.py`

- 同样只对最大 shape `(4096, 5120, 4096)` 特化，其余 shape fallback 到 `torch.mm(a, b, out=c)`。
- 不编译 extension，直接用 `ctypes.CDLL("libcublasLt.so.12")` 调 cuBLASLt C API，减少构建依赖和 extension 编译不确定性。
- 建立固定 row-major fp16 layout，设置 FP32 compute，并提供 256 MiB workspace。
- cuBLASLt preference 打开 reduction scheme mask，允许 split-K 等更多候选算法；这通常会扩大性能搜索空间，但也放大 correctness 风险。
- import 阶段做一次面向当前 B200 机器的 autotune：
  - 先取 64 个 heuristic algorithm；
  - 对每个候选先跑正确性，要求与 `torch.mm` 的参考结果 `torch.equal`；
  - 用冷 L2 条件做 median timing；
  - 再扫部分 algo config，例如 tile id 和 CTA swizzling，继续用“正确性 + 冷 L2 timing”选最优。
- 运行时只复用 import 阶段选出的 `_best_algo`。这类方案把调参成本前置到 import，但 leaderboard timeout 足够时很划算。

### H100: 直接走 PyTorch `a @ b`，只调 cuBLAS workspace config

对应文件：`top1_H100_submission_512472.py`

- 没有自定义 CUDA/Triton/cuBLASLt 调度，`custom_kernel` 直接返回 `a @ b`。
- 设置 `CUBLAS_WORKSPACE_CONFIG=":4096:8"`，随后让 PyTorch/cuBLAS 在 H100 上选择默认 matmul 实现。
- 它也不写入传入的 `c`，而是返回新 tensor；这能通过题目 checker，因为参考也是返回 `a @ b`。
- 这个 top1 的信息量反而很强：在 H100 runner 上，PyTorch/cuBLAS 默认路径已经足够强，自定义包装、out buffer、Triton kernel 或错误的 cuBLASLt algo tuning 可能只会增加开销或引入数值偏差。

### L4: Triton 128x128x32 tile kernel 特化两个大 shape

对应文件：`top1_L4_submission_780611.py`

- 对 `(2048, 3072, 2048)` 和 `(4096, 5120, 4096)` 两个大 shape 使用 Triton kernel；其他 shape fallback 到 `torch.mm(a, b, out=c)`。
- tile 形状固定为 `128 x 128 x 32`：
  - 一个 program 计算一个 `128x128` 的 C tile；
  - K 维按 32 分块；
  - accumulator 用 fp32，最后 cast 回 fp16 存到 `c`。
- grid 是完整 tile 网格，无边界 mask；这依赖被特化的两个 shape 都能被 128 整除。
- program id 做 M 方向 grouping：
  - `2048x3072x2048` 用 `GROUP_M=8`；
  - `4096x5120x4096` 用 `GROUP_M=16`。
  目的是改善 tile 遍历顺序，让相邻 program 更容易复用 B 或 A 的 cache 行。
- 最大 shape 版本额外使用 `cache_modifier=".cg"`，并给 A/B 设置不同 eviction policy：A 更早驱逐，B 更倾向保留。这是在 L4 这类资源较紧的 GPU 上针对冷 L2 benchmark 做 cache 行为调参。

## 3. 共享的 high-level insight

- 这道题的 top1 不是在写一个通用 matmul library，而是在利用“shape 集合固定、runner 硬件固定、输入 dtype/layout 固定”的 benchmark 结构做特化 dispatch。
- 最关键的优化对象是大 shape，尤其是 `(4096, 5120, 4096)`；小 shape 和中等 shape 多数 submission 直接交给 `torch.mm`，因为它们对总分影响较小，且自定义 kernel 的 launch/JIT/数值风险更不划算。
- 最强路径通常不是手写完整 GEMM，而是选择合适层级：
  - A100/B200：直接控制 cuBLASLt algo、workspace、tile/swizzle 等内部策略；
  - H100：完全委托 PyTorch/cuBLAS 默认选择；
  - L4：当库默认路径不够理想时，用 Triton 做少数 shape 的固定 tile kernel。
- 评测每轮清 L2，因此缓存热启动收益不可靠。B200 的 autotune 显式模拟冷 L2，L4 的 grouping/cache modifier 也围绕冷 cache 条件优化。
- correctness 比表面上的 `allclose` 更严格：fp16 输出下，很多算法只要 reduction 顺序导致 1 ulp 差异就可能失败。所以 B200 在 autotune 中用 `torch.equal` 过滤 algo，是很重要的防守动作。

## 4. 复现时需要注意的 correctness 风险和硬件依赖

- cuBLASLt algorithm、heuristic 顺序、tile id、swizzle、workspace 需求都依赖 GPU 架构、CUDA/cuBLASLt 版本和 driver。A100 的“heuristic idx=2”尤其脆弱，换 CUDA 版本后不一定仍是同一个快且正确的算法。
- B200 方案依赖 `libcublasLt.so.12`、ctypes 中手写的结构体布局、CUDA 12 ABI、当前 stream 获取方式以及足够的 256 MiB workspace；迁移时要先在目标机器重新 autotune。
- 打开 split-K/reduction scheme 后，性能可能提升，但 reduction 顺序可能导致 fp16 输出与 PyTorch 参考不一致。复现时必须在目标硬件上对候选 algo 做 correctness 过滤，最好像 B200 一样先要求 bit-exact。
- Triton L4 kernel 没有边界 mask，只能安全用于代码里显式 guard 的两个 128 整除 shape。若 benchmark 增加“只是 16 的倍数但不是 128 的倍数”的隐藏 shape，必须 fallback 或补 mask。
- Triton 的 `cache_modifier`、eviction policy 和最佳 `GROUP_M` 都是硬件相关调参；在 A100/H100/B200 上未必同向收益。
- H100 的 `a @ b` 方案依赖当前 PyTorch/cuBLAS 默认 matmul 的选择。升级 PyTorch/CUDA、改变 `CUBLAS_WORKSPACE_CONFIG` 或改成 `torch.mm(..., out=c)` 都可能改变速度和数值路径。
- 所有方案都默认输入为 CUDA contiguous row-major fp16。若复现环境传入非 contiguous tensor、不同 dtype，或要求一定原地写 `c`，这些 top1 submission 的假设都需要重审。
