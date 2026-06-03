# GPUMode TriMul Top1 Submission Idea 提炼

## 1. Kernel 要求摘要

TriMul 题目要求实现 AlphaFold3 风格的 outgoing triangle multiplicative update forward pass。输入是 `(input, mask, weights, config)`：

- `input`: `[B, N, N, C]` float32。
- `mask`: `[B, N, N]`，用于屏蔽 pair。
- `weights`: 包含两组 LayerNorm、三组 projection/gate、最终 `to_out` 线性层权重。
- `config`: 至少包含 `dim=C` 和 `hidden_dim=H`。

计算流程可以概括为：

1. 对 `input` 最后一维做 LayerNorm。
2. 从归一化后的输入生成 `left_proj/right_proj/left_gate/right_gate/out_gate`。
3. `left = left_proj * sigmoid(left_gate) * mask`，`right = right_proj * sigmoid(right_gate) * mask`。
4. outgoing contraction：对每个 hidden channel 独立计算  
   `out[b, i, j, h] = sum_k left[b, i, k, h] * right[b, j, k, h]`。
5. 对 contraction 输出在 hidden 维做 LayerNorm，再乘 `sigmoid(out_gate)`。
6. 最后用 `to_out.weight` 投回 `C` 维，输出 `[B, N, N, C]`。

评测覆盖 `N=32..1024`、`B=1/2`、`C=128/256/384/768`、`H=128`，包含 normal/cauchy 输入和 masked/unmasked case。reference 使用 float32 PyTorch，判定容忍度为 `rtol=2e-2, atol=2e-2`，这给了 top solution 大量 fp16/Tensor Core 近似空间。

## 2. 各 runner top1 关键 idea

### A100: Triton 分段融合 + `torch.bmm`

A100 版是相对朴素但清晰的 Triton pipeline：

- 第一段 Triton kernel 做 row-wise LayerNorm，fp32 统计、fp16 写回，把输入从 `[B,N,N,C]` 压成 Tensor Core 友好的 half buffer。
- 第二段 Triton kernel 一次性算五个矩阵乘累加：left/right projection、left/right gate、out gate。它在同一个 tile 内完成 sigmoid、mask、layout scatter，把 left/right 直接写成 `[B,H,N,N]`，并保留 `out_gate`。
- 中间最重的 triangle contraction 直接交给 `torch.bmm`：把 left/right reshape 为 `B*H` 个 `N x N` 矩阵，计算 `left @ right^T`。
- 最后一段 Triton kernel 融合 hidden 维 LayerNorm、out gate、final linear projection，减少 out_norm/gated 中间张量。

核心思想是“把两端 elementwise/normalization 融起来，中间大 GEMM 交给库”。实现复杂度低，适合 A100 上快速吃到 Tensor Core。

### B200: 手写 CUDA 前后处理 + cuBLAS Tensor Core

B200 版完全走 C++/CUDA extension，尽量把大矩阵乘交给 cuBLAS：

- 自定义 LN1 kernel 按 `dim` 特化，常见 `128/256/384/768` 用 `float4` load、warp reduce、多 row per CTA，输出 fp16。
- 把 5 组 projection/gate 权重和 `to_out` 权重每次调用打包/转换到 fp16，生成 `[5H, C]` 与 `[C,H]`，避免 Python 侧多个 transpose/cast。
- 用 cuBLAS `gemm_f16` 一次完成 `x_norm @ packed_weights^T`，得到所有 projection/gate。
- 手写 `_mask_gate_lr_fuse` kernel 用 vectorized load 和 half2/float2 sigmoid 融合 `mask * sigmoid(gate)`，left/right 原地变成 contraction 输入。
- triangle contraction 用 `cublasGemmStridedBatchedEx`，batch 是 `B*H`，输出也用 fp16 存，fp32 accumulate。
- LN2 + out_gate + transpose 用一个共享内存 tile kernel 完成，随后再用 cuBLAS 做最终 `to_out` GEMM。

这版的重点不是最大程度 fusion，而是“库 GEMM 吞吐最大化 + 轻量 CUDA kernel 衔接 layout”。它把 hidden_dim 明确锁定为 128，贴合本题 benchmark。

### H100: cuBLASLt 形状缓存/择优 + 更深 layout 特化

H100 版也是 C++/CUDA extension，但相比 B200 更激进地使用 cuBLASLt：

- 强制 `TORCH_CUDA_ARCH_LIST=9.0a`，为 H100 编译。
- 为 projection GEMM、triangle contraction、final GEMM 建立 cuBLASLt descriptor cache，并给每类 shape 缓存多个 heuristic algo。
- 每个新 shape 首次出现时用 CUDA event 对候选 algo 做轻量 benchmark，之后复用最快 algo。大 workspace 约 768MB，用显存换更好的 Lt algo。
- LN1 针对 `dim=128/256/384/768` 有独立 kernel：每 CTA 处理多个 row，减少超大 `M=B*N*N` 时的 CTA 数和 launch/调度开销。
- projection 输出直接写成 `projT=[5H,M]` 的 d-major 布局，后续 pack kernel 顺序读连续 `M` 维；out_gate 不单独落地为新张量，而是在 LN2 kernel 中直接从 `projT` 第五段读取。
- pack4 kernel 同时支持 mask half/float32，并有整除 fast path，left/right 以 `[B*H, N*N]` 布局服务后续 batched contract。
- LN2 kernel 以 64 个 pair position 为 tile，跨 hidden 做规约，并直接输出 `[H,M]`，方便 final GEMM 写出 `y_T=[C,M]`，最后只做 view+permute。

H100 的 high score 主要来自“把数学改写成三次 Lt GEMM，并把每个固定 shape 的 Lt algo 选到最优”，而不是只靠手写算子。

### MI300: Triton autotune 全流程，避开 NVIDIA 专用库

MI300 版使用 Triton，整体更接近可移植 GPU kernel 设计：

- 打开 matmul TF32/半精度 reduced precision 开关，并对三个关键 Triton kernel 做 autotune。
- 第一段 `fused_ln_dual_matmul_kernel` 把 LN1 和 projection 融在一起：只算一次 LayerNorm 统计，然后用 packed `4H` 权重同时生成 left/right projection 和 gate；out_gate 另一路只在部分 program 中计算。
- 权重 packing 把 left/left_gate/right/right_gate interleave 成 `[C,4H]`，便于一个 matmul tile 后 reshape 出四个角色。
- left/right 写出时直接做 mask 和 sigmoid，并把 right 写成转置布局，减少后续 contraction 的非连续访问。
- triangle contraction 由自写 Triton `bmm_coalesced_kernel` 处理，按 `(B,H)` 分 batch，block size/warp/stage 通过 autotune 适配 MI300。
- 最后 `fused_final_kernel` 融合 hidden 维 LN、out_gate、final projection，一次 kernel 完成统计、gate 和 `to_out` dot。
- 对小 `N<100` 单独回退到 PyTorch path，避免小 shape 上自定义 kernel launch/autotune 的固定开销压过收益。

MI300 的核心是“用 Triton autotune 替代 cuBLASLt 生态”，在 AMD runner 上通过融合和布局控制拿到稳定吞吐。

## 3. 共享 high-level insight

这些 top1 的共同点非常一致：

- 不按原始 PyTorch module 逐算子执行，而是把 TriMul 拆成“LN/projection 前处理 + per-hidden batched GEMM contraction + LN/gate/final projection 后处理”。
- 中间的 `sum_k left[i,k,h] * right[j,k,h]` 是主 FLOPs，应尽量变成 `B*H` 个标准 `N x N` GEMM，让 Tensor Core/矩阵库或 Triton dot 吃满。
- 允许 fp16 中间存储、fp32 accumulate。题目容忍度较松，最优解都用 half buffer 降带宽、降显存，并只在 LN 统计和 GEMM accumulate 保持 fp32。
- 权重和 layout 比算术本身更关键：提前 pack 五组 projection/gate，选择 `[5H,M]`、`[B*H,N,N]`、`[H,M]` 等布局，目的是让后续读写连续，少做 transpose。
- Elementwise 操作要贴着生产者/消费者融合：sigmoid gate、mask、LN2、out_gate 都避免作为独立 PyTorch op 执行。
- top runner 会对固定评测 shape 做强特化：`H=128`、`C in {128,256,384,768}`、`N` 固定集合，这比泛化 kernel 更有价值。

## 4. 复现 correctness 风险与硬件依赖

- Mask 语义要特别核对。problem 的 `config` 只包含 `dim/hidden_dim`，没有 `nomask` 字段；A100 版从 `cfg.get("nomask", True)` 判断是否应用 mask，按当前题目文件原样复现 masked case 时有默认跳过 mask 的风险。
- fp16 中间结果依赖评测容忍度。cauchy 输入长尾可能放大 LN、sigmoid、GEMM 写回 fp16 的误差；若容忍度收紧或 reference 改成更严格，就可能失效。
- sigmoid 使用 fast math / `__expf` / Triton `tl.sigmoid`，和 PyTorch 精确 sigmoid 不是逐位一致，只能依赖 `2e-2` tolerance。
- B200 版硬编码 `hidden_dim == 128`，多个 kernel 和 shape check 都不支持泛化 hidden；H100 虽有 `hidden <= 256` 的 LN2 分支，但整体调优仍围绕本题 `H=128`。
- H100 版依赖 CUDA、cuBLASLt、`sm_90a` 编译目标和较大 Lt workspace；在 A100/B200 或较旧 CUDA/cuBLASLt 上 algo 选择、workspace、甚至编译都可能不同。
- B200 版依赖 NVIDIA CUDA/cuBLAS 和 `CUBLAS_COMPUTE_32F_FAST_16F`/Tensor Op math；不能直接搬到 AMD。
- MI300 版依赖 Triton 在 AMD backend 上的 autotune 表现；同一代码在 NVIDIA 上未必最优，block size、num warps、num stages 都要重新扫。
- top 解法大多每次调用重新 cast/pack 权重，这是为了 benchmark 输入权重每次变化的正确性；复现时不要跨调用缓存权重，除非能证明权重对象和内容不变。
- H100/B200 的输出多为 fp16 view/permute 返回，A100 最后输出 fp32。reference allclose 会自动转 fp32 比较，但下游如果要求 dtype 精确，需额外处理。
- 所有实现都假设输入 contiguous 或主动 `.contiguous()`；改变输入 stride、非方阵 `N x N`、或 hidden/dim 超出 benchmark 集合，需要重新检查 offset 计算和 fast path。
