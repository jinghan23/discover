# WhestBench Sampling 方法调研

你的抽象基本正确，但可以再精确一步：

> 这些方法不是简单地让少量样本“看起来更像分布”，而是选择一个有限离散测度，使它对**与目标函数相关的某一类测试函数**尽可能像真实分布。

设目标为

[
\mu_W=\mathbb E_{X\sim P}[F_W(X)],
\qquad P=\mathcal N(0,I_{256}),
]

其中 (F_W(X)\in\mathbb R^{256}) 是固定网络 (W) 的最后一层输出。所有方法最终都在构造

[
\hat\mu_W=\sum_{i=1}^{N}a_iF_W(x_i),
\qquad
\nu_N=\sum_{i=1}^{N}a_i\delta_{x_i},
]

希望有限离散分布 (\nu_N) 能代替真实分布 (P)。

误差就是：

[
\hat\mu_W-\mu_W
===============

\int F_W(x),d(\nu_N-P)(x).
]

因此不存在脱离函数类的“绝对好样本”。不同方法其实是在优化不同的函数类：

* German：让低阶多项式看到的分布正确；
* Sphere：利用齐次性，把径向随机性直接解析积分掉；
* QMC：让低维、低频、较平滑的函数看到的覆盖更均匀；
* spherical design：让低阶球谐多项式的积分完全正确；
* kernel quadrature：对某个 RKHS 函数类优化；
* control variate：先解释掉容易预测的函数成分，只采样剩余残差；
* importance sampling：把样本放到对当前 (F_W) 更重要的区域；
* multifidelity：用大量廉价近似计算换取少量真实网络计算。

下面按这个统一视角做一个 survey。

---

# 一、German 和 Sphere 在整个文献地图中的位置

## German：moment matching cubature

German 的核心是：

[
\frac1N\sum_i x_i=0,
\qquad
\frac1N\sum_i x_ix_i^\top=I.
]

其中 antithetic sampling 精确匹配一阶矩，whitening 精确匹配二阶矩。仓库将方法概括为 antithetic sampling、input whitening、folded whitening 和 half-covariance。([GitHub][1])

考虑任意二次函数

[
q(x)=c+a^\top x+x^\top Bx.
]

那么满足上述条件的有限点集有：

[
\frac1N\sum_iq(x_i)
===================

# c+\operatorname{tr}(B)

\mathbb E_{X\sim\mathcal N(0,I)}q(X).
]

所以忽略后续非均匀 radial weights 的细节，German 的等权核心可以理解为：

[
\boxed{\text{一个对所有一阶、二阶高斯多项式完全精确的随机 cubature rule}}
]

它不是简单“分布更均匀”，而是主动把低阶积分误差设为零。

## Sphere：Rao–Blackwellization / conditional Monte Carlo

写成极坐标：

[
X=RU,
\qquad
R\sim\chi_{256},
\qquad
U\sim\operatorname{Unif}(S^{255}),
\qquad R\perp U.
]

无偏置 ReLU 网络满足正齐次性：

[
F_W(RU)=R F_W(U),\qquad R\geq0.
]

因此：

[
\mathbb E[F_W(X)]
=================

\mathbb E[R],
\mathbb E[F_W(U)].
]

它不是让半径采得更均匀，而是直接将半径从随机变量变成精确常数：

[
\boxed{\text{不再估计径向期望，只估计方向期望}}
]

这属于 conditional Monte Carlo / Rao–Blackwellization。现代 preintegration 文献进一步研究了如何选择最值得解析积分的线性方向，并证明对 scrambled-net RQMC，preintegration 同样不会增大方差。([应用数学学会][2])

---

# 二、第一大类：分层、对偶、正交和矩匹配

这类方法不需要真正理解 (F_W)，主要通过设计样本之间的依赖关系减少随机波动。

## 1. Latin hypercube sampling

Latin hypercube sampling，LHS，把每个坐标轴的边缘分布分成 (N) 个等概率区间，并保证每个区间恰好采一次。McKay、Beckman 和 Conover 在 1979 年的 Technometrics 论文中证明，对一类估计器，LHS 的方差优于简单随机采样。([泰恩德在线][3])

进一步的 orthogonal-array Latin hypercube 不只保证每个一维边缘均匀，还保证每个低维坐标子集都得到分层覆盖。Tang 在 JASA 1993 年的工作中证明，强度为 (r) 的 OA-LHS 会对所有 (r) 维边缘进行分层。([泰恩德在线][4])

但对本题，原始 LHS 未必很强，因为：

* 高斯分布和网络权重具有旋转结构；
* LHS 是坐标轴对齐的；
* 网络真正敏感的方向通常不是标准基 (e_1,\ldots,e_{256})。

因此更合理的是：

[
X=Q\Phi^{-1}(U_{\rm LHS}),
]

其中 (Q) 是随机或网络相关的正交矩阵。

## 2. Antithetic sampling 的一般化

最简单的 antithetic pair 是：

[
x,,-x.
]

更一般地，可以选择一组保持目标分布不变的变换：

[
Q_1,\ldots,Q_m\in O(256),
]

然后使用轨道：

[
Q_1x,\ldots,Q_mx.
]

因为各向同性高斯满足

[
Q_jX\overset d=X,
]

每个样本的边缘分布仍然正确，但可以设计 (Q_j) 使输出之间负相关。

AISTATS 2019 的 differentiable antithetic sampling 直接学习能匹配目标矩的相关样本；ICML 2019 的 adaptive antithetic sampling 则学习降低方差的相关结构，同时保持估计无偏。([Proceedings of Machine Learning Research][5])

对 ARC 问题，一个自然但尚需实验验证的方向是：

[
\boxed{\text{用网络权重学习比 }-I\text{ 更好的正交 antithetic transformations}}
]

例如根据前几层的活跃方向，寻找 (Q) 使

[
\operatorname{Cov}\big(F_W(X),F_W(QX)\big)
]

尽可能负。

## 3. Orthogonal Monte Carlo

Orthogonal MC 不让方向独立抽取，而是让同一批方向尽量正交。ICML 2019 的工作统一了多种 orthogonal MC 方法，并为部分近似正交构造给出了统计保证。([Proceedings of Machine Learning Research][6])

如果使用单位球面方向，一个正交矩阵的 256 行 (u_i) 自动满足：

[
\frac1{256}\sum_{i=1}^{256}u_iu_i^\top
======================================

\frac1{256}I.
]

所以一个完整正交块就是 unit-norm tight frame，能够精确匹配球面二阶矩。

这可以理解为 Sphere 方法对应的“球面 whitening”：

[
|u_i|=1,
\qquad
\frac1N\sum_i u_iu_i^\top=\frac1{256}I.
]

---

# 三、第二大类：Quasi-Monte Carlo 和 effective dimension

## 1. QMC 的基本目标

普通 MC 的点会聚团、留空；QMC 用低差异序列强制更均匀地覆盖空间，例如 Sobol、Halton 和 lattice rules。

随机化 QMC，RQMC，尤其是 Owen scrambling，同时保留：

* 无偏性；
* 可以通过多次 scramble 估计误差；
* 比 iid MC 更均匀的覆盖。

Owen 1997 年证明，对任意平方可积函数，scrambled-net quadrature 的方差渐近地比普通 MC 的 (1/N) 更快趋于零；其多分辨率解释是，某些低维、低频成分根本不贡献抽样方差。([应用数学学会][7])

QMC 已经在机器学习中用于降低 Monte Carlo 梯度方差。例如 ICML 2018 的 QMC Variational Inference 用低差异点替代 iid 样本，在其任务上获得更稳定的梯度和更快收敛。([Proceedings of Machine Learning Research][8])

## 2. 256 维为什么未必太高，也未必不高

QMC 的关键通常不是名义维度，而是 **effective dimension**。

如果

[
F_W(x)\approx g(a_1^\top x,\ldots,a_r^\top x),
\qquad r\ll256,
]

那么即使输入有 256 维，QMC 仍可能主要面对一个低维积分。

对于 Lipschitz ridge functions，有研究表明其 mean dimension 可以在环境维度增长时保持有界；preintegration 还可能把 mean dimension 从随维度增长降至常数级。([应用数学学会][9])

因此对本题，单纯做

[
x_j=\Phi^{-1}(u_j)
]

不够，应该考虑：

[
x=Q\Phi^{-1}(u),
]

其中 (Q) 把 Sobol 的前几个高质量坐标对齐到网络的 active subspace。

Active-subspace preintegration 和 conditional QMC 文献正是在做这种“让最重要的随机性集中到前几个坐标”的工作。([应用数学学会][2])

## 3. ReLU 对 QMC 的限制

本题中的 (F_W) 是连续、分片线性、但在大量激活边界上不可微。

所以不能直接期待光滑函数上接近 (N^{-1}) 甚至更高阶的漂亮速率。对不连续函数的 RQMC 理论显示，收敛率提升强烈依赖所谓 irregular dimension；环境维度高且边界方向复杂时，优势可能非常有限。([应用数学学会][10])

虽然 ReLU 网络输出本身连续，比 indicator 好，但它仍然有大量 kink。因此我的判断是：

> RQMC 值得作为强 baseline，但需要 active rotation、preintegration 或 control variate 配合；裸 Sobol→Gaussian 很可能不是最终答案。

---

# 四、第三大类：Cubature、spherical design 和 moment compression

这类方法与 German 最接近：选择有限点和权重，使某些函数的积分完全正确。

## 1. Spherical (t)-design

一个 spherical (t)-design 是球面有限点集，使所有次数不超过 (t) 的多项式都被精确积分。

2026 年 SIAM Journal on Discrete Mathematics 的 fixed-strength spherical designs 工作研究了固定 (t)、维度增大时所需的点数，并建立了 spherical design 与 Gaussian design 的显式联系；signed designs 可以使用比简单自由度计数更少的点。([应用数学学会][11])

对 Sphere 路线：

* (\pm u) 自动消除所有奇数阶球谐成分；
* tight frame 精确匹配二阶矩；
* 更高阶 spherical design 可进一步消除低阶球谐误差。

但维度增长很快。对于 (d=256)，仅二阶 trace-free quadratic harmonics 就有

[
\frac{256\cdot257}{2}-1=32,895
]

个自由度。

所以构造“完整高阶 design”通常太贵，更现实的是只匹配：

* 网络相关方向上的低阶矩；
* 低秩 quadratic forms；
* 少量重要 spherical harmonics。

2026 年的 sphere QMC-design 文献给出了 Sobolev 函数类上的最坏情况误差

[
O(N^{-s/d}).
]

在这里球面维度是 (255)，所以如果只依靠通用平滑性，这个指数很弱；必须利用低有效维度或网络特有结构。([Springer Link][12])

## 2. Monte Carlo cubature / Tchakaloff compression

这一方向非常符合你说的：

> 先产生大量廉价候选样本，再把它们压缩成少量“具有相同统计效果”的代表点。

假设选了一组可廉价计算、且真实期望已知的特征：

[
\phi(x)=
\begin{bmatrix}
\phi_1(x)\
\vdots\
\phi_m(x)
\end{bmatrix},
\qquad
\mathbb E_P[\phi(X)]=b.
]

先产生 (M\gg N) 个候选点，再寻找少量点和正权重，使：

[
\sum_{i=1}^{N}a_i\phi(x_i)=b,
\qquad
a_i\geq0,\quad\sum_i a_i=1.
]

Monte Carlo cubature 文献证明，当可以从目标分布采样并知道测试函数的精确均值时，可以通过线性规划从随机候选集中构造这样的 cubature rule；它也可以被理解为数据压缩。([Springer Link][13])

这可能是本题最直接的下一代 German 方法。

### 一个特别适合 ARC 的构造

对第一层第 (j) 个神经元：

[
\phi_j(x)=\operatorname{ReLU}(w_j^\top x).
]

由于

[
w_j^\top X\sim\mathcal N(0,|w_j|^2),
]

其精确均值是：

[
\mathbb E[\phi_j(X)]
====================

\frac{|w_j|}{\sqrt{2\pi}}.
]

所以可以：

1. 产生 (M) 个候选高斯点；
2. 只运行第一层，获得 (M\times256) 个第一层激活；
3. 从中选出 (N) 个带权点，使 256 个第一层神经元的均值同时精确匹配；
4. 只让选中的 (N) 个点继续运行剩余 31 层。

总层计算量大致为：

[
M\cdot1+N\cdot31,
]

而不是：

[
M\cdot32.
]

例如 (M=5N)，代价为 (36N) 个 layer-equivalents，只比直接运行 (N) 个样本的 (32N) 多约 12.5%，但选择阶段看过了五倍的候选输入。

还可以匹配：

* 第一层激活的低秩协方差；
* 第一层 PCA 投影；
* 少数第二层 cheap proxy；
* 网络相关的 Hermite 特征。

这类“在浅层空间压缩、只让代表点继续前向”的方法，我认为非常值得实验。

---

# 五、第四大类：Repulsive sampling、support points 和 kernel methods

## 1. DPP：让点互相排斥

Determinantal point processes，DPP，通过负相关让样本彼此排斥，减少聚团。

Annals of Applied Probability 2020 的理论结果对某类光滑积分给出：

[
\operatorname{RMSE}
===================

O\left(
N^{-(1+1/d)/2}
\right).
]

([arXiv][14])

但在 (d=256) 时：

[
\frac{1+1/256}{2}
\approx0.50195.
]

相对普通 MC 的 (0.5)，纯维度理论上的指数收益极小。

因此：

[
\boxed{\text{通用的 256 维 DPP 不太可能是首选}}
]

除非 DPP 是在低维 active representation 中构造，或者其核专门根据网络设计。AISTATS 2020 的 DPPMC 正是把 DPP 用于高维非各向同性采样，并把它与 orthogonal MC 联系起来。([Proceedings of Machine Learning Research][15])

## 2. Support points

Annals of Statistics 2018 的 support points 通过最小化 energy distance，把连续分布压缩为少量代表点。论文证明这些点收敛到目标分布，并在一大类积分问题中获得优于普通 Monte Carlo 的误差率；也将其用于压缩昂贵模拟和 MCMC 样本。([Scholars@Duke][16])

它的优点是：

* 不需要指定具体积分函数；
* 直接优化点集对分布的代表性；
* 可以先生成点再运行昂贵模型。

问题仍然是 256 维距离集中。Projected support points 尝试通过稀疏投影结构缓解高维问题。([arXiv][17])

## 3. Kernel herding 和 kernel quadrature

Kernel quadrature 不要求点集对所有函数都好，而是假设 (F) 属于某个 RKHS。它最小化：

[
\operatorname{MMD}_K^2(\nu_N,P)
===============================

\iint K(x,x'),d(\nu_N-P)(x)d(\nu_N-P)(x').
]

Kernel herding 用 Frank–Wolfe 方法逐点选择样本；AISTATS 2015 的 sequential kernel herding 在其粒子过滤实验中优于随机和 QMC 点。([Proceedings of Machine Learning Research][18])

ICML 2017 的 kernel quadrature 工作强调，采样分布和点集选择对误差常数影响极大，并在其实验中通过自适应采样将误差降低多个数量级。([Proceedings of Machine Learning Research][19])

### 对随机 ReLU 网络特别有趣的 NNGP-kernel 方案

这是我认为最值得深入推导的方向之一。

设网络权重本身来自挑战的随机网络分布。定义网络函数族的协方差核：

[
K(x,x')
=======

\mathbb E_W
\left[
F_W(x)^\top F_W(x')
\right].
]

对于一个 quadrature rule (\nu_N)，对网络随机性平均后的积分 MSE 为：

[
\begin{aligned}
\mathbb E_W
\left[
\left|
\int F_W,d(\nu_N-P)
\right|^2
\right]
&=
\iint
K(x,x'),
d(\nu_N-P)(x)
d(\nu_N-P)(x').
\end{aligned}
]

右边正是 kernel discrepancy / MMD。

因此：

[
\boxed{
\text{最小化 NNGP kernel MMD 的点集，等价于最小化随机网络族上的平均积分误差}
}
]

无限宽随机深度网络对应于可递归计算的 Gaussian-process kernel；ReLU 的 arc-cosine kernel 以及深层 NNGP 递归分别在 NeurIPS 2009 和 ICLR 2018 的工作中系统研究。([NeurIPS 会议记录][20])

这给出一个很自然的方案：

1. 从网络生成分布推出 depth-32 ReLU NNGP kernel；
2. 在 Gaussian 空间或球面上做 kernel herding / Bayesian quadrature；
3. 得到一个对“随机深 ReLU 网络函数类”最优的通用点集；
4. 再根据当前具体权重做少量 instance-specific 修正。

它比通用 RBF kernel 更符合比赛的函数分布。

需要注意：宽度只有 256，不是无穷宽，所以 NNGP 只提供 prior-average proxy，不能完全捕获具体网络的有限宽异常。

## 4. Kernel thinning

COLT 2021 的 kernel thinning 从 (n) 个点中压缩到 (\sqrt n) 个点，同时在指定 RKHS 中保持与原 (n) 点集相近的最坏情况积分误差；同样大小的 iid 子样本通常误差更大。([Proceedings of Machine Learning Research][21])

这又对应一个直接可用的框架：

[
\text{先产生 }M\text{ 个廉价候选}
\rightarrow
\text{kernel thinning 到 }N
\rightarrow
\text{只运行这 }N\text{ 个完整网络}.
]

核可以选择：

* Gaussian kernel；
* first-layer activation kernel；
* NNGP kernel；
* 某个浅层隐藏表示上的 kernel。

---

# 六、第五大类：Control variates——不改善点，而是让剩余函数更容易积分

这是从“少量样本达到更多样本效果”角度最有潜力的一类。

构造一个廉价近似 (G_W(x))，其期望可精确或廉价计算：

[
\mathbb E[F_W(X)]
=================

\mathbb E[G_W(X)]
+
\mathbb E[F_W(X)-G_W(X)].
]

估计器为：

[
\hat\mu
=======

\mathbb E[G_W(X)]
+
\frac1N\sum_i
\left[
F_W(x_i)-G_W(x_i)
\right].
]

如果 (G_W) 能解释 (F_W) 的大部分变化，那么残差方差远小于原输出方差：

[
\operatorname{Var}(F_W-G_W)
\ll
\operatorname{Var}(F_W).
]

这等价于同样 (N) 个样本拥有更大的“有效样本倍数”。

AISTATS 2016 的 control functionals 把函数逼近和 QMC 结合起来；ICML 2023 的 vector-valued control variates 则允许多个相关积分共享信息，这与本题同时估计 256 个输出神经元高度吻合。([Proceedings of Machine Learning Research][22])

## 1. Gaussian Hermite control variates

高斯分布下最自然的正交基是 Hermite polynomials。

选择网络相关方向 (a_k)，使用：

[
\phi_{k,m}(x)
=============

H_m(a_k^\top x),
\qquad
\mathbb E[\phi_{k,m}(X)]=0,\quad m\geq1.
]

拟合：

[
F_W(x)
\approx
c+\sum_{k,m}B_{k,m}\phi_{k,m}(x).
]

然后估计：

[
\hat\mu
=======

\frac1N\sum_i
\left[
F_W(x_i)
--------

\sum_{k,m}B_{k,m}\phi_{k,m}(x_i)
\right],
]

因为所有非零阶 Hermite 特征的真实期望都是零。

方向 (a_k) 可以来自：

* 第一层权重行；
* 第一层或第 8 层 active subspace；
* 权重乘积的主奇异方向；
* pilot 样本的 Jacobian covariance。

German 已经让全空间的一、二阶成分近似不产生误差，因此下一步更可能有价值的是：

[
m=3,4
]

的少量、网络对齐 Hermite 成分，而不是继续增加通用二阶约束。

## 2. Spherical-harmonic control variates

如果先解析积分 (R)，方向积分最自然的基是 spherical harmonics。

ICML 2024 的 Spherical Harmonics Control Variates 在球面积分中用球谐函数作为 control variates，并证明了优于普通 MC 的收敛性质；在其特定 Gaussian dependency 条件下还存在 no-error 性质。([Proceedings of Machine Learning Research][23])

这几乎是 Sphere 路线的直接文献对应：

[
\mathbb E[F_W(X)]
=================

\mathbb E[R]
\left{
\frac1N\sum_i
\left[
F_W(U_i)-B\Phi(U_i)
\right]
\right},
]

其中：

[
\mathbb E_U[\Phi(U)]=0.
]

为了避免 32,895 个完整二阶球谐特征，可以只用：

* 低秩 quadratic harmonics；
* 与第一层权重方向对齐的 zonal harmonics；
* 从多个训练网络学习出的共享低维 harmonic basis。

## 3. Meta-learned control variates

如果可以生成很多同分布随机网络，那么每个网络都是一个相关积分任务：

[
W\mapsto \mathbb E_X[F_W(X)].
]

UAI 2023 的 Meta-CVs 研究了大量相关积分任务、但每个任务只有极少样本的场景，并通过跨任务共享信息降低方差。([Proceedings of Machine Learning Research][24])

在 ARC 中可以考虑学习：

[
B(W)=\text{由权重决定的 control-variate 系数},
]

而 control basis 仍选用期望已知的 Hermite 或 spherical-harmonic 特征。

这样最终仍可保持无偏：

[
\hat\mu_W
=========

\frac1N\sum_i
\left[
F_W(x_i)-B(W)\phi(x_i)
\right],
\qquad
\mathbb E[\phi]=0.
]

为避免用同一批小样本拟合系数造成有限样本偏差，应使用：

* 独立 pilot；
* sample splitting；
* cross-fitting。

---

# 七、第六大类：Multifidelity / multilevel Monte Carlo

这里的核心是：

> 不一定每个样本都要运行完整的 32 层高保真网络。

设：

* (F_W(x))：完整 32 层网络；
* (G_W(x))：廉价近似网络。

则：

[
\mathbb E[F_W]
==============

\mathbb E[G_W]
+
\mathbb E[F_W-G_W].
]

可以：

* 用大量样本估计廉价的 (\mathbb E[G_W])；
* 用少量配对样本估计昂贵残差 (F_W-G_W)。

SIAM Journal on Scientific Computing 的 optimal multifidelity MC 给出了在任意多个代理模型间优化样本分配的方法，并保持对高保真统计量的无偏估计；SIAM Review 2018 对 multifidelity uncertainty quantification 做了系统 survey。([应用数学学会][25])

本题的低保真模型可以是：

* 每层权重的 rank-(r) 近似；
* 只保留部分神经元；
* 前几层精确、后几层线性化；
* 较浅网络；
* Gaussian mean/covariance propagation；
* cumulant propagation；
* 对激活模式进行低秩近似。

例如：

[
G_W(x)=
\text{低秩或剪枝网络输出}.
]

如果：

[
\operatorname{Corr}(F_W,G_W)\approx1
]

且 (G_W) 成本远低于 (F_W)，multifidelity 可能比仅优化输入点集有更大的收益。

但随机高斯矩阵本身通常不低秩，因此需要实测深层激活是否出现低有效秩，不能默认 rank-(r) 权重近似有效。

---

# 八、第七大类：Importance sampling

普通采样来自：

[
p(x)=\mathcal N(0,I).
]

Importance sampling 改从 (q(x)) 采样：

[
\mu
===

\mathbb E_{q}
\left[
F_W(X)\frac{p(X)}{q(X)}
\right].
]

理想 proposal 会把更多样本放到对积分贡献大的区域。对于单个非负标量函数，形式上最优 proposal 与

[
p(x)F_W(x)
]

相关。

RQMC 与 importance sampling 也可以结合，相关 SIAM 工作分析了 Gaussian 积分下 drift IS、Laplace IS 及其与 RQMC 的误差性质。([应用数学学会][26])

但本题有 256 个输出，且每个神经元的高贡献方向不同。因此一个统一 proposal 很难同时对所有输出最优。可以尝试：

[
q(x)\propto p(x)\left(
\epsilon+|F_W^{\rm cheap}(x)|_2
\right),
]

或者使用多个 proposal 的 mixture。

主要风险是：

* proposal 由少量 pilot 学错；
* importance weights 出现重尾；
* 某些神经元方差降低，另一些反而上升。

因此我会把它放在 control variate、cubature compression 和 active-subspace RQMC 之后。

---

# 九、针对这个挑战，我认为最值得研究的五条路线

## 方向 1：German + 高阶 network-aligned control variates

保留 German 的：

* antithetic；
* whitening；
* radial stratification；

再增加少量期望为零的三、四阶 Hermite 特征：

[
H_3(a_k^\top x),\qquad H_4(a_k^\top x),
]

其中 (a_k) 来自网络的 active directions。

这是最接近现有强方案、风险最低的增量路线。

## 方向 2：Exact radius integration + spherical tight frame + SHCV

组合为：

[
\boxed{
\text{解析积分 }R
+
\text{antipodal tight frame}
+
\text{低秩 spherical-harmonic control variates}
}
]

它不使用普通 whitening，而是在球面上：

* 精确一阶；
* 精确二阶；
* 用 CV 去掉部分更高阶成分。

这是 Sphere 路线理论上最自然的升级。

## 方向 3：大量浅层候选，少量深层代表点

先让 (M) 个候选只通过前 (L_0) 层：

[
x_i\rightarrow h_i^{(L_0)}.
]

然后在隐藏空间中使用：

* empirical cubature；
* kernel thinning；
* support points；
* clustering + stratification；
* moment matching；

压缩到 (N) 个代表点，只让它们继续通过剩余网络。

总成本：

[
M L_0 + N(32-L_0).
]

当 (L_0=1) 或 2 时，可以看很多候选而不付完整前向的代价。这最直接实现了：

> 用大量廉价观察，决定少量昂贵评估该落在哪里。

## 方向 4：NNGP-kernel quadrature

使用 depth-32 ReLU 的 NNGP kernel 设计一个对随机网络分布最优的点集：

[
\min_{{x_i,a_i}}
\operatorname{MMD}*{K*{\rm NNGP}}^2(\nu_N,P).
]

再用具体网络权重或第一层 hidden features 对点集做轻量 instance adaptation。

这是最“AI-theory-aware”的路线，也是 kernel quadrature 文献与本挑战最自然的结合。

## 方向 5：Active-subspace RQMC

流程为：

1. 用廉价 weight-only proxy 或少量 pilot 找到主要方向 (Q)；
2. 对 Owen-scrambled Sobol 点做 inverse Gaussian；
3. 用 (Q) 将高质量早期坐标对齐到 active directions；
4. 加 antithetic；
5. 使用多个 independent scramble 获得误差估计。

即：

[
x_i=Q\Phi^{-1}(u_i^{\rm scrambled}).
]

它实现简单，适合快速建立强实验基线。

---

# 十、怎样公平衡量“相当于多采了多少样本”

可以定义一个非标准但直观的 **effective sample multiplier**：

[
G(N)
====

\frac{
\operatorname{MSE}*{\rm iid}(N)
}{
\operatorname{MSE}*{\rm method}(N)
}.
]

如果：

[
G(N)=5,
]

那么在这个具体问题和样本数区间内，(N) 个结构化样本大致相当于 (5N) 个 iid 样本。

实验时应同时测：

[
\operatorname{MSE}
==================

\underbrace{|\operatorname{Bias}|^2}*{\text{系统偏差}}
+
\underbrace{\operatorname{Variance}}*{\text{抽样波动}}.
]

尤其要注意：

* antithetic、标准 RQMC 通常可以保持无偏；
* 用同一批数据估计 whitening transform、control coefficients 或 optimized weights，可能产生有限样本偏差；
* 比赛最终看 MSE，因此有小偏差但大幅降方差的方法仍可能更优；
* 所有方法必须按总 FLOPs 比较，而不是只比较完整前向次数；
* 应在未参与设计的网络 seed 上测试，防止把 point design 或 meta-CV 过拟合到公开网络。

---

## 总体判断

从这批文献看，方法发展大致经历了三个层次：

[
\text{iid sampling}
\rightarrow
\text{更均匀、更匹配矩的 sampling}
\rightarrow
\text{针对函数类或具体 integrand 的 sampling}.
]

German 和 Sphere 主要处于前两个层次：

* German 利用高斯矩结构；
* Sphere 利用网络齐次结构。

真正可能再提高一个数量级的方向，往往需要进入第三层：

[
\boxed{
\text{让点集、control variate 或 surrogate 看到网络本身的结构}
}
]

其中最有希望的三个具体组合是：

[
\boxed{
\begin{aligned}
&\text{German}+\text{network-aligned Hermite CV},\
&\text{Sphere}+\text{spherical-harmonic CV},\
&\text{large candidate pool}+\text{hidden-space cubature compression}.
\end{aligned}
}
]

再往更理论化的方向，则是：

[
\boxed{\text{NNGP-kernel quadrature}}
]

因为它直接把“对随机深 ReLU 网络的平均积分误差”转化成了一个可以优化的 kernel discrepancy。

[1]: https://github.com/galfaroi/Can-You-Predict-a-Network-Without-Running-It- "https://github.com/galfaroi/Can-You-Predict-a-Network-Without-Running-It-"
[2]: https://epubs.siam.org/doi/10.1137/22M1479129 "https://epubs.siam.org/doi/10.1137/22M1479129"
[3]: https://www.tandfonline.com/doi/abs/10.1080/00401706.1979.10489755 "https://www.tandfonline.com/doi/abs/10.1080/00401706.1979.10489755"
[4]: https://www.tandfonline.com/doi/abs/10.1080/01621459.1993.10476423 "https://www.tandfonline.com/doi/abs/10.1080/01621459.1993.10476423"
[5]: https://proceedings.mlr.press/v89/wu19c.html "https://proceedings.mlr.press/v89/wu19c.html"
[6]: https://proceedings.mlr.press/v97/choromanski19a.html "https://proceedings.mlr.press/v97/choromanski19a.html"
[7]: https://epubs.siam.org/doi/pdf/10.1137/S0036142994277468 "https://epubs.siam.org/doi/pdf/10.1137/S0036142994277468"
[8]: https://proceedings.mlr.press/v80/buchholz18a.html "https://proceedings.mlr.press/v80/buchholz18a.html"
[9]: https://epubs.siam.org/doi/10.1137/19M127149X "https://epubs.siam.org/doi/10.1137/19M127149X"
[10]: https://epubs.siam.org/doi/pdf/10.1137/15M1007963 "https://epubs.siam.org/doi/pdf/10.1137/15M1007963"
[11]: https://epubs.siam.org/doi/epdf/10.1137/25M1739790 "https://epubs.siam.org/doi/epdf/10.1137/25M1739790"
[12]: https://link.springer.com/article/10.1007/s00365-026-09760-9 "https://link.springer.com/article/10.1007/s00365-026-09760-9"
[13]: https://link.springer.com/article/10.1007/s13160-020-00451-x "https://link.springer.com/article/10.1007/s13160-020-00451-x"
[14]: https://arxiv.org/abs/1605.00361 "https://arxiv.org/abs/1605.00361"
[15]: https://proceedings.mlr.press/v108/choromanski20a.html "https://proceedings.mlr.press/v108/choromanski20a.html"
[16]: https://scholars.duke.edu/publication/1657620 "https://scholars.duke.edu/publication/1657620"
[17]: https://arxiv.org/abs/1708.06897 "https://arxiv.org/abs/1708.06897"
[18]: https://proceedings.mlr.press/v38/lacoste-julien15.html "https://proceedings.mlr.press/v38/lacoste-julien15.html"
[19]: https://proceedings.mlr.press/v70/briol17a.html "https://proceedings.mlr.press/v70/briol17a.html"
[20]: https://proceedings.neurips.cc/paper/2009/hash/5751ec3e9a4feab575962e78e006250d-Abstract.html "https://proceedings.neurips.cc/paper/2009/hash/5751ec3e9a4feab575962e78e006250d-Abstract.html"
[21]: https://proceedings.mlr.press/v134/dwivedi21a.html "https://proceedings.mlr.press/v134/dwivedi21a.html"
[22]: https://proceedings.mlr.press/v51/oates16.html "https://proceedings.mlr.press/v51/oates16.html"
[23]: https://proceedings.mlr.press/v235/leluc24a.html "https://proceedings.mlr.press/v235/leluc24a.html"
[24]: https://proceedings.mlr.press/v216/sun23a.html "https://proceedings.mlr.press/v216/sun23a.html"
[25]: https://epubs.siam.org/doi/10.1137/15M1046472 "https://epubs.siam.org/doi/10.1137/15M1046472"
[26]: https://epubs.siam.org/doi/10.1137/19M1280065 "https://epubs.siam.org/doi/10.1137/19M1280065"
