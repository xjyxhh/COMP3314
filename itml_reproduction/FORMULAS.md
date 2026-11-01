# 公式核对与实现约定

2026-10-31已重新逐页、逐式审计，完整对应表、独立凸模型对照与保留差异见 [FORMULA_REAUDIT.md](FORMULA_REAUDIT.md)。单步更新一致不代表整个实验协议或修正版与原文Algorithm 1完全相同。

核对源为工作区 `../paper/Distance Metric Learning using Privileged Information (T-NNLS 2015).pdf`（13页）及 `../paper/Informationtheoretic metric learning.pdf`（Davis et al., ICML 2007）。已直接读取这两份 PDF，不仅依赖此前聊天或中文摘录。

主文献作者公开版本：https://wenli-vision.github.io/papers/Xu_TNNLS2015.pdf
DOI：https://doi.org/10.1109/TNNLS.2015.2405574

## 变量、目标、约束

样本特征按行储存。`X[i]` 对应 x_i∈R^h，`Z[i]` 对应 z_i∈R^g。
M∈S++^h、P∈S++^g；代码从单位矩阵初始化。论文写半正定，但普通逆与有限 LogDet 的定义域实际是正定。
每行 pair 为 `(i,j,y)`；同身份 y=+1，不同身份 y=−1。`xi0[k]` 对应 ξ⁰_ij，相似对取 u，不相似对取 l；`xi[k]` 为当前 ξ_ij。

式 (1)：r=d²_M(x_i,x_j)=(x_i−x_j)ᵀM(x_i−x_j)。s 同理使用 Z、P。

式 (3)：D_ld(A,A0)=tr(A A0⁻¹)−logdet(A A0⁻¹)−dim(A)。本项目 A0=I。
标量损失 ℓ(a,b)=a/b−log(a/b)−1，L(ξ,ξ0)=Σ_k ℓ(ξ_k,ξ0_k)。不使用平方损失或 hinge loss 替换。

ITML（式 2）：

    min_{M≻0, ξ>0} D_ld(M,I) + γ Σ_k ℓ(ξ_k,ξ0_k)
    s.t. y_k [d²_M(x_i,x_j) − ξ_k] ≤ 0

ITML+（式 4；式 6/7 是等价增广表示）：

    min_{M≻0,P≻0} D_ld(M,I) + λ D_ld(P,I)
                         + γ Σ_k ℓ(d²_P(z_i,z_j),ξ0_k)
    s.t. y_k [d²_M(x_i,x_j) − d²_P(z_i,z_j)] ≤ 0

式 6/7 显式增加 ξ_k=d²_P(z_i,z_j)。这不是把 RGB 与 Depth 拼接，也不是预先学习 P 后固定界限。

## ITML+ 单对更新：严格对应 Algorithm 1

以下 r,s,ξ,M,P 都是更新前的量：

    α = max(0, [γ/ξ + λ/s − (λ+γ)/r] / [y(λ+γ+1)])   (23)
    β = λ/(λ+γ) [γ/s − γ/ξ + yα]                       (24)
    M_new = M − yα/(1+yαr) · (Mv)(Mv)ᵀ                 (11)
    P_new = P + β/(λ−βs) · (Pw)(Pw)ᵀ                   (12)
    ξ_new = λs/(λ−βs)                                  (13)

其中 v=x_i−x_j，w=z_i−z_j。代码在 `metrics.plus_step`，没有把 β 限为非负，没有将式 (27) 在 α=0 时误用为 ξ 更新。

每次只更新选中对的 ξ。其他对的 ξ 不随着共享 P 同步重算，这忠实于 Algorithm 1。必须区分：

- 当前选中对子问题有闭式最优解，可逐步验证 KKT。
- 后续更新 P 会改变以前训练对的深度距离；中间状态不必全局可行。
- Algorithm 1 未给出 ITML 式的累计对偶修正；这里没有偷偷补入未报告的修正。
- 因而本实现是**论文公开迭代步骤的复现**，不宣称有限次循环就精确求得式 (4) 的全局最优解。历史中使用当前 P 的全部深度距离计算式 (4) 目标和约束违例，另报 ξ 与该距离的最大等式残差。目标也不要求在每次循环后单调下降。

如果要另外研究经过修正的全局优化器，应作为单独方法并与 Algorithm 1 分开报告。

## ITML 基线：保留对偶修正

后续已补齐 Partial ITML+：`metrics.fit(..., privileged_mask=mask)` 按式(5)/(28)混合有/无特权信息的分支，目标与残差仅在适用集合计算，详见 [PARTIAL_ITML_PROGRESS.md](PARTIAL_ITML_PROGRESS.md)。不传掩码时原有行为不变。

按 Davis 原始 ITML 的 soft-constraint Bregman 对偶坐标方法实现，记累计非负乘子 q_k（对应 Davis 的 λ_ij，**不是**主论文深度正则 λ）：

    a = min(q_k, y γ/(γ+1) · (1/r − 1/ξ_k))
    M_new = M + ya/(1−yar) · (Mv)(Mv)ᵀ
    ξ_new = γ ξ_k / (γ + ya ξ_k)
    q_k ← q_k − a

a 的符号与主论文 α 的增量相反。系数 γ/(γ+1) 来自 soft objective 的驻点条件，不能把公开旧版伪代码中印出的 δ/2(1/p−γ/ξ) 不经核对直接照搬到主文献式 (2) 的变量定义。令 r_new=r/(1−yar)、ξ_new=γξ/(γ+yaξ)，两者相等直接解得 a=yγ/(γ+1)(1/r−1/ξ)。γ=1 时系数才退化为 1/2。

测试不仅检查正定性，也检查整个多约束问题的原始可行性、互补松弛及两个驻点条件：

    M⁻¹ = I + Σ_k y_k q_k v_k v_kᵀ
    γ/ξ_k = γ/ξ0_k − y_k q_k

这防止用没有累计对偶的反复投影冒充原始 ITML 优化器。

## 原文歧义与显式约定

1. **u/l**：III-B 定义 u 为相似上界，l 为不相似下界；V-B3 却把 l、u 依次指向第 3、97 百分位。配置 `bound_assignment=constraint_consistent` 默认 u=q03,l=q97；`literal_v_b3` 则 l=q03,u=q97。二者都保留，不声称作者真实采用哪一种。
2. **距离单位**：V-B3 写 L2 distances，但目标用平方距离。默认 `bound_distance=euclidean` 按实验文字计算训练集全体无序样本对的 L2 分位数并直接用作 ξ0；也支持 `squared` 按平方距离取分位数。这是明确的实验解释选项，不混为定论。两方法完全共用同一数值界限。
3. **停止**：IV-D 给出相邻 α/β 向量范数相对变化 <1e−3 或最多 10×pair_count 次更新，但没写比较时刻和零分母处理。实现按完整 sweep 比较 `abs(||new||−||old||)/max(||old||,eps)`，至少两遍；默认最多十遍。返回停止原因和残差，不伪称全局收敛。
4. **顺序**：每轮先随机排列一次 pairs，再固定顺序循环。五轮共用身份 split；正对集合固定，负对重新抽样。原文未公布 split IDs、随机种子、具体 pair 顺序。
5. **PCA**：两模态独立、仅训练 fit、中心化不白化，不额外做归一化。训练样本数/数值秩不足时明确失败，不静默降低目标维数。
6. **GLBP**：V-B2 明确双眼对齐裁剪 120×105、8×7 个 15×15 块、每模态 6888 维；具体 gradient-LBP 依赖参考文献 [9]/[45]，仅主文献不足以唯一确定其编码、梯度与归一化。后续已实现G-LBP并完成FaceGrabber真实传感器深度实验，RGB眼点检测、点云投影与归一化约定详见 [最新实验报告](EXTERNAL_RGBD_RESULTS.md) 和 [获取记录](EXTERNAL_RGBD_PROGRESS.md)；这些预处理不能声称与作者程序逐细节一致。
7. **评价**：AP 用按不同分数分组的阶梯积分（非梯形 PR-AUC）；AUC 用分组 ROC 梯形积分。相同分数不会因排序顺序虚增结果。均值/标准差采用配置中的 `std_ddof`，默认总体标准差 ddof=0，原文未说明该选择。
8. **Bregman 定义笔误**：III-B 一般定义梯度写在 M，标准形式应在参考点 M0；直接采用明确式 (3) 及后续推导，不由该疑似笔误导出别的散度。

不通过加 epsilon 改写距离、裁剪特征、裁剪特征值或重置矩阵隐藏问题。遇到零 pair 距离、非正更新分母、非法特征、PCA 秩不足时抛错，保留研究问题的可见性。

## 累计对偶修正版：多轮不变量（2026-10-26）

`corrected.fit_corrected` 单独实现式(7)的对偶块优化，不修改 Algorithm 1。
令累计 α≥0、β 不限号，则每一遍都应满足：

    M⁻¹ = I + Σ y_k α_k v_k v_kᵀ
    λP⁻¹ = λI − Σ β_k w_k w_kᵀ
    γ/ξ_k = γ/ξ0_k − y_k α_k + β_k

更新变量是增量：u=y·Δα·r，b=Δβ·s/λ。全部 r,s,ξ,Mv,Pw 取自旧状态。
三个定义域分母是 1+u、1−b、1−ξu/(γr)+ξλb/(γs)，均须严格为正。
块对偶增益为 log(1+u)+λlog(1−b)+γlog(第三分母)。代码最小化其负值。
α 的界限约束累计 α_new≥0；β 无非负限制。
ξ_new=ξ_old/(第三分母)，不能每次直接强制写成当前 P 的距离，否则破坏累计驻点关系。
各遍记录三项归一化驻点误差、乘子非负违例；停止同时要求它们与可行性、等式残差、互补松弛达标。

独立对照见 `python -m itml_repro.verify_corrected --output runs/new_acceptance`，额外需要 CVXPY 和 CLARABEL。
参考模型直接构造原始凸问题，不复用修正版更新公式；三个随机小问题均以 I,I 为显式严格可行证据。
参考求解器要求 optimal、正定和约束残差≤1e-6；修正版要求两种相对残差与互补残差<1e-6、驻点误差<1e-7、目标相对差<1e-5。
矩阵差异只记录，不要求逐元素相同。数值验收不是对全部输入的形式化证明。
