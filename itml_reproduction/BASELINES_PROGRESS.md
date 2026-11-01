# 原文四种基线补齐

2026-10-31。接续 Partial ITML+。此次新增 LMNN、线性 SVM、线性 SVM+、ITML-S 的显式模型和验证选参入口。已能在小规模固定输入上运行原文全部七种方法；这不等于七种方法与作者软件/实验细节完全相同，也不等于原规模实验已通过。

## 依据与实现约定

主文献 V-A/V-B 使用工作区 PDF 第7–8页：[作者公开版](https://wenli-vision.github.io/papers/Xu_TNNLS2015.pdf)。参数网格与方法描述来自这里。

LMNN 目标核对 [Weinberger & Saul, JMLR 2009，第3.4节式(14)](https://www.jmlr.org/papers/volume10/weinberger09a/weinberger09a.pdf)；SVM+ 核对 [Vapnik & Vashist, Neural Networks 2009，第3.2节，第548页](https://dl1.icdst.org/pdfs/files/3e25a143988b3a8f7005c8cb756b4113.pdf)。求解器使用本机已有 CVXPY/CLARABEL，独立重写原始凸模型，未复制作者代码。这里完成的是目标与结构的复现，没有恢复作者未公布的核、容差、软件版本与内部运行顺序。

### LMNN

```text
min_M (1-mu) * sum_(i,j target) d²_M(x_i,x_j)
      + mu * sum_(i,j target,l different identity)
                 max(0, 1+d²_M(x_i,x_j)-d²_M(x_i,x_l))
subject to M >= 0
```

目标邻居在训练集按同身份欧氏距离固定，排除自身，同分按训练输入顺序。默认 k=5；主文“parameter for KNN is set to 5”在这里解释为目标邻居数。若每人不足 k+1 个训练样本，明确失败，不偷偷降低 k。所有异身份样本均参与 triplets，没有抽样。

使用完整身份标签而非仅已抽取的训练 pairs，与主文所说 LMNN 使用更强监督信息一致。mu={0.1,0.2,...,1.0}，损失为求和，未按 triplet 数归一化。记录原始目标重算、PSD、约束、驻点和互补残差；不裁剪特征值。

**规模限制：** 当前为直接 SDP 参考求解器，默认拒绝超过1,000,000个 triplets。按原文规模、k=5：EURECOM约637,000，CurtinFaces约29,331,250，BIWI约1,200,000。后两者超出默认限制；即便数量未超过，也没有验证150维的大规模耗时/内存。需要后续接入高效 LMNN active-set 求解器并核对同一目标，不能通过悄悄抽样把实验改小。配置可显式提高限制，但提高限制不保证资源足够。

### SVM 和 SVM+

先按主文 V-A4 构造对称样本对特征：

```text
g = standard_normal_pdf((x_i+x_j)/2)
F_ij = concat(abs(x_i-x_j)*g, x_i*x_j*g)
```

这里将“零均值、单位方差 Gaussian function”解释为归一化标准正态密度。不额外白化、标准化或加权；这项尺度选择会影响 C，必须保留记录。深度按相同公式单独构造 G。

SVM 为标准线性软间隔目标：0.5||w||²+C*sum(xi)，xi>=0，y(Fw+b)>=1-xi。

SVM+ 为：

```text
min 0.5||w||² + 0.5*gamma||w*||² + C*sum(Gw*+b*)
s.t. y*(Fw+b) >= 1-(Gw*+b*)
     Gw*+b* >= 0
```

两个 bias 均不正则化；没有在结构化松弛外额外添加自由 slack。验证/推断仅计算 Fw+b，不访问 G、w* 或 b*。C和SVM+的gamma分别取{0.01,0.1,1,10,100}。SVM+ 的 gamma 与 ITML 的 gamma 是不同模型的参数，使用不同配置键。

**核函数约定：** 主文未明确核函数与带宽。当前必须在配置中显式设置 `svm_kernel=linear`，不是假定作者也使用线性核。尚未实现 RBF/其他核变体；不能据此声称已恢复原表 SVM/SVM+ 数值。原始QP与独立对偶问题对照，记录原始可行性、驻点、乘子和互补残差。

### ITML-S

按主文 V-A6 的两阶段顺序：先在深度空间训练 ITML → 识别/删除深度空间异常训练对 → 用剩余训练对及原顺序训练普通空间 ITML。

主文未给出完整异常阈值规则，也未明确两阶段是否独立选 gamma；其参考文献 [26] 全文此次未成功获取，因此不能冒称已核对该论文完整实现。当前接口要求显式传入 `filter_rule=shared_xi0`，两个阶段共用一个 gamma：

```text
depth_model = ITML(Z, original_pairs, original_xi0, gamma)
keep = y * (d²_depth_model(z_i,z_j)-original_xi0) <= 0
ordinary_model = ITML(X, original_pairs[keep], original_xi0[keep], gamma)
```

等于阈值的训练对保留。输入 xi0 即给定训练普通特征的边界值，此处不会偷偷改成深度分位数、学习新的过滤阈值、改标签或重抽 pairs。保留 depth_M、keep_mask、保留/删除数量、两个阶段的训练历史。若某个标签被全部删除，候选失败并记原因，没有退回原 ITML 冒充训练成功。这是公开的阈值解释版本，后续拿到原始实现时需再核对。

## 统一的验证比较入口

新入口 `python -m itml_repro.baseline_compare`，不改原有 `run.py` 默认行为。七种方法按以下顺序使用共用固定输入：

```text
Euclidean → ITML → LMNN → SVM → SVM+ → ITML-S → ITML+
```

所需文件：

- `features_pca.npz`：X_train、Z_train、X_val。
- `round_XX/pairs.npz`：train、val；`round_XX/xi0.npy`。
- LMNN 标签：`train_manifest.csv` 中按训练特征行顺序的 identity；也支持既有 Olivetti 的 `manifest.csv`+`split.npz`。

入口校验身份标签与训练 pairs 一致。它依赖预先核验的划分/PCA/特征对齐，不自行重建这些数据。测试特征和测试 pairs 可以完全不存在。此次未接入 BIWI 的 probe/gallery 主协议。

`configs/baselines_paper.json` 保留主文参数网格，每轮候选数：Euclidean 1、ITML 9、LMNN 10、SVM 5、SVM+ 25、ITML-S 9、ITML+ 81，共140。只按验证 AP 选参（配置可选AUC），平分保留第一个。输入有几轮就运行几轮，不自动补足五轮。每个候选均保留耗时/诊断/失败；保存选定模型、训练历史和输入/代码哈希，输出目录不得已存在。

分类器使用 decision_scores，度量使用负平方距离，共同调用 `verification_scores` 的 AP/AUC。同分分组规则与旧 `verification` 一致，旧回归检查通过。

本机已安装依赖；其他环境可选安装 `requirements-baselines.txt`。基本 ITML/ITML+ 不因新增模块而强制引入 CVXPY。

从 itml_reproduction 目录执行，输出目录使用新名称：

```bash
/opt/miniconda3/bin/python -m itml_repro.baseline_compare \
  --input runs/baselines_smoke_20261031/input \
  --config runs/baselines_smoke_20261031/config.json \
  --output runs/baselines_smoke_repeat
```

默认不传 `--config` 时使用 baselines_paper.json；本次合成检查缩小了部分网格、样本数和维度，不能将其结果作为原文主实验。

## 实际验证与限制

- 31项测试通过。新增5项检查覆盖样本对表示/普通输入推断、SVM与SVM+独立对偶比较、LMNN一维解析最优与规模限制、ITML-S过滤分支、移除全部测试输入后的七方法入口。
- 独立对偶比较不是将训练代码再调用一次：另建 alpha/beta 对偶问题，对比原始/对偶目标及主决策权重。
- LMNN四个一维样本的解析最优 M=1.25、目标0.025通过；拒绝不足邻居及超出triplet限制。此检查不保证任意规模数值稳定性。
- 两轮合成流程检查：14个选定模型、50个候选全部成功；14个保存模型重新加载后的验证 AP/AUC 均与记录相符。输入哈希不变，测试数组根本不存在。
- 合成候选中凸模型最大原始违例约1.39e-9，最大记录驻点误差约2.89e-8，最大互补残差约3.23e-9。它们仅对应本次小输入。
- `runs/baselines_smoke_20261031/acceptance.json` 保存指标复算与残差；最终使用 `comparison_final/` 保存全候选/模型，并为迭代模型逐候选记录末遍残差及ITML-S两个阶段的末遍状态。`comparison/`是补充这些记录前的过程检查，保留不覆盖。生成脚本为 `generate_inputs.py`，从 itml_reproduction 可用 `python -m runs.baselines_smoke_20261031.generate_inputs` 生成；它拒绝覆盖已有input。可用 `python -m runs.baselines_smoke_20261031.recheck` 重算保存模型的验证指标。
- 合成训练/验证使用相同三个人工类中心、不同噪声，仅检查流程。其验证成绩不支持真实泛化结论。此次未重跑或调整任何已有真实数据测试结果。

**尚未解决：** 核/阈值未公开约定、LMNN大规模求解、原文特征和符合任务/模态的替代数据。当前耗时包含不同的凸建模及求解器开销，不能用来验证原文“ITML+约为ITML两倍”的时间结论。下一项工作是落实 GLBP 特征定义/实现和主实验数据，再进行真实数据统一训练。
