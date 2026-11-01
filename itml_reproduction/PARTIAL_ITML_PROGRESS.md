# Partial ITML+：方法补齐与数据范围

后续更新：四种基线已增加显式模型与小规模验证入口，详见 [BASELINES_PROGRESS.md](BASELINES_PROGRESS.md)。本文末尾“缺四种基线”是新增前的阶段记录。

2026-10-31，接续 CURRENT_REPRODUCTION_REVIEW.md。此次完成原文 III-D / IV-E 的方法实现与比例实验入口；未完成原文图2正式实验，未增加真实数据测试成绩。

## 数据范围

以“允许替换数据集，其余尽量依照原文”为主线，固定以下要求：

- 人脸主实验仍是同人/异人验证；行人主实验仍是 probe/gallery 重识别。
- 训练时使用配对 RGB-D，验证/测试评分仅用普通特征。
- 方法按原文组织：Euclidean、ITML、LMNN、SVM、SVM+、ITML-S、原版 ITML+；修正版另作补充。
- 保留原文特征、PCA维度、参数网格、五轮重抽负对及分析比例；若替代数据规模不足以满足数量/150维数值秩，明确记为额外差异，不重复抽样或静默降维凑数。
- 论文未公开的随机种子、完整顺序、边界单位等使用公开的实施约定；不能声称已恢复作者历史运行。

本地盘点：Olivetti 为40人400张灰度脸，没有物理深度；Washington 为51类300物体实例的41,877组 RGB-D，没有原文人脸/行人任务。二者继续用于诊断，不能直接升级为“只换数据集”的正式复刻。符合任务和模态的主实验替代数据尚未锁定。本次没有下载其他数据，也没有假定新的访问权限。

## 式(5)/(28)与实现

`metrics.fit` 新增可选布尔向量 `privileged_mask`，长度等于训练 pairs；True 表示该训练对具备特权信息。未传时保持原 ITML/ITML+ 行为。

令 H 是有特权信息的训练对，U 是其余训练对。目标为：

```text
Dld(M,I) + lambda*Dld(P,I)
  + gamma*[sum_H Burg(d²_P(z_i,z_j), xi0_ij)
           + sum_U Burg(xi_ij, xi0_ij)]

H: y_ij*(d²_M(x_i,x_j)-d²_P(z_i,z_j)) <= 0
U: y_ij*(d²_M(x_i,x_j)-xi_ij) <= 0
```

内部对 H 保留辅助等式 xi=d²_P，按式(11)–(13)/(23)–(24)更新；对 U 使用现有目标一致 ITML 更新，仅更新 M/xi/该对的 ITML 累计乘子，P 保持不变，符合 IV-E 的分支说明。没有加入修正版的累计 alpha/beta 块优化。

每遍目标按上式计算；不等式诊断覆盖所有训练对，等式残差只检查 H。仅 H 中涉及的样本行要求深度有限，未涉及的行允许 NaN，不使用它们填补训练信息。

- 0%：精确退化到既有 ITML；P 不参与计算，返回 None，其正则最小值对应 I。
- 100%：精确退化到既有原版 ITML+。
- 中间比例：原版投影与 ITML 分支混合；单步对照通过不意味着整个混合迭代已达到式(5)全局最优。
- 继续保留原实现的整遍范数停止约定和残差记录；不将早停或固定预算自动称作收敛。

Python 调用：

```python
from itml_repro.metrics import fit
from itml_repro.partial import privileged_prefix

mask = privileged_prefix(pairs, 0.5)
model = fit(X_train, pairs, xi0, gamma=0.1, Z=Z_train,
            lam=1.0, max_sweeps=10, tol=1e-3,
            privileged_mask=mask)
```

`privileged_prefix` 对正对和负对分别按已保存顺序取前0/25/50/75/100%，比例间掩码嵌套，不改变训练 pairs 或训练顺序。非整数量按各标签 floor，保存实际计数。这与“减少训练对”是两种不同实验：本模块所有比例都使用全部训练约束，仅改变可用特权信息范围。

## 验证集比例实验入口

`python -m itml_repro.partial` 接受已有固定输入目录：

```text
features_pca.npz: X_train, Z_train, X_val
round_XX/pairs.npz: train, val
round_XX/xi0.npy
```

它不会读取 X_test 或 test pairs，测试输入可以完全不存在。各比例分别用同一网格按验证 AP 选参，平分保留第一个候选。0% 的 lambda 无效，因此只运行一次；其余比例搜索 gamma×lambda。未完成的候选记录失败原因。输出包含协议、代码/输入哈希、每档掩码、全部候选、选定模型及历史；禁止覆盖既有输出目录。

默认配置为 eurecom.json 的9×9网格、最多10遍及1e-3范数阈值；输入中的配对数量和维度保持原样，程序不会把替代输入自动变成原文规模。每轮最多9+4×81=333个候选。输入目录包含多少个 round 就使用多少个，不会自动创建五轮。

在 itml_reproduction 目录复跑本次合成检查，输出目录使用新名称：

```bash
/opt/miniconda3/bin/python -m itml_repro.partial \
  --input runs/smoke_verified \
  --output runs/partial_smoke_new --config configs/smoke.json
```

若要在已有 Olivetti 第一组上做验证诊断，可执行：

```bash
/opt/miniconda3/bin/python -m itml_repro.partial \
  --input runs/olivetti_clean_privileged_20261031/split_00/input \
  --output runs/partial_olivetti_validation_new
```

后一条本次未运行；它仍属于非深度、16维特征的替代设置，不能作为原论文图2数值复现。

## 本次实际验收

- 26项测试通过，包括新增5项 Partial 测试：端点多遍轨迹、分层嵌套掩码、缺失深度和式(5)目标、混合首遍原始凸投影对照、完全移除测试输入的实验入口。
- 混合首遍的12个连续投影与独立 CVXPY/CLARABEL 原始子问题比较通过，覆盖有/无特权信息及两种标签；没有将整轮目标冒称全局最优。后续普通分支的累计对偶行为由既有 ITML KKT 和多遍端点测试检查。
- 原有12个 ITML+ 单步投影及3个 ITML 完整目标对照重新通过，证据在 `runs/partial_regression_20261031`。
- 三组已存 Olivetti 原版 ITML+ 按原参数重训100遍，M/P/xi/dual/beta均与旧模型逐元素完全一致，记录在 `runs/partial_default_regression_20261031.json`。不是对已有测试模型重新选参。
- 合成两轮×五档，共10档、36个候选全部运行成功，记录在 `runs/partial_smoke_20261031`。输入哈希未变，未评估测试集。该检查只证明实现和流程可运行，不证明比例越高性能越好。

## 剩余顺序

下一方法模块是原文完整基线，仍缺 LMNN、SVM、SVM+、ITML-S；同时需要锁定满足任务/模态的替代数据及实现 GLBP。完成这些之后再做主比较、行人重识别和 V-D 四组分析。Partial 的实现完成并不等于其比例实验已完成。
