# 全算法公式复核（2026-10-31）

2026-11-01状态补充：本页保存初期公式审计；下文“Partial、LMNN、SVM、SVM+、ITML-S未实现”是当时状态，之后均已补齐，见 [Partial记录](PARTIAL_ITML_PROGRESS.md)、[基线记录](BASELINES_PROGRESS.md)及[最新实验报告](EXTERNAL_RGBD_RESULTS.md)。原文歧义、独立单步对照和原版/累计修正版的区别仍适用；不把单步一致当作全局收敛证明。

## 结论与范围

**现有论文版ITML+的核心更新式与主文献一致；未发现用于有效输入的符号、系数、矩阵更新顺序或损失方向错误。ITML基线与主文献式(2)/Davis式(4.5)目标一致。但不能将整个工程称为“原文所有算法、所有实验细节完全一致”。** 原版、修正版、论文印刷歧义和扩展实验必须分别说明。

本次直接重新读取两份工作区PDF的正文，并渲染主文献PDF第4–7页和Davis算法页核对上下标与分母，未只依赖旧中文摘录。

- 主文献：Xu, Li & Xu, Distance Metric Learning Using Privileged Information for Face Verification and Person Re-Identification，TNNLS 2015，13页。
- ITML来源：Davis et al., Information-Theoretic Metric Learning，ICML 2007，8页。
- 本次检查覆盖所有已实现训练方法（Euclidean、ITML、原版ITML+、累计对偶修正版），以及PCA、pairs、指标、训练调用入口和诊断工具。
- Partial ITML+、多种特权信息、LMNN、SVM、SVM+及ITML-S未实现；本次没有将它们写成“已通过复现”。

## 逐式对应

|原文位置|公式/内容|代码与核对结果|
|---|---|---|
|主文献PDF第3页，式(1)|平方Mahalanobis距离 vᵀMv|metrics.distances；与逐对二次型一致。测试/验证只使用普通特征和M。|
|第4页，式(2)|Dld(M,I)+gamma·ΣBurg(xi,xi0)，同人r≤xi、异人r≥xi|metrics.fit的Z=None分支；统一标签y=+1/-1，符合y(r−xi)≤0。3个完整混合约束问题与独立凸模型吻合。|
|第4页，式(3)|tr(M M0⁻¹)−logdet(M M0⁻¹)−h|metrics.logdet实现M0=I的特例，函数返回Dld(M,I)而非裸logdet；identity初始化与论文选择一致。|
|第4页，式(4)|Dld(M,I)+lambda·Dld(P,I)+gamma·ΣBurg(s,xi0)，y(r−s)≤0|原版和修正版均按当前P的全部距离s记录这一目标，而非把不一致的中间xi误当作式(4)损失。累计修正版使用这一目标，更新规则独立。|
|第5页，式(5)|Partial ITML+，有/无特权信息的pair混合|未实现；不能声称覆盖本式。|
|第5页，式(6)/(7)|引入中间xi，加入xi=s的等价表示|原版按Algorithm 1保留每个pair的xi；修正版按式(7)累计对偶。中间状态可能不满足全部等式，单独记录残差，不将中间状态当成已可行解。|
|第5页，式(8)–(10)|相对旧M/P/xi的单pair Bregman投影|metrics.plus_step；12个直接构造的原始凸子问题对照通过，含两种标签和活跃/非活跃分支。|
|第6页，式(11)–(13)|M、P的秩一更新，xi_new=lambda·s/(lambda−beta·s)|逐项一致。全部右端使用更新前状态，不把M或P先更新后的量误代入。|
|第6页，式(14)|拉格朗日函数，alpha≥0，beta不限制符号|符号与标签一致；没有错误地把beta截为非负。|
|第6页，式(15)–(18)|M、P驻点与逆矩阵更新|检查N⁻¹=M⁻¹+y·alpha·vvᵀ，lambda·Q⁻¹=lambda·P⁻¹−beta·wwᵀ；12例通过。|
|第6页，式(19)/(20)|Sherman–Morrison展开|与代码秩一更新等价，由逆矩阵残差和独立原始模型双重检验。|
|第6页，式(21)/(22)|所选pair的xi_new=wᵀP_new w|每次单pair更新一致；后续其他pair改变P后，旧pair的等式未必保持，记录全局残差。|
|第6页，式(23)|alpha=max(0,[gamma/xi+lambda/s−(lambda+gamma)/r]/[y(lambda+gamma+1)])|代码符号、系数和括号一致；通过独立子问题最优目标与KKT检查。|
|第6页，式(24)|beta=lambda/(lambda+gamma)·(gamma/s−gamma/xi+y·alpha)|代码一致；beta可正可负。|
|第6页，式(25)/(26)|gamma/xi_new=gamma/xi_old−y·alpha+beta|单步检查通过；累计修正版以累计alpha/beta维护同类全局关系。|
|第6页，式(27)|活跃约束时xi_new=r/(1+y·alpha·r)|6个活跃例单独检查通过；不在alpha=0时把该式冒充通用xi更新。|
|第7页，Algorithm 1|I初始化、xi0按正负pair赋值、选pair、按式(23)/(24)/(11)/(12)/(13)更新|原版更新步骤一致；修正版不属于这一算法。停止时刻与实验预算的区别见下文。|
|第7页，式(28)|Partial ITML+的增广表示|未实现，亦未在当前实验调用。|

## ITML原论文伪代码与目标的冲突

Davis PDF第3页式(4.5)明确使用gamma·Dld(diag(xi),diag(xi0))；第4页Algorithm 1第3.4行却印为 a=min(q,y/2·(1/r−gamma/xi))，同时第3.6行使用xi_new=gamma·xi/(gamma+y·a·xi)。

若按同一个目标求单约束驻点，应使r_new=r/(1−y·a·r)与xi_new相等，得到：

```text
a = min(q, y * gamma/(gamma+1) * (1/r - 1/xi))
M_new = M + y*a/(1-y*a*r) * (Mv)(Mv)^T
xi_new = gamma*xi/(gamma+y*a*xi)
q_new = q-a
```

当前ITML使用这一与目标一致的更新，并保留累计对偶修正。不能同时宣称它与Davis印刷伪代码逐字相同。这里是从所给PDF和代数推导得到的不一致证据，不冒称作者已发布勘误。

一维反例：gamma=0.7、r=4、xi0=0.2、相似pair。印刷伪代码一步得到M=0.133333、xi=0.373333，r_new=0.533333，约束仍违例0.16；继续使用同一印刷更新已接近零步长。目标一致更新得到M=0.113333、xi=0.453333，与解析最优M=(1+gamma)/(1+gamma*r/xi0)一致。数值反例保存于independent_final/davis_printed_counterexample.json。

## 原版ITML+与累计修正版必须分开

原版按公开Algorithm 1，每个投影的alpha为局部非负乘子；代码没有偷偷加入累计Dykstra修正。修正版每个块优化累计alpha≥0和自由beta，保留其他pair对逆矩阵及逆xi的贡献，以数值方法求式(7)的对偶块子问题。

修正版全局驻点关系：

```text
M^-1 = I + sum(y_k * alpha_k * v_k v_k^T)
lambda * P^-1 = lambda*I - sum(beta_k * w_k w_k^T)
gamma / xi_k = gamma / xi0_k - y_k*alpha_k + beta_k
```

修正版的SLSQP、1e-12定义域分母余量及KKT停止阈值都是独立数值实现细节，不是主文献式(23)/(24)的闭式原版。代码和结果持续单独命名itml_plus_corrected。

三个已知可行的小问题再次表明：原版可达到很小原始残差，却不一定达到式(4)的参考最优值。以下目标仅在每行相同问题和参数内比较：

|seed|原版目标|修正版目标|独立原始凸模型目标|修正版相对目标误差|
|---|---:|---:|---:|---:|
|17|7.97944905|7.82977458|7.82977462|5.065e-09|
|23|31.82685673|31.67287257|31.67287251|1.774e-09|
|42|24.97851866|24.79690087|24.79690115|1.122e-08|

所以“单步公式和公开伪代码一致”不能自动推出“所有输入都已求到主目标的全局最优”。这只是对上述明确案例的数值核验，也不能由此断言作者未公开的实际代码如何实现。

## 不能声称与原实验完全一致的部分

|内容|原文|当前处理/结论|
|---|---|---|
|u/l|III-B定义u为同人小上界、l为异人大下界；V-B3却按l、u的顺序写3%、97%|默认u=q03、l=q97遵循约束含义；run保留literal_v_b3选项。原文内部存在顺序歧义，未替作者断言实际代码。|
|距离分位数|V-B3写L2距离，约束和损失使用平方Mahalanobis距离|默认按L2分位数直接赋xi0，另支持squared。是明确约定，不能称已唯一确认作者采用的单位。|
|停止判据|IV-D：相邻dual向量范数相对变化<1e-3，最多10×pair_count次pair更新|默认上限10个sweep与最多更新次数一致；实现按整遍比较范数。原文t按pair更新，未清楚规定整向量比较时刻/零分母；整遍比较是实施约定，不称逐pair停止完全一致。|
|Olivetti/诊断预算|原实验最多10遍，阈值1e-3|当前100/500/1000遍且tol=0，是明确扩展；不能冒称原论文停止条件。|
|PCA与特征|人脸GLBP 6888维，各模态PCA150维|Olivetti是像素特征、16维PCA；清晰图像代替真实深度，还有训练RMS尺度匹配，均为扩展。没有实现或伪称GLBP。|
|划分/配对/重复|原文52人、26/13/13划分，五轮固定划分并重抽负对，特定pair数量|论文配置文件保留这些数值；Olivetti40人、20/10/10、三组重新划分、90+180pairs是替代设置。|
|参数范围|gamma=10^(-4:0.5:0)，lambda=10^(-2:0.5:2)|eurecom/curtinfaces配置一致；Olivetti采用三点小网格，非完整论文范围。|
|其他baseline/partial|原论文还有LMNN、SVM、SVM+、ITML-S与partial模型|当前未实现，不纳入已完成公式一致性清单。|

## 指标与辅助模块

- AP/AUC：论文没有明确给出同分处理与插值公式；代码采用按同分分组的非插值AP和ROC梯形AUC。混合同分手算例AP=7/12、AUC=0.625通过；负平方距离与负L2距离排序理论等价，所以Euclidean对照使用平方距离不改变这些排序指标。不能确认原作者未公开的指标细节。
- Rank-1/CMC：通用函数通过手算例，Rank-1=0.5、CMC=[0.5,1,1]。BIWI专用人体特征及完整gallery协议仍未实现，不能由接口测试冒称复现BIWI实验。
- PCA/pairs：重新检查仅训练拟合、两模态独立、相同ID对齐、标签方向、唯一无序pair及无跨split训练引用；现有测试覆盖这些规则。
- 可行性审计：trace(M)+trace(P)=1和最大化最小特征值是利用齐次约束构造的辅助检查，不进入训练目标。证书的数值余量处理不等于训练中裁剪M/P；原训练未加入特征值裁剪或额外松弛项。
- 后续比较/特征/诊断入口均调用同一metrics.fit或corrected.fit_corrected，没有另藏一套训练公式；推断指标只使用M和普通输入。

## 本次修复及对已有实验的影响

修复的是定义域检查，不是有效输入的训练公式：

1. 标量Burg此前只检查x/x0>0，两个负数可能错误通过。现在要求x、x0各自为有限正数，符合正对角LogDet定义域。
2. 矩阵LogDet此前只依赖Cholesky，底层可能忽略非对称矩阵上三角。现在先检查有限、方阵、数值对称，再由Cholesky检查正定。未对输入做对称化或特征值裁剪。
3. 补齐gamma、lambda、tolerance的有限性和单步标签±1检查，防止非法数值混入推导。

对三组Olivetti的ITML、原版ITML+、修正版ITML+共9个已选100遍模型重新训练，M/P/xi/dual/beta所有已存数组均逐元素完全相同。因此不需要改写此前实验结果；此前低于基线的结果没有因本次定义域修复而变化。未重新评估或据测试成绩调参。

## 验证证据

|独立检查|数量|结果|
|---|---:|---|
|式(8)原始凸子问题与plus_step比较|12，含6活跃/6非活跃|全部通过；最大相对目标误差 3.662e-09，最大单步KKT残差 3.109e-15。|
|式(2)/Davis式(4.5)完整ITML与原始凸模型比较|3，gamma=0.01/0.7/2|全部通过；最大相对目标误差 2.526e-09。|
|式(4)/(7)修正版与原始凸模型比较|3|全部通过；最大相对目标误差 1.122e-08。|
|独立复算参考解正定性、等式、不等式和活跃式(27)|15组参考解|全部通过；不只依赖求解器status。|
|既有Olivetti模型完全一致性|9|全部逐元素一致。|
|回归测试|21|全部通过，含新增定义域、印刷冲突反例、混合同分AP/AUC与CMC。|

数值验证覆盖明确输入，不是对所有维度、所有条件数或所有数据的形式化正确性证明。原版全局收敛与原文实验歧义仍须保留。

## 文件与复跑

证据目录：runs/formula_reaudit_20261031。xu.txt/davis.txt为重新提取的原文，xu_page*.png/davis_page*.png为公式页图像；sources.json记录两篇PDF和核心代码SHA256。

本次最终结果使用independent_final/与corrected_reference_final/；无final的目录是加完最后定义域检查前的过程记录，予以保留。reference_feasibility.json保存对最终独立参考解的直接复算，olivetti_unchanged.json保存9个模型的位级一致性。

从itml_reproduction目录执行，输出目录须不存在：

```bash
/opt/miniconda3/bin/python -m itml_repro.audit_formulas --output runs/formula_audit_new
/opt/miniconda3/bin/python -m itml_repro.verify_corrected --output runs/corrected_audit_new
/opt/miniconda3/bin/python -m unittest discover -s tests -v
```
