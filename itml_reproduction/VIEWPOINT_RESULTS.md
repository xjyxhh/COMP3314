# 替代数据上的论文观点检验

日期：2026-10-29。目标：检验训练时引入深度特权信息，能否改善仅使用RGB的验证表现。不要求原文数据或数值一致。

## 结论

**实现和训练流程已跑通，但当前Washington基础特征实验不支持“加入深度后优于RGB-only ITML”的性能结论。** 原版与修正版均保留，未按测试表现筛选数据、参数或遍数。

## 设置

直接读取 washington_audit_verified_20261026 的PCA后特征、划分、pairs和xi0，不重新抽样或拟合。288样本、24实例、8类别；训练/验证/测试实例互斥。两轮共用身份划分和正对，负对不同。测试仅使用RGB。特征为基础空间统计，并非论文GLBP。

固定参数诊断：gamma=0.1、lambda=1，每个方法从相同初始化独立运行100/500/1000遍，关闭提前停止，不按验证或测试选择遍数。

性能实验：gamma∈{0.01,0.1,1}，ITML+额外搜索lambda∈{0.1,1,10}，固定100遍。每轮每种方法仅按验证AP选参，平分保留首个候选，选定后计算测试指标。该小网格与固定预算不等于完整参数搜索或所有候选收敛证明。

## 验证集选参后的测试结果

|方法|两轮平均AP (%)|两轮平均AUC (%)|
|---|---:|---:|
|euclidean|74.80|81.66|
|itml|83.77|91.93|
|itml_plus|70.39|84.35|
|itml_plus_corrected|70.25|83.62|

|轮次|方法|选定gamma|选定lambda|验证AP|测试AP|测试AUC|
|---|---|---:|---:|---:|---:|---:|
|round_00|euclidean|None|1.0|0.787679|0.781306|0.823785|
|round_00|itml|1.0|1.0|0.841896|0.889457|0.932943|
|round_00|itml_plus|1.0|0.1|0.786588|0.760294|0.868490|
|round_00|itml_plus_corrected|1.0|0.1|0.785039|0.747464|0.858290|
|round_01|euclidean|None|1.0|0.745765|0.714679|0.809462|
|round_01|itml|1.0|1.0|0.850359|0.785935|0.905599|
|round_01|itml_plus|0.1|0.1|0.846968|0.647426|0.818576|
|round_01|itml_plus_corrected|0.1|0.1|0.856349|0.657579|0.814019|

## 固定参数的优化诊断（1000遍）

|轮次|方法|目标值|相对不等式违例|相对等式残差|
|---|---|---:|---:|---:|
|round_00|itml_plus|30.908181|4.534e-16|3.456e-16|
|round_00|itml_plus_corrected|29.413161|1.362e-08|9.561e-09|
|round_01|itml_plus|35.200111|5.032e-16|4.136e-16|
|round_01|itml_plus_corrected|31.586363|1.401e-08|9.419e-09|

修正版目标更低，但这不等于测试性能更好；原版低原始残差也不足以证明全局最优。固定预算运行的stop=max_sweeps不应改称触发了收敛停止。

## 限制与后续

当前每个split每类别仅一个实例，没有同类不同实例的困难负对；两轮共用划分，不能当作两个独立数据集。深度基础统计是否提供额外判别信息尚未证实。此次结果不能否定论文在其特征和数据条件下的结论，也不能作为支持论文性能结论的证据。

下一步应预先固定更有判别力的RGB/深度描述子及含同类负对的协议，再做深度质量或随机深度对照；根据验证集选参，最后一次性评估测试集。不能为获得正向结果而修改已有测试数据或隐藏本次结果。

## 复跑与产物

从itml_reproduction目录运行（输出目录必须不存在）：

```bash
/opt/miniconda3/bin/python -m itml_repro.compare_fixed --input runs/washington_audit_verified_20261026 --output runs/new_fixed_compare
/opt/miniconda3/bin/python -m itml_repro.viewpoint --input runs/washington_audit_verified_20261026 --output runs/new_validation_grid
/opt/miniconda3/bin/python -m unittest discover -s tests -v
```

正式输出：runs/washington_viewpoint_20261029（固定参数诊断与基线）和 runs/washington_validation_grid_20261029（验证选参）。所有训练、候选搜索、模型、输入SHA256及输入未变检查均保留。初次固定诊断另保留在runs/washington_fixed_compare_20261029。10项测试通过，包含固定配对入口、验证集选参和模型指标重算的检查。
