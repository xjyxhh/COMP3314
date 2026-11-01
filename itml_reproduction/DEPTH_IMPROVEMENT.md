# 深度特征改进与打乱对照

日期：2026-10-29。继续以替代数据验证论文观点；原算法未修改，所有新实验保留在新目录。

## 结果

加入形状信息后，原版ITML+平均测试AP由70.39%提高到76.00%，修正版由70.25%提高到74.58%。仍低于RGB-only ITML的83.77%，并且打乱训练深度的对照更好。因此只能报告特征改进带来的局部改善，不能宣称论文的特权信息收益已得到支持。

|方法|原始特征AP (%)|改进特征AP (%)|改进后AUC (%)|打乱深度AP (%)|
|---|---:|---:|---:|---:|
|euclidean|74.80|74.80|81.66|—|
|itml|83.77|83.77|91.93|—|
|itml_plus|70.39|76.00|89.32|82.95|
|itml_plus_corrected|70.25|74.58|87.51|82.80|

## 改进内容与协议

1. scale_only：只按训练集整体RMS将深度尺度对齐到RGB，保留所有原特征。
2. shape：每个4×4网格提取相对深度均值、标准差、10/50/90分位数、掩膜/有效深度占比，另加宽高比、填充率、深度覆盖率，共115维。每图使用有效深度中位数居中、90%-10%分位差归一化；空格统计量置0，无样本删除。仅在训练集PCA到8维。
3. shape_augmented：保留原深度8维，追加形状8维，各块按训练RMS对齐，整体再与训练RGB对齐。RGB、实例划分、pairs、xi0完全不变。共16维深度，属于特征工程扩展，不是论文GLBP。

shape替换方案两轮均得到正定不可行的数值证据，故没有进入训练搜索。最初增广方案的SDP求解器失败；随后从原来严格可行矩阵构造块对角见证，并直接复算全部原始约束和特征值，两轮均严格可行。未删除约束、裁剪训练矩阵或掩盖求解失败。

训练参数网格和预算与此前相同：gamma={0.01,0.1,1}，lambda={0.1,1,10}，100遍。分别完成尺度对齐和形状增广的验证集选参；本次搜索不计算测试分数。最后每轮每个方法按验证AP，从原始/尺度对齐/形状增广三个方案中选择，之后才计算测试分数。两版ITML+在两轮均选中形状增广。

打乱对照：仅对训练深度按固定种子3329+轮次置换，RGB和pairs不动，用已选超参数及100遍重新训练。没有为打乱对照单独选参。打乱后约束可行性未审计，有限预算结果不代表相同优化精度，因此该对照是警示性诊断，不是严格因果证明。

## 逐轮结果

|轮次|方法|特征|验证AP|测试AP|打乱深度测试AP|
|---|---|---|---:|---:|---:|
|round_00|euclidean|original|0.787679|0.781306|—|
|round_00|itml|original|0.841896|0.889457|—|
|round_00|itml_plus|shape_augmented|0.846768|0.811797|0.886136|
|round_00|itml_plus_corrected|shape_augmented|0.844909|0.779261|0.870754|
|round_01|euclidean|original|0.745765|0.714679|—|
|round_01|itml|original|0.850359|0.785935|—|
|round_01|itml_plus|shape_augmented|0.898515|0.708181|0.772824|
|round_01|itml_plus_corrected|shape_augmented|0.896623|0.712385|0.785329|

## 如何理解

当前数据中，身份跨训练/验证/测试互斥，但每个split每类别只有一个物体实例；两轮共用实例划分。深度可能主要反映类别/视角/缺失模式，而非稳定可迁移的实例判别信息。此解释尚未验证，不能当作已证实原因。

此前测试集已被查看，本次属于同一划分上的探索性改进，即便参数/特征方案按验证集选择，也不能称为全新未接触测试集上的确认性评估。下一阶段应预注册包含同类困难负对的新实例划分，并扩大样本量，比较正确/打乱深度，最后在新的保留集确认。不能继续只按当前测试AP寻找有利特征。

## 文件

- 特征生成与审计：runs/depth_improvement_20261029、runs/depth_augmented_20261029。
- 验证选参：runs/depth_improvement_20261029/scale_search、runs/depth_augmented_20261029/shape_search。
- 最终选定模型、测试结果、打乱深度索引和模型：runs/depth_selected_20261029。
- shape_augmented/pca.npz保存包含块尺度的完整投影；depth_pca.npz保存形状块未缩放PCA。

复跑从itml_reproduction目录执行，输出目录须不存在：

```bash
/opt/miniconda3/bin/python -m itml_repro.improve_depth --input runs/washington_audit_verified_20261026 --output runs/new_depth_features
/opt/miniconda3/bin/python -m itml_repro.viewpoint --input runs/new_depth_features/shape_augmented --output runs/new_shape_search --defer-test
/opt/miniconda3/bin/python -m unittest discover -s tests -v
```
