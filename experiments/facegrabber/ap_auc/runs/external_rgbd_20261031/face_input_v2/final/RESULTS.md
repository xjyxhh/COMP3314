# 固定协议扩展实验结果

**范围：FaceGrabber real sensor-depth alternative face verification; not EURECOM/Curtin reproduction。不是原论文数值复现。**
固定输入，5轮重抽负对，特征及参数网格见输入protocol与搜索protocol。
本次新数据在模型冻结后统一测试。标准差是固定人物划分下负对重抽的描述性波动，不是独立人物泛化的置信区间。

|方法|AP均值±标准差 (%)|AUC均值±标准差 (%)|
|---|---:|---:|
|euclidean|77.84 ± 0.16|84.53 ± 0.16|
|itml|90.22 ± 0.23|95.18 ± 0.11|
|lmnn|94.37 ± 0.30|97.66 ± 0.11|
|svm|77.87 ± 0.44|87.21 ± 0.17|
|svm_plus|76.18 ± 1.19|88.64 ± 0.48|
|itml_s|82.97 ± 0.19|88.31 ± 0.13|
|itml_plus|91.20 ± 0.31|95.85 ± 0.20|

原版ITML+相对ITML的平均AP差为 +0.978 个百分点。结果按实际差值报告，不要求出现正收益。
比例分析中间档固定主比较参数；Partial的0%复用独立选参的ITML端点，其余档使用ITML+主比较参数，端点参数差异须单独考虑。
各候选和选定模型的停止原因/残差见训练目录；有限预算及LMNN数值停止不等于全局最优。
未根据这些测试成绩再次选参或更改特征。不进行将相关pairs当独立样本的显著性检验。
四项分析：partial_curve.png、sample_count_curve.png、distance_analysis.png、training_time.png。
距离图使用训练对，不能单独证明减少泛化过拟合；本机时间不能验证作者时间比例。
行人重识别没有对应真实RGB-D数据，不能把人脸/物体结果称为BIWI验证。
