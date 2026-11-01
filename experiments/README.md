# 实验数据与检验指标

按数据来源与指标归档；正收益、负收益和失败候选的记录均保留。manifest.json逐文件记录原始路径、SHA256、存放位置、重复关系或排除原因。混合指标的报告放在cross_dataset或joint_metrics_and_report中，不将不同任务AP合并。模型、共享输入及源码快照各有独立目录。

|数据来源|内容|
|---|---|
|facegrabber|真实传感器深度人脸验证、机制对照、10个重叠人物划分|
|weightgait|展示深度人物检索、跨片段/负重动作；不是原BIWI Walking/Still|
|orl_olivetti|清晰图像特权信息，包括早期负收益及后续G-LBP扩展|
|washington|RGB-D物体实例验证及深度改进，非人物任务|
|synthetic_and_formula_validation|合成流程、Partial、凸解及公式核验|
|feature_validation|G-LBP/KDES实现与标量对照|
|cross_dataset|跨数据集报告、计时、独立核验及来源记录|

各数据来源下的ap_auc、rank1_cmc、privileged_ratio、training_pair_count、distance_distribution、training_time、convergence_kkt等目录对应检验主题；inputs_features_and_protocol、models、integrity_and_provenance保存复算依据。一个文件同时包含多个指标时原样保留一次，不拆改原实验文件。

同内容文件只保存一份。超过40 MiB文件按字节分块。执行`python tools/restore_experiments.py --output /path/to/new_directory`，可校验并恢复被纳入清单的文件到原始相对目录；目标必须不存在。未纳入的人物图像、下载归档、失败尝试的大型二进制缓存仍在原工作区，排除原因见manifest。图表保留，人物示例/质量检查图不上传。

恢复后原实验报告的相对链接和复算脚本可使用原目录结构；部分记录仍含原机器绝对路径，跨机器运行需指定新路径。原始数据及外部依赖须按项目说明另行取得，不能把“可还原实验产物”当作完整原论文数值复现。

## 各来源与指标文件数（含去重引用）

|来源|指标/用途|文件数|
|---|---|---:|
|cross_dataset|convergence_kkt|2|
|cross_dataset|failed_attempts|1|
|cross_dataset|inputs_features_and_protocol|1|
|cross_dataset|integrity_and_provenance|7|
|cross_dataset|joint_metrics_and_report|6|
|cross_dataset|mechanism_controls|3|
|cross_dataset|privileged_ratio|1|
|cross_dataset|protocol_and_logs|20|
|cross_dataset|rank1_cmc|1|
|cross_dataset|source_snapshots_and_scripts|7|
|cross_dataset|training_time|2|
|facegrabber|ap_auc|85|
|facegrabber|convergence_kkt|70|
|facegrabber|distance_distribution|2|
|facegrabber|inputs_features_and_protocol|18|
|facegrabber|integrity_and_provenance|16|
|facegrabber|mechanism_controls|94|
|facegrabber|models|70|
|facegrabber|privileged_ratio|141|
|facegrabber|protocol_and_logs|27|
|facegrabber|robustness|176|
|facegrabber|source_snapshots_and_scripts|35|
|facegrabber|training_pair_count|341|
|facegrabber|training_time|2|
|feature_validation|convergence_kkt|2|
|feature_validation|inputs_features_and_protocol|2|
|feature_validation|protocol_and_logs|3|
|orl_olivetti|ap_auc|155|
|orl_olivetti|convergence_kkt|84|
|orl_olivetti|distance_distribution|2|
|orl_olivetti|inputs_features_and_protocol|44|
|orl_olivetti|integrity_and_provenance|22|
|orl_olivetti|models|83|
|orl_olivetti|privileged_ratio|153|
|orl_olivetti|protocol_and_logs|49|
|orl_olivetti|source_snapshots_and_scripts|35|
|orl_olivetti|training_pair_count|341|
|orl_olivetti|training_time|2|
|synthetic_and_formula_validation|convergence_kkt|122|
|synthetic_and_formula_validation|distance_distribution|2|
|synthetic_and_formula_validation|inputs_features_and_protocol|60|
|synthetic_and_formula_validation|integrity_and_provenance|9|
|synthetic_and_formula_validation|joint_metrics_and_report|78|
|synthetic_and_formula_validation|models|61|
|synthetic_and_formula_validation|privileged_ratio|78|
|synthetic_and_formula_validation|protocol_and_logs|32|
|synthetic_and_formula_validation|source_snapshots_and_scripts|5|
|synthetic_and_formula_validation|training_pair_count|35|
|synthetic_and_formula_validation|training_time|2|
|washington|ap_auc|78|
|washington|convergence_kkt|24|
|washington|inputs_features_and_protocol|64|
|washington|integrity_and_provenance|9|
|washington|models|26|
|washington|protocol_and_logs|30|
|weightgait|convergence_kkt|77|
|weightgait|distance_distribution|2|
|weightgait|failed_attempts|44|
|weightgait|inputs_features_and_protocol|21|
|weightgait|integrity_and_provenance|18|
|weightgait|mechanism_controls|94|
|weightgait|models|77|
|weightgait|protocol_and_logs|25|
|weightgait|rank1_cmc|90|
|weightgait|source_snapshots_and_scripts|70|
