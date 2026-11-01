# ITML 与 ITML+ 复现实验

当前状态（2026-11-01）：核心算法、七方法比较、G-LBP/KDES特征和本次计划的FaceGrabber、WeightGait替代数据实验已完成。**尚未完成原论文原数据的数值复现。** EURECOM、CurtinFaces、BIWI及作者部分历史实现约定仍缺；WeightGait展示深度和normal_A/B不能当作原始物理深度及Walking/Still。

主文献：Xu, Li & Xu, *Distance Metric Learning Using Privileged Information for Face Verification and Person Re-Identification*, TNNLS 2015。

## 当前结果与证据

最新入口为[替代数据实验报告](EXTERNAL_RGBD_RESULTS.md)、[完整数值报告](runs/external_rgbd_20261031/final_report/RESULTS.md)及[最终核验](runs/external_rgbd_20261031/final_checks.json)。

- FaceGrabber：40人560张RGB，训练特权输入来自传感器点云z。ITML AP 90.22%，ITML+ 91.20%；人脸最佳为LMNN 94.37%。
- WeightGait：32人1920张RGB及额外1280张负重动作RGB测试；训练使用320帧压缩彩色展示深度。normal_A/B的ITML Rank-1为31.25%/33.12%，ITML+为59.38%/70.00%。测试人物与训练/验证人物有重叠。
- 已完成五轮七方法、两类比例分析、距离图、十个人物划分、90次机制对照、40次串行计时；2300候选中2299成功，70个主比较模型可用，545个条件评估独立复算一致。

这些证据支持当前设置下的描述性收益；ITML+并非所有条件最优，小样本优势更大未复现，打乱/随机特权信息也有收益，不能把全部提升归因于真实深度语义。重复划分共享人物，不作为独立人物置信区间或显著性证明。

## 安装与快速检查

在本目录执行，建议Python 3.10+：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-full.txt
python -m unittest discover -s tests -v
python -m itml_repro.demo --output data/synthetic_new
python -m itml_repro.run --config configs/smoke.json \
  --manifest data/synthetic_new/manifest.csv \
  --features data/synthetic_new/features.npz --output runs/smoke_new
```

输出目录须不存在，重复时改用新名称。基础数值依赖见requirements.txt；凸优化基线增加requirements-baselines.txt；完整特征、图表、下载和全部现有测试采用requirements-full.txt。下载依赖requests已显式列出。仅安装基础依赖不能运行所有特征与基线测试。smoke检查不代替真实实验。

## 当前程序入口

|模块|用途|
|---|---|
|metrics.py / corrected.py|论文公开ITML+及普通ITML / 独立累计对偶修正版；分别命名与报告|
|partial.py / baselines.py|有/无特权训练对混合；七种比较方法|
|data.py / pca.py / evaluation.py|身份与pairs、训练集PCA、AP/AUC与CMC|
|run.py|通用人脸验证输入准备、选参和测试；默认含Euclidean、ITML、ITML+及独立修正版|
|facegrabber.py / weightgait.py|已用替代数据集的准备；RGB-only眼点或RGB坐标裁剪|
|glbp.py / kdes_native.py|完整G-LBP与固定来源KDES移植|
|kdes.py|显式参数数学参考及预计算KDES来源/维数校验；并非废弃的完整提取器|
|baseline_compare.py|七方法统一验证选参|
|paper_analyses.py / finalize_suite.py|固定参数比例分析 / 冻结模型测试、图表和串行计时|
|reid.py|帧选择、KDES准备、七方法Rank-1选参及CMC评价|
|supplemental.py|固定主参数的机制对照|
|audit_formulas.py / verify_corrected.py|独立凸模型与修正版数值验收|
|complete_pipeline.py|仅保留冻结五轮基线后的finish_only；仍被FaceGrabber历史study调用|

常规七方法流程采用独立目录顺序运行，输入须已有features_pca.npz、每轮pairs/xi0及baseline_config.json：

```bash
python -m itml_repro.baseline_compare --input INPUT \
  --config INPUT/baseline_config.json --output BASELINES
python -m itml_repro.paper_analyses --input INPUT --baseline BASELINES \
  --config INPUT/baseline_config.json --output ANALYSES
python -m itml_repro.finalize_suite --input INPUT --baseline BASELINES \
  --analyses ANALYSES --output FINAL
```

INPUT、BASELINES、ANALYSES、FINAL为需替换的路径。分析默认fixed_main；若研究逐档重调参数，需在新实验中显式指定`--parameter-policy retune_each_fraction`，不要根据测试成绩选择政策。完整技术及行人命令见[实现与运行说明](REMAINING_IMPLEMENTATION.md)。

## 通用人脸数据接入与原论文配置

UTF-8 CSV列为`sample_id,identity,rgb_path,depth_path`，路径相对manifest，sample_id全局唯一，同帧RGB/Depth必须对齐。推荐预计算NPZ，包含字符串sample_ids、N×d的X和Z；按ID对齐，禁止pickle。训练Z须有限完整，验证/测试Z可为NaN，推断只使用普通特征。

也可使用`--extractor my_features:extract`代替`--features`；接口`extract(path, modality)`返回一维特征，modality为rgb或depth。提取器不得拟合全数据PCA/标准化，需记录特征来源、对齐与深度尺度处理。

`configs/eurecom.json`要求52人、26/13/13划分，每人14张配对图像，训练2366正+7634负对，验证/测试各1183正+3817负对；`curtinfaces.json`要求52人、每人95张，各split 15000正+15000负对。两者为PCA150、原文参数网格和五轮配置；配置存在不代表实验已完成。超出可用pairs会失败，不重复采样凑数。

```bash
python -m itml_repro.run --config configs/eurecom.json \
  --manifest data/eurecom/manifest.csv --features data/eurecom/glbp.npz \
  --output runs/eurecom_new
```

人脸配置不可直接套用于行人任务；行人专用入口与训练/Walking/Still协议已实现，但原BIWI真实条件尚未执行。

## 协议与结论边界

人脸身份train/val/test互斥；两模态PCA分别仅用训练集拟合，各方法共用pairs、xi0及验证规则。仅验证选参，全部选定后才测试，不合并验证重训。保存输入哈希、划分、PCA、pairs、候选、模型与停止/残差记录。pairs索引是各split内部索引；AP/AUC数值0–1，百分比须乘100。

原文u/l顺序、L2/平方距离、停止检查时刻、SVM核与ITML-S过滤存在未确认约定，详见[公式说明](FORMULAS.md)与[公式审计](FORMULA_REAUDIT.md)。ITML采用目标一致更新，不能同时声称与Davis印刷伪代码逐字一致。原版单步正确、范数稳定或达到预算不证明全局可行/最优；请报告残差与停止原因。修正版独立命名itml_plus_corrected，不替代论文Algorithm 1。

当前46项测试覆盖核心公式、Partial、基线、特征、协议和输入检查；有限案例通过不是任意输入正确性证明。

## 历史实验与维护

[ORL G-LBP状态](FINAL_REPRODUCTION_STATUS.md)、[低维ORL负收益](OLIVETTI_RESULTS.md)、[ORL诊断](OLIVETTI_STEPS12.md)、[Washington结果](VIEWPOINT_RESULTS.md)与[深度改进](DEPTH_IMPROVEMENT.md)均保留，不合并不同任务指标制造总体提升。Washington已提取特征并完成探索训练，任务为物体实例验证，不是人脸复现。

旧ORL分片/PID控制器及相关修改前文件保存于[维护归档](archive/cleanup_20261101/README.md)。一次性内存调度、第五轮恢复、求解器探索脚本仍原样保留，状态见[历史工具说明](runs/external_rgbd_20261031/HISTORICAL_TOOLS.md)。源码快照、独立审计和失败尝试作为证据保留，勿当作当前入口批量重跑。

数据来源、预处理修订与许可见[获取记录](EXTERNAL_RGBD_PROGRESS.md)；KDES及YuNet署名/许可保留于third_party。人物图像、数据归档和实验中的敏感元信息不应随源码默认发布；先按数据许可选定可分享内容。
