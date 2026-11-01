# 完整复刻工作：实现、运行与边界

2026-11-01更新：本页保留此前ORL阶段的实现清单。后续已完成FaceGrabber真实深度人脸实验、WeightGait展示深度人物实验、负重动作检索及全部预定补充核验，最新入口为 [EXTERNAL_RGBD_RESULTS.md](EXTERNAL_RGBD_RESULTS.md)。下文有关缺少人物样本或尚未运行人物实验的描述属于历史状态；原论文三库和未公开实现细节仍缺，详细边界以新报告为准。

2026-10-31。此次目标：继续完成已列出的其余方法/特征/实验协议，而不是只再补一个模块。此文件与运行目录一起作为当前入口。

## 当前范围

|部分|代码/核验|真实实验状态|
|---|---|---|
|ITML/ITML+、Partial，式(1)–(28)涉及的已列模型|已实现；主算法和累计修正版分别命名|已有旧扩展实验，新实验见下文|
|七种对比方法|已接入统一选参；线性核/过滤阈值等约定公开|五轮G-LBP扩展已完成，700个主比较候选均运行成功|
|G-LBP|123维/块，56块6888维；双眼对齐函数和原生深度读取接口|400张本地ORL已提取；没有眼点，使用resize的额外偏差已标记|
|LMNN规模限制|增加O(n²)内存的完整损失/梯度投影求解，不再构造全部triplet张量|保留SDP参考解；新后端有限预算，不宣称全局最优|
|Partial信息比例|0/25/50/75/100%，中间档固定主比较验证选定参数，0/100%复用各自基线端点|五轮完成；逐档重新选参保留为可选协议|
|训练对比例|20/40/60/80/100%，固定各方法主比较参数|五轮ITML/ITML+均完成|
|距离分布|固定前200个正训练对和前200个负训练对，共400对，无挑选|已生成图与原始数组|
|训练时间|选定参数重新串行计时，校验模型与原选定模型相同|20次模型复算均一致；本机时间不能对照作者秒数|
|AP/AUC汇总|统一固定模型测试、五轮均值/总体标准差、图表和Markdown结果报告|110个冻结模型完成；独立重算AP/AUC全部一致|
|行人重识别|20帧均匀采样、训练/验证人物分组、首帧probe、余帧gallery、七方法Rank-1选参、Walking/Still独立测试及CMC|合成协议检查通过；没有对应真实数据，未做真实实验|
|KDES|后续找到固定BSD源码镜像，完整移植梯度、basis/KPCA、训练码本和图像级EMK-CKSVD；21000/14000维接口|小规模执行和独立标量核验通过；MATLAB运行时及2015实际软件版本一致性仍未确认，见KDES_PORT.md|

这不是“整篇完全复刻已完成”的声明。公开论文的歧义、未取得的作者特征参数和缺失的真实RGB-D人脸/行人数据，不会通过造数据或改名消除。

## 原始特征依据

G-LBP依据 [Huynh、Min、Dugelay，原文第4节](https://www.eurecom.fr/fr/publication/3849/download/mm-publi-3849.pdf)，PDF已保存在工作区paper，提取正文及第6/7页渲染在 `runs/glbp_source_20261031`。式(5)的有符号差分截到[-8,7]，与59类uniform LBP拼接。代码明确选择3×3格点邻居、块内有效中心、升序uniform编码；支持原始计数和每直方图L1两种显式设置。新ORL实验使用后者，不冒称与作者未公开实现逐元素一致。

`glbp.align_face`根据给定的两眼坐标和显式目标坐标做相似变换；主文没有公布目标眼点，不自行声称默认值来自作者。原生深度保持整数层级；不擅自压成8位。整数图像的差分截断是原描述符定义，不是训练矩阵的数值裁剪。

KDES依据 [Bo、Ren、Fox，式(5)/(12)](https://research.cs.washington.edu/istc/lfb/paper/nips10.pdf)。`kdes.dense_gradient_kdes`要求外部给定梯度、方向/位置basis、KPCA系数与核参数，返回16×16、步长8的局部描述子。使用单位方向向量避免角度环绕。它没有伪造原始SIFT梯度算子、1000词码本或图像级EMK/CKSVD，故不能把这个局部实现称为完整21000/14000维原版KDES。

作者 [软件页](https://research.cs.washington.edu/istc/lfb/software.htm) 指向的旧地址本次实际请求返回HTTP404；后续已找到带许可和署名的源码镜像并完成完整移植，详见 [KDES_PORT.md](KDES_PORT.md)。此前kdes.py仍是式(12)数学参考；完整源码版本入口为kdes_native.py。默认行人准备要求JSON提供feature_name、producer/codebook哈希、训练拟合sample IDs或明确external_frozen来源；不能仅由维数证明特征正确。

## 新的五轮扩展协议

目录：`runs/orl_glbp_five_rounds_20261031`。协议在本轮训练前保存：

- 全部40人400张原始PGM保留；固定seed2026，20/10/10人物互斥。
- 普通输入仍是固定8×8退化/噪声，再放到120×105；清晰原图放到同尺寸作为训练特权输入。**清晰图像不是物理深度；resize不是双眼对齐。**
- 两模态均G-LBP6888维，分别仅训练集拟合PCA150维；未加入原文之外的RMS模态匹配。
- 训练全部900正对+2904负对；验证/测试各450正+1452负。由于每人仅10张，无法构造原文正对数量，不重复抽样凑数；尽量保留原文正负比例。
- 五轮人物划分与正对集合固定，只重抽负对。完整主文参数网格；ITML系最多10遍/整遍范数阈值1e-3。
- LMNN采用显式projected后端、最多1000步，目标未改变；目标停止不是全局最优证明。SVM/SVM+为显式线性核；ITML-S为shared_xi0阈值解释。
- 40人已经参与过去的探索，本轮不是从未接触过的人群保留集。结果只作为探索性扩展，不能用其证明或推翻原RGB-D论文。

各轮独立计算以缩短等待；汇总检查同一输入哈希与配置、恰好5×7个选定模型，另存分片来源。时间实验在末尾串行重新测量，不拿并发搜索时间代替训练时间。

本次可用数据上的流程依次执行主比较 → Partial比例/训练对数量 → 固定模型统一测试 → 距离图与串行时间 → 结果汇总。代码保留原文行人研究，但因无数据未执行；因此不宣称原文所有研究顺序均已实际走完。分析中间比例固定主比较验证选定的参数，端点复用主比较模型。Partial的0%使用普通ITML独立选定参数，其余档使用ITML+主比较参数，因此0%到25%的变化还包含该端点参数差异。训练对减少时保留PCA/阈值/样本划分，对两类pairs采用嵌套前缀。

主文V-D未明确逐档是否重新调参，不能把任一种选择冒称作者事实。本次先前控制器已开始逐档调参，随后在新测试评估前改用固定参数；旧的不完整`analysis_shards/`完整保留且不进入汇总，正式分析来自`analysis_fixed_shards/`，变更记录见`analysis_policy_revision.json`。主比较特征、参数网格和停止标准未改动。可使用`--parameter-policy retune_each_fraction`在全新目录执行另一协议；不得根据本次测试成绩挑选更好协议。

状态和日志：

- `pipeline_state.json`：当前阶段，仅在真实推进时更新。
- `baseline_training.log`、`shard_*.log`、`pipeline_fixed.log`：进度和异常，旧日志保留。
- `baselines_final/`：主比较汇总；`analyses/`：两类比例实验。
- `final/RESULTS.md`、summary.json、test_results.json：仅在选参全部完成后生成。
- `final/partial_curve.png`、sample_count_curve.png、distance_analysis.png、training_time.png：四项分析图。
- `final/completion.json`：只有完整扩展流程完成后存在；不存在就不能称完成。
- `final/independent_checks.json`：110个模型的独立指标、哈希、PSD和固定参数核验；最大指标误差2.23e-16以内。

当前已完成：ITML平均AP 85.66%，ITML+ 86.53%，相差+0.869个百分点；五轮差值均为正，但不是独立人物重复实验。25%特权信息AP为83.63%，低于0%的85.66%，不能声称收益单调。五个ITML+均在10遍上限停止，最大等式残差0.00695–0.76711，相对残差0.000365–0.01314，不能声称已充分收敛。详见 [FINAL_REPRODUCTION_STATUS.md](FINAL_REPRODUCTION_STATUS.md)。

正式结果不应从搜索未结束的日志中挑选。失败候选、约束残差和停止原因均保留；不会为了产生正收益重选特征/超参数。没有把相关图像pairs当独立观测进行显著性检验；原文t检验具体形式未公开，不能恢复其统计实现。

## 运行方法

安装可选完整依赖：`python -m pip install -r requirements-full.txt`。本机使用 `/opt/miniconda3/bin/python`。Matplotlib缓存应放可写位置，例如环境变量 `MPLCONFIGDIR=/tmp/itml-mpl`。

```bash
# 新输入和输出目录均须不存在
python -m itml_repro.prepare_glbp --source ../orl_faces --output runs/glbp_new
python -m itml_repro.baseline_compare --input runs/glbp_new \
  --config runs/glbp_new/baseline_config.json --output runs/glbp_new/baselines
python -m itml_repro.paper_analyses --input runs/glbp_new \
  --baseline runs/glbp_new/baselines --config runs/glbp_new/baseline_config.json \
  --output runs/glbp_new/analyses
python -m itml_repro.finalize_suite --input runs/glbp_new \
  --baseline runs/glbp_new/baselines --analyses runs/glbp_new/analyses \
  --output runs/glbp_new/final
```

`complete_pipeline`是本次已启动任务的合并/后续控制器，使用已核验的独立轮次分片；常规复跑使用上面四条顺序命令即可，无需手工管理PID。

## 行人协议入口

`reid.py`输入CSV：`sample_id,identity,subset,frame`，subset为train/walking/still；原始训练序列提供50人，按输入人物顺序（或显式identity_order）前25人训练、后25人验证。每人按数值frame排序后均匀选择20帧；两个测试条件各28人且来自原始50人，但允许与模型训练人物重叠，这一点不同于人脸验证。

每个集合首个选定样本作probe，其余19张作gallery；禁止同一样本出现在两侧。训练4750正/4750负对、五轮重抽负对；验证只按Rank-1选参。SVM/SVM+使用普通特征的正向决策分数排序，度量用负距离，均报告CMC。

```bash
python -m itml_repro.reid --stage prepare --manifest data/reid/manifest.csv \
  --features data/reid/kdes.npz --config configs/reid_protocol.json --output runs/reid_input
python -m itml_repro.reid --stage train --input runs/reid_input \
  --config configs/baselines_paper.json --output runs/reid_search
python -m itml_repro.reid --stage evaluate --input runs/reid_input \
  --search runs/reid_search --output runs/reid_test
```

这三条是可用数据的运行接口，本次没有对应真实数据，因此没有执行真实行人实验。合成单元检查明确关闭KDES维数契约，不能冒充原始数据验证。

## 验证与剩余外部条件

最新42项测试通过，包括完整KDES移植、先选帧后拟合码本、固定参数与逐档重调参两个分支，日志在 `runs/remaining_tests_completed_20261031.log`。真实ORL扩展的700个主比较候选、110个冻结模型和20次串行计时已全部完成，4张图已逐张检查；独立AP/AUC重算不调用项目评分函数。此前合成端到端验收见runs/full_suite_smoke_20261031，不将其当作真实实验结果。G-LBP另有两个6888维随机图与独立标量oracle逐元素一致，见runs/glbp_source_20261031/acceptance.json。

要满足“只更换数据集，其余全部复刻”，仍需：

1. 可访问且许可合适的RGB-D人脸及行人数据，包含对齐/帧/身份信息。查到的[IIIT-D Kinect](https://iab-rubric.org/old/index.php/rgb-d-face-recognition)、[RGB-D-Face](https://github.com/AvLab-CV/RGB-D-Face-Database)等仍需申请；本次未替用户发送邮件或签署协议。
2. 确认已移植的固定KDES源码与2015实验实际软件版本/参数的一致性，补充MATLAB运行交叉验证；以及确认G-LBP未公开的边界/归一化约定。源码获取和完整Python流程已不再是未完成项。
3. 作者未写明的核函数、过滤规则、停止时刻、u/l与距离单位的处理。现有代码明确记录约定，无法替作者补写历史事实。

这些外部条件不满足时，最完整且诚实的交付是“公式与算法部分复现、实验协议实现、现有数据上的扩展”，不是原论文全量复刻成功。
