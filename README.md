# COMP3314 — ITML+ 论文复现与扩展实验

研究训练阶段特权信息能否改善普通输入上的度量学习。包含代码、实验结果、选定模型、特征与协议、逐轮指标及独立核验。

- [按数据来源与检验指标浏览实验](experiments/README.md)
- [当前算法、结果与运行说明](itml_reproduction/README.md)
- [最新替代RGB-D数据实验结论](itml_reproduction/EXTERNAL_RGBD_RESULTS.md)
- [早期ORL实验与负结果](itml_reproduction/OLIVETTI_RESULTS.md)
- [逐文件来源、SHA256与排除清单](experiments/manifest.json)

## 目录

```
experiments/
  facegrabber/                       # 人脸验证
  weightgait/                        # 人物检索与跨动作
  orl_olivetti/                      # 清晰图像特权信息扩展
  washington/                       # RGB-D物体实例验证
  synthetic_and_formula_validation/  # 合成流程与公式/优化核验
  feature_validation/               # G-LBP与KDES核验
  cross_dataset/                    # 混合指标报告与全局审计
```

每个来源下按AP/AUC、Rank-1/CMC、特权比例、训练对数、距离分布、训练时间、收敛/KKT等主题分目录；共享特征、模型、协议和源码快照另存。完整路径映射见manifest；相同文件去重、大文件分块，原文件内容不改动。

校验全部上传产物：

```bash
python tools/restore_experiments.py
```

恢复原实验目录结构，以便使用原报告的相对链接和核验脚本：

```bash
python tools/restore_experiments.py --output /path/to/new_directory
```

输出目录须不存在。恢复后的itml_reproduction/runs包含清单所列的原实验文件；再按itml_reproduction/README.md安装所需依赖。人物图像、视频、下载归档及被替代的失败尝试二进制缓存留在本地，排除原因完整记录；原始数据须按来源与许可另行获取。

## 结论边界

本项目完成核心算法与替代数据实验，未完成原论文原数据的数值复现。ITML+在FaceGrabber和WeightGait的部分比较中优于ITML，但并非所有条件下最优；随机/打乱特权信息也有收益。早期负收益、失败候选和残差保留。重叠人物划分及相关pairs不能当作独立样本证明显著性；WeightGait压缩展示深度不是原始物理深度，normal_A/B不是BIWI Walking/Still。

第三方KDES、YuNet模型及署名/许可见itml_reproduction/third_party。本文档未为第三方数据授予新的使用或再分发许可。
