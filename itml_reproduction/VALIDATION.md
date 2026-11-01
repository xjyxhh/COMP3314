# 本地验证记录

日期：2026-10-26。使用 Codex bundled Python 3.12 与 NumPy，项目本身仅依赖 NumPy。

- `python -m unittest discover -s tests -v`：5 项测试全部通过。
- `python -m itml_repro.demo --output data/synthetic`：生成8身份×5样本的配对合成特征。
- `python -m itml_repro.run --config configs/smoke.json --manifest data/synthetic/manifest.csv --features data/synthetic/features.npz --output runs/smoke_verified`：两轮、三种方法、完整网格/验证选参/测试/落盘流程成功。

所选 ITML+ 模型的最终诊断（不是论文数据结果）：

| round | 停止原因 | 式(4)最大不等式违例 | ξ/Depth最大等式残差 |
|---|---|---:|---:|
| 0 | max_sweeps (10) | 0.0235213 | 0.0284179 |
| 1 | max_sweeps (10) | 4.38877 | 2.29712 |

M、P 保持正定；这仍不等于全局约束满足。保留论文公开更新步骤而不擅自修改优化器，观察到的非零残差明确暴露在输出中。这个合成例子也未显示 ITML+ 性能优势，不能用于支持真实任务上的性能主张。

未开展真实 EURECOM/CurtinFaces 数据训练、作者代码数值对照或真实 GLBP 特征一致性验证。正式实验前须接入经核验的特征，并明确 FORMULAS.md 中的 u/l 与距离单位约定。
