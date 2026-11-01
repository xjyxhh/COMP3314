# 历史工具状态（2026-11-01维护补充）

本目录为已完成的固定实验记录。原脚本、源码快照、模型和原报告均保留。

|程序|状态与用途|
|---|---|
|schedule_memory.py|一次性SIGSTOP/SIGCONT内存调度；研究已完成，不作为新实验调度入口。|
|complete_weight_parallel.py|补跑并合并第五轮的恢复工具；硬编码当时目录和进程，退出当前运行流程。|
|benchmark_qp.py|固定训练输入上的求解器探索，保留作历史证据。|
|acquire_facegrabber.py|初始selection生成及获取；不能由读取既有selection的fast版本完全替代。|
|acquire_facegrabber_fast.py|既有selection的续传及补充深度获取。|
|其余study、diagnostics、report脚本|固定实验复跑/核验证据；会写固定输出，不应用于已完成目录的日常重跑。|

新研究请采用独立输出目录及项目README中的准备、训练、分析、汇总入口。原脚本原样保留以维持来源与哈希可追溯性。
