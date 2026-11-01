# 2026-11-01维护归档

本目录保存本次清理前的8个程序/依赖/实现说明文件，供历史复核，不是当前运行入口。原样文件的SHA256见checksums_archived.json；checksums_before.json另记录修改前的9个文件哈希，其中旧主README已重写，未保留逐字副本。其研究历史仍保留在各日期报告和实验目录中。

itml_repro/complete_pipeline.py包含旧ORL固定分片和PID恢复逻辑，使用包内相对导入，不能直接作为独立脚本运行。如需恢复旧环境，应结合实验的完整source_snapshot和依赖。工作模块保留仍被调用的finish_only；现有调用无需改动。

runs内的原脚本、源码快照、模型和实验报告未改。历史工具新增说明单独保存。
