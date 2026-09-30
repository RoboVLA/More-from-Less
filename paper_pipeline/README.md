# 可用流程入口

在仓库根目录运行 `python paper_pipeline/run_morefromless_pipeline.py`，执行倒水与擦拭的离线融合示例。

`--dry-run --include-disabled` 展示完整清单。当前缺失的 VLA 训练、失败区域迭代和在线控制阶段不配置虚构命令，也不能通过 `--include-disabled` 强行运行。配置中的示例参数不是论文训练配置。详见 `../docs/REPRODUCIBILITY.md`。
