# 论文与代码对应关系

当前方法说明对应基于 v162 的匿名英文稿，见 `docs/current_manuscript.json` 和 `paper/anonymous_review.pdf`。`docs/manuscript_snapshot.json` 仅保留旧 v155 稿件快照。代码来源快照未因稿件更新而替换。

| 方法 | 对应路径 | 使用方式 |
|---|---|---|
| 对象级 3DGS 外观建模 | `reference_code/gaussian_grouping/` | 配合 Gaussian Grouping 上游环境、相机与实例掩码 |
| 高斯对象与碰撞资产处理 | `reference_code/3dgs_proxy/` | 裁剪、分割、网格、USDZ 和铰链组装参考代码 |
| 仿真策略和数据接口 | `reference_code/leisaac/` | 覆盖到对应上游目录后配置 Isaac Sim；不是完整拓展循环 |
| RGB-D 采集 | `tools/gemini_camera/` | 配合设备与驱动 |
| 离线轨迹诊断 | `morefromless/trajectory/`, `tools/trajectory_compensation/` | 示例读写与融合，历史视频处理脚本 |
| 三项主贡献的完整运行系统 | 未完整打包 | 见 `REPRODUCIBILITY.md`，不能用示例运行器替代 |

`paper_pipeline` 只执行有输入、可用的离线示例。`--dry-run --include-disabled` 会显示缺失阶段；显式运行缺失阶段会以非零状态退出，不创建伪实验结果。
