# 当前数据与素材

- `examples/trajectories/` 保存原倒水/擦白板CSV和JSON。本次没有修改这些文件；新增入口将重算结果写入 `outputs/appendix_reproduction/`。
- `static/media/v162/` 是当前v164稿件继续使用的图1–8、A.1–A.4。版本目录标识图像来源，图像没有因资源说明更新而重绘。
- 34个历史图片与旧v155快照已经完整归档到公开仓库之外。`static/media/current/`、`static/media/paper/` 不再是发布资源；旧A.3示意响应曲线不再出现在当前文件树。归档哈希见 `historical_assets.json`。
- `static/media/experiments/real_*_global.mp4` 是已有真实操作短片，仅展示任务交互，不构成补偿触发或恢复结果的同步证据。
- `paper/anonymous_review.pdf` 是当前v164匿名阅读稿；图注、版本和SHA-256由 `current_manuscript.json` 索引。
- 完整示教、原始评测日志、模型权重和大型场景/点云不在本仓库中。

新增方法代码的解析测试样例不是论文实验数据。表格汇总值直接取自论文，不能通过测试生成。媒体、论文与实验数据不自动适用代码许可证。
