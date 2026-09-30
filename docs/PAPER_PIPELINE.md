# 论文与代码对应关系

当前方法说明对应v164匿名稿。当前图表沿用v162；v164更新资源获取与代码发布说明。稿件和图像哈希见 `current_manuscript.json`。

| 论文部分 | 路径 | 状态 |
|---|---|---|
| 式(4)–(7)，结构化监督 | `morefromless/method/structured.py`, `kinematics.py`, `train.py` | 新参考实现；依赖实际主干/数据/几何适配器 |
| 算法1，失败区域拓展 | `morefromless/method/expansion.py` | 新参考实现；模拟与判据由适配器提供 |
| 式(24)、(26)–(31)，映射门控过滤 | `morefromless/method/control.py` | 新参考实现；不连接机器人 |
| 附录A.4离线曲线 | `morefromless/trajectory/reproduce_appendix.py` | 从原G/R重算并核对保存F |
| 对象重建与物理资产 | `reference_code/gaussian_grouping/`, `reference_code/3dgs_proxy/` | 未改动的来源快照 |
| 仿真/相机接口 | `reference_code/leisaac/`, `tools/gemini_camera/` | 未改动的来源快照 |
| Table 1–8原始实验 | 原检查点、划分与回合日志未恢复 | 不宣称已完整复现 |

`paper_pipeline` 默认执行附录复现。显式选择三项方法阶段执行工程合约测试，不会以假参数启动所谓原论文实验。完整API和运行约束见 [METHOD_REFERENCE.md](METHOD_REFERENCE.md)。
