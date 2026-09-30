# 现有研究代码快照

这些文件从当前工程复制，原件保持不变。来源与 SHA-256 见 `../docs/source_manifest.json`。目录保留原始实现和版权头；没有为缺失的原始训练/控制流程编造替代实现。

来源快照保留原始换行和空格，`.gitattributes` 禁止 Git 自动转换其换行，便于克隆后按来源哈希核对。历史代码中的尾随空格不属于本次算法修改。

## gaussian_grouping

包含当前 `train.py`、渲染/掩码入口及 arguments、scene、utils、gaussian_renderer 模块。上游：[Gaussian Grouping](https://github.com/lkeab/gaussian-grouping)。本地基线提交见来源清单；本地文件有修改，快照哈希用于精确识别。

先根据上游说明准备 Gaussian Splatting CUDA 扩展、PyTorch、COLMAP 相机及实例 ID 掩码数据，再把此目录作为同结构覆盖层使用。未打包子模块、模型权重、DEVA/SAM/LAMA 或训练数据。`config/gaussian_dataset/train.json` 是现存文件快照，不宣称它是 Table 1–8 对应的实际配置。此模块训练对象级高斯表示，不训练双臂 VLA。

## 3dgs_proxy

包含 `crop_gaussian_ply.py`、`split_gaussian_ply_by_plane.py`、网格转换依赖以及 `create_hinged_usd_assembly.py`。`run_split_3dgrut_hinge_pipeline.py` 依赖的两个组装入口已经从现有 leisaac/tools 中一起整理，来源分别记录。

CPU/Open3D 工具依赖见 `../requirements-geometry.txt`。如需导出 Gaussian USDZ，需先配置 [NVIDIA 3DGRUT](https://github.com/nv-tlabs/3dgrut)；USD/PhysX 组装使用 Isaac Sim 的 Python/`pxr`。环境就绪后可先查看：

```bash
python reference_code/3dgs_proxy/split_gaussian_ply_by_plane.py --help
python reference_code/3dgs_proxy/crop_gaussian_ply.py --help
```

全流程脚本保留 Linux 编译器、CUDA 及场景默认值；使用自己的路径/轴线/尺度和输出目录。质量、摩擦和关节参数来自测量或先验，几何转换不自动提供物理验证。

## leisaac

保留对应上游相对目录的策略推理、动作处理、格式转换和铰链交互文件。需完整 [leisaac](https://github.com/LightwheelAI/leisaac) 与 Isaac Lab 环境。原始示教集和模型权重未附带。这些通用接口不是本文关系监督训练器、失败区域验收器或共享 QP 控制器。

## 许可

每个来源目录保留原有 LICENSE 和文件级版权声明。上游依赖的子模块可能采用其他条款，使用时同时保留其原有声明。项目根目录 MIT 不替换这些来源代码的许可证。
