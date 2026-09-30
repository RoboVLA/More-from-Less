# 方法参考实现与实验实现的边界

本目录说明 2026-09-30 新增的参考实现。它依据当前论文公式实现可检查的算法流程，使用解析测试样例验证；它不是从已删除文件中恢复的原实验系统，也没有生成或替换论文结果。

| 模块 | 实际提供的实现 | 对应论文 | 接入真实系统时仍需提供 |
|---|---|---|---|
| `morefromless/method/structured.py` | 两臂动作 L1、共享关系映射与归一化、固定训练上下文、有效步掩码、距离安全先验、反向传播和优化器更新 | 式(4)–(7) | VLA 主干/检查点、示教划分、实际损失参数、可微完整几何查询 |
| `morefromless/method/kinematics.py` | 每臂五转动关节的可微串联 FK；相对位姿采用局部 SO(3) 对数 | 第3.1节 | 测量或 URDF 提供的关节原点/轴、基座/工具变换、碰撞几何 |
| `morefromless/method/expansion.py` | 当前策略首轮失败发现、局部有界采样、可行性检查、完整成功轨迹联合验收、去重、迭代再训练回调与逐轮记录 | 算法1、式(18)–(19) | Isaac Sim 场景、状态适配器、真实任务/物理/安全判据、训练适配器 |
| `morefromless/method/control.py` | 异步参考请求、按原观测时间计龄、当前深度平面映射、双臂任务掩码 DLS、门控融合、共享 QP 与非线性复检出口 | 式(24)、(26)–(31) | 视频服务、跟踪/当前状态对齐、雅可比与任务掩码、全碰撞检查、风险量与真实硬件驱动 |

新模块已经包含算法主体。未留存的实际主干、物理模型、运行参数和设备适配器不能由公开公式唯一恢复，因此不能声称可用这些模块复现 Table 1–8。

## 运行与验证

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-method.txt
python -m unittest discover -s tests -v
python paper_pipeline/run_morefromless_pipeline.py --dry-run --include-disabled
python paper_pipeline/run_morefromless_pipeline.py --stage structured_vla
python paper_pipeline/run_morefromless_pipeline.py --stage failure_expansion
python paper_pipeline/run_morefromless_pipeline.py --stage online_compensation
```

最后三条命令运行对应模块的工程检查；不是模型训练、机器人实验或性能评估。未安装 PyTorch 时，通用测试会明确跳过梯度检查，显式请求结构化阶段则报依赖错误。本次验证安装 PyTorch 2.14.0+cpu，全部梯度检查实际执行。

## 结构化训练接入

`train_step` 只向策略传入 `rgb_global / rgb_left / rgb_right / language / joints`，输出必须为 `B×32×12`。训练张量可以由配置指定的均值/尺度归一化；BC与几何计算先还原绝对关节坐标。部署策略适配器直接返回绝对目标。示教和接触/角色上下文停止梯度；预测和示教使用同一反归一化、FK、相对位姿/距离映射与尺度。关系向量为相对平移3维、局部旋转对数3维、最小跨臂距离1维。`relation_diagonal` 位于平方范数内部，实际平方项系数为其平方。接触和角色不新增独立动作损失。

FK 接收五个关节的原点变换和单位旋转轴；不内置猜测的 SO101 尺寸。碰撞距离回调应覆盖实际臂链、夹爪和环境，不能用末端点距离冒充。SO(3) 对数拒绝接近 π 的分支切点。位置、距离及其尺度必须使用一致单位。

可用一个明确的适配工厂进行新训练：

```bash
python -m morefromless.method.train --factory your_adapter:build --config your_run.json --out-dir outputs/new_reference_run
```

`your_adapter` 由实际模型/数据环境提供，不是本仓库内的隐藏依赖。工厂返回 `policy / optimizer / batches / geometry / settings`：`batches()` 可重复遍历，每批提供 `observation / demonstrated / valid_steps / context`。配置必须记录 `checkpoint_identity / dataset_manifest / epochs / seed`，并设置 `implementation_status: new_reference_run`。训练输出保存新检查点和逐批损失，不覆盖旧实验。

## 拓展接入

`expand` 接收真实示教、独立训练/适配初始状态、模拟 rollout、局部采样、可行性和再训练回调。首轮失败来自当前策略在适配集的试运行。候选必须同时满足任务成功、物理有效、安全有效以及完整状态—动作序列；测试集轨迹不能回流。质量权重在虚拟组内部归一化，真实组总权重由显式 `real_mass > 0.5` 控制。再训练回调必须使用传入的组权重，不能重新按池大小直接平均。每轮记录发现次数、失败数、可行状态数、候选数、接受数、累计量和停止原因。

残差探索器仅传入模拟 rollout；在线控制器没有残差策略入口。这里未重建原实验的 RL 训练器。

## 在线参考与过滤接入

`AsyncReferences` 将生成工作放在单个后台工作线程；控制 tick 不等待生成完成。参考龄从请求使用的原始观测计时，返回时不重置。有效期限必须由运行配置显式给出；约两分钟的论文生成延迟不是此代码中假定的控制周期或默认有效期。

`map_reference` 必须针对当前观测执行时间/状态对齐、轨迹有效性、当前深度与标定映射、任务掩码 IK 和可达性检查。生成接口仅交付二维像素参考。`task_masked_reference` 要求双臂雅可比、每臂1–5个独立选中分量和支持臂目标；未观测姿态不保证保持。两个夹爪继承当前 VLA 输出。

风险量按 `u,e,kappa,zeta` 顺序传入，阈值、归一化尺度和门控权重全部显式配置。基础与融合动作均进入共享 QP；候选平滑在约束重建之前完成；门控平滑之后再次施加硬门，等待、失效或低风险不能残留非零参考权重。

QP 使用带正定对角目标的对偶坐标迭代，硬约束无松弛，软松弛有上界；不收敛或约束不可行时返回 hold。通过 QP 后仍必须由调用者提供的完整非线性关节/碰撞检查确认，之后不再修改命令。该便携参考求解器没有实测控制频率承诺。`Decision.command=None` 表示交给硬件保持/减速机制，不会自动转成未经检查的位置命令。没有真实几何和硬件验证时，解析测试不能支持安全认证。

## 附录数据复现

```bash
python -m morefromless.trajectory.reproduce_appendix
```

从原 G/R CSV 重新计算 `w=0.55+0.10 sin(πt)`、五帧截断居中均值、Euler 解绕/回绕；倒水使用 G/R 合并深度的中位数644 mm。历史 JSON 只用于读取处理参数；保存的 F CSV 只用于核对，不作为重算输入。输出写入 `outputs/appendix_reproduction/`，原数据不改动。程序同时逐分量比较全部72/75点和打印论文附录所用统计量；不吻合就失败。

原固定权重 `fuse_trajectories` 仍可作通用示例，但不作为论文附录复现入口。这些离线坐标采用估计内参与建模深度，不能解释为标定误差或在线恢复效果。
