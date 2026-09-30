# 离线轨迹示例

`pour` 和 `wipe` 分别保存既有生成、真实视频重定向及融合轨迹。CSV 字段为 `frame,time_s,x,y,z,rx,ry,rz`；这些示例使用相机系毫米和角度制。JSON 内历史元数据保留原样。

示例是离线诊断材料，不是完整实机回合日志，也不是策略输出的 12 维关节命令。默认运行器读取 generated 与 real_retargeted，并把新演示输出写入被 Git 忽略的 outputs/，不会改写这里的历史 fused 文件。演示参数不能被视为原实验配置。
