# 流程入口

```bash
python paper_pipeline/run_morefromless_pipeline.py
python paper_pipeline/run_morefromless_pipeline.py --dry-run --include-disabled
```

默认只运行 `appendix_reproduction`，按历史时变融合和平滑过程重算倒水与擦白板曲线，并逐分量核对原CSV。失败即返回非零状态，原数据不改动。

`--stage structured_vla`、`--stage failure_expansion`、`--stage online_compensation` 分别运行新增参考模块的工程测试。第一项需安装 `requirements-method.txt`。配置明确标记 `scope: new_reference_contract_tests` 和 `historical_run_reproduced: false`；测试不替代原训练或实机评测。实际训练/模拟/控制适配器要求见 `../docs/METHOD_REFERENCE.md`。
