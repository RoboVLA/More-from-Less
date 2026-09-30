# Pipeline entry point

From the repository root:

```bash
python paper_pipeline/run_morefromless_pipeline.py
python paper_pipeline/run_morefromless_pipeline.py --dry-run --include-disabled
```

The default stage is `appendix_reproduction`. It reconstructs pouring and wiping curves with the historical time-varying weights and smoothing, comparing every component with the saved CSVs. Mismatch returns a nonzero exit status. Original data are not overwritten.

`--stage structured_vla`, `--stage failure_expansion`, and `--stage online_compensation` run engineering tests of the reference modules. The first requires [requirements-method.txt](../requirements-method.txt). Configuration explicitly records `scope: new_reference_contract_tests` and `historical_run_reproduced: false`. These tests do not replace training or robot evaluation. See [adapter requirements](../docs/METHOD_REFERENCE.md).
