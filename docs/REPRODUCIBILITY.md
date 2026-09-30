# Reproducibility and evidence scope

Updated 2026-09-30. The reference manuscript is v164; its experimental results and figures are retained from v162. This release combines retained source snapshots, new method reference implementations, and a verifiable appendix-processing entry point.

| Component | What can be run or checked | What this does not reproduce |
|---|---|---|
| Offline appendix diagnostics | Time-varying fusion, smoothing and pouring depth from G/R; all 72/75 points match saved F CSVs | Online risk, reference adoption or robot recovery |
| Structured-training reference | Shared geometric mapping, normalization, detached context, gradients, training loop and FK | Original checkpoint, demonstrations, loss settings or Table 1 |
| Expansion reference | Failure discovery, local sampling, joint acceptance of complete successful trajectories, group weighting and iteration records | Original Isaac Sim runs, candidate/acceptance counts or Tables 2–3 |
| Control reference | Asynchronous references, task-masked DLS, gating, shared QP and nonlinear-check ordering | Original tracker settings, calibrated geometry, drivers, control frequency or Tables 4–8 |
| Source snapshots | Reconstruction, physical proxies, camera and simulator interfaces with source hashes | Complete upstream environments or the original training system |

See [method interfaces and commands](METHOD_REFERENCE.md). Analytic engineering fixtures do not produce experimental success rates. Reported manuscript values remain reported values; missing original logs are not evidence that those values are incorrect.

## Commands

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m morefromless.trajectory.reproduce_appendix
python paper_pipeline/run_morefromless_pipeline.py --dry-run --include-disabled
python -m pip install -r requirements-method.txt
python -m unittest discover -s tests -v
```

Lightweight processing, expansion and control references use NumPy. Differentiable supervision/FK and training use PyTorch. Geometry, GPU, camera and Isaac Sim snapshots need separately configured environments. Validation did not run robot experiments or recover the original training runs.

## Inputs and interpretation

Policy inputs are synchronized three-view RGB, language and joint states. Output is 32 × 12 absolute joint targets, with five driven joints and one gripper per arm. Contact/role are fixed training context; pose/distance relations provide differentiable supervision. Depth supports proxy scaling, reference mapping and geometry checks, not additional VLA inference inputs.

Offline 6-D camera-frame trajectories are not 12-D joint actions. Depth is modeled and intrinsics are estimated. R participates in constructing F, so reduced F-to-R distance describes internal consistency. Videos retain their documented source meanings.

Website Tables 1, 2, 4 and 6 reproduce the manuscript's summary values. Their evaluation units/task pools differ; Table 1 summarizes 15 paired units. Raw episodes cannot be reconstructed from these aggregates. The [asset manifest](current_manuscript.json) identifies current figures. Historical figures and the old v155 snapshot are no longer in the published file tree. The manuscript link is temporarily hidden from the website.
