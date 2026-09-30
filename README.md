# More from Less

**3D Gaussian Splatting Digital Twins with Iterative Expansion and Trajectory Compensation for Sparse-Demonstration Bimanual Manipulation**

[Project website](https://robovla.github.io/More-from-Less/) · [GitHub repository](https://github.com/RoboVLA/More-from-Less) · [Source browser](https://robovla.github.io/More-from-Less/code.html) · [Reproducibility](docs/REPRODUCIBILITY.md) · [Method reference](docs/METHOD_REFERENCE.md)

This release provides method reference implementations, reproducible offline trajectory diagnostics, and retained reconstruction and simulator utilities. The reference implementations were added for the release; they are **not recovered historical experiment code**. Original checkpoints and full evaluation logs are unavailable, so this repository does not reproduce Tables 1–8 end to end. The manuscript download is temporarily hidden from the project website.

![Method overview](static/media/v162/figure-1.png)

## Method

1. **Structured bimanual VLA.** Synchronized global and left/right wrist RGB, language, and joint states produce 32 × 12 absolute joint targets. Pose and distance relations provide training-only geometric supervision; contact and role labels are fixed training context.
2. **Physical proxies and failure-region expansion.** 3DGS appearance, simplified collision geometry, and measured/CAD/URDF/prior physics are modeled separately. Only complete successful virtual trajectories that pass task, physics, and safety acceptance enter real-dominant retraining.
3. **Step-level recovery.** Asynchronous image-plane references undergo current-state alignment, depth/calibration mapping, task-masked IK, validity gating, and shared QP/nonlinear checks. Grippers inherit the VLA targets. The approximately two-minute video-generation latency is outside the fast control loop.

## Quick start

Python 3.10 or newer. Clone the repository before running the examples:

```bash
git clone https://github.com/RoboVLA/More-from-Less.git
cd More-from-Less
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on Linux/macOS, then run:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m morefromless.trajectory.reproduce_appendix
python paper_pipeline/run_morefromless_pipeline.py --dry-run --include-disabled
```

The appendix command reconstructs the time-varying fusion, smoothing, and pouring depth treatment from the retained G/R input CSVs. It checks every pose component against the saved F CSVs (72 pouring points and 75 wiping points) and writes results to `outputs/appendix_reproduction/`. Input data are not overwritten. `python paper_pipeline/run_morefromless_pipeline.py` runs this same procedure.

For PyTorch gradient and training-interface checks:

```bash
python -m pip install -r requirements-method.txt
python -m unittest discover -s tests -v
```

Without PyTorch, gradient tests are explicitly skipped. Optional pipeline stages `structured_vla`, `failure_expansion`, and `online_compensation` run engineering checks, not the reported robot experiments. See the [adapter requirements](docs/METHOD_REFERENCE.md) for new training and system integration. An editable install is optional: `python -m pip install -e .`.

## Repository guide

| Location | Contents and scope |
|---|---|
| [Method reference](morefromless/method/) | Differentiable relation supervision/FK, training interface, expansion/acceptance engine, asynchronous references, DLS and shared filtering. Actual model, geometry, simulation and device adapters are required. |
| [Trajectory tools](morefromless/trajectory/) | Pose I/O, generic fusion and exact appendix recomputation; offline diagnostics only. |
| [Trajectory examples](examples/trajectories/) | Retained generated, retargeted and fused trajectories; no new real-robot data. |
| [Reconstruction/proxy snapshots](reference_code/) | Object reconstruction, collision assets, USD/hinge assembly and simulator integration. Upstream environments must be configured separately. |
| [Camera utilities](tools/gemini_camera/) | RGB-D acquisition and tuning; hardware, drivers and SDK required. |
| [Historical video tools](tools/trajectory_compensation/) | Original offline processing scripts with scene-specific defaults. |
| [Pipeline entry](paper_pipeline/) | Appendix reproduction and optional reference-module checks. |
| [Current figures](static/media/v162/) | Figures 1–8 and A.1–A.4 retained by manuscript v164. The directory identifies the figure source version. |
| [Validation record](docs/VALIDATION.md) | Verified scope, preserved data and known limits. |

The [paper-to-code map](docs/PAPER_PIPELINE.md), [source manifest](docs/source_manifest.json), and [asset guide](docs/DATA_AND_ASSETS.md) record provenance and version boundaries.

## Generic fusion example

This fixed-weight example is separate from appendix reproduction:

```bash
python -m morefromless.trajectory.fuse_trajectories --generated examples/trajectories/pour/generated_pose_xyzrpy.csv --real-retargeted examples/trajectories/pour/real_retargeted_pose_xyzrpy.csv --out-dir outputs/pour_demo --real-weight 0.55
```

Input columns are `frame,time_s,x,y,z,rx,ry,rz`. Examples use millimeters, degrees, and camera coordinates (+X right, +Y down, +Z forward). Both inputs must share time origin, frame and units. Component-wise interpolation/fusion holds endpoints outside the reference interval; it is not SE(3) interpolation or a 12-D robot joint command. The example weight is not the online gate weight.

## Website and local preview

The public website is **https://robovla.github.io/More-from-Less/**. GitHub Pages publishes the root of `main`. To preview a clone locally:

```bash
python scripts/serve.py --port 18765
```

Open the loopback address printed by the server. It is a local preview, not the public project address. If the port is occupied, choose another with `--port`; do not stop another project's service. The source browser requires HTTP. See [release and hosting notes](docs/LOCAL_RELEASE.md).

## Dependencies, citation and licensing

NumPy supports the lightweight trajectory, expansion and control references. PyTorch is required for differentiable supervision/FK and training. Optional video and geometry dependencies are in [requirements-vision.txt](requirements-vision.txt) and [requirements-geometry.txt](requirements-geometry.txt). Camera, GPU and Isaac Sim snapshots require their upstream environments.

Use [CITATION.cff](CITATION.cff) for this software release. No journal acceptance or DOI is claimed. Project-authored code is MIT licensed; third-party snapshots retain their own terms. Manuscripts, figures, videos and example data are not automatically covered by the code license. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
