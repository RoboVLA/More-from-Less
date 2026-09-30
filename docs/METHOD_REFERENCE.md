# Method reference implementations

The reference modules added on 2026-09-30 implement inspectable algorithmic steps from the current manuscript and use analytic test fixtures. They are not recovered historical experimental code and did not generate or replace the reported results.

| Module | Provided implementation | Paper mapping | Required system adapters |
|---|---|---|---|
| [structured.py](../morefromless/method/structured.py) | Bimanual L1 actions, shared relation mapping/normalization, detached context, valid-step masks, distance safety prior, gradients and optimizer update | Eqs. (4)–(7) | VLA/checkpoint, data splits, loss parameters, differentiable complete geometry |
| [kinematics.py](../morefromless/method/kinematics.py) | Differentiable five-revolute-joint FK per arm; local SO(3) log for relative pose | Section 3.1 | Measured/URDF origins and axes, base/tool transforms, collision geometry |
| [expansion.py](../morefromless/method/expansion.py) | Initial failure discovery, bounded sampling, feasibility checks, joint acceptance, deduplication, retraining callbacks and iteration records | Algorithm 1; Eqs. (18)–(19) | Isaac Sim scene, state adapter, task/physics/safety criteria, training adapter |
| [control.py](../morefromless/method/control.py) | Async requests, capture-time age, depth mapping, bimanual masked DLS, gated fusion, shared QP and nonlinear checks | Eqs. (24), (26)–(31) | Video service, tracking/alignment, Jacobians/masks, full collision checks, risk signals, hardware drivers |

The original models, physical parameters, run settings and device adapters cannot be uniquely recovered from the formulas. These modules alone do not reproduce Tables 1–8.

## Run and validate

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-method.txt
python -m unittest discover -s tests -v
python paper_pipeline/run_morefromless_pipeline.py --dry-run --include-disabled
python paper_pipeline/run_morefromless_pipeline.py --stage structured_vla
python paper_pipeline/run_morefromless_pipeline.py --stage failure_expansion
python paper_pipeline/run_morefromless_pipeline.py --stage online_compensation
```

The final three commands run engineering checks, not training, robot experiments or performance evaluation. Without PyTorch, generic tests explicitly skip gradient checks; explicitly selecting the structured stage reports the missing dependency. The recorded 24-test validation used CPU PyTorch 2.14.0+cpu and executed the gradient checks.

## Structured training

`train_step` passes only `rgb_global / rgb_left / rgb_right / language / joints` to the policy. Output shape is `B × 32 × 12`. Configured means/scales may normalize training tensors; BC and geometry computations first restore absolute joint coordinates. The deployment adapter returns absolute targets directly. Demonstrations and contact/role context are detached. Predictions and demonstrations share denormalization, FK, relative-pose/distance mapping and scales.

The relation vector contains three relative-translation components, three local rotation-log components and one minimum cross-arm distance. `relation_diagonal` is inside the squared norm, so its squared entries are the effective squared-error coefficients. Contact and role do not introduce separate differentiable action losses.

FK takes five joint-origin transforms and unit rotation axes. It does not assume robot dimensions. The collision callback must cover actual arm links, grippers and environment; endpoint distance is not a substitute. The SO(3) log rejects configurations near the pi branch cut. Position, distance and normalization scales must use consistent units.

New training can use an explicit adapter factory:

```bash
python -m morefromless.method.train --factory your_adapter:build --config your_run.json --out-dir outputs/new_reference_run
```

`your_adapter` must be supplied by the actual model/data environment. It returns `policy / optimizer / batches / geometry / settings`. The repeatable `batches()` iterator supplies `observation / demonstrated / valid_steps / context`. Configuration records `checkpoint_identity / dataset_manifest / epochs / seed` and `implementation_status: new_reference_run`. Outputs contain new checkpoints and per-batch losses without overwriting historical experiments.

## Expansion

`expand` accepts real demonstrations, separate training/adaptation initial states, simulated rollouts, local sampling, feasibility and retraining callbacks. First-round failures come from the current policy on the adaptation set. Accepted candidates must satisfy task success, physics validity, safety validity and complete state–action sequences. Evaluation trajectories must not enter retraining.

Quality weights are normalized within the virtual group. Explicit `real_mass > 0.5` controls the real group's total weight. Retraining must use the supplied weights rather than average directly by pool size. Each iteration records discovery, failures, feasible states, candidates, accepted trajectories, cumulative count and stopping reason. Residual exploration is passed only to simulated rollouts; the online controller has no residual-policy entry. The original RL trainer is not reconstructed.

## Online references and filtering

`AsyncReferences` uses one background worker; a control tick does not wait for generation. Reference age starts at the observation used by the request, not at generation completion. Lifetime must be supplied explicitly. The approximately two-minute generation latency is neither a control period nor an assumed default lifetime.

`map_reference` must check timestamp/current-state alignment, tracking validity, depth/calibration mapping, task-masked IK and reachability against the current observation. The provider returns only image-plane targets. `task_masked_reference` requires both Jacobians, one to five independent selected components per arm and a supporting-arm target. It does not guarantee preservation of unobserved orientation. Both grippers inherit current VLA outputs.

Risk inputs are ordered `u,e,kappa,zeta`; thresholds, scales and gate weights are explicit. Base and fused actions both enter the shared QP. Candidate smoothing precedes constraint reconstruction. A hard gate is reapplied after gate smoothing, so waiting, invalid or low-risk states cannot retain a nonzero reference weight.

The portable QP uses dual coordinate iteration with a positive diagonal objective, hard constraints without slack and bounded soft slack. Non-convergence or infeasibility returns hold. A caller-provided full nonlinear joint/collision check must accept the QP result, and the command is not modified afterward. No measured control-frequency claim is made. `Decision.command=None` delegates to the hardware hold/slow mechanism; it is not an unchecked position command. Analytic tests cannot establish hardware safety certification.

## Appendix reproduction

```bash
python -m morefromless.trajectory.reproduce_appendix
```

The command recomputes `w = 0.55 + 0.10 sin(pi*t)`, a truncated centered five-frame mean, and Euler unwrapping/wrapping from G/R CSVs. Pouring uses the combined G/R depth median, 644 mm. Historical JSON supplies processing parameters; saved F CSVs are comparison targets, not reconstruction inputs. Outputs go to `outputs/appendix_reproduction/`. All components of the 72/75 points are compared, and a mismatch fails the command.

The generic fixed-weight `fuse_trajectories` example is separate. Estimated intrinsics and modeled depth mean these curves describe offline consistency, not calibrated pose error or online recovery. Public JSON copies use portable file references; their trajectory values and processing parameters are unchanged.
