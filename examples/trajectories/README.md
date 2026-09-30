# Offline trajectory examples

[pour](pour/) and [wipe](wipe/) contain retained generated (G), real-video-retargeted (R), and fused (F) trajectories. CSV columns are `frame,time_s,x,y,z,rx,ry,rz`, using camera-frame millimeters and degrees.

These are offline diagnostics, not complete robot evaluation logs or 12-D policy actions. Run `python -m morefromless.trajectory.reproduce_appendix` from the repository root. It reads G/R, reproduces the saved processing, and compares against F without overwriting any input. Results go to the Git-ignored `outputs/appendix_reproduction/` directory.

Public JSON metadata lists portable CSV references under `release_files`. The old workstation-specific `generated_pose_json`, `real_retarget_pose_json`, `target_video`, and `output_dir` fields are null because those external artifacts are not shipped at those paths. Original metadata is backed up outside this repository. Every numerical trajectory value and historical processing parameter is retained. The provided real-task teaser videos are not substitutes for the missing generated-video sources.

Coordinate intrinsics are estimated and depth is modeled. R contributes to F; lower F-to-R distance is an internal consistency measure, not independent tracking accuracy or online recovery performance.
