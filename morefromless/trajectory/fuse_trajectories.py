from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .pose_io import PoseSample, load_pose_trajectory, save_pose_csv, save_pose_json


def _as_array(samples: list[PoseSample]) -> np.ndarray:
    return np.array(
        [[s.time_s, s.x, s.y, s.z, s.rx, s.ry, s.rz] for s in samples],
        dtype=np.float64,
    )


def _resample(samples: list[PoseSample], target_time: np.ndarray) -> np.ndarray:
    source = _as_array(samples)
    source_time = source[:, 0]
    result = np.zeros((len(target_time), 6), dtype=np.float64)
    for i in range(6):
        result[:, i] = np.interp(target_time, source_time, source[:, i + 1])
    return result


def fuse_trajectories(
    generated: list[PoseSample],
    real_retargeted: list[PoseSample],
    real_weight: float,
    keep_generated_z: bool,
) -> list[PoseSample]:
    if not generated:
        raise ValueError("Generated trajectory is empty.")
    if not real_retargeted:
        raise ValueError("Real-retargeted trajectory is empty.")
    if not 0.0 <= real_weight <= 1.0:
        raise ValueError("real_weight must be in [0, 1].")

    gen = _as_array(generated)
    target_time = gen[:, 0]
    real = _resample(real_retargeted, target_time)
    fused_values = (1.0 - real_weight) * gen[:, 1:] + real_weight * real
    if keep_generated_z:
        fused_values[:, 2] = gen[:, 3]

    fused: list[PoseSample] = []
    for i, row in enumerate(fused_values):
        fused.append(
            PoseSample(
                frame=generated[i].frame,
                time_s=float(target_time[i]),
                x=float(row[0]),
                y=float(row[1]),
                z=float(row[2]),
                rx=float(row[3]),
                ry=float(row[4]),
                rz=float(row[5]),
            )
        )
    return fused


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse generated and real-retargeted 6D trajectories.")
    parser.add_argument("--generated", required=True, type=Path, help="CSV or JSON generated-video trajectory.")
    parser.add_argument("--real-retargeted", required=True, type=Path, help="CSV or JSON real-video retargeted trajectory.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--real-weight", type=float, default=0.55, help="Weight for real-retargeted path features.")
    parser.add_argument(
        "--keep-generated-z",
        action="store_true",
        help="Keep generated Z instead of fusing Z. Use this when Z is already metric-stabilized.",
    )
    args = parser.parse_args()

    generated = load_pose_trajectory(args.generated)
    real_retargeted = load_pose_trajectory(args.real_retargeted)
    fused = fuse_trajectories(generated, real_retargeted, args.real_weight, args.keep_generated_z)

    metadata = {
        "generated": str(args.generated),
        "real_retargeted": str(args.real_retargeted),
        "real_weight": args.real_weight,
        "keep_generated_z": args.keep_generated_z,
        "coordinate_frame": "+X right, +Y down, +Z forward",
        "position_unit": "mm",
        "rotation_unit": "deg",
    }
    save_pose_csv(args.out_dir / "fused_pose_xyzrpy.csv", fused)
    save_pose_json(args.out_dir / "fused_pose_trajectory_xyzrpy.json", fused, metadata)
    print(f"Wrote {len(fused)} fused samples to {args.out_dir}")


if __name__ == "__main__":
    main()

