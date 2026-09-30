#!/usr/bin/env python3
"""
Split one Gaussian PLY into two parts, run the 3DGRUT USDZ+collision pipeline
for both parts, then create a hinged USD assembly.

The split outputs keep the original coordinate frame. This is important:
both generated USDZ files will align when referenced into the final assembly.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("\n[run]", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split Gaussian PLY into two collision-enabled USDZ parts and create hinged assembly USD.")
    parser.add_argument("input_ply")
    parser.add_argument("--out-dir", default=None)

    parser.add_argument("--pre-filter", action="store_true", help="Filter input PLY before splitting.")
    parser.add_argument("--filtered-ply", default=None, help="Filtered PLY path. Defaults to out-dir/point_cloud_filtered.ply.")
    parser.add_argument("--filter-mode", choices=["cluster-distance", "bbox"], default="cluster-distance")
    parser.add_argument("--filter-min-opacity", type=float, default=0.005)
    parser.add_argument("--filter-max-points", type=int, default=0)
    parser.add_argument("--filter-crop-min", default=None, help="Optional x,y,z lower crop bound.")
    parser.add_argument("--filter-crop-max", default=None, help="Optional x,y,z upper crop bound.")
    parser.add_argument("--filter-cluster-opacity", type=float, default=0.7)
    parser.add_argument("--filter-cluster-eps", type=float, default=0.5)
    parser.add_argument("--filter-cluster-min-points", type=int, default=10)
    parser.add_argument("--filter-cluster-include-radius", type=float, default=0.12)

    split = parser.add_mutually_exclusive_group(required=True)
    split.add_argument("--axis", choices=["x", "y", "z"], help="Axis-aligned split plane.")
    split.add_argument("--plane-normal", help="Non-axis split plane normal x,y,z.")
    split.add_argument("--split-json", help="Use a JSON file produced by pick_hinge_split_open3d.py.")
    parser.add_argument("--value", type=float, default=0.0, help="Axis split value.")
    parser.add_argument("--plane-point", default=None, help="Point on non-axis plane x,y,z.")
    parser.add_argument("--overlap-width", type=float, default=None, help="Overrides overlap_width from --split-json when provided.")

    parser.add_argument("--hinge-point", default=None, help="Point on hinge line x,y,z, in original PLY coordinates.")
    parser.add_argument("--hinge-axis", default=None, help="Hinge axis direction x,y,z.")
    parser.add_argument("--lower-limit-deg", type=float, default=0.0)
    parser.add_argument("--upper-limit-deg", type=float, default=90.0)
    parser.add_argument("--fixed-body", choices=["A", "B", "none"], default="A")

    parser.add_argument("--simplify-target-faces", type=int, default=25000)
    parser.add_argument("--mesh-min-opacity", type=float, default=0.16)
    parser.add_argument("--set-invisible", action="store_true", help="Pass --set_invisible to add_mesh_to_usdz if supported.")

    parser.add_argument("--cc", default="/usr/bin/gcc-12")
    parser.add_argument("--cxx", default="/usr/bin/g++-12")
    parser.add_argument("--cuda-host-cxx", default="/usr/bin/g++-12")
    return parser.parse_args()


def vec_to_text(values) -> str:
    return ",".join(f"{float(value):.9g}" for value in values)


def load_split_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("plane_normal", "plane_point", "hinge_point", "hinge_axis"):
        if key not in data:
            raise ValueError(f"{path} is missing required key: {key}")
    return data


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_ply = Path(args.input_ply)
    if not input_ply.is_file():
        raise FileNotFoundError(input_ply)

    out_dir = Path(args.out_dir) if args.out_dir else input_ply.parent / "split_hinge_output"
    split_dir = out_dir / "split_ply"
    part_a_dir = out_dir / "part_a"
    part_b_dir = out_dir / "part_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    part_a_dir.mkdir(parents=True, exist_ok=True)
    part_b_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CC"] = args.cc
    env["CXX"] = args.cxx
    env["CUDAHOSTCXX"] = args.cuda_host_cxx

    input_for_split = input_ply
    if args.pre_filter:
        filtered_ply = Path(args.filtered_ply) if args.filtered_ply else out_dir / "point_cloud_filtered.ply"
        filter_cmd = [
            sys.executable,
            str(script_dir / "crop_gaussian_ply.py"),
            str(input_ply),
            "--output",
            str(filtered_ply),
            "--filter-mode",
            args.filter_mode,
            "--min-opacity",
            str(args.filter_min_opacity),
            "--cluster-opacity",
            str(args.filter_cluster_opacity),
            "--cluster-eps",
            str(args.filter_cluster_eps),
            "--cluster-min-points",
            str(args.filter_cluster_min_points),
            "--cluster-include-radius",
            str(args.filter_cluster_include_radius),
            "--max-points",
            str(args.filter_max_points),
        ]
        if args.filter_crop_min:
            filter_cmd.append(f"--crop-min={args.filter_crop_min}")
        if args.filter_crop_max:
            filter_cmd.append(f"--crop-max={args.filter_crop_max}")
        run(filter_cmd, env)
        input_for_split = filtered_ply

    split_data = load_split_json(Path(args.split_json)) if args.split_json else None

    plane_normal = args.plane_normal
    plane_point = args.plane_point
    hinge_point = args.hinge_point
    hinge_axis = args.hinge_axis
    overlap_width = args.overlap_width

    if split_data is not None:
        plane_normal = vec_to_text(split_data["plane_normal"])
        plane_point = vec_to_text(split_data["plane_point"])
        hinge_point = hinge_point or vec_to_text(split_data["hinge_point"])
        hinge_axis = hinge_axis or vec_to_text(split_data["hinge_axis"])
        if overlap_width is None:
            overlap_width = float(split_data.get("overlap_width", 0.0))

    if overlap_width is None:
        overlap_width = 0.0

    if not hinge_point or not hinge_axis:
        raise ValueError("--hinge-point and --hinge-axis are required unless --split-json provides them")

    split_cmd = [
        sys.executable,
        str(script_dir / "split_gaussian_ply_by_plane.py"),
        str(input_for_split),
        "--output-dir",
        str(split_dir),
        "--negative-name",
        "part_a",
        "--positive-name",
        "part_b",
        "--overlap-width",
        str(overlap_width),
        f"--hinge-point={hinge_point}",
        f"--hinge-axis={hinge_axis}",
    ]
    if args.axis:
        split_cmd.extend(["--axis", args.axis, "--value", str(args.value)])
    else:
        split_cmd.append(f"--plane-normal={plane_normal}")
        if plane_point:
            split_cmd.append(f"--plane-point={plane_point}")
    run(split_cmd, env)

    common_pipeline_args = [
        "--skip-filter",
        "--mesh-min-opacity",
        str(args.mesh_min_opacity),
        "--simplify-target-faces",
        str(args.simplify_target_faces),
    ]
    if args.set_invisible:
        common_pipeline_args.append("--set-invisible")

    run(
        [
            sys.executable,
            str(script_dir / "run_3dgrut_usdz_collision_pipeline.py"),
            str(split_dir / "part_a.ply"),
            "--out-dir",
            str(part_a_dir),
            "--visual-usdz",
            str(part_a_dir / "part_a.usdz"),
            "--mesh-ply",
            str(part_a_dir / "part_a_mesh.ply"),
            "--output-usdz",
            str(part_a_dir / "part_a_with_collision.usdz"),
            *common_pipeline_args,
        ],
        env,
    )

    run(
        [
            sys.executable,
            str(script_dir / "run_3dgrut_usdz_collision_pipeline.py"),
            str(split_dir / "part_b.ply"),
            "--out-dir",
            str(part_b_dir),
            "--visual-usdz",
            str(part_b_dir / "part_b.usdz"),
            "--mesh-ply",
            str(part_b_dir / "part_b_mesh.ply"),
            "--output-usdz",
            str(part_b_dir / "part_b_with_collision.usdz"),
            *common_pipeline_args,
        ],
        env,
    )

    assembly_path = out_dir / "hinged_assembly.usda"
    run(
        [
            sys.executable,
            str(script_dir / "create_hinged_usd_assembly.py"),
            "--part-a-usdz",
            str(part_a_dir / "part_a_with_collision.usdz"),
            "--part-b-usdz",
            str(part_b_dir / "part_b_with_collision.usdz"),
            "--output",
            str(assembly_path),
            f"--hinge-point={hinge_point}",
            f"--hinge-axis={hinge_axis}",
            "--lower-limit-deg",
            str(args.lower_limit_deg),
            "--upper-limit-deg",
            str(args.upper_limit_deg),
            "--fixed-body",
            args.fixed_body,
        ],
        env,
    )

    print("\n== Done ==")
    print("[part_a_usdz]", part_a_dir / "part_a_with_collision.usdz")
    print("[part_b_usdz]", part_b_dir / "part_b_with_collision.usdz")
    print("[assembly]   ", assembly_path)


if __name__ == "__main__":
    main()



'''

python run_split_3dgrut_hinge_pipeline.py \
  /home/ubuntu/Desktop/gaussian-grouping01/output/microwave/point_cloud_object_removal/iteration_10000/point_cloud_filtered.ply \
  --out-dir /home/ubuntu/Desktop/gaussian-grouping01/output/microwave/point_cloud_object_removal/iteration_10000/hinge_split \
  --split-json /home/ubuntu/Desktop/gaussian-grouping01/output/microwave/point_cloud_object_removal/iteration_10000/manual_split/picked_hinge_split.json \
  --fixed-body A \
  --lower-limit-deg 0 \
  --upper-limit-deg 90 \
  --simplify-target-faces 25000 \
  --set-invisible

'''