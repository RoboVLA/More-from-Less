#!/usr/bin/env python3
"""
Run the full 3DGRUT USDZ + collision-mesh pipeline.

Pipeline:
1. Filter/crop Gaussian PLY while preserving all 3DGS fields.
2. Convert the filtered Gaussian PLY to USDZ with 3DGRUT.
3. Generate a 3DGRUT-compatible triangle mesh PLY from the filtered Gaussian PLY.
4. Add the mesh PLY to the USDZ with collision enabled.

Example:
    python tools/run_3dgrut_usdz_collision_pipeline.py \
        /media/ubuntu/L/output/hu/point_cloud_object_removal/iteration_10000/point_cloud.ply
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_INPUT = "/media/ubuntu/L/output/hu/point_cloud_object_removal/iteration_10000/point_cloud.ply"


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("\n[run]", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter Gaussian PLY, create USDZ and mesh PLY, then add collision mesh to USDZ.")
    parser.add_argument("input_ply", nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", default=None, help="Defaults to input PLY directory.")

    parser.add_argument("--filtered-ply", default=None)
    parser.add_argument("--visual-usdz", default=None)
    parser.add_argument("--mesh-ply", default=None)
    parser.add_argument("--output-usdz", default=None)

    parser.add_argument("--cc", default="/usr/bin/gcc-12")
    parser.add_argument("--cxx", default="/usr/bin/g++-12")
    parser.add_argument("--cuda-host-cxx", default="/usr/bin/g++-12")

    parser.add_argument("--filter-min-opacity", type=float, default=0.005)
    parser.add_argument("--skip-filter", action="store_true", help="Use input_ply directly for 3DGRUT visual and mesh generation.")
    parser.add_argument(
        "--use-existing-filtered-ply",
        action="store_true",
        help="Do not run crop_gaussian_ply.py; use --filtered-ply as the already tuned collision/filter PLY.",
    )
    parser.add_argument(
        "--preserve-visual",
        action="store_true",
        help="Convert the original input PLY to the visible USDZ, and use the filtered PLY only for the collision mesh.",
    )
    parser.add_argument("--filter-mode", choices=["cluster-distance", "bbox"], default="cluster-distance")
    parser.add_argument("--filter-cluster-margin", type=float, default=0.12)
    parser.add_argument("--filter-cluster-opacity", type=float, default=0.7)
    parser.add_argument("--filter-cluster-eps", type=float, default=0.5)
    parser.add_argument("--filter-cluster-min-points", type=int, default=10)
    parser.add_argument("--filter-cluster-include-radius", type=float, default=0.12)
    parser.add_argument("--filter-max-points", type=int, default=0)
    parser.add_argument("--mesh-min-opacity", type=float, default=0.16)
    parser.add_argument("--simplify-target-faces", type=int, default=25000)
    parser.add_argument("--set-invisible", action="store_true", help="Also pass --set_invisible to add_mesh_to_usdz if your 3DGRUT version supports it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    input_ply = Path(args.input_ply)
    out_dir = Path(args.out_dir) if args.out_dir else input_ply.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    filtered_ply = Path(args.filtered_ply) if args.filtered_ply else out_dir / "point_cloud_filtered.ply"
    visual_usdz = Path(args.visual_usdz) if args.visual_usdz else out_dir / "point_cloud.usdz"
    mesh_ply = Path(args.mesh_ply) if args.mesh_ply else out_dir / "point_cloud_mesh.ply"
    output_usdz = Path(args.output_usdz) if args.output_usdz else out_dir / "point_cloudoutput.usdz"

    if not input_ply.is_file():
        raise FileNotFoundError(input_ply)

    env = os.environ.copy()
    env["CC"] = args.cc
    env["CXX"] = args.cxx
    env["CUDAHOSTCXX"] = args.cuda_host_cxx

    print("[env] CC =", env["CC"])
    print("[env] CXX =", env["CXX"])
    print("[env] CUDAHOSTCXX =", env["CUDAHOSTCXX"])
    print("[input]", input_ply)
    print("[out_dir]", out_dir)

    if args.use_existing_filtered_ply:
        if args.filtered_ply is None:
            raise ValueError("--use-existing-filtered-ply requires --filtered-ply")
        if not filtered_ply.is_file():
            raise FileNotFoundError(filtered_ply)
        print("[filter] using existing filtered PLY:", filtered_ply)
    elif args.skip_filter:
        filtered_ply = input_ply
        print("[filter] skipped; using input PLY directly")
    else:
        run(
            [
                sys.executable,
                str(script_dir / "crop_gaussian_ply.py"),
                str(input_ply),
                "--output",
                str(filtered_ply),
                "--min-opacity",
                str(args.filter_min_opacity),
                "--filter-mode",
                str(args.filter_mode),
                "--max-points",
                str(args.filter_max_points),
                "--cluster-margin",
                str(args.filter_cluster_margin),
                "--cluster-opacity",
                str(args.filter_cluster_opacity),
                "--cluster-eps",
                str(args.filter_cluster_eps),
                "--cluster-min-points",
                str(args.filter_cluster_min_points),
                "--cluster-include-radius",
                str(args.filter_cluster_include_radius),
            ],
            env,
        )

    visual_input_ply = input_ply if args.preserve_visual else filtered_ply
    print("[visual_input]", visual_input_ply)
    print("[collision_input]", filtered_ply)

    run(
        [
            sys.executable,
            "-m",
            "threedgrut.export.scripts.ply_to_usd",
            str(visual_input_ply),
            "--output_file",
            str(visual_usdz),
        ],
        env,
    )

    run(
        [
            sys.executable,
            str(script_dir / "gaussian_ply_to_3dgrut_mesh_ply.py"),
            str(filtered_ply),
            "--output",
            str(mesh_ply),
            "--simplify-target-faces",
            str(args.simplify_target_faces),
            "--min-opacity",
            str(args.mesh_min_opacity),
            "--no-auto-crop-largest-cluster",
        ],
        env,
    )

    add_cmd = [
        sys.executable,
        "-m",
        "threedgrut.export.scripts.add_mesh_to_usdz",
        "--input_usdz",
        str(visual_usdz),
        "--output_usdz",
        str(output_usdz),
        "--mesh_ply",
        str(mesh_ply),
        "--set_collision",
    ]
    if args.set_invisible:
        add_cmd.append("--set_invisible")

    run(add_cmd, env)

    print("\n== Done ==")
    print("[filtered_ply]", filtered_ply)
    print("[visual_usdz] ", visual_usdz)
    print("[mesh_ply]    ", mesh_ply)
    print("[output_usdz] ", output_usdz)


if __name__ == "__main__":
    main()
