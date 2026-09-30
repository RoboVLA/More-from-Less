#!/usr/bin/env python3
"""
Interactively pick a hinge split with Open3D.

Pick three points in this order:
  1. one endpoint on the hinge line
  2. another endpoint on the hinge line
  3. a point on the desired cut plane

The script writes a JSON file that can be passed to:
  run_split_3dgrut_hinge_pipeline.py --split-json path/to/json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ply_to_mesh import extract_points_colors_normals, read_vertex_ply


def normalized(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        raise ValueError("Vector length is zero")
    return vec / norm


def canonical_axis(vec: np.ndarray) -> np.ndarray:
    vec = normalized(vec)
    idx = int(np.argmax(np.abs(vec)))
    if vec[idx] < 0.0:
        vec = -vec
    return vec


def vec_text(vec: np.ndarray) -> str:
    return ",".join(f"{float(v):.9g}" for v in vec)


def run_filter(input_path: Path, args: argparse.Namespace) -> Path:
    if args.filtered_ply:
        filtered_path = Path(args.filtered_ply)
    elif args.output_dir:
        filtered_path = Path(args.output_dir) / f"{input_path.stem}_filtered.ply"
    else:
        filtered_path = input_path.with_name(f"{input_path.stem}_filtered.ply")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "crop_gaussian_ply.py"),
        str(input_path),
        "--output",
        str(filtered_path),
        "--filter-mode",
        args.filter_mode,
        "--min-opacity",
        str(args.filter_min_opacity),
        "--cluster-opacity",
        str(args.cluster_opacity),
        "--cluster-eps",
        str(args.cluster_eps),
        "--cluster-min-points",
        str(args.cluster_min_points),
        "--cluster-include-radius",
        str(args.cluster_include_radius),
        "--max-points",
        str(args.filter_max_points),
    ]
    if args.crop_min:
        cmd.append(f"--crop-min={args.crop_min}")
    if args.crop_max:
        cmd.append(f"--crop-max={args.crop_max}")
    print("[filter]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return filtered_path


def load_visual_points(path: Path, min_opacity: float, max_points: int, seed: int):
    vertices, names = read_vertex_ply(path)
    points, colors, _ = extract_points_colors_normals(vertices, names, min_opacity=min_opacity)
    if len(points) == 0:
        raise ValueError("No valid points to show")
    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(int(seed))
        indices = rng.choice(len(points), size=int(max_points), replace=False)
        indices = np.sort(indices)
        points = points[indices]
        colors = colors[indices]
    return points, colors


def make_split_preview_geometries(
    o3d,
    points: np.ndarray,
    colors: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    hinge_point: np.ndarray,
    hinge_axis: np.ndarray,
    overlap_width: float,
):
    signed = (points - plane_point[None, :]) @ plane_normal
    side_colors = np.empty_like(colors)
    side_colors[signed < 0.0] = np.array([0.10, 0.35, 1.00])
    side_colors[signed >= 0.0] = np.array([1.00, 0.42, 0.08])
    if overlap_width > 0.0:
        side_colors[np.abs(signed) <= overlap_width] = np.array([1.00, 0.92, 0.05])
    preview_colors = np.clip(colors * 0.35 + side_colors * 0.65, 0.0, 1.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(preview_colors)

    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    diag = max(float(np.linalg.norm(pmax - pmin)), 1e-6)
    half_hinge = diag * 0.35

    plane_dir = np.cross(plane_normal, hinge_axis)
    if np.linalg.norm(plane_dir) < 1e-8:
        plane_dir = np.cross(plane_normal, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(plane_dir) < 1e-8:
            plane_dir = np.cross(plane_normal, np.array([0.0, 1.0, 0.0]))
    plane_dir = normalized(plane_dir)

    plane_half_u = diag * 0.45
    plane_half_v = diag * 0.30
    corners = np.array(
        [
            hinge_point - hinge_axis * plane_half_u - plane_dir * plane_half_v,
            hinge_point + hinge_axis * plane_half_u - plane_dir * plane_half_v,
            hinge_point + hinge_axis * plane_half_u + plane_dir * plane_half_v,
            hinge_point - hinge_axis * plane_half_u + plane_dir * plane_half_v,
        ],
        dtype=np.float64,
    )
    hinge_line = np.array(
        [
            hinge_point - hinge_axis * half_hinge,
            hinge_point + hinge_axis * half_hinge,
        ],
        dtype=np.float64,
    )
    line_points = np.vstack([hinge_line, corners])
    line_indices = [
        [0, 1],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 2],
        [2, 4],
        [3, 5],
    ]
    line_colors = [
        [0.0, 1.0, 0.1],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.75, 0.75, 0.75],
        [0.75, 0.75, 0.75],
    ]
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(line_points)
    lines.lines = o3d.utility.Vector2iVector(line_indices)
    lines.colors = o3d.utility.Vector3dVector(line_colors)

    return [pcd, lines]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pick hinge split points in an Open3D window.")
    parser.add_argument("input_ply")
    parser.add_argument("--output-dir", default=None, help="Directory for picked JSON, optional filtered PLY, and optional split preview.")
    parser.add_argument("--output-json", default=None)

    parser.add_argument("--filter-first", action="store_true", help="Filter the input Gaussian PLY before picking and splitting.")
    parser.add_argument("--filtered-ply", default=None, help="Filtered PLY output path. Defaults to input stem + _filtered.ply.")
    parser.add_argument("--filter-mode", choices=["cluster-distance", "bbox"], default="cluster-distance")
    parser.add_argument("--filter-min-opacity", type=float, default=0.3)
    parser.add_argument("--filter-max-points", type=int, default=0)
    parser.add_argument("--crop-min", default=None, help="Optional x,y,z lower crop bound for filtering.")
    parser.add_argument("--crop-max", default=None, help="Optional x,y,z upper crop bound for filtering.")
    parser.add_argument("--cluster-opacity", type=float, default=0.1)
    parser.add_argument("--cluster-eps", type=float, default=0.5)
    parser.add_argument("--cluster-min-points", type=int, default=10)
    parser.add_argument("--cluster-include-radius", type=float, default=0.12)

    parser.add_argument("--min-opacity", type=float, default=0.05)
    parser.add_argument("--max-points", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overlap-width", type=float, default=0.0, help="Duplicate points inside this cut band when --run-split is used.")
    parser.add_argument("--negative-name", default="part_a")
    parser.add_argument("--positive-name", default="part_b")
    parser.add_argument("--no-preview", action="store_true", help="Do not show the colored split preview after picking.")
    parser.add_argument("--run-split", action="store_true", help="Run split_gaussian_ply_by_plane.py after picking.")
    parser.add_argument("--split-output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_input_path = Path(args.input_ply)
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    input_path = run_filter(source_input_path, args) if args.filter_first else source_input_path
    if args.output_json:
        output_json = Path(args.output_json)
    elif output_dir is not None:
        output_json = output_dir / "picked_hinge_split.json"
    else:
        output_json = input_path.with_name("picked_hinge_split.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    import open3d as o3d

    points, colors = load_visual_points(input_path, float(args.min_opacity), int(args.max_points), int(args.seed))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("\nOpen3D picker controls:")
    print("  Shift + left click : pick a point")
    print("  Shift + right click: undo the last pick")
    print("  Q or Esc           : finish")
    print("\nPick exactly three points:")
    print("  1) hinge line point A")
    print("  2) hinge line point B")
    print("  3) any point on the desired split plane")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Pick hinge split: A, B, plane point", width=1280, height=800)
    vis.add_geometry(pcd)
    vis.run()
    picked = list(vis.get_picked_points())
    vis.destroy_window()

    if len(picked) < 3:
        raise ValueError(f"Need at least 3 picked points, got {len(picked)}")

    p0 = points[picked[0]]
    p1 = points[picked[1]]
    p2 = points[picked[2]]
    hinge_axis = canonical_axis(p1 - p0)
    plane_normal = normalized(np.cross(hinge_axis, p2 - p0))
    if float(np.linalg.norm(plane_normal)) <= 1e-12:
        raise ValueError("Picked points are collinear; pick a third point away from the hinge line")
    plane_normal = canonical_axis(plane_normal)
    hinge_point = 0.5 * (p0 + p1)
    plane_point = p0

    metadata = {
        "source_input": str(source_input_path),
        "input": str(input_path),
        "filtered": bool(args.filter_first),
        "method": "open3d_three_point_picker",
        "picked_indices": picked[:3],
        "picked_points": [p0.tolist(), p1.tolist(), p2.tolist()],
        "plane_point": plane_point.tolist(),
        "plane_normal": plane_normal.tolist(),
        "hinge_point": hinge_point.tolist(),
        "hinge_axis": hinge_axis.tolist(),
        "overlap_width": float(args.overlap_width),
        "negative_name": args.negative_name,
        "positive_name": args.positive_name,
    }
    output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[save] {output_json.resolve()}")
    print(f"[plane] normal={vec_text(plane_normal)} point={vec_text(plane_point)}")
    print(f"[hinge] axis={vec_text(hinge_axis)} point={vec_text(hinge_point)}")
    print("[pipeline args]")
    print("  --split-json " + str(output_json.resolve()))

    if not args.no_preview:
        print("\nPreview colors:")
        print("  blue   : part_a / negative side")
        print("  orange : part_b / positive side")
        print("  yellow : overlap band")
        print("  green  : hinge axis")
        geometries = make_split_preview_geometries(
            o3d,
            points,
            colors,
            plane_point,
            plane_normal,
            hinge_point,
            hinge_axis,
            max(0.0, float(args.overlap_width)),
        )
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Preview split: blue=part_a, orange=part_b, green=hinge",
            width=1280,
            height=800,
        )

    if args.run_split:
        if args.split_output_dir:
            split_output_dir = Path(args.split_output_dir)
        elif output_dir is not None:
            split_output_dir = output_dir / "split_preview"
        else:
            split_output_dir = input_path.parent / "picked_split_ply"
        split_output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "split_gaussian_ply_by_plane.py"),
                str(input_path),
                "--output-dir",
                str(split_output_dir),
                "--negative-name",
                args.negative_name,
                "--positive-name",
                args.positive_name,
                "--overlap-width",
                str(args.overlap_width),
                f"--plane-normal={vec_text(plane_normal)}",
                f"--plane-point={vec_text(plane_point)}",
                f"--hinge-point={vec_text(hinge_point)}",
                f"--hinge-axis={vec_text(hinge_axis)}",
            ],
            check=True,
        )
        print("\n== Saved files ==")
        if args.filter_first:
            print(f"[filtered_ply] {input_path.resolve()}")
        print(f"[split_json]   {output_json.resolve()}")
        print(f"[part_a_ply]   {(split_output_dir / (args.negative_name + '.ply')).resolve()}")
        print(f"[part_b_ply]   {(split_output_dir / (args.positive_name + '.ply')).resolve()}")
        print(f"[metadata]     {(split_output_dir / 'split_metadata.json').resolve()}")
    else:
        print("\n== Saved files ==")
        if args.filter_first:
            print(f"[filtered_ply] {input_path.resolve()}")
        print(f"[split_json]   {output_json.resolve()}")
        print("[note] No part_a/part_b PLY files were generated because --run-split was not set.")


if __name__ == "__main__":
    main()

'''

python pick_hinge_split_open3d.py \
  /home/ubuntu/Desktop/gaussian-grouping01/output/microwave/point_cloud_object_removal/iteration_10000/point_cloud.ply \
  --filter-first \
  --filtered-ply /home/ubuntu/Desktop/gaussian-grouping01/output/microwave/point_cloud_object_removal/iteration_10000/point_cloud_filtered.ply \
  --output-json /home/ubuntu/Desktop/gaussian-grouping01/output/microwave/point_cloud_object_removal/iteration_10000/picked_hinge_split.json \
  --overlap-width 0.002 \
  --run-split \
  --split-output-dir /home/ubuntu/Desktop/gaussian-grouping01/output/microwave/point_cloud_object_removal/iteration_10000

python 01_pick_hinge_split_open3d.py    /home/ubuntu/Desktop/gaussian-grouping01/output/microwave/point_cloud_object_removal/iteration_10000/point_cloud.ply   --filter-first   --filtered-ply   /home/ubuntu/Desktop/gaussian-grouping01/output/microwave/point_cloud_object_removal/iteration_10000/point_cloud_filtered.ply   --output-json   /home/ubuntu/Desktop/gaussian-grouping01/output/microwave/point_cloud_object_removal/iteration_10000/picked_hinge_split.json   --overlap-width 0.002   --run-split   --split-output-dir   /home/ubuntu/Desktop/gaussian-grouping01/output/microwave/point_cloud_object_removal/iteration_10000

python 01_pick_hinge_split_open3d.py \
  /home/ubuntu/Desktop/gaussian-grouping01/output/microwave/point_cloud_object_removal/iteration_10000/point_cloud.ply \
  --filtered-ply /home/ubuntu/Desktop/gaussian-grouping01/output/microwave/point_cloud_object_removal/iteration_10000/point_cloud_filtered.ply \
  --filter-first \
  --output-json /home/ubuntu/Desktop/gaussian-grouping01/output/microwave/point_cloud_object_removal/iteration_10000/picked_hinge_split.json \
  --overlap-width 0.002 \
  --run-split \
  --split-output-dir /home/ubuntu/Desktop/gaussian-grouping01/output/microwave/point_cloud_object_removal/iteration_10000


python 01_pick_hinge_split_open3d.py \
       /home/ubuntu/Desktop/gaussian-grouping01/output/medicine_cabinet/point_cloud_object_removal/iteration_20000/filter_sweep/08_cluster_stronger.ply  \
  --filtered-ply     /home/ubuntu/Desktop/gaussian-grouping01/output/medicine_cabinet/point_cloud_object_removal/iteration_20000/filter_sweep/08_cluster_stronger.ply  \
  --output-json    /home/ubuntu/Desktop/gaussian-grouping01/output/microwave/temp/filter_sweep/picked_hinge_split.json \
  --overlap-width 0.002 \
    --filter-first \
  --run-split \
  --split-output-dir    /home/ubuntu/Desktop/gaussian-grouping01/output/microwave/temp




'''
