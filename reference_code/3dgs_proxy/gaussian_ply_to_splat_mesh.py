#!/usr/bin/env python3
"""
Convert a Gaussian-splat PLY into a colored "splat mesh".

Unlike Poisson reconstruction, this does not try to create one continuous
surface. Each Gaussian becomes a tiny low-poly ellipsoid, which is much closer
to a splat-cloud appearance when imported into UE/Isaac as a conventional mesh.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

from ply_to_mesh import SH_C0, read_vertex_ply, sigmoid


OCT_VERTICES = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float32,
)

OCT_FACES = np.array(
    [
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ],
    dtype=np.int64,
)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float64)
    quat /= np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-12)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    mats = np.empty((len(quat), 3, 3), dtype=np.float32)
    mats[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    mats[:, 0, 1] = 2.0 * (x * y - z * w)
    mats[:, 0, 2] = 2.0 * (x * z + y * w)
    mats[:, 1, 0] = 2.0 * (x * y + z * w)
    mats[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    mats[:, 1, 2] = 2.0 * (y * z - x * w)
    mats[:, 2, 0] = 2.0 * (x * z - y * w)
    mats[:, 2, 1] = 2.0 * (y * z + x * w)
    mats[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return mats


def apply_bounds(points: np.ndarray, keep: np.ndarray, crop_min, crop_max) -> np.ndarray:
    if crop_min is not None:
        keep &= (points >= np.array(crop_min, dtype=np.float32)).all(axis=1)
    if crop_max is not None:
        keep &= (points <= np.array(crop_max, dtype=np.float32)).all(axis=1)
    return keep


def largest_cluster_bounds(points: np.ndarray, opacity: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    cluster_keep = np.isfinite(points).all(axis=1) & (opacity >= float(args.cluster_opacity))
    cluster_keep = apply_bounds(points, cluster_keep, args.crop_min, args.crop_max)
    cluster_points = points[cluster_keep]
    if len(cluster_points) == 0:
        raise RuntimeError("No points available for auto cluster crop. Lower --cluster-opacity.")

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cluster_points.astype(np.float64)))
    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=float(args.cluster_eps),
            min_points=int(args.cluster_min_points),
            print_progress=False,
        )
    )
    valid_labels = [label for label in set(labels.tolist()) if label >= 0]
    if not valid_labels:
        raise RuntimeError("DBSCAN found no clusters. Increase --cluster-eps or lower --cluster-min-points.")

    largest_label = max(valid_labels, key=lambda label: int((labels == label).sum()))
    cluster = cluster_points[labels == largest_label]
    margin = float(args.cluster_margin)
    return cluster.min(axis=0) - margin, cluster.max(axis=0) + margin


def load_gaussians(path: Path, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
    vertices, names = read_vertex_ply(path)
    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}
    missing = sorted(required.difference(names))
    if missing:
        raise ValueError(f"Input PLY is missing Gaussian fields: {missing}")

    points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float32)
    opacity = sigmoid(vertices["opacity"].astype(np.float32))
    scales = np.exp(np.column_stack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]]).astype(np.float32))
    rotations = np.column_stack([vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]]).astype(np.float32)

    colors = np.full((len(vertices), 3), 0.75, dtype=np.float32)
    if all(name in names for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
        dc = np.column_stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]]).astype(np.float32)
        colors = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)

    keep = np.isfinite(points).all(axis=1) & np.isfinite(scales).all(axis=1) & np.isfinite(rotations).all(axis=1)
    keep &= opacity >= float(args.min_opacity)

    auto_bounds = None
    crop_min = args.crop_min
    crop_max = args.crop_max
    if args.auto_crop_largest_cluster:
        auto_min, auto_max = largest_cluster_bounds(points, opacity, args)
        crop_min = tuple(auto_min.tolist()) if crop_min is None else tuple(np.maximum(auto_min, np.array(crop_min)).tolist())
        crop_max = tuple(auto_max.tolist()) if crop_max is None else tuple(np.minimum(auto_max, np.array(crop_max)).tolist())
        auto_bounds = (np.array(crop_min, dtype=np.float32), np.array(crop_max, dtype=np.float32))

    keep = apply_bounds(points, keep, crop_min, crop_max)

    idx = np.where(keep)[0]
    if args.max_splats and len(idx) > args.max_splats:
        order = np.argsort(opacity[idx])[::-1]
        idx = idx[order[: args.max_splats]]

    return points[idx], colors[idx], opacity[idx], scales[idx], rotations[idx], auto_bounds


def build_splat_mesh(
    points: np.ndarray,
    colors: np.ndarray,
    opacity: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    args: argparse.Namespace,
) -> trimesh.Trimesh:
    count = len(points)
    if count == 0:
        raise RuntimeError("No splats survived filtering. Lower --min-opacity or change crop bounds.")

    radii = scales * float(args.radius_scale)
    radii = np.clip(radii, float(args.min_radius), float(args.max_radius)).astype(np.float32)
    mats = quat_wxyz_to_matrix(rotations)

    local = OCT_VERTICES[None, :, :] * radii[:, None, :]
    world = np.einsum("nij,nvj->nvi", mats, local) + points[:, None, :]
    mesh_vertices = world.reshape((-1, 3))

    offsets = (np.arange(count, dtype=np.int64) * len(OCT_VERTICES))[:, None, None]
    mesh_faces = (OCT_FACES[None, :, :] + offsets).reshape((-1, 3))

    alpha = np.clip(opacity * 255.0, 24.0, 255.0).astype(np.uint8)
    rgba = np.concatenate(
        [
            np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8),
            alpha[:, None],
        ],
        axis=1,
    )
    vertex_colors = np.repeat(rgba, len(OCT_VERTICES), axis=0)

    mesh = trimesh.Trimesh(
        vertices=mesh_vertices,
        faces=mesh_faces,
        vertex_colors=vertex_colors,
        process=False,
    )
    return mesh


def parse_vec3(text: str | None) -> tuple[float, float, float] | None:
    if text is None:
        return None
    parts = [float(part.strip()) for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated values, e.g. 0,0,0")
    return (parts[0], parts[1], parts[2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Gaussian PLY to an ellipsoid splat mesh.")
    parser.add_argument("input", nargs="?", default="assets/point_cloud.ply")
    parser.add_argument("--output", "-o", default="assets/point_cloud_splat_mesh.glb")
    parser.add_argument("--min-opacity", type=float, default=0.05)
    parser.add_argument("--max-splats", type=int, default=35000)
    parser.add_argument("--radius-scale", type=float, default=0.75)
    parser.add_argument("--min-radius", type=float, default=0.0015)
    parser.add_argument("--max-radius", type=float, default=0.035)
    parser.add_argument("--crop-min", type=parse_vec3, default=None, help="Optional xyz crop min, e.g. -1,-1,-1")
    parser.add_argument("--crop-max", type=parse_vec3, default=None, help="Optional xyz crop max, e.g. 1,1,1")
    parser.add_argument("--auto-crop-largest-cluster", action="store_true", help="Find the largest high-opacity DBSCAN cluster and crop around it.")
    parser.add_argument("--cluster-opacity", type=float, default=0.7)
    parser.add_argument("--cluster-eps", type=float, default=0.5)
    parser.add_argument("--cluster-min-points", type=int, default=10)
    parser.add_argument("--cluster-margin", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    points, colors, opacity, scales, rotations, auto_bounds = load_gaussians(input_path, args)
    mesh = build_splat_mesh(points, colors, opacity, scales, rotations, args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)

    print(f"[load] {input_path.resolve()}")
    if auto_bounds is not None:
        print(f"[auto-crop] min={auto_bounds[0]} max={auto_bounds[1]}")
    print(f"[splats] {len(points)}")
    print(f"[mesh] vertices={len(mesh.vertices)} faces={len(mesh.faces)}")
    print(f"[bbox] min={mesh.bounds[0]} max={mesh.bounds[1]}")
    print(f"[save] {output_path.resolve()}")


if __name__ == "__main__":
    main()
