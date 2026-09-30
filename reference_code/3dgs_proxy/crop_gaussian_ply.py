#!/usr/bin/env python3
"""
Crop a Gaussian-splat PLY while preserving all Gaussian attributes.

The output is still a real 3DGS PLY, not a mesh. Import it with a UE5
Gaussian Splat plugin for the closest visual match to the original splat.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gaussian_ply_to_splat_mesh import apply_bounds, largest_cluster_bounds, parse_vec3
from ply_to_mesh import read_vertex_ply, sigmoid


PLY_TYPE_BY_DTYPE = {
    "int8": "char",
    "uint8": "uchar",
    "int16": "short",
    "uint16": "ushort",
    "int32": "int",
    "uint32": "uint",
    "float32": "float",
    "float64": "double",
}


def ply_type(dtype) -> str:
    key = np.dtype(dtype).name
    if key not in PLY_TYPE_BY_DTYPE:
        raise ValueError(f"Unsupported dtype for PLY export: {dtype}")
    return PLY_TYPE_BY_DTYPE[key]


def write_vertex_ply(path: Path, vertices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment cropped from Gaussian-splat PLY by crop_gaussian_ply.py",
        f"element vertex {len(vertices)}",
    ]
    for name in vertices.dtype.names or []:
        lines.append(f"property {ply_type(vertices.dtype[name])} {name}")
    lines.append("end_header")
    header = ("\n".join(lines) + "\n").encode("ascii")

    with path.open("wb") as f:
        f.write(header)
        f.write(np.ascontiguousarray(vertices).tobytes())


def largest_cluster_indices(points: np.ndarray, opacity: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    import open3d as o3d

    cluster_keep = np.isfinite(points).all(axis=1) & (opacity >= float(args.cluster_opacity))
    cluster_keep = apply_bounds(points, cluster_keep, args.crop_min, args.crop_max)
    candidate_indices = np.where(cluster_keep)[0]
    if len(candidate_indices) == 0:
        raise RuntimeError("No high-opacity points available for clustering. Lower --cluster-opacity.")

    cluster_points = points[candidate_indices]
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
        raise RuntimeError("DBSCAN found no valid cluster. Increase --cluster-eps or lower --cluster-min-points.")

    largest_label = max(valid_labels, key=lambda label: int((labels == label).sum()))
    largest = candidate_indices[labels == largest_label]
    print(
        f"[cluster] high-opacity candidates={len(candidate_indices)} "
        f"clusters={len(valid_labels)} largest={len(largest)}"
    )
    return largest


def keep_near_largest_cluster(points: np.ndarray, opacity: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    from scipy.spatial import cKDTree

    largest = largest_cluster_indices(points, opacity, args)
    cluster_points = points[largest]
    radius = float(args.cluster_include_radius)
    tree = cKDTree(cluster_points.astype(np.float64))
    try:
        distances, _ = tree.query(points.astype(np.float64), k=1, workers=-1)
    except TypeError:
        distances, _ = tree.query(points.astype(np.float64), k=1)

    keep = np.isfinite(points).all(axis=1)
    keep &= opacity >= float(args.min_opacity)
    keep &= distances <= radius
    keep = apply_bounds(points, keep, args.crop_min, args.crop_max)

    print(f"[cluster-distance] include_radius={radius} kept={int(keep.sum())}")
    print(f"[cluster-distance] cluster_bbox_min={cluster_points.min(axis=0)}")
    print(f"[cluster-distance] cluster_bbox_max={cluster_points.max(axis=0)}")
    return keep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop Gaussian-splat PLY and preserve original fields.")
    parser.add_argument("input", nargs="?", default="assets/point_cloud.ply")
    parser.add_argument("--output", "-o", default="assets/ue5_gaussian_splat/PointCloudObject_GaussianSplat.ply")
    parser.add_argument("--min-opacity", type=float, default=0.005)
    parser.add_argument("--max-points", type=int, default=0, help="0 keeps all cropped points.")
    parser.add_argument("--crop-min", type=parse_vec3, default=None)
    parser.add_argument("--crop-max", type=parse_vec3, default=None)
    parser.add_argument(
        "--filter-mode",
        choices=["cluster-distance", "bbox"],
        default="cluster-distance",
        help="cluster-distance removes isolated splats by distance to the largest high-opacity cluster; bbox is the old crop-by-bounding-box mode.",
    )
    parser.add_argument("--auto-crop-largest-cluster", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cluster-opacity", type=float, default=0.7)
    parser.add_argument("--cluster-eps", type=float, default=0.5)
    parser.add_argument("--cluster-min-points", type=int, default=10)
    parser.add_argument("--cluster-margin", type=float, default=0.12)
    parser.add_argument(
        "--cluster-include-radius",
        type=float,
        default=0.12,
        help="For cluster-distance mode, keep points within this distance from the largest high-opacity cluster.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vertices, names = read_vertex_ply(Path(args.input))
    points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float32)
    opacity = sigmoid(vertices["opacity"].astype(np.float32)) if "opacity" in names else np.ones(len(vertices), dtype=np.float32)

    if args.filter_mode == "cluster-distance":
        keep = keep_near_largest_cluster(points, opacity, args)
    else:
        crop_min = args.crop_min
        crop_max = args.crop_max
        if args.auto_crop_largest_cluster:
            auto_min, auto_max = largest_cluster_bounds(points, opacity, args)
            crop_min = tuple(auto_min.tolist()) if crop_min is None else tuple(np.maximum(auto_min, np.array(crop_min)).tolist())
            crop_max = tuple(auto_max.tolist()) if crop_max is None else tuple(np.minimum(auto_max, np.array(crop_max)).tolist())
            print(f"[auto-crop] min={np.array(crop_min)} max={np.array(crop_max)}")

        keep = np.isfinite(points).all(axis=1)
        keep &= opacity >= float(args.min_opacity)
        keep = apply_bounds(points, keep, crop_min, crop_max)
    idx = np.where(keep)[0]
    if args.max_points and len(idx) > args.max_points:
        order = np.argsort(opacity[idx])[::-1]
        idx = idx[order[: args.max_points]]

    cropped = vertices[idx].copy()
    write_vertex_ply(Path(args.output), cropped)
    cropped_points = points[idx]
    print(f"[load] {Path(args.input).resolve()}")
    print(f"[points] original={len(vertices)} cropped={len(cropped)}")
    print(f"[bbox] min={cropped_points.min(axis=0)} max={cropped_points.max(axis=0)}")
    print(f"[save] {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
