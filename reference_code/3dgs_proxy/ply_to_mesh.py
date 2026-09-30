#!/usr/bin/env python3
"""
Convert a Gaussian-splat PLY point cloud into a colored triangle mesh.

The output is an approximation. Gaussian splats are view-dependent translucent
primitives, while this script creates a conventional mesh with vertex colors.
For Isaac Sim physics, use this mesh or simpler boxes as hidden collision, and
keep the original Gaussian USDZ as the visible asset when exact appearance
matters.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


SH_C0 = 0.28209479177387814


PLY_DTYPE_MAP = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def read_vertex_ply(path: Path) -> tuple[np.ndarray, list[str]]:
    raw = path.read_bytes()
    marker = b"end_header"
    idx = raw.find(marker)
    if idx < 0:
        raise ValueError(f"{path} is not a PLY file: missing end_header")

    header_end = idx + len(marker)
    if raw[header_end : header_end + 2] == b"\r\n":
        data_start = header_end + 2
    elif raw[header_end : header_end + 1] == b"\n":
        data_start = header_end + 1
    else:
        data_start = header_end

    header = raw[:data_start].decode("ascii", errors="replace")
    lines = [line.strip() for line in header.splitlines() if line.strip()]
    if not lines or lines[0] != "ply":
        raise ValueError(f"{path} is not a PLY file")

    fmt = None
    vertex_count = None
    vertex_props: list[tuple[str, str]] = []
    current_element = None

    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if parts[1] == "list":
                raise ValueError("List properties on vertex elements are not supported")
            prop_type, prop_name = parts[1], parts[2]
            if prop_type not in PLY_DTYPE_MAP:
                raise ValueError(f"Unsupported PLY property type: {prop_type}")
            vertex_props.append((prop_name, PLY_DTYPE_MAP[prop_type]))

    if fmt != "binary_little_endian":
        raise ValueError(f"Only binary_little_endian PLY is supported, got: {fmt}")
    if vertex_count is None:
        raise ValueError("PLY file has no vertex element")
    if not vertex_props:
        raise ValueError("PLY file has no vertex properties")

    dtype = np.dtype(vertex_props)
    need = dtype.itemsize * vertex_count
    chunk = raw[data_start : data_start + need]
    if len(chunk) != need:
        raise ValueError(f"PLY vertex data is truncated: expected {need} bytes, got {len(chunk)}")

    return np.frombuffer(chunk, dtype=dtype, count=vertex_count), [name for name, _ in vertex_props]


def require_fields(names: Iterable[str], required: Iterable[str]) -> None:
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"PLY is missing required fields: {missing}")


def extract_points_colors_normals(vertices: np.ndarray, names: list[str], min_opacity: float) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    require_fields(names, ["x", "y", "z"])
    points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float64)

    mask = np.isfinite(points).all(axis=1)
    if "opacity" in names and min_opacity > 0.0:
        mask &= sigmoid(vertices["opacity"].astype(np.float64)) >= float(min_opacity)

    points = points[mask]

    colors = np.full((len(points), 3), 0.75, dtype=np.float64)
    if all(name in names for name in ("red", "green", "blue")):
        raw_rgb = np.column_stack([vertices["red"], vertices["green"], vertices["blue"]]).astype(np.float64)[mask]
        colors = np.clip(raw_rgb / 255.0, 0.0, 1.0)
    elif all(name in names for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
        dc = np.column_stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]]).astype(np.float64)[mask]
        colors = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)

    normals = None
    if all(name in names for name in ("nx", "ny", "nz")):
        n = np.column_stack([vertices["nx"], vertices["ny"], vertices["nz"]]).astype(np.float64)[mask]
        lengths = np.linalg.norm(n, axis=1)
        valid = np.isfinite(n).all(axis=1) & (lengths > 1e-8)
        if valid.mean() > 0.8:
            n = n / np.maximum(lengths[:, None], 1e-8)
            normals = n

    return points, colors, normals


def estimate_normal_radius(pcd, voxel_size: float) -> float:
    dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    dists = dists[np.isfinite(dists) & (dists > 0.0)]
    if len(dists) == 0:
        return max(voxel_size * 8.0, 0.01)
    return max(float(np.median(dists)) * 8.0, voxel_size * 8.0, 1e-4)


def make_mesh(points: np.ndarray, colors: np.ndarray, normals: np.ndarray | None, args: argparse.Namespace):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    if args.voxel_size > 0.0:
        pcd = pcd.voxel_down_sample(args.voxel_size)

    if args.remove_outliers:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.0)

    if not pcd.has_normals():
        radius = estimate_normal_radius(pcd, args.voxel_size)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50))

    if args.orient_normals:
        try:
            pcd.orient_normals_consistent_tangent_plane(args.orientation_neighbors)
        except RuntimeError as exc:
            print(f"[warn] normal orientation failed, continuing with estimated normals: {exc}")

    if args.method == "poisson":
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=args.depth)
        densities = np.asarray(densities)
        if 0.0 < args.density_trim_quantile < 1.0:
            keep = densities > np.quantile(densities, args.density_trim_quantile)
            mesh = mesh.select_by_index(np.where(keep)[0])
    else:
        dists = np.asarray(pcd.compute_nearest_neighbor_distance())
        dists = dists[np.isfinite(dists) & (dists > 0.0)]
        radius = float(np.median(dists)) * args.ball_radius_scale
        radii = o3d.utility.DoubleVector([radius, radius * 2.0, radius * 4.0])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    if args.keep_largest:
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        if len(cluster_n_triangles) > 0:
            largest = int(cluster_n_triangles.argmax())
            mesh.remove_triangles_by_mask(triangle_clusters != largest)
            mesh.remove_unreferenced_vertices()

    if args.smooth_iterations > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=args.smooth_iterations)

    mesh.compute_vertex_normals()
    return mesh


def export_mesh(mesh, output: Path) -> None:
    import open3d as o3d

    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()

    if suffix in {".glb", ".gltf"}:
        import trimesh

        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        colors = np.asarray(mesh.vertex_colors)
        vertex_colors = None
        if len(colors) == len(vertices):
            alpha = np.full((len(colors), 1), 255, dtype=np.uint8)
            vertex_colors = np.concatenate([np.clip(colors * 255.0, 0, 255).astype(np.uint8), alpha], axis=1)
        tm = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=False)
        tm.export(output)
    else:
        ok = o3d.io.write_triangle_mesh(str(output), mesh, write_vertex_colors=True, write_triangle_uvs=False)
        if not ok:
            raise RuntimeError(f"Open3D failed to write {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Gaussian-splat PLY to a colored triangle mesh.")
    parser.add_argument("input", nargs="?", default="assets/point_cloud.ply", help="Input Gaussian-splat PLY.")
    parser.add_argument("--output", "-o", default="assets/point_cloud_mesh.glb", help="Output mesh path: .glb, .gltf, .ply, .obj, .stl.")
    parser.add_argument("--method", choices=["poisson", "ball_pivoting"], default="poisson")
    parser.add_argument("--voxel-size", type=float, default=0.0, help="Optional point voxel downsample size in scene units.")
    parser.add_argument("--depth", type=int, default=10, help="Poisson reconstruction depth.")
    parser.add_argument("--density-trim-quantile", type=float, default=0.03, help="Trim low-density Poisson vertices.")
    parser.add_argument("--min-opacity", type=float, default=0.02, help="Drop Gaussian points below this sigmoid opacity.")
    parser.add_argument("--remove-outliers", action="store_true", help="Run statistical outlier removal before meshing.")
    parser.add_argument("--orient-normals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--orientation-neighbors", type=int, default=30)
    parser.add_argument("--ball-radius-scale", type=float, default=3.0)
    parser.add_argument("--keep-largest", action="store_true", help="Keep only the largest connected triangle component.")
    parser.add_argument("--smooth-iterations", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    vertices, names = read_vertex_ply(input_path)
    points, colors, normals = extract_points_colors_normals(vertices, names, args.min_opacity)
    print(f"[load] {input_path}: {len(vertices)} vertices, {len(points)} kept")

    mesh = make_mesh(points, colors, normals, args)
    export_mesh(mesh, output_path)
    print(f"[mesh] vertices={len(mesh.vertices)} triangles={len(mesh.triangles)}")
    print(f"[save] {output_path.resolve()}")


if __name__ == "__main__":
    main()
