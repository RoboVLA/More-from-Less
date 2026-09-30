#!/usr/bin/env python3
"""
Create a solid continuous textured mesh from a Gaussian-splat PLY.

This is a more robust fallback than Poisson for sparse 3DGS points:
- crop the object,
- convert points into a volumetric occupancy field,
- dilate/smooth/fill the volume,
- extract a continuous surface using marching cubes,
- transfer color and bake it to a texture for UE5.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy import ndimage as ndi
from skimage import measure

from gaussian_ply_to_splat_mesh import parse_vec3
from gaussian_ply_to_textured_mesh import (
    export_textured_fbx_with_blender,
    load_cropped_points,
    transfer_colors,
)


def make_source_pcd(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count <= 1:
        return mask
    sizes = np.bincount(labels.reshape(-1))
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    return labels == largest


def points_to_volume(points: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float]:
    voxel = float(args.voxel_size)
    pad = int(args.padding_voxels)

    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    origin = pmin - pad * voxel
    dims = np.ceil((pmax - pmin) / voxel).astype(np.int32) + pad * 2 + 1
    dims = np.maximum(dims, 8)

    if int(np.prod(dims)) > int(args.max_voxels):
        scale = (np.prod(dims) / float(args.max_voxels)) ** (1.0 / 3.0)
        voxel *= scale
        origin = pmin - pad * voxel
        dims = np.ceil((pmax - pmin) / voxel).astype(np.int32) + pad * 2 + 1

    grid = np.zeros(tuple(int(x) for x in dims), dtype=np.float32)
    idx = np.floor((points - origin) / voxel).astype(np.int32)
    valid = ((idx >= 0) & (idx < dims)).all(axis=1)
    idx = idx[valid]
    np.add.at(grid, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)

    density = ndi.gaussian_filter(grid, sigma=float(args.sigma_voxels))
    threshold = float(args.density_threshold)
    if threshold <= 0.0:
        threshold = float(density.max()) * float(args.relative_threshold)
    mask = density >= threshold

    if args.dilate_iterations > 0:
        mask = ndi.binary_dilation(mask, iterations=int(args.dilate_iterations))
    if args.close_iterations > 0:
        mask = ndi.binary_closing(mask, iterations=int(args.close_iterations))
    if args.fill_holes:
        mask = ndi.binary_fill_holes(mask)
    if args.keep_largest:
        mask = keep_largest_component(mask)

    return mask.astype(bool), origin.astype(np.float32), voxel


def volume_to_mesh(mask: np.ndarray, origin: np.ndarray, voxel: float, args: argparse.Namespace) -> o3d.geometry.TriangleMesh:
    volume = mask.astype(np.float32)
    vertices, faces, normals, _ = measure.marching_cubes(
        volume,
        level=0.5,
        spacing=(voxel, voxel, voxel),
    )
    vertices = vertices.astype(np.float32) + origin[None, :]

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices.astype(np.float64)),
        triangles=o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    if args.smooth_iterations > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=int(args.smooth_iterations))

    if args.simplify_target_faces and len(mesh.triangles) > args.simplify_target_faces:
        mesh = mesh.simplify_quadric_decimation(int(args.simplify_target_faces))

    mesh.compute_vertex_normals()
    return mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a solid textured UE5 mesh from Gaussian PLY.")
    parser.add_argument("input", nargs="?", default="assets/point_cloud.ply")
    parser.add_argument("--fbx", default="assets/ue5_point_cloud_object/PointCloudObject_SolidTexturedMesh_UE5.fbx")
    parser.add_argument("--texture", default="assets/ue5_point_cloud_object/T_PointCloudObject_SolidTexturedMesh_BaseColor.png")
    parser.add_argument("--name", default="PointCloudObject_SolidTexturedMesh")
    parser.add_argument("--texture-size", type=int, default=2048)

    parser.add_argument("--min-opacity", type=float, default=0.16)
    parser.add_argument("--max-points", type=int, default=30000)
    parser.add_argument("--crop-min", type=parse_vec3, default=None)
    parser.add_argument("--crop-max", type=parse_vec3, default=None)
    parser.add_argument("--auto-crop-largest-cluster", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cluster-opacity", type=float, default=0.7)
    parser.add_argument("--cluster-eps", type=float, default=0.5)
    parser.add_argument("--cluster-min-points", type=int, default=10)
    parser.add_argument("--cluster-margin", type=float, default=0.10)

    parser.add_argument("--voxel-size", type=float, default=0.012)
    parser.add_argument("--max-voxels", type=int, default=4_000_000)
    parser.add_argument("--padding-voxels", type=int, default=8)
    parser.add_argument("--sigma-voxels", type=float, default=2.2)
    parser.add_argument("--density-threshold", type=float, default=0.0)
    parser.add_argument("--relative-threshold", type=float, default=0.030)
    parser.add_argument("--dilate-iterations", type=int, default=2)
    parser.add_argument("--close-iterations", type=int, default=3)
    parser.add_argument("--fill-holes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-largest", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--smooth-iterations", type=int, default=8)
    parser.add_argument("--simplify-target-faces", type=int, default=45000)
    parser.add_argument("--color-neighbors", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points, colors, _, auto_bounds = load_cropped_points(Path(args.input), args)
    source_pcd = make_source_pcd(points, colors)

    mask, origin, voxel = points_to_volume(points, args)
    mesh = volume_to_mesh(mask, origin, voxel, args)
    mesh_colors = transfer_colors(mesh, source_pcd, args)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int64)

    export_textured_fbx_with_blender(
        vertices,
        faces,
        mesh_colors,
        Path(args.fbx),
        Path(args.texture),
        args.name,
        int(args.texture_size),
    )

    if auto_bounds is not None:
        print(f"[auto-crop] min={auto_bounds[0]} max={auto_bounds[1]}")
    print(f"[points] {len(points)}")
    print(f"[volume] shape={mask.shape} occupied={int(mask.sum())} voxel={voxel:.6f}")
    print(f"[mesh] vertices={len(vertices)} faces={len(faces)}")
    print(f"[save] FBX {Path(args.fbx).resolve()}")
    print(f"[save] texture {Path(args.texture).resolve()}")


if __name__ == "__main__":
    main()
