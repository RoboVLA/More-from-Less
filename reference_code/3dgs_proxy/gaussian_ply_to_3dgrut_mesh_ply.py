#!/usr/bin/env python3
"""
Generate a triangle-mesh PLY suitable for 3DGRUT's add_mesh_to_usdz.py.

3DGRUT expects:
- element vertex with x, y, z float properties
- element face with list uchar int vertex_indices
- every face must be a triangle

This script keeps the original Gaussian PLY for rendering and creates a
separate solid-ish mesh PLY for USDZ proxy/collision geometry.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gaussian_ply_to_solid_textured_mesh import (
    load_cropped_points,
    points_to_volume,
    volume_to_mesh,
)


def write_triangle_mesh_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, ascii: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must be Nx3, got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be Mx3 triangles, got {faces.shape}")
    if len(faces) == 0:
        raise ValueError("mesh has no faces")

    fmt = "ascii 1.0" if ascii else "binary_little_endian 1.0"
    header = "\n".join(
        [
            "ply",
            f"format {fmt}",
            "comment triangle mesh generated for 3DGRUT add_mesh_to_usdz.py",
            f"element vertex {len(vertices)}",
            "property float x",
            "property float y",
            "property float z",
            f"element face {len(faces)}",
            "property list uchar int vertex_indices",
            "end_header",
            "",
        ]
    )

    if ascii:
        with path.open("w", encoding="ascii", newline="\n") as f:
            f.write(header)
            for x, y, z in vertices:
                f.write(f"{float(x):.9g} {float(y):.9g} {float(z):.9g}\n")
            for a, b, c in faces:
                f.write(f"3 {int(a)} {int(b)} {int(c)}\n")
        return

    vertex_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    vertex_data = np.empty(len(vertices), dtype=vertex_dtype)
    vertex_data["x"] = vertices[:, 0]
    vertex_data["y"] = vertices[:, 1]
    vertex_data["z"] = vertices[:, 2]

    # PLY list uchar int is encoded as one uint8 count followed by int32 indices.
    face_dtype = np.dtype([("n", "u1"), ("v0", "<i4"), ("v1", "<i4"), ("v2", "<i4")])
    face_data = np.empty(len(faces), dtype=face_dtype)
    face_data["n"] = 3
    face_data["v0"] = faces[:, 0]
    face_data["v1"] = faces[:, 1]
    face_data["v2"] = faces[:, 2]

    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex_data.tobytes())
        f.write(face_data.tobytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a 3DGRUT-compatible mesh PLY from Gaussian PLY.")
    parser.add_argument("input", nargs="?", default="assets/point_cloud.ply")
    parser.add_argument("--output", "-o", default="assets/3dgrut_mesh/PointCloudObject_collision_mesh.ply")
    parser.add_argument("--ascii", action="store_true", help="Write ASCII PLY instead of binary.")

    parser.add_argument("--min-opacity", type=float, default=0.16)
    parser.add_argument("--max-points", type=int, default=30000)
    parser.add_argument("--crop-min", type=str, default=None)
    parser.add_argument("--crop-max", type=str, default=None)
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
    parser.add_argument("--simplify-target-faces", type=int, default=25000)
    return parser.parse_args()


def _parse_vec3_arg(text: str | None):
    if text is None:
        return None
    parts = [float(part.strip()) for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got: {text}")
    return tuple(parts)


def main() -> None:
    args = parse_args()
    args.crop_min = _parse_vec3_arg(args.crop_min)
    args.crop_max = _parse_vec3_arg(args.crop_max)

    points, _, _, auto_bounds = load_cropped_points(Path(args.input), args)
    mask, origin, voxel = points_to_volume(points, args)
    mesh = volume_to_mesh(mask, origin, voxel, args)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    write_triangle_mesh_ply(Path(args.output), vertices, faces, ascii=bool(args.ascii))

    if auto_bounds is not None:
        print(f"[auto-crop] min={auto_bounds[0]} max={auto_bounds[1]}")
    print(f"[points] {len(points)}")
    print(f"[volume] shape={mask.shape} occupied={int(mask.sum())} voxel={voxel:.6f}")
    print(f"[mesh] vertices={len(vertices)} faces={len(faces)}")
    print(f"[save] {Path(args.output).resolve()}")
    print("[3dgrut] use with: --mesh_ply", Path(args.output).resolve())


if __name__ == "__main__":
    main()
