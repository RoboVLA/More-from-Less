#!/usr/bin/env python3
"""
Split a Gaussian-splat PLY into two Gaussian PLY files by a plane.

All vertex properties are preserved, including opacity, scale, rotation and
spherical-harmonic color fields. The output PLY files remain valid 3DGS PLYs
and can be fed into 3DGRUT's ply_to_usd pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from crop_gaussian_ply import write_vertex_ply
from ply_to_mesh import read_vertex_ply


def parse_vec3(text: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected x,y,z")
    return (parts[0], parts[1], parts[2])


def normalized(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        raise ValueError("Vector length is zero")
    return vec / norm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a Gaussian PLY into two files by a plane.")
    parser.add_argument("input", help="Input Gaussian-splat PLY.")
    parser.add_argument("--output-dir", default=None, help="Defaults to input PLY directory.")
    parser.add_argument("--negative-name", default="part_a", help="Output stem for signed_distance < 0 side.")
    parser.add_argument("--positive-name", default="part_b", help="Output stem for signed_distance >= 0 side.")

    plane = parser.add_mutually_exclusive_group(required=True)
    plane.add_argument("--axis", choices=["x", "y", "z"], help="Axis-aligned split plane normal.")
    plane.add_argument("--plane-normal", type=parse_vec3, help="Plane normal as x,y,z.")
    parser.add_argument("--value", type=float, default=0.0, help="Axis-aligned plane value, e.g. x=value.")
    parser.add_argument("--plane-point", type=parse_vec3, default=None, help="Point on non-axis-aligned split plane.")

    parser.add_argument(
        "--overlap-width",
        type=float,
        default=0.0,
        help="Duplicate points within this signed-distance band into both files to hide a visual seam.",
    )
    parser.add_argument("--hinge-point", type=parse_vec3, default=None, help="Optional hinge line point stored in metadata.")
    parser.add_argument("--hinge-axis", type=parse_vec3, default=None, help="Optional hinge line direction stored in metadata.")
    parser.add_argument("--metadata", default=None, help="Defaults to output-dir/split_metadata.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    vertices, _ = read_vertex_ply(input_path)
    points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float64)

    if args.axis:
        axis_index = {"x": 0, "y": 1, "z": 2}[args.axis]
        normal = np.zeros(3, dtype=np.float64)
        normal[axis_index] = 1.0
        plane_point = np.zeros(3, dtype=np.float64)
        plane_point[axis_index] = float(args.value)
    else:
        normal = normalized(np.array(args.plane_normal, dtype=np.float64))
        plane_point = np.array(args.plane_point if args.plane_point is not None else (0.0, 0.0, 0.0), dtype=np.float64)

    signed = (points - plane_point[None, :]) @ normal
    overlap = max(0.0, float(args.overlap_width))
    negative_mask = signed <= overlap
    positive_mask = signed >= -overlap

    negative = vertices[np.where(negative_mask)[0]].copy()
    positive = vertices[np.where(positive_mask)[0]].copy()

    negative_path = output_dir / f"{args.negative_name}.ply"
    positive_path = output_dir / f"{args.positive_name}.ply"
    write_vertex_ply(negative_path, negative)
    write_vertex_ply(positive_path, positive)

    metadata_path = Path(args.metadata) if args.metadata else output_dir / "split_metadata.json"
    metadata = {
        "input": str(input_path),
        "negative_ply": str(negative_path),
        "positive_ply": str(positive_path),
        "plane_point": plane_point.tolist(),
        "plane_normal": normal.tolist(),
        "overlap_width": overlap,
        "negative_count": int(len(negative)),
        "positive_count": int(len(positive)),
        "hinge_point": list(args.hinge_point) if args.hinge_point is not None else plane_point.tolist(),
        "hinge_axis": list(args.hinge_axis) if args.hinge_axis is not None else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[input] {input_path.resolve()}")
    print(f"[plane] point={plane_point} normal={normal}")
    print(f"[split] negative={len(negative)} positive={len(positive)} overlap_width={overlap}")
    print(f"[save] {negative_path.resolve()}")
    print(f"[save] {positive_path.resolve()}")
    print(f"[metadata] {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
