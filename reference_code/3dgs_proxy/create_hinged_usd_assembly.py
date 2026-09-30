#!/usr/bin/env python3
"""
Create a top-level USD assembly that references two USDZ parts and connects
them with a PhysicsRevoluteJoint.

This local assembly step does not require 3DGRUT. It only needs the USD Python
bindings that come with Isaac Sim / IsaacLab.

The two USDZ files are expected to stay in the same original coordinate frame
as the source Gaussian PLY. This means they can both be referenced at identity
transform and will line up along the split plane.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

try:
    from pxr import PhysxSchema
except ImportError:
    PhysxSchema = None


def parse_vec3(text: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected x,y,z")
    return (parts[0], parts[1], parts[2])


def load_split_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("hinge_point", "hinge_axis"):
        if key not in data:
            raise ValueError(f"{path} is missing required key: {key}")
    return data


def normalized(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        raise ValueError("Vector length is zero")
    return vec / norm


def quat_from_to(src: np.ndarray, dst: np.ndarray) -> Gf.Quatf:
    src = normalized(src)
    dst = normalized(dst)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if dot > 1.0 - 1e-8:
        return Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
    if dot < -1.0 + 1e-8:
        ortho = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(src, ortho))) > 0.9:
            ortho = np.array([0.0, 1.0, 0.0])
        axis = normalized(np.cross(src, ortho))
        return Gf.Quatf(0.0, Gf.Vec3f(float(axis[0]), float(axis[1]), float(axis[2])))
    axis = np.cross(src, dst)
    quat = np.array([1.0 + dot, axis[0], axis[1], axis[2]], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return Gf.Quatf(float(quat[0]), Gf.Vec3f(float(quat[1]), float(quat[2]), float(quat[3])))


def ensure_scope(stage: Usd.Stage, path: str):
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        return prim
    return UsdGeom.Xform.Define(stage, path).GetPrim()


def make_reference_path(asset_path: Path, output_dir: Path, mode: str) -> str:
    resolved = asset_path.resolve()
    if mode == "absolute":
        return str(resolved)
    return os.path.relpath(resolved, start=output_dir.resolve()).replace(os.sep, "/")


def add_reference_body(stage: Usd.Stage, body_path: str, reference_path: str, kinematic: bool, disable_gravity: bool, density: float):
    body = UsdGeom.Xform.Define(stage, body_path).GetPrim()
    body.GetReferences().AddReference(reference_path)

    UsdPhysics.RigidBodyAPI.Apply(body)
    mass = UsdPhysics.MassAPI.Apply(body)
    mass.CreateDensityAttr(float(density))
    if hasattr(UsdPhysics.RigidBodyAPI(body), "CreateKinematicEnabledAttr"):
        try:
            UsdPhysics.RigidBodyAPI(body).CreateKinematicEnabledAttr(bool(kinematic))
        except Exception:
            pass

    if PhysxSchema is not None:
        physx = PhysxSchema.PhysxRigidBodyAPI.Apply(body)
        physx.CreateDisableGravityAttr(bool(disable_gravity))
        physx.CreateKinematicEnabledAttr(bool(kinematic))
        physx.CreateSolverPositionIterationCountAttr(16)
        physx.CreateSolverVelocityIterationCountAttr(4)
    return body


def set_collision_approximation_under(stage: Usd.Stage, root_path: str, approximation: str) -> int:
    if PhysxSchema is None or not approximation:
        return 0

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return 0

    count = 0
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() != "Mesh":
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        physx_col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        physx_col.CreateCollisionApproximationAttr().Set(approximation)
        count += 1
    return count


def axis_token_from_vector(axis: np.ndarray) -> tuple[str, Gf.Quatf]:
    axis = normalized(axis)
    candidates = {
        "X": np.array([1.0, 0.0, 0.0]),
        "Y": np.array([0.0, 1.0, 0.0]),
        "Z": np.array([0.0, 0.0, 1.0]),
    }
    token = max(candidates, key=lambda key: abs(float(np.dot(axis, candidates[key]))))
    base_axis = candidates[token]
    if float(np.dot(axis, base_axis)) < 0.0:
        base_axis = -base_axis
    return token, quat_from_to(base_axis, axis)


def create_revolute_joint(
    stage: Usd.Stage,
    joint_path: str,
    body0_path: str,
    body1_path: str,
    hinge_point: np.ndarray,
    hinge_axis: np.ndarray,
    lower_deg: float,
    upper_deg: float,
):
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])

    axis_token, local_rot = axis_token_from_vector(hinge_axis)
    joint.CreateAxisAttr(axis_token)
    joint.CreateLocalPos0Attr(Gf.Vec3f(float(hinge_point[0]), float(hinge_point[1]), float(hinge_point[2])))
    joint.CreateLocalPos1Attr(Gf.Vec3f(float(hinge_point[0]), float(hinge_point[1]), float(hinge_point[2])))
    joint.CreateLocalRot0Attr(local_rot)
    joint.CreateLocalRot1Attr(local_rot)
    joint.CreateLowerLimitAttr(float(lower_deg))
    joint.CreateUpperLimitAttr(float(upper_deg))
    return joint


def build_assembly_stage(
    output_layer: Path,
    part_a_reference: str,
    part_b_reference: str,
    args: argparse.Namespace,
    hinge_point_arg,
    hinge_axis_arg,
) -> None:
    stage = Usd.Stage.CreateNew(str(output_layer.resolve()))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, float(args.meters_per_unit))

    root_prim = ensure_scope(stage, args.root_path)
    root_path = root_prim.GetPath()
    if not root_path.IsRootPrimPath():
        raise ValueError(
            f"--root-path must be a top-level prim path for drag-and-drop assets, got {root_path}. "
            "Use /HingedObject instead of /World/HingedObject."
        )
    stage.SetDefaultPrim(root_prim)
    if args.use_articulation_root:
        if args.fixed_body != "none":
            raise ValueError("--use-articulation-root cannot be used with --fixed-body A/B because PhysX does not support kinematic bodies in articulations.")
        UsdPhysics.ArticulationRootAPI.Apply(root_prim)

    body_a_path = f"{args.root_path}/PartA"
    body_b_path = f"{args.root_path}/PartB"
    joint_path = f"{args.root_path}/HingeJoint"

    add_reference_body(
        stage,
        body_a_path,
        part_a_reference,
        kinematic=(args.fixed_body == "A"),
        disable_gravity=(args.fixed_body == "A"),
        density=float(args.density),
    )
    add_reference_body(
        stage,
        body_b_path,
        part_b_reference,
        kinematic=(args.fixed_body == "B"),
        disable_gravity=(args.fixed_body == "B"),
        density=float(args.density),
    )

    if args.dynamic_collision_approximation:
        dynamic_paths = []
        if args.fixed_body != "A":
            dynamic_paths.append(body_a_path)
        if args.fixed_body != "B":
            dynamic_paths.append(body_b_path)
        total = 0
        for path in dynamic_paths:
            total += set_collision_approximation_under(stage, path, args.dynamic_collision_approximation)
        print(f"[collision_approximation] {args.dynamic_collision_approximation} applied to {total} dynamic collision Mesh prim(s)")

    hinge_point = np.array(hinge_point_arg, dtype=np.float64)
    hinge_axis = normalized(np.array(hinge_axis_arg, dtype=np.float64))
    create_revolute_joint(
        stage,
        joint_path,
        body_a_path,
        body_b_path,
        hinge_point,
        hinge_axis,
        float(args.lower_limit_deg),
        float(args.upper_limit_deg),
    )

    stage.GetRootLayer().Save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create hinged USD assembly from two collision-enabled USDZ files.")
    parser.add_argument("--part-a-usdz", required=True)
    parser.add_argument("--part-b-usdz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root-path", default="/HingedObject")
    parser.add_argument("--split-json", help="JSON produced by pick_hinge_split_open3d.py. Provides hinge_point and hinge_axis.")
    parser.add_argument("--hinge-point", type=parse_vec3, default=None, help="World/local point on the hinge line, x,y,z.")
    parser.add_argument("--hinge-axis", type=parse_vec3, default=None, help="Hinge axis direction, x,y,z.")
    parser.add_argument("--lower-limit-deg", type=float, default=0.0)
    parser.add_argument("--upper-limit-deg", type=float, default=90.0)
    parser.add_argument("--fixed-body", choices=["A", "B", "none"], default="A")
    parser.add_argument("--density", type=float, default=300.0)
    parser.add_argument("--meters-per-unit", type=float, default=1.0)
    parser.add_argument("--reference-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument(
        "--dynamic-collision-approximation",
        default="convexHull",
        help="PhysX approximation for collision Mesh prims under dynamic bodies. Use '' to disable.",
    )
    parser.add_argument(
        "--use-articulation-root",
        action="store_true",
        help="Apply ArticulationRootAPI. Do not use with --fixed-body A/B; regular joints are the default.",
    )
    parser.add_argument(
        "--keep-package-source",
        action="store_true",
        help="When output is .usdz, keep the temporary USD package source directory for inspection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_data = load_split_json(Path(args.split_json)) if args.split_json else None

    hinge_point_arg = args.hinge_point
    hinge_axis_arg = args.hinge_axis
    if split_data is not None:
        hinge_point_arg = hinge_point_arg or tuple(split_data["hinge_point"])
        hinge_axis_arg = hinge_axis_arg or tuple(split_data["hinge_axis"])
    if hinge_point_arg is None or hinge_axis_arg is None:
        raise ValueError("--hinge-point and --hinge-axis are required unless --split-json provides them")

    part_a_usdz = Path(args.part_a_usdz)
    part_b_usdz = Path(args.part_b_usdz)
    if not part_a_usdz.is_file():
        raise FileNotFoundError(part_a_usdz)
    if not part_b_usdz.is_file():
        raise FileNotFoundError(part_b_usdz)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix.lower() == ".usdz":
        temp_parent = output.parent
        temp_context = tempfile.TemporaryDirectory(prefix=f"{output.stem}_package_", dir=str(temp_parent))
        package_source = Path(temp_context.name)
        try:
            packaged_part_a = package_source / "part_a.usdz"
            packaged_part_b = package_source / "part_b.usdz"
            shutil.copy2(part_a_usdz, packaged_part_a)
            shutil.copy2(part_b_usdz, packaged_part_b)

            package_root = package_source / f"{output.stem}.usda"
            build_assembly_stage(
                package_root,
                "part_a.usdz",
                "part_b.usdz",
                args,
                hinge_point_arg,
                hinge_axis_arg,
            )
            if not UsdUtils.CreateNewUsdzPackage(str(package_root.resolve()), str(output.resolve())):
                raise RuntimeError(f"UsdUtils.CreateNewUsdzPackage failed for {output}")
            if args.keep_package_source:
                kept = output.with_suffix(".package_source")
                if kept.exists():
                    raise FileExistsError(f"Package source output already exists: {kept}")
                shutil.copytree(package_source, kept)
                print(f"[package_source] {kept.resolve()}")
        finally:
            temp_context.cleanup()
    else:
        build_assembly_stage(
            output,
            make_reference_path(part_a_usdz, output.parent, args.reference_mode),
            make_reference_path(part_b_usdz, output.parent, args.reference_mode),
            args,
            hinge_point_arg,
            hinge_axis_arg,
        )

    hinge_point = np.array(hinge_point_arg, dtype=np.float64)
    hinge_axis = normalized(np.array(hinge_axis_arg, dtype=np.float64))
    print(f"[part_a] {part_a_usdz.resolve()}")
    print(f"[part_b] {part_b_usdz.resolve()}")
    print(f"[hinge] point={hinge_point} axis={hinge_axis} limits=({args.lower_limit_deg}, {args.upper_limit_deg})")
    print(f"[fixed_body] {args.fixed_body}")
    print(f"[save] {output.resolve()}")


if __name__ == "__main__":
    main()
