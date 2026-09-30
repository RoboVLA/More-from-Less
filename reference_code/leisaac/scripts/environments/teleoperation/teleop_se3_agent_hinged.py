# -*- coding: utf-8 -*-
"""
Teleoperation entry point for the hinge_case USDZ scene.

This script keeps the normal LeIsaac teleop/recording flow, but replaces the
default task scene with assets under assets/hinge_case:

- table/table_vis.usdz: static/kinematic support with collision
- stick/stick_vis.usdz: dynamic rigid asset with gravity
- microwave/microwave_hinged_assembly.usdz: dynamic hinged assembly
- medicine_cabinet/medicine_cabinet_vis_hinged_assembly.usdz: dynamic hinged assembly
- cup/cup_vis.usdz: dynamic rigid asset with gravity

The USDZ files are spawned as AssetBaseCfg extras, then patched on the live USD
stage so nested rigid bodies and collision meshes keep working.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
from pathlib import Path
from typing import Iterable

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

from isaaclab.app import AppLauncher


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_hinge_root() -> str:
    return str(_repo_root() / "assets" / "hinge_case")


def parse_vec3(text: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in str(text).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected x,y,z")
    return (parts[0], parts[1], parts[2])


def parse_scale3(text: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in str(text).split(",")]
    if len(parts) == 1:
        return (parts[0], parts[0], parts[0])
    if len(parts) == 3:
        return (parts[0], parts[1], parts[2])
    raise argparse.ArgumentTypeError("Expected scale or sx,sy,sz")


def quat_wxyz_from_euler_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    cr = math.cos(r * 0.5)
    sr = math.sin(r * 0.5)
    cp = math.cos(p * 0.5)
    sp = math.sin(p * 0.5)
    cy = math.cos(y * 0.5)
    sy = math.sin(y * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    yv = cy * sp * cr + sy * cp * sr
    z = sy * cp * cr - cy * sp * sr
    return (w, x, yv, z)


parser = argparse.ArgumentParser(description="LeIsaac teleoperation with the prebuilt hinge_case USDZ scene.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    choices=[
        "keyboard",
        "gamepad",
        "keyboard-world",
        "macro-keyboard",
        "so101leader",
        "bi-so101leader",
        "lekiwi-keyboard",
        "lekiwi-gamepad",
        "lekiwi-leader",
        "rule-grasp",
    ],
)
parser.add_argument("--port", type=str, default="/dev/ttyACM0")
parser.add_argument("--left_arm_port", type=str, default="/dev/ttyACM0")
parser.add_argument("--right_arm_port", type=str, default="/dev/ttyACM1")
parser.add_argument("--task", type=str, default="LeIsaac-SO101-PickUnkownObject-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--sensitivity", type=float, default=1.0)
parser.add_argument(
    "--debug-leader",
    action="store_true",
    help="Print SO101 leader shoulder_pan input and converted follower joint command.",
)
parser.add_argument(
    "--debug-leader-period",
    type=float,
    default=0.25,
    help="Seconds between --debug-leader prints. Use 0 to print every frame.",
)

parser.add_argument("--record", action="store_true")
parser.add_argument("--step_hz", type=int, default=60)
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset_hinged.hdf5")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--num_demos", type=int, default=0)

parser.add_argument("--recalibrate", action="store_true")
parser.add_argument("--quality", action="store_true")

# Rule-grasp compatibility with leisaac.utils.teleop_device_factory.
parser.add_argument("--ee_prim_path", type=str, default="", help="EE USD prim relative path, without per-env root.")
parser.add_argument("--autograsp_pregrasp_dz", type=float, default=0.10)
parser.add_argument("--autograsp_grasp_dz", type=float, default=0.02)
parser.add_argument("--autograsp_lift_dz", type=float, default=0.15)
parser.add_argument("--autograsp_xy_gain", type=float, default=6.0)
parser.add_argument("--autograsp_z_gain", type=float, default=6.0)
parser.add_argument("--autograsp_yaw_gain", type=float, default=3.0)
parser.add_argument("--autograsp_close_steps", type=int, default=25)
parser.add_argument("--autograsp_settle_steps", type=int, default=10)
parser.add_argument("--autograsp_success_on_lift", action="store_true")
parser.add_argument("--autograsp_open_val", type=float, default=1.0)
parser.add_argument("--autograsp_close_val", type=float, default=-1.0)

# Hinge-case assets and placement.
parser.add_argument("--hinge-assets-root", type=str, default=_default_hinge_root())
parser.add_argument("--table-usdz", type=str, default=None)
parser.add_argument("--stick-usdz", type=str, default=None)
parser.add_argument("--microwave-usdz", type=str, default=None)
parser.add_argument("--medicine-cabinet-usdz", type=str, default=None)
parser.add_argument("--cup-usdz", type=str, default=None)
parser.add_argument(
    "--scene-pos",
    type=parse_vec3,
    default=(1.2, -0.095, 0.06),
    help="Common world translation for assets that share the same reconstruction frame.",
)
parser.add_argument(
    "--scene-rot-deg",
    type=parse_vec3,
    default=(225.0, 0.0, 0.0),
    help="Common roll,pitch,yaw rotation in degrees. Use 35,-1,5 only if you want the old MY_SCENE tilt.",
)
parser.add_argument("--scene-scale", type=float, default=0.08)
parser.add_argument("--table-offset", type=parse_vec3, default=(0.0, 0.0, 0.0))
parser.add_argument("--stick-offset", type=parse_vec3, default=(0.0, 0.0, 0.0))
parser.add_argument("--microwave-offset", type=parse_vec3, default=(0.0, 0.0, 0.0))
parser.add_argument("--medicine-cabinet-offset", type=parse_vec3, default=(0.0, 0.0, 0.0))
parser.add_argument("--cup-offset", type=parse_vec3, default=(0.0, 0.0, 0.0))
parser.add_argument("--table-pos", type=parse_vec3, default=None)
parser.add_argument("--table-rot-deg", type=parse_vec3, default=None)
parser.add_argument("--table-scale", type=parse_scale3, default=None)
parser.add_argument("--stick-pos", type=parse_vec3, default=None)
parser.add_argument("--stick-rot-deg", type=parse_vec3, default=None)
parser.add_argument("--stick-scale", type=parse_scale3, default=None)
parser.add_argument("--microwave-pos", type=parse_vec3, default=None)
parser.add_argument("--microwave-rot-deg", type=parse_vec3, default=None)
parser.add_argument("--microwave-scale", type=parse_scale3, default=None)
parser.add_argument("--medicine-cabinet-pos", type=parse_vec3, default=None)
parser.add_argument("--medicine-cabinet-rot-deg", type=parse_vec3, default=None)
parser.add_argument("--medicine-cabinet-scale", type=parse_scale3, default=None)
parser.add_argument("--cup-pos", type=parse_vec3, default=None)
parser.add_argument("--cup-rot-deg", type=parse_vec3, default=None)
parser.add_argument("--cup-scale", type=parse_scale3, default=None)
parser.add_argument("--robot-pos", type=parse_vec3, default=(1.33321, 0.11138, -0.3846))
parser.add_argument("--robot-rot-deg", type=parse_vec3, default=(-16.0, 2.0, 180.0))
parser.add_argument("--robot-scale", type=parse_scale3, default=(0.7, 0.7, 0.7))
parser.add_argument("--viewer-eye", type=parse_vec3, default=(1.33321, 0.11138, 1.35))
parser.add_argument("--viewer-lookat", type=parse_vec3, default=(1.33321, 0.11138, -0.4646))
parser.add_argument("--density", type=float, default=300.0)
parser.add_argument("--dynamic-collision-approximation", type=str, default="convexHull")
parser.add_argument(
    "--disable-gauss-collision",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Disable collision on visual 3DGS /gauss/ meshes and keep them render-only. "
        "Default is false so add_mesh_to_usdz meshes can provide convexHull collision."
    ),
)
parser.add_argument(
    "--disable-dynamic-ccd",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Disable CCD on hinge-case dynamic bodies. This avoids PhysX kinematic/CCD warnings in USDZ assemblies.",
)
parser.add_argument("--snap-to-table", action="store_true", help="Move dynamic objects upward so their AABB rests on the table top.")
parser.add_argument("--table-clearance", type=float, default=0.003)
parser.add_argument("--add-hidden-floor", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug-aabbs", action=argparse.BooleanOptionalAction, default=True)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import TerminationTermCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone

from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
from leisaac.utils.rate_limiter import RateLimiter
from leisaac.utils.recording_utils import DemoCounter, get_resume_demo_count, start_recording, stop_recording
from leisaac.utils.teleop_device_factory import create_teleop_and_agent
from leisaac.utils.teleop_env_factory import build_env_cfg, _replace_recorder_manager_if_needed


ASSET_SPECS = {
    "table": {
        "prim_name": "Table",
        "field_name": "hinge_table",
        "arg_name": "table_usdz",
        "default_rel": ("table", "table_vis.usdz"),
        "dynamic": False,
        "offset_arg": "table_offset",
    },
    "stick": {
        "prim_name": "Stick",
        "field_name": "hinge_stick",
        "arg_name": "stick_usdz",
        "default_rel": ("stick", "stick_vis.usdz"),
        "dynamic": True,
        "offset_arg": "stick_offset",
    },
    "microwave": {
        "prim_name": "Microwave",
        "field_name": "hinge_microwave",
        "arg_name": "microwave_usdz",
        "default_rel": ("microwave", "microwave_hinged_assembly.usdz"),
        "dynamic": True,
        "offset_arg": "microwave_offset",
    },
    "medicine_cabinet": {
        "prim_name": "MedicineCabinet",
        "field_name": "hinge_medicine_cabinet",
        "arg_name": "medicine_cabinet_usdz",
        "default_rel": ("medicine_cabinet", "medicine_cabinet_vis_hinged_assembly.usdz"),
        "dynamic": True,
        "offset_arg": "medicine_cabinet_offset",
    },
    "cup": {
        "prim_name": "Cup",
        "field_name": "hinge_cup",
        "arg_name": "cup_usdz",
        "default_rel": ("cup", "cup_vis.usdz"),
        "dynamic": True,
        "offset_arg": "cup_offset",
    },
}


@clone
def spawn_xform(prim_path: str, cfg: SpawnerCfg, translation=None, orientation=None):
    import isaacsim.core.utils.prims as prim_utils

    return prim_utils.create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)


def make_zero_actions(env):
    if hasattr(env, "action_manager") and hasattr(env.action_manager, "total_action_dim"):
        action_dim = int(env.action_manager.total_action_dim)
        return torch.zeros((env.num_envs, action_dim), device=env.device)
    space = getattr(env, "action_space", None)
    if space is not None and getattr(space, "shape", None) is not None:
        return torch.zeros((env.num_envs,) + tuple(space.shape), device=env.device)
    raise RuntimeError("Cannot infer action dimension.")


def manual_terminate(env: ManagerBasedRLEnv | DirectRLEnv, success: bool):
    if hasattr(env, "termination_manager"):
        value = bool(success)
        env.termination_manager.set_term_cfg(
            "success",
            TerminationTermCfg(
                func=lambda env: torch.full((env.num_envs,), value, dtype=torch.bool, device=env.device)
            ),
        )
        env.termination_manager.compute()
    elif hasattr(env, "_get_dones"):
        env.cfg.return_success_status = success


def _asset_path(args: argparse.Namespace, key: str) -> Path:
    spec = ASSET_SPECS[key]
    explicit = getattr(args, spec["arg_name"])
    path = Path(explicit) if explicit else Path(args.hinge_assets_root).joinpath(*spec["default_rel"])
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing hinge-case asset for {key}: {path}")
    return path


def _add_vec3(a: Iterable[float], b: Iterable[float]) -> tuple[float, float, float]:
    ax, ay, az = a
    bx, by, bz = b
    return (float(ax) + float(bx), float(ay) + float(by), float(az) + float(bz))


def _asset_pose_args(args: argparse.Namespace, key: str) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    spec = ASSET_SPECS[key]
    pos_override = getattr(args, f"{key}_pos")
    rot_override = getattr(args, f"{key}_rot_deg")
    scale_override = getattr(args, f"{key}_scale")

    offset = getattr(args, spec["offset_arg"])
    pos = tuple(pos_override) if pos_override is not None else _add_vec3(args.scene_pos, offset)
    rot_deg = tuple(rot_override) if rot_override is not None else tuple(args.scene_rot_deg)
    scale = tuple(scale_override) if scale_override is not None else (
        float(args.scene_scale),
        float(args.scene_scale),
        float(args.scene_scale),
    )
    return pos, rot_deg, scale


def apply_hinge_case_scene_cfg(env_cfg, args: argparse.Namespace) -> None:
    """Replace the default scene asset with the hinge_case USDZ objects."""

    # Disable the old MY_SCENE asset from PickUnkownObject; the objects below
    # become the whole scene content for this teleop variant.
    if hasattr(env_cfg.scene, "scene"):
        env_cfg.scene.scene = None

    if hasattr(env_cfg.scene, "robot"):
        env_cfg.scene.robot.init_state.pos = tuple(args.robot_pos)
        env_cfg.scene.robot.init_state.rot = quat_wxyz_from_euler_deg(*args.robot_rot_deg)
        if getattr(env_cfg.scene.robot, "spawn", None) is not None and hasattr(env_cfg.scene.robot.spawn, "scale"):
            env_cfg.scene.robot.spawn.scale = tuple(args.robot_scale)
        print(
            f"[hinge-scene] robot pose: pos={tuple(args.robot_pos)}, "
            f"rot_deg={tuple(args.robot_rot_deg)}, scale={tuple(args.robot_scale)}"
        )

    env_cfg.viewer.eye = tuple(args.viewer_eye)
    env_cfg.viewer.lookat = tuple(args.viewer_lookat)
    print(f"[hinge-scene] viewer: eye={tuple(args.viewer_eye)}, lookat={tuple(args.viewer_lookat)}")

    # IsaacLab's cloned spawners require the regex parent prim to exist before
    # spawning nested assets such as /World/envs/env_.*/HingeCase/Table.
    env_cfg.scene.hinge_case_root = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/HingeCase",
        spawn=SpawnerCfg(func=spawn_xform),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    for key, spec in ASSET_SPECS.items():
        pos, rot_deg, scale = _asset_pose_args(args, key)
        rot = quat_wxyz_from_euler_deg(*rot_deg)
        prim_path = f"{{ENV_REGEX_NS}}/HingeCase/{spec['prim_name']}"
        usd_path = str(_asset_path(args, key))
        cfg = AssetBaseCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(usd_path=usd_path, scale=scale),
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
        )
        setattr(env_cfg.scene, spec["field_name"], cfg)
        print(f"[hinge-scene] {key}: {usd_path} -> {prim_path}, pos={pos}, rot_deg={rot_deg}, scale={scale}")


def _set_bool_attr(api, names: Iterable[str], value: bool) -> bool:
    ok = False
    for name in names:
        get_name = f"Get{name}Attr"
        create_name = f"Create{name}Attr"
        attr = None
        if hasattr(api, get_name):
            try:
                attr = getattr(api, get_name)()
            except Exception:
                attr = None
        if attr is None and hasattr(api, create_name):
            try:
                attr = getattr(api, create_name)(bool(value))
            except Exception:
                attr = None
        if attr is not None:
            try:
                attr.Set(bool(value))
                ok = True
            except Exception:
                pass
    return ok


def _set_int_attr(api, name: str, value: int) -> None:
    get_name = f"Get{name}Attr"
    create_name = f"Create{name}Attr"
    attr = None
    if hasattr(api, get_name):
        try:
            attr = getattr(api, get_name)()
        except Exception:
            attr = None
    if attr is None and hasattr(api, create_name):
        try:
            attr = getattr(api, create_name)(int(value))
        except Exception:
            attr = None
    if attr is not None:
        try:
            attr.Set(int(value))
        except Exception:
            pass


def _rigid_body_prims(stage, root_path: str):
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return []
    return [prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]


def _configure_rigid_body(prim, *, dynamic: bool, density: float, enable_ccd: bool = False) -> None:
    from pxr import UsdPhysics

    try:
        from pxr import PhysxSchema
    except ImportError:
        PhysxSchema = None

    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    _set_bool_attr(rb, ["KinematicEnabled", "Kinematic"], not dynamic)

    mass = UsdPhysics.MassAPI.Apply(prim)
    try:
        attr = mass.GetDensityAttr()
        if attr:
            attr.Set(float(density))
        else:
            mass.CreateDensityAttr(float(density))
    except Exception:
        pass

    if PhysxSchema is not None:
        physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        _set_bool_attr(physx_rb, ["KinematicEnabled", "Kinematic"], not dynamic)
        _set_bool_attr(physx_rb, ["DisableGravity"], not dynamic)
        _set_bool_attr(physx_rb, ["EnableCCD"], bool(dynamic and enable_ccd))
        _set_int_attr(physx_rb, "SolverPositionIterationCount", 16)
        _set_int_attr(physx_rb, "SolverVelocityIterationCount", 4)


def _load_stage_payloads_under(stage, root_path: str) -> None:
    try:
        from pxr import Sdf, Usd
    except ImportError:
        return

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return
    try:
        stage.Load(Sdf.Path(root_path), Usd.LoadWithDescendants)
    except Exception:
        try:
            root.Load(Usd.LoadWithDescendants)
        except Exception:
            pass


def _is_gauss_visual_mesh(prim) -> bool:
    path = str(prim.GetPath()).replace("\\", "/").lower()
    return "/gauss/" in path or path.endswith("/gauss")


def _disable_collision_api(prim) -> bool:
    try:
        from pxr import PhysxSchema, UsdPhysics
    except ImportError:
        return False

    changed = False
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        try:
            collision_api = UsdPhysics.CollisionAPI.Apply(prim)
            attr = collision_api.GetCollisionEnabledAttr()
            if attr:
                attr.Set(False)
            else:
                collision_api.CreateCollisionEnabledAttr(False)
            changed = True
        except Exception:
            pass
        try:
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
            changed = True
        except Exception:
            pass

    try:
        if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            prim.RemoveAPI(PhysxSchema.PhysxCollisionAPI)
            changed = True
    except Exception:
        pass

    return changed


def _set_collision_approximation_under(
    stage,
    root_path: str,
    approximation: str,
    *,
    disable_gauss_collision: bool = True,
) -> tuple[int, int]:
    if not approximation:
        return (0, 0)
    try:
        from pxr import PhysxSchema, Usd, UsdPhysics
    except ImportError:
        return (0, 0)

    _load_stage_payloads_under(stage, root_path)
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return (0, 0)

    approx_count = 0
    disabled_count = 0
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() != "Mesh":
            continue

        if disable_gauss_collision and _is_gauss_visual_mesh(prim):
            if _disable_collision_api(prim):
                disabled_count += 1
            continue

        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        try:
            physx_col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            physx_col.CreateCollisionApproximationAttr().Set(str(approximation))
            approx_count += 1
        except Exception:
            pass
    return (approx_count, disabled_count)


def _compute_world_aabb(stage, root_path: str):
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return None
    included = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), included, True)
    box = bbox_cache.ComputeWorldBound(root).ComputeAlignedBox()
    return box.GetMin(), box.GetMax()


def _format_aabb(stage, root_path: str) -> str:
    aabb = _compute_world_aabb(stage, root_path)
    if aabb is None:
        return "missing"
    mn, mx = aabb
    size = (float(mx[0] - mn[0]), float(mx[1] - mn[1]), float(mx[2] - mn[2]))
    center = (float((mn[0] + mx[0]) * 0.5), float((mn[1] + mx[1]) * 0.5), float((mn[2] + mx[2]) * 0.5))
    return (
        f"min=({float(mn[0]):.3f},{float(mn[1]):.3f},{float(mn[2]):.3f}) "
        f"max=({float(mx[0]):.3f},{float(mx[1]):.3f},{float(mx[2]):.3f}) "
        f"center=({center[0]:.3f},{center[1]:.3f},{center[2]:.3f}) "
        f"size=({size[0]:.3f},{size[1]:.3f},{size[2]:.3f})"
    )


def _translate_root_z(stage, root_path: str, dz: float) -> None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(root_path)
    if not prim.IsValid() or abs(float(dz)) <= 1e-9:
        return
    xform = UsdGeom.Xformable(prim)
    translate_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        cur = Gf.Vec3d(0.0, 0.0, 0.0)
    else:
        cur = translate_op.Get()
        if cur is None:
            cur = Gf.Vec3d(0.0, 0.0, 0.0)
    translate_op.Set(Gf.Vec3d(float(cur[0]), float(cur[1]), float(cur[2]) + float(dz)))


def postprocess_hinge_case_stage(stage, num_envs: int, args: argparse.Namespace) -> None:
    """Patch physics on the spawned USDZ assets after IsaacLab creates the stage."""

    for env_id in range(num_envs):
        base = f"/World/envs/env_{env_id}/HingeCase"
        robot_path = f"/World/envs/env_{env_id}/Robot"

        if args.debug_aabbs:
            robot_prim = stage.GetPrimAtPath(robot_path)
            print(f"[hinge-scene] env_{env_id}: robot prim valid={robot_prim.IsValid()} path={robot_path}")

        if args.add_hidden_floor:
            try:
                from leisaac.utils.physics_prims import create_static_collider_box

                create_static_collider_box(
                    prim_path=f"/World/envs/env_{env_id}/HingeCaseProxy/Floor",
                    pos=(0.55, 0.0, -0.469),
                    size_xyz=(6.0, 6.0, 0.1),
                    visible=False,
                )
            except Exception as exc:
                print(f"[hinge-scene] WARN: failed to create hidden floor for env_{env_id}: {exc}")

        table_path = f"{base}/Table"
        _load_stage_payloads_under(stage, table_path)
        table_bodies = _rigid_body_prims(stage, table_path)
        for body in table_bodies:
            _configure_rigid_body(body, dynamic=False, density=args.density, enable_ccd=False)
        print(f"[hinge-scene] env_{env_id}: table rigid bodies fixed={len(table_bodies)}")
        if args.debug_aabbs:
            print(f"[hinge-scene] env_{env_id}: table AABB {_format_aabb(stage, table_path)}")

        table_top_z = None
        if args.snap_to_table:
            aabb = _compute_world_aabb(stage, table_path)
            if aabb is not None:
                table_top_z = float(aabb[1][2])

        for key, spec in ASSET_SPECS.items():
            if key == "table":
                continue
            root_path = f"{base}/{spec['prim_name']}"
            _load_stage_payloads_under(stage, root_path)
            bodies = _rigid_body_prims(stage, root_path)
            if not bodies:
                root = stage.GetPrimAtPath(root_path)
                if root.IsValid():
                    bodies = [root]
            for body in bodies:
                _configure_rigid_body(
                    body,
                    dynamic=True,
                    density=args.density,
                    enable_ccd=not bool(args.disable_dynamic_ccd),
                )
            approx_count, disabled_gauss_count = _set_collision_approximation_under(
                stage,
                root_path,
                str(args.dynamic_collision_approximation),
                disable_gauss_collision=bool(args.disable_gauss_collision),
            )
            if table_top_z is not None:
                aabb = _compute_world_aabb(stage, root_path)
                if aabb is not None:
                    min_z = float(aabb[0][2])
                    dz = table_top_z + float(args.table_clearance) - min_z
                    if dz > 0.0:
                        _translate_root_z(stage, root_path, dz)
                        print(f"[hinge-scene] env_{env_id}: snapped {key} upward by {dz:.4f} m")
            print(
                f"[hinge-scene] env_{env_id}: {key} dynamic bodies={len(bodies)}, "
                f"collision_approx={args.dynamic_collision_approximation} ({approx_count} mesh prims), "
                f"gauss_collision_disabled={disabled_gauss_count}"
            )
            if args.debug_aabbs:
                print(f"[hinge-scene] env_{env_id}: {key} AABB {_format_aabb(stage, root_path)}")


def reactivate_physics_after_stage_patch(env) -> None:
    """Force PhysX to rebuild handles after USD physics APIs are authored."""

    try:
        env.sim.reset()
        env.scene.update(dt=env.physics_dt)
        print("[hinge-scene] PhysX reset after hinge-case physics patch -> OK")
        return
    except Exception as exc:
        print(f"[hinge-scene] WARN: env.sim.reset() after physics patch failed: {exc}")

    try:
        env.sim.forward()
        print("[hinge-scene] sim.forward after hinge-case physics patch -> OK")
    except Exception as exc:
        print(f"[hinge-scene] WARN: sim.forward() after physics patch failed: {exc}")


def create_hinged_env(args: argparse.Namespace):
    env_cfg, task_name, is_direct_env, output_dir, output_file_stem = build_env_cfg(args)
    apply_hinge_case_scene_cfg(env_cfg, args)
    env = gym.make(task_name, cfg=env_cfg).unwrapped
    _replace_recorder_manager_if_needed(env, env_cfg, args)
    return env, env_cfg, task_name, is_direct_env, output_dir, output_file_stem


def main():  # noqa: C901
    env, env_cfg, task_name, is_direct_env, _, _ = create_hinged_env(args_cli)

    print("[task]", task_name, "direct_env=", is_direct_env)
    print("[action_dim]", getattr(env.action_manager, "total_action_dim", None), "action_space:", env.action_space)
    try:
        print("[action_terms]", list(getattr(env.action_manager, "_terms").keys()))
    except Exception:
        pass

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    postprocess_hinge_case_stage(stage, env.num_envs, args_cli)
    reactivate_physics_after_stage_patch(env)

    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()

    should_reset = False
    should_success = False

    demo_counter = None
    resume_recorded_demo_count = 0
    if args_cli.record:
        if args_cli.resume:
            resume_recorded_demo_count = get_resume_demo_count(env)
            print(f"Resume recording from {resume_recorded_demo_count} demonstrations.")
        demo_counter = DemoCounter(resume_count=resume_recorded_demo_count, num_demos=args_cli.num_demos)
        demo_counter.init_from_env(env)
        start_recording(env)

    def on_reset():
        nonlocal should_reset
        should_reset = True

    built_teleop = create_teleop_and_agent(env, stage, args_cli)
    teleop_interface = built_teleop.teleop_interface
    rule_agent = built_teleop.rule_agent
    teleop_interface.reset()
    teleop_interface.add_callback("R", on_reset)

    rate_limiter = RateLimiter(args_cli.step_hz)

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                if hasattr(teleop_interface, "update_macro"):
                    teleop_interface.update_macro()

                if getattr(env.cfg, "dynamic_reset_gripper_effort_limit", False):
                    dynamic_reset_gripper_effort_limit_sim(env, args_cli.teleop_device)

                if args_cli.teleop_device == "rule-grasp":
                    actions, _req_pose_sync_off, req_success = rule_agent.act()
                    if req_success:
                        should_success = True
                else:
                    actions = teleop_interface.advance()

                if should_success:
                    print("Task Success!!!")
                    should_success = False
                    if args_cli.record:
                        manual_terminate(env, True)

                if should_reset:
                    should_reset = False
                    if args_cli.record:
                        stop_recording(env)

                    postprocess_hinge_case_stage(stage, env.num_envs, args_cli)
                    reactivate_physics_after_stage_patch(env)
                    env.reset()
                    rate_limiter.reset()

                    if args_cli.record:
                        manual_terminate(env, False)
                    if args_cli.teleop_device == "rule-grasp":
                        rule_agent.reset()
                    if args_cli.record:
                        start_recording(env)

                if actions is None:
                    actions = make_zero_actions(env)

                env.step(actions)

                if args_cli.record and demo_counter is not None:
                    demo_counter.maybe_print_update(env)
                    if demo_counter.reached_limit(env):
                        print(f"All {args_cli.num_demos} demonstrations recorded. Exiting.")
                        break

                rate_limiter.sleep(env)

    finally:
        try:
            if args_cli.record:
                stop_recording(env)
        except Exception:
            pass
        try:
            if hasattr(teleop_interface, "close"):
                teleop_interface.close()
        except Exception:
            pass
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
