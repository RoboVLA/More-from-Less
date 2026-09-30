from dataclasses import MISSING, fields
import time
from typing import Any

import isaaclab.envs.mdp as mdp
import torch
from leisaac.assets.robots.lerobot import SO101_FOLLOWER_USD_JOINT_LIMLITS


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _normalize_device_name(device: str) -> str:
    """Normalize teleop device names to existing action cfg keys."""
    if device in ["keyboard-world", "macro-keyboard"]:
        return "keyboard"
    return device


def _has_any(action: dict[str, Any], keys: list[str]) -> bool:
    """Check whether any of given keys exists and is not None in action dict."""
    return any(action.get(k) is not None for k in keys)


# ------------------------------------------------------------
# Action cfg init
# ------------------------------------------------------------
def init_action_cfg(action_cfg, device):
    """SO101 Follower action configuration: arm_action and gripper_action"""
    device = _normalize_device_name(device)

    if device in ["so101leader", "lekiwi-leader"]:
        action_cfg.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=1.0,
        )
        action_cfg.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=1.0,
        )

    elif device in ["keyboard", "gamepad", "lekiwi-keyboard", "lekiwi-gamepad"]:
        action_cfg.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            body_name="gripper",
            controller=mdp.DifferentialIKControllerCfg(command_type="pose", ik_method="dls", use_relative_mode=True),
        )
        action_cfg.gripper_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "gripper"],
            scale=1.0,
        )

    elif device in ["bi-so101leader"]:
        action_cfg.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="left_arm",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=1.0,
        )
        action_cfg.left_gripper_action = mdp.JointPositionActionCfg(
            asset_name="left_arm",
            joint_names=["gripper"],
            scale=1.0,
        )
        action_cfg.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="right_arm",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=1.0,
        )
        action_cfg.right_gripper_action = mdp.JointPositionActionCfg(
            asset_name="right_arm",
            joint_names=["gripper"],
            scale=1.0,
        )

    elif device in ["mimic_so101leader"]:
        action_cfg.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            body_name="gripper",
            controller=mdp.DifferentialIKControllerCfg(command_type="pose", ik_method="dls", use_relative_mode=False),
        )
        action_cfg.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=1.0,
        )

    elif device in ["mimic_keyboard", "mimic_gamepad"]:
        action_cfg.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            body_name="gripper",
            controller=mdp.DifferentialIKControllerCfg(command_type="pose", ik_method="dls", use_relative_mode=False),
        )
        action_cfg.gripper_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=1.0,
        )

    """LeKiwi action configuration"""
    if device in ["lekiwi-leader", "lekiwi-keyboard", "lekiwi-gamepad"]:
        action_cfg.wheel_action = mdp.JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["base_x", "base_y", "base_theta"],
            scale=1.0,
        )

    """Check if all the action configurations are set"""
    for field in fields(action_cfg):
        value = getattr(action_cfg, field.name, None)
        if value is None or value is MISSING:
            raise ValueError(f"Action configuration '{field.name}' for {device} is not set")

    return action_cfg


# ------------------------------------------------------------
# SO101 leader conversion
# ------------------------------------------------------------
joint_names_to_motor_ids = {
    "shoulder_pan": 0,
    "shoulder_lift": 1,
    "elbow_flex": 2,
    "wrist_flex": 3,
    "wrist_roll": 4,
    "gripper": 5,
}


def _to_debug_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().flatten()[0].cpu().item())
    return float(value)


def _maybe_print_so101_leader_debug(
    joint_state: dict[str, float],
    motor_limits: dict[str, tuple[float, float]],
    processed_action: torch.Tensor,
    teleop_device,
    label: str,
) -> None:
    if not bool(getattr(teleop_device, "debug_leader", False)):
        return

    period = max(0.0, float(getattr(teleop_device, "debug_leader_period", 0.25)))
    now = time.perf_counter()
    last_by_label = getattr(teleop_device, "_debug_leader_last_print_by_label", None)
    if last_by_label is None:
        last_by_label = {}
        setattr(teleop_device, "_debug_leader_last_print_by_label", last_by_label)

    last = float(last_by_label.get(label, 0.0))
    if period > 0.0 and now - last < period:
        return
    last_by_label[label] = now

    shoulder_pan_raw = _to_debug_float(joint_state["shoulder_pan"])
    shoulder_pan_cmd_rad = _to_debug_float(processed_action[0, joint_names_to_motor_ids["shoulder_pan"]])
    shoulder_pan_cmd_deg = shoulder_pan_cmd_rad * 180.0 / float(torch.pi)
    motor_min, motor_max = motor_limits["shoulder_pan"]
    print(
        "[leader-debug]"
        f" {label} shoulder_pan_raw={shoulder_pan_raw:.4f}"
        f" motor_limit=({float(motor_min):.4f},{float(motor_max):.4f})"
        f" -> follower_cmd_rad={shoulder_pan_cmd_rad:.6f}"
        f" follower_cmd_deg={shoulder_pan_cmd_deg:.3f}",
        flush=True,
    )


def convert_action_from_so101_leader(
    joint_state: dict[str, float],
    motor_limits: dict[str, tuple[float, float]],
    teleop_device,
    debug_label: str = "leader",
) -> torch.Tensor:
    processed_action = torch.zeros(teleop_device.env.num_envs, 6, device=teleop_device.env.device)
    joint_limits = SO101_FOLLOWER_USD_JOINT_LIMLITS
    for joint_name, motor_id in joint_names_to_motor_ids.items():
        motor_limit_range = motor_limits[joint_name]
        joint_limit_range = joint_limits[joint_name]
        motor_range = motor_limit_range[1] - motor_limit_range[0]
        joint_range = joint_limit_range[1] - joint_limit_range[0]
        motor_degree = joint_state[joint_name] - motor_limit_range[0]
        processed_degree = motor_degree / motor_range * joint_range + joint_limit_range[0]
        processed_radius = processed_degree / 180.0 * torch.pi  # convert degree to radius
        processed_action[:, motor_id] = processed_radius
    _maybe_print_so101_leader_debug(joint_state, motor_limits, processed_action, teleop_device, debug_label)
    return processed_action


# ------------------------------------------------------------
# Action preprocess (device output -> env action tensor)
# ------------------------------------------------------------
def preprocess_device_action(action: dict[str, Any], teleop_device) -> torch.Tensor:
    # Compatibility: allow upstream action dict to use keyboard aliases.
    # Mirror them to "keyboard" so existing logic works.
    _keyboard_aliases = ["keyboard-world", "macro-keyboard"]
    for k in _keyboard_aliases:
        if action.get(k) is not None and action.get("keyboard") is None:
            action["keyboard"] = action[k]
            break

    if action.get("so101_leader") is not None:
        processed_action = convert_action_from_so101_leader(
            action["joint_state"], action["motor_limits"], teleop_device, debug_label="so101_leader"
        )

    elif _has_any(action, ["keyboard", "gamepad"]):
        # keyboard-like: 8-dim delta action
        processed_action = torch.zeros(teleop_device.env.num_envs, 8, device=teleop_device.env.device)
        processed_action[:, :] = action["joint_state"]

    elif action.get("bi_so101_leader") is not None:
        processed_action = torch.zeros(teleop_device.env.num_envs, 12, device=teleop_device.env.device)
        processed_action[:, :6] = convert_action_from_so101_leader(
            action["joint_state"]["left_arm"],
            action["motor_limits"]["left_arm"],
            teleop_device,
            debug_label="left_so101_leader",
        )
        processed_action[:, 6:] = convert_action_from_so101_leader(
            action["joint_state"]["right_arm"],
            action["motor_limits"]["right_arm"],
            teleop_device,
            debug_label="right_so101_leader",
        )

    elif action.get("lekiwi-leader") is not None:
        processed_action = torch.zeros(teleop_device.env.num_envs, 9, device=teleop_device.env.device)
        processed_action[:, :6] = convert_action_from_so101_leader(
            action["joint_state"]["arm_action"], action["motor_limits"], teleop_device, debug_label="lekiwi_arm"
        )
        processed_action[:, 6:] = action["joint_state"]["wheel_action"]

    elif _has_any(action, ["lekiwi-keyboard", "lekiwi-gamepad"]):
        processed_action = torch.zeros(teleop_device.env.num_envs, 11, device=teleop_device.env.device)
        processed_action[:, :] = action["joint_state"]

    else:
        raise NotImplementedError(f"Not implemented for this device now: {teleop_device.device_type}")

    return processed_action
