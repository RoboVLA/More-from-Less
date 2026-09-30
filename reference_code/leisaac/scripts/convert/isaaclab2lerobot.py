import argparse
import copy
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

"""
NOTE: Please use the environment of lerobot.

Because lerobot is rapidly developing, we don't guarantee the compatibility for the latest version of lerobot.
Currently, the commit we used is https://github.com/huggingface/lerobot/tree/v0.3.3
"""

# Feature definition for single-arm so101_follower
SINGLE_ARM_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ],
    },
    "observation.images.front": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.wrist": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
}

# Feature definition for bi-arm so101_follower
BI_ARM_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (12,),
        "names": [
            "left_shoulder_pan.pos",
            "left_shoulder_lift.pos",
            "left_elbow_flex.pos",
            "left_wrist_flex.pos",
            "left_wrist_roll.pos",
            "left_gripper.pos",
            "right_shoulder_pan.pos",
            "right_shoulder_lift.pos",
            "right_elbow_flex.pos",
            "right_wrist_flex.pos",
            "right_wrist_roll.pos",
            "right_gripper.pos",
        ],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (12,),
        "names": [
            "left_shoulder_pan.pos",
            "left_shoulder_lift.pos",
            "left_elbow_flex.pos",
            "left_wrist_flex.pos",
            "left_wrist_roll.pos",
            "left_gripper.pos",
            "right_shoulder_pan.pos",
            "right_shoulder_lift.pos",
            "right_elbow_flex.pos",
            "right_wrist_flex.pos",
            "right_wrist_roll.pos",
            "right_gripper.pos",
        ],
    },
    "observation.images.left_wrist": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.top": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.right_wrist": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
}

# preprocess actions and joint pos
ISAACLAB_JOINT_POS_LIMIT_RANGE = [
    (-110.0, 110.0),
    (-100.0, 100.0),
    (-100.0, 90.0),
    (-95.0, 95.0),
    (-160.0, 160.0),
    (-10, 100.0),
]
LEROBOT_JOINT_POS_LIMIT_RANGE = [
    (-100, 100),
    (-100, 100),
    (-100, 100),
    (-100, 100),
    (-100, 100),
    (0, 100),
]


ACTION_SOURCE_CHOICES = [
    "auto",
    "actions",
    "processed_actions",
    "obs/actions",
    "obs/joint_pos_target",
    "obs/joint_pos",
]


def _dataset_exists(group: h5py.Group, path: str) -> bool:
    try:
        group[path]
        return True
    except KeyError:
        return False


def _read_dataset(group: h5py.Group, path: str) -> np.ndarray:
    return np.asarray(group[path], dtype=np.float32)


def _last_dim(array: np.ndarray) -> int:
    if array.ndim < 2:
        raise ValueError(f"Expected a 2D array, got shape={array.shape}")
    return int(array.shape[-1])


def preprocess_joint_pos(joint_pos: np.ndarray) -> np.ndarray:
    """Convert SO101 Isaac joint radians to LeRobot motor-range values.

    Works for one arm (6 dims) and bi-arm concatenated joint arrays (12 dims).
    """

    joint_pos = np.asarray(joint_pos, dtype=np.float32).copy()
    if joint_pos.shape[-1] % 6 != 0:
        raise ValueError(f"Joint position array must have a multiple of 6 dims, got shape={joint_pos.shape}")

    joint_pos = joint_pos / np.pi * 180
    for block_start in range(0, joint_pos.shape[-1], 6):
        for i in range(6):
            isaaclab_min, isaaclab_max = ISAACLAB_JOINT_POS_LIMIT_RANGE[i]
            lerobot_min, lerobot_max = LEROBOT_JOINT_POS_LIMIT_RANGE[i]
            isaac_range = isaaclab_max - isaaclab_min
            lerobot_range = lerobot_max - lerobot_min
            col = block_start + i
            joint_pos[:, col] = (joint_pos[:, col] - isaaclab_min) / isaac_range * lerobot_range + lerobot_min
    return joint_pos


def resolve_single_arm_action_source(demo_group: h5py.Group, requested_source: str) -> str:
    if requested_source != "auto":
        return requested_source

    action_dim = _last_dim(demo_group["actions"])
    state_dim = _last_dim(demo_group["obs/joint_pos"])
    if action_dim == state_dim:
        return "actions"

    for candidate in ("obs/joint_pos_target", "obs/actions", "obs/joint_pos"):
        if _dataset_exists(demo_group, candidate) and _last_dim(demo_group[candidate]) == state_dim:
            print(
                f"[action-source:auto] raw actions dim={action_dim}, state dim={state_dim}; "
                f"using {candidate} as 6-D LeRobot action."
            )
            return candidate

    raise ValueError(
        f"Cannot infer a LeRobot action source. HDF5 actions dim={action_dim}, state dim={state_dim}. "
        f"Pass --action-source explicitly. Choices: {ACTION_SOURCE_CHOICES}"
    )


def resolve_bi_arm_action_source(demo_group: h5py.Group, requested_source: str) -> str:
    if requested_source != "auto":
        return requested_source

    action_dim = _last_dim(demo_group["actions"])
    state_dim = _last_dim(demo_group["obs/left_joint_pos"]) + _last_dim(demo_group["obs/right_joint_pos"])
    if action_dim == state_dim:
        return "actions"

    # Manager-based bi-arm recordings often expose left/right targets separately.
    if _dataset_exists(demo_group, "obs/left_joint_pos_target") and _dataset_exists(demo_group, "obs/right_joint_pos_target"):
        return "obs/joint_pos_target"

    raise ValueError(
        f"Cannot infer bi-arm LeRobot action source. HDF5 actions dim={action_dim}, state dim={state_dim}. "
        f"Pass --action-source explicitly."
    )


def read_single_arm_actions(demo_group: h5py.Group, requested_source: str) -> tuple[np.ndarray, str]:
    source = resolve_single_arm_action_source(demo_group, requested_source)
    actions = _read_dataset(demo_group, source)
    if _last_dim(actions) % 6 == 0:
        actions = preprocess_joint_pos(actions)
    return actions, source


def read_bi_arm_actions(demo_group: h5py.Group, requested_source: str) -> tuple[np.ndarray, str]:
    source = resolve_bi_arm_action_source(demo_group, requested_source)
    if source == "obs/joint_pos_target":
        left = _read_dataset(demo_group, "obs/left_joint_pos_target")
        right = _read_dataset(demo_group, "obs/right_joint_pos_target")
        actions = np.concatenate([left, right], axis=-1)
    else:
        actions = _read_dataset(demo_group, source)
    if _last_dim(actions) % 6 == 0:
        actions = preprocess_joint_pos(actions)
    return actions, source


def make_features(robot_type: str, action_dim: int) -> dict:
    features = copy.deepcopy(SINGLE_ARM_FEATURES if robot_type == "so101_follower" else BI_ARM_FEATURES)
    features["action"]["shape"] = (int(action_dim),)
    if robot_type == "so101_follower" and action_dim == 6:
        features["action"]["names"] = SINGLE_ARM_FEATURES["action"]["names"]
    elif robot_type == "bi_so101_follower" and action_dim == 12:
        features["action"]["names"] = BI_ARM_FEATURES["action"]["names"]
    else:
        features["action"]["names"] = [f"action_{i}" for i in range(int(action_dim))]
    return features


def infer_action_dim_and_source(hdf5_files: list[str], robot_type: str, action_source: str) -> tuple[int, str]:
    for hdf5_file in hdf5_files:
        with h5py.File(hdf5_file, "r") as f:
            for demo_name in f["data"].keys():
                demo_group = f["data"][demo_name]
                if robot_type == "so101_follower":
                    actions, resolved_source = read_single_arm_actions(demo_group, action_source)
                else:
                    actions, resolved_source = read_bi_arm_actions(demo_group, action_source)
                return _last_dim(actions), resolved_source
    raise ValueError("No demos found in input HDF5 files.")


def process_single_arm_data(
    dataset: LeRobotDataset, task: str, demo_group: h5py.Group, demo_name: str, action_source: str
) -> bool:
    try:
        joint_pos = np.array(demo_group["obs/joint_pos"])
        front_images = np.array(demo_group["obs/front"])
        wrist_images = np.array(demo_group["obs/wrist"])
    except KeyError:
        print(f"Demo {demo_name} is not valid, skip it")
        return False

    actions, resolved_action_source = read_single_arm_actions(demo_group, action_source)
    if actions.shape[0] < 10:
        print(f"Demo {demo_name} has less than 10 frames, skip it")
        return False

    # preprocess joint pos
    joint_pos = preprocess_joint_pos(joint_pos)
    print(f"[{demo_name}] action_source={resolved_action_source}, action_shape={actions.shape}")

    assert actions.shape[0] == joint_pos.shape[0] == front_images.shape[0] == wrist_images.shape[0]
    total_state_frames = actions.shape[0]
    # skip the first 5 frames
    for frame_index in tqdm(range(5, total_state_frames), desc="Processing each frame"):
        frame = {
            "action": actions[frame_index],
            "observation.state": joint_pos[frame_index],
            "observation.images.front": front_images[frame_index],
            "observation.images.wrist": wrist_images[frame_index],
        }
        dataset.add_frame(frame=frame, task=task)

    return True


def process_bi_arm_data(
    dataset: LeRobotDataset, task: str, demo_group: h5py.Group, demo_name: str, action_source: str
) -> bool:
    try:
        left_joint_pos = np.array(demo_group["obs/left_joint_pos"])
        right_joint_pos = np.array(demo_group["obs/right_joint_pos"])
        left_images = np.array(demo_group["obs/left_wrist"])
        right_images = np.array(demo_group["obs/right_wrist"])
        top_images = np.array(demo_group["obs/top"])
    except KeyError:
        print(f"Demo {demo_name} is not valid, skip it")
        return False

    actions, resolved_action_source = read_bi_arm_actions(demo_group, action_source)
    if actions.shape[0] < 10:
        print(f"Demo {demo_name} has less than 10 frames, skip it")
        return False

    # preprocess joint pos
    left_joint_pos = preprocess_joint_pos(left_joint_pos)
    right_joint_pos = preprocess_joint_pos(right_joint_pos)
    print(f"[{demo_name}] action_source={resolved_action_source}, action_shape={actions.shape}")

    assert (
        actions.shape[0]
        == left_joint_pos.shape[0]
        == right_joint_pos.shape[0]
        == left_images.shape[0]
        == right_images.shape[0]
        == top_images.shape[0]
    )
    total_state_frames = actions.shape[0]
    # skip the first 5 frames
    for frame_index in tqdm(range(5, total_state_frames), desc="Processing each frame"):
        frame = {
            "action": actions[frame_index],
            "observation.state": np.concatenate([left_joint_pos[frame_index], right_joint_pos[frame_index]]),
            "observation.images.left_wrist": left_images[frame_index],
            "observation.images.top": top_images[frame_index],
            "observation.images.right_wrist": right_images[frame_index],
        }
        dataset.add_frame(frame=frame, task=task)

    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Convert IsaacLab HDF5 demos to a LeRobotDataset.")
    parser.add_argument("--repo_id", default="EverNorif/so101_test_orange_pick01")
    parser.add_argument("--task_name", default="Grab orange and place into plate")
    parser.add_argument("--task", default=None, help="Natural-language LeRobot task string. Defaults to --task_name.")
    parser.add_argument("--robot_type", choices=["auto", "so101_follower", "bi_so101_follower"], default="auto")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hdf5_root", default="./datasets")
    parser.add_argument("--hdf5_files", nargs="+", default=None)
    parser.add_argument("--action-source", choices=ACTION_SOURCE_CHOICES, default="auto")
    parser.add_argument("--root", default=None, help="Optional local LeRobot dataset root.")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Inspect input/action source without writing a dataset.")
    return parser.parse_args()


def resolve_hdf5_files(hdf5_root: str, hdf5_files: list[str] | None) -> list[str]:
    if not hdf5_files:
        hdf5_files = ["dataset.hdf5"]

    flat_files = []
    for item in hdf5_files:
        flat_files.extend(part.strip() for part in str(item).split(",") if part.strip())

    resolved = []
    root = Path(hdf5_root)
    for item in flat_files:
        path = Path(item)
        if not path.is_absolute() and not path.exists():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved.append(str(path))
    return resolved


def convert_isaaclab_to_lerobot():
    args = parse_args()

    repo_id = args.repo_id
    task_name = args.task_name
    robot_type = args.robot_type
    fps = args.fps
    hdf5_files = resolve_hdf5_files(args.hdf5_root, args.hdf5_files)
    task = args.task if args.task is not None else task_name
    push_to_hub = args.push_to_hub

    if robot_type == "auto":
        robot_type = "bi_so101_follower" if "BiArm" in str(task_name) else "so101_follower"

    """parameters check"""
    assert robot_type in [
        "so101_follower",
        "bi_so101_follower",
    ], "robot_type must be so101_follower or bi_so101_follower"

    action_dim, resolved_action_source = infer_action_dim_and_source(hdf5_files, robot_type, args.action_source)
    print(
        f"[config] repo_id={repo_id}, robot_type={robot_type}, task={task}, "
        f"action_source={resolved_action_source}, action_dim={action_dim}"
    )

    if args.dry_run:
        print("[dry-run] Input inspection finished. No LeRobot dataset was written.")
        return

    """convert to LeRobotDataset"""
    now_episode_index = 0
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=make_features(robot_type, action_dim),
        root=args.root,
    )

    for hdf5_id, hdf5_file in enumerate(hdf5_files):
        print(f"[{hdf5_id+1}/{len(hdf5_files)}] Processing hdf5 file: {hdf5_file}")
        with h5py.File(hdf5_file, "r") as f:
            demo_names = list(f["data"].keys())
            print(f"Found {len(demo_names)} demos: {demo_names}")

            for demo_name in tqdm(demo_names, desc="Processing each demo"):
                demo_group = f["data"][demo_name]
                # if "success" in demo_group.attrs and not demo_group.attrs["success"]:
                #     print(f"Demo {demo_name} is not successful, skip it")
                #     continue

                if robot_type == "so101_follower":
                    valid = process_single_arm_data(dataset, task, demo_group, demo_name, args.action_source)
                elif robot_type == "bi_so101_follower":
                    valid = process_bi_arm_data(dataset, task, demo_group, demo_name, args.action_source)

                if valid:
                    now_episode_index += 1
                    dataset.save_episode()
                    print(f"Saving episode {now_episode_index} successfully")

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    convert_isaaclab_to_lerobot()
