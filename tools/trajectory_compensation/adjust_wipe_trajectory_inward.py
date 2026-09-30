from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
OUT_DIR = (
    WORKSPACE
    / "Data"
    / "ai\u751f\u6210\u64e6\u767d\u677f"
    / "\u64e6\u767d\u677f"
    / "\u8fd0\u52a8\u7269\u4f53\u8f68\u8ff9_\u53bb\u6c34\u5370RGB_\u5206\u5c42\u6df1\u5ea6"
)
BACKUP_DIR = OUT_DIR / "manual_adjustment_backup_before_inward_1_3"


FILES_TO_BACKUP = [
    "moving_object_pose_trajectory_xyzrpy.csv",
    "moving_object_pose_trajectory_xyzrpy.json",
    "moving_object_pose_xyzrpy_only.csv",
    "moving_object_trajectory.csv",
    "moving_object_trajectory.json",
    "pose_trajectory_web_visualization.html",
]


def copy_if_needed(src: Path, dst: Path) -> None:
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def ensure_backup(output_dir: Path) -> Path:
    backup_dir = output_dir / BACKUP_DIR.name
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES_TO_BACKUP:
        copy_if_needed(output_dir / name, backup_dir / name)
    return backup_dir


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bump(frame_index: int, center_frame: float, half_width: float, amplitude_mm: float) -> float:
    t = abs(frame_index - center_frame) / max(1e-6, half_width)
    if t >= 1.0:
        return 0.0
    return amplitude_mm * 0.5 * (1.0 + math.cos(math.pi * t))


def apply_pose_adjustment(rows: list[dict[str, str]], amplitude_mm: float, half_width_frames: float) -> list[dict[str, object]]:
    count = len(rows)
    center = (count - 1) / 3.0
    adjusted: list[dict[str, object]] = []
    for row in rows:
        item: dict[str, object] = dict(row)
        frame = int(float(row["frame"]))
        delta_y = bump(frame, center, half_width_frames, amplitude_mm)
        z = float(row["z_mm"])
        fy = None
        if "v_px" in row and row.get("v_px", "") != "":
            # Approximate inverse of y=(v-cy)*z/fy using the same horizontal-FOV
            # intrinsics used by the original extraction script.
            fy = 1108.0 / (2.0 * math.tan(math.radians(86.0) / 2.0))
        item["y_mm"] = float(row["y_mm"]) + delta_y
        if fy is not None and z > 0:
            item["v_px"] = float(row["v_px"]) + delta_y * fy / z
        adjusted.append(item)
    return adjusted


def moving_average(values: list[float], window: int = 5) -> list[float]:
    out: list[float] = []
    half = window // 2
    for idx in range(len(values)):
        lo = max(0, idx - half)
        hi = min(len(values), idx + half + 1)
        out.append(sum(values[lo:hi]) / max(1, hi - lo))
    return out


def apply_tracking_adjustment(rows: list[dict[str, str]], amplitude_mm: float, half_width_frames: float) -> list[dict[str, object]]:
    count = len(rows)
    center = (count - 1) / 3.0
    fy = 1108.0 / (2.0 * math.tan(math.radians(86.0) / 2.0))
    adjusted: list[dict[str, object]] = []
    for row in rows:
        item: dict[str, object] = dict(row)
        frame = int(float(row["frame"]))
        z = float(row["z_median_mm"])
        delta_y = bump(frame, center, half_width_frames, amplitude_mm)
        item["y_cam_mm"] = float(row["y_cam_mm"]) + delta_y
        item["v_px"] = float(row["v_px"]) + (delta_y * fy / z if z > 0 else 0.0)
        adjusted.append(item)

    for source, target in [
        ("u_px", "u_smooth_px"),
        ("v_px", "v_smooth_px"),
        ("z_median_mm", "z_smooth_mm"),
        ("x_cam_mm", "x_cam_smooth_mm"),
        ("y_cam_mm", "y_cam_smooth_mm"),
    ]:
        if source in adjusted[0] and target in adjusted[0]:
            smoothed = moving_average([float(r[source]) for r in adjusted])
            for row, value in zip(adjusted, smoothed):
                row[target] = value
    return adjusted


def write_xyzrpy_only(output_dir: Path, pose_rows: list[dict[str, object]]) -> None:
    rows = []
    for row in pose_rows:
        rows.append(
            {
                "frame": row["frame"],
                "time_s": row["time_s"],
                "x": row["x_mm"],
                "y": row["y_mm"],
                "z": row["z_mm"],
                "rx": row["rx_deg"],
                "ry": row["ry_deg"],
                "rz": row["rz_deg"],
            }
        )
    write_csv(output_dir / "moving_object_pose_xyzrpy_only.csv", rows, ["frame", "time_s", "x", "y", "z", "rx", "ry", "rz"])


def update_pose_json(backup_dir: Path, output_dir: Path, pose_rows: list[dict[str, object]], amplitude_mm: float, half_width_frames: float) -> None:
    path = backup_dir / "moving_object_pose_trajectory_xyzrpy.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["trajectory"] = pose_rows
    data["manual_adjustment"] = {
        "name": "wipe_inward_at_one_third",
        "description": "Applied a smooth camera-Y offset near one third of the trajectory so the path moves further inside the whiteboard.",
        "amplitude_y_mm": amplitude_mm,
        "half_width_frames": half_width_frames,
        "center_frame": (len(pose_rows) - 1) / 3.0,
        "note": "Negative camera Y means upward in the image, which is used here as the whiteboard-inward direction.",
    }
    (output_dir / "moving_object_pose_trajectory_xyzrpy.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def update_tracking_json(backup_dir: Path, output_dir: Path, tracking_rows: list[dict[str, object]], amplitude_mm: float, half_width_frames: float) -> None:
    path = backup_dir / "moving_object_trajectory.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["trajectory"] = tracking_rows
    data["manual_adjustment"] = {
        "name": "wipe_inward_at_one_third",
        "amplitude_y_mm": amplitude_mm,
        "half_width_frames": half_width_frames,
        "center_frame": (len(tracking_rows) - 1) / 3.0,
    }
    (output_dir / "moving_object_trajectory.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def rebuild_html(output_dir: Path) -> None:
    builder = WORKSPACE / "code" / "gemini_camera" / "build_task_pose_trajectory_web_visualization.py"
    subprocess.run(
        [
            "python",
            "-X",
            "utf8",
            str(builder),
            "--pose-json",
            str(output_dir / "moving_object_pose_trajectory_xyzrpy.json"),
            "--output-html",
            str(output_dir / "pose_trajectory_web_visualization.html"),
            "--title",
            "Whiteboard Wipe 6D Pose Trajectory",
            "--overlay-video",
            "\u64e6\u767d\u677f_\u53bb\u6c34\u5370_\u8fd0\u52a8\u7269\u4f53\u8f68\u8ff9_overlay.mp4",
        ],
        cwd=str(WORKSPACE),
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move the wipe trajectory inward near one third of the path.")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--amplitude-y-mm", type=float, default=-18.0)
    parser.add_argument("--half-width-frames", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    backup_dir = ensure_backup(output_dir)

    pose_csv = backup_dir / "moving_object_pose_trajectory_xyzrpy.csv"
    pose_rows_raw = read_csv(pose_csv)
    pose_rows = apply_pose_adjustment(pose_rows_raw, args.amplitude_y_mm, args.half_width_frames)
    write_csv(output_dir / "moving_object_pose_trajectory_xyzrpy.csv", pose_rows, list(pose_rows_raw[0].keys()))
    write_xyzrpy_only(output_dir, pose_rows)
    update_pose_json(backup_dir, output_dir, pose_rows, args.amplitude_y_mm, args.half_width_frames)

    tracking_csv = backup_dir / "moving_object_trajectory.csv"
    if tracking_csv.exists():
        tracking_rows_raw = read_csv(tracking_csv)
        tracking_rows = apply_tracking_adjustment(tracking_rows_raw, args.amplitude_y_mm, args.half_width_frames)
        write_csv(output_dir / "moving_object_trajectory.csv", tracking_rows, list(tracking_rows_raw[0].keys()))
        update_tracking_json(backup_dir, output_dir, tracking_rows, args.amplitude_y_mm, args.half_width_frames)

    rebuild_html(output_dir)
    center = int(round((len(pose_rows) - 1) / 3.0))
    sample = [
        {
            "frame": pose_rows[i]["frame"],
            "x_mm": pose_rows[i]["x_mm"],
            "y_mm": pose_rows[i]["y_mm"],
            "z_mm": pose_rows[i]["z_mm"],
        }
        for i in range(max(0, center - 2), min(len(pose_rows), center + 3))
    ]
    print(json.dumps({"backup_dir": str(backup_dir), "center_frame": center, "sample": sample}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
