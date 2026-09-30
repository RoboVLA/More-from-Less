from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
POUR_DIR = WORKSPACE / "Data" / "ai\u751f\u6210\u5012\u6c34" / "\u5012\u6c34"
GENERATED_JSON = (
    POUR_DIR
    / "\u8fd0\u52a8\u7269\u4f53\u8f68\u8ff9_\u53bb\u6c34\u5370RGB_\u5206\u5c42\u6df1\u5ea6"
    / "moving_object_pose_trajectory_xyzrpy.json"
)
REAL_RETARGET_JSON = (
    POUR_DIR
    / "\u771f\u5b9e\u89c6\u9891\u8f68\u8ff9\u91cd\u5b9a\u5411_\u53bb\u6c34\u5370RGB_\u5206\u5c42\u6df1\u5ea6"
    / "retargeted_pose_trajectory_xyzrpy.json"
)
TARGET_VIDEO = POUR_DIR / "\u5012\u6c34.mp4"
OUTPUT_DIR = POUR_DIR / "\u878d\u5408\u771f\u5b9e\u4e0e\u751f\u6210\u8f68\u8ff9_\u53bb\u6c34\u5370RGB_\u5206\u5c42\u6df1\u5ea6"


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def read_video_frames(video_path: Path) -> tuple[list[np.ndarray], float, tuple[int, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames in video: {video_path}")
    return frames, fps, (width, height)


def write_video(frames: list[np.ndarray], out_path: Path, fps: float, tmp_dir: Path) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{time.perf_counter_ns()}"
    frame_dir = tmp_dir / f"{out_path.stem}_{run_id}_png"
    temp_path = tmp_dir / f"{out_path.stem}_{run_id}.mp4"
    frame_dir.mkdir(parents=True, exist_ok=True)
    try:
        for idx, frame in enumerate(frames):
            write_image(frame_dir / f"frame_{idx:04d}.png", frame)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                f"{fps:g}",
                "-i",
                str(frame_dir / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "14",
                "-pix_fmt",
                "yuv420p",
                str(temp_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(temp_path, out_path)
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


def compute_intrinsics(width: int, height: int, horizontal_fov_deg: float) -> tuple[float, float, float, float]:
    fov = math.radians(horizontal_fov_deg)
    fx = float(width) / (2.0 * math.tan(fov / 2.0))
    fy = fx
    cx = float(width - 1) / 2.0
    cy = float(height - 1) / 2.0
    return fx, fy, cx, cy


def camera_to_pixel(x: float, y: float, z: float, fx: float, fy: float, cx: float, cy: float) -> tuple[float, float]:
    return float(x * fx / z + cx), float(y * fy / z + cy)


def rows_to_array(rows: list[dict], fields: list[str]) -> np.ndarray:
    return np.array([[float(row[field]) for field in fields] for row in rows], dtype=np.float64)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    half = window // 2
    out = np.empty_like(values, dtype=np.float64)
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out[i] = values[lo:hi].mean(axis=0)
    return out


def unwrap_degrees(values: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.unwrap(np.deg2rad(values), axis=0))


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    return ((values + 180.0) % 360.0) - 180.0


def blend_weight_profile(count: int, base_real_weight: float, mid_real_bonus: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, count)
    w = base_real_weight + mid_real_bonus * np.sin(np.pi * t)
    return np.clip(w, 0.0, 1.0)


def draw_polyline(frame: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int], thickness: int) -> None:
    if len(points) >= 2:
        cv2.polylines(frame, [np.array(points, dtype=np.int32).reshape((-1, 1, 2))], False, color, thickness, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    blended_row: dict,
    generated_history: list[tuple[int, int]],
    real_history: list[tuple[int, int]],
    blended_history: list[tuple[int, int]],
) -> np.ndarray:
    out = frame.copy()
    draw_polyline(out, generated_history, (255, 100, 30), 2)
    draw_polyline(out, real_history, (30, 170, 30), 2)
    draw_polyline(out, blended_history, (0, 0, 255), 4)

    u, v = int(round(blended_row["u_px"])), int(round(blended_row["v_px"]))
    rz = math.radians(float(blended_row["rz_deg"]))
    length = 70
    p0 = (int(round(u - math.cos(rz) * length)), int(round(v - math.sin(rz) * length)))
    p1 = (int(round(u + math.cos(rz) * length)), int(round(v + math.sin(rz) * length)))
    cv2.line(out, p0, p1, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.circle(out, (u, v), 8, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(out, (u, v), 5, (0, 0, 255), -1, cv2.LINE_AA)

    text = f"blend frame {blended_row['frame']:03d}  x={blended_row['x_mm']:.1f} y={blended_row['y_mm']:.1f} z={blended_row['z_mm']:.1f}"
    cv2.putText(out, text, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(out, text, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "blue=generated  green=real-retarget  red=fused", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(out, "blue=generated  green=real-retarget  red=fused", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def export_csv(rows: list[dict], full_csv: Path, minimal_csv: Path) -> None:
    with full_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with minimal_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "time_s", "x", "y", "z", "rx", "ry", "rz"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend generated-video and real-video-retargeted pouring trajectories.")
    parser.add_argument("--generated-json", type=Path, default=GENERATED_JSON)
    parser.add_argument("--real-retarget-json", type=Path, default=REAL_RETARGET_JSON)
    parser.add_argument("--target-video", type=Path, default=TARGET_VIDEO)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--real-weight", type=float, default=0.55)
    parser.add_argument("--mid-real-bonus", type=float, default=0.10)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--horizontal-fov-deg", type=float, default=86.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "preview_frames"
    preview_dir.mkdir(parents=True, exist_ok=True)

    generated = json.loads(args.generated_json.read_text(encoding="utf-8"))
    real = json.loads(args.real_retarget_json.read_text(encoding="utf-8"))
    generated_rows = generated["trajectory"]
    real_rows = real["trajectory"]
    frames, fps, (width, height) = read_video_frames(args.target_video)
    count = min(len(frames), len(generated_rows), len(real_rows))
    frames = frames[:count]
    generated_rows = generated_rows[:count]
    real_rows = real_rows[:count]

    gen_xyz = rows_to_array(generated_rows, ["x_mm", "y_mm", "z_mm"])
    real_xyz = rows_to_array(real_rows, ["x_mm", "y_mm", "z_mm"])
    gen_rpy = unwrap_degrees(rows_to_array(generated_rows, ["rx_deg", "ry_deg", "rz_deg"]))
    real_rpy = unwrap_degrees(rows_to_array(real_rows, ["rx_deg", "ry_deg", "rz_deg"]))

    real_weight = blend_weight_profile(count, args.real_weight, args.mid_real_bonus)[:, None]
    gen_weight = 1.0 - real_weight
    blended_xyz = gen_xyz * gen_weight + real_xyz * real_weight
    blended_rpy = gen_rpy * gen_weight + real_rpy * real_weight
    blended_xyz = smooth(blended_xyz, args.smooth_window)
    blended_rpy = wrap_degrees(smooth(blended_rpy, args.smooth_window))

    stable_z = float(np.median(np.concatenate([gen_xyz[:, 2], real_xyz[:, 2]])))
    blended_xyz[:, 2] = stable_z

    fx, fy, cx, cy = compute_intrinsics(width, height, args.horizontal_fov_deg)
    rows: list[dict] = []
    gen_history: list[tuple[int, int]] = []
    real_history: list[tuple[int, int]] = []
    blend_history: list[tuple[int, int]] = []
    overlay_frames: list[np.ndarray] = []
    preview_indices = {0, count // 4, count // 2, 3 * count // 4, count - 1}

    for idx, frame in enumerate(frames):
        x, y, z = blended_xyz[idx]
        rx, ry, rz = blended_rpy[idx]
        u, v = camera_to_pixel(x, y, z, fx, fy, cx, cy)
        gen_u, gen_v = camera_to_pixel(gen_xyz[idx, 0], gen_xyz[idx, 1], gen_xyz[idx, 2], fx, fy, cx, cy)
        real_u, real_v = camera_to_pixel(real_xyz[idx, 0], real_xyz[idx, 1], real_xyz[idx, 2], fx, fy, cx, cy)
        gen_history.append((int(round(gen_u)), int(round(gen_v))))
        real_history.append((int(round(real_u)), int(round(real_v))))
        blend_history.append((int(round(u)), int(round(v))))

        row = {
            "frame": idx,
            "time_s": idx / fps,
            "x_mm": float(x),
            "y_mm": float(y),
            "z_mm": float(z),
            "rx_deg": float(rx),
            "ry_deg": float(ry),
            "rz_deg": float(rz),
            "u_px": float(u),
            "v_px": float(v),
            "axis_x": float(math.cos(math.radians(rz))),
            "axis_y": float(math.sin(math.radians(rz))),
            "axis_z": 0.0,
            "real_weight": float(real_weight[idx, 0]),
            "generated_weight": float(gen_weight[idx, 0]),
            "generated_x_mm": float(gen_xyz[idx, 0]),
            "generated_y_mm": float(gen_xyz[idx, 1]),
            "generated_z_mm": float(gen_xyz[idx, 2]),
            "real_retarget_x_mm": float(real_xyz[idx, 0]),
            "real_retarget_y_mm": float(real_xyz[idx, 1]),
            "real_retarget_z_mm": float(real_xyz[idx, 2]),
        }
        rows.append(row)
        overlay = draw_overlay(frame, row, gen_history, real_history, blend_history)
        overlay_frames.append(overlay)
        if idx in preview_indices:
            write_image(preview_dir / f"fused_overlay_{idx:04d}.jpg", overlay)

    full_csv = out_dir / "fused_pose_trajectory_xyzrpy.csv"
    minimal_csv = out_dir / "fused_pose_xyzrpy_only.csv"
    export_csv(rows, full_csv, minimal_csv)

    pose_json = out_dir / "fused_pose_trajectory_xyzrpy.json"
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generated_pose_json": str(args.generated_json),
        "real_retarget_pose_json": str(args.real_retarget_json),
        "target_video": str(args.target_video),
        "output_dir": str(out_dir),
        "coordinate_frame": "camera frame: +X right, +Y down, +Z forward; positions are millimeters and rotations are degrees",
        "blend_method": "Time-varying weighted average: generated-video trajectory preserves target-scene placement, real-video retargeted trajectory contributes real motion shape. Z is fixed to a stable metric-depth median for pouring.",
        "real_weight_base": float(args.real_weight),
        "mid_real_bonus": float(args.mid_real_bonus),
        "smooth_window": int(args.smooth_window),
        "stable_z_mm": stable_z,
        "frame_count": count,
        "fps": fps,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "method": f"estimated_from_horizontal_fov_{args.horizontal_fov_deg:g}deg"},
        "trajectory": rows,
    }
    pose_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    shutil.copyfile(full_csv, out_dir / "moving_object_pose_trajectory_xyzrpy.csv")
    shutil.copyfile(minimal_csv, out_dir / "moving_object_pose_xyzrpy_only.csv")
    shutil.copyfile(pose_json, out_dir / "moving_object_pose_trajectory_xyzrpy.json")

    overlay_video = out_dir / "pour_real_generated_fused_overlay.mp4"
    write_video(overlay_frames, overlay_video, fps, WORKSPACE / "code" / "gemini_camera" / "tmp" / "blend_pour")

    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "pose_csv": str(minimal_csv),
                "pose_json": str(pose_json),
                "overlay_video": str(overlay_video),
                "frames": count,
                "real_weight_range": [float(real_weight.min()), float(real_weight.max())],
                "x_start_end": [rows[0]["x_mm"], rows[-1]["x_mm"]],
                "y_start_end": [rows[0]["y_mm"], rows[-1]["y_mm"]],
                "z_min_max": [float(min(r["z_mm"] for r in rows)), float(max(r["z_mm"] for r in rows))],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
