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
REAL_VIDEO = (
    WORKSPACE
    / "Data"
    / "\u771f\u5b9e\u53cc\u81c2-\u5012\u6c34"
    / "\u5168\u5c40\u89c6\u89d2"
    / "\u5168\u5c40\u89c6\u89d2.mp4"
)
POUR_DIR = WORKSPACE / "Data" / "ai\u751f\u6210\u5012\u6c34" / "\u5012\u6c34"
TARGET_RGB_VIDEO = POUR_DIR / "\u5012\u6c34.mp4"
BASE_POSE_JSON = (
    POUR_DIR
    / "\u8fd0\u52a8\u7269\u4f53\u8f68\u8ff9_\u53bb\u6c34\u5370RGB_\u5206\u5c42\u6df1\u5ea6"
    / "moving_object_pose_trajectory_xyzrpy.json"
)
OUTPUT_DIR = POUR_DIR / "\u771f\u5b9e\u89c6\u9891\u8f68\u8ff9\u91cd\u5b9a\u5411_\u53bb\u6c34\u5370RGB_\u5206\u5c42\u6df1\u5ea6"


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
    fov_rad = math.radians(horizontal_fov_deg)
    fx = float(width) / (2.0 * math.tan(fov_rad / 2.0))
    fy = fx
    cx = float(width - 1) / 2.0
    cy = float(height - 1) / 2.0
    return fx, fy, cx, cy


def pixel_to_camera(u: float, v: float, z: float, fx: float, fy: float, cx: float, cy: float) -> tuple[float, float, float]:
    return float((u - cx) * z / fx), float((v - cy) * z / fy), float(z)


def camera_to_pixel(x: float, y: float, z: float, fx: float, fy: float, cx: float, cy: float) -> tuple[float, float]:
    return float(x * fx / z + cx), float(y * fy / z + cy)


def interpolate_missing(values: np.ndarray) -> np.ndarray:
    out = values.astype(np.float64).copy()
    n, dim = out.shape
    for j in range(dim):
        col = out[:, j]
        good = np.isfinite(col)
        if not np.any(good):
            col[:] = 0.0
        elif np.count_nonzero(good) == 1:
            col[:] = col[good][0]
        else:
            x = np.arange(n)
            col[~good] = np.interp(x[~good], x[good], col[good])
        out[:, j] = col
    return out


def resample(values: np.ndarray, out_count: int) -> np.ndarray:
    if len(values) == out_count:
        return values.copy()
    src = np.linspace(0.0, 1.0, len(values))
    dst = np.linspace(0.0, 1.0, out_count)
    out = np.zeros((out_count, values.shape[1]), dtype=np.float64)
    for j in range(values.shape[1]):
        out[:, j] = np.interp(dst, src, values[:, j])
    return out


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    result = np.empty_like(values, dtype=np.float64)
    half = window // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        result[i] = values[lo:hi].mean(axis=0)
    return result


def pca_angle(points: np.ndarray) -> float:
    if points.shape[0] < 8:
        return 0.0
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    if angle < -90.0:
        angle += 180.0
    if angle > 90.0:
        angle -= 180.0
    return float(angle)


def unwrap_degrees(values: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.unwrap(np.deg2rad(values), axis=0))


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    return ((values + 180.0) % 360.0) - 180.0


def track_real_tool_region(
    frames: list[np.ndarray],
    init_box: tuple[int, int, int, int],
    preview_dir: Path,
) -> tuple[list[dict], list[np.ndarray]]:
    gray0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    x, y, w, h = init_box
    mask = np.zeros_like(gray0)
    mask[y : y + h, x : x + w] = 255
    p0 = cv2.goodFeaturesToTrack(gray0, mask=mask, maxCorners=80, qualityLevel=0.01, minDistance=5, blockSize=5)
    if p0 is None or len(p0) < 8:
        raise RuntimeError("Cannot initialize real pouring tool track from the first frame.")

    rows: list[dict] = []
    previews: list[np.ndarray] = []
    prev_gray = gray0
    points = p0.astype(np.float32)
    preview_indices = {0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, len(frames) - 1}

    for idx, frame in enumerate(frames):
        if idx == 0:
            good = points.reshape(-1, 2)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                points,
                None,
                winSize=(25, 25),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            if next_points is None or status is None:
                good = np.empty((0, 2), dtype=np.float32)
            else:
                good = next_points[status.reshape(-1) == 1].reshape(-1, 2)
                height, width = gray.shape[:2]
                keep = (good[:, 0] > width * 0.22) & (good[:, 0] < width * 0.88) & (good[:, 1] > height * 0.05) & (good[:, 1] < height * 0.72)
                good = good[keep]
            if len(good) >= 8:
                points = good.reshape(-1, 1, 2).astype(np.float32)
                prev_gray = gray
            else:
                # Keep a missing marker; interpolation fills short gaps.
                rows.append({"frame": idx, "u_px": math.nan, "v_px": math.nan, "angle_deg": math.nan, "tracked_points": int(len(good))})
                prev_gray = gray
                continue

        u = float(np.median(good[:, 0]))
        v = float(np.median(good[:, 1]))
        angle = pca_angle(good.astype(np.float64))
        rows.append({"frame": idx, "u_px": u, "v_px": v, "angle_deg": angle, "tracked_points": int(len(good))})

        if idx in preview_indices:
            vis = frame.copy()
            for px, py in good.astype(int):
                cv2.circle(vis, (int(px), int(py)), 2, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(vis, (int(round(u)), int(round(v))), 7, (0, 0, 255), -1, cv2.LINE_AA)
            text = f"real frame {idx:03d}  u={u:.1f} v={v:.1f} points={len(good)}"
            cv2.putText(vis, text, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(vis, text, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            write_image(preview_dir / f"real_tracking_{idx:04d}.jpg", vis)
            previews.append(vis)

    return rows, previews


def draw_overlay(frame: np.ndarray, row: dict, history: list[tuple[int, int]]) -> np.ndarray:
    out = frame.copy()
    if len(history) >= 2:
        cv2.polylines(out, [np.array(history, dtype=np.int32).reshape((-1, 1, 2))], False, (0, 0, 255), 3, cv2.LINE_AA)

    u, v = int(round(row["u_px"])), int(round(row["v_px"]))
    rz = math.radians(float(row["rz_deg"]))
    axis_len = 70
    p0 = (int(round(u - math.cos(rz) * axis_len)), int(round(v - math.sin(rz) * axis_len)))
    p1 = (int(round(u + math.cos(rz) * axis_len)), int(round(v + math.sin(rz) * axis_len)))
    cv2.line(out, p0, p1, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.circle(out, (u, v), 8, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(out, (u, v), 5, (0, 0, 255), -1, cv2.LINE_AA)
    text = f"retarget frame {row['frame']:03d}  x={row['x_mm']:.1f} y={row['y_mm']:.1f} z={row['z_mm']:.1f}"
    cv2.putText(out, text, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(out, text, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def export_minimal_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
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
    parser = argparse.ArgumentParser(description="Retarget a real global-view pouring trajectory to the AI pouring clip.")
    parser.add_argument("--real-video", type=Path, default=REAL_VIDEO)
    parser.add_argument("--target-rgb-video", type=Path, default=TARGET_RGB_VIDEO)
    parser.add_argument("--base-pose-json", type=Path, default=BASE_POSE_JSON)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--horizontal-fov-deg", type=float, default=86.0)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--init-box", type=int, nargs=4, default=(340, 130, 90, 130), metavar=("X", "Y", "W", "H"))
    parser.add_argument("--map-mode", choices=("bbox_to_generated", "resolution_scale"), default="bbox_to_generated")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "preview_frames"
    preview_dir.mkdir(parents=True, exist_ok=True)

    real_frames, real_fps, (real_w, real_h) = read_video_frames(args.real_video)
    target_frames, target_fps, (target_w, target_h) = read_video_frames(args.target_rgb_video)
    base = json.loads(args.base_pose_json.read_text(encoding="utf-8"))
    base_rows = base["trajectory"]
    frame_count = min(len(target_frames), len(base_rows))
    target_frames = target_frames[:frame_count]
    base_rows = base_rows[:frame_count]

    track_rows, _ = track_real_tool_region(real_frames, tuple(args.init_box), preview_dir)
    raw_track = np.array([[r["u_px"], r["v_px"], r["angle_deg"]] for r in track_rows], dtype=np.float64)
    raw_track = interpolate_missing(raw_track)
    raw_track[:, 2] = unwrap_degrees(raw_track[:, 2:3]).reshape(-1)
    raw_track = smooth(raw_track, args.smooth_window)
    retarget_values = resample(raw_track, frame_count)
    retarget_values[:, 2] = wrap_degrees(retarget_values[:, 2])

    scaled_u = retarget_values[:, 0] * target_w / real_w
    scaled_v = retarget_values[:, 1] * target_h / real_h
    if args.map_mode == "bbox_to_generated":
        base_u = np.array([float(r["u_px"]) for r in base_rows], dtype=np.float64)
        base_v = np.array([float(r["v_px"]) for r in base_rows], dtype=np.float64)

        def fit_to_base(values: np.ndarray, base_values: np.ndarray) -> np.ndarray:
            lo = float(np.min(values))
            hi = float(np.max(values))
            if abs(hi - lo) < 1e-6:
                return np.full_like(values, float(np.median(base_values)))
            norm = (values - lo) / (hi - lo)
            return float(np.min(base_values)) + norm * (float(np.max(base_values)) - float(np.min(base_values)))

        target_u_values = fit_to_base(scaled_u, base_u)
        target_v_values = fit_to_base(scaled_v, base_v)
    else:
        target_u_values = scaled_u
        target_v_values = scaled_v

    z_base = float(np.median([float(r["z_mm"]) for r in base_rows]))
    fx, fy, cx, cy = compute_intrinsics(target_w, target_h, args.horizontal_fov_deg)
    rows: list[dict] = []
    history: list[tuple[int, int]] = []
    overlay_frames: list[np.ndarray] = []
    preview_indices = {0, frame_count // 4, frame_count // 2, 3 * frame_count // 4, frame_count - 1}

    for idx, (frame, base_row) in enumerate(zip(target_frames, base_rows)):
        real_u, real_v, real_angle = retarget_values[idx]
        target_u = float(target_u_values[idx])
        target_v = float(target_v_values[idx])
        x_mm, y_mm, z_mm = pixel_to_camera(target_u, target_v, z_base, fx, fy, cx, cy)
        rx_deg = float(base_row["rx_deg"])
        ry_deg = float(base_row["ry_deg"])
        rz_deg = float(real_angle)
        row = {
            "frame": idx,
            "time_s": idx / target_fps,
            "x_mm": x_mm,
            "y_mm": y_mm,
            "z_mm": z_mm,
            "rx_deg": rx_deg,
            "ry_deg": ry_deg,
            "rz_deg": rz_deg,
            "u_px": target_u,
            "v_px": target_v,
            "axis_x": float(math.cos(math.radians(rz_deg))),
            "axis_y": float(math.sin(math.radians(rz_deg))),
            "axis_z": 0.0,
            "source_real_u_px": float(real_u),
            "source_real_v_px": float(real_v),
            "source_real_angle_deg": float(real_angle),
            "source_real_frame_float": float(idx * (len(real_frames) - 1) / max(1, frame_count - 1)),
        }
        rows.append(row)
        history.append((int(round(target_u)), int(round(target_v))))
        overlay = draw_overlay(frame, row, history)
        overlay_frames.append(overlay)
        if idx in preview_indices:
            write_image(preview_dir / f"retarget_overlay_{idx:04d}.jpg", overlay)

    full_csv = out_dir / "retargeted_pose_trajectory_xyzrpy.csv"
    with full_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    minimal_csv = out_dir / "retargeted_pose_xyzrpy_only.csv"
    export_minimal_csv(rows, minimal_csv)

    real_track_csv = out_dir / "real_video_pour_tool_track_2d.csv"
    with real_track_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(track_rows[0].keys()))
        writer.writeheader()
        writer.writerows(track_rows)

    pose_json = out_dir / "retargeted_pose_trajectory_xyzrpy.json"
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_real_video": str(args.real_video),
        "target_rgb_video": str(args.target_rgb_video),
        "base_pose_json": str(args.base_pose_json),
        "output_dir": str(out_dir),
        "coordinate_frame": "camera frame: +X right, +Y down, +Z forward; positions are millimeters, rotations are degrees",
        "retarget_method": "Track the held bottle/end-effector region in the real global-view pouring video with Lucas-Kanade feature flow, resample it to the AI clip length, scale the 2D path to the target RGB resolution, and lift it to camera-space XYZ with stable real-depth scale from the generated RGB-D pose.",
        "map_mode": args.map_mode,
        "z_policy": "constant median object Z from the existing layered metric depth trajectory; this keeps the pouring bottle depth stable while it moves in image space.",
        "frame_count": frame_count,
        "target_fps": target_fps,
        "real_fps": real_fps,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "method": f"estimated_from_horizontal_fov_{args.horizontal_fov_deg:g}deg"},
        "trajectory": rows,
    }
    pose_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    shutil.copyfile(full_csv, out_dir / "moving_object_pose_trajectory_xyzrpy.csv")
    shutil.copyfile(minimal_csv, out_dir / "moving_object_pose_xyzrpy_only.csv")
    shutil.copyfile(pose_json, out_dir / "moving_object_pose_trajectory_xyzrpy.json")

    overlay_video = out_dir / "pour_real_retarget_overlay.mp4"
    write_video(overlay_frames, overlay_video, target_fps, WORKSPACE / "code" / "gemini_camera" / "tmp" / "retarget_pour")

    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "pose_csv": str(minimal_csv),
                "pose_json": str(pose_json),
                "real_track_csv": str(real_track_csv),
                "overlay_video": str(overlay_video),
                "frames": frame_count,
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
