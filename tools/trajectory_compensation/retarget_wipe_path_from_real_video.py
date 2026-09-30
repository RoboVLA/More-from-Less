from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
REAL_VIDEO = (
    WORKSPACE
    / "Data"
    / "\u771f\u5b9e\u53cc\u81c2-\u64e6\u9ed1\u677f-\u9636\u6bb5\u4e09\u53f3\u81c2\u79fb\u52a8\u9ed1\u677f\u64e6"
    / "\u5168\u5c40\u89c6\u89d2"
    / "\u5168\u5c40\u89c6\u89d2.mp4"
)
AI_WIPE_DIR = WORKSPACE / "Data" / "ai\u751f\u6210\u64e6\u767d\u677f" / "\u64e6\u767d\u677f"
TARGET_RGB_VIDEO = AI_WIPE_DIR / "\u53bb\u6c34\u5370" / "\u64e6\u767d\u677f_\u53bb\u6c34\u5370.mp4"
TARGET_DEPTH_DIR = AI_WIPE_DIR / "\u6df1\u5ea6\u89c6\u9891_\u64e6\u767d\u677f_\u5206\u5c42\u51e0\u4f55\u4e00\u81f4\u771f\u5b9e\u6df1\u5ea6\u7248"
BASE_POSE_JSON = AI_WIPE_DIR / "\u8fd0\u52a8\u7269\u4f53\u8f68\u8ff9_\u53bb\u6c34\u5370RGB_\u5206\u5c42\u6df1\u5ea6" / "moving_object_pose_trajectory_xyzrpy.json"
OUTPUT_DIR = AI_WIPE_DIR / "\u771f\u5b9e\u89c6\u9891\u8f68\u8ff9\u91cd\u5b9a\u5411_\u53bb\u6c34\u5370RGB_\u5206\u5c42\u6df1\u5ea6"


def read_image(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def natural_key(path: Path) -> tuple:
    parts: list[int | str] = []
    token = ""
    for ch in path.stem:
        if ch.isdigit():
            token += ch
        else:
            if token:
                parts.append(int(token))
                token = ""
            parts.append(ch)
    if token:
        parts.append(int(token))
    return tuple(parts)


def list_frames(directory: Path, suffixes: Iterable[str] = (".png",)) -> list[Path]:
    return sorted((p for p in directory.iterdir() if p.suffix.lower() in suffixes), key=natural_key)


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


def compute_intrinsics(
    width: int,
    height: int,
    fx: float | None,
    fy: float | None,
    cx: float | None,
    cy: float | None,
    horizontal_fov_deg: float,
) -> tuple[float, float, float, float, str]:
    cx_value = float(width - 1) / 2.0 if cx is None else float(cx)
    cy_value = float(height - 1) / 2.0 if cy is None else float(cy)
    if fx is not None and fy is not None:
        return float(fx), float(fy), cx_value, cy_value, "user_fx_fy"
    fov_rad = math.radians(horizontal_fov_deg)
    fx_value = float(width) / (2.0 * math.tan(fov_rad / 2.0))
    fy_value = fx_value if fy is None else float(fy)
    if fx is not None:
        fx_value = float(fx)
    return fx_value, fy_value, cx_value, cy_value, f"estimated_from_horizontal_fov_{horizontal_fov_deg:g}deg"


def pixel_to_camera(u: float, v: float, z: float, fx: float, fy: float, cx: float, cy: float) -> tuple[float, float, float]:
    return float((u - cx) * z / fx), float((v - cy) * z / fy), float(z)


def eraser_mask(frame: np.ndarray, roi: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = (((hue < 13) | (hue > 168)) & (sat > 55) & (val > 35) & (roi > 0)).astype(np.uint8) * 255
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((red > 0).astype(np.uint8), 8)
    best_label = 0
    best_score = -1.0
    h, w = frame.shape[:2]
    for label in range(1, num):
        x, y, cw, ch, area = stats[label]
        if int(area) < 30 or int(area) > 9000:
            continue
        if cw < 5 or ch < 5:
            continue
        cx = x + cw / 2
        cy = y + ch / 2
        board_score = 1.0 - min(1.0, abs(cy - h * 0.56) / (h * 0.30))
        center_score = 1.0 - min(1.0, abs(cx - w * 0.58) / (w * 0.30))
        score = float(area) * (0.80 + 0.12 * board_score + 0.08 * center_score)
        if score > best_score:
            best_score = score
            best_label = label

    if best_label == 0:
        return np.zeros(frame.shape[:2], dtype=np.uint8)

    core = (labels == best_label).astype(np.uint8) * 255
    support = cv2.dilate(core, np.ones((25, 25), np.uint8), iterations=1)
    dark = ((val < 88) & (roi > 0)).astype(np.uint8) * 255
    dark_near = cv2.bitwise_and(dark, support)
    mask = cv2.bitwise_or(core, dark_near)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    pts = cv2.findNonZero(mask)
    if pts is not None and len(pts) >= 8:
        hull = cv2.convexHull(pts)
        filled = np.zeros_like(mask)
        cv2.fillConvexPoly(filled, hull, 255)
        mask = cv2.bitwise_or(mask, filled)
    return cv2.GaussianBlur(mask, (9, 9), 0)


def centroid_and_angle(mask: np.ndarray) -> tuple[float, float, float, int]:
    ys, xs = np.nonzero(mask > 16)
    area = int(xs.size)
    if area == 0:
        return math.nan, math.nan, math.nan, 0
    weights = mask[ys, xs].astype(np.float64)
    u = float(np.average(xs, weights=weights))
    v = float(np.average(ys, weights=weights))
    angle = 0.0
    points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    if points.shape[0] >= 8:
        centered = points - points.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
        angle = math.degrees(math.atan2(axis[1], axis[0]))
        if angle < -90:
            angle += 180
        if angle > 90:
            angle -= 180
    return u, v, angle, area


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
    src_x = np.linspace(0.0, 1.0, len(values))
    dst_x = np.linspace(0.0, 1.0, out_count)
    out = np.zeros((out_count, values.shape[1]), dtype=np.float64)
    for j in range(values.shape[1]):
        out[:, j] = np.interp(dst_x, src_x, values[:, j])
    return out


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    result = values.copy().astype(np.float64)
    half = window // 2
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        result[i] = np.mean(values[lo:hi], axis=0)
    return result


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
    cv2.circle(out, (u, v), 7, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(out, (u, v), 4, (0, 0, 255), -1, cv2.LINE_AA)
    text = f"retarget frame {row['frame']:03d}  x={row['x_mm']:.1f} y={row['y_mm']:.1f} z={row['z_mm']:.1f}"
    cv2.putText(out, text, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(out, text, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def export_minimal_csv(rows: list[dict], path: Path) -> None:
    fields = ["frame", "time_s", "x", "y", "z", "rx", "ry", "rz"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
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
    parser = argparse.ArgumentParser(description="Retarget the AI wipe 6D path from a real global-view eraser video.")
    parser.add_argument("--real-video", type=Path, default=REAL_VIDEO)
    parser.add_argument("--target-rgb-video", type=Path, default=TARGET_RGB_VIDEO)
    parser.add_argument("--target-depth-dir", type=Path, default=TARGET_DEPTH_DIR)
    parser.add_argument("--base-pose-json", type=Path, default=BASE_POSE_JSON)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--horizontal-fov-deg", type=float, default=86.0)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--smooth-window", type=int, default=5)
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

    roi = np.zeros((real_h, real_w), dtype=np.uint8)
    roi[int(real_h * 0.34) : int(real_h * 0.75), int(real_w * 0.30) : int(real_w * 0.78)] = 255
    track_rows: list[dict] = []
    preview_real_indices = {0, len(real_frames) // 4, len(real_frames) // 2, 3 * len(real_frames) // 4, len(real_frames) - 1}
    for idx, frame in enumerate(real_frames):
        mask = eraser_mask(frame, roi)
        u, v, angle, area = centroid_and_angle(mask)
        track_rows.append({"frame": idx, "time_s": idx / real_fps, "u_px": u, "v_px": v, "angle_deg": angle, "area_px": area})
        if idx in preview_real_indices:
            overlay = frame.copy()
            overlay[mask > 16] = (0, 80, 255)
            vis = cv2.addWeighted(overlay, 0.38, frame, 0.62, 0)
            if np.isfinite(u) and np.isfinite(v):
                cv2.circle(vis, (int(round(u)), int(round(v))), 6, (0, 0, 255), -1, cv2.LINE_AA)
            write_image(preview_dir / f"real_tracking_{idx:04d}.jpg", vis)

    real_values = interpolate_missing(np.array([[r["u_px"], r["v_px"], r["angle_deg"]] for r in track_rows], dtype=np.float64))
    real_values = smooth(real_values, args.smooth_window)
    retarget_values = resample(real_values, frame_count)

    depth_paths = list_frames(args.target_depth_dir / "depth_mm_frames")
    depths = [read_image(p, cv2.IMREAD_UNCHANGED) for p in depth_paths[:frame_count]]
    z_base = float(np.median([float(r["z_mm"]) for r in base_rows]))
    rx_base = float(np.median([float(r["rx_deg"]) for r in base_rows]))
    ry_base = float(np.median([float(r["ry_deg"]) for r in base_rows]))
    fx, fy, cx, cy, intrinsics_method = compute_intrinsics(target_w, target_h, args.fx, args.fy, args.cx, args.cy, args.horizontal_fov_deg)

    rows: list[dict] = []
    history: list[tuple[int, int]] = []
    overlay_frames: list[np.ndarray] = []
    for idx, (frame, base_row) in enumerate(zip(target_frames, base_rows)):
        real_u, real_v, real_angle = retarget_values[idx]
        target_u = float(real_u * target_w / real_w)
        target_v = float(real_v * target_h / real_h)

        depth = depths[idx] if idx < len(depths) else None
        z_mm = z_base
        if depth is not None:
            u_i, v_i = int(round(target_u)), int(round(target_v))
            radius = 18
            x0, x1 = max(0, u_i - radius), min(depth.shape[1], u_i + radius + 1)
            y0, y1 = max(0, v_i - radius), min(depth.shape[0], v_i + radius + 1)
            vals = depth[y0:y1, x0:x1]
            vals = vals[vals > 0]
            if vals.size >= 20:
                local_z = float(np.median(vals))
                if abs(local_z - z_base) < 12:
                    z_mm = local_z

        x_mm, y_mm, z_mm = pixel_to_camera(target_u, target_v, z_mm, fx, fy, cx, cy)
        rz_deg = float(real_angle)
        row = {
            "frame": idx,
            "time_s": idx / target_fps,
            "x_mm": x_mm,
            "y_mm": y_mm,
            "z_mm": z_mm,
            "rx_deg": rx_base,
            "ry_deg": ry_base,
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
        if idx in {0, frame_count // 4, frame_count // 2, 3 * frame_count // 4, frame_count - 1}:
            write_image(preview_dir / f"retarget_overlay_{idx:04d}.jpg", overlay)

    full_csv = out_dir / "retargeted_pose_trajectory_xyzrpy.csv"
    with full_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    minimal_csv = out_dir / "retargeted_pose_xyzrpy_only.csv"
    export_minimal_csv(rows, minimal_csv)

    real_track_csv = out_dir / "real_video_eraser_track_2d.csv"
    with real_track_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(track_rows[0].keys()))
        writer.writeheader()
        writer.writerows(track_rows)

    overlay_video = out_dir / "擦白板_真实轨迹重定向_overlay.mp4"
    write_video(overlay_frames, overlay_video, target_fps, WORKSPACE / "code" / "gemini_camera" / "tmp" / "retarget_wipe")

    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_real_video": str(args.real_video),
        "target_rgb_video": str(args.target_rgb_video),
        "target_depth_dir": str(args.target_depth_dir),
        "base_pose_json": str(args.base_pose_json),
        "output_dir": str(out_dir),
        "coordinate_frame": "camera frame: +X right, +Y down, +Z forward; positions are millimeters, rotations are degrees",
        "retarget_method": "Detect red/black eraser in real global-view video, resample its 2D center/axis to the AI wipe clip length, scale pixels by target/source resolution, then lift to camera-space XYZ using the target depth layer and estimated RGB intrinsics.",
        "frame_count": frame_count,
        "target_fps": target_fps,
        "real_fps": real_fps,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "method": intrinsics_method},
        "trajectory": rows,
    }
    pose_json = out_dir / "retargeted_pose_trajectory_xyzrpy.json"
    pose_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(
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
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
