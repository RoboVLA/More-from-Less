from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
WIPE_DIR = WORKSPACE / "Data" / "ai\u751f\u6210\u64e6\u767d\u677f" / "\u64e6\u767d\u677f"
DEFAULT_VIDEO = WIPE_DIR / "\u53bb\u6c34\u5370" / "\u64e6\u767d\u677f_\u53bb\u6c34\u5370.mp4"
DEFAULT_REAL_DEPTH = WORKSPACE / "\u6df1\u5ea6\u56fe\u76f8\u5173" / "openni_depth_20260609_043616_000_mm_u16.png"
DEFAULT_REAL_RGB = WORKSPACE / "\u6df1\u5ea6\u56fe\u76f8\u5173" / "orbbec_rgb_20260609_043637.jpg"
DEFAULT_OUTPUT_DIR = WIPE_DIR / "\u6df1\u5ea6\u89c6\u9891_\u64e6\u767d\u677f_\u5206\u5c42\u51e0\u4f55\u4e00\u81f4\u771f\u5b9e\u6df1\u5ea6\u7248"


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
    if not frames:
        raise RuntimeError("No frames to write.")
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


def colorize_depth(depth_mm: np.ndarray, near_mm: float, far_mm: float) -> np.ndarray:
    clipped = np.clip(depth_mm.astype(np.float32), near_mm, far_mm)
    norm = (far_mm - clipped) / max(1.0, far_mm - near_mm)
    return cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
    divider = np.full((left.shape[0], 8, 3), 255, dtype=np.uint8)
    return np.hstack([left, divider, right])


def overlay_alpha(frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    overlay[alpha > 16] = (0, 80, 255)
    return cv2.addWeighted(overlay, 0.38, frame, 0.62, 0)


def inpaint_depth(depth_mm: np.ndarray, mask: np.ndarray) -> np.ndarray:
    depth_f = depth_mm.astype(np.float32)
    repair = ((depth_mm == 0) | (mask > 0)).astype(np.uint8) * 255
    if not np.any(repair):
        return depth_mm
    valid = depth_f[depth_mm > 0]
    fill_value = float(np.median(valid)) if valid.size else 600.0
    norm = depth_f.copy()
    norm[depth_mm == 0] = fill_value
    lo, hi = np.percentile(norm[norm > 0], [2, 98])
    scaled = np.clip((norm - lo) * 255.0 / max(1.0, hi - lo), 0, 255).astype(np.uint8)
    repaired = cv2.inpaint(scaled, repair, 7, cv2.INPAINT_TELEA).astype(np.float32)
    restored = repaired * max(1.0, hi - lo) / 255.0 + lo
    keep = repair > 0
    out = depth_f.copy()
    out[keep] = restored[keep]
    return np.clip(np.rint(out), 0, 65535).astype(np.uint16)


def old_real_eraser_mask(real_rgb: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    h, w = real_rgb.shape[:2]
    hsv = cv2.cvtColor(real_rgb, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    board_roi = np.zeros((h, w), dtype=np.uint8)
    board_roi[int(h * 0.34) : int(h * 0.62), int(w * 0.30) : int(w * 0.78)] = 255
    blue_purple = (((hue > 105) & (hue < 155) & (sat > 45) & (val > 35)) & (board_roi > 0)).astype(np.uint8) * 255
    dark_near = ((val < 70) & (board_roi > 0)).astype(np.uint8) * 255
    blue = cv2.morphologyEx(blue_purple, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    near = cv2.bitwise_and(dark_near, cv2.dilate(blue, np.ones((35, 35), np.uint8), iterations=1))
    mask = cv2.bitwise_or(blue, near)
    mask = cv2.dilate(mask, np.ones((13, 13), np.uint8), iterations=1)
    return cv2.resize(mask, target_shape, interpolation=cv2.INTER_NEAREST)


def build_background_depth(real_depth: np.ndarray, real_rgb: np.ndarray, target_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    old_mask = old_real_eraser_mask(real_rgb, (real_depth.shape[1], real_depth.shape[0]))
    repaired = inpaint_depth(real_depth, old_mask)
    background = cv2.resize(repaired, target_size, interpolation=cv2.INTER_LINEAR).astype(np.uint16)
    old_mask_resized = cv2.resize(old_mask, target_size, interpolation=cv2.INTER_NEAREST)
    background = inpaint_depth(background, old_mask_resized)
    return background, old_mask_resized


def eraser_alpha(frame: np.ndarray, background_rgb: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    diff = cv2.cvtColor(cv2.absdiff(frame, background_rgb), cv2.COLOR_BGR2GRAY)

    roi = np.zeros((h, w), dtype=np.uint8)
    roi[int(h * 0.31) : int(h * 0.70), int(w * 0.28) : int(w * 0.80)] = 255
    red = (((hue < 12) | (hue > 168)) & (sat > 55) & (val > 45) & (roi > 0)).astype(np.uint8) * 255
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((red > 0).astype(np.uint8), 8)
    if num <= 1:
        changed = ((diff > 22) & (roi > 0) & (val < 150)).astype(np.uint8) * 255
        changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
        num, labels, stats, _ = cv2.connectedComponentsWithStats((changed > 0).astype(np.uint8), 8)
        source = changed
    else:
        source = red

    best_label = 0
    best_score = -1.0
    for label in range(1, num):
        x, y, cw, ch, area = stats[label]
        if area < 60 or area > 25000:
            continue
        if cw < 8 or ch < 8:
            continue
        aspect = max(cw / max(1, ch), ch / max(1, cw))
        if aspect > 5.0:
            continue
        cx = x + cw / 2.0
        cy = y + ch / 2.0
        board_score = 1.0 - min(1.0, abs(cy - h * 0.47) / (h * 0.28))
        center_score = 1.0 - min(1.0, abs(cx - w * 0.52) / (w * 0.33))
        score = float(area) * (0.8 + 0.12 * board_score + 0.08 * center_score)
        if score > best_score:
            best_score = score
            best_label = label

    if best_label == 0:
        return np.zeros((h, w), dtype=np.uint8)

    comp = (labels == best_label).astype(np.uint8) * 255
    comp = cv2.dilate(comp, np.ones((17, 17), np.uint8), iterations=1)
    dark = ((val < 85) & (roi > 0)).astype(np.uint8) * 255
    dark_near = cv2.bitwise_and(dark, comp)
    alpha = cv2.bitwise_or(source, dark_near)
    alpha = cv2.bitwise_and(alpha, cv2.dilate(comp, np.ones((9, 9), np.uint8), iterations=1))

    pts = cv2.findNonZero(alpha)
    if pts is not None and len(pts) >= 8:
        hull = cv2.convexHull(pts)
        filled = np.zeros_like(alpha)
        cv2.fillConvexPoly(filled, hull, 255)
        alpha = cv2.bitwise_or(alpha, filled)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
    alpha = cv2.dilate(alpha, np.ones((5, 5), np.uint8), iterations=1)
    return cv2.GaussianBlur(alpha, (13, 13), 0)


def eraser_depth_layer(alpha: np.ndarray, background_depth: np.ndarray, thickness_mm: float, relief_mm: float) -> np.ndarray:
    depth = background_depth.astype(np.float32)
    ys, xs = np.nonzero(alpha > 16)
    if xs.size < 20:
        return depth
    pad = 18
    x0, x1 = max(0, xs.min() - pad), min(depth.shape[1], xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(depth.shape[0], ys.max() + pad + 1)
    local = background_depth[y0:y1, x0:x1]
    valid = local[local > 0]
    board_depth = float(np.median(valid)) if valid.size else float(np.median(background_depth[background_depth > 0]))

    points = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    short_axis = eigvecs[:, int(np.argmin(eigvals))]
    short_coord = centered @ short_axis
    radius = max(1.0, float(np.percentile(np.abs(short_coord), 95)))
    grid_x, grid_y = np.meshgrid(np.arange(depth.shape[1], dtype=np.float32), np.arange(depth.shape[0], dtype=np.float32))
    local_short = (grid_x - center[0]) * short_axis[0] + (grid_y - center[1]) * short_axis[1]
    edge = np.clip(np.abs(local_short) / radius, 0.0, 1.0)
    relief = relief_mm * (edge * edge - 0.35)
    return (board_depth - thickness_mm + relief).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate layered metric depth for the AI whiteboard-wiping video.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--real-depth", type=Path, default=DEFAULT_REAL_DEPTH)
    parser.add_argument("--real-rgb", type=Path, default=DEFAULT_REAL_RGB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eraser-thickness-mm", type=float, default=24.0)
    parser.add_argument("--eraser-relief-mm", type=float, default=10.0)
    parser.add_argument("--near-mm", type=float, default=300.0)
    parser.add_argument("--far-mm", type=float, default=850.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames, fps, (width, height) = read_video_frames(args.video)
    real_depth = read_image(args.real_depth, cv2.IMREAD_UNCHANGED)
    real_rgb = read_image(args.real_rgb, cv2.IMREAD_COLOR)
    background_depth, real_eraser_mask = build_background_depth(real_depth, real_rgb, (width, height))
    background_rgb = np.median(np.stack(frames, axis=0).astype(np.uint8), axis=0).astype(np.uint8)

    out_dir: Path = args.out_dir
    depth_dir = out_dir / "depth_mm_frames"
    color_dir = out_dir / "depth_metric_color_frames"
    alpha_dir = out_dir / "moving_layer_alpha_frames"
    preview_dir = out_dir / "preview_frames"
    for directory in [depth_dir, color_dir, alpha_dir, preview_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    color_frames: list[np.ndarray] = []
    compare_frames: list[np.ndarray] = []
    alpha_areas: list[int] = []
    z_values: list[float] = []
    preview_indices = {0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, len(frames) - 1}

    write_image(preview_dir / "static_background_depth_color.png", colorize_depth(background_depth, args.near_mm, args.far_mm))
    write_image(preview_dir / "real_eraser_removed_mask.png", real_eraser_mask)

    for idx, frame in enumerate(frames):
        alpha = eraser_alpha(frame, background_rgb)
        alpha_f = np.clip(alpha.astype(np.float32) / 255.0, 0.0, 1.0)
        obj_depth = eraser_depth_layer(alpha, background_depth, args.eraser_thickness_mm, args.eraser_relief_mm)
        final_depth = background_depth.astype(np.float32) * (1.0 - alpha_f) + obj_depth * alpha_f
        final_depth_u16 = np.clip(np.rint(final_depth), 0, 65535).astype(np.uint16)
        write_image(depth_dir / f"depth_{idx:04d}_mm_u16.png", final_depth_u16)
        write_image(alpha_dir / f"moving_layer_alpha_{idx:04d}.png", alpha)
        color = colorize_depth(final_depth_u16, args.near_mm, args.far_mm)
        write_image(color_dir / f"depth_metric_color_{idx:04d}.png", color)
        color_frames.append(color)
        compare_frames.append(side_by_side(frame, color))
        area = int(np.count_nonzero(alpha > 16))
        alpha_areas.append(area)
        vals = final_depth_u16[(alpha > 16) & (final_depth_u16 > 0)]
        if vals.size:
            z_values.append(float(np.median(vals)))
        if idx in preview_indices:
            write_image(preview_dir / f"rgb_depth_layered_geometry_compare_{idx:04d}.jpg", side_by_side(frame, color))
            write_image(preview_dir / f"moving_layer_alpha_overlay_{idx:04d}.jpg", overlay_alpha(frame, alpha))

    tmp_dir = WORKSPACE / "code" / "gemini_camera" / "tmp" / "ai_wipe_layered_geometry"
    color_video = out_dir / "\u64e6\u767d\u677f_depth_metric_layered_geometry_color_video.mp4"
    compare_video = out_dir / "\u64e6\u767d\u677f_rgb_depth_layered_geometry_compare.mp4"
    write_video(color_frames, color_video, fps, tmp_dir)
    write_video(compare_frames, compare_video, fps, tmp_dir)

    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_rgb_video": str(args.video),
        "source_real_depth": str(args.real_depth),
        "source_real_rgb": str(args.real_rgb),
        "output_dir": str(out_dir),
        "output_depth_color_video": str(color_video),
        "output_rgb_depth_compare_video": str(compare_video),
        "output_depth_mm_frames": str(depth_dir),
        "output_moving_layer_alpha_frames": str(alpha_dir),
        "frame_count": len(frames),
        "fps": fps,
        "width": width,
        "height": height,
        "eraser_thickness_mm": args.eraser_thickness_mm,
        "eraser_relief_mm": args.eraser_relief_mm,
        "moving_layer_alpha_area_px_min_median_max": [
            int(np.min(alpha_areas)) if alpha_areas else 0,
            int(np.median(alpha_areas)) if alpha_areas else 0,
            int(np.max(alpha_areas)) if alpha_areas else 0,
        ],
        "moving_eraser_depth_mm_min_median_max": [
            float(np.min(z_values)) if z_values else None,
            float(np.median(z_values)) if z_values else None,
            float(np.max(z_values)) if z_values else None,
        ],
        "method": "Static Gemini real depth is repaired to remove the captured blue eraser, then the AI red/black eraser is detected as a moving layer and composited near the whiteboard plane with fixed thickness.",
    }
    (out_dir / "depth_video_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "output_dir": str(out_dir),
            "depth_mm_frames": str(depth_dir),
            "alpha_frames": str(alpha_dir),
            "compare_video": str(compare_video),
            "frames": len(frames),
            "alpha_area_min_median_max": metadata["moving_layer_alpha_area_px_min_median_max"],
            "eraser_depth_min_median_max": metadata["moving_eraser_depth_mm_min_median_max"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
