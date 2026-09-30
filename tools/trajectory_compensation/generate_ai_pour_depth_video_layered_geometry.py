from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


AI_POUR_DIR = Path("Data") / "ai生成倒水" / "倒水"
DEPTH_SOURCE_DIR = "深度图相关"
DEFAULT_VIDEO_NAME = "倒水.mp4"
DEFAULT_PRIOR_DIR_NAME = "深度视频_倒水_DepthAnything_真实深度归一化"
DEFAULT_OUTPUT_DIR_NAME = "深度视频_倒水_分层几何一致真实深度版"


def imread_unicode(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return img


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    buf.tofile(str(path))


def latest_file(root: Path, pattern: str) -> Path:
    files = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise RuntimeError(f"No files matching {pattern} in {root}")
    return files[-1]


def read_video_frames(video_path: Path) -> tuple[list[np.ndarray], float, tuple[int, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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
    png_dir = tmp_dir / f"{out_path.stem}_{run_id}_png_frames"
    temp_path = tmp_dir / f"{out_path.stem}_{run_id}{out_path.suffix}"
    png_dir.mkdir(parents=True, exist_ok=True)
    try:
        for idx, frame in enumerate(frames):
            imwrite_unicode(png_dir / f"frame_{idx:04d}.png", frame)
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
                str(png_dir / "frame_%04d.png"),
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
        return
    except Exception:
        pass
    finally:
        shutil.rmtree(png_dir, ignore_errors=True)

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {temp_path}")
    for frame in frames:
        writer.write(frame)
    writer.release()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(temp_path, out_path)


def colorize_metric_depth(depth_mm: np.ndarray, near_mm: float, far_mm: float) -> np.ndarray:
    clipped = np.clip(depth_mm.astype(np.float32), near_mm, far_mm)
    norm = (far_mm - clipped) / max(1.0, far_mm - near_mm)
    return cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
    divider = np.full((left.shape[0], 8, 3), 255, dtype=np.uint8)
    return np.hstack([left, divider, right])


def overlay_alpha(frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    red = np.zeros_like(frame)
    red[:, :, 2] = np.clip(alpha, 0, 255).astype(np.uint8)
    return cv2.addWeighted(frame, 0.85, red, 0.50, 0)


def auto_bottle_depth_mm(real_depth_path: Path) -> int:
    depth = imread_unicode(real_depth_path, cv2.IMREAD_UNCHANGED)
    if depth.ndim != 2 or depth.dtype != np.uint16:
        return 648
    h, w = depth.shape[:2]
    vals = depth[int(h * 0.33) : int(h * 0.55), int(w * 0.56) : int(w * 0.66)]
    vals = vals[vals > 0]
    if vals.size < 200:
        return 648
    return int(np.median(vals))


def fill_depth_region(depth_mm: np.ndarray, fill_region: np.ndarray) -> np.ndarray:
    region = fill_region > 0
    if not np.any(region):
        return depth_mm
    original = depth_mm.astype(np.float32)
    filled = original.copy()
    h, w = filled.shape[:2]
    x_all = np.arange(w, dtype=np.float32)

    for y in range(h):
        row_region = region[y]
        if not np.any(row_region):
            continue
        row = filled[y]
        valid = (~row_region) & (row > 0)
        valid_x = x_all[valid]
        if valid_x.size >= 2:
            filled[y, row_region] = np.interp(x_all[row_region], valid_x, row[valid])
        elif valid_x.size == 1:
            filled[y, row_region] = row[valid][0]

    # Smooth only the repaired region boundary, keeping original static depth
    # outside the moving layer untouched.
    soft = cv2.GaussianBlur(region.astype(np.float32), (31, 31), 0)
    soft = np.clip(soft, 0.0, 1.0)
    smoothed = cv2.GaussianBlur(filled, (15, 15), 0)
    repaired = original * (1.0 - soft) + smoothed * soft
    repaired[region] = filled[region] * 0.65 + smoothed[region] * 0.35
    return np.clip(np.rint(repaired), 0, 65535).astype(np.uint16)


def build_static_background_depth(
    prior_paths: list[Path],
    shape: tuple[int, int],
    moving_alphas: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    stack: list[np.ndarray] = []
    raw_stack: list[np.ndarray] = []
    union_moving = np.zeros((h, w), dtype=np.uint8)
    if moving_alphas is not None and len(moving_alphas) != len(prior_paths):
        raise RuntimeError("Moving alpha count and depth prior count differ.")

    for idx, path in enumerate(prior_paths):
        depth = imread_unicode(path, cv2.IMREAD_UNCHANGED)
        if depth.ndim != 2 or depth.dtype != np.uint16:
            raise RuntimeError(f"Expected uint16 depth frame: {path}")
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.uint16)
        raw_stack.append(depth.astype(np.float32))
        depth_f = depth.astype(np.float32)
        if moving_alphas is not None:
            moving = (moving_alphas[idx] > 4).astype(np.uint8) * 255
            moving = cv2.dilate(moving, np.ones((17, 17), np.uint8), iterations=1)
            union_moving = cv2.bitwise_or(union_moving, moving)
            depth_f[moving > 0] = np.nan
        stack.append(depth_f)

    depth_stack = np.stack(stack, axis=0).astype(np.float32)
    with np.errstate(all="ignore"):
        background = np.nanmedian(depth_stack, axis=0)
    fallback = np.median(np.stack(raw_stack, axis=0), axis=0)
    background = np.where(np.isfinite(background), background, fallback)
    background_u16 = np.clip(np.rint(background), 0, 65535).astype(np.uint16)

    # Remove any remaining history of the moving object from the background layer.
    union_moving = cv2.dilate(union_moving, np.ones((19, 19), np.uint8), iterations=1)
    background_u16 = fill_depth_region(background_u16, union_moving)
    return background_u16, union_moving


def component_candidates(alpha_u8: np.ndarray, min_area: int, max_area: int) -> list[tuple[np.ndarray, tuple[int, int, int, int], int]]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats((alpha_u8 > 0).astype(np.uint8), 8)
    comps: list[tuple[np.ndarray, tuple[int, int, int, int], int]] = []
    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        area = int(area)
        if area < min_area or area > max_area:
            continue
        if w < 10 or h < 10:
            continue
        aspect = max(w / max(1, h), h / max(1, w))
        if aspect > 7.0:
            continue
        comp = (labels == idx).astype(np.uint8) * 255
        comps.append((comp, (int(x), int(y), int(w), int(h)), area))
    return comps


def initial_bottle_layer(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    alpha = np.zeros((h, w), dtype=np.uint8)
    cx = int(w * 0.61)
    cv2.rectangle(alpha, (cx - int(w * 0.020), int(h * 0.35)), (cx + int(w * 0.020), int(h * 0.43)), 255, -1)
    cv2.ellipse(alpha, (cx, int(h * 0.35)), (int(w * 0.022), int(h * 0.012)), 0, 0, 360, 255, -1)
    cv2.rectangle(alpha, (cx - int(w * 0.031), int(h * 0.43)), (cx + int(w * 0.031), int(h * 0.525)), 255, -1)
    cv2.ellipse(alpha, (cx, int(h * 0.525)), (int(w * 0.031), int(h * 0.019)), 0, 0, 360, 255, -1)
    return alpha


def bottle_layer_alpha(frame: np.ndarray, background_rgb: np.ndarray, frame_index: int) -> np.ndarray:
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    diff = cv2.cvtColor(cv2.absdiff(frame, background_rgb), cv2.COLOR_BGR2GRAY)

    roi = np.zeros((h, w), dtype=np.uint8)
    roi[int(h * 0.16) : int(h * 0.59), int(w * 0.38) : int(w * 0.74)] = 255

    low_sat_object = ((sat < 135) & (val > 35) & (roi > 0)).astype(np.uint8) * 255
    changed = ((diff > 14) & (roi > 0)).astype(np.uint8) * 255
    alpha = cv2.bitwise_and(low_sat_object, changed)

    if frame_index <= 10:
        alpha = cv2.bitwise_or(alpha, initial_bottle_layer((h, w)))

    # Remove fixed mechanical plates. Do not blank the original bottle area:
    # later upright bottle frames pass through the same image region, and a hard
    # rectangle there cuts the bottle body out of the generated depth.
    alpha[int(h * 0.28) : int(h * 0.48), int(w * 0.64) : int(w * 0.81)] = 0

    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    candidates = component_candidates(alpha, min_area=180, max_area=32_000)
    if not candidates:
        return np.zeros((h, w), dtype=np.uint8)

    scored: list[tuple[float, np.ndarray]] = []
    for comp, (x, y, cw, ch), area in candidates:
        cx = x + cw / 2.0
        cy = y + ch / 2.0
        if cy < h * 0.16 or cy > h * 0.60:
            continue
        center_score = 1.0 - min(1.0, abs(cx - w * 0.56) / (w * 0.30))
        upper_score = 1.0 - min(1.0, abs(cy - h * 0.38) / (h * 0.30))
        compact = area / max(1, cw * ch)
        score = area * (0.65 + 0.15 * center_score + 0.10 * upper_score + 0.10 * compact)
        scored.append((score, comp))
    if not scored:
        return np.zeros((h, w), dtype=np.uint8)

    scored.sort(key=lambda item: item[0], reverse=True)
    chosen = scored[0][1]
    pts = cv2.findNonZero(chosen)
    if pts is not None and len(pts) >= 8:
        hull = cv2.convexHull(pts)
        filled = np.zeros_like(chosen)
        cv2.fillConvexPoly(filled, hull, 255)
        chosen = cv2.bitwise_or(chosen, filled)
    chosen = cv2.morphologyEx(chosen, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    chosen = cv2.dilate(chosen, np.ones((5, 5), np.uint8), iterations=1)
    return cv2.GaussianBlur(chosen, (13, 13), 0)


def bottle_geometry_depth(alpha: np.ndarray, center_depth_mm: int, relief_mm: float) -> np.ndarray:
    h, w = alpha.shape[:2]
    depth = np.full((h, w), float(center_depth_mm), dtype=np.float32)
    ys, xs = np.nonzero(alpha > 16)
    if xs.size < 50:
        return depth

    points = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    long_axis = eigvecs[:, order[0]]
    short_axis = eigvecs[:, order[1]]
    short_coord = centered @ short_axis
    radius = float(np.percentile(np.abs(short_coord), 95))
    if radius < 1.0:
        return depth

    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    all_centered = np.stack([grid_x - center[0], grid_y - center[1]], axis=2)
    v = all_centered[:, :, 0] * short_axis[0] + all_centered[:, :, 1] * short_axis[1]
    edge = np.clip(np.abs(v) / radius, 0.0, 1.0)

    # Smooth cylindrical hint: center is slightly closer, edges slightly farther.
    # The mean stays close to center_depth_mm so the object Z does not drift.
    relief = relief_mm * (edge * edge - 0.42)
    depth = float(center_depth_mm) + relief
    return depth.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a layered, geometry-consistent metric depth video for the AI pouring RGB video."
    )
    parser.add_argument("--video", type=Path, default=None, help="Input AI pouring RGB video.")
    parser.add_argument("--prior-dir", type=Path, default=None, help="Metric Depth Anything prior directory.")
    parser.add_argument("--real-depth", type=Path, default=None, help="Gemini real depth png used to set bottle Z.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument("--bottle-depth-mm", type=int, default=None, help="Override moving bottle center depth.")
    parser.add_argument("--bottle-relief-mm", type=float, default=28.0, help="Small cylindrical depth relief for bottle layer.")
    parser.add_argument("--near-mm", type=float, default=279.0, help="Near value for color preview.")
    parser.add_argument("--far-mm", type=float, default=837.0, help="Far value for color preview.")
    parser.add_argument("--save-every-frame-color", action="store_true", help="Also save every color preview frame.")
    parser.add_argument("--save-alpha", action="store_true", help="Save every moving-layer alpha frame.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path.cwd()
    video_path = args.video if args.video else workspace / AI_POUR_DIR / DEFAULT_VIDEO_NAME
    prior_dir = args.prior_dir if args.prior_dir else workspace / AI_POUR_DIR / DEFAULT_PRIOR_DIR_NAME
    real_depth_path = args.real_depth if args.real_depth else latest_file(workspace / DEPTH_SOURCE_DIR, "openni_depth_*_mm_u16.png")
    out_dir = args.out_dir if args.out_dir else workspace / AI_POUR_DIR / DEFAULT_OUTPUT_DIR_NAME
    tmp_dir = workspace / "code" / "gemini_camera" / "tmp" / "ai_pour_layered_geometry"

    frames, fps, (width, height) = read_video_frames(video_path)
    prior_paths = sorted((prior_dir / "depth_mm_frames").glob("depth_*_mm_u16.png"))
    if len(prior_paths) != len(frames):
        raise RuntimeError(f"RGB frames ({len(frames)}) and prior depth frames ({len(prior_paths)}) differ.")

    bottle_depth_mm = args.bottle_depth_mm if args.bottle_depth_mm is not None else auto_bottle_depth_mm(real_depth_path)
    background_rgb = np.median(np.stack(frames, axis=0).astype(np.uint8), axis=0).astype(np.uint8)
    moving_alphas = [bottle_layer_alpha(frame, background_rgb, idx) for idx, frame in enumerate(frames)]
    background_depth, excluded_background_region = build_static_background_depth(
        prior_paths,
        (height, width),
        moving_alphas=moving_alphas,
    )

    out_depth_dir = out_dir / "depth_mm_frames"
    out_color_dir = out_dir / "depth_metric_color_frames"
    out_alpha_dir = out_dir / "moving_layer_alpha_frames"
    preview_dir = out_dir / "preview_frames"
    out_depth_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    if args.save_every_frame_color:
        out_color_dir.mkdir(parents=True, exist_ok=True)
    if args.save_alpha:
        out_alpha_dir.mkdir(parents=True, exist_ok=True)

    color_video_frames: list[np.ndarray] = []
    compare_video_frames: list[np.ndarray] = []
    alpha_areas: list[int] = []
    preview_indices = {0, len(frames) // 2, len(frames) - 1}

    imwrite_unicode(preview_dir / "static_background_depth_color.png", colorize_metric_depth(background_depth, args.near_mm, args.far_mm))
    imwrite_unicode(preview_dir / "static_background_excluded_region.png", excluded_background_region)

    for idx, (frame, prior_path) in enumerate(zip(frames, prior_paths)):
        alpha = moving_alphas[idx]
        alpha_f = np.clip(alpha.astype(np.float32) / 255.0, 0.0, 1.0)
        bottle_depth = bottle_geometry_depth(alpha, bottle_depth_mm, args.bottle_relief_mm)
        final_depth = background_depth.astype(np.float32) * (1.0 - alpha_f) + bottle_depth * alpha_f
        final_depth_u16 = np.clip(np.rint(final_depth), 0, 65535).astype(np.uint16)

        imwrite_unicode(out_depth_dir / prior_path.name, final_depth_u16)
        if args.save_alpha:
            imwrite_unicode(out_alpha_dir / f"moving_layer_alpha_{idx:04d}.png", alpha)

        color = colorize_metric_depth(final_depth_u16, args.near_mm, args.far_mm)
        if args.save_every_frame_color:
            imwrite_unicode(out_color_dir / f"depth_metric_color_{idx:04d}.png", color)
        color_video_frames.append(color)
        compare_video_frames.append(side_by_side(frame, color))

        if idx in preview_indices:
            imwrite_unicode(preview_dir / f"rgb_depth_layered_geometry_compare_{idx:04d}.jpg", side_by_side(frame, color))
            imwrite_unicode(preview_dir / f"depth_layered_geometry_color_{idx:04d}.png", color)
            imwrite_unicode(preview_dir / f"moving_layer_alpha_overlay_{idx:04d}.jpg", overlay_alpha(frame, alpha))

        alpha_areas.append(int(np.count_nonzero(alpha > 16)))

    color_video_path = out_dir / "倒水_depth_metric_layered_geometry_color_video.mp4"
    compare_video_path = out_dir / "倒水_rgb_depth_layered_geometry_compare.mp4"
    write_video(color_video_frames, color_video_path, fps, tmp_dir)
    write_video(compare_video_frames, compare_video_path, fps, tmp_dir)

    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_rgb_video": str(video_path),
        "source_metric_prior_dir": str(prior_dir),
        "source_real_depth_for_bottle_z": str(real_depth_path),
        "output_dir": str(out_dir),
        "output_depth_color_video": str(color_video_path),
        "output_rgb_depth_compare_video": str(compare_video_path),
        "output_depth_mm_frames": str(out_depth_dir),
        "frame_count": len(frames),
        "fps": fps,
        "width": width,
        "height": height,
        "moving_bottle_center_depth_mm": int(bottle_depth_mm),
        "bottle_relief_mm": float(args.bottle_relief_mm),
        "moving_layer_alpha_area_px_min_median_max": [
            int(np.min(alpha_areas)) if alpha_areas else 0,
            int(np.median(alpha_areas)) if alpha_areas else 0,
            int(np.max(alpha_areas)) if alpha_areas else 0,
        ],
        "method": (
            "Regenerates the depth video as layered geometry. The moving bottle layer is extracted first, then that "
            "moving region is excluded while building the static background depth from the real-metric Depth Anything "
            "priors. The moving bottle is rendered as a separate geometry layer at one Gemini-derived metric Z value "
            "with a small cylindrical relief. The final depth is composited during generation, so the bottle does not "
            "inherit per-frame monocular depth drift."
        ),
    }
    (out_dir / "depth_video_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"source_video={video_path}")
    print(f"source_metric_prior_dir={prior_dir}")
    print(f"moving_bottle_center_depth_mm={bottle_depth_mm}")
    print(f"output_dir={out_dir}")
    print(f"depth_color_video={color_video_path}")
    print(f"rgb_depth_compare_video={compare_video_path}")
    print(f"depth_mm_frames={out_depth_dir}")
    print(f"moving_layer_alpha_area_px_min_median_max={metadata['moving_layer_alpha_area_px_min_median_max']}")


if __name__ == "__main__":
    main()
