from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


AI_POUR_DIR = Path("Data") / "ai\u751f\u6210\u5012\u6c34" / "\u5012\u6c34"
DEFAULT_VIDEO_NAME = "\u5012\u6c34.mp4"
METRIC_DIR_NAME = "\u6df1\u5ea6\u89c6\u9891_\u5012\u6c34_DepthAnything_\u771f\u5b9e\u6df1\u5ea6\u5f52\u4e00\u5316"
OUTPUT_DIR_NAME = "\u6df1\u5ea6\u89c6\u9891_\u5012\u6c34_DepthAnything_\u771f\u5b9e\u6df1\u5ea6\u5f52\u4e00\u5316_\u74f6\u5b50Z\u7a33\u5b9a_\u7ec6\u8282\u4fdd\u6301\u7248"
DEPTH_SOURCE_DIR = "\u6df1\u5ea6\u56fe\u76f8\u5173"


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


def colorize_metric_depth(depth_mm: np.ndarray, near_mm: float, far_mm: float) -> np.ndarray:
    clipped = np.clip(depth_mm.astype(np.float32), near_mm, far_mm)
    norm = (far_mm - clipped) / max(1.0, far_mm - near_mm)
    return cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)


def write_video(frames: list[np.ndarray], out_path: Path, fps: float, tmp_dir: Path) -> None:
    if not frames:
        raise RuntimeError("No frames to write.")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{time.perf_counter_ns()}"
    temp_path = tmp_dir / f"{out_path.stem}_{run_id}{out_path.suffix}"
    png_dir = tmp_dir / f"{out_path.stem}_{run_id}_png_frames"
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
        # Fall back to OpenCV if ffmpeg is unavailable in a different runtime.
        pass

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {temp_path}")
    for frame in frames:
        writer.write(frame)
    writer.release()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(temp_path, out_path)


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
    divider = np.full((left.shape[0], 8, 3), 255, dtype=np.uint8)
    return np.hstack([left, divider, right])


def auto_bottle_depth_mm(real_depth_path: Path) -> int:
    depth = imread_unicode(real_depth_path, cv2.IMREAD_UNCHANGED)
    if depth.ndim != 2 or depth.dtype != np.uint16:
        return 644
    h, w = depth.shape[:2]
    # Bottle in the Gemini depth/RGB captures is near the right-center tabletop.
    x1 = int(w * 0.56)
    x2 = int(w * 0.66)
    y1 = int(h * 0.33)
    y2 = int(h * 0.55)
    vals = depth[y1:y2, x1:x2]
    vals = vals[vals > 0]
    if vals.size < 200:
        return 644
    return int(np.median(vals))


def build_temporal_background(frames: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(frames, axis=0).astype(np.uint8)
    return np.median(stack, axis=0).astype(np.uint8)


def component_filter(mask: np.ndarray, min_area: int, max_area: int) -> list[tuple[np.ndarray, tuple[int, int, int, int], int]]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    comps: list[tuple[np.ndarray, tuple[int, int, int, int], int]] = []
    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        area = int(area)
        if area < min_area or area > max_area:
            continue
        if w < 12 or h < 12:
            continue
        aspect = max(w / max(1, h), h / max(1, w))
        if aspect > 6.0:
            continue
        comp = (labels == idx).astype(np.uint8) * 255
        comps.append((comp, (int(x), int(y), int(w), int(h)), area))
    return comps


def initial_bottle_prior(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cx = int(w * 0.61)
    cv2.rectangle(mask, (cx - int(w * 0.020), int(h * 0.35)), (cx + int(w * 0.020), int(h * 0.43)), 255, -1)
    cv2.ellipse(mask, (cx, int(h * 0.35)), (int(w * 0.022), int(h * 0.012)), 0, 0, 360, 255, -1)
    cv2.rectangle(mask, (cx - int(w * 0.031), int(h * 0.43)), (cx + int(w * 0.031), int(h * 0.525)), 255, -1)
    cv2.ellipse(mask, (cx, int(h * 0.525)), (int(w * 0.031), int(h * 0.019)), 0, 0, 360, 255, -1)
    return mask


def bottle_mask_from_rgb(
    frame: np.ndarray,
    background: np.ndarray,
    frame_index: int,
    fill_convex_hull: bool,
    dilate_px: int,
) -> np.ndarray:
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    gray_diff = cv2.cvtColor(cv2.absdiff(frame, background), cv2.COLOR_BGR2GRAY)

    roi = np.zeros((h, w), dtype=np.uint8)
    roi[int(h * 0.17) : int(h * 0.58), int(w * 0.38) : int(w * 0.73)] = 255

    # The bottle is grey/silver with low-to-mid saturation. Shadows are usually
    # darker and larger; the component filtering below removes most of them.
    bottle_like = ((sat < 135) & (val > 35) & (roi > 0)).astype(np.uint8) * 255
    changed = ((gray_diff > 16) & (roi > 0)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(bottle_like, changed)

    if frame_index <= 10:
        # Early frames still include the upright source bottle at its original
        # pose; use a conservative geometric prior so its dark body is not lost.
        mask = cv2.bitwise_or(mask, initial_bottle_prior((h, w)))

    # Suppress right fixture and exposed original-location table/shadow regions.
    right_fixture = (int(w * 0.64), int(h * 0.28), int(w * 0.17), int(h * 0.20))
    rx, ry, rw, rh = right_fixture
    mask[ry : ry + rh, rx : rx + rw] = 0
    if frame_index > 10:
        original_table_shadow = (int(w * 0.57), int(h * 0.39), int(w * 0.12), int(h * 0.17))
        ox, oy, ow, oh = original_table_shadow
        mask[oy : oy + oh, ox : ox + ow] = 0

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)

    comps = component_filter(mask, min_area=180, max_area=32000)
    if not comps:
        return np.zeros((h, w), dtype=np.uint8)

    # Keep components in the bottle band. If there are several, keep the most
    # plausible compact changed object rather than shadows.
    scored: list[tuple[float, np.ndarray]] = []
    for comp, (x, y, cw, ch), area in comps:
        cx = x + cw / 2
        cy = y + ch / 2
        if cy > h * 0.58 or cy < h * 0.16:
            continue
        compact_bonus = area / max(1, cw * ch)
        center_bonus = 1.0 - min(1.0, abs(cx - w * 0.56) / (w * 0.28))
        upper_bonus = 1.0 - min(1.0, abs(cy - h * 0.36) / (h * 0.30))
        score = area * (0.65 + 0.20 * compact_bonus + 0.10 * center_bonus + 0.05 * upper_bonus)
        scored.append((score, comp))

    if not scored:
        return np.zeros((h, w), dtype=np.uint8)
    scored.sort(key=lambda item: item[0], reverse=True)
    chosen = scored[0][1]
    pts = cv2.findNonZero(chosen)
    if fill_convex_hull and pts is not None and len(pts) >= 8:
        hull = cv2.convexHull(pts)
        hull_mask = np.zeros_like(chosen)
        cv2.fillConvexPoly(hull_mask, hull, 255)
        chosen = cv2.bitwise_or(chosen, hull_mask)
    chosen = cv2.morphologyEx(chosen, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    # Cover dark bottle rims/undersides that RGB differencing can miss.
    if dilate_px > 0:
        kernel = max(1, int(dilate_px))
        if kernel % 2 == 0:
            kernel += 1
        chosen = cv2.dilate(chosen, np.ones((kernel, kernel), np.uint8), iterations=1)
    return chosen


def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    red = np.zeros_like(frame)
    red[:, :, 2] = mask
    return cv2.addWeighted(overlay, 0.82, red, 0.55, 0)


def stabilize_bottle_depth_with_relief(
    depth: np.ndarray,
    mask: np.ndarray,
    fixed_depth_mm: int,
    surface_relief_mm: float,
    feather_px: int,
) -> np.ndarray:
    valid_mask = mask > 0
    if not np.any(valid_mask):
        return depth.copy()

    depth_f = depth.astype(np.float32)
    bottle_values = depth_f[valid_mask]
    bottle_median = float(np.median(bottle_values))

    # Keep the bottle's local cylindrical surface cues but remove the per-frame
    # Z drift. Center the clipped relief so the bottle median stays fixed.
    relief = np.clip(depth_f - bottle_median, -surface_relief_mm, surface_relief_mm)
    relief_offset = float(np.median(relief[valid_mask]))
    target = fixed_depth_mm + relief - relief_offset

    kernel = max(3, int(feather_px) * 2 + 1)
    if kernel % 2 == 0:
        kernel += 1
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (kernel, kernel), max(1.0, feather_px / 2.0))
    soft = np.clip(soft, 0.0, 1.0)

    blended = depth_f * (1.0 - soft) + target * soft
    return np.clip(np.rint(blended), 0, 65535).astype(np.uint16)


def stabilize_bottle_depth_detail_preserving(
    depth: np.ndarray,
    mask: np.ndarray,
    fixed_depth_mm: int,
    relief_scale: float,
    max_relief_mm: float,
    feather_px: int,
) -> np.ndarray:
    valid_mask = mask > 0
    if not np.any(valid_mask):
        return depth.copy()

    depth_f = depth.astype(np.float32)
    bottle_values = depth_f[valid_mask]
    bottle_median = float(np.median(bottle_values))

    # Shift the whole bottle to the Gemini-derived Z value, but keep the local
    # per-pixel structure predicted by Depth Anything. This avoids the flat
    # block/mosaic look caused by overwriting the bottle with one constant.
    relief = depth_f - bottle_median
    if max_relief_mm > 0:
        relief = np.clip(relief, -max_relief_mm, max_relief_mm)
    target = fixed_depth_mm + relief * float(relief_scale)

    kernel = max(3, int(feather_px) * 2 + 1)
    if kernel % 2 == 0:
        kernel += 1
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (kernel, kernel), max(1.0, feather_px / 2.0))
    soft = np.clip(soft, 0.0, 1.0)

    blended = depth_f * (1.0 - soft) + target * soft
    return np.clip(np.rint(blended), 0, 65535).astype(np.uint16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stabilize bottle Z depth while preserving bottle detail.")
    parser.add_argument("--video", type=Path, default=None, help="Input RGB AI pouring video.")
    parser.add_argument("--metric-dir", type=Path, default=None, help="Directory with calibrated depth_mm_frames.")
    parser.add_argument("--real-depth", type=Path, default=None, help="Gemini raw depth png for auto bottle depth.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument("--fixed-bottle-depth-mm", type=int, default=None, help="Override fixed bottle depth in millimeters.")
    parser.add_argument("--near-mm", type=float, default=279.0, help="Near value for color preview.")
    parser.add_argument("--far-mm", type=float, default=837.0, help="Far value for color preview.")
    parser.add_argument("--mode", choices=["detail", "relief"], default="detail", help="Bottle Z stabilization mode.")
    parser.add_argument("--relief-scale", type=float, default=0.75, help="Detail mode: keep this fraction of original bottle local depth relief.")
    parser.add_argument("--max-relief-mm", type=float, default=160.0, help="Detail mode: clamp local bottle relief before shifting.")
    parser.add_argument("--surface-relief-mm", type=float, default=18.0, help="Allowed bottle surface depth variation around the fixed Z.")
    parser.add_argument("--feather-px", type=int, default=9, help="Soft edge feather radius for bottle/background blending.")
    parser.add_argument("--mask-dilate-px", type=int, default=3, help="Bottle mask dilation kernel size; smaller values avoid chunky mask boundaries.")
    parser.add_argument("--fill-convex-hull", action="store_true", help="Fill bottle mask convex hull. Disabled by default to avoid blocky patches.")
    parser.add_argument("--save-all-masks", action="store_true", help="Save every bottle mask png.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path.cwd()
    video_path = args.video if args.video else workspace / AI_POUR_DIR / DEFAULT_VIDEO_NAME
    metric_dir = args.metric_dir if args.metric_dir else workspace / AI_POUR_DIR / METRIC_DIR_NAME
    depth_frame_dir = metric_dir / "depth_mm_frames"
    real_depth_path = args.real_depth if args.real_depth else latest_file(workspace / DEPTH_SOURCE_DIR, "openni_depth_*_mm_u16.png")
    out_dir = args.out_dir if args.out_dir else workspace / AI_POUR_DIR / OUTPUT_DIR_NAME
    tmp_dir = workspace / "code" / "gemini_camera" / "tmp" / "bottle_z_stabilized"

    bottle_depth_mm = args.fixed_bottle_depth_mm if args.fixed_bottle_depth_mm is not None else auto_bottle_depth_mm(real_depth_path)
    frames, fps, (width, height) = read_video_frames(video_path)
    depth_paths = sorted(depth_frame_dir.glob("depth_*_mm_u16.png"))
    if len(depth_paths) != len(frames):
        raise RuntimeError(f"RGB frame count ({len(frames)}) and depth frame count ({len(depth_paths)}) differ.")

    background = build_temporal_background(frames)

    out_depth_dir = out_dir / "depth_mm_frames"
    out_mask_dir = out_dir / "bottle_masks"
    preview_dir = out_dir / "preview_frames"
    out_depth_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    color_frames: list[np.ndarray] = []
    compare_frames: list[np.ndarray] = []
    mask_areas: list[int] = []
    before_bottle_medians: list[float] = []
    after_bottle_medians: list[float] = []
    preview_indices = {0, len(frames) // 2, len(frames) - 1}

    for idx, (frame, depth_path) in enumerate(zip(frames, depth_paths)):
        depth = imread_unicode(depth_path, cv2.IMREAD_UNCHANGED)
        if depth.ndim != 2 or depth.dtype != np.uint16:
            raise RuntimeError(f"Expected uint16 depth frame: {depth_path}")
        mask = bottle_mask_from_rgb(
            frame,
            background,
            idx,
            fill_convex_hull=args.fill_convex_hull,
            dilate_px=args.mask_dilate_px,
        )
        if mask.shape[:2] != depth.shape[:2]:
            mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid_mask = mask > 0
        if np.any(valid_mask):
            before_bottle_medians.append(float(np.median(depth[valid_mask])))
        else:
            before_bottle_medians.append(float("nan"))
        if args.mode == "relief":
            stabilized = stabilize_bottle_depth_with_relief(
                depth,
                mask,
                bottle_depth_mm,
                surface_relief_mm=args.surface_relief_mm,
                feather_px=args.feather_px,
            )
        else:
            stabilized = stabilize_bottle_depth_detail_preserving(
                depth,
                mask,
                bottle_depth_mm,
                relief_scale=args.relief_scale,
                max_relief_mm=args.max_relief_mm,
                feather_px=args.feather_px,
            )
        if np.any(valid_mask):
            after_bottle_medians.append(float(np.median(stabilized[valid_mask])))
        else:
            after_bottle_medians.append(float("nan"))

        imwrite_unicode(out_depth_dir / depth_path.name, stabilized)
        if args.save_all_masks or idx in preview_indices:
            imwrite_unicode(out_mask_dir / f"bottle_mask_{idx:04d}.png", mask)

        color = colorize_metric_depth(stabilized, args.near_mm, args.far_mm)
        color_frames.append(color)
        compare_frames.append(side_by_side(frame, color))

        if idx in preview_indices:
            imwrite_unicode(preview_dir / f"bottle_mask_overlay_{idx:04d}.jpg", overlay_mask(frame, mask))
            imwrite_unicode(preview_dir / f"depth_color_bottle_z_{args.mode}_{idx:04d}.png", color)
            imwrite_unicode(preview_dir / f"rgb_depth_bottle_z_{args.mode}_compare_{idx:04d}.jpg", side_by_side(frame, color))

        mask_areas.append(int(np.count_nonzero(mask)))

    if args.mode == "relief":
        color_video_path = out_dir / "\u5012\u6c34_depth_metric_bottle_z_stable_relief_color_video.mp4"
        compare_video_path = out_dir / "\u5012\u6c34_rgb_depth_bottle_z_stable_relief_compare.mp4"
    else:
        color_video_path = out_dir / "\u5012\u6c34_depth_metric_bottle_z_stable_detail_color_video.mp4"
        compare_video_path = out_dir / "\u5012\u6c34_rgb_depth_bottle_z_stable_detail_compare.mp4"
    write_video(color_frames, color_video_path, fps, tmp_dir)
    write_video(compare_frames, compare_video_path, fps, tmp_dir)

    before_arr = np.array(before_bottle_medians, dtype=np.float32)
    after_arr = np.array(after_bottle_medians, dtype=np.float32)
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_rgb_video": str(video_path),
        "source_metric_depth_dir": str(metric_dir),
        "source_real_depth_for_fixed_bottle_depth": str(real_depth_path),
        "output_dir": str(out_dir),
        "output_depth_color_video": str(color_video_path),
        "output_rgb_depth_compare_video": str(compare_video_path),
        "output_depth_mm_frames": str(out_depth_dir),
        "output_bottle_masks": str(out_mask_dir),
        "frame_count": len(frames),
        "fps": fps,
        "width": width,
        "height": height,
        "fixed_bottle_depth_mm": int(bottle_depth_mm),
        "mode": args.mode,
        "relief_scale": float(args.relief_scale),
        "max_relief_mm": float(args.max_relief_mm),
        "surface_relief_mm": float(args.surface_relief_mm),
        "feather_px": int(args.feather_px),
        "mask_dilate_px": int(args.mask_dilate_px),
        "fill_convex_hull": bool(args.fill_convex_hull),
        "preview_color_scale_mm": {"near_mm": args.near_mm, "far_mm": args.far_mm},
        "bottle_mask_area_px_min_median_max": [
            int(np.min(mask_areas)) if mask_areas else 0,
            int(np.median(mask_areas)) if mask_areas else 0,
            int(np.max(mask_areas)) if mask_areas else 0,
        ],
        "bottle_depth_before_fix_mm_min_median_max": [
            float(np.nanmin(before_arr)),
            float(np.nanmedian(before_arr)),
            float(np.nanmax(before_arr)),
        ],
        "bottle_depth_after_fix_mm_min_median_max": [
            float(np.nanmin(after_arr)),
            float(np.nanmedian(after_arr)),
            float(np.nanmax(after_arr)),
        ],
        "method": (
            "Post-processes the metric-calibrated Depth Anything frames. A temporal median RGB background is used "
            "to find the moving grey bottle in the central workspace. In detail mode the bottle is shifted as a "
            "whole so its median depth stays at the Gemini-derived millimeter value, while per-pixel local depth "
            "detail from the original Depth Anything frame is retained and softly blended at the mask boundary. "
            "The default mask avoids convex-hull filling to reduce blocky/mosaic patches."
        ),
    }
    (out_dir / "depth_video_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"source_video={video_path}")
    print(f"source_metric_dir={metric_dir}")
    print(f"fixed_bottle_depth_mm={bottle_depth_mm}")
    print(f"output_dir={out_dir}")
    print(f"depth_color_video={color_video_path}")
    print(f"rgb_depth_compare_video={compare_video_path}")
    print(f"depth_mm_frames={out_depth_dir}")
    print(f"bottle_masks={out_mask_dir}")
    print(f"mask_area_min_median_max={metadata['bottle_mask_area_px_min_median_max']}")
    print(f"bottle_depth_before_fix_mm_min_median_max={metadata['bottle_depth_before_fix_mm_min_median_max']}")
    print(f"bottle_depth_after_fix_mm_min_median_max={metadata['bottle_depth_after_fix_mm_min_median_max']}")


if __name__ == "__main__":
    main()
