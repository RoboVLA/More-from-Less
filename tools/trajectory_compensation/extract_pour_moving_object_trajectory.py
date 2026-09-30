from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
POUR_DIR = WORKSPACE / "Data" / "ai生成倒水" / "倒水"
DEFAULT_RGB_VIDEO = POUR_DIR / "倒水.mp4"
DEFAULT_DEPTH_RESULT_DIR = POUR_DIR / "深度视频_倒水_分层几何一致真实深度版"
DEFAULT_OUTPUT_DIR = POUR_DIR / "运动物体轨迹_去水印RGB_分层深度"


def read_image(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Cannot encode image for: {path}")
    encoded.tofile(str(path))


def natural_key(path: Path) -> tuple:
    stem = path.stem
    parts: list[int | str] = []
    token = ""
    for ch in stem:
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
    paths = [p for p in directory.iterdir() if p.suffix.lower() in suffixes]
    return sorted(paths, key=natural_key)


def largest_component(alpha: np.ndarray, threshold: int, min_area: int) -> np.ndarray:
    if alpha.ndim == 3:
        alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
    binary = (alpha > threshold).astype(np.uint8)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if labels_count <= 1:
        return np.zeros_like(binary, dtype=np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_label = int(np.argmax(areas) + 1)
    if int(stats[keep_label, cv2.CC_STAT_AREA]) < min_area:
        return np.zeros_like(binary, dtype=np.uint8)
    return (labels == keep_label).astype(np.uint8)


def weighted_centroid(alpha: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    if alpha.ndim == 3:
        alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(mask)
    weights = alpha[ys, xs].astype(np.float64)
    if weights.sum() <= 0:
        return float(xs.mean()), float(ys.mean())
    return float(np.average(xs, weights=weights)), float(np.average(ys, weights=weights))


def robust_depth(depth_mm: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float]:
    values = depth_mm[(mask > 0) & (depth_mm > 0)].astype(np.float64)
    if values.size == 0:
        return math.nan, math.nan, math.nan, math.nan
    return (
        float(np.median(values)),
        float(np.mean(values)),
        float(np.percentile(values, 10)),
        float(np.percentile(values, 90)),
    )


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0, 0, 0, 0, 0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1, y1, int(xs.size)


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


def project_pixel_to_camera(
    u: float,
    v: float,
    z_mm: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float, float]:
    if not np.isfinite(z_mm):
        return math.nan, math.nan, math.nan
    x_mm = (u - cx) * z_mm / fx
    y_mm = (v - cy) * z_mm / fy
    return float(x_mm), float(y_mm), float(z_mm)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    result = np.empty_like(values, dtype=np.float64)
    half = window // 2
    for i in range(values.shape[0]):
        lo = max(0, i - half)
        hi = min(values.shape[0], i + half + 1)
        result[i] = np.nanmean(values[lo:hi], axis=0)
    return result


def draw_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    row: dict,
    history: list[tuple[int, int]],
) -> np.ndarray:
    out = frame.copy()
    overlay = out.copy()
    overlay[mask > 0] = (0, 210, 255)
    out = cv2.addWeighted(overlay, 0.28, out, 0.72, 0)

    x0, y0, x1, y1 = int(row["bbox_x0"]), int(row["bbox_y0"]), int(row["bbox_x1"]), int(row["bbox_y1"])
    u, v = int(round(row["u_px"])), int(round(row["v_px"]))
    cv2.rectangle(out, (x0, y0), (x1, y1), (20, 255, 255), 2)
    if len(history) >= 2:
        pts = np.array(history, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], False, (40, 40, 255), 3, cv2.LINE_AA)
    cv2.circle(out, (u, v), 7, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(out, (u, v), 10, (255, 255, 255), 2, cv2.LINE_AA)

    label = f"frame {row['frame']:03d}  z={row['z_median_mm']:.0f} mm"
    cv2.putText(out, label, (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(out, label, (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def normalize_to_range(values: np.ndarray, dst_min: float, dst_max: float) -> np.ndarray:
    values = values.astype(np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full_like(values, (dst_min + dst_max) / 2.0, dtype=np.float64)
    src_min = float(np.nanmin(values[finite]))
    src_max = float(np.nanmax(values[finite]))
    if abs(src_max - src_min) < 1e-6:
        return np.full_like(values, (dst_min + dst_max) / 2.0, dtype=np.float64)
    return dst_min + (values - src_min) * (dst_max - dst_min) / (src_max - src_min)


def draw_cv2_line_plot(
    x: np.ndarray,
    y: np.ndarray,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    y_increases_down: bool = False,
) -> None:
    width, height = 1000, 560
    margin_left, margin_right, margin_top, margin_bottom = 92, 34, 58, 78
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (margin_left, margin_top), (width - margin_right, height - margin_bottom), (40, 40, 40), 1)

    x_plot = normalize_to_range(x, margin_left, width - margin_right)
    if y_increases_down:
        y_plot = normalize_to_range(y, margin_top, height - margin_bottom)
    else:
        y_plot = normalize_to_range(y, height - margin_bottom, margin_top)
    points = np.column_stack([x_plot, y_plot]).round().astype(np.int32)
    if len(points) >= 2:
        cv2.polylines(canvas, [points.reshape((-1, 1, 2))], False, (40, 40, 210), 3, cv2.LINE_AA)
    for idx, pt in enumerate(points):
        color = (0, 150, 0) if idx == 0 else (0, 0, 220) if idx == len(points) - 1 else (70, 120, 230)
        cv2.circle(canvas, tuple(pt), 4, color, -1, cv2.LINE_AA)

    cv2.putText(canvas, title, (margin_left, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, x_label, (width // 2 - 70, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, y_label, (16, margin_top - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, "start", tuple(points[0] + np.array([8, -8])), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 130, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "end", tuple(points[-1] + np.array([8, -8])), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 180), 2, cv2.LINE_AA)

    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    cv2.putText(canvas, f"x: {x_min:.1f}..{x_max:.1f}", (margin_left, height - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"y: {y_min:.1f}..{y_max:.1f}", (margin_left, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 70, 70), 1, cv2.LINE_AA)
    write_image(output_path, canvas)


def save_cv2_plots(rows: list[dict], output_dir: Path, width: int, height: int) -> None:
    frames = np.array([r["frame"] for r in rows], dtype=np.float64)
    u = np.array([r["u_px"] for r in rows], dtype=np.float64)
    v = np.array([r["v_px"] for r in rows], dtype=np.float64)
    z = np.array([r["z_median_mm"] for r in rows], dtype=np.float64)
    xs = np.array([r["x_cam_mm"] for r in rows], dtype=np.float64)

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (40, 40, 40), 2)
    pts = np.column_stack([u, v]).round().astype(np.int32)
    if len(pts) >= 2:
        cv2.polylines(canvas, [pts.reshape((-1, 1, 2))], False, (40, 40, 210), 3, cv2.LINE_AA)
    for idx, pt in enumerate(pts):
        color = (0, 150, 0) if idx == 0 else (0, 0, 220) if idx == len(pts) - 1 else (70, 120, 230)
        cv2.circle(canvas, tuple(pt), 6, color, -1, cv2.LINE_AA)
    cv2.putText(canvas, "Moving object trajectory in RGB image pixels", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "start", tuple(pts[0] + np.array([9, -9])), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 130, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "end", tuple(pts[-1] + np.array([9, -9])), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 180), 2, cv2.LINE_AA)
    write_image(output_dir / "trajectory_2d_image.png", canvas)

    draw_cv2_line_plot(
        u,
        v,
        output_dir / "trajectory_2d_image_zoom.png",
        "Moving object trajectory in RGB pixels (zoomed)",
        "u pixel",
        "v pixel",
        y_increases_down=True,
    )

    draw_cv2_line_plot(
        frames,
        z,
        output_dir / "trajectory_z_time.png",
        "Moving object median depth over time",
        "frame",
        "z depth (mm)",
    )
    draw_cv2_line_plot(
        xs,
        z,
        output_dir / "trajectory_3d_camera_approx.png",
        "Approximate camera-space X-Z trajectory",
        "X camera (mm, estimated)",
        "Z depth (mm)",
    )


def save_plots(rows: list[dict], output_dir: Path, width: int, height: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency fallback
        print(f"[warn] matplotlib unavailable, writing OpenCV fallback plots: {exc}")
        save_cv2_plots(rows, output_dir, width, height)
        return

    frames = np.array([r["frame"] for r in rows], dtype=np.float64)
    u = np.array([r["u_px"] for r in rows], dtype=np.float64)
    v = np.array([r["v_px"] for r in rows], dtype=np.float64)
    z = np.array([r["z_median_mm"] for r in rows], dtype=np.float64)
    xs = np.array([r["x_cam_mm"] for r in rows], dtype=np.float64)
    ys = np.array([r["y_cam_mm"] for r in rows], dtype=np.float64)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(u, v, c=frames, cmap="viridis", s=32)
    plt.plot(u, v, color="black", linewidth=1.2, alpha=0.65)
    plt.gca().invert_yaxis()
    plt.xlabel("u pixel")
    plt.ylabel("v pixel")
    plt.title("Moving object trajectory in RGB image")
    plt.colorbar(sc, label="frame")
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_2d_image.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(frames, z, color="#1f77b4", linewidth=2)
    plt.xlabel("frame")
    plt.ylabel("median depth z (mm)")
    plt.title("Moving object depth over time")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_z_time.png", dpi=180)
    plt.close()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, z, color="#d62728", linewidth=2)
    ax.scatter(xs, ys, z, c=frames, cmap="viridis", s=26)
    ax.set_xlabel("X camera (mm)")
    ax.set_ylabel("Y camera (mm)")
    ax.set_zlabel("Z depth (mm)")
    ax.set_title("Approximate 3D trajectory")
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_3d_camera_approx.png", dpi=180)
    plt.close()


def open_video_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {path}")
    return writer


def file_is_locked(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("rb+"):
            return False
    except PermissionError:
        return True


def finalize_video_file(tmp_path: Path, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        try:
            final_path.unlink()
        except PermissionError:
            if final_path.stat().st_size > 0:
                print(f"[warn] final video is locked; keeping existing copy: {final_path}")
                for _ in range(10):
                    try:
                        tmp_path.unlink()
                        break
                    except PermissionError:
                        time.sleep(0.2)
                return
            raise
    shutil.copy2(str(tmp_path), str(final_path))

    # On Windows, OpenCV's mp4 writer can keep the just-written file locked
    # briefly even after release. The final copy is the deliverable; cleanup is
    # best-effort so the run does not fail after successful output generation.
    for _ in range(10):
        try:
            tmp_path.unlink()
            return
        except PermissionError:
            time.sleep(0.2)
    print(f"[warn] temporary video is still locked, left in place: {tmp_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the moving bottle trajectory from the de-watermarked RGB video and metric depth frames."
    )
    parser.add_argument("--rgb-video", type=Path, default=DEFAULT_RGB_VIDEO)
    parser.add_argument("--depth-result-dir", type=Path, default=DEFAULT_DEPTH_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alpha-threshold", type=int, default=24)
    parser.add_argument("--min-area", type=int, default=200)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--horizontal-fov-deg", type=float, default=86.0)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--force-overlay-video", action="store_true", help="Regenerate the overlay mp4 even if it already exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "preview_frames"
    preview_dir.mkdir(parents=True, exist_ok=True)

    depth_dir = args.depth_result_dir / "depth_mm_frames"
    alpha_dir = args.depth_result_dir / "moving_layer_alpha_frames"
    depth_paths = list_frames(depth_dir)
    alpha_paths = list_frames(alpha_dir)
    if not depth_paths:
        raise RuntimeError(f"No depth frames found: {depth_dir}")
    if len(depth_paths) != len(alpha_paths):
        raise RuntimeError(f"Depth/alpha count mismatch: {len(depth_paths)} vs {len(alpha_paths)}")

    cap = cv2.VideoCapture(str(args.rgb_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RGB video: {args.rgb_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    usable_count = min(frame_count, len(depth_paths), len(alpha_paths))
    fx, fy, cx, cy, intrinsics_method = compute_intrinsics(
        width,
        height,
        args.fx,
        args.fy,
        args.cx,
        args.cy,
        args.horizontal_fov_deg,
    )

    overlay_video = output_dir / f"{args.rgb_video.stem}_\u8fd0\u52a8\u7269\u4f53\u8f68\u8ff9_overlay.mp4"
    if overlay_video.exists() and not args.force_overlay_video:
        print(f"[info] overlay video already exists; skip regenerating it: {overlay_video}")
        tmp_overlay: Path | None = None
        writer: cv2.VideoWriter | None = None
    elif file_is_locked(overlay_video):
        print(f"[warn] final video is locked; skip regenerating overlay video: {overlay_video}")
        tmp_overlay: Path | None = None
        writer: cv2.VideoWriter | None = None
    else:
        tmp_overlay = Path(f"trajectory_overlay_tmp_{os.getpid()}.mp4")
        writer = open_video_writer(tmp_overlay, fps, width, height)

    rows: list[dict] = []
    history: list[tuple[int, int]] = []

    for index in range(usable_count):
        ok, frame = cap.read()
        if not ok:
            break
        depth_mm = read_image(depth_paths[index], cv2.IMREAD_UNCHANGED)
        alpha = read_image(alpha_paths[index], cv2.IMREAD_GRAYSCALE)
        if depth_mm.shape[:2] != frame.shape[:2]:
            depth_mm = cv2.resize(depth_mm, (width, height), interpolation=cv2.INTER_NEAREST)
        if alpha.shape[:2] != frame.shape[:2]:
            alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)

        mask = largest_component(alpha, args.alpha_threshold, args.min_area)
        x0, y0, x1, y1, area = bbox_from_mask(mask)
        if area > 0:
            u, v = weighted_centroid(alpha, mask)
            z_median, z_mean, z_p10, z_p90 = robust_depth(depth_mm, mask)
            x_cam, y_cam, z_cam = project_pixel_to_camera(u, v, z_median, fx, fy, cx, cy)
            history.append((int(round(u)), int(round(v))))
        else:
            u = v = z_median = z_mean = z_p10 = z_p90 = x_cam = y_cam = z_cam = math.nan

        row = {
            "frame": index,
            "time_s": index / fps,
            "u_px": u,
            "v_px": v,
            "z_median_mm": z_median,
            "z_mean_mm": z_mean,
            "z_p10_mm": z_p10,
            "z_p90_mm": z_p90,
            "x_cam_mm": x_cam,
            "y_cam_mm": y_cam,
            "z_cam_mm": z_cam,
            "bbox_x0": x0,
            "bbox_y0": y0,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "mask_area_px": area,
        }
        rows.append(row)
        overlay = draw_overlay(frame, mask, row, history)
        if writer is not None:
            writer.write(overlay)
        if index in {0, usable_count // 4, usable_count // 2, 3 * usable_count // 4, usable_count - 1}:
            write_image(preview_dir / f"trajectory_overlay_{index:04d}.jpg", overlay)

    cap.release()
    if writer is not None:
        writer.release()
        del writer
        cv2.destroyAllWindows()
        gc.collect()
        time.sleep(0.2)

    if not rows:
        raise RuntimeError("No frames were processed.")

    raw_points = np.array([[r["u_px"], r["v_px"], r["z_median_mm"], r["x_cam_mm"], r["y_cam_mm"]] for r in rows])
    smooth_points = moving_average(raw_points, int(args.smooth_window))
    for row, smooth in zip(rows, smooth_points):
        row["u_smooth_px"] = float(smooth[0])
        row["v_smooth_px"] = float(smooth[1])
        row["z_smooth_mm"] = float(smooth[2])
        row["x_cam_smooth_mm"] = float(smooth[3])
        row["y_cam_smooth_mm"] = float(smooth[4])

    csv_path = output_dir / "moving_object_trajectory.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    metadata = {
        "source_rgb_video": str(args.rgb_video),
        "source_compare_video_reference": str(args.depth_result_dir / "倒水_rgb_depth_layered_geometry_compare.mp4"),
        "source_depth_mm_frames": str(depth_dir),
        "source_moving_layer_alpha_frames": str(alpha_dir),
        "output_dir": str(output_dir),
        "frame_count_processed": len(rows),
        "fps": fps,
        "width": width,
        "height": height,
        "alpha_threshold": args.alpha_threshold,
        "min_area": args.min_area,
        "intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "method": intrinsics_method,
            "note": "x/y camera coordinates are approximate unless real Gemini RGB intrinsics are supplied. u/v/z_mm are directly measured from frames.",
        },
        "trajectory": rows,
    }
    json_path = output_dir / "moving_object_trajectory.json"
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if tmp_overlay is not None:
        finalize_video_file(tmp_overlay, overlay_video)

    save_plots(rows, output_dir, width, height)

    print(json.dumps(
        {
            "output_dir": str(output_dir),
            "csv": str(csv_path),
            "json": str(json_path),
            "overlay_video": str(overlay_video),
            "frames": len(rows),
            "z_median_min_mm": float(np.nanmin([r["z_median_mm"] for r in rows])),
            "z_median_max_mm": float(np.nanmax([r["z_median_mm"] for r in rows])),
            "intrinsics_method": intrinsics_method,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
