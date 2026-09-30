from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
POUR_DIR = WORKSPACE / "Data" / "ai\u751f\u6210\u5012\u6c34" / "\u5012\u6c34"
DEFAULT_RGB_VIDEO = POUR_DIR / "\u5012\u6c34.mp4"
DEFAULT_DEPTH_RESULT_DIR = POUR_DIR / "\u6df1\u5ea6\u89c6\u9891_\u5012\u6c34_\u5206\u5c42\u51e0\u4f55\u4e00\u81f4\u771f\u5b9e\u6df1\u5ea6\u7248"
DEFAULT_OUTPUT_DIR = POUR_DIR / "\u8fd0\u52a8\u7269\u4f53\u8f68\u8ff9_\u53bb\u6c34\u5370RGB_\u5206\u5c42\u6df1\u5ea6"


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


def weighted_pixel_centroid(alpha: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    weights = alpha[ys, xs].astype(np.float64)
    if weights.sum() <= 0:
        return float(xs.mean()), float(ys.mean())
    return float(np.average(xs, weights=weights)), float(np.average(ys, weights=weights))


def project_pixel_to_camera(
    u: float,
    v: float,
    z_mm: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float, float]:
    x_mm = (u - cx) * z_mm / fx
    y_mm = (v - cy) * z_mm / fy
    return float(x_mm), float(y_mm), float(z_mm)


def mask_point_cloud(
    depth_mm: np.ndarray,
    alpha: np.ndarray,
    mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero((mask > 0) & (depth_mm > 0))
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)
    z = depth_mm[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    points = np.column_stack([x, y, z])
    weights = np.maximum(alpha[ys, xs].astype(np.float64), 1.0)
    return points, weights


def principal_axis_3d(points: np.ndarray, weights: np.ndarray, previous_axis: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] < 6:
        if previous_axis is not None:
            return previous_axis, np.zeros(3, dtype=np.float64)
        return np.array([0.0, -1.0, 0.0], dtype=np.float64), np.zeros(3, dtype=np.float64)

    w = weights / np.sum(weights)
    center = np.sum(points * w[:, None], axis=0)
    shifted = points - center
    covariance = (shifted * w[:, None]).T @ shifted
    eigvals, eigvecs = np.linalg.eigh(covariance)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    axis = axis / max(np.linalg.norm(axis), 1e-12)

    if previous_axis is not None:
        if float(np.dot(axis, previous_axis)) < 0.0:
            axis = -axis
    else:
        # Choose a deterministic first-frame sign: local +Z points upward in
        # the image/camera-Y direction. The sign is kept continuous afterward.
        if axis[1] > 0.0:
            axis = -axis
    return axis, eigvals


def rotation_from_object_axis(axis_z: np.ndarray) -> np.ndarray:
    z_axis = axis_z / max(np.linalg.norm(axis_z), 1e-12)
    camera_forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = np.cross(camera_forward, z_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-12)
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1.0
    return rotation


def matrix_to_euler_xyz_deg(rotation: np.ndarray) -> tuple[float, float, float]:
    # XYZ Tait-Bryan angles represented as R = Rz(rz) * Ry(ry) * Rx(rx).
    sy = math.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0])
    singular = sy < 1e-8
    if not singular:
        rx = math.atan2(rotation[2, 1], rotation[2, 2])
        ry = math.atan2(-rotation[2, 0], sy)
        rz = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        rx = math.atan2(-rotation[1, 2], rotation[1, 1])
        ry = math.atan2(-rotation[2, 0], sy)
        rz = 0.0
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0, 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), int(len(xs))


def draw_axis_preview(frame: np.ndarray, row: dict, output_path: Path) -> None:
    out = frame.copy()
    u = int(round(row["u_px"]))
    v = int(round(row["v_px"]))
    axis_px = np.array([row["axis_u"], row["axis_v"]], dtype=np.float64)
    axis_px = axis_px / max(np.linalg.norm(axis_px), 1e-12)
    p0 = (int(round(u - axis_px[0] * 70)), int(round(v - axis_px[1] * 70)))
    p1 = (int(round(u + axis_px[0] * 70)), int(round(v + axis_px[1] * 70)))
    cv2.line(out, p0, p1, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.circle(out, (u, v), 8, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(out, (u, v), 5, (0, 0, 255), -1, cv2.LINE_AA)
    text = f"x={row['x_mm']:.1f} y={row['y_mm']:.1f} z={row['z_mm']:.1f} mm  r=({row['rx_deg']:.1f},{row['ry_deg']:.1f},{row['rz_deg']:.1f})"
    cv2.putText(out, text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(out, text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)
    write_image(output_path, out)


def draw_simple_plot(rows: list[dict], fields: list[str], output_path: Path, title: str) -> None:
    width, height = 1100, 650
    left, right, top, bottom = 88, 24, 56, 56
    graph_w = width - left - right
    graph_h = height - top - bottom
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (left, top), (width - right, height - bottom), (50, 50, 50), 1)
    frames = np.array([r["frame"] for r in rows], dtype=np.float64)
    colors = [(220, 60, 30), (40, 150, 40), (40, 80, 220), (180, 80, 180), (40, 170, 190), (200, 130, 20)]
    all_values = np.concatenate([np.array([r[field] for r in rows], dtype=np.float64) for field in fields])
    y_min = float(np.nanmin(all_values))
    y_max = float(np.nanmax(all_values))
    if abs(y_max - y_min) < 1e-6:
        y_min -= 1.0
        y_max += 1.0
    x_min, x_max = float(frames.min()), float(frames.max())

    for idx, field in enumerate(fields):
        values = np.array([r[field] for r in rows], dtype=np.float64)
        xs = left + (frames - x_min) * graph_w / max(x_max - x_min, 1e-6)
        ys = height - bottom - (values - y_min) * graph_h / max(y_max - y_min, 1e-6)
        pts = np.column_stack([xs, ys]).round().astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], False, colors[idx % len(colors)], 2, cv2.LINE_AA)
        cv2.putText(canvas, field, (left + 18 + idx * 150, top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, colors[idx % len(colors)], 2, cv2.LINE_AA)

    cv2.putText(canvas, title, (left, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"value range: {y_min:.1f} .. {y_max:.1f}", (left, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 70, 70), 1, cv2.LINE_AA)
    write_image(output_path, canvas)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract x,y,z,rx,ry,rz pose trajectory for the moving pouring object.")
    parser.add_argument("--rgb-video", type=Path, default=DEFAULT_RGB_VIDEO)
    parser.add_argument("--depth-result-dir", type=Path, default=DEFAULT_DEPTH_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alpha-threshold", type=int, default=24)
    parser.add_argument("--min-area", type=int, default=200)
    parser.add_argument("--horizontal-fov-deg", type=float, default=86.0)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "pose_preview_frames"
    preview_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.rgb_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RGB video: {args.rgb_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fx, fy, cx, cy, intrinsics_method = compute_intrinsics(width, height, args.fx, args.fy, args.cx, args.cy, args.horizontal_fov_deg)

    depth_paths = list_frames(args.depth_result_dir / "depth_mm_frames")
    alpha_paths = list_frames(args.depth_result_dir / "moving_layer_alpha_frames")
    usable_count = min(frame_count, len(depth_paths), len(alpha_paths))
    if usable_count == 0:
        raise RuntimeError("No usable aligned RGB/depth/alpha frames.")

    rows: list[dict] = []
    previous_axis: np.ndarray | None = None
    preview_indices = {0, usable_count // 4, usable_count // 2, 3 * usable_count // 4, usable_count - 1}

    for frame_index in range(usable_count):
        ok, frame = cap.read()
        if not ok:
            break
        depth_mm = read_image(depth_paths[frame_index], cv2.IMREAD_UNCHANGED)
        alpha = read_image(alpha_paths[frame_index], cv2.IMREAD_GRAYSCALE)
        if depth_mm.shape[:2] != (height, width):
            depth_mm = cv2.resize(depth_mm, (width, height), interpolation=cv2.INTER_NEAREST)
        if alpha.shape[:2] != (height, width):
            alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)

        mask = largest_component(alpha, args.alpha_threshold, args.min_area)
        x0, y0, x1, y1, area = bbox_from_mask(mask)
        if area == 0:
            continue

        u, v = weighted_pixel_centroid(alpha, mask)
        z_values = depth_mm[(mask > 0) & (depth_mm > 0)].astype(np.float64)
        z_mm = float(np.median(z_values))
        x_mm, y_mm, z_mm = project_pixel_to_camera(u, v, z_mm, fx, fy, cx, cy)

        points, weights = mask_point_cloud(depth_mm, alpha, mask, fx, fy, cx, cy)
        axis, eigvals = principal_axis_3d(points, weights, previous_axis)
        previous_axis = axis
        rotation = rotation_from_object_axis(axis)
        rx_deg, ry_deg, rz_deg = matrix_to_euler_xyz_deg(rotation)

        axis_u = axis[0] * fx / max(axis[2] + z_mm / max(abs(z_mm), 1e-6), 1e-6)
        axis_v = axis[1] * fy / max(axis[2] + z_mm / max(abs(z_mm), 1e-6), 1e-6)
        if not np.isfinite(axis_u) or not np.isfinite(axis_v) or abs(axis_u) + abs(axis_v) < 1e-6:
            axis_u, axis_v = axis[0], axis[1]

        row = {
            "frame": frame_index,
            "time_s": frame_index / fps,
            "x_mm": x_mm,
            "y_mm": y_mm,
            "z_mm": z_mm,
            "rx_deg": rx_deg,
            "ry_deg": ry_deg,
            "rz_deg": rz_deg,
            "u_px": u,
            "v_px": v,
            "axis_x": float(axis[0]),
            "axis_y": float(axis[1]),
            "axis_z": float(axis[2]),
            "axis_u": float(axis_u),
            "axis_v": float(axis_v),
            "bbox_x0": x0,
            "bbox_y0": y0,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "mask_area_px": area,
            "pca_eig_0": float(eigvals[0]) if eigvals.size else 0.0,
            "pca_eig_1": float(eigvals[1]) if eigvals.size else 0.0,
            "pca_eig_2": float(eigvals[2]) if eigvals.size else 0.0,
        }
        rows.append(row)
        if frame_index in preview_indices:
            draw_axis_preview(frame, row, preview_dir / f"pose_axis_{frame_index:04d}.jpg")

    cap.release()
    if not rows:
        raise RuntimeError("No pose rows were produced.")

    csv_path = output_dir / "moving_object_pose_trajectory_xyzrpy.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "moving_object_pose_trajectory_xyzrpy.json"
    json_path.write_text(
        json.dumps(
            {
                "source_rgb_video": str(args.rgb_video),
                "source_depth_mm_frames": str(args.depth_result_dir / "depth_mm_frames"),
                "source_moving_layer_alpha_frames": str(args.depth_result_dir / "moving_layer_alpha_frames"),
                "coordinate_frame": "camera frame: +X right, +Y down, +Z forward; units are millimeters and degrees",
                "rotation_convention": "rx,ry,rz are XYZ Tait-Bryan degrees with R = Rz(rz) * Ry(ry) * Rx(rx); local +Z is the bottle long axis from weighted 3D PCA; roll about the bottle axis is constrained by camera-forward reference because it is not fully observable from this input.",
                "intrinsics": {
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy,
                    "method": intrinsics_method,
                    "note": "Replace with real Gemini RGB intrinsics via --fx --fy --cx --cy for metrically exact X/Y and angles.",
                },
                "frame_count_processed": len(rows),
                "fps": fps,
                "trajectory": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    draw_simple_plot(rows, ["x_mm", "y_mm", "z_mm"], output_dir / "pose_xyz_over_time.png", "Object position trajectory")
    draw_simple_plot(rows, ["rx_deg", "ry_deg", "rz_deg"], output_dir / "pose_rpy_over_time.png", "Object orientation trajectory")

    print(json.dumps(
        {
            "csv": str(csv_path),
            "json": str(json_path),
            "frames": len(rows),
            "x_mm_start_end": [rows[0]["x_mm"], rows[-1]["x_mm"]],
            "y_mm_start_end": [rows[0]["y_mm"], rows[-1]["y_mm"]],
            "z_mm_min_max": [min(r["z_mm"] for r in rows), max(r["z_mm"] for r in rows)],
            "rx_deg_start_end": [rows[0]["rx_deg"], rows[-1]["rx_deg"]],
            "ry_deg_start_end": [rows[0]["ry_deg"], rows[-1]["ry_deg"]],
            "rz_deg_start_end": [rows[0]["rz_deg"], rows[-1]["rz_deg"]],
            "intrinsics_method": intrinsics_method,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
