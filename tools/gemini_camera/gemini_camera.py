from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from pyorbbecsdk import (  # type: ignore
        AlignFilter,
        Config,
        Context,
        OBAlignMode,
        OBError,
        OBFormat,
        OBFrameAggregateOutputMode,
        OBLogLevel,
        OBSensorType,
        OBStreamType,
        Pipeline,
        PointCloudFilter,
        save_point_cloud_to_ply,
    )
except ImportError as exc:
    raise SystemExit(
        "pyorbbecsdk is not installed. Run:\n"
        "  python -m pip install --no-index --find-links wheels pyorbbecsdk2 opencv-python numpy"
    ) from exc


ESC_KEY = 27
MIN_DEPTH_MM = 100
MAX_DEPTH_MM = 5000


def set_quiet_logging() -> None:
    try:
        Context.set_logger_to_console(OBLogLevel.WARNING)
    except Exception:
        pass


def frame_to_bgr_image(frame: Any) -> np.ndarray | None:
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    data = np.asanyarray(frame.get_data())

    if fmt == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return np.resize(data, (height, width, 3))
    if fmt == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    if fmt == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == OBFormat.NV12:
        image = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_NV12)
    if fmt == OBFormat.NV21:
        image = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_NV21)
    if fmt == OBFormat.I420:
        image = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_I420)

    print(f"Unsupported color format: {fmt}")
    return None


def depth_frame_to_mm(depth_frame: Any) -> np.ndarray:
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = depth_frame.get_depth_scale()
    raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    return raw.reshape(height, width).astype(np.float32) * scale


def depth_mm_to_color(depth_mm: np.ndarray, min_mm: int, max_mm: int) -> np.ndarray:
    valid = np.where((depth_mm >= min_mm) & (depth_mm <= max_mm), depth_mm, 0)
    clipped = np.clip(valid, min_mm, max_mm)
    normalized = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)


def query_devices() -> Any:
    set_quiet_logging()
    ctx = Context()
    device_list = ctx.query_devices()
    if device_list.get_count() == 0:
        raise RuntimeError(
            "No Orbbec device found. Plug in the Gemini camera, use a USB 3 port, "
            "and verify it with OrbbecViewer first."
        )
    return device_list


def print_info(_: argparse.Namespace) -> int:
    try:
        device_list = query_devices()
    except RuntimeError as exc:
        print(exc)
        return 1

    print(f"Found {device_list.get_count()} Orbbec device(s)")
    for index in range(device_list.get_count()):
        device = device_list.get_device_by_index(index)
        info = device.get_device_info()
        print()
        print(f"Device #{index}")
        print(f"  Name       : {info.get_name()}")
        print(f"  Serial     : {info.get_serial_number()}")
        print(f"  Firmware   : {info.get_firmware_version()}")
        print(f"  Hardware   : {info.get_hardware_version()}")
        print(f"  VID/PID    : 0x{info.get_vid():04X}/0x{info.get_pid():04X}")
        print(f"  Connection : {info.get_connection_type()}")

        pipeline = Pipeline(device)
        for sensor_type, label in (
            (OBSensorType.DEPTH_SENSOR, "Depth"),
            (OBSensorType.COLOR_SENSOR, "Color"),
            (OBSensorType.IR_SENSOR, "IR"),
        ):
            try:
                profiles = pipeline.get_stream_profile_list(sensor_type)
                profile = profiles.get_default_video_stream_profile()
            except Exception:
                continue
            print(
                f"  {label:<5} default: "
                f"{profile.get_width()}x{profile.get_height()} "
                f"@ {profile.get_fps()} fps format={profile.get_format()}"
            )
    return 0


def make_depth_pipeline() -> Pipeline:
    pipeline = Pipeline()
    config = Config()
    profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profiles.get_default_video_stream_profile()
    config.enable_stream(depth_profile)
    pipeline.start(config)
    print(f"Depth stream: {depth_profile}")
    return pipeline


def view_depth(args: argparse.Namespace) -> int:
    try:
        query_devices()
        pipeline = make_depth_pipeline()
    except Exception as exc:
        print(f"Cannot start depth stream: {exc}")
        return 1

    window = "Gemini depth | q/Esc quit | s save"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    try:
        while True:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                continue

            depth_mm = depth_frame_to_mm(depth_frame)
            display = depth_mm_to_color(depth_mm, args.min_depth, args.max_depth)
            h, w = depth_mm.shape
            cx, cy = w // 2, h // 2
            center = depth_mm[cy, cx]
            label = f"{center:.0f} mm" if center > 0 else "invalid"
            cv2.circle(display, (cx, cy), 5, (255, 255, 255), -1)
            cv2.putText(display, label, (cx + 8, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ESC_KEY):
                break
            if key == ord("s"):
                save_depth_frame(out_dir, saved, depth_mm, display, depth_frame)
                saved += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0


def make_rgbd_pipeline(use_hw_align: bool) -> tuple[Pipeline, Config, AlignFilter | None]:
    pipeline = Pipeline()
    config = Config()

    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    try:
        color_profile = color_profiles.get_video_stream_profile(0, 0, OBFormat.RGB, 0)
    except Exception:
        color_profile = color_profiles.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profiles.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    align_filter: AlignFilter | None = None
    if use_hw_align:
        config.set_align_mode(OBAlignMode.HW_MODE)
    else:
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    try:
        pipeline.enable_frame_sync()
    except Exception:
        pass
    pipeline.start(config)
    print(f"Color stream: {color_profile}")
    print(f"Depth stream: {depth_profile}")
    return pipeline, config, align_filter


def view_rgbd(args: argparse.Namespace) -> int:
    try:
        query_devices()
        pipeline, _, align_filter = make_rgbd_pipeline(args.hw_align)
    except Exception as exc:
        print(f"Cannot start RGBD stream: {exc}")
        return 1

    window = "Gemini RGBD | q/Esc quit | s save"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    try:
        while True:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue
            if align_filter is not None:
                frames = align_filter.process(frames)
                if frames is None:
                    continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue

            color = frame_to_bgr_image(color_frame)
            if color is None:
                continue
            depth_mm = depth_frame_to_mm(depth_frame)
            depth_color = depth_mm_to_color(depth_mm, args.min_depth, args.max_depth)
            if depth_color.shape[:2] != color.shape[:2]:
                depth_color = cv2.resize(depth_color, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
            overlay = cv2.addWeighted(color, 0.6, depth_color, 0.4, 0)

            cv2.imshow(window, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ESC_KEY):
                break
            if key == ord("s"):
                save_rgbd_frame(out_dir, saved, color, depth_mm, depth_color, overlay, color_frame, depth_frame)
                saved += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0


def save_depth_frame(out_dir: Path, index: int, depth_mm: np.ndarray, depth_color: np.ndarray, depth_frame: Any) -> None:
    stem = time.strftime(f"depth_%Y%m%d_%H%M%S_{index:03d}")
    raw_path = out_dir / f"{stem}_depth_mm_u16.png"
    color_path = out_dir / f"{stem}_depth_color.png"
    meta_path = out_dir / f"{stem}_meta.json"

    cv2.imwrite(str(raw_path), np.clip(depth_mm, 0, 65535).astype(np.uint16))
    cv2.imwrite(str(color_path), depth_color)
    write_json(
        meta_path,
        {
            "type": "depth",
            "width": depth_frame.get_width(),
            "height": depth_frame.get_height(),
            "depth_scale": depth_frame.get_depth_scale(),
            "unit": "millimeter",
            "raw_depth_png": raw_path.name,
            "colored_depth_png": color_path.name,
        },
    )
    print(f"Saved {raw_path}")


def save_rgbd_frame(
    out_dir: Path,
    index: int,
    color: np.ndarray,
    depth_mm: np.ndarray,
    depth_color: np.ndarray,
    overlay: np.ndarray,
    color_frame: Any,
    depth_frame: Any,
) -> None:
    stem = time.strftime(f"rgbd_%Y%m%d_%H%M%S_{index:03d}")
    color_path = out_dir / f"{stem}_color.png"
    raw_path = out_dir / f"{stem}_depth_mm_u16.png"
    depth_vis_path = out_dir / f"{stem}_depth_color.png"
    overlay_path = out_dir / f"{stem}_overlay.png"
    meta_path = out_dir / f"{stem}_meta.json"

    cv2.imwrite(str(color_path), color)
    cv2.imwrite(str(raw_path), np.clip(depth_mm, 0, 65535).astype(np.uint16))
    cv2.imwrite(str(depth_vis_path), depth_color)
    cv2.imwrite(str(overlay_path), overlay)
    write_json(
        meta_path,
        {
            "type": "rgbd",
            "color_width": color_frame.get_width(),
            "color_height": color_frame.get_height(),
            "depth_width": depth_frame.get_width(),
            "depth_height": depth_frame.get_height(),
            "depth_scale": depth_frame.get_depth_scale(),
            "unit": "millimeter",
            "color_png": color_path.name,
            "raw_depth_png": raw_path.name,
            "colored_depth_png": depth_vis_path.name,
            "overlay_png": overlay_path.name,
        },
    )
    print(f"Saved {color_path} and {raw_path}")


def capture_once(args: argparse.Namespace) -> int:
    try:
        query_devices()
        pipeline, _, align_filter = make_rgbd_pipeline(args.hw_align)
    except Exception as exc:
        print(f"Cannot start capture stream: {exc}")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue
            if align_filter is not None:
                frames = align_filter.process(frames)
                if frames is None:
                    continue
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue

            color = frame_to_bgr_image(color_frame)
            if color is None:
                continue
            depth_mm = depth_frame_to_mm(depth_frame)
            depth_color = depth_mm_to_color(depth_mm, args.min_depth, args.max_depth)
            if depth_color.shape[:2] != color.shape[:2]:
                depth_color = cv2.resize(depth_color, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
            overlay = cv2.addWeighted(color, 0.6, depth_color, 0.4, 0)
            save_rgbd_frame(out_dir, 0, color, depth_mm, depth_color, overlay, color_frame, depth_frame)
            return 0
        print(f"No complete RGBD frame arrived within {args.timeout:.1f} seconds")
        return 1
    finally:
        pipeline.stop()


def capture_point_cloud(args: argparse.Namespace) -> int:
    try:
        query_devices()
        pipeline, _, align_filter = make_rgbd_pipeline(args.hw_align)
    except Exception as exc:
        print(f"Cannot start point-cloud stream: {exc}")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    point_cloud_filter = PointCloudFilter()
    point_cloud_filter.set_create_point_format(OBFormat.RGB_POINT)

    try:
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue
            if align_filter is not None:
                frames = align_filter.process(frames)
                if frames is None:
                    continue
            if frames.get_depth_frame() is None or frames.get_color_frame() is None:
                continue

            point_cloud_frame = point_cloud_filter.process(frames)
            if point_cloud_frame is None:
                continue

            ply_path = out_dir / time.strftime("gemini_point_cloud_%Y%m%d_%H%M%S.ply")
            save_point_cloud_to_ply(str(ply_path), point_cloud_frame)
            print(f"Saved {ply_path}")
            return 0
        print(f"No complete point cloud arrived within {args.timeout:.1f} seconds")
        return 1
    finally:
        pipeline.stop()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drive an Orbbec Gemini depth camera with Python.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info_parser = subparsers.add_parser("info", help="List connected Orbbec/Gemini cameras.")
    info_parser.set_defaults(func=print_info)

    depth_parser = subparsers.add_parser("depth", help="Show live depth image.")
    add_depth_args(depth_parser)
    depth_parser.add_argument("--out", default="captures", help="Output folder for saved frames.")
    depth_parser.set_defaults(func=view_depth)

    rgbd_parser = subparsers.add_parser("rgbd", help="Show live RGB plus aligned depth overlay.")
    add_depth_args(rgbd_parser)
    rgbd_parser.add_argument("--out", default="captures", help="Output folder for saved frames.")
    rgbd_parser.add_argument("--hw-align", action="store_true", help="Use hardware D2C alignment when supported.")
    rgbd_parser.set_defaults(func=view_rgbd)

    capture_parser = subparsers.add_parser("capture", help="Save one RGBD frame set.")
    add_depth_args(capture_parser)
    capture_parser.add_argument("--out", default="captures", help="Output folder.")
    capture_parser.add_argument("--timeout", type=float, default=10.0, help="Seconds to wait for a complete frame.")
    capture_parser.add_argument("--hw-align", action="store_true", help="Use hardware D2C alignment when supported.")
    capture_parser.set_defaults(func=capture_once)

    point_parser = subparsers.add_parser("pointcloud", help="Save one RGB point cloud as PLY.")
    point_parser.add_argument("--out", default="captures", help="Output folder.")
    point_parser.add_argument("--timeout", type=float, default=10.0, help="Seconds to wait for a complete frame.")
    point_parser.add_argument("--hw-align", action="store_true", help="Use hardware D2C alignment when supported.")
    point_parser.set_defaults(func=capture_point_cloud)

    return parser


def add_depth_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-depth", type=int, default=MIN_DEPTH_MM, help="Minimum displayed depth in millimeters.")
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH_MM, help="Maximum displayed depth in millimeters.")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
