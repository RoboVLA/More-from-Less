from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from openni import openni2


DEFAULT_OPENNI_PATHS = [
    r"C:\Program Files (x86)\iPiSoft\iPi Recorder 4\OpenNI-2.3-Orbbec\amd64",
    r"C:\Program Files\OpenNI2\Redist",
]

MIN_DEPTH_MM = 100
MAX_DEPTH_MM = 5000
ESC_KEY = 27


def find_openni_path(explicit_path: str | None) -> str:
    candidates = [explicit_path] if explicit_path else DEFAULT_OPENNI_PATHS
    for candidate in candidates:
        if candidate and (Path(candidate) / "OpenNI2.dll").exists():
            return candidate
    raise RuntimeError(
        "OpenNI2.dll not found. Pass --openni-path or install the Orbbec/OpenNI runtime."
    )


def initialize(openni_path: str | None) -> str:
    path = find_openni_path(openni_path)
    openni2.initialize(path)
    return path


def depth_units_to_mm(video_mode) -> float:
    if video_mode.pixelFormat == openni2.PIXEL_FORMAT_DEPTH_100_UM:
        return 0.1
    return 1.0


def frame_to_depth_mm(frame, unit_scale: float) -> np.ndarray:
    data = np.ctypeslib.as_array(frame.get_buffer_as_uint16())
    return data.reshape(frame.height, frame.width).astype(np.float32) * unit_scale


def depth_to_display(depth_mm: np.ndarray, min_mm: int, max_mm: int) -> np.ndarray:
    clipped = np.where((depth_mm >= min_mm) & (depth_mm <= max_mm), depth_mm, 0)
    normalized = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)


def configure_depth_stream(stream, width: int | None, height: int | None, fps: int | None) -> None:
    if width is None and height is None and fps is None:
        return

    sensor_info = stream.get_sensor_info()
    modes = sensor_info.videoModes if sensor_info else []
    for mode in modes:
        width_ok = width is None or mode.resolutionX == width
        height_ok = height is None or mode.resolutionY == height
        fps_ok = fps is None or mode.fps == fps
        is_depth = mode.pixelFormat in (
            openni2.PIXEL_FORMAT_DEPTH_1_MM,
            openni2.PIXEL_FORMAT_DEPTH_100_UM,
        )
        if width_ok and height_ok and fps_ok and is_depth:
            stream.video_mode = mode
            return

    requested = f"{width or '*'}x{height or '*'}@{fps or '*'}"
    raise RuntimeError(f"No matching depth video mode: {requested}")


def print_info(args: argparse.Namespace) -> int:
    try:
        path = initialize(args.openni_path)
        uris = openni2.Device.enumerate_uris()
    except Exception as exc:
        print(f"OpenNI init failed: {exc}")
        return 1

    try:
        print(f"OpenNI runtime: {path}")
        print(f"Found {len(uris)} OpenNI device(s)")
        for index, uri in enumerate(uris):
            device = openni2.Device(uri)
            info = device.get_device_info()
            print()
            print(f"Device #{index}")
            print(f"  URI    : {info.uri.decode(errors='replace')}")
            print(f"  Vendor : {info.vendor.decode(errors='replace')}")
            print(f"  Name   : {info.name.decode(errors='replace')}")
            print(f"  VID/PID: 0x{info.usbVendorId:04X}/0x{info.usbProductId:04X}")

            depth_info = device.get_sensor_info(openni2.SENSOR_DEPTH)
            if depth_info:
                print("  Depth modes:")
                for mode in depth_info.videoModes:
                    print(
                        f"    {mode.resolutionX}x{mode.resolutionY} "
                        f"@ {mode.fps} fps {mode.pixelFormat}"
                    )
            else:
                print("  Depth modes: none")
            device.close()
    finally:
        openni2.unload()
    return 0


def open_depth_stream(args: argparse.Namespace):
    path = initialize(args.openni_path)
    device = openni2.Device.open_any()
    stream = device.create_depth_stream()
    if stream is None:
        raise RuntimeError("The OpenNI device has no depth stream.")
    configure_depth_stream(stream, args.width, args.height, args.fps)
    stream.start()
    mode = stream.video_mode
    unit_scale = depth_units_to_mm(mode)
    print(f"OpenNI runtime: {path}")
    print(f"Depth stream: {mode.resolutionX}x{mode.resolutionY} @ {mode.fps} fps {mode.pixelFormat}")
    return device, stream, unit_scale


def view_depth(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        device, stream, unit_scale = open_depth_stream(args)
    except Exception as exc:
        print(f"Cannot open depth stream: {exc}")
        return 1

    window = "OpenNI Gemini depth | q/Esc quit | s save"
    saved = 0
    try:
        while True:
            frame = stream.read_frame()
            depth_mm = frame_to_depth_mm(frame, unit_scale)
            display = depth_to_display(depth_mm, args.min_depth, args.max_depth)

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
                save_depth(out_dir, saved, depth_mm, display, stream.video_mode)
                saved += 1
    finally:
        stream.stop()
        device.close()
        openni2.unload()
        cv2.destroyAllWindows()
    return 0


def capture_once(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        device, stream, unit_scale = open_depth_stream(args)
    except Exception as exc:
        print(f"Cannot open depth stream: {exc}")
        return 1

    try:
        frame = stream.read_frame()
        depth_mm = frame_to_depth_mm(frame, unit_scale)
        display = depth_to_display(depth_mm, args.min_depth, args.max_depth)
        save_depth(out_dir, 0, depth_mm, display, stream.video_mode)
        return 0
    finally:
        stream.stop()
        device.close()
        openni2.unload()


def save_depth(out_dir: Path, index: int, depth_mm: np.ndarray, display: np.ndarray, video_mode) -> None:
    stem = time.strftime(f"openni_depth_%Y%m%d_%H%M%S_{index:03d}")
    raw_path = out_dir / f"{stem}_mm_u16.png"
    vis_path = out_dir / f"{stem}_color.png"
    meta_path = out_dir / f"{stem}_meta.json"

    cv2.imwrite(str(raw_path), np.clip(depth_mm, 0, 65535).astype(np.uint16))
    cv2.imwrite(str(vis_path), display)
    meta_path.write_text(
        json.dumps(
            {
                "type": "openni_depth",
                "width": int(video_mode.resolutionX),
                "height": int(video_mode.resolutionY),
                "fps": int(video_mode.fps),
                "pixel_format": str(video_mode.pixelFormat),
                "unit": "millimeter",
                "raw_depth_png": raw_path.name,
                "colored_depth_png": vis_path.name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    valid = depth_mm[depth_mm > 0]
    mean_depth = float(valid.mean()) if valid.size else 0.0
    print(f"Saved {raw_path}  valid_mean={mean_depth:.1f} mm")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drive older Orbbec Gemini/Astra depth cameras through OpenNI2.")
    parser.add_argument("--openni-path", default=None, help="Folder containing OpenNI2.dll.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info_parser = subparsers.add_parser("info", help="List OpenNI devices and depth modes.")
    info_parser.set_defaults(func=print_info)

    depth_parser = subparsers.add_parser("depth", help="Show live depth.")
    add_stream_args(depth_parser)
    depth_parser.add_argument("--out", default="captures", help="Output folder for saved frames.")
    depth_parser.set_defaults(func=view_depth)

    capture_parser = subparsers.add_parser("capture", help="Save one depth frame.")
    add_stream_args(capture_parser)
    capture_parser.add_argument("--out", default="captures", help="Output folder.")
    capture_parser.set_defaults(func=capture_once)
    return parser


def add_stream_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--width", type=int, default=None, help="Requested depth width.")
    parser.add_argument("--height", type=int, default=None, help="Requested depth height.")
    parser.add_argument("--fps", type=int, default=None, help="Requested depth FPS.")
    parser.add_argument("--min-depth", type=int, default=MIN_DEPTH_MM, help="Minimum displayed depth in millimeters.")
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH_MM, help="Maximum displayed depth in millimeters.")


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
