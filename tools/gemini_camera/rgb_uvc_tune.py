from __future__ import annotations

import argparse
import sys

import cv2


PROPS = {
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
    "contrast": cv2.CAP_PROP_CONTRAST,
    "saturation": cv2.CAP_PROP_SATURATION,
    "hue": cv2.CAP_PROP_HUE,
    "gain": cv2.CAP_PROP_GAIN,
    "exposure": cv2.CAP_PROP_EXPOSURE,
    "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
}

BACKENDS = {
    "any": 0,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}


def open_camera(index: int, backend: str):
    api = BACKENDS[backend]
    return cv2.VideoCapture(index, api) if api else cv2.VideoCapture(index)


def list_cameras(args: argparse.Namespace) -> int:
    for backend in BACKENDS:
        print(f"[{backend}]")
        for index in range(args.max_index + 1):
            cap = open_camera(index, backend)
            ok = cap.isOpened()
            ret, shape = False, None
            if ok:
                ret, frame = cap.read()
                shape = None if not ret else frame.shape
            print(f"  index={index} opened={ok} frame={shape}")
            cap.release()
    return 0


def print_props(cap) -> None:
    for name, prop_id in PROPS.items():
        print(f"{name:>13}: {cap.get(prop_id)}")


def set_if_given(cap, name: str, value: float | None) -> None:
    if value is None:
        return
    prop_id = PROPS[name]
    ok = cap.set(prop_id, value)
    print(f"set {name}={value}: {ok}; now={cap.get(prop_id)}")


def preview(args: argparse.Namespace) -> int:
    cap = open_camera(args.index, args.backend)
    if not cap.isOpened():
        print(
            "Cannot open camera. Close other camera apps first, then try another "
            "--index or --backend. Use `list` to probe indices."
        )
        return 1

    set_if_given(cap, "auto_exposure", args.auto_exposure)
    set_if_given(cap, "exposure", args.exposure)
    set_if_given(cap, "gain", args.gain)
    set_if_given(cap, "brightness", args.brightness)
    set_if_given(cap, "contrast", args.contrast)
    set_if_given(cap, "saturation", args.saturation)

    print("Current camera properties:")
    print_props(cap)

    window = "RGB UVC tune | q/Esc quit | +/- exposure | g/G gain | b/B brightness"
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("+"), ord("=")):
            cap.set(cv2.CAP_PROP_EXPOSURE, cap.get(cv2.CAP_PROP_EXPOSURE) + args.exposure_step)
            print(f"exposure={cap.get(cv2.CAP_PROP_EXPOSURE)}")
        elif key in (ord("-"), ord("_")):
            cap.set(cv2.CAP_PROP_EXPOSURE, cap.get(cv2.CAP_PROP_EXPOSURE) - args.exposure_step)
            print(f"exposure={cap.get(cv2.CAP_PROP_EXPOSURE)}")
        elif key == ord("g"):
            cap.set(cv2.CAP_PROP_GAIN, cap.get(cv2.CAP_PROP_GAIN) + args.gain_step)
            print(f"gain={cap.get(cv2.CAP_PROP_GAIN)}")
        elif key == ord("G"):
            cap.set(cv2.CAP_PROP_GAIN, cap.get(cv2.CAP_PROP_GAIN) - args.gain_step)
            print(f"gain={cap.get(cv2.CAP_PROP_GAIN)}")
        elif key == ord("b"):
            cap.set(cv2.CAP_PROP_BRIGHTNESS, cap.get(cv2.CAP_PROP_BRIGHTNESS) + args.brightness_step)
            print(f"brightness={cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
        elif key == ord("B"):
            cap.set(cv2.CAP_PROP_BRIGHTNESS, cap.get(cv2.CAP_PROP_BRIGHTNESS) - args.brightness_step)
            print(f"brightness={cap.get(cv2.CAP_PROP_BRIGHTNESS)}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview and tune a UVC RGB camera with OpenCV properties.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="Probe camera indices.")
    list_parser.add_argument("--max-index", type=int, default=8)
    list_parser.set_defaults(func=list_cameras)

    preview_parser = subparsers.add_parser("preview", help="Open one camera and tune common UVC properties.")
    preview_parser.add_argument("--index", type=int, default=0)
    preview_parser.add_argument("--backend", choices=BACKENDS.keys(), default="any")
    preview_parser.add_argument("--auto-exposure", type=float, default=None)
    preview_parser.add_argument("--exposure", type=float, default=None)
    preview_parser.add_argument("--gain", type=float, default=None)
    preview_parser.add_argument("--brightness", type=float, default=None)
    preview_parser.add_argument("--contrast", type=float, default=None)
    preview_parser.add_argument("--saturation", type=float, default=None)
    preview_parser.add_argument("--exposure-step", type=float, default=1.0)
    preview_parser.add_argument("--gain-step", type=float, default=1.0)
    preview_parser.add_argument("--brightness-step", type=float, default=1.0)
    preview_parser.set_defaults(func=preview)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
