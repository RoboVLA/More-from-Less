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
DEFAULT_OUTPUT_DIR_NAME = "\u53bb\u6c34\u5370"


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    buf.tofile(str(path))


def write_video_from_pngs(frame_dir: Path, pattern: str, out_path: Path, fps: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
            str(frame_dir / pattern),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "14",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def write_gif(video_path: Path, gif_path: Path, fps: float) -> None:
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps:g},scale=720:-1:flags=lanczos",
            str(gif_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def text_stroke_mask(roi_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    local = ((gray > 145) & (hsv[:, :, 1] < 105)).astype(np.uint8) * 255

    num, labels, stats, _ = cv2.connectedComponentsWithStats(local, 8)
    keep = np.zeros_like(local)
    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        if 5 <= int(area) <= 2500 and h <= 55 and w <= 90:
            keep[labels == idx] = 255
    keep = cv2.dilate(keep, np.ones((5, 5), np.uint8), iterations=1)
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return keep


def build_watermark_masks(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = frame.shape[:2]
    top = np.zeros((h, w), dtype=np.uint8)
    bottom = np.zeros((h, w), dtype=np.uint8)

    # Top-left "AI generated" card. The whole translucent rounded rectangle is
    # removed because masking only text leaves the card background visible.
    tx1 = int(round(w * 0.013))
    ty1 = int(round(h * 0.022))
    tx2 = int(round(w * 0.131))
    ty2 = int(round(h * 0.087))
    cv2.rectangle(top, (tx1, ty1), (tx2, ty2), 255, -1)

    # Bottom-right "Kling AI 3.0" watermark. Only bright text/icon strokes are
    # masked so the robot and cable behind it are not erased as a large block.
    bx1 = int(round(w * 0.758))
    by1 = int(round(h * 0.906))
    bx2 = min(w, int(round(w * 0.998)))
    by2 = min(h, int(round(h * 0.999)))
    roi = frame[by1:by2, bx1:bx2]
    if roi.size:
        bottom[by1:by2, bx1:bx2] = text_stroke_mask(roi)

    return top, bottom


def remove_watermark(frame: np.ndarray, top_radius: int, bottom_radius: int) -> tuple[np.ndarray, np.ndarray]:
    top_mask, bottom_mask = build_watermark_masks(frame)
    cleaned = cv2.inpaint(frame, top_mask, top_radius, cv2.INPAINT_TELEA)
    cleaned = cv2.inpaint(cleaned, bottom_mask, bottom_radius, cv2.INPAINT_TELEA)
    return cleaned, cv2.bitwise_or(top_mask, bottom_mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove fixed AI-platform watermarks from the AI pouring RGB video.")
    parser.add_argument("--video", type=Path, default=None, help="Input AI video.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument("--top-radius", type=int, default=7, help="Inpaint radius for the top-left card.")
    parser.add_argument("--bottom-radius", type=int, default=4, help="Inpaint radius for the bottom-right text.")
    parser.add_argument("--gif", action="store_true", help="Also write a GIF preview.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path.cwd()
    video_path = args.video if args.video else workspace / AI_POUR_DIR / DEFAULT_VIDEO_NAME
    out_dir = args.out_dir if args.out_dir else video_path.parent / DEFAULT_OUTPUT_DIR_NAME
    frame_dir = out_dir / "frames_png"
    mask_dir = out_dir / "watermark_masks"
    preview_dir = out_dir / "preview_frames"
    tmp_dir = workspace / "code" / "gemini_camera" / "tmp" / "remove_watermark"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    if mask_dir.exists():
        shutil.rmtree(mask_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    preview_indices: set[int] = set()
    frame_count_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count_hint > 0:
        preview_indices = {0, frame_count_hint // 2, frame_count_hint - 1}

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cleaned, mask = remove_watermark(frame, args.top_radius, args.bottom_radius)
        imwrite_unicode(frame_dir / f"frame_{frame_count:04d}.png", cleaned)
        if frame_count in preview_indices:
            imwrite_unicode(mask_dir / f"watermark_mask_{frame_count:04d}.png", mask)
            compare = np.hstack(
                [
                    cv2.resize(frame, (width // 2, height // 2), interpolation=cv2.INTER_AREA),
                    cv2.resize(cleaned, (width // 2, height // 2), interpolation=cv2.INTER_AREA),
                ]
            )
            imwrite_unicode(preview_dir / f"before_after_{frame_count:04d}.jpg", compare)
        frame_count += 1
    cap.release()

    out_video = out_dir / f"{video_path.stem}_\u53bb\u6c34\u5370.mp4"
    write_video_from_pngs(frame_dir, "frame_%04d.png", out_video, fps)

    out_gif = None
    if args.gif:
        out_gif = out_dir / f"{video_path.stem}_\u53bb\u6c34\u5370.gif"
        write_gif(out_video, out_gif, min(fps, 15.0))

    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_video": str(video_path),
        "output_video": str(out_video),
        "output_gif": str(out_gif) if out_gif else None,
        "output_frames": str(frame_dir),
        "output_masks": str(mask_dir),
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "top_left_mask": "full translucent card region",
        "bottom_right_mask": "bright text/icon strokes only",
        "method": "OpenCV Telea inpaint over fixed AI-platform watermark regions.",
    }
    (out_dir / "watermark_removal_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"source_video={video_path}")
    print(f"output_dir={out_dir}")
    print(f"output_video={out_video}")
    if out_gif:
        print(f"output_gif={out_gif}")
    print(f"output_frames={frame_dir}")
    print(f"frame_count={frame_count}")


if __name__ == "__main__":
    main()
