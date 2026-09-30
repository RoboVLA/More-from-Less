#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从视频中按设定规则截取图片帧。

用法示例：
1）每隔 30 帧保存 1 张
python extract_frames.py --input /home/ubuntu/Desktop/gaussian-grouping01/data/Astronaut_video/video/input.mp4 --output /home/ubuntu/Desktop/gaussian-grouping01/data/Astronaut_video/input --every-n-frames 30

2）每隔 2 秒保存 1 张
python extract_frames.py --input input.mp4 --output frames --every-n-seconds 2

3）总共均匀截取 20 张
python video_to_img.py --input /home/ubuntu/Desktop/gaussian-grouping01/data/Astronaut_video/video/input.mp4 --output /home/ubuntu/Desktop/gaussian-grouping01/data/Astronaut_video/input --num-frames 100

可选：
--start-sec 5        从第 5 秒开始
--end-sec 20         截到第 20 秒结束
--prefix img         输出文件名前缀
--jpg-quality 95     JPG质量
"""

import argparse
import os
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="从视频中截取不同帧图片")
    parser.add_argument("--input", required=True, help="/home/ubuntu/Desktop/gaussian-grouping01/data/Astronaut_video/video/input.mp4")
    parser.add_argument("--output", default="frames", help="/home/ubuntu/Desktop/gaussian-grouping01/data/Astronaut_video/input")
    parser.add_argument("--prefix", default="frame", help="输出图片名前缀")
    parser.add_argument("--ext", default="jpg", choices=["jpg", "png"], help="输出图片格式")
    parser.add_argument("--jpg-quality", type=int, default=95, help="JPG质量，1~100")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--every-n-frames", type=int, help="每隔多少帧保存一张")
    group.add_argument("--every-n-seconds", type=float, help="每隔多少秒保存一张")
    group.add_argument("--num-frames", type=int, help="从视频中均匀截取多少张")

    parser.add_argument("--start-sec", type=float, default=0.0, help="起始时间（秒）")
    parser.add_argument("--end-sec", type=float, default=None, help="结束时间（秒）")
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def get_video_info(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        raise ValueError("无法读取视频 FPS，请检查视频文件是否损坏。")

    duration = total_frames / fps if total_frames > 0 else 0
    return fps, total_frames, duration


def save_frame(frame, save_path: Path, ext: str, jpg_quality: int):
    if ext == "jpg":
        ok = cv2.imwrite(str(save_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
    else:
        ok = cv2.imwrite(str(save_path), frame)
    return ok


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"输入视频不存在: {input_path}")

    ensure_dir(output_dir)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {input_path}")

    try:
        fps, total_frames, duration = get_video_info(cap)

        start_sec = max(0.0, args.start_sec)
        end_sec = duration if args.end_sec is None else min(args.end_sec, duration)

        if end_sec <= start_sec:
            raise ValueError("结束时间必须大于起始时间。")

        start_frame = int(start_sec * fps)
        end_frame = min(int(end_sec * fps), total_frames - 1)

        # 计算要保存的帧序号集合
        target_frames = []

        if args.every_n_frames is not None:
            if args.every_n_frames <= 0:
                raise ValueError("--every-n-frames 必须大于 0")
            target_frames = list(range(start_frame, end_frame + 1, args.every_n_frames))

        elif args.every_n_seconds is not None:
            if args.every_n_seconds <= 0:
                raise ValueError("--every-n-seconds 必须大于 0")
            step = max(1, int(round(args.every_n_seconds * fps)))
            target_frames = list(range(start_frame, end_frame + 1, step))

        elif args.num_frames is not None:
            if args.num_frames <= 0:
                raise ValueError("--num-frames 必须大于 0")
            span = end_frame - start_frame + 1
            if args.num_frames == 1:
                target_frames = [start_frame]
            else:
                target_frames = [
                    start_frame + int(i * (span - 1) / (args.num_frames - 1))
                    for i in range(args.num_frames)
                ]

        # 去重并排序
        target_frames = sorted(set(target_frames))

        saved = 0
        for idx, frame_id in enumerate(target_frames, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"[警告] 读取第 {frame_id} 帧失败，已跳过。")
                continue

            timestamp = frame_id / fps
            filename = f"{args.prefix}_{idx:04d}_f{frame_id:06d}_t{timestamp:.3f}.{args.ext}"
            save_path = output_dir / filename

            if save_frame(frame, save_path, args.ext, args.jpg_quality):
                saved += 1
                print(f"[已保存] {save_path}")
            else:
                print(f"[失败] 保存图片失败: {save_path}")

        print("\n处理完成")
        print(f"视频路径: {input_path}")
        print(f"输出目录: {output_dir}")
        print(f"视频 FPS: {fps:.3f}")
        print(f"视频总帧数: {total_frames}")
        print(f"视频总时长: {duration:.3f} 秒")
        print(f"截取范围: {start_sec:.3f} ~ {end_sec:.3f} 秒")
        print(f"成功保存: {saved} 张")

    finally:
        cap.release()


if __name__ == "__main__":
    main()