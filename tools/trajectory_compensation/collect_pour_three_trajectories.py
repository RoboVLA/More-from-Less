from __future__ import annotations

import json
import shutil
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
POUR_DIR = WORKSPACE / "Data" / "ai\u751f\u6210\u5012\u6c34" / "\u5012\u6c34"
SUMMARY_DIR = POUR_DIR / "\u4e09\u79cd\u8f68\u8ff9\u6c47\u603b"

SOURCES = [
    (
        "01_\u751f\u6210\u89c6\u9891\u539f\u8f68\u8ff9",
        POUR_DIR / "\u8fd0\u52a8\u7269\u4f53\u8f68\u8ff9_\u53bb\u6c34\u5370RGB_\u5206\u5c42\u6df1\u5ea6",
        "根据 AI 生成倒水视频本身提取的运动物体轨迹，最贴合生成画面中的瓶子落点。",
    ),
    (
        "02_\u771f\u5b9e\u89c6\u9891\u91cd\u5b9a\u5411\u8f68\u8ff9",
        POUR_DIR / "\u771f\u5b9e\u89c6\u9891\u8f68\u8ff9\u91cd\u5b9a\u5411_\u53bb\u6c34\u5370RGB_\u5206\u5c42\u6df1\u5ea6",
        "从真实双臂倒水全局视频提取瓶子/末端执行器轨迹，再重定向到 AI 倒水场景。",
    ),
    (
        "03_\u878d\u5408\u771f\u5b9e\u4e0e\u751f\u6210\u8f68\u8ff9",
        POUR_DIR / "\u878d\u5408\u771f\u5b9e\u4e0e\u751f\u6210\u8f68\u8ff9_\u53bb\u6c34\u5370RGB_\u5206\u5c42\u6df1\u5ea6",
        "融合真实视频路径特征和生成视频物体位置约束，当前作为更均衡的推荐版本。",
    ),
]

COPY_SUFFIXES = {".csv", ".json", ".html", ".mp4", ".png", ".jpg", ".jpeg", ".md"}
COPY_DIRS = {"preview_frames", "pose_preview_frames"}


def copy_subset(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_dir():
            if item.name in COPY_DIRS:
                shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
            continue
        if item.suffix.lower() in COPY_SUFFIXES:
            shutil.copy2(item, dst / item.name)


def trajectory_stats(json_path: Path) -> dict:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    rows = data["trajectory"]
    return {
        "frames": len(rows),
        "x": [min(float(r["x_mm"]) for r in rows), max(float(r["x_mm"]) for r in rows)],
        "y": [min(float(r["y_mm"]) for r in rows), max(float(r["y_mm"]) for r in rows)],
        "z": [min(float(r["z_mm"]) for r in rows), max(float(r["z_mm"]) for r in rows)],
    }


def write_readme(stats: dict[str, dict]) -> None:
    text = """# 倒水三种 6D 轨迹汇总

本目录汇总同一个倒水任务的三种轨迹版本：

1. `01_生成视频原轨迹`
   - 根据 AI 生成倒水视频本身提取运动物体轨迹。
   - 更贴合生成视频画面中的瓶子位置和落点。

2. `02_真实视频重定向轨迹`
   - 从 `Data/真实双臂-倒水/全局视角/全局视角.mp4` 提取瓶子/末端执行器路径。
   - 将真实路径形状重定向到 AI 倒水场景的瓶子运动范围内。
   - Z 采用稳定真实深度尺度，避免瓶子移动时深度值漂移。

3. `03_融合真实与生成轨迹`
   - 融合真实视频轨迹特征与生成视频轨迹特征。
   - 默认真实重定向轨迹权重 55%，运动中段最高约 65%，同时保留生成视频场景约束。
   - Z 固定为稳定真实深度中位值，是当前更均衡的推荐版本。

常用文件：

- `pose_trajectory_web_visualization.html`：网页可视化入口。
- `moving_object_pose_xyzrpy_only.csv`：精简 6D 轨迹，字段为 `frame,time_s,x,y,z,rx,ry,rz`。
- `moving_object_pose_trajectory_xyzrpy.json`：完整轨迹和元数据。
- `*.mp4`：对应版本的轨迹叠加检查视频。

坐标定义：

- `x,y,z` 单位为毫米。
- `rx,ry,rz` 单位为角度。
- 相机坐标系：`+X` 向右，`+Y` 向下，`+Z` 向前。

轨迹范围：

"""
    for name, values in stats.items():
        text += (
            f"- `{name}`：{values['frames']} 帧，"
            f"x={values['x'][0]:.1f}..{values['x'][1]:.1f} mm，"
            f"y={values['y'][0]:.1f}..{values['y'][1]:.1f} mm，"
            f"z={values['z'][0]:.1f}..{values['z'][1]:.1f} mm。\n"
        )
    (SUMMARY_DIR / "README.md").write_text(text, encoding="utf-8")


def write_index() -> None:
    cards = []
    for name, _src, description in SOURCES:
        cards.append(
            f"""
      <section>
        <h2>{name.replace('_', ' ', 1)}</h2>
        <p>{description}</p>
        <a href="./{name}/pose_trajectory_web_visualization.html">打开网页</a>
        <a href="./{name}/moving_object_pose_xyzrpy_only.csv">CSV</a>
        <a href="./{name}/moving_object_pose_trajectory_xyzrpy.json">JSON</a>
      </section>"""
        )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>倒水三种 6D 轨迹汇总</title>
  <style>
    body {{ margin: 0; background: #f4f6f8; color: #17202a; font: 14px/1.5 "Segoe UI", "Microsoft YaHei", Arial, sans-serif; }}
    main {{ max-width: 1080px; margin: 0 auto; padding: 28px 18px 40px; }}
    h1 {{ font-size: 22px; margin: 0 0 8px; }}
    .note {{ color: #657384; margin-bottom: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    section {{ background: #fff; border: 1px solid #d7dde5; border-radius: 8px; padding: 14px; box-shadow: 0 8px 24px rgba(20, 30, 42, 0.07); }}
    h2 {{ font-size: 16px; margin: 0 0 8px; }}
    p {{ color: #657384; min-height: 86px; margin: 0 0 12px; }}
    a {{ display: inline-block; margin: 4px 6px 4px 0; padding: 7px 10px; border: 1px solid #d7dde5; border-radius: 6px; color: #17202a; text-decoration: none; background: #fff; }}
    a:hover {{ background: #f8fafc; border-color: #b8c2d0; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} p {{ min-height: 0; }} }}
  </style>
</head>
<body>
  <main>
    <h1>倒水三种 6D 轨迹汇总</h1>
    <div class="note">打开对应网页可视化，或直接读取 CSV/JSON。坐标为相机坐标系，位置单位 mm，姿态单位 deg。</div>
    <div class="grid">{''.join(cards)}
    </div>
  </main>
</body>
</html>
"""
    (SUMMARY_DIR / "index.html").write_text(html, encoding="utf-8")


def main() -> int:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict] = {}
    for name, src, _description in SOURCES:
        dst = SUMMARY_DIR / name
        copy_subset(src, dst)
        stats[name] = trajectory_stats(dst / "moving_object_pose_trajectory_xyzrpy.json")
    write_readme(stats)
    write_index()
    print(json.dumps({"summary_dir": str(SUMMARY_DIR), "sets": list(stats.keys()), "stats": stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
