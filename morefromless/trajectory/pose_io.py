from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


POSE_FIELDS = ("frame", "time_s", "x", "y", "z", "rx", "ry", "rz")


@dataclass(slots=True)
class PoseSample:
    frame: int
    time_s: float
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float


def _coerce_pose(row: dict[str, Any], index: int) -> PoseSample:
    return PoseSample(
        frame=int(float(row.get("frame", index))),
        time_s=float(row.get("time_s", row.get("timestamp", index))),
        x=float(row["x"]),
        y=float(row["y"]),
        z=float(row["z"]),
        rx=float(row.get("rx", 0.0)),
        ry=float(row.get("ry", 0.0)),
        rz=float(row.get("rz", 0.0)),
    )


def _extract_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if not isinstance(payload, dict):
        raise ValueError("JSON trajectory must be a list or an object.")
    for key in (
        "trajectory",
        "poses",
        "samples",
        "frames",
        "moving_object_pose_trajectory",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value]
    if all(field in payload for field in ("x", "y", "z")):
        return [payload]
    raise ValueError("Cannot find pose records in JSON trajectory.")


def load_pose_trajectory(path: str | Path) -> list[PoseSample]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        return [_coerce_pose(row, i) for i, row in enumerate(rows)]

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return [_coerce_pose(row, i) for i, row in enumerate(_extract_records(payload))]

    raise ValueError(f"Unsupported trajectory format: {path.suffix}")


def save_pose_csv(path: str | Path, samples: Iterable[PoseSample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=POSE_FIELDS)
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))


def save_pose_json(path: str | Path, samples: Iterable[PoseSample], metadata: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "trajectory": [asdict(sample) for sample in samples],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def pose_bounds(samples: Iterable[PoseSample]) -> dict[str, tuple[float, float]]:
    values = list(samples)
    bounds: dict[str, tuple[float, float]] = {}
    for key in ("x", "y", "z", "rx", "ry", "rz"):
        column = [float(getattr(sample, key)) for sample in values]
        if column:
            bounds[key] = (min(column), max(column))
    return bounds

