"""Convert one SinD vehicle scene to the CSV layout expected by HDAAGT.

The converter keeps the original SinD metric coordinates in Xreal/Yreal and
maps x/y to a 0..1023 integer grid for HDAAGT's categorical position tokens.
It writes one self-contained scene at a time; scenes must not be concatenated
because their coordinate systems and road layouts differ.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


OUT_HEADERS = [
    "Frame", "ID", "BBx", "BBy", "W", "L", "Cls",
    "Tr1", "Tr2", "Tr3", "Tr4", "Zone", "Xreal", "Yreal",
]


def find_traffic_file(scene_dir: Path) -> Path | None:
    candidates = sorted(scene_dir.glob("*Traffic*.csv"))
    return candidates[0] if candidates else None


def read_traffic(path: Path | None) -> dict[int, list[int]]:
    if path is None:
        return {}
    result: dict[int, list[int]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        light_fields = [
            f for f in fields
            if "traffic light" in f.lower() or "trafficlight" in f.lower()
        ]
        for row in reader:
            raw_frame = row.get("RawFrameID", "")
            try:
                frame = int(float(raw_frame))
            except (TypeError, ValueError):
                continue
            values = []
            for field in light_fields[:4]:
                try:
                    value = int(float(row.get(field, 0) or 0))
                except (TypeError, ValueError):
                    value = 0
                # SinD uses 0 for unavailable; HDAAGT accepts 0 as padding.
                values.append(max(0, min(value, 8)))
            result[frame] = (values + [0, 0, 0, 0])[:4]
    return result


def load_rows(vehicle_file: Path) -> list[dict[str, str]]:
    with vehicle_file.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    # Scene_Process() scans the CSV frame by frame.  SinD's source files are
    # commonly track-major (all frames of one vehicle first), so explicitly
    # reorder them to frame-major order before writing the HDAAGT CSV.
    rows.sort(
        key=lambda row: (
            int(float(row["frame_id"])),
            int(float(row["track_id"])),
        )
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vehicle", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--traffic", type=Path)
    parser.add_argument("--grid-min-x", type=float)
    parser.add_argument("--grid-max-x", type=float)
    parser.add_argument("--grid-min-y", type=float)
    parser.add_argument("--grid-max-y", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.vehicle)
    if not rows:
        raise SystemExit("Vehicle CSV is empty")

    xs = [float(row["x"]) for row in rows]
    ys = [float(row["y"]) for row in rows]
    min_x = min(xs) if args.grid_min_x is None else args.grid_min_x
    max_x = max(xs) if args.grid_max_x is None else args.grid_max_x
    min_y = min(ys) if args.grid_min_y is None else args.grid_min_y
    max_y = max(ys) if args.grid_max_y is None else args.grid_max_y
    if max_x <= min_x or max_y <= min_y:
        raise SystemExit("Coordinate range must be non-zero")

    traffic = read_traffic(args.traffic or find_traffic_file(args.vehicle.parent))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        # HDAAGT's loadcsv() reads header=None and assigns Headers itself.
        # Therefore the converted file must not contain a header row.
        for row in rows:
            frame = int(float(row["frame_id"]))
            x = float(row["x"])
            y = float(row["y"])
            bx = round((x - min_x) / (max_x - min_x) * 1023.0)
            by = round((y - min_y) / (max_y - min_y) * 1023.0)
            bx = max(0, min(1023, bx))
            by = max(0, min(1023, by))
            lights = traffic.get(frame, [0, 0, 0, 0])
            writer.writerow([
                frame,
                int(float(row["track_id"])),
                bx,
                by,
                float(row.get("width", 0) or 0),
                float(row.get("length", 0) or 0),
                1,
                *lights,
                0,
                x,
                y,
            ])

    print(f"Wrote {len(rows)} rows to {args.out}")
    print(f"Coordinate bounds: x=[{min_x}, {max_x}], y=[{min_y}, {max_y}]")
    print(f"Traffic rows matched: {sum(1 for r in rows if int(float(r['frame_id'])) in traffic)}")

    meta_path = args.out.with_suffix(".meta.json")
    meta = {
        "coordinate_system": "SinD metric coordinates linearly discretized to 0..1023",
        "grid_size": 1024,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "source_vehicle_file": str(args.vehicle),
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(f"Wrote metric metadata to {meta_path}")


if __name__ == "__main__":
    main()
