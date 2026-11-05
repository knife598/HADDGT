"""Convert one official SinD vehicle recording to HDAAGT's mask-aware CSV.

Valid position tokens occupy 1..1023 (0 is padding). SinD raw signal states
0/1/3 (red/green/yellow) are recoded to 1/2/3. Five independent mask columns
encode Zone and four signal-channel availability; they are never inferred from
the category value. Pedestrians are excluded by reading Veh_smoothed_tracks.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


OUT_HEADERS = [
    "Frame", "ID", "BBx", "BBy", "W", "L", "Cls",
    "Tr1", "Tr2", "Tr3", "Tr4", "Zone", "Xreal", "Yreal",
    "ZoneMask", "Tr1Mask", "Tr2Mask", "Tr3Mask", "Tr4Mask",
]

STATE_MAP = {0: 1, 1: 2, 3: 3}  # red, green, yellow; token 0 remains missing/padding


def find_traffic_file(scene_dir: Path) -> Path | None:
    candidates = sorted(scene_dir.glob("*Traffic*.csv"))
    return candidates[0] if candidates else None


def read_traffic(path: Path | None) -> tuple[list[tuple[float, list[int], list[int]]], list[str]]:
    if path is None:
        return [], []
    result: list[tuple[float, list[int], list[int]]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        light_fields = [f for f in fields if "vehicle traffic light" in f.lower()]
        if not light_fields:
            light_fields = [f for f in fields if "traffic light" in f.lower() or "trafficlight" in f.lower()]
        light_fields = light_fields[:4]
        if not light_fields:
            raise ValueError(f"No vehicle traffic-light columns found in {path}")
        for row in reader:
            timestamp = row.get("timestamp(ms)") or row.get("timestamp_ms") or row.get("timestamp")
            try:
                timestamp_ms = float(timestamp)
            except (TypeError, ValueError):
                continue
            values, masks = [], []
            for field in light_fields:
                try:
                    raw = row.get(field, "")
                    if raw is None or not raw.strip():
                        values.append(0)
                        masks.append(0)
                        continue
                    value = int(float(raw))
                except (TypeError, ValueError):
                    values.append(0)
                    masks.append(0)
                    continue
                if value not in STATE_MAP:
                    raise ValueError(f"Unexpected SinD signal state {value} in {path}")
                values.append(STATE_MAP[value])
                masks.append(1)
            result.append((timestamp_ms, values, masks))
    result.sort(key=lambda event: event[0])
    return result, light_fields


def load_rows(vehicle_file: Path, allowed_agent_types: set[str]) -> tuple[list[dict[str, str]], int]:
    with vehicle_file.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    original_count = len(rows)
    if rows and "agent_type" in rows[0]:
        rows = [row for row in rows if row.get("agent_type", "").strip().lower() in allowed_agent_types]
    # Scene_Process() scans the CSV frame by frame.  SinD's source files are
    # commonly track-major (all frames of one vehicle first), so explicitly
    # reorder them to frame-major order before writing the HDAAGT CSV.
    rows.sort(
        key=lambda row: (
            int(float(row["frame_id"])),
            int(float(row["track_id"])),
        )
    )
    return rows, original_count - len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vehicle", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--traffic", type=Path)
    parser.add_argument(
        "--agent-types",
        default="car,truck,bus,motorcycle",
        help="Comma-separated SinD classes to retain; ambiguous tricycles, bicycles and pedestrians are excluded by default.",
    )
    parser.add_argument("--grid-min-x", type=float)
    parser.add_argument("--grid-max-x", type=float)
    parser.add_argument("--grid-min-y", type=float)
    parser.add_argument("--grid-max-y", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed_agent_types = {item.strip().lower() for item in args.agent_types.split(",") if item.strip()}
    rows, excluded_rows = load_rows(args.vehicle, allowed_agent_types)
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

    traffic_events, traffic_fields = read_traffic(args.traffic or find_traffic_file(args.vehicle.parent))
    event_index = 0
    current_values = [0, 0, 0, 0]
    current_masks = [0, 0, 0, 0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        # HDAAGT's loadcsv() reads header=None and assigns Headers itself.
        # Therefore the converted file must not contain a header row.
        for row in rows:
            frame = int(float(row["frame_id"]))
            timestamp_ms = float(row["timestamp_ms"])
            x = float(row["x"])
            y = float(row["y"])
            # Reserve token 0 for padding and preserve the embedding's 1024 entries.
            bx = 1 + round((x - min_x) / (max_x - min_x) * 1022.0)
            by = 1 + round((y - min_y) / (max_y - min_y) * 1022.0)
            bx = max(1, min(1023, bx))
            by = max(1, min(1023, by))
            while event_index < len(traffic_events) and traffic_events[event_index][0] <= timestamp_ms:
                _, values, masks = traffic_events[event_index]
                for light_i, (value, mask) in enumerate(zip(values, masks)):
                    current_values[light_i] = value
                    current_masks[light_i] = mask
                event_index += 1
            lights = current_values[:]
            light_masks = current_masks[:]
            lights += [0] * (4 - len(lights))
            light_masks += [0] * (4 - len(light_masks))
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
                0,  # SinD lane/zone is not mapped in this first round.
                *light_masks,
            ])

    print(f"Wrote {len(rows)} rows to {args.out}")
    print(f"Excluded {excluded_rows} rows outside vehicle classes: {sorted(allowed_agent_types)}")
    print(f"Coordinate bounds: x=[{min_x}, {max_x}], y=[{min_y}, {max_y}]")
    print(f"Vehicle signal columns used: {traffic_fields}")
    print(f"Signal states aligned by timestamp (ms); events: {len(traffic_events)}")

    meta_path = args.out.with_suffix(".meta.json")
    meta = {
        "coordinate_system": "SinD metric coordinates linearly discretized to tokens 1..1023; 0 reserved for padding",
        "grid_size": 1024,
        "grid_min_token": 1,
        "grid_max_token": 1023,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "x_grid_to_m": [(max_x - min_x) / 1022.0, min_x - (max_x - min_x) / 1022.0],
        "y_grid_to_m": [(max_y - min_y) / 1022.0, min_y - (max_y - min_y) / 1022.0],
        "source_vehicle_file": str(args.vehicle),
        "source_traffic_file": str(args.traffic or find_traffic_file(args.vehicle.parent)) if (args.traffic or find_traffic_file(args.vehicle.parent)) else None,
        "traffic_state_mapping": {"raw_0_red": 1, "raw_1_green": 2, "raw_3_yellow": 3, "0_reserved_for_missing": True},
        "traffic_alignment": "forward-fill traffic-light change events by timestamp_ms; unknown before first event",
        "mask_columns": ["ZoneMask", "Tr1Mask", "Tr2Mask", "Tr3Mask", "Tr4Mask"],
        "agent_mask": "one for each CSV vehicle row; Scene_Process writes zero for absent/padded agents",
        "vehicle_only": True,
        "included_agent_types": sorted(allowed_agent_types),
        "excluded_source_rows": excluded_rows,
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(f"Wrote metric metadata to {meta_path}")


if __name__ == "__main__":
    main()
