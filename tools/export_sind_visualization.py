"""Export HDAAGT SinD predictions for the interactive Chinese HTML viewer."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export HDAAGT trajectories for the Chinese HTML viewer.")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split", default="Test", choices=("Train", "Test", "Validation"))
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--all", action="store_true", help="Export every sample in the split.")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--count", type=int, default=None)
    p.add_argument("--metadata", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--output", default=None, help="HTML path, or folder with --all.")
    return p.parse_args()


def load_metadata(config: dict, override: str | None) -> dict | None:
    path = Path(override) if override else Path(config["detection_path"]).with_suffix(".meta.json")
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        value = json.load(f)
    if "x_grid_to_m" not in value or "y_grid_to_m" not in value:
        raise ValueError(f"{path} lacks x_grid_to_m/y_grid_to_m")
    print(f"Using coordinate metadata: {path}")
    return value


def decode_top5(logits):
    import torch
    probabilities = logits.softmax(dim=-1)
    values, indices = torch.topk(probabilities, k=5, dim=-1)
    return (indices * values).sum(dim=-1) / values.sum(dim=-1)


def to_metric(points, metadata):
    import numpy as np
    if metadata is None:
        return points.astype(float)
    result = np.empty_like(points, dtype=float)
    ax, bx = metadata["x_grid_to_m"]
    ay, by = metadata["y_grid_to_m"]
    result[..., 0] = points[..., 0] * ax + bx
    result[..., 1] = points[..., 1] * ay + by
    return result


def finite_point(point) -> list[float] | None:
    if point is None or not all(math.isfinite(float(v)) for v in point):
        return None
    return [round(float(point[0]), 6), round(float(point[1]), 6)]


def prepare_config(config: dict, torch, device):
    config = dict(config)
    config["NFeatures"] = len(config["Columns_to_keep"])
    config["input_size"] = config["NFeatures"] + 9 + 8
    config["output_size"] = len(config["xy_indx"])
    config["device"] = device
    raw_sos = torch.tensor(config["sos"], dtype=torch.float32)
    raw_eos = torch.tensor(config["eos"], dtype=torch.float32)
    config["sos"] = torch.cat((raw_sos, torch.zeros(17))).repeat(config["Nnodes"], 1).to(device)
    config["eos"] = torch.cat((raw_eos, torch.zeros(17))).repeat(config["Nnodes"], 1).to(device)
    return config


def serialise_sample(scene, target, prediction, metadata, split: str, sample_index: int) -> dict[str, Any]:
    import numpy as np
    scene_grid = scene[:, :, :2].astype(float)
    target_grid = target[:, :, :2].astype(float)
    prediction_grid = prediction.astype(float)
    nodes = []
    for node_id in range(scene_grid.shape[1]):
        hist_valid = np.isfinite(scene_grid[:, node_id, 0]) & (scene_grid[:, node_id, 0] != 0)
        gt_valid = np.isfinite(target_grid[:, node_id, 0]) & (target_grid[:, node_id, 0] != 0)
        hist_metric = to_metric(np.where(hist_valid[:, None], scene_grid[:, node_id], np.nan), metadata)
        gt_metric = to_metric(np.where(gt_valid[:, None], target_grid[:, node_id], np.nan), metadata)
        pred_metric = to_metric(prediction_grid[:, node_id], metadata)
        error = np.linalg.norm(pred_metric - gt_metric, axis=1)
        valid_errors = error[gt_valid]
        ade = float(valid_errors.mean()) if valid_errors.size else None
        fde = float(np.linalg.norm(pred_metric[-1] - gt_metric[-1])) if gt_valid[-1] else None
        max_error = float(valid_errors.max()) if valid_errors.size else None
        nodes.append({
            "node_id": node_id,
            "history": [finite_point(x) for x in hist_metric],
            "ground_truth": [finite_point(x) for x in gt_metric],
            "prediction": [finite_point(x) for x in pred_metric],
            "valid_ground_truth": [bool(x) for x in gt_valid],
            "ade": round(ade, 6) if ade is not None else None,
            "fde": round(fde, 6) if fde is not None else None,
            "max_error": round(max_error, 6) if max_error is not None else None,
        })
    return {
        "split": split, "sample_index": sample_index,
        "coordinate_unit": "m" if metadata else "grid",
        "grid_to_metric": metadata, "history_steps": int(scene_grid.shape[0]),
        "future_steps": int(target_grid.shape[0]), "nodes": nodes,
        "fingerprint": {"scene_sum": round(float(np.nansum(scene)), 6),
                        "target_sum": round(float(np.nansum(target)), 6)},
    }


def main() -> None:
    args = parse_args()
    try:
        import torch
        import yaml
    except ImportError as exc:
        raise SystemExit(f"Missing model dependency: {exc}") from exc
    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"Config file not found: {config_path}")
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    requested = args.device or str(config.get("device", "cpu"))
    device = torch.device(requested if requested.startswith("cuda") and torch.cuda.is_available() else "cpu")
    config = prepare_config(config, torch, device)
    checkpoint_path = Path(args.checkpoint or config["Load_Model_Path"])
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    split_dir = PROJECT_ROOT / "Pickled" / args.split
    required = [split_dir / "Scene.pt", split_dir / "Target.pt", split_dir / "Adj_Mat_Scene.pt"]
    missing = [str(x) for x in required if not x.is_file()]
    if missing:
        raise SystemExit("Missing Pickled files:\n  " + "\n  ".join(missing))
    scene_all = torch.load(required[0], map_location="cpu", weights_only=True)
    target_all = torch.load(required[1], map_location="cpu", weights_only=True)
    adj_all = torch.load(required[2], map_location="cpu", weights_only=True)
    count = int(scene_all.shape[0])
    if scene_all.shape[2] != config["Nnodes"]:
        raise SystemExit(f"Pickled nodes={scene_all.shape[2]} but config Nnodes={config['Nnodes']}")
    from models.HDAAGT import HDAAGT
    from train_test import prep_model_input
    model = HDAAGT(config).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    metadata = load_metadata(config, args.metadata)
    template_path = PROJECT_ROOT / "tools" / "sind_trajectory_viewer.html"
    template = template_path.read_text(encoding="utf-8")

    def export_one(index: int):
        scene = scene_all[index:index + 1].to(device)
        target = target_all[index:index + 1].to(device)
        adj = adj_all[index:index + 1].to(device)
        with torch.no_grad():
            model_scene, scene_mask, model_adj, _ = prep_model_input(
                scene, adj, target, config["sos"], config["eos"], config)
            prediction = decode_top5(model(model_scene, scene_mask, model_adj)[:, 1:-1])[0].cpu().numpy()
        return serialise_sample(scene[0, :, :, :2].cpu().numpy(),
                                target[0, :, :, :2].cpu().numpy(),
                                prediction, metadata, args.split, index)

    if not args.all:
        if not 0 <= args.sample_index < count:
            raise SystemExit(f"sample-index out of range; split has {count} samples")
        data = export_one(args.sample_index)
        output = Path(args.output) if args.output else PROJECT_ROOT / "outputs" / f"sind_{args.split.lower()}_{args.sample_index:05d}.html"
        if not output.is_absolute(): output = PROJECT_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        bootstrap = "window.__HDAAGT_DATA__ = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "; window.__HDAAGT_MANIFEST_URL__ = null;"
        output.write_text(template.replace("/* __HDAAGT_BOOTSTRAP__ */", bootstrap), encoding="utf-8")
        print(f"Generated interactive HTML: {output}")
        return

    start = max(0, args.start_index)
    end = count if args.count is None else min(count, start + max(0, args.count))
    if start >= end: raise SystemExit(f"No samples selected: {start}:{end}")
    output_dir = Path(args.output) if args.output else PROJECT_ROOT / "outputs" / f"sind_{args.split.lower()}_viewer"
    if not output_dir.is_absolute(): output_dir = PROJECT_ROOT / output_dir
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index in range(start, end):
        filename = f"sample_{index:05d}.json"
        (samples_dir / filename).write_text(json.dumps(export_one(index), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        manifest.append({"sample_index": index, "file": f"samples/{filename}"})
        print(f"Exported sample {index + 1 - start}/{end - start}")
    (output_dir / "manifest.json").write_text(json.dumps({"split": args.split, "samples": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    bootstrap = "window.__HDAAGT_DATA__ = null; window.__HDAAGT_MANIFEST_URL__ = 'manifest.json';"
    (output_dir / "index.html").write_text(template.replace("/* __HDAAGT_BOOTSTRAP__ */", bootstrap), encoding="utf-8")
    print(f"Generated interactive viewer directory: {output_dir}")


if __name__ == "__main__":
    main()

