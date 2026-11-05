"""Plot HDAAGT predictions and ground truth for one saved SinD test sample.

This is an inference/visualization utility. It reads an existing checkpoint
and Pickled/<split> files and does not train, regenerate, or modify them.
Run from the HADDGT project directory, for example:

    python tools/visualize_sind_trajectories.py \
        --config configs/config_SinD.yaml \
        --checkpoint Pickled/best_trained_model.pth \
        --split Test --sample-index 0 --output outputs/sind_test_0.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize HDAAGT history, predicted future, and SinD ground truth."
    )
    parser.add_argument("--config", default="configs/config_SinD.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path; defaults to Load_Model_Path in config.")
    parser.add_argument("--split", default="Test", choices=("Train", "Test", "Validation"))
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Zero-based index within the selected Pickled split.")
    parser.add_argument("--agents", default=None,
                        help="Optional comma-separated node indices, e.g. 0,3,7.")
    parser.add_argument("--focus-agent", type=int, default=None,
                        help="Plot only one node and print its exact endpoint errors.")
    parser.add_argument("--selection", choices=("moving", "node"), default="moving",
                        help="Automatic agent selection: moving agents (default) or lowest node indices.")
    parser.add_argument("--max-agents", type=int, default=8,
                        help="Maximum valid agents to draw when --agents is omitted.")
    parser.add_argument("--metadata", default=None,
                        help="Optional SinD grid-to-meter metadata JSON path.")
    parser.add_argument("--device", default=None,
                        help="Override config device, for example cpu or cuda:0.")
    parser.add_argument("--output", default=None,
                        help="Output PNG path; defaults to outputs/sind_<split>_<index>.png.")
    return parser.parse_args()


def load_metadata(config: dict, override: str | None) -> dict | None:
    if override:
        path = Path(override)
    else:
        csv_path = Path(config["detection_path"])
        path = csv_path.with_suffix(".meta.json")
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if "x_grid_to_m" not in metadata or "y_grid_to_m" not in metadata:
        raise ValueError(
            f"{path} lacks x_grid_to_m/y_grid_to_m; cannot safely label axes in meters."
        )
    print(f"Using coordinate metadata: {path}")
    return metadata


def grid_to_world(xy, metadata):
    """Map [..., (x_grid,y_grid)] to SinD meters using the saved affine map."""
    import numpy as np

    result = np.empty_like(xy, dtype=float)
    ax, bx = metadata["x_grid_to_m"]
    ay, by = metadata["y_grid_to_m"]
    result[..., 0] = xy[..., 0] * ax + bx
    result[..., 1] = xy[..., 1] * ay + by
    return result


def decode_top5(logits):
    """Match the project's current top-5 probability-weighted coordinate decoder."""
    import torch

    probabilities = logits.softmax(dim=-1)
    values, indices = torch.topk(probabilities, k=5, dim=-1)
    return (indices * values).sum(dim=-1) / values.sum(dim=-1)


def main() -> None:
    args = parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        import numpy as np
        import torch
        import yaml
    except ImportError as exc:
        raise SystemExit(
            f"Missing visualization/runtime dependency: {exc}. "
            "Install matplotlib in the model's Python environment."
        ) from exc

    from models.HDAAGT import HDAAGT
    from train_test import prep_model_input

    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"Config file not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["NFeatures"] = len(config["Columns_to_keep"])
    config["input_size"] = config["NFeatures"] + 9 + 8
    config["output_size"] = len(config["xy_indx"])
    requested_device = args.device or str(config.get("device", "cpu"))
    if requested_device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(requested_device)
    else:
        if requested_device.startswith("cuda"):
            print(f"CUDA unavailable for {requested_device}; using CPU.")
        device = torch.device("cpu")
    config["device"] = device

    raw_sos = torch.tensor(config["sos"], dtype=torch.float32)
    raw_eos = torch.tensor(config["eos"], dtype=torch.float32)
    config["sos"] = torch.cat((raw_sos, torch.zeros(17))).repeat(config["Nnodes"], 1).to(device)
    config["eos"] = torch.cat((raw_eos, torch.zeros(17))).repeat(config["Nnodes"], 1).to(device)

    checkpoint_path = Path(args.checkpoint or config["Load_Model_Path"])
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    split_dir = PROJECT_ROOT / "Pickled" / args.split
    required = [split_dir / "Scene.pt", split_dir / "Target.pt", split_dir / "Adj_Mat_Scene.pt"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing Pickled test data files:\n  " + "\n  ".join(missing))

    scene_all = torch.load(required[0], map_location="cpu", weights_only=True)
    target_all = torch.load(required[1], map_location="cpu", weights_only=True)
    adj_all = torch.load(required[2], map_location="cpu", weights_only=True)
    count = scene_all.shape[0]
    if not 0 <= args.sample_index < count:
        raise SystemExit(f"sample-index {args.sample_index} out of range; split has {count} samples.")
    if scene_all.shape[2] != config["Nnodes"]:
        raise SystemExit(
            f"Pickled nodes={scene_all.shape[2]} but config Nnodes={config['Nnodes']}; "
            "use the matching checkpoint/config/Pickled set."
        )

    sample_scene = scene_all[args.sample_index:args.sample_index + 1].to(device)
    sample_target = target_all[args.sample_index:args.sample_index + 1].to(device)
    sample_adj = adj_all[args.sample_index:args.sample_index + 1].to(device)

    model = HDAAGT(config).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        model_scene, scene_mask, model_adj, model_target = prep_model_input(
            sample_scene, sample_adj, sample_target, config["sos"], config["eos"], config
        )
        logits = model(model_scene, scene_mask, model_adj)[:, 1:-1]
        pred_grid = decode_top5(logits)[0].cpu().numpy()  # [future, nodes, xy]

    print(
        f"Sample fingerprint: Scene sum={float(sample_scene.sum()):.3f}, "
        f"Target sum={float(sample_target.sum()):.3f}, "
        f"Adj sum={float(sample_adj.sum()):.3f}"
    )
    if args.sample_index > 0:
        previous_scene = scene_all[args.sample_index - 1]
        previous_target = target_all[args.sample_index - 1]
        print(
            f"Difference from sample {args.sample_index - 1}: "
            f"Scene max={float((scene_all[args.sample_index] - previous_scene).abs().max()):.3f}, "
            f"Target max={float((target_all[args.sample_index] - previous_target).abs().max()):.3f}"
        )

    hist_grid = sample_scene[0, :, :, :2].cpu().numpy()  # [history, nodes, xy]
    gt_grid = sample_target[0, :, :, :2].cpu().numpy()  # [future, nodes, xy]
    valid_hist = hist_grid[..., 0] != 0
    valid_future = gt_grid[..., 0] != 0

    if args.focus_agent is not None:
        if args.focus_agent < 0 or args.focus_agent >= config["Nnodes"]:
            raise SystemExit(f"focus-agent must be in [0,{config['Nnodes'] - 1}]")
        agent_indices = [args.focus_agent]
    elif args.agents:
        agent_indices = [int(value.strip()) for value in args.agents.split(",") if value.strip()]
        invalid = [idx for idx in agent_indices if idx < 0 or idx >= config["Nnodes"]]
        if invalid:
            raise SystemExit(f"Agent indices out of range [0,{config['Nnodes'] - 1}]: {invalid}")
        agent_indices = [idx for idx in agent_indices if valid_hist[:, idx].any() and valid_future[:, idx].any()]
    else:
        candidates = [idx for idx in range(config["Nnodes"])
                     if valid_hist[:, idx].any() and valid_future[:, idx].any()]
        if args.selection == "moving":
            def movement_score(idx):
                valid = valid_future[:, idx]
                positions = gt_grid[valid, idx]
                if len(positions) < 2:
                    return 0.0
                return float(np.linalg.norm(positions[-1] - positions[0]))

            candidates.sort(key=movement_score, reverse=True)
        agent_indices = candidates[:max(0, args.max_agents)]
    if not agent_indices:
        raise SystemExit("This sample has no selected agent with both history and future ground truth.")

    metadata = load_metadata(config, args.metadata)
    to_plot = grid_to_world if metadata else (lambda points, _: points.astype(float))
    axis_unit = "m" if metadata else "grid units"
    if not metadata:
        print("No metric metadata found; plotting discrete grid coordinates (not meters).")

    shown_ade_sum = shown_ade_count = shown_fde_sum = shown_fde_count = 0.0
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for order, agent in enumerate(agent_indices):
        color = colors(order % 10)
        h = hist_grid[:, agent, :]
        gt = gt_grid[:, agent, :]
        pred = pred_grid[:, agent, :]
        h = np.where(valid_hist[:, agent, None], h, np.nan)
        gt = np.where(valid_future[:, agent, None], gt, np.nan)
        # Predictions exist for the whole future horizon. Keep them visible
        # even when the ground-truth track has missed frames.
        h = to_plot(h, metadata)
        gt = to_plot(gt, metadata)
        pred = to_plot(pred, metadata)

        ax.plot(h[:, 0], h[:, 1], color=color, linewidth=1.6, alpha=0.75)
        ax.plot(gt[:, 0], gt[:, 1], color=color, linewidth=2.0,
                marker=".", markersize=4)
        ax.plot(pred[:, 0], pred[:, 1], color=color, linewidth=2.0,
                linestyle="--", marker=".", markersize=4)
        h_ok = np.isfinite(h).all(axis=1)
        gt_ok = np.isfinite(gt).all(axis=1)
        if h_ok.any():
            ax.scatter(h[h_ok, 0][-1], h[h_ok, 1][-1], color=color, marker="o", s=24, zorder=4)
            ax.annotate(f"N{agent}", (h[h_ok, 0][-1], h[h_ok, 1][-1]), xytext=(3, 3),
                        textcoords="offset points", fontsize=8, color=color)
        if gt_ok.any():
            last_gt_idx = np.flatnonzero(gt_ok)[-1]
            ax.scatter(gt[last_gt_idx, 0], gt[last_gt_idx, 1], color=color,
                       marker="x", s=42, zorder=5)
        ax.scatter(pred[-1, 0], pred[-1, 1], color=color, marker="*", s=58, zorder=5)

        print(
            f"node {agent}: history points={int(valid_hist[:, agent].sum())}, "
            f"valid GT future points={int(valid_future[:, agent].sum())}, "
            f"predicted future points={len(pred_grid)}, "
            f"history_last_grid={hist_grid[-1, agent].round(2).tolist()}, "
            f"gt_first_grid={gt_grid[0, agent].round(2).tolist()}, "
            f"pred_first_grid={pred_grid[0, agent].round(2).tolist()}, "
            f"gt_last_grid={gt_grid[-1, agent].round(2).tolist()}, "
            f"pred_last_grid={pred_grid[-1, agent].round(2).tolist()}"
        )

        valid = valid_future[:, agent]
        if valid.any():
            delta_grid = pred_grid[valid, agent] - gt_grid[valid, agent]
            dist_grid = np.sqrt((delta_grid ** 2).sum(axis=-1))
            shown_ade_sum += float(dist_grid.sum())
            shown_ade_count += int(valid.sum())
            if valid_future[-1, agent]:
                fde_grid = float(np.linalg.norm(pred_grid[-1, agent] - gt_grid[-1, agent]))
                shown_fde_sum += fde_grid
                shown_fde_count += 1

            if metadata:
                pred_agent_m = to_plot(pred_grid[:, agent], metadata)
                gt_agent_m = to_plot(gt_grid[:, agent], metadata)
                dist_m = np.linalg.norm(pred_agent_m[valid] - gt_agent_m[valid], axis=-1)
                final_text = "not evaluated (GT missing at final horizon)"
                if valid_future[-1, agent]:
                    final_text = f"{np.linalg.norm(pred_agent_m[-1] - gt_agent_m[-1]):.3f} m"
                print(
                    f"node {agent}: per-agent ADE={dist_m.mean():.3f} m, "
                    f"FDE={final_text}, "
                    f"max point error={dist_m.max():.3f} m"
                )

    legend_items = [
        Line2D([0], [0], color="0.25", linewidth=1.6, label="History"),
        Line2D([0], [0], color="0.25", linewidth=2, label="Ground truth future"),
        Line2D([0], [0], color="0.25", linewidth=2, linestyle="--", label="Predicted future"),
        Line2D([0], [0], color="0.25", marker="o", linestyle="None", label="Observed endpoint"),
        Line2D([0], [0], color="0.25", marker="x", linestyle="None", label="Last valid GT point"),
        Line2D([0], [0], color="0.25", marker="*", linestyle="None", label="Prediction horizon endpoint"),
    ]
    ax.legend(handles=legend_items, loc="best", fontsize=8)
    sample_ade_grid = shown_ade_sum / max(1, shown_ade_count)
    sample_fde_grid = shown_fde_sum / max(1, shown_fde_count)
    if metadata:
        # Use the same affine metric map to calculate the displayed-agent summary in meters.
        shown_ade_m_sum = shown_fde_m_sum = 0.0
        shown_ade_m_count = shown_fde_m_count = 0
        for agent in agent_indices:
            valid = valid_future[:, agent]
            if not valid.any():
                continue
            pred_m = to_plot(pred_grid[:, agent], metadata)
            gt_m = to_plot(gt_grid[:, agent], metadata)
            d = np.linalg.norm(pred_m[valid] - gt_m[valid], axis=-1)
            shown_ade_m_sum += float(d.sum())
            shown_ade_m_count += int(valid.sum())
            if valid_future[-1, agent]:
                shown_fde_m_sum += float(np.linalg.norm(pred_m[-1] - gt_m[-1]))
                shown_fde_m_count += 1
        subtitle = (
            f"Shown-agent ADE {shown_ade_m_sum/max(1, shown_ade_m_count):.2f} m; "
            f"FDE {shown_fde_m_sum/max(1, shown_fde_m_count):.2f} m"
        )
    else:
        subtitle = f"Shown-agent ADE {sample_ade_grid:.1f} grid; FDE {sample_fde_grid:.1f} grid"

    ax.set_title(f"SinD {args.split} sample {args.sample_index} — {len(agent_indices)} agents\n{subtitle}")
    ax.set_xlabel(f"X ({axis_unit})")
    ax.set_ylabel(f"Y ({axis_unit})")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)

    output = Path(args.output) if args.output else PROJECT_ROOT / "outputs" / f"sind_{args.split.lower()}_{args.sample_index:05d}.png"
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Split samples: {count}; selected sample: {args.sample_index}")
    print(f"Agent selection: {'explicit --agents' if args.agents else args.selection}")
    print(f"Plotted node indices: {agent_indices}")
    print(f"Displayed-agent ADE/FDE: {subtitle}")
    print(f"Saved figure: {output}")


if __name__ == "__main__":
    main()
