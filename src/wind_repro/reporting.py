from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .config import load_config, resolve_project_path
from .utils import read_json


def _plot_history(run_path: Path) -> None:
    history = pd.read_csv(run_path / "history.csv")
    output = run_path / "figures" / "loss.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(history["epoch"], history["train_loss"], label="Train")
    axis.plot(history["epoch"], history["validation_loss"], label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Weighted MSE")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_predictions(run_path: Path, sample_limit: int) -> None:
    with np.load(run_path / "predictions.npz") as archive:
        prediction = archive["prediction"]
        target = archive["target"]
    samples = min(sample_limit, len(prediction))
    output = run_path / "figures" / "speed_prediction.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9.0, 4.2))
    axis.plot(target[:samples, 0, 0], label="Observed", linewidth=1.5)
    axis.plot(prediction[:samples, 0, 0], label="Predicted", linewidth=1.2)
    axis.set_xlabel("Test window")
    axis.set_ylabel("Wind speed")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_horizon_error(run_path: Path, num_turbines: int) -> None:
    with np.load(run_path / "predictions.npz") as archive:
        prediction = archive["prediction"]
        target = archive["target"]
    speed_mae = np.mean(
        np.abs(prediction[..., :num_turbines] - target[..., :num_turbines]),
        axis=(0, 2),
    )
    direction_mae = np.mean(
        np.abs(
            (
                prediction[..., num_turbines:]
                - target[..., num_turbines:]
                + np.pi
            )
            % (2 * np.pi)
            - np.pi
        ),
        axis=(0, 2),
    )
    direction_mae = np.rad2deg(direction_mae)
    output = run_path / "figures" / "horizon_mae.png"
    steps = np.arange(1, len(speed_mae) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    axes[0].bar(steps, speed_mae)
    axes[0].set_title("Wind speed")
    axes[1].bar(steps, direction_mae)
    axes[1].set_title("Wind direction")
    axes[0].set_ylabel("MAE (m/s)")
    axes[1].set_ylabel("Circular MAE (degrees)")
    for axis in axes:
        axis.set_xlabel("Forecast step")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _paper_comparison(
    config: dict[str, Any], metrics: dict[str, Any]
) -> tuple[list[str], str]:
    path = resolve_project_path(
        config.get("evaluation", {}).get(
            "paper_results_path", "reports/paper_results.yaml"
        )
    )
    if not path.exists():
        return [], "Paper result registry was not found."
    with path.open("r", encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)

    dataset = str(config["data"].get("dataset_name", "")).lower()
    horizon = int(config["data"]["horizon"])
    model = str(config["model"].get("name", "dlinear")).lower()
    reference = (
        registry.get("results", {})
        .get(dataset, {})
        .get(horizon, {})
        .get(model)
    )
    if not reference:
        return [], "No matching paper result exists for this dataset/model/horizon."

    rows = []
    for feature in ("speed", "direction"):
        for metric in ("mae", "mse"):
            reproduced = float(metrics[feature][metric])
            paper_value = float(reference[feature][metric])
            difference = (reproduced - paper_value) / paper_value * 100.0
            rows.append(
                f"| {feature} | {metric.upper()} | {paper_value:.4f} | "
                f"{reproduced:.4f} | {difference:+.1f}% |"
            )
    return rows, str(registry["paper"]["metrics_note"])


def make_report(run_dir: str | Path) -> Path:
    run_path = Path(run_dir).resolve()
    config = load_config(run_path / "config.yaml")
    metrics = read_json(run_path / "metrics.json")
    environment = read_json(run_path / "environment.json")
    metadata = read_json(run_path / "data_metadata.json")
    summary = read_json(run_path / "training_summary.json")
    direction_metrics = metrics["direction"]
    direction_mae = direction_metrics.get(
        "mae_rad", direction_metrics["mae"]
    )
    direction_mse = direction_metrics.get(
        "mse_rad2", direction_metrics["mse"]
    )
    direction_rmse = direction_metrics.get(
        "rmse_rad", direction_metrics["rmse"]
    )

    _plot_history(run_path)
    _plot_predictions(
        run_path,
        int(config.get("evaluation", {}).get("prediction_plot_samples", 144)),
    )
    _plot_horizon_error(run_path, int(metadata["num_turbines"]))
    comparison_rows, comparison_note = _paper_comparison(config, metrics)
    horizon_rows = []
    for horizon, horizon_metrics in metrics.get("horizons", {}).items():
        direction = horizon_metrics["direction"]
        horizon_rows.append(
            f"| {horizon} | {horizon_metrics['speed']['mae']:.6f} | "
            f"{horizon_metrics['speed']['mse']:.6f} | "
            f"{horizon_metrics['speed']['rmse']:.6f} | "
            f"{direction.get('mae_rad', direction['mae']):.6f} | "
            f"{direction.get('rmse_rad', direction['rmse']):.6f} |"
        )
    horizon_table = (
        "\n".join(
            [
                "| Horizon (steps) | Speed MAE | Speed MSE | Speed RMSE | "
                "Direction MAE (rad) | Direction RMSE (rad) |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
                *horizon_rows,
            ]
        )
        if horizon_rows
        else "_No horizon-specific metrics were configured._"
    )
    model_name = str(metrics["run"]["model"]).lower()
    if model_name == "mstnet":
        interpretation = (
            "This run evaluates a PyTorch engineering equivalent of the "
            "paper's MST-Net modules. It is not an official-source or "
            "bitwise reproduction. A close numerical comparison with Table "
            "II is meaningful only after matching the dataset release, "
            "chronological coverage, column definitions, normalization, "
            "denoising threshold, and five-seed protocol."
        )
    else:
        interpretation = (
            f"This run evaluates the `{model_name}` engineering baseline. "
            "A close numerical comparison with Table II is meaningful only "
            "after matching the dataset release, chronological coverage, "
            "column definitions, normalization, denoising threshold, and "
            "five-seed protocol."
        )

    comparison = (
        "\n".join(
            [
                "| Output | Metric | Paper | Reproduced | Difference |",
                "| --- | --- | ---: | ---: | ---: |",
                *comparison_rows,
            ]
        )
        if comparison_rows
        else f"_Comparison unavailable: {comparison_note}_"
    )
    report = f"""# Reproduction Report

## Run

- Dataset: `{metrics['run']['dataset']}`
- Model: `{metrics['run']['model']}`
- Forecast horizon: {metrics['run']['horizon']} steps
- Test windows: {metrics['run']['samples']}
- Turbines: {metadata['num_turbines']}
- Data period: {metadata['timestamp_start']} to {metadata['timestamp_end']}
- Device: {environment['device_name']}
- Parameters: {summary['parameters']:,}
- Training time: {summary['training_seconds']:.2f} seconds
- Best checkpoint epoch: {metrics['run']['checkpoint_epoch']}

## Reproduced Metrics

| Output | MAE | MSE | RMSE | MAPE |
| --- | ---: | ---: | ---: | ---: |
| Wind speed | {metrics['speed']['mae']:.6f} m/s | {metrics['speed']['mse']:.6f} | {metrics['speed']['rmse']:.6f} m/s | {metrics['speed']['mape']:.2f}% |
| Wind direction | {direction_mae:.6f} | {direction_mse:.6f} | {direction_rmse:.6f} | N/A |

## Metrics By Forecast Horizon

{horizon_table}

## Paper Comparison

{comparison}

## Figures

![Training history](figures/loss.png)

![Wind-speed prediction](figures/speed_prediction.png)

![MAE by forecast step](figures/horizon_mae.png)

## Interpretation

{interpretation}
Wind-direction MAE, MSE, and RMSE are saved and reported in radians (MSE in
rad^2), matching the paper table convention. The corresponding MAE and RMSE
in degrees are retained in `metrics.json` as auxiliary fields. The metric
protocol is recorded in `metrics.json`; this run uses the paper-style scalar
direction error.
"""
    output = run_path / "report.md"
    output.write_text(report, encoding="utf-8")
    return output
