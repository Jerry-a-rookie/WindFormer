from __future__ import annotations

import csv
import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import project_root
from .data import WindWindowDataset, inverse_transform, load_prepared, make_dataset
from .engine import _build_loss_context, _run_epoch
from .metrics import split_feature_metrics
from .models import build_model
from .utils import choose_device, read_json, set_seed, write_json


DEFAULT_NOISE_LEVELS = (0.10, 0.20, 0.30)

CONDITION_NAMES = (
    "normal",
    "weak_period",
    "extreme_0",
    "extreme_1_2",
    "extreme_3_5",
    "extreme_gt5",
    "direction_shear_low",
    "direction_shear_high",
    "missing_10pct",
)

CONDITION_ALIASES = {
    "weak": ("weak_period",),
    "extreme": (
        "extreme_0",
        "extreme_1_2",
        "extreme_3_5",
        "extreme_gt5",
    ),
    "direction_shear": (
        "direction_shear_low",
        "direction_shear_high",
    ),
    "shear": (
        "direction_shear_low",
        "direction_shear_high",
    ),
    "missing": ("missing_10pct",),
}


def format_noise_condition(level: float) -> str:
    """Return the stable condition name used in output files."""
    return f"noise_{float(level):g}"


def normalize_noise_levels(levels: Iterable[float]) -> tuple[float, ...]:
    normalized = tuple(sorted({round(float(level), 8) for level in levels}))
    if not normalized or any(level <= 0.0 for level in normalized):
        raise ValueError("Noise levels must contain positive numbers.")
    return normalized


def _deep_update(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


def _benchmark_config(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    output = deepcopy(config)
    overrides = config.get("benchmark", {}).get("models", {}).get(model_name)
    if overrides:
        _deep_update(output, overrides)
    output.pop("benchmark", None)
    output.setdefault("model", {})["name"] = model_name
    return output


def _feature_layout(prepared: dict[str, Any]) -> tuple[str, int]:
    metadata = prepared["metadata"]
    n = int(metadata["num_turbines"])
    representation = str(metadata.get("direction_representation", "scalar"))
    return representation, n


def _batch_windows(dataset: WindWindowDataset, indices: np.ndarray) -> np.ndarray:
    lookback = int(dataset.lookback)
    return np.stack(
        [dataset.values[start : start + lookback] for start in indices],
        axis=0,
    ).astype(np.float32, copy=False)


def _window_targets(dataset: WindWindowDataset, indices: np.ndarray) -> np.ndarray:
    lookback = int(dataset.lookback)
    horizon = int(dataset.horizon)
    return np.stack(
        [
            dataset.values[
                start + lookback : start + lookback + horizon
            ]
            for start in indices
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _direction_from_features(
    features: np.ndarray, prepared: dict[str, Any]
) -> np.ndarray:
    representation, n = _feature_layout(prepared)
    if representation == "circular":
        sine = features[..., n : 2 * n]
        cosine = features[..., 2 * n : 3 * n]
        return np.arctan2(sine, cosine) % (2.0 * np.pi)
    return (
        features[..., n:]
        * np.asarray(prepared["direction_std"], dtype=np.float32)[None, None, :]
        + np.asarray(prepared["direction_mean"], dtype=np.float32)[
            None, None, :
        ]
    )


def _spectral_peak_ratio(speed_windows: np.ndarray) -> np.ndarray:
    values = speed_windows - speed_windows.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(values, axis=1)
    power = np.abs(spectrum[:, 1:, :]) ** 2
    total = power.sum(axis=(1, 2))
    peak = power.max(axis=(1, 2))
    return (peak / np.maximum(total, 1e-12)).astype(np.float32)


def _direction_shear(direction_windows: np.ndarray) -> np.ndarray:
    difference = (
        direction_windows[:, 1:] - direction_windows[:, :-1] + np.pi
    ) % (2.0 * np.pi) - np.pi
    return np.mean(np.abs(difference), axis=(1, 2)).astype(np.float32)


def _extreme_counts(
    speed_windows: np.ndarray, threshold: float
) -> np.ndarray:
    delta = np.abs(np.diff(speed_windows, axis=1))
    return (delta > threshold).sum(axis=(1, 2)).astype(np.int32)


def _collect_window_scores(
    prepared: dict[str, Any],
    split: str,
    lookback: int,
    horizon: int,
    batch_size: int,
    extreme_threshold: float | None = None,
) -> dict[str, np.ndarray]:
    dataset = make_dataset(prepared, split, lookback, horizon)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    spectral: list[np.ndarray] = []
    shear: list[np.ndarray] = []
    extreme_counts: list[np.ndarray] = []
    for features, _ in loader:
        values = features.numpy()
        representation, n = _feature_layout(prepared)
        speed = values[..., :n]
        spectral.append(_spectral_peak_ratio(speed))
        directions = _direction_from_features(values, prepared)
        shear.append(_direction_shear(directions))
        if extreme_threshold is not None:
            extreme_counts.append(_extreme_counts(speed, extreme_threshold))
    if not spectral:
        raise ValueError(f"No windows were created for split '{split}'.")
    return {
        "spectral_peak_ratio": np.concatenate(spectral),
        "direction_shear": np.concatenate(shear),
        "starts": dataset.starts.copy(),
        "extreme_count": (
            np.concatenate(extreme_counts)
            if extreme_counts
            else np.zeros(len(np.concatenate(spectral)), dtype=np.int32)
        ),
    }


def _condition_masks(
    prepared: dict[str, Any],
    lookback: int,
    horizon: int,
    batch_size: int,
    noise_levels: Iterable[float] = DEFAULT_NOISE_LEVELS,
) -> tuple[dict[str, np.ndarray], dict[str, Any], pd.DataFrame]:
    noise_levels = normalize_noise_levels(noise_levels)
    train_scores = _collect_window_scores(
        prepared, "train", lookback, horizon, batch_size
    )
    train_speed = np.asarray(prepared["train_speed"], dtype=np.float32)
    extreme_threshold = float(
        np.quantile(np.abs(np.diff(train_speed, axis=0)), 0.95)
    )
    test_scores = _collect_window_scores(
        prepared,
        "test",
        lookback,
        horizon,
        batch_size,
        extreme_threshold=extreme_threshold,
    )
    train_shear_q50 = float(np.quantile(train_scores["direction_shear"], 0.50))
    train_shear_q75 = float(np.quantile(train_scores["direction_shear"], 0.75))
    weak_threshold = float(
        np.quantile(train_scores["spectral_peak_ratio"], 0.30)
    )
    test_extreme_count = test_scores["extreme_count"]
    spectral = test_scores["spectral_peak_ratio"]
    shear = test_scores["direction_shear"]
    masks: dict[str, np.ndarray] = {
        "normal": np.ones(len(spectral), dtype=bool),
        "weak_period": spectral <= weak_threshold,
        "extreme_0": test_extreme_count == 0,
        "extreme_1_2": (test_extreme_count >= 1) & (test_extreme_count <= 2),
        "extreme_3_5": (test_extreme_count >= 3) & (test_extreme_count <= 5),
        "extreme_gt5": test_extreme_count > 5,
        "direction_shear_low": shear <= train_shear_q50,
        "direction_shear_high": shear >= train_shear_q75,
        "missing_10pct": np.ones(len(spectral), dtype=bool),
    }
    for level in noise_levels:
        masks[format_noise_condition(level)] = np.ones(
            len(spectral), dtype=bool
        )
    thresholds = {
        "weak_period_train_q30": weak_threshold,
        "extreme_train_q95_abs_speed_difference": extreme_threshold,
        "direction_shear_train_q50": train_shear_q50,
        "direction_shear_train_q75": train_shear_q75,
        "noise_levels": list(noise_levels),
        "test_windows": int(len(spectral)),
    }
    rows = []
    test_start = pd.Timestamp(
        prepared["metadata"]["split_ranges"]["test"]["start"]
    )
    interval = pd.Timedelta(
        minutes=int(prepared["metadata"].get("interval_minutes", 10))
    )
    for index in range(len(spectral)):
        timestamp = test_start + int(test_scores["starts"][index]) * interval
        rows.append(
            {
                "sample_index": index,
                "timestamp_start": str(timestamp),
                "spectral_peak_ratio": float(spectral[index]),
                "direction_shear": float(shear[index]),
                "extreme_count": int(test_extreme_count[index]),
                "weak_period": bool(masks["weak_period"][index]),
                "extreme_group": (
                    "0"
                    if test_extreme_count[index] == 0
                    else "1-2"
                    if test_extreme_count[index] <= 2
                    else "3-5"
                    if test_extreme_count[index] <= 5
                    else ">5"
                ),
                "direction_shear_group": (
                    "low"
                    if masks["direction_shear_low"][index]
                    else "high"
                    if masks["direction_shear_high"][index]
                    else "middle"
                ),
            }
        )
    return masks, thresholds, pd.DataFrame(rows)


class _MetricAccumulator:
    def __init__(self, direction_metric: str = "circular") -> None:
        self.direction_metric = direction_metric
        self.speed_abs = 0.0
        self.speed_sq = 0.0
        self.speed_ape = 0.0
        self.speed_count = 0
        self.direction_abs = 0.0
        self.direction_sq = 0.0
        self.direction_ape = 0.0
        self.direction_count = 0

    def update(
        self, target: np.ndarray, prediction: np.ndarray, n: int, mask: np.ndarray
    ) -> None:
        if not np.any(mask):
            return
        true_speed = target[mask, ..., :n].astype(np.float64)
        pred_speed = prediction[mask, ..., :n].astype(np.float64)
        speed_error = pred_speed - true_speed
        self.speed_abs += float(np.abs(speed_error).sum())
        self.speed_sq += float(np.square(speed_error).sum())
        self.speed_ape += float(
            (np.abs(speed_error) / np.maximum(np.abs(true_speed), 1e-6)).sum()
        )
        self.speed_count += int(speed_error.size)

        true_direction = target[mask, ..., n:].astype(np.float64)
        pred_direction = prediction[mask, ..., n:].astype(np.float64)
        if self.direction_metric == "circular":
            direction_error = (
                pred_direction - true_direction + np.pi
            ) % (2.0 * np.pi) - np.pi
        else:
            direction_error = pred_direction - true_direction
        self.direction_abs += float(np.abs(direction_error).sum())
        self.direction_sq += float(np.square(direction_error).sum())
        self.direction_ape += float(
            (np.abs(direction_error) / np.maximum(np.abs(true_direction), 1e-6)).sum()
        )
        self.direction_count += int(direction_error.size)

    def result(self, sample_count: int) -> dict[str, Any]:
        if sample_count <= 0:
            return {
                "samples": 0,
                "speed": {
                    "mae": None,
                    "mse": None,
                    "rmse": None,
                    "mape": None,
                },
                "direction": {
                    "unit": "radians",
                    "mae": None,
                    "mse": None,
                    "rmse": None,
                    "mape": None,
                    "mae_rad": None,
                    "mse_rad2": None,
                    "rmse_rad": None,
                    "mae_degrees": None,
                    "mse_degrees2": None,
                    "rmse_degrees": None,
                },
            }
        speed_mae = self.speed_abs / max(self.speed_count, 1)
        speed_mse = self.speed_sq / max(self.speed_count, 1)
        direction_mae = self.direction_abs / max(self.direction_count, 1)
        direction_mse = self.direction_sq / max(self.direction_count, 1)
        direction_rmse = math.sqrt(direction_mse)
        output = {
            "samples": int(sample_count),
            "speed": {
                "mae": speed_mae,
                "mse": speed_mse,
                "rmse": math.sqrt(speed_mse),
                "mape": self.speed_ape / max(self.speed_count, 1) * 100.0,
            },
            "direction": {
                "unit": "radians",
                "mae": direction_mae,
                "mse": direction_mse,
                "rmse": direction_rmse,
                "mape": self.direction_ape
                / max(self.direction_count, 1)
                * 100.0,
                "mae_rad": direction_mae,
                "mse_rad2": direction_mse,
                "rmse_rad": direction_rmse,
                "mae_degrees": float(np.rad2deg(direction_mae)),
                "mse_degrees2": float(direction_mse * (180.0 / np.pi) ** 2),
                "rmse_degrees": float(np.rad2deg(direction_rmse)),
            },
        }
        return output


def _interpolate_missing(values: np.ndarray) -> np.ndarray:
    output = values.copy()
    time_axis = np.arange(output.shape[1])
    for batch_index in range(output.shape[0]):
        for feature_index in range(output.shape[2]):
            series = output[batch_index, :, feature_index]
            valid = np.isfinite(series)
            if valid.all():
                continue
            if not valid.any():
                output[batch_index, :, feature_index] = 0.0
                continue
            output[batch_index, :, feature_index] = np.interp(
                time_axis, time_axis[valid], series[valid]
            )
    return output


def _perturb_input(
    features: np.ndarray,
    prepared: dict[str, Any],
    condition: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = features.astype(np.float32, copy=True)
    representation, n = _feature_layout(prepared)
    if condition.startswith("noise_"):
        try:
            noise_level = float(condition.removeprefix("noise_"))
        except ValueError as exc:
            raise ValueError(f"Invalid noise condition: {condition}") from exc
        if noise_level <= 0.0:
            raise ValueError("Noise level must be positive.")

        # Add speed noise in physical units, then apply the train-fitted
        # normalization used by the model.
        speed_mean = np.asarray(
            prepared["speed_mean"], dtype=np.float32
        )[None, None, :]
        speed_std = np.asarray(
            prepared["speed_std"], dtype=np.float32
        )[None, None, :]
        speed_raw = values[..., :n] * speed_std + speed_mean
        speed_raw += noise_level * rng.standard_normal(
            speed_raw.shape
        ).astype(np.float32)
        values[..., :n] = (
            speed_raw - speed_mean
        ) / np.maximum(speed_std, 1e-6)

        angles = _direction_from_features(values, prepared)
        angles = (
            angles
            + noise_level
            * rng.standard_normal(angles.shape).astype(np.float32)
        ) % (2.0 * np.pi)
        if representation == "circular":
            values[..., n : 2 * n] = np.sin(angles)
            values[..., 2 * n : 3 * n] = np.cos(angles)
        else:
            direction_mean = np.asarray(
                prepared["direction_mean"], dtype=np.float32
            )
            direction_std = np.asarray(
                prepared["direction_std"], dtype=np.float32
            )
            values[..., n:] = (
                (angles - direction_mean[None, None, :])
                / np.maximum(direction_std[None, None, :], 1e-6)
            )
        return values, {
            "noise_level": noise_level,
            "speed_noise_std_mps": noise_level,
            "direction_noise_std_radians": noise_level,
            "noise_domain": "physical",
        }
    if condition == "missing_10pct":
        mask = rng.random(values.shape) < 0.10
        values[mask] = np.nan
        values = _interpolate_missing(values)
        return values, {
            "missing_rate_requested": 0.10,
            "missing_rate_realized": float(mask.mean()),
        }
    return values, {}


def _load_model_for_run(
    run_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    import yaml

    with (run_path / "config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    prepared = load_prepared(config["data"]["prepared_path"])
    checkpoint = torch.load(
        run_path / "best.pt", map_location=device, weights_only=False
    )
    model = build_model(
        str(checkpoint["model_name"]),
        config,
        int(checkpoint["num_features"]),
        prepared,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config, prepared


def evaluate_condition_runs(
    config: dict[str, Any],
    model_names: Iterable[str],
    horizons: Iterable[int],
    conditions: Iterable[str],
    output_root: str | Path | None = None,
    seed: int = 2026,
    noise_levels: Iterable[float] = DEFAULT_NOISE_LEVELS,
) -> list[dict[str, Any]]:
    """Evaluate existing benchmark checkpoints under selected test conditions."""
    noise_levels = normalize_noise_levels(noise_levels)
    training_config = config["training"]
    device = choose_device(str(training_config.get("device", "auto")))
    batch_size = int(training_config.get("batch_size", 128))
    prefix = str(
        config.get("benchmark", {}).get(
            "run_prefix", config["project"]["name"]
        )
    )
    dataset_name = str(config["data"].get("dataset_name", config["project"]["name"]))
    output_base = (
        Path(output_root)
        if output_root is not None
        else project_root() / "experiments" / "multi_condition"
    )
    dataset_root = output_base / dataset_name
    rows: list[dict[str, Any]] = []
    requested_conditions: set[str] = set()
    for value in conditions:
        normalized = str(value).strip().lower()
        requested_conditions.update(
            CONDITION_ALIASES.get(normalized, (normalized,))
        )

    for horizon in sorted({int(value) for value in horizons}):
        masks, thresholds, sample_table = _condition_masks(
            load_prepared(config["data"]["prepared_path"]),
            int(config["data"]["lookback"]),
            horizon,
            batch_size,
            noise_levels=noise_levels,
        )
        horizon_root = dataset_root / f"h{horizon}"
        horizon_root.mkdir(parents=True, exist_ok=True)
        write_json(thresholds, horizon_root / "condition_thresholds.json")
        sample_table.to_csv(
            horizon_root / "condition_samples.csv",
            index=False,
            encoding="utf-8",
        )

        for model_name in model_names:
            run_path = project_root() / "experiments" / (
                f"{prefix}-h{horizon}-{model_name}"
            )
            if not (run_path / "best.pt").exists():
                print(f"condition_skip_missing_run={run_path}")
                continue
            model, run_config, prepared = _load_model_for_run(run_path, device)
            test_dataset = make_dataset(
                prepared,
                "test",
                int(run_config["data"]["lookback"]),
                horizon,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(training_config.get("num_workers", 0)),
                pin_memory=device.type == "cuda",
            )
            direction_metric = str(
                run_config.get("evaluation", {}).get(
                    "direction_metric", "circular"
                )
            )
            available_conditions = [
                *CONDITION_NAMES,
                *(format_noise_condition(level) for level in noise_levels),
            ]
            selected = [
                name for name in available_conditions
                if name in requested_conditions
                or (name == "normal" and requested_conditions)
            ]
            accumulators = {
                name: _MetricAccumulator(direction_metric)
                for name in selected
            }
            sample_counts = {name: 0 for name in selected}
            rng = np.random.default_rng(seed)
            global_index = 0
            started = time.perf_counter()
            with torch.no_grad():
                for features, target in test_loader:
                    batch_count = int(features.shape[0])
                    batch_slice = slice(global_index, global_index + batch_count)
                    clean_features = features.numpy()
                    target_tensor = target.to(device, non_blocking=True)
                    for condition in selected:
                        batch_features = clean_features
                        if (
                            condition.startswith("noise_")
                            or condition == "missing_10pct"
                        ):
                            batch_features, _ = _perturb_input(
                                clean_features,
                                prepared,
                                condition,
                                rng,
                            )
                        else:
                            batch_features = clean_features
                        feature_tensor = torch.from_numpy(batch_features).to(
                            device, non_blocking=True
                        )
                        output = model(feature_tensor)
                        prediction = inverse_transform(
                            output.cpu().numpy(), prepared
                        )
                        target_values = inverse_transform(
                            target_tensor.cpu().numpy(), prepared
                        )
                        sample_mask = masks[condition][batch_slice]
                        accumulators[condition].update(
                            target_values,
                            prediction,
                            int(prepared["metadata"]["num_turbines"]),
                            sample_mask,
                        )
                        sample_counts[condition] += int(sample_mask.sum())
                    global_index += batch_count
            elapsed = time.perf_counter() - started
            model_root = horizon_root / model_name
            model_root.mkdir(parents=True, exist_ok=True)
            condition_results: dict[str, Any] = {}
            normal_speed_mae = None
            normal_direction_mae = None
            for condition, accumulator in accumulators.items():
                result = accumulator.result(sample_counts[condition])
                condition_results[condition] = result
                if condition == "normal":
                    normal_speed_mae = result["speed"]["mae"]
                    normal_direction_mae = result["direction"]["mae"]
                row = {
                    "dataset": dataset_name,
                    "horizon": horizon,
                    "model": model_name,
                    "condition": condition,
                    "samples": result["samples"],
                    "speed_mae": result["speed"]["mae"],
                    "speed_mse": result["speed"]["mse"],
                    "speed_rmse": result["speed"]["rmse"],
                    "direction_mae_rad": result["direction"]["mae_rad"],
                    "direction_mse_rad2": result["direction"]["mse_rad2"],
                    "direction_rmse_rad": result["direction"]["rmse_rad"],
                    "direction_mae_degrees": result["direction"][
                        "mae_degrees"
                    ],
                    "direction_rmse_degrees": result["direction"][
                        "rmse_degrees"
                    ],
                    "evaluation_seconds": elapsed,
                    "run_dir": str(run_path),
                }
                rows.append(row)
            for row in rows:
                if row["dataset"] == dataset_name and row["horizon"] == horizon and row["model"] == model_name:
                    if (
                        row["speed_mae"] is not None
                        and normal_speed_mae
                        and normal_speed_mae > 0
                    ):
                        row["speed_mae_degradation_pct"] = (
                            row["speed_mae"] - normal_speed_mae
                        ) / normal_speed_mae * 100.0
                    else:
                        row["speed_mae_degradation_pct"] = None
                    if (
                        row["direction_mae_rad"] is not None
                        and normal_direction_mae
                        and normal_direction_mae > 0
                    ):
                        row["direction_mae_degradation_pct"] = (
                            row["direction_mae_rad"] - normal_direction_mae
                        ) / normal_direction_mae * 100.0
                    else:
                        row["direction_mae_degradation_pct"] = None
            write_json(
                {
                    "dataset": dataset_name,
                    "horizon": horizon,
                    "model": model_name,
                    "conditions": condition_results,
                    "evaluation_seconds": elapsed,
                    "source_run": str(run_path),
                    "seed": seed,
                },
                model_root / "metrics.json",
            )
            write_json(
                {
                    "noise_levels": list(noise_levels),
                    "missing_rate": 0.10,
                    "seed": seed,
                    "condition_thresholds": thresholds,
                },
                model_root / "condition_metadata.json",
            )

    output_base.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with (dataset_root / "metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        write_json({"rows": rows}, dataset_root / "metrics.json")
    return rows


class _TransferWindowDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        turbine_indices: list[int],
        starts: np.ndarray,
        lookback: int,
        horizon: int,
        include_turbine: bool = False,
    ) -> None:
        self.values = values.astype(np.float32, copy=False)
        self.turbine_indices = list(turbine_indices)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.include_turbine = include_turbine

    def __len__(self) -> int:
        return len(self.turbine_indices) * len(self.starts)

    def __getitem__(self, index: int):
        start_index = index // len(self.starts)
        window_index = index % len(self.starts)
        turbine = self.turbine_indices[start_index]
        start = int(self.starts[window_index])
        middle = start + self.lookback
        end = middle + self.horizon
        features = torch.from_numpy(self.values[turbine, start:middle])
        target = torch.from_numpy(self.values[turbine, middle:end])
        if self.include_turbine:
            return features, target, turbine
        return features, target


def _single_turbine_prepared(
    prepared: dict[str, Any],
    turbine_index: int,
    normalization: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    metadata = deepcopy(prepared["metadata"])
    turbine_name = metadata["turbines"][turbine_index]
    metadata["turbines"] = [turbine_name]
    metadata["num_turbines"] = 1
    output = {
        "metadata": metadata,
        "speed_mean": np.asarray(
            prepared["speed_mean"][turbine_index : turbine_index + 1]
        ),
        "speed_std": np.asarray(
            prepared["speed_std"][turbine_index : turbine_index + 1]
        ),
        "direction_mean": np.asarray(
            prepared["direction_mean"][turbine_index : turbine_index + 1]
        ),
        "direction_std": np.asarray(
            prepared["direction_std"][turbine_index : turbine_index + 1]
        ),
    }
    for split in ("train", "validation", "test"):
        speed = prepared[f"{split}_speed"][
            :, turbine_index : turbine_index + 1
        ]
        if normalization is not None:
            speed = (
                speed * np.asarray(prepared["speed_std"])[turbine_index]
                + np.asarray(prepared["speed_mean"])[turbine_index]
            )
            speed = (speed - normalization["speed_mean"]) / normalization[
                "speed_std"
            ]
        output[f"{split}_speed"] = speed
        output[f"{split}_segment_ids"] = prepared.get(
            f"{split}_segment_ids"
        )
        representation = metadata.get("direction_representation", "scalar")
        if representation == "circular":
            output[f"{split}_direction_sin"] = prepared[
                f"{split}_direction_sin"
            ][:, turbine_index : turbine_index + 1]
            output[f"{split}_direction_cos"] = prepared[
                f"{split}_direction_cos"
            ][:, turbine_index : turbine_index + 1]
        else:
            direction = prepared[
                f"{split}_direction"
            ][:, turbine_index : turbine_index + 1]
            if normalization is not None:
                direction = (
                    direction * np.asarray(prepared["direction_std"])[
                        turbine_index
                    ]
                    + np.asarray(prepared["direction_mean"])[turbine_index]
                )
                direction = (
                    direction - normalization["direction_mean"]
                ) / normalization["direction_std"]
            output[f"{split}_direction"] = direction
    if normalization is not None:
        output["speed_mean"] = normalization["speed_mean"].copy()
        output["speed_std"] = normalization["speed_std"].copy()
        output["direction_mean"] = normalization["direction_mean"].copy()
        output["direction_std"] = normalization["direction_std"].copy()
    for key in ("coordinates", "geodetic_coordinates", "elevations"):
        if key in prepared:
            output[key] = prepared[key][turbine_index : turbine_index + 1]
    return output


def _source_normalization(
    prepared: dict[str, Any], source_indices: list[int]
) -> dict[str, np.ndarray]:
    """Fit transfer normalization only on the selected source turbines."""
    source = np.asarray(source_indices, dtype=np.int64)
    speed = (
        prepared["train_speed"][:, source]
        * np.asarray(prepared["speed_std"])[source]
        + np.asarray(prepared["speed_mean"])[source]
    )
    result = {
        "speed_mean": np.asarray([np.nanmean(speed)], dtype=np.float32),
        "speed_std": np.asarray([np.nanstd(speed)], dtype=np.float32),
    }
    result["speed_std"] = np.where(
        result["speed_std"] < 1e-6, 1.0, result["speed_std"]
    )
    if prepared["metadata"].get("direction_representation", "scalar") == "scalar":
        direction = (
            prepared["train_direction"][:, source]
            * np.asarray(prepared["direction_std"])[source]
            + np.asarray(prepared["direction_mean"])[source]
        )
        result["direction_mean"] = np.asarray(
            [np.nanmean(direction)], dtype=np.float32
        )
        result["direction_std"] = np.asarray(
            [np.nanstd(direction)], dtype=np.float32
        )
        result["direction_std"] = np.where(
            result["direction_std"] < 1e-6, 1.0, result["direction_std"]
        )
    else:
        result["direction_mean"] = np.zeros(1, dtype=np.float32)
        result["direction_std"] = np.ones(1, dtype=np.float32)
    return result


def _transfer_values(
    prepared: dict[str, Any],
    split: str,
    n: int,
    normalization: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    representation = str(
        prepared["metadata"].get("direction_representation", "scalar")
    )
    speed = prepared[f"{split}_speed"]
    if normalization is not None:
        speed = (
            speed * np.asarray(prepared["speed_std"])[None, :]
            + np.asarray(prepared["speed_mean"])[None, :]
        )
        speed = (
            speed - normalization["speed_mean"]
        ) / normalization["speed_std"]
    if representation == "circular":
        values = np.stack(
            [
                speed,
                prepared[f"{split}_direction_sin"],
                prepared[f"{split}_direction_cos"],
            ],
            axis=-1,
        )
    else:
        direction = prepared[f"{split}_direction"]
        if normalization is not None:
            direction = (
                direction * np.asarray(prepared["direction_std"])[None, :]
                + np.asarray(prepared["direction_mean"])[None, :]
            )
            direction = (
                direction - normalization["direction_mean"]
            ) / normalization["direction_std"]
        values = np.stack(
            [speed, direction],
            axis=-1,
        )
    return values.transpose(1, 0, 2).astype(np.float32, copy=False)


def _transfer_training_config(config: dict[str, Any]) -> dict[str, Any]:
    output = deepcopy(config)
    output["training"] = deepcopy(config["training"])
    # The transfer task uses one shared source-fitted normalization and avoids
    # adding graph or per-turbine direction losses to the single-turbine model.
    output["training"]["circular_direction_loss"] = False
    output["training"]["vector_loss_weight"] = 0.0
    output.setdefault("model", {})["use_graph"] = False
    return output


def _sample_transfer_split(
    num_turbines: int,
    seed: int,
    source_fraction: float = 0.25,
) -> tuple[list[int], list[int]]:
    """Sample source turbines and return the remaining zero-shot targets."""
    if num_turbines < 2:
        raise ValueError(
            "Cross-turbine experiments require at least two turbines."
        )
    if not 0.0 < source_fraction < 1.0:
        raise ValueError("source_fraction must be between 0 and 1.")
    source_count = min(
        num_turbines - 1,
        max(1, int(math.ceil(num_turbines * source_fraction))),
    )
    rng = np.random.default_rng(int(seed))
    source = sorted(
        int(value)
        for value in rng.choice(num_turbines, size=source_count, replace=False)
    )
    targets = [index for index in range(num_turbines) if index not in source]
    return source, targets


def _evaluate_transfer_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    prepared_by_turbine: dict[int, dict[str, Any]],
    direction_metric: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    aggregate = _MetricAccumulator(direction_metric)
    per_turbine: dict[int, _MetricAccumulator] = {}
    counts: dict[int, int] = {}
    with torch.no_grad():
        for features, target, turbine_ids in loader:
            output = model(features.to(device, non_blocking=True))
            target_np = target.numpy()
            output_np = output.cpu().numpy()
            turbine_ids_np = turbine_ids.numpy()
            for turbine in np.unique(turbine_ids_np):
                mask = turbine_ids_np == turbine
                single_prepared = prepared_by_turbine[int(turbine)]
                target_values = inverse_transform(
                    target_np[mask], single_prepared
                )
                prediction_values = inverse_transform(
                    output_np[mask], single_prepared
                )
                if int(turbine) not in per_turbine:
                    per_turbine[int(turbine)] = _MetricAccumulator(
                        direction_metric
                    )
                    counts[int(turbine)] = 0
                per_turbine[int(turbine)].update(
                    target_values,
                    prediction_values,
                    1,
                    np.ones(len(target_values), dtype=bool),
                )
                counts[int(turbine)] += len(target_values)
                aggregate.update(
                    target_values,
                    prediction_values,
                    1,
                    np.ones(len(target_values), dtype=bool),
                )
    per_rows = []
    for turbine, accumulator in sorted(per_turbine.items()):
        result = accumulator.result(counts[turbine])
        per_rows.append(
            {
                "turbine_index": turbine,
                "turbine_id": prepared_by_turbine[turbine]["metadata"][
                    "turbines"
                ][0],
                "samples": result["samples"],
                "speed_mae": result["speed"]["mae"],
                "speed_mse": result["speed"]["mse"],
                "speed_rmse": result["speed"]["rmse"],
                "direction_mae_rad": result["direction"]["mae_rad"],
                "direction_mse_rad2": result["direction"]["mse_rad2"],
                "direction_rmse_rad": result["direction"]["rmse_rad"],
                "direction_mae_degrees": result["direction"]["mae_degrees"],
                "direction_rmse_degrees": result["direction"][
                    "rmse_degrees"
                ],
            }
        )
    return aggregate.result(sum(counts.values())), per_rows


def _evaluate_transfer_model_with_noise(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    prepared_by_turbine: dict[int, dict[str, Any]],
    direction_metric: str,
    noise_levels: Iterable[float],
    seed: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate clean and noisy inputs only on unseen target turbines."""
    noise_levels = normalize_noise_levels(noise_levels)
    condition_names = (
        "normal",
        *(format_noise_condition(level) for level in noise_levels),
    )
    aggregate = {
        condition: _MetricAccumulator(direction_metric)
        for condition in condition_names
    }
    per_turbine: dict[tuple[str, int], _MetricAccumulator] = {}
    counts: dict[tuple[str, int], int] = {}
    rng = np.random.default_rng(int(seed))

    with torch.no_grad():
        for features, target, turbine_ids in loader:
            clean_features = features.numpy()
            target_np = target.numpy()
            turbine_ids_np = turbine_ids.numpy()
            batch_outputs: dict[str, np.ndarray] = {}
            for condition in condition_names:
                if condition == "normal":
                    batch_features = clean_features
                else:
                    batch_features = np.empty_like(clean_features)
                    for turbine in np.unique(turbine_ids_np):
                        mask = turbine_ids_np == turbine
                        batch_features[mask], _ = _perturb_input(
                            clean_features[mask],
                            prepared_by_turbine[int(turbine)],
                            condition,
                            rng,
                        )
                output = model(
                    torch.from_numpy(batch_features).to(
                        device, non_blocking=True
                    )
                )
                batch_outputs[condition] = output.cpu().numpy()

            for condition in condition_names:
                for turbine in np.unique(turbine_ids_np):
                    mask = turbine_ids_np == turbine
                    single_prepared = prepared_by_turbine[int(turbine)]
                    target_values = inverse_transform(
                        target_np[mask], single_prepared
                    )
                    prediction_values = inverse_transform(
                        batch_outputs[condition][mask], single_prepared
                    )
                    key = (condition, int(turbine))
                    if key not in per_turbine:
                        per_turbine[key] = _MetricAccumulator(direction_metric)
                        counts[key] = 0
                    per_turbine[key].update(
                        target_values,
                        prediction_values,
                        1,
                        np.ones(len(target_values), dtype=bool),
                    )
                    counts[key] += len(target_values)
                    aggregate[condition].update(
                        target_values,
                        prediction_values,
                        1,
                        np.ones(len(target_values), dtype=bool),
                    )

    condition_results = {
        condition: aggregate[condition].result(
            sum(
                counts.get((condition, turbine), 0)
                for turbine in prepared_by_turbine
            )
        )
        for condition in condition_names
    }
    per_rows = []
    for (condition, turbine), accumulator in sorted(per_turbine.items()):
        result = accumulator.result(counts[(condition, turbine)])
        per_rows.append(
            {
                "condition": condition,
                "turbine_index": turbine,
                "turbine_id": prepared_by_turbine[turbine]["metadata"][
                    "turbines"
                ][0],
                "samples": result["samples"],
                "speed_mae": result["speed"]["mae"],
                "speed_mse": result["speed"]["mse"],
                "speed_rmse": result["speed"]["rmse"],
                "direction_mae_rad": result["direction"]["mae_rad"],
                "direction_mse_rad2": result["direction"]["mse_rad2"],
                "direction_rmse_rad": result["direction"]["rmse_rad"],
                "direction_mae_degrees": result["direction"]["mae_degrees"],
                "direction_rmse_degrees": result["direction"][
                    "rmse_degrees"
                ],
            }
        )
    return condition_results, per_rows


def run_cross_turbine_experiments(
    config: dict[str, Any],
    model_names: Iterable[str],
    horizons: Iterable[int],
    seeds: Iterable[int] = (2026,),
    output_root: str | Path | None = None,
    skip_existing: bool = False,
    noise_levels: Iterable[float] = DEFAULT_NOISE_LEVELS,
) -> list[dict[str, Any]]:
    """Train on random 1/4 source turbines and test on unseen turbines."""
    noise_levels = normalize_noise_levels(noise_levels)
    base_prepared = load_prepared(config["data"]["prepared_path"])
    metadata = base_prepared["metadata"]
    n_turbines = int(metadata["num_turbines"])
    lookback = int(config["data"]["lookback"])
    batch_size = int(config["training"].get("batch_size", 128))
    device = choose_device(str(config["training"].get("device", "auto")))
    dataset_name = str(config["data"].get("dataset_name", config["project"]["name"]))
    base_output = (
        Path(output_root)
        if output_root is not None
        else project_root() / "experiments" / "multi_condition"
    )
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        source, targets = _sample_transfer_split(n_turbines, int(seed))
        seed_root = base_output / dataset_name / "cross_turbine" / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        write_json(
            {
                "dataset": dataset_name,
                "seed": int(seed),
                "source_count": len(source),
                "target_count": len(targets),
                "source_indices": source,
                "target_indices": targets,
                "source_turbines": [metadata["turbines"][i] for i in source],
                "target_turbines": [metadata["turbines"][i] for i in targets],
            },
            seed_root / "split.json",
        )

        for horizon in sorted({int(value) for value in horizons}):
            for model_name in model_names:
                model_config = _benchmark_config(config, model_name)
                model_config["data"]["horizon"] = horizon
                model_config = _transfer_training_config(model_config)
                run_root = seed_root / f"h{horizon}" / model_name
                if (
                    skip_existing
                    and (run_root / "metrics.json").exists()
                    and (run_root / "training_summary.json").exists()
                ):
                    metrics = read_json(run_root / "metrics.json")
                    condition_results = metrics.get(
                        "conditions", {"normal": metrics["aggregate"]}
                    )
                    for condition, aggregate in condition_results.items():
                        rows.append(
                            {
                                "dataset": dataset_name,
                                "seed": int(seed),
                                "horizon": horizon,
                                "model": model_name,
                                "condition": condition,
                                "samples": aggregate["samples"],
                                "speed_mae": aggregate["speed"]["mae"],
                                "speed_mse": aggregate["speed"]["mse"],
                                "speed_rmse": aggregate["speed"]["rmse"],
                                "direction_mae_rad": aggregate["direction"][
                                    "mae_rad"
                                ],
                                "direction_mse_rad2": aggregate["direction"][
                                    "mse_rad2"
                                ],
                                "direction_rmse_rad": aggregate["direction"][
                                    "rmse_rad"
                                ],
                                "direction_mae_degrees": aggregate[
                                    "direction"
                                ]["mae_degrees"],
                                "direction_rmse_degrees": aggregate[
                                    "direction"
                                ]["rmse_degrees"],
                                "parameters": metrics["parameters"],
                                "best_epoch": metrics["best_epoch"],
                                "training_seconds": metrics["training_seconds"],
                                "run_dir": str(run_root),
                            }
                        )
                    print(
                        f"skipping_completed_cross_turbine="
                        f"seed_{seed}/h{horizon}/{model_name}"
                    )
                    continue
                normalization = _source_normalization(base_prepared, source)
                prepared_by_turbine = {
                    i: _single_turbine_prepared(
                        base_prepared,
                        i,
                        normalization=normalization,
                    )
                    for i in range(n_turbines)
                }
                source_prepared = prepared_by_turbine[source[0]]
                source_values = {
                    split: _transfer_values(
                        base_prepared,
                        split,
                        n_turbines,
                        normalization=normalization,
                    )
                    for split in ("train", "validation")
                }
                base_dataset = make_dataset(
                    source_prepared, "train", lookback, horizon
                )
                validation_dataset = make_dataset(
                    source_prepared, "validation", lookback, horizon
                )
                train_dataset = _TransferWindowDataset(
                    source_values["train"],
                    source,
                    base_dataset.starts,
                    lookback,
                    horizon,
                )
                validation_values = {
                    "values": _transfer_values(
                        base_prepared,
                        "validation",
                        n_turbines,
                        normalization=normalization,
                    )
                }
                validation_dataset_transfer = _TransferWindowDataset(
                    validation_values["values"],
                    source,
                    validation_dataset.starts,
                    lookback,
                    horizon,
                )
                test_values = _transfer_values(
                    base_prepared,
                    "test",
                    n_turbines,
                    normalization=normalization,
                )
                test_dataset = _TransferWindowDataset(
                    test_values,
                    targets,
                    make_dataset(
                        source_prepared, "test", lookback, horizon
                    ).starts,
                    lookback,
                    horizon,
                    include_turbine=True,
                )
                loader_options = {
                    "batch_size": batch_size,
                    "num_workers": int(
                        config["training"].get("num_workers", 0)
                    ),
                    "pin_memory": device.type == "cuda",
                }
                generator = torch.Generator().manual_seed(int(seed))
                train_loader = DataLoader(
                    train_dataset,
                    shuffle=True,
                    generator=generator,
                    **loader_options,
                )
                validation_loader = DataLoader(
                    validation_dataset_transfer,
                    shuffle=False,
                    **loader_options,
                )
                test_loader = DataLoader(
                    test_dataset,
                    shuffle=False,
                    **loader_options,
                )
                direction_representation = str(
                    metadata.get("direction_representation", "scalar")
                )
                num_features = 3 if direction_representation == "circular" else 2
                set_seed(int(seed), deterministic=False)
                model = build_model(
                    model_name,
                    model_config,
                    num_features,
                    source_prepared,
                ).to(device)
                loss_context = _build_loss_context(source_prepared, device)
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=float(config["training"].get("learning_rate", 1e-4)),
                    weight_decay=float(
                        config["training"].get("weight_decay", 0.0)
                    ),
                )
                run_root.mkdir(parents=True, exist_ok=True)
                checkpoint_path = run_root / "best.pt"
                best_validation = float("inf")
                stale = 0
                history: list[dict[str, Any]] = []
                amp_enabled = bool(config["training"].get("amp", False)) and (
                    device.type == "cuda"
                )
                scaler = torch.amp.GradScaler(
                    "cuda", enabled=amp_enabled
                )
                started = time.perf_counter()
                for epoch in range(
                    1, int(config["training"].get("epochs", 50)) + 1
                ):
                    train_loss = _run_epoch(
                        model,
                        train_loader,
                        device,
                        1,
                        1.0,
                        1.0,
                        optimizer,
                        direction_representation=direction_representation,
                        loss_context=loss_context,
                        vector_loss_weight=0.0,
                        circular_direction_loss=False,
                        amp_enabled=amp_enabled,
                        scaler=scaler,
                    )
                    validation_loss = _run_epoch(
                        model,
                        validation_loader,
                        device,
                        1,
                        1.0,
                        1.0,
                        None,
                        direction_representation=direction_representation,
                        loss_context=loss_context,
                        vector_loss_weight=0.0,
                        circular_direction_loss=False,
                        amp_enabled=amp_enabled,
                    )
                    history.append(
                        {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "validation_loss": validation_loss,
                        }
                    )
                    print(
                        f"cross_turbine seed={seed} horizon={horizon} "
                        f"model={model_name} epoch={epoch:03d} "
                        f"train_loss={train_loss:.6f} "
                        f"validation_loss={validation_loss:.6f}"
                    )
                    if validation_loss < best_validation:
                        best_validation = validation_loss
                        stale = 0
                        torch.save(
                            {
                                "model_name": model_name,
                                "model_state": model.state_dict(),
                                "num_features": num_features,
                                "epoch": epoch,
                                "best_validation_loss": best_validation,
                            },
                            checkpoint_path,
                        )
                    else:
                        stale += 1
                        if stale >= int(
                            config["training"].get("patience", 10)
                        ):
                            break
                training_seconds = time.perf_counter() - started
                checkpoint = torch.load(
                    checkpoint_path, map_location=device, weights_only=False
                )
                model.load_state_dict(checkpoint["model_state"])
                condition_results, per_rows = _evaluate_transfer_model_with_noise(
                    model,
                    test_loader,
                    device,
                    prepared_by_turbine,
                    str(
                        config.get("evaluation", {}).get(
                            "direction_metric", "circular"
                        )
                    ),
                    noise_levels=noise_levels,
                    seed=int(seed),
                )
                aggregate = condition_results["normal"]
                write_json(
                    {
                        "dataset": dataset_name,
                        "seed": int(seed),
                        "horizon": horizon,
                        "model": model_name,
                        "source_turbines": [
                            metadata["turbines"][i] for i in source
                        ],
                        "target_turbines": [
                            metadata["turbines"][i] for i in targets
                        ],
                        "noise_levels": list(noise_levels),
                        "aggregate": aggregate,
                        "conditions": condition_results,
                        "parameters": sum(
                            parameter.numel()
                            for parameter in model.parameters()
                        ),
                        "best_epoch": int(checkpoint["epoch"]),
                        "training_seconds": training_seconds,
                    },
                    run_root / "metrics.json",
                )
                with (run_root / "per_turbine.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=sorted(
                            {
                                key
                                for row in per_rows
                                for key in row.keys()
                            }
                        ),
                    )
                    writer.writeheader()
                    writer.writerows(per_rows)
                write_json(
                    {
                        "parameters": sum(
                            parameter.numel()
                            for parameter in model.parameters()
                        ),
                        "best_epoch": int(checkpoint["epoch"]),
                        "training_seconds": training_seconds,
                        "source_turbines": [
                            metadata["turbines"][i] for i in source
                        ],
                        "target_turbines": [
                            metadata["turbines"][i] for i in targets
                        ],
                        "history": history,
                    },
                    run_root / "training_summary.json",
                )
                for condition, condition_result in condition_results.items():
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "seed": int(seed),
                            "horizon": horizon,
                            "model": model_name,
                            "condition": condition,
                            "samples": condition_result["samples"],
                            "speed_mae": condition_result["speed"]["mae"],
                            "speed_mse": condition_result["speed"]["mse"],
                            "speed_rmse": condition_result["speed"]["rmse"],
                            "direction_mae_rad": condition_result[
                                "direction"
                            ]["mae_rad"],
                            "direction_mse_rad2": condition_result[
                                "direction"
                            ]["mse_rad2"],
                            "direction_rmse_rad": condition_result[
                                "direction"
                            ]["rmse_rad"],
                            "direction_mae_degrees": condition_result[
                                "direction"
                            ]["mae_degrees"],
                            "direction_rmse_degrees": condition_result[
                                "direction"
                            ]["rmse_degrees"],
                            "parameters": sum(
                                parameter.numel()
                                for parameter in model.parameters()
                            ),
                            "best_epoch": int(checkpoint["epoch"]),
                            "training_seconds": training_seconds,
                            "run_dir": str(run_root),
                        }
                    )
    final_dir = base_output / dataset_name / "cross_turbine"
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with (final_dir / "summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        write_json({"rows": rows}, final_dir / "summary.json")
    return rows
