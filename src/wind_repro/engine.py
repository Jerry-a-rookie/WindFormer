from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import load_config, save_config
from .data import (
    fft_denoise_window_batch,
    inverse_transform,
    load_prepared,
    make_dataset,
)
from .metrics import split_feature_metrics
from .models import build_model
from .utils import choose_device, environment_info, set_seed, write_json


def _weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_turbines: int,
    speed_weight: float,
    direction_weight: float,
) -> torch.Tensor:
    speed_loss = nn.functional.mse_loss(
        prediction[..., :num_turbines], target[..., :num_turbines]
    )
    direction_loss = nn.functional.mse_loss(
        prediction[..., num_turbines:], target[..., num_turbines:]
    )
    return speed_weight * speed_loss + direction_weight * direction_loss


def _build_loss_context(
    prepared: dict[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    num_turbines = int(prepared["metadata"]["num_turbines"])
    return {
        "speed_mean": torch.as_tensor(
            prepared.get("speed_mean", np.zeros(num_turbines)),
            device=device,
            dtype=torch.float32,
        ),
        "speed_std": torch.as_tensor(
            prepared.get("speed_std", np.ones(num_turbines)),
            device=device,
            dtype=torch.float32,
        ),
        "direction_mean": torch.as_tensor(
            prepared.get("direction_mean", np.zeros(num_turbines)),
            device=device,
            dtype=torch.float32,
        ),
        "direction_std": torch.as_tensor(
            prepared.get("direction_std", np.ones(num_turbines)),
            device=device,
            dtype=torch.float32,
        ),
    }


def _direction_components(
    values: torch.Tensor,
    num_turbines: int,
    direction_representation: str,
    loss_context: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if direction_representation == "circular":
        sine = values[..., num_turbines : 2 * num_turbines]
        cosine = values[..., 2 * num_turbines : 3 * num_turbines]
        norm = torch.sqrt(sine.square() + cosine.square() + 1e-6)
        sine = sine / norm
        cosine = cosine / norm
        angle = torch.atan2(sine, cosine)
        return angle, cosine, sine

    angle = (
        values[..., num_turbines:]
        * loss_context["direction_std"].view(1, 1, -1)
        + loss_context["direction_mean"].view(1, 1, -1)
    )
    return angle, torch.cos(angle), torch.sin(angle)


def _forecast_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_turbines: int,
    speed_weight: float,
    direction_weight: float,
    direction_representation: str,
    loss_context: dict[str, torch.Tensor],
    vector_weight: float = 0.0,
    circular_direction_loss: bool = False,
    low_wind_direction_threshold: float = 1.0,
) -> torch.Tensor:
    speed_loss = nn.functional.mse_loss(
        prediction[..., :num_turbines], target[..., :num_turbines]
    )
    if (
        direction_representation == "circular"
        or circular_direction_loss
    ):
        _, prediction_cosine, prediction_sine = _direction_components(
            prediction,
            num_turbines,
            direction_representation,
            loss_context,
        )
        _, target_cosine, target_sine = _direction_components(
            target,
            num_turbines,
            direction_representation,
            loss_context,
        )
        angular_error = 1.0 - (
            prediction_cosine * target_cosine
            + prediction_sine * target_sine
        )
        target_speed = (
            target[..., :num_turbines]
            * loss_context["speed_std"].view(1, 1, -1)
            + loss_context["speed_mean"].view(1, 1, -1)
        )
        direction_weighting = (
            target_speed.clamp_min(0.0)
            / max(float(low_wind_direction_threshold), 1e-6)
        ).clamp(0.0, 1.0)
        direction_loss = (
            (angular_error * direction_weighting).sum()
            / direction_weighting.sum().clamp_min(1.0)
        )
    else:
        direction_loss = nn.functional.mse_loss(
            prediction[..., num_turbines:],
            target[..., num_turbines:],
        )

    total = speed_weight * speed_loss + direction_weight * direction_loss
    if vector_weight > 0.0:
        _, prediction_cosine, prediction_sine = _direction_components(
            prediction,
            num_turbines,
            direction_representation,
            loss_context,
        )
        _, target_cosine, target_sine = _direction_components(
            target,
            num_turbines,
            direction_representation,
            loss_context,
        )
        prediction_speed = (
            prediction[..., :num_turbines]
            * loss_context["speed_std"].view(1, 1, -1)
            + loss_context["speed_mean"].view(1, 1, -1)
        )
        target_speed = (
            target[..., :num_turbines]
            * loss_context["speed_std"].view(1, 1, -1)
            + loss_context["speed_mean"].view(1, 1, -1)
        )
        prediction_vector = torch.stack(
            [prediction_speed * prediction_cosine,
             prediction_speed * prediction_sine],
            dim=-1,
        )
        target_vector = torch.stack(
            [target_speed * target_cosine, target_speed * target_sine],
            dim=-1,
        )
        speed_scale = loss_context["speed_std"].mean().clamp_min(1e-3)
        vector_loss = nn.functional.smooth_l1_loss(
            prediction_vector / speed_scale,
            target_vector / speed_scale,
        )
        total = total + vector_weight * vector_loss
    return total


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_turbines: int,
    speed_weight: float,
    direction_weight: float,
    optimizer: torch.optim.Optimizer | None,
    window_fft_quantile: float | None = None,
    window_fft_target: bool = False,
    direction_representation: str = "scalar",
    loss_context: dict[str, torch.Tensor] | None = None,
    vector_loss_weight: float = 0.0,
    circular_direction_loss: bool = False,
    low_wind_direction_threshold: float = 1.0,
    amp_enabled: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for features, target in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if window_fft_quantile is not None:
                speed = fft_denoise_window_batch(
                    features[..., :num_turbines], window_fft_quantile
                )
                features = torch.cat([speed, features[..., num_turbines:]], dim=-1)
                if window_fft_target:
                    speed = fft_denoise_window_batch(
                        target[..., :num_turbines], window_fft_quantile
                    )
                    target = torch.cat(
                        [speed, target[..., num_turbines:]], dim=-1
                    )
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                prediction = model(features)
                if loss_context is None:
                    raise ValueError("loss_context is required for training.")
                loss = _forecast_loss(
                    prediction,
                    target,
                    num_turbines,
                    speed_weight,
                    direction_weight,
                    direction_representation,
                    loss_context,
                    vector_weight=vector_loss_weight,
                    circular_direction_loss=circular_direction_loss,
                    low_wind_direction_threshold=(
                        low_wind_direction_threshold
                    ),
                )
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            batch_size = features.shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_samples += batch_size
    return total_loss / max(total_samples, 1)


def train_experiment(
    config: dict[str, Any], model_name: str, run_dir: str | Path
) -> Path:
    run_path = Path(run_dir).resolve()
    run_path.mkdir(parents=True, exist_ok=True)
    save_config(config, run_path / "config.yaml")

    seed = int(config["project"].get("seed", 2026))
    training_config = config["training"]
    deterministic = bool(training_config.get("deterministic", True))
    set_seed(seed, deterministic=deterministic)
    data_config = config["data"]
    device = choose_device(str(training_config.get("device", "auto")))
    prepared = load_prepared(data_config["prepared_path"])
    metadata = prepared["metadata"]
    num_turbines = int(metadata["num_turbines"])
    direction_representation = str(
        metadata.get("direction_representation", "scalar")
    )
    direction_metric = str(
        config.get("evaluation", {}).get("direction_metric", "circular")
    )
    num_features = (
        3 * num_turbines
        if direction_representation == "circular"
        else 2 * num_turbines
    )

    train_dataset = make_dataset(
        prepared,
        "train",
        int(data_config["lookback"]),
        int(data_config["horizon"]),
    )
    validation_dataset = make_dataset(
        prepared,
        "validation",
        int(data_config["lookback"]),
        int(data_config["horizon"]),
    )
    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": int(training_config.get("batch_size", 32)),
        "num_workers": int(training_config.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_options
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_options
    )

    model = build_model(model_name, config, num_features, prepared).to(device)
    loss_context = _build_loss_context(prepared, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-4)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    speed_weight = float(training_config.get("speed_loss_weight", 1.0))
    direction_weight = float(training_config.get("direction_loss_weight", 1.0))
    vector_loss_weight = float(
        training_config.get("vector_loss_weight", 0.0)
    )
    circular_direction_loss = bool(
        training_config.get("circular_direction_loss", False)
    )
    low_wind_direction_threshold = float(
        training_config.get("low_wind_direction_threshold", 1.0)
    )
    denoise_config = data_config.get("fft_denoise", {})
    window_fft_enabled = bool(denoise_config.get("enabled", False)) and str(
        denoise_config.get("scope", "")
    ).lower() == "window"
    window_fft_apply_to = {
        str(value).lower()
        for value in denoise_config.get("apply_to", ["input"])
    }
    window_fft_quantile = (
        float(denoise_config.get("amplitude_quantile", 0.1))
        if window_fft_enabled and "input" in window_fft_apply_to
        else None
    )
    window_fft_target = (
        window_fft_enabled and "target" in window_fft_apply_to
    )
    epochs = int(training_config.get("epochs", 100))
    patience = int(training_config.get("patience", 3))
    amp_enabled = bool(training_config.get("amp", False)) and (
        device.type == "cuda"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history: list[dict[str, float | int]] = []
    best_validation = float("inf")
    stale_epochs = 0
    checkpoint_path = run_path / "best.pt"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_started = time.perf_counter()
        train_loss = _run_epoch(
            model,
            train_loader,
            device,
            num_turbines,
            speed_weight,
            direction_weight,
            optimizer,
            window_fft_quantile,
            window_fft_target,
            direction_representation,
            loss_context=loss_context,
            vector_loss_weight=vector_loss_weight,
            circular_direction_loss=circular_direction_loss,
            low_wind_direction_threshold=low_wind_direction_threshold,
            amp_enabled=amp_enabled,
            scaler=scaler,
        )
        validation_loss = _run_epoch(
            model,
            validation_loader,
            device,
            num_turbines,
            speed_weight,
            direction_weight,
            optimizer=None,
            window_fft_quantile=window_fft_quantile,
            window_fft_target=window_fft_target,
            direction_representation=direction_representation,
            loss_context=loss_context,
            vector_loss_weight=vector_loss_weight,
            circular_direction_loss=circular_direction_loss,
            low_wind_direction_threshold=low_wind_direction_threshold,
            amp_enabled=amp_enabled,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_started
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "epoch_seconds": epoch_seconds,
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f} "
            f"seconds={epoch_seconds:.2f}"
        )

        if validation_loss < best_validation:
            best_validation = validation_loss
            stale_epochs = 0
            torch.save(
                {
                    "model_name": model_name,
                    "model_state": model.state_dict(),
                    "num_features": num_features,
                    "num_turbines": num_turbines,
                    "best_validation_loss": best_validation,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"early_stopping epoch={epoch}")
                break

    with (run_path / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "validation_loss",
                "epoch_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else 0.0
    )
    write_json(environment_info(device), run_path / "environment.json")
    write_json(metadata, run_path / "data_metadata.json")
    write_json(
        {
            "model_name": model_name,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "best_validation_loss": best_validation,
            "epochs_completed": len(history),
            "training_seconds": training_seconds,
            "mean_epoch_seconds": (
                sum(float(row["epoch_seconds"]) for row in history)
                / max(len(history), 1)
            ),
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "amp_enabled": amp_enabled,
            "deterministic": deterministic,
        },
        run_path / "training_summary.json",
    )
    return run_path


def evaluate_experiment(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    config = load_config(run_path / "config.yaml")
    training_config = config["training"]
    data_config = config["data"]
    device = choose_device(str(training_config.get("device", "auto")))
    prepared = load_prepared(data_config["prepared_path"])
    metadata = prepared["metadata"]
    num_turbines = int(metadata["num_turbines"])
    direction_representation = str(
        metadata.get("direction_representation", "scalar")
    )
    direction_metric = str(
        config.get("evaluation", {}).get("direction_metric", "circular")
    )

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

    test_dataset = make_dataset(
        prepared,
        "test",
        int(data_config["lookback"]),
        int(data_config["horizon"]),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(training_config.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(training_config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    denoise_config = data_config.get("fft_denoise", {})
    window_fft_enabled = bool(denoise_config.get("enabled", False)) and str(
        denoise_config.get("scope", "")
    ).lower() == "window"
    window_fft_apply_to = {
        str(value).lower()
        for value in denoise_config.get("apply_to", ["input"])
    }
    window_fft_quantile = float(
        denoise_config.get("amplitude_quantile", 0.1)
    )
    amp_enabled = bool(training_config.get("amp", False)) and (
        device.type == "cuda"
    )

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    evaluation_started = time.perf_counter()
    with torch.no_grad():
        for features, target in test_loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if window_fft_enabled and "input" in window_fft_apply_to:
                speed = fft_denoise_window_batch(
                    features[..., :num_turbines], window_fft_quantile
                )
                features = torch.cat([speed, features[..., num_turbines:]], dim=-1)
            if window_fft_enabled and "target" in window_fft_apply_to:
                speed = fft_denoise_window_batch(
                    target[..., :num_turbines], window_fft_quantile
                )
                target = torch.cat(
                    [speed, target[..., num_turbines:]], dim=-1
                )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = model(features)
            predictions.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    evaluation_seconds = time.perf_counter() - evaluation_started

    normalized_prediction = np.concatenate(predictions, axis=0)
    normalized_target = np.concatenate(targets, axis=0)
    prediction = inverse_transform(normalized_prediction, prepared)
    target = inverse_transform(normalized_target, prepared)
    metrics = split_feature_metrics(
        target,
        prediction,
        num_turbines,
        direction_metric=direction_metric,
    )
    configured_horizons = config.get("evaluation", {}).get(
        "report_horizons", [int(data_config["horizon"])]
    )
    report_horizons = sorted(
        {
            int(horizon)
            for horizon in configured_horizons
            if 1 <= int(horizon) <= int(data_config["horizon"])
        }
    )
    if not report_horizons:
        report_horizons = [int(data_config["horizon"])]
    metrics["horizons"] = {
        str(horizon): split_feature_metrics(
            target[:, :horizon],
            prediction[:, :horizon],
            num_turbines,
            direction_metric=direction_metric,
        )
        for horizon in report_horizons
    }
    metrics["run"] = {
        "samples": int(len(prediction)),
        "horizon": int(data_config["horizon"]),
        "report_horizons": report_horizons,
        "dataset": str(data_config.get("dataset_name", "unknown")),
        "model": str(checkpoint["model_name"]),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "evaluation_seconds": evaluation_seconds,
        "samples_per_second": (
            len(prediction) / max(evaluation_seconds, 1e-9)
        ),
        "fft_denoise": {
            "enabled": bool(denoise_config.get("enabled", False)),
            "scope": (
                "window"
                if window_fft_enabled
                else str(denoise_config.get("scope", "disabled"))
            ),
            "apply_to": sorted(window_fft_apply_to)
            if window_fft_enabled
            else [],
        },
        "direction_metric": direction_metric,
    }

    np.savez_compressed(
        run_path / "predictions.npz",
        prediction=prediction,
        target=target,
    )
    write_json(metrics, run_path / "metrics.json")
    return metrics
