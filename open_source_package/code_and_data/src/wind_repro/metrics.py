from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    error = y_pred - y_true
    absolute = np.abs(error)
    return {
        "mae": float(np.mean(absolute)),
        "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": float(np.mean(absolute / np.maximum(np.abs(y_true), 1e-6)) * 100.0),
    }


def circular_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    error = (y_pred - y_true + np.pi) % (2 * np.pi) - np.pi
    absolute = np.abs(error)
    mae = float(np.mean(absolute))
    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))
    degree_scale = 180.0 / np.pi
    return {
        "unit": "radians",
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mae_rad": mae,
        "mse_rad2": mse,
        "rmse_rad": rmse,
        "mae_degrees": float(np.rad2deg(mae)),
        "mse_degrees2": float(mse * degree_scale**2),
        "rmse_degrees": float(np.rad2deg(rmse)),
    }


def split_feature_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_turbines: int,
    direction_metric: str = "circular",
) -> dict[str, dict[str, float]]:
    direction_metric = direction_metric.lower()
    if direction_metric == "scalar":
        direction = regression_metrics(
            y_true[..., num_turbines:], y_pred[..., num_turbines:]
        )
        direction.update(
            {
                "unit": "radians",
                "mae_rad": direction["mae"],
                "mse_rad2": direction["mse"],
                "rmse_rad": direction["rmse"],
                "mae_degrees": float(np.rad2deg(direction["mae"])),
                "mse_degrees2": float(
                    direction["mse"] * (180.0 / np.pi) ** 2
                ),
                "rmse_degrees": float(np.rad2deg(direction["rmse"])),
            }
        )
    elif direction_metric == "circular":
        direction = circular_regression_metrics(
            y_true[..., num_turbines:], y_pred[..., num_turbines:]
        )
    else:
        raise ValueError(
            "direction_metric must be either 'scalar' or 'circular'."
        )
    return {
        "speed": regression_metrics(
            y_true[..., :num_turbines], y_pred[..., :num_turbines]
        ),
        "direction": direction,
    }
