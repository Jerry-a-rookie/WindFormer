import numpy as np

from wind_repro.metrics import (
    circular_regression_metrics,
    regression_metrics,
    split_feature_metrics,
)


def test_regression_metrics_match_manual_values() -> None:
    target = np.array([1.0, 2.0, 3.0])
    prediction = np.array([2.0, 2.0, 1.0])
    metrics = regression_metrics(target, prediction)
    assert np.isclose(metrics["mae"], 1.0)
    assert np.isclose(metrics["mse"], 5.0 / 3.0)
    assert np.isclose(metrics["rmse"], np.sqrt(5.0 / 3.0))


def test_split_metrics_separate_speed_and_direction() -> None:
    target = np.zeros((2, 3, 4), dtype=np.float32)
    prediction = np.zeros_like(target)
    prediction[..., :2] = 1.0
    prediction[..., 2:] = 2.0
    metrics = split_feature_metrics(target, prediction, num_turbines=2)
    assert metrics["speed"]["mae"] == 1.0
    assert metrics["direction"]["mae"] == 2.0


def test_circular_metrics_use_shortest_angular_distance() -> None:
    target = np.deg2rad(np.array([359.0]))
    prediction = np.deg2rad(np.array([1.0]))
    metrics = circular_regression_metrics(target, prediction)
    assert metrics["unit"] == "radians"
    assert np.isclose(metrics["mae_rad"], np.deg2rad(2.0))
    assert np.isclose(metrics["rmse_rad"], np.deg2rad(2.0))
    assert np.isclose(metrics["mae_degrees"], 2.0)
