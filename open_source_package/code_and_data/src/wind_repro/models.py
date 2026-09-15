from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import resolve_project_path


ModelBuilder = Callable[[dict[str, Any], int, dict[str, Any] | None], nn.Module]
MODEL_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("Model names cannot be empty.")

    def decorator(builder: ModelBuilder) -> ModelBuilder:
        if normalized_name in MODEL_REGISTRY:
            raise ValueError(f"Model '{normalized_name}' is already registered.")
        MODEL_REGISTRY[normalized_name] = builder
        return builder

    return decorator


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("moving_average must be a positive odd integer")
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = (self.kernel_size - 1) // 2
        values = x.transpose(1, 2)
        values = F.pad(values, (padding, padding), mode="replicate")
        return F.avg_pool1d(values, self.kernel_size, stride=1).transpose(1, 2)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_average = MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_average(x)
        seasonal = x - trend
        return seasonal, trend


class ExponentialMovingAverage(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError("EMA alpha must lie in (0, 1).")
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, features], got {tuple(x.shape)}")
        alpha = self.alpha
        ema = x[:, 0]
        values = [ema]
        for index in range(1, x.shape[1]):
            ema = alpha * x[:, index] + (1.0 - alpha) * ema
            values.append(ema)
        return torch.stack(values, dim=1)


class GraphLaplacianRefiner(nn.Module):
    def __init__(
        self,
        num_turbines: int,
        coordinates: np.ndarray | None,
        k_neighbors: int = 4,
        kernel_scale: float | None = None,
        strength: float = 0.5,
    ) -> None:
        super().__init__()
        if num_turbines < 1:
            raise ValueError("num_turbines must be positive.")
        self.num_turbines = num_turbines
        self.logit_mix = nn.Parameter(torch.tensor(2.0))

        if (
            coordinates is None
            or len(coordinates) == 0
            or not np.isfinite(coordinates[:, :2]).all()
        ):
            laplacian = self._build_index_laplacian(num_turbines)
        else:
            laplacian = self._build_coordinate_laplacian(
                np.asarray(coordinates, dtype=np.float32),
                max(int(k_neighbors), 0),
                kernel_scale,
            )
        self.register_buffer("laplacian", laplacian)
        system = torch.eye(num_turbines, dtype=torch.float32) + float(
            max(strength, 1e-6)
        ) * laplacian
        self.register_buffer("smoothing_matrix", torch.linalg.inv(system))

    @staticmethod
    def _build_index_laplacian(num_turbines: int) -> torch.Tensor:
        adjacency = torch.zeros(num_turbines, num_turbines, dtype=torch.float32)
        for index in range(num_turbines - 1):
            adjacency[index, index + 1] = 1.0
            adjacency[index + 1, index] = 1.0
        degree = torch.diag(adjacency.sum(dim=1))
        return degree - adjacency

    @staticmethod
    def _build_coordinate_laplacian(
        coordinates: np.ndarray,
        k_neighbors: int,
        kernel_scale: float | None,
    ) -> torch.Tensor:
        if coordinates.ndim != 2 or coordinates.shape[1] < 2:
            raise ValueError("coordinates must have shape [num_turbines, 2+].")
        xy = coordinates[:, :2]
        distances = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1))
        if kernel_scale is None:
            nonzero = distances[distances > 0]
            kernel_scale = float(np.median(nonzero)) if nonzero.size else 1.0
        kernel_scale = max(float(kernel_scale), 1e-6)
        weights = np.exp(-(distances**2) / (kernel_scale**2))
        np.fill_diagonal(weights, 0.0)
        if 0 < k_neighbors < len(weights):
            knn = np.zeros_like(weights)
            for index in range(len(weights)):
                nearest = np.argsort(distances[index])[1 : k_neighbors + 1]
                knn[index, nearest] = weights[index, nearest]
            weights = np.maximum(knn, knn.T)
        degree = np.diag(weights.sum(axis=1))
        return torch.tensor(degree - weights, dtype=torch.float32)

    def forward(self, residual_prediction: torch.Tensor) -> torch.Tensor:
        if residual_prediction.ndim != 3:
            raise ValueError(
                "Expected residual prediction with shape [batch, horizon, features], "
                f"got {tuple(residual_prediction.shape)}"
            )
        batch, horizon, features = residual_prediction.shape
        if features != self.num_turbines:
            raise ValueError(
                "Residual prediction feature count does not match turbine layout."
            )
        flat = residual_prediction.transpose(0, 2).reshape(
            self.num_turbines, batch * horizon
        )
        solved = torch.matmul(
            self.smoothing_matrix.to(
                device=residual_prediction.device,
                dtype=residual_prediction.dtype,
            ),
            flat,
        )
        mix = torch.sigmoid(self.logit_mix)
        refined = flat + mix * (solved - flat)
        return refined.reshape(self.num_turbines, batch, horizon).permute(
            1, 2, 0
        )


class DirectionalGraphRefiner(nn.Module):
    """One-step directed Laplacian smoothing with a residual connection."""

    def __init__(
        self,
        num_turbines: int,
        coordinates: np.ndarray | None,
        k_neighbors: int = 4,
        kernel_scale: float | None = None,
        direction_convention: str = "from",
        temperature: float = 0.35,
    ) -> None:
        super().__init__()
        if num_turbines < 1:
            raise ValueError("num_turbines must be positive.")
        convention = str(direction_convention).lower()
        if convention not in {"from", "to"}:
            raise ValueError("direction_convention must be 'from' or 'to'.")
        self.num_turbines = num_turbines
        self.direction_convention = convention
        self.temperature = max(float(temperature), 1e-3)
        self.logit_mix = nn.Parameter(torch.tensor(-1.0))

        if coordinates is None or len(coordinates) == 0:
            xy = np.stack(
                [
                    np.arange(num_turbines, dtype=np.float32),
                    np.zeros(num_turbines, dtype=np.float32),
                ],
                axis=1,
            )
        else:
            xy = np.asarray(coordinates, dtype=np.float32)[:, :2]
            if xy.shape != (num_turbines, 2) or not np.isfinite(xy).all():
                xy = np.stack(
                    [
                        np.arange(num_turbines, dtype=np.float32),
                        np.zeros(num_turbines, dtype=np.float32),
                    ],
                    axis=1,
                )

        distances = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
        nonzero = distances[distances > 0]
        scale = (
            float(np.median(nonzero))
            if kernel_scale is None and nonzero.size
            else float(kernel_scale or 1.0)
        )
        scale = max(scale, 1e-6)
        weights = np.exp(-(distances**2) / (scale**2))
        np.fill_diagonal(weights, 0.0)
        if 0 < int(k_neighbors) < num_turbines:
            knn = np.zeros_like(weights)
            for index in range(num_turbines):
                nearest = np.argsort(distances[index])[1 : int(k_neighbors) + 1]
                knn[index, nearest] = weights[index, nearest]
            weights = np.maximum(knn, knn.T)

        relative = xy[:, None, :] - xy[None, :, :]
        relative_norm = np.linalg.norm(relative, axis=-1, keepdims=True)
        relative_unit = relative / np.maximum(relative_norm, 1e-6)
        self.register_buffer(
            "base_weights",
            torch.tensor(weights, dtype=torch.float32),
        )
        self.register_buffer(
            "relative_unit",
            torch.tensor(relative_unit, dtype=torch.float32),
        )

    def forward(
        self, residual_prediction: torch.Tensor, flow_direction: torch.Tensor
    ) -> torch.Tensor:
        if residual_prediction.ndim != 3:
            raise ValueError(
                "Expected residual prediction [batch, horizon, turbines]."
            )
        if flow_direction.shape != residual_prediction.shape[:2]:
            raise ValueError(
                "flow_direction must have shape [batch, horizon]."
            )
        if residual_prediction.shape[-1] != self.num_turbines:
            raise ValueError("Residual turbine count does not match graph.")

        flow = torch.stack(
            [torch.cos(flow_direction), torch.sin(flow_direction)], dim=-1
        )
        if self.direction_convention == "from":
            flow = -flow
        alignment = torch.einsum(
            "bhd,ijd->bhij", flow, self.relative_unit.to(flow)
        )
        gate = torch.sigmoid(alignment / self.temperature)
        weights = self.base_weights.to(
            device=residual_prediction.device,
            dtype=residual_prediction.dtype,
        )[None, None] * gate
        weights = weights * (
            1.0
            - torch.eye(
                self.num_turbines,
                device=weights.device,
                dtype=weights.dtype,
            )[None, None]
        )
        normalized = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        smoothed = torch.einsum(
            "bhij,bhj->bhi", normalized, residual_prediction
        )
        mix = torch.sigmoid(self.logit_mix)
        return residual_prediction + mix * (smoothed - residual_prediction)


class DLinear(nn.Module):
    """Channel-independent DLinear forecaster for wind speed and direction."""

    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        num_features: int,
        moving_average: int = 25,
        individual: bool = True,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.num_features = num_features
        self.individual = individual
        self.decomposition = SeriesDecomposition(moving_average)

        if individual:
            self.seasonal_layers = nn.ModuleList(
                nn.Linear(sequence_length, prediction_length)
                for _ in range(num_features)
            )
            self.trend_layers = nn.ModuleList(
                nn.Linear(sequence_length, prediction_length)
                for _ in range(num_features)
            )
            for layer in [*self.seasonal_layers, *self.trend_layers]:
                nn.init.constant_(layer.weight, 1.0 / sequence_length)
                nn.init.zeros_(layer.bias)
        else:
            self.seasonal_layer = nn.Linear(sequence_length, prediction_length)
            self.trend_layer = nn.Linear(sequence_length, prediction_length)
            nn.init.constant_(self.seasonal_layer.weight, 1.0 / sequence_length)
            nn.init.constant_(self.trend_layer.weight, 1.0 / sequence_length)
            nn.init.zeros_(self.seasonal_layer.bias)
            nn.init.zeros_(self.trend_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, features], got {tuple(x.shape)}")
        seasonal, trend = self.decomposition(x)
        seasonal = seasonal.transpose(1, 2)
        trend = trend.transpose(1, 2)

        if self.individual:
            seasonal_output = torch.stack(
                [self.seasonal_layers[i](seasonal[:, i]) for i in range(self.num_features)],
                dim=1,
            )
            trend_output = torch.stack(
                [self.trend_layers[i](trend[:, i]) for i in range(self.num_features)],
                dim=1,
            )
        else:
            seasonal_output = self.seasonal_layer(seasonal)
            trend_output = self.trend_layer(trend)
        return (seasonal_output + trend_output).transpose(1, 2)


class DecomposedLinear(nn.Module):
    """DLinear-style forecaster with a selectable trend decomposition."""

    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        num_features: int,
        decomposition: str = "moving_average",
        moving_average: int = 25,
        ema_alpha: float = 0.15,
        individual: bool = True,
    ) -> None:
        super().__init__()
        method = str(decomposition).lower().strip()
        if method in {"moving_average", "ma", "dlinear"}:
            self.decomposition = SeriesDecomposition(moving_average)
            self.decomposition_name = "moving_average"
        elif method in {"ema", "exponential_moving_average"}:
            self.decomposition = _EmaSeriesDecomposition(ema_alpha)
            self.decomposition_name = "ema"
        else:
            raise ValueError(
                "decomposition must be 'moving_average' or 'ema'."
            )

        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.num_features = num_features
        self.individual = individual

        if individual:
            self.seasonal_layers = nn.ModuleList(
                nn.Linear(sequence_length, prediction_length)
                for _ in range(num_features)
            )
            self.trend_layers = nn.ModuleList(
                nn.Linear(sequence_length, prediction_length)
                for _ in range(num_features)
            )
            layers = [*self.seasonal_layers, *self.trend_layers]
        else:
            self.seasonal_layer = nn.Linear(sequence_length, prediction_length)
            self.trend_layer = nn.Linear(sequence_length, prediction_length)
            layers = [self.seasonal_layer, self.trend_layer]

        for layer in layers:
            nn.init.constant_(layer.weight, 1.0 / sequence_length)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected [batch, time, features], got {tuple(x.shape)}"
            )
        if x.shape[1:] != (self.sequence_length, self.num_features):
            raise ValueError(
                f"Expected [batch, {self.sequence_length}, {self.num_features}], "
                f"got {tuple(x.shape)}"
            )

        seasonal, trend = self.decomposition(x)
        seasonal = seasonal.transpose(1, 2)
        trend = trend.transpose(1, 2)
        if self.individual:
            seasonal_output = torch.stack(
                [
                    self.seasonal_layers[index](seasonal[:, index])
                    for index in range(self.num_features)
                ],
                dim=1,
            )
            trend_output = torch.stack(
                [
                    self.trend_layers[index](trend[:, index])
                    for index in range(self.num_features)
                ],
                dim=1,
            )
        else:
            seasonal_output = self.seasonal_layer(seasonal)
            trend_output = self.trend_layer(trend)
        return (seasonal_output + trend_output).transpose(1, 2)


class _EmaSeriesDecomposition(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.ema = ExponentialMovingAverage(alpha)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.ema(x)
        return x - trend, trend


class DLinearGeo(nn.Module):
    """DLinear with common-field, circular direction, and coordinate refinement.

    The model keeps the original DLinear linear heads. It only changes the
    representation being forecast:

    - speed is decomposed into a farm-level common signal and turbine residuals;
    - direction is forecast in the sine/cosine plane;
    - coordinate graphs smooth only the predicted residuals.
    """

    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        num_features: int,
        num_turbines: int,
        direction_representation: str = "scalar",
        speed_mean: np.ndarray | None = None,
        speed_std: np.ndarray | None = None,
        direction_mean: np.ndarray | None = None,
        direction_std: np.ndarray | None = None,
        coordinates: np.ndarray | None = None,
        decomposition: str = "moving_average",
        moving_average: int = 25,
        ema_alpha: float = 0.15,
        graph_k: int = 4,
        graph_kernel_scale: float | None = None,
        graph_strength: float = 0.5,
        direction_convention: str = "from",
        graph_temperature: float = 0.35,
        use_graph: bool = True,
    ) -> None:
        super().__init__()
        representation = str(direction_representation).lower()
        expected_features = (
            3 * num_turbines if representation == "circular"
            else 2 * num_turbines
        )
        if representation not in {"scalar", "circular"}:
            raise ValueError(
                "direction_representation must be 'scalar' or 'circular'."
            )
        if num_features != expected_features:
            raise ValueError(
                f"Expected {expected_features} features for {representation} "
                f"direction input, got {num_features}."
            )
        if num_turbines < 1:
            raise ValueError("num_turbines must be positive.")
        if use_graph and (
            coordinates is None
            or np.asarray(coordinates).ndim != 2
            or np.asarray(coordinates).shape[0] != num_turbines
            or np.asarray(coordinates).shape[1] < 2
            or not np.isfinite(np.asarray(coordinates)[:, :2]).all()
        ):
            raise ValueError(
                "DLinearGeo requires finite coordinates for every turbine "
                "when use_graph=True."
            )

        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.num_features = num_features
        self.num_turbines = num_turbines
        self.direction_representation = representation
        self.decomposition_name = str(decomposition).lower()
        self.use_graph = bool(use_graph)

        self.common_weights = nn.Parameter(torch.zeros(num_turbines))
        self.turbine_scale = nn.Parameter(torch.ones(num_turbines))
        self.turbine_bias = nn.Parameter(torch.zeros(num_turbines))

        self.speed_common = DecomposedLinear(
            sequence_length,
            prediction_length,
            1,
            decomposition=decomposition,
            moving_average=moving_average,
            ema_alpha=ema_alpha,
            individual=False,
        )
        self.speed_residual = DecomposedLinear(
            sequence_length,
            prediction_length,
            num_turbines,
            decomposition=decomposition,
            moving_average=moving_average,
            ema_alpha=ema_alpha,
            individual=True,
        )
        self.direction_common = DecomposedLinear(
            sequence_length,
            prediction_length,
            2,
            decomposition=decomposition,
            moving_average=moving_average,
            ema_alpha=ema_alpha,
            individual=True,
        )
        self.direction_residual = DecomposedLinear(
            sequence_length,
            prediction_length,
            2 * num_turbines,
            decomposition=decomposition,
            moving_average=moving_average,
            ema_alpha=ema_alpha,
            individual=True,
        )

        self.register_buffer(
            "speed_mean",
            torch.tensor(
                np.zeros(num_turbines, dtype=np.float32)
                if speed_mean is None else np.asarray(speed_mean, dtype=np.float32)
            ),
        )
        self.register_buffer(
            "speed_std",
            torch.tensor(
                np.ones(num_turbines, dtype=np.float32)
                if speed_std is None else np.asarray(speed_std, dtype=np.float32)
            ),
        )
        self.register_buffer(
            "direction_mean",
            torch.tensor(
                np.zeros(num_turbines, dtype=np.float32)
                if direction_mean is None
                else np.asarray(direction_mean, dtype=np.float32)
            ),
        )
        self.register_buffer(
            "direction_std",
            torch.tensor(
                np.ones(num_turbines, dtype=np.float32)
                if direction_std is None
                else np.asarray(direction_std, dtype=np.float32)
            ),
        )

        if self.use_graph:
            self.speed_refiner = GraphLaplacianRefiner(
                num_turbines,
                coordinates,
                k_neighbors=graph_k,
                kernel_scale=graph_kernel_scale,
                strength=graph_strength,
            )
            self.direction_refiner = DirectionalGraphRefiner(
                num_turbines,
                coordinates,
                k_neighbors=graph_k,
                kernel_scale=graph_kernel_scale,
                direction_convention=direction_convention,
                temperature=graph_temperature,
            )

    def _direction_components(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.direction_representation == "scalar":
            angle = (
                x[..., self.num_turbines :]
                * self.direction_std.view(1, 1, -1)
                + self.direction_mean.view(1, 1, -1)
            )
            return torch.cos(angle), torch.sin(angle)
        sine = x[..., self.num_turbines : 2 * self.num_turbines]
        cosine = x[..., 2 * self.num_turbines :]
        norm = torch.sqrt(sine.square() + cosine.square() + 1e-6)
        return cosine / norm, sine / norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (
            self.sequence_length,
            self.num_features,
        ):
            raise ValueError(
                f"Expected [batch, {self.sequence_length}, {self.num_features}], "
                f"got {tuple(x.shape)}"
            )

        weights = torch.softmax(self.common_weights, dim=0)
        speed = x[..., : self.num_turbines]
        common_speed = torch.einsum("bln,n->bl", speed, weights)
        speed_residual = speed - common_speed.unsqueeze(-1)
        common_prediction = self.speed_common(
            common_speed.unsqueeze(-1)
        ).squeeze(-1)
        residual_prediction = self.speed_residual(speed_residual)
        if self.use_graph:
            residual_prediction = self.speed_refiner(residual_prediction)
        speed_prediction = (
            common_prediction.unsqueeze(-1)
            * self.turbine_scale.view(1, 1, -1)
            + self.turbine_bias.view(1, 1, -1)
            + residual_prediction
        )

        cosine, sine = self._direction_components(x)
        common_cosine = torch.einsum("bln,n->bl", cosine, weights)
        common_sine = torch.einsum("bln,n->bl", sine, weights)
        common_vector = torch.stack([common_cosine, common_sine], dim=-1)
        common_vector = common_vector / torch.sqrt(
            common_vector.square().sum(dim=-1, keepdim=True) + 1e-6
        )
        direction_common_prediction = self.direction_common(common_vector)

        direction_vector = torch.stack([cosine, sine], dim=-1)
        common_vector_input = common_vector.unsqueeze(2)
        direction_residual = direction_vector - common_vector_input
        direction_residual_prediction = self.direction_residual(
            direction_residual.reshape(
                x.shape[0],
                self.sequence_length,
                2 * self.num_turbines,
            )
        )
        direction_residual_prediction = direction_residual_prediction.view(
            x.shape[0],
            self.prediction_length,
            self.num_turbines,
            2,
        )

        common_angle = torch.atan2(
            direction_common_prediction[..., 1],
            direction_common_prediction[..., 0],
        )
        if self.use_graph:
            direction_residual_prediction = torch.stack(
                [
                    self.direction_refiner(
                        direction_residual_prediction[..., component],
                        common_angle,
                    )
                    for component in range(2)
                ],
                dim=-1,
            )
        direction_prediction = (
            direction_common_prediction.unsqueeze(2)
            + direction_residual_prediction
        )
        direction_prediction = direction_prediction / torch.sqrt(
            direction_prediction.square().sum(dim=-1, keepdim=True) + 1e-6
        )

        if self.direction_representation == "circular":
            return torch.cat(
                [
                    speed_prediction,
                    direction_prediction[..., 1],
                    direction_prediction[..., 0],
                ],
                dim=-1,
            )

        angle = torch.remainder(
            torch.atan2(
                direction_prediction[..., 1],
                direction_prediction[..., 0],
            ),
            2.0 * torch.pi,
        )
        normalized_angle = (
            angle - self.direction_mean.view(1, 1, -1)
        ) / self.direction_std.view(1, 1, -1).clamp_min(1e-6)
        return torch.cat([speed_prediction, normalized_angle], dim=-1)


class SharedResidualMLP(nn.Module):
    """Shared temporal MLP applied independently to every turbine residual."""

    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        hidden_size: int = 32,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.network = nn.Sequential(
            nn.Linear(sequence_length, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, prediction_length),
        )
        nn.init.constant_(
            self.network[0].weight,
            1.0 / max(sequence_length, 1),
        )
        nn.init.zeros_(self.network[0].bias)
        nn.init.constant_(
            self.network[3].weight,
            1.0 / max(hidden_size, 1),
        )
        nn.init.zeros_(self.network[3].bias)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.ndim != 3:
            raise ValueError(
                "Expected residual [batch, time, turbines], "
                f"got {tuple(residual.shape)}"
            )
        if residual.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, "
                f"got {residual.shape[1]}"
            )
        return self.network(residual.transpose(1, 2)).transpose(1, 2)


class SharedResidualMLPEncoder(nn.Module):
    """Shared MLP encoder for [batch, time, turbine, channel] residuals."""

    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        input_dim: int,
        hidden_size: int = 32,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.input_dim = input_dim
        self.output = nn.Sequential(
            nn.Linear(sequence_length * input_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, prediction_length * input_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4:
            raise ValueError(
                "Expected residual [batch, time, turbine, channel], "
                f"got {tuple(values.shape)}"
            )
        batch, length, turbines, channels = values.shape
        if (length, channels) != (self.sequence_length, self.input_dim):
            raise ValueError(
                f"Expected time/channel ({self.sequence_length}, "
                f"{self.input_dim}), got ({length}, {channels})"
            )
        flattened = values.permute(0, 2, 1, 3).reshape(
            batch * turbines,
            length * channels,
        )
        output = self.output(flattened).view(
            batch,
            turbines,
            self.prediction_length,
            channels,
        )
        return output.permute(0, 2, 1, 3)


class SharedResidualTransformerEncoder(nn.Module):
    """Shared temporal Transformer encoder applied turbine-wise."""

    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        input_dim: int,
        d_model: int = 16,
        nhead: int = 2,
        num_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.input_dim = input_dim
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = nn.Parameter(
            torch.empty(1, sequence_length, d_model)
        )
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.head = nn.Linear(d_model, prediction_length * input_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4:
            raise ValueError(
                "Expected residual [batch, time, turbine, channel], "
                f"got {tuple(values.shape)}"
            )
        batch, length, turbines, channels = values.shape
        if (length, channels) != (self.sequence_length, self.input_dim):
            raise ValueError(
                f"Expected time/channel ({self.sequence_length}, "
                f"{self.input_dim}), got ({length}, {channels})"
            )
        flattened = values.permute(0, 2, 1, 3).reshape(
            batch * turbines,
            length,
            channels,
        )
        encoded = self.encoder(
            self.input_projection(flattened) + self.position
        )
        output = self.head(encoded[:, -1]).view(
            batch,
            turbines,
            self.prediction_length,
            channels,
        )
        return output.permute(0, 2, 1, 3)


class SharedResidualSSMEncoder(nn.Module):
    """Shared diagonal SSM followed by a feed-forward prediction head."""

    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        input_dim: int,
        d_model: int = 16,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.input_dim = input_dim
        self.input_projection = nn.Linear(input_dim, d_model)
        self.input_gate = nn.Linear(input_dim, d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.log_decay = nn.Parameter(torch.full((d_model,), -1.5))
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(d_model, prediction_length * input_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4:
            raise ValueError(
                "Expected residual [batch, time, turbine, channel], "
                f"got {tuple(values.shape)}"
            )
        batch, length, turbines, channels = values.shape
        if (length, channels) != (self.sequence_length, self.input_dim):
            raise ValueError(
                f"Expected time/channel ({self.sequence_length}, "
                f"{self.input_dim}), got ({length}, {channels})"
            )
        flattened = values.permute(0, 2, 1, 3).reshape(
            batch * turbines,
            length,
            channels,
        )
        projected = self.input_projection(flattened)
        gate = torch.sigmoid(self.input_gate(flattened))
        projected = projected * gate
        decay = torch.sigmoid(self.log_decay).view(1, -1)
        state = torch.zeros_like(projected[:, 0])
        for index in range(length):
            state = decay * state + (1.0 - decay) * projected[:, index]
        encoded = self.output_norm(state)
        encoded = encoded + self.feed_forward(encoded)
        output = self.head(encoded).view(
            batch,
            turbines,
            self.prediction_length,
            channels,
        )
        return output.permute(0, 2, 1, 3)


class ScreenedPeriodEncoder(nn.Module):
    """Multi-period phase encoder with fixed, data-screened period branches."""

    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        input_dim: int,
        periods: list[int],
        hidden_size: int = 32,
        inter_tokens: int = 8,
    ) -> None:
        super().__init__()
        normalized_periods = list(dict.fromkeys(int(value) for value in periods))
        if not normalized_periods:
            raise ValueError("At least one screened period is required.")
        if any(
            period < 2 or period > sequence_length
            for period in normalized_periods
        ):
            raise ValueError(
                "Screened periods must lie between 2 and sequence_length."
            )
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.input_dim = input_dim
        self.periods = tuple(normalized_periods)
        self.inter_tokens = inter_tokens
        self.intra_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(period * input_dim, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, prediction_length * input_dim),
                )
                for period in self.periods
            ]
        )
        self.inter_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(inter_tokens * input_dim, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, prediction_length * input_dim),
                )
                for _ in self.periods
            ]
        )
        self.period_logits = nn.Parameter(torch.zeros(len(self.periods)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError(
                f"Expected [batch, time, channel], got {tuple(values.shape)}"
            )
        batch, length, channels = values.shape
        if (length, channels) != (self.sequence_length, self.input_dim):
            raise ValueError(
                f"Expected ({self.sequence_length}, {self.input_dim}), "
                f"got ({length}, {channels})"
            )
        predictions = []
        for index, period in enumerate(self.periods):
            padded_length = int(np.ceil(length / period) * period)
            padded = F.pad(
                values,
                (0, 0, 0, padded_length - length),
            )
            folded = padded.reshape(
                batch,
                padded_length // period,
                period,
                channels,
            )
            intra = folded.mean(dim=1).reshape(batch, period * channels)
            inter = folded.mean(dim=2).transpose(1, 2)
            inter = F.adaptive_avg_pool1d(
                inter,
                self.inter_tokens,
            ).reshape(batch, self.inter_tokens * channels)
            prediction = self.intra_branches[index](intra)
            prediction = prediction + self.inter_branches[index](inter)
            predictions.append(
                prediction.view(
                    batch,
                    self.prediction_length,
                    channels,
                )
            )
        stacked = torch.stack(predictions, dim=0)
        weights = torch.softmax(self.period_logits, dim=0)
        return (stacked * weights[:, None, None, None]).sum(dim=0)


class PhaseAttentionEncoder(nn.Module):
    """Compact PhaseFormer-inspired encoder for a common-field sequence.

    Each screened period is folded into phase tokens.  A small set of learned
    routers first aggregates the cycle tokens and then distributes the
    aggregated context back to them, matching the efficient cross-phase
    routing pattern while keeping attention away from the turbine dimension.
    """

    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        input_dim: int,
        periods: list[int],
        d_model: int = 16,
        nhead: int = 2,
        num_routers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("common phase d_model must be divisible by nhead.")
        normalized_periods = list(dict.fromkeys(int(value) for value in periods))
        if not normalized_periods:
            raise ValueError("At least one phase period is required.")
        if any(
            period < 2 or period > sequence_length
            for period in normalized_periods
        ):
            raise ValueError(
                "Phase periods must lie between 2 and sequence_length."
            )
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.input_dim = input_dim
        self.periods = tuple(normalized_periods)
        self.d_model = d_model
        self.num_routers = num_routers

        self.period_branches = nn.ModuleList()
        for period in self.periods:
            num_cycles = int(np.ceil(sequence_length / period))
            self.period_branches.append(
                nn.ModuleDict(
                    {
                        "input_projection": nn.Sequential(
                            nn.Linear(period * input_dim, d_model),
                            nn.LayerNorm(d_model),
                        ),
                        "position": nn.ParameterDict(
                            {
                                "value": nn.Parameter(
                                    torch.zeros(num_cycles, d_model)
                                )
                            }
                        ),
                        "router": nn.ParameterDict(
                            {
                                "value": nn.Parameter(
                                    torch.empty(num_routers, d_model)
                                )
                            }
                        ),
                        "sender": nn.MultiheadAttention(
                            d_model,
                            nhead,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "receiver": nn.MultiheadAttention(
                            d_model,
                            nhead,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "norm1": nn.LayerNorm(d_model),
                        "norm2": nn.LayerNorm(d_model),
                        "feed_forward": nn.Sequential(
                            nn.Linear(d_model, 4 * d_model),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(4 * d_model, d_model),
                            nn.Dropout(dropout),
                        ),
                        "head": nn.Linear(
                            2 * d_model,
                            prediction_length * input_dim,
                        ),
                    }
                )
            )

        for branch in self.period_branches:
            nn.init.trunc_normal_(branch["position"]["value"], std=0.02)
            nn.init.trunc_normal_(branch["router"]["value"], std=0.02)
        self.period_logits = nn.Parameter(torch.zeros(len(self.periods)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError(
                f"Expected [batch, time, channel], got {tuple(values.shape)}"
            )
        batch, length, channels = values.shape
        if (length, channels) != (self.sequence_length, self.input_dim):
            raise ValueError(
                f"Expected ({self.sequence_length}, {self.input_dim}), "
                f"got ({length}, {channels})"
            )

        predictions = []
        for period, branch in zip(self.periods, self.period_branches):
            padded_length = int(np.ceil(length / period) * period)
            if padded_length > length:
                padded = F.pad(
                    values.transpose(1, 2),
                    (0, padded_length - length),
                    mode="circular",
                ).transpose(1, 2)
            else:
                padded = values
            num_cycles = padded_length // period
            folded = padded.reshape(
                batch,
                num_cycles,
                period,
                channels,
            )
            phase_tokens = folded.reshape(
                batch,
                num_cycles,
                period * channels,
            )
            tokens = branch["input_projection"](phase_tokens)
            tokens = tokens + branch["position"]["value"][:num_cycles].unsqueeze(0)

            routers = branch["router"]["value"].unsqueeze(0).expand(
                batch,
                -1,
                -1,
            )
            router_context, _ = branch["sender"](
                routers,
                tokens,
                tokens,
                need_weights=False,
            )
            tokens_from_router, _ = branch["receiver"](
                tokens,
                router_context,
                router_context,
                need_weights=False,
            )
            tokens = branch["norm1"](tokens + tokens_from_router)
            tokens = branch["norm2"](
                tokens + branch["feed_forward"](tokens)
            )

            summary = torch.cat(
                [tokens[:, -1], tokens.mean(dim=1)],
                dim=-1,
            )
            predictions.append(
                branch["head"](summary).view(
                    batch,
                    self.prediction_length,
                    channels,
                )
            )

        stacked = torch.stack(predictions, dim=0)
        weights = torch.softmax(self.period_logits, dim=0)
        return (stacked * weights[:, None, None, None]).sum(dim=0)


class PhaseCommonResidualGeo(nn.Module):
    """Screened-period common-field/residual model with switchable encoders."""

    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        num_features: int,
        num_turbines: int,
        direction_representation: str = "scalar",
        speed_mean: np.ndarray | None = None,
        speed_std: np.ndarray | None = None,
        direction_mean: np.ndarray | None = None,
        direction_std: np.ndarray | None = None,
        coordinates: np.ndarray | None = None,
        periods: list[int] | None = None,
        common_hidden_size: int = 32,
        common_encoder: str = "screened",
        common_speed_encoder: str | None = None,
        common_direction_encoder: str | None = None,
        common_phase_d_model: int = 16,
        common_phase_nhead: int = 2,
        common_phase_routers: int = 4,
        residual_encoder: str = "mlp",
        residual_hidden_size: int = 32,
        residual_d_model: int = 16,
        residual_nhead: int = 2,
        residual_layers: int = 1,
        residual_ffn_size: int = 64,
        residual_persistence: bool = False,
        dropout: float = 0.1,
        graph_k: int = 4,
        graph_kernel_scale: float | None = None,
        graph_strength: float = 0.5,
        direction_convention: str = "from",
        graph_temperature: float = 0.35,
        use_graph: bool = True,
    ) -> None:
        super().__init__()
        representation = str(direction_representation).lower()
        expected_features = (
            3 * num_turbines
            if representation == "circular"
            else 2 * num_turbines
        )
        if representation not in {"scalar", "circular"}:
            raise ValueError(
                "direction_representation must be 'scalar' or 'circular'."
            )
        if num_features != expected_features:
            raise ValueError(
                f"Expected {expected_features} features for {representation}, "
                f"got {num_features}."
            )
        if use_graph and (
            coordinates is None
            or np.asarray(coordinates).ndim != 2
            or np.asarray(coordinates).shape[0] != num_turbines
            or np.asarray(coordinates).shape[1] < 2
            or not np.isfinite(np.asarray(coordinates)[:, :2]).all()
        ):
            raise ValueError(
                "PhaseCommonResidualGeo requires finite coordinates when "
                "use_graph=True."
            )

        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.num_features = num_features
        self.num_turbines = num_turbines
        self.direction_representation = representation
        self.use_graph = bool(use_graph)
        self.residual_persistence = bool(residual_persistence)
        self.common_weights = nn.Parameter(torch.zeros(num_turbines))
        self.turbine_scale = nn.Parameter(torch.ones(num_turbines))
        self.turbine_bias = nn.Parameter(torch.zeros(num_turbines))
        selected_periods = periods or [6, 12]

        def make_common_encoder(input_dim: int, encoder_name: str) -> nn.Module:
            normalized = str(encoder_name).lower()
            if normalized in {"phase", "phaseformer", "phase_attention"}:
                return PhaseAttentionEncoder(
                    sequence_length,
                    prediction_length,
                    input_dim,
                    selected_periods,
                    d_model=common_phase_d_model,
                    nhead=common_phase_nhead,
                    num_routers=common_phase_routers,
                    dropout=dropout,
                )
            return ScreenedPeriodEncoder(
                sequence_length,
                prediction_length,
                input_dim,
                selected_periods,
                hidden_size=common_hidden_size,
            )

        self.common_speed_encoder = make_common_encoder(
            1,
            common_speed_encoder or common_encoder,
        )
        self.common_direction_encoder = make_common_encoder(
            2,
            common_direction_encoder or common_encoder,
        )

        def make_encoder(input_dim: int) -> nn.Module:
            normalized = str(residual_encoder).lower()
            if normalized == "mlp":
                return SharedResidualMLPEncoder(
                    sequence_length,
                    prediction_length,
                    input_dim,
                    hidden_size=residual_hidden_size,
                    dropout=dropout,
                )
            if normalized == "transformer":
                return SharedResidualTransformerEncoder(
                    sequence_length,
                    prediction_length,
                    input_dim,
                    d_model=residual_d_model,
                    nhead=residual_nhead,
                    num_layers=residual_layers,
                    dim_feedforward=residual_ffn_size,
                    dropout=dropout,
                )
            if normalized in {"ssm", "ssm_ffn"}:
                return SharedResidualSSMEncoder(
                    sequence_length,
                    prediction_length,
                    input_dim,
                    d_model=residual_d_model,
                    dim_feedforward=residual_ffn_size,
                    dropout=dropout,
                )
            raise ValueError(
                "residual_encoder must be one of: mlp, transformer, ssm."
            )

        self.speed_residual = make_encoder(1)
        self.direction_residual = make_encoder(2)
        self.register_buffer(
            "speed_mean",
            torch.tensor(
                np.zeros(num_turbines, dtype=np.float32)
                if speed_mean is None
                else np.asarray(speed_mean, dtype=np.float32)
            ),
        )
        self.register_buffer(
            "speed_std",
            torch.tensor(
                np.ones(num_turbines, dtype=np.float32)
                if speed_std is None
                else np.asarray(speed_std, dtype=np.float32)
            ),
        )
        self.register_buffer(
            "direction_mean",
            torch.tensor(
                np.zeros(num_turbines, dtype=np.float32)
                if direction_mean is None
                else np.asarray(direction_mean, dtype=np.float32)
            ),
        )
        self.register_buffer(
            "direction_std",
            torch.tensor(
                np.ones(num_turbines, dtype=np.float32)
                if direction_std is None
                else np.asarray(direction_std, dtype=np.float32)
            ),
        )
        if self.use_graph:
            self.speed_refiner = GraphLaplacianRefiner(
                num_turbines,
                coordinates,
                k_neighbors=graph_k,
                kernel_scale=graph_kernel_scale,
                strength=graph_strength,
            )
            self.direction_refiner = DirectionalGraphRefiner(
                num_turbines,
                coordinates,
                k_neighbors=graph_k,
                kernel_scale=graph_kernel_scale,
                direction_convention=direction_convention,
                temperature=graph_temperature,
            )

    def _direction_components(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.direction_representation == "scalar":
            angles = (
                values[..., self.num_turbines :]
                * self.direction_std.view(1, 1, -1)
                + self.direction_mean.view(1, 1, -1)
            )
            return torch.cos(angles), torch.sin(angles)
        sine = values[..., self.num_turbines : 2 * self.num_turbines]
        cosine = values[..., 2 * self.num_turbines :]
        norm = torch.sqrt(sine.square() + cosine.square() + 1e-6)
        return cosine / norm, sine / norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (
            self.sequence_length,
            self.num_features,
        ):
            raise ValueError(
                f"Expected [batch, {self.sequence_length}, {self.num_features}], "
                f"got {tuple(x.shape)}"
            )
        weights = torch.softmax(self.common_weights, dim=0)
        speed = x[..., : self.num_turbines]
        common_speed = torch.einsum("bln,n->bl", speed, weights)
        speed_residual = speed - common_speed.unsqueeze(-1)
        common_speed_prediction = self.common_speed_encoder(
            common_speed.unsqueeze(-1)
        ).squeeze(-1)
        speed_residual_baseline = speed_residual[:, -1:, :].expand(
            -1,
            self.prediction_length,
            -1,
        )
        speed_residual_input = speed_residual
        if self.residual_persistence:
            speed_residual_input = (
                speed_residual - speed_residual[:, -1:, :]
            )
        speed_residual_prediction = self.speed_residual(
            speed_residual_input.unsqueeze(-1)
        ).squeeze(-1)
        if self.residual_persistence:
            speed_residual_prediction = (
                speed_residual_baseline + speed_residual_prediction
            )
        if self.use_graph:
            speed_residual_prediction = self.speed_refiner(
                speed_residual_prediction
            )
        speed_prediction = (
            common_speed_prediction.unsqueeze(-1)
            * self.turbine_scale.view(1, 1, -1)
            + self.turbine_bias.view(1, 1, -1)
            + speed_residual_prediction
        )

        cosine, sine = self._direction_components(x)
        common_cosine = torch.einsum("bln,n->bl", cosine, weights)
        common_sine = torch.einsum("bln,n->bl", sine, weights)
        common_direction = torch.stack(
            [common_cosine, common_sine],
            dim=-1,
        )
        common_direction = common_direction / torch.sqrt(
            common_direction.square().sum(dim=-1, keepdim=True) + 1e-6
        )
        common_direction_prediction = self.common_direction_encoder(
            common_direction
        )
        common_direction_prediction = (
            common_direction_prediction
            / torch.sqrt(
                common_direction_prediction.square().sum(
                    dim=-1,
                    keepdim=True,
                )
                + 1e-6
            )
        )
        direction_vectors = torch.stack([cosine, sine], dim=-1)
        direction_residual = (
            direction_vectors - common_direction.unsqueeze(2)
        )
        direction_residual_baseline = direction_residual[:, -1:, :, :].expand(
            -1,
            self.prediction_length,
            -1,
            -1,
        )
        direction_residual_input = direction_residual
        if self.residual_persistence:
            direction_residual_input = (
                direction_residual - direction_residual[:, -1:, :, :]
            )
        direction_residual_prediction = self.direction_residual(
            direction_residual_input
        )
        if self.residual_persistence:
            direction_residual_prediction = (
                direction_residual_baseline + direction_residual_prediction
            )
        common_angle = torch.atan2(
            common_direction_prediction[..., 1],
            common_direction_prediction[..., 0],
        )
        if self.use_graph:
            direction_residual_prediction = torch.stack(
                [
                    self.direction_refiner(
                        direction_residual_prediction[..., component],
                        common_angle,
                    )
                    for component in range(2)
                ],
                dim=-1,
            )
        direction_prediction = (
            common_direction_prediction.unsqueeze(2)
            + direction_residual_prediction
        )
        direction_prediction = direction_prediction / torch.sqrt(
            direction_prediction.square().sum(dim=-1, keepdim=True) + 1e-6
        )
        if self.direction_representation == "circular":
            return torch.cat(
                [
                    speed_prediction,
                    direction_prediction[..., 1],
                    direction_prediction[..., 0],
                ],
                dim=-1,
            )
        angle = torch.remainder(
            torch.atan2(
                direction_prediction[..., 1],
                direction_prediction[..., 0],
            ),
            2.0 * torch.pi,
        )
        normalized_angle = (
            angle - self.direction_mean.view(1, 1, -1)
        ) / self.direction_std.view(1, 1, -1).clamp_min(1e-6)
        return torch.cat([speed_prediction, normalized_angle], dim=-1)


class DLinearSharedResidualMLPGeo(DLinearGeo):
    """DLinear common-field forecast with shared MLP residual forecasting."""

    def __init__(
        self,
        *args: Any,
        residual_hidden_size: int = 32,
        residual_dropout: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.speed_residual = SharedResidualMLP(
            sequence_length=self.sequence_length,
            prediction_length=self.prediction_length,
            hidden_size=residual_hidden_size,
            dropout=residual_dropout,
        )


class EmaFrequencyGraphNet(nn.Module):
    """EMA decomposition, spectral linear residual prediction, and graph smoothing."""

    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        num_features: int,
        ema_alpha: float = 0.15,
        graph_k: int = 4,
        graph_kernel_scale: float | None = None,
        graph_strength: float = 0.5,
        coordinates: np.ndarray | None = None,
        num_turbines: int | None = None,
    ) -> None:
        super().__init__()
        if num_features % 2 != 0:
            raise ValueError("EmaFrequencyGraphNet expects speed and direction pairs.")
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.num_features = num_features
        self.num_turbines = (
            int(num_turbines)
            if num_turbines is not None
            else num_features // 2
        )
        self.ema = ExponentialMovingAverage(ema_alpha)
        freq_bins = sequence_length // 2 + 1
        self.trend_weight = nn.Parameter(
            torch.full(
                (num_features, prediction_length, sequence_length),
                1.0 / max(sequence_length, 1),
            )
        )
        self.trend_bias = nn.Parameter(torch.zeros(num_features, prediction_length))
        self.frequency_weight = nn.Parameter(
            torch.full(
                (num_features, prediction_length, freq_bins * 2),
                1.0 / max(sequence_length, 1),
            )
        )
        self.frequency_bias = nn.Parameter(
            torch.zeros(num_features, prediction_length)
        )
        self.refiner = GraphLaplacianRefiner(
            self.num_turbines,
            coordinates,
            k_neighbors=graph_k,
            kernel_scale=graph_kernel_scale,
            strength=graph_strength,
        )

    @staticmethod
    def _apply_featurewise_linear(
        weight: torch.Tensor, bias: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum("bfi,foi->bfo", values, weight) + bias.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, features], got {tuple(x.shape)}")
        if x.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, got {x.shape[1]}"
            )
        if x.shape[2] != self.num_features:
            raise ValueError(
                f"Expected feature count {self.num_features}, got {x.shape[2]}"
            )

        trend = self.ema(x)
        residual = x - trend
        trend_values = trend.transpose(1, 2)
        trend_prediction = self._apply_featurewise_linear(
            self.trend_weight,
            self.trend_bias,
            trend_values,
        ).transpose(1, 2)

        speed_residual = residual[..., : self.num_turbines]
        spectrum = torch.fft.rfft(speed_residual, dim=1)
        spectral_values = torch.cat(
            [spectrum.real, spectrum.imag], dim=1
        ).transpose(1, 2)
        speed_weight = self.frequency_weight[
            : self.num_turbines
        ]
        speed_bias = self.frequency_bias[: self.num_turbines]
        speed_prediction = self._apply_featurewise_linear(
            speed_weight,
            speed_bias,
            spectral_values,
        ).transpose(1, 2)
        residual_prediction = torch.zeros_like(trend_prediction)
        residual_prediction[..., : self.num_turbines] = (
            self.refiner(speed_prediction)
        )

        return trend_prediction + residual_prediction


class CommonResidualGraphNet(nn.Module):
    """Efficient common-field/residual forecaster for speed and direction."""

    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        num_features: int,
        num_turbines: int,
        direction_representation: str = "scalar",
        speed_mean: np.ndarray | None = None,
        speed_std: np.ndarray | None = None,
        direction_mean: np.ndarray | None = None,
        direction_std: np.ndarray | None = None,
        coordinates: np.ndarray | None = None,
        ema_alpha: float = 0.15,
        graph_k: int = 4,
        graph_kernel_scale: float | None = None,
        direction_convention: str = "from",
        graph_temperature: float = 0.35,
    ) -> None:
        super().__init__()
        representation = str(direction_representation).lower()
        expected_features = (
            3 * num_turbines if representation == "circular" else 2 * num_turbines
        )
        if representation not in {"scalar", "circular"}:
            raise ValueError(
                "direction_representation must be 'scalar' or 'circular'."
            )
        if num_features != expected_features:
            raise ValueError(
                f"Expected {expected_features} features for {representation} "
                f"direction input, got {num_features}."
            )
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.num_features = num_features
        self.num_turbines = num_turbines
        self.direction_representation = representation
        self.ema = ExponentialMovingAverage(ema_alpha)
        freq_bins = sequence_length // 2 + 1

        self.common_weights = nn.Parameter(torch.zeros(num_turbines))
        self.turbine_scale = nn.Parameter(torch.ones(num_turbines))
        self.turbine_bias = nn.Parameter(torch.zeros(num_turbines))
        self.common_trend = nn.Linear(sequence_length, prediction_length)
        self.common_frequency = nn.Linear(2 * freq_bins, prediction_length)
        self.residual_trend = nn.Linear(sequence_length, prediction_length)
        self.residual_frequency = nn.Linear(2 * freq_bins, prediction_length)
        for layer in (
            self.common_trend,
            self.common_frequency,
            self.residual_trend,
            self.residual_frequency,
        ):
            nn.init.zeros_(layer.bias)
        nn.init.constant_(self.common_trend.weight, 1.0 / sequence_length)
        nn.init.zeros_(self.common_frequency.weight)
        nn.init.constant_(self.residual_trend.weight, 1.0 / sequence_length)
        nn.init.zeros_(self.residual_frequency.weight)

        self.direction_common = nn.Linear(sequence_length, prediction_length)
        self.direction_residual = nn.Linear(sequence_length, prediction_length)
        nn.init.constant_(self.direction_common.weight, 1.0 / sequence_length)
        nn.init.zeros_(self.direction_common.bias)
        nn.init.constant_(self.direction_residual.weight, 1.0 / sequence_length)
        nn.init.zeros_(self.direction_residual.bias)

        self.register_buffer(
            "speed_mean",
            torch.tensor(
                np.zeros(num_turbines)
                if speed_mean is None
                else np.asarray(speed_mean, dtype=np.float32)
            ),
        )
        self.register_buffer(
            "speed_std",
            torch.tensor(
                np.ones(num_turbines)
                if speed_std is None
                else np.asarray(speed_std, dtype=np.float32)
            ),
        )
        self.register_buffer(
            "direction_mean",
            torch.tensor(
                np.zeros(num_turbines)
                if direction_mean is None
                else np.asarray(direction_mean, dtype=np.float32)
            ),
        )
        self.register_buffer(
            "direction_std",
            torch.tensor(
                np.ones(num_turbines)
                if direction_std is None
                else np.asarray(direction_std, dtype=np.float32)
            ),
        )
        self.refiner = DirectionalGraphRefiner(
            num_turbines=num_turbines,
            coordinates=coordinates,
            k_neighbors=graph_k,
            kernel_scale=graph_kernel_scale,
            direction_convention=direction_convention,
            temperature=graph_temperature,
        )

    def _direction_components(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.direction_representation == "scalar":
            direction = (
                x[..., self.num_turbines :]
                * self.direction_std.view(1, 1, -1)
                + self.direction_mean.view(1, 1, -1)
            )
            return torch.cos(direction), torch.sin(direction)
        sine = x[..., self.num_turbines : 2 * self.num_turbines]
        cosine = x[..., 2 * self.num_turbines :]
        norm = torch.sqrt(sine.square() + cosine.square() + 1e-6)
        return cosine / norm, sine / norm

    @staticmethod
    def _spectral_features(values: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(values, dim=-1)
        return torch.cat([spectrum.real, spectrum.imag], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (
            self.sequence_length,
            self.num_features,
        ):
            raise ValueError(
                f"Expected [batch, {self.sequence_length}, {self.num_features}], "
                f"got {tuple(x.shape)}"
            )
        speed = x[..., : self.num_turbines]
        weights = torch.softmax(self.common_weights, dim=0)
        common = torch.einsum("bln,n->bl", speed, weights)
        common_ema = self.ema(common.unsqueeze(-1)).squeeze(-1)
        common_residual = common.unsqueeze(-1) - common_ema.unsqueeze(-1)
        common_prediction = self.common_trend(common_ema)
        common_prediction = common_prediction + self.common_frequency(
            self._spectral_features(common_residual.squeeze(-1))
        )

        residual = speed - common.unsqueeze(-1)
        residual_ema = self.ema(residual)
        residual_prediction = self.residual_trend(
            residual_ema.transpose(1, 2)
        )
        residual_prediction = residual_prediction + self.residual_frequency(
            self._spectral_features(
                (residual - residual_ema).transpose(1, 2)
            )
        )
        residual_prediction = residual_prediction.transpose(1, 2)

        cosine, sine = self._direction_components(x)
        common_cosine = torch.einsum("bln,n->bl", cosine, weights)
        common_sine = torch.einsum("bln,n->bl", sine, weights)
        common_norm = torch.sqrt(
            common_cosine.square() + common_sine.square() + 1e-6
        )
        common_cosine = common_cosine / common_norm
        common_sine = common_sine / common_norm
        common_direction = torch.stack(
            [
                self.direction_common(common_cosine),
                self.direction_common(common_sine),
            ],
            dim=-1,
        )
        local_direction = torch.stack([cosine, sine], dim=-1)
        local_common = torch.stack(
            [common_cosine, common_sine], dim=-1
        ).unsqueeze(2)
        local_direction = local_direction - local_common
        local_prediction = self.direction_residual(
            local_direction.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)
        direction_prediction = common_direction.unsqueeze(2) + local_prediction
        direction_prediction = direction_prediction / torch.sqrt(
            direction_prediction.square().sum(dim=-1, keepdim=True) + 1e-6
        )
        common_angle = torch.atan2(
            common_direction[..., 1], common_direction[..., 0]
        )

        residual_prediction = self.refiner(
            residual_prediction, common_angle
        )
        speed_prediction = (
            common_prediction.unsqueeze(-1) * self.turbine_scale
            + self.turbine_bias
            + residual_prediction
        )
        if self.direction_representation == "circular":
            return torch.cat(
                [
                    speed_prediction,
                    direction_prediction[..., 1],
                    direction_prediction[..., 0],
                ],
                dim=-1,
            )
        angle = torch.atan2(
            direction_prediction[..., 1], direction_prediction[..., 0]
        )
        angle = torch.remainder(angle, 2.0 * torch.pi)
        normalized_angle = (
            angle - self.direction_mean.view(1, 1, -1)
        ) / self.direction_std.view(1, 1, -1).clamp_min(1e-6)
        return torch.cat([speed_prediction, normalized_angle], dim=-1)


class MSTNet(nn.Module):
    """Runnable PyTorch equivalent of the paper's MTSR/IPTSP/CLIR design."""

    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        num_features: int,
        num_turbines: int,
        coordinates: np.ndarray | None,
        direction_mean: np.ndarray | None = None,
        direction_std: np.ndarray | None = None,
        num_scales: int = 3,
        top_components: int = 3,
        hidden_size: int = 32,
        upstream_turbines: int = 5,
        leading_segments: int = 3,
        dropout: float = 0.1,
        candidate_periods: list[int] | None = None,
        period_top_k: int | None = None,
    ) -> None:
        super().__init__()
        if num_features != 2 * num_turbines:
            raise ValueError(
                "MSTNet reproduction currently expects scalar speed/direction pairs."
            )
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.num_features = num_features
        self.num_turbines = num_turbines
        self.num_scales = max(int(num_scales), 1)
        self.top_components = max(int(top_components), 1)
        self.upstream_turbines = max(int(upstream_turbines), 1)
        self.leading_segments = max(int(leading_segments), 1)
        self.hidden_size = hidden_size
        self.period_top_k = max(
            int(period_top_k) if period_top_k is not None else self.num_scales,
            1,
        )
        configured_periods = (
            list(candidate_periods) if candidate_periods is not None else []
        )
        normalized_periods = []
        for value in configured_periods:
            period = int(value)
            if period < 2 or period > sequence_length:
                raise ValueError(
                    "candidate_periods must lie between 2 and "
                    f"sequence_length ({sequence_length})."
                )
            if period not in normalized_periods:
                normalized_periods.append(period)
        self.candidate_periods = tuple(normalized_periods)

        self.intra_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(sequence_length, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, prediction_length),
                )
                for _ in range(self.num_scales)
            ]
        )
        self.inter_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, prediction_length),
                )
                for _ in range(self.num_scales)
            ]
        )
        self.inter_input = nn.Linear(8, hidden_size)
        self.scale_logits = nn.Parameter(torch.zeros(self.num_scales))
        self.leading_fusion = nn.Linear(
            self.leading_segments * prediction_length,
            prediction_length,
        )
        self.refinement = nn.Sequential(
            nn.Linear(2 * prediction_length, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, prediction_length),
        )
        self.register_buffer(
            "direction_mean",
            torch.tensor(
                np.zeros(num_turbines)
                if direction_mean is None
                else np.asarray(direction_mean, dtype=np.float32)
            ),
        )
        self.register_buffer(
            "direction_std",
            torch.tensor(
                np.ones(num_turbines)
                if direction_std is None
                else np.asarray(direction_std, dtype=np.float32)
            ),
        )
        coordinate_tensor = (
            np.zeros((num_turbines, 2), dtype=np.float32)
            if coordinates is None
            else np.asarray(coordinates, dtype=np.float32)[:, :2]
        )
        if coordinate_tensor.shape != (num_turbines, 2) or not np.isfinite(
            coordinate_tensor
        ).all():
            coordinate_tensor = np.arange(
                num_turbines, dtype=np.float32
            )[:, None]
            coordinate_tensor = np.concatenate(
                [coordinate_tensor, np.zeros_like(coordinate_tensor)], axis=1
            )
        self.register_buffer(
            "coordinates",
            torch.tensor(coordinate_tensor, dtype=torch.float32),
        )

    def _select_periods(
        self, x: torch.Tensor
    ) -> tuple[list[int], torch.Tensor]:
        spectrum = torch.fft.rfft(x, dim=1).abs().mean(dim=(0, 2))
        if len(spectrum) > 1:
            spectrum[0] = 0.0
        if self.candidate_periods:
            # Limit batch-wise FFT selection to periods screened offline.
            scores = []
            for period in self.candidate_periods:
                index = max(
                    1,
                    min(
                        len(spectrum) - 1,
                        int(round(self.sequence_length / period)),
                    ),
                )
                left = max(index - 1, 1)
                right = min(index + 2, len(spectrum))
                scores.append(spectrum[left:right].max())
            scores = torch.stack(scores)
            count = min(self.period_top_k, len(self.candidate_periods))
            selected = torch.topk(scores, count).indices
            periods = [
                self.candidate_periods[int(index)]
                for index in selected.detach().cpu().tolist()
            ]
            return periods, scores[selected].to(dtype=x.dtype)
        count = min(self.top_components, max(len(spectrum) - 1, 1))
        indices = torch.topk(spectrum, count).indices
        unique_periods: list[int] = []
        period_amplitudes: list[float] = []
        for index in indices.detach().cpu().tolist():
            period = max(
                2,
                min(
                    self.sequence_length,
                    int(np.ceil(self.sequence_length / max(int(index), 1))),
                ),
            )
            if period not in unique_periods:
                unique_periods.append(period)
                period_amplitudes.append(float(spectrum[index].detach().cpu()))
        while len(unique_periods) < self.num_scales:
            fallback = max(
                2,
                min(
                    self.sequence_length,
                    self.sequence_length // (len(unique_periods) + 2),
                ),
            )
            if fallback not in unique_periods:
                unique_periods.append(fallback)
                frequency_index = max(
                    1,
                    min(
                        len(spectrum) - 1,
                        round(self.sequence_length / fallback),
                    ),
                )
                period_amplitudes.append(
                    float(spectrum[frequency_index].detach().cpu())
                )
            else:
                unique_periods.append(2)
                period_amplitudes.append(float(spectrum[1].detach().cpu()))
        periods = unique_periods[: self.num_scales]
        amplitudes = torch.tensor(
            period_amplitudes[: self.num_scales],
            device=x.device,
            dtype=x.dtype,
        )
        return periods, amplitudes

    def _temporal_backbone(self, x: torch.Tensor) -> torch.Tensor:
        periods, amplitudes = self._select_periods(x)
        predictions = []
        for scale_index, period in enumerate(periods):
            padded_length = int(
                np.ceil(self.sequence_length / period) * period
            )
            padded = F.pad(
                x,
                (0, 0, 0, padded_length - self.sequence_length),
            )
            folded = padded.reshape(
                x.shape[0],
                padded_length // period,
                period,
                self.num_features,
            ).permute(0, 3, 1, 2)
            intraperiod = folded.mean(dim=2)
            intra = self.intra_mlps[scale_index](
                F.adaptive_avg_pool1d(
                    intraperiod,
                    self.sequence_length,
                )
            )
            interperiod = folded.mean(dim=3)
            inter = self.inter_mlps[scale_index](
                self.inter_input(
                    F.adaptive_avg_pool1d(
                        interperiod,
                        8,
                    )
                )
            )
            predictions.append((intra + inter).transpose(1, 2))
        weights = torch.softmax(amplitudes + self.scale_logits, dim=0)
        return torch.stack(predictions, dim=0).mul(
            weights[:, None, None, None]
        ).sum(dim=0)

    def _upstream_indices(
        self, direction: torch.Tensor
    ) -> torch.Tensor:
        flow = torch.stack(
            [torch.cos(direction), torch.sin(direction)],
            dim=-1,
        )
        relative = (
            self.coordinates[:, None, :] - self.coordinates[None, :, :]
        )
        scores = torch.einsum("bnd,nsd->bns", flow, relative)
        diagonal = torch.eye(
            self.num_turbines,
            dtype=torch.bool,
            device=direction.device,
        )
        scores = scores.masked_fill(diagonal.unsqueeze(0), float("-inf"))
        count = min(self.upstream_turbines, max(self.num_turbines - 1, 1))
        result = torch.topk(scores, count, dim=-1).indices
        if count < self.upstream_turbines:
            padding = result[..., -1:].expand(
                -1, -1, self.upstream_turbines - count
            )
            result = torch.cat([result, padding], dim=-1)
        return result

    def _clir(
        self,
        x: torch.Tensor,
        initial: torch.Tensor,
    ) -> torch.Tensor:
        speed = x[..., : self.num_turbines]
        direction = (
            x[..., self.num_turbines :]
            * self.direction_std.view(1, 1, -1)
            + self.direction_mean.view(1, 1, -1)
        )
        direction_angle = torch.atan2(
            torch.sin(direction).mean(dim=1),
            torch.cos(direction).mean(dim=1),
        )
        upstream = self._upstream_indices(direction_angle)
        batch = x.shape[0]
        horizon = self.prediction_length
        lag_count = self.sequence_length
        lag_starts = torch.arange(
            lag_count, device=x.device
        ).mul(-1).add(self.sequence_length - horizon)
        lag_starts = lag_starts.clamp(0, self.sequence_length - horizon)
        refined = initial[..., : self.num_turbines].clone()

        for target in range(self.num_turbines):
            candidate_segments = []
            candidate_scores = []
            target_prediction = initial[..., target]
            source_ids = upstream[:, target]
            for candidate in range(self.upstream_turbines):
                source_id = source_ids[:, candidate]
                source = speed.gather(
                    2,
                    source_id.view(batch, 1, 1).expand(
                        batch, self.sequence_length, 1
                    ),
                ).squeeze(-1)
                windows = source.unfold(1, horizon, 1)
                windows = windows.gather(
                    1,
                    lag_starts.view(1, -1, 1).expand(batch, -1, horizon),
                )
                centered_source = windows - windows.mean(dim=2, keepdim=True)
                centered_target = target_prediction - target_prediction.mean(
                    dim=1, keepdim=True
                )
                numerator = (
                    centered_source * centered_target[:, None, :]
                ).sum(dim=2)
                denominator = (
                    torch.sqrt(
                        (centered_source**2).sum(dim=2) + 1e-6
                    )
                    * torch.sqrt(
                        (centered_target**2).sum(dim=1, keepdim=True) + 1e-6
                    )
                )
                candidate_segments.append(windows)
                candidate_scores.append(numerator / denominator)
            segments = torch.stack(candidate_segments, dim=1)
            scores = torch.stack(candidate_scores, dim=1)
            flat_segments = segments.reshape(
                batch, self.upstream_turbines * lag_count, horizon
            )
            flat_scores = scores.reshape(
                batch, self.upstream_turbines * lag_count
            )
            top = torch.topk(
                flat_scores,
                min(self.leading_segments, flat_scores.shape[1]),
                dim=1,
            ).indices
            selected = flat_segments.gather(
                1,
                top.unsqueeze(-1).expand(-1, -1, horizon),
            ).reshape(batch, -1)
            fused = self.leading_fusion(selected)
            correction = self.refinement(
                torch.cat([target_prediction, fused], dim=-1)
            )
            refined[..., target] = target_prediction + correction
        output = initial.clone()
        output[..., : self.num_turbines] = refined
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (
            self.sequence_length,
            self.num_features,
        ):
            raise ValueError(
                f"Expected [batch, {self.sequence_length}, {self.num_features}], "
                f"got {tuple(x.shape)}"
            )
        initial = self._temporal_backbone(x)
        return self._clir(x, initial)


@register_model("mstnet")
def _build_mstnet(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    data_config = config["data"]
    model_config = config.get("model", {})
    if prepared is None:
        raise ValueError("MSTNet requires prepared data metadata.")
    metadata = prepared["metadata"]
    num_turbines = int(metadata["num_turbines"])
    direction_mean = prepared.get("direction_mean")
    direction_std = prepared.get("direction_std")
    return MSTNet(
        sequence_length=int(data_config["lookback"]),
        prediction_length=int(data_config["horizon"]),
        num_features=num_features,
        num_turbines=num_turbines,
        coordinates=_model_coordinates(config, prepared),
        direction_mean=direction_mean,
        direction_std=direction_std,
        num_scales=int(model_config.get("num_scales", 3)),
        top_components=int(model_config.get("top_components", 3)),
        hidden_size=int(model_config.get("hidden_size", 32)),
        upstream_turbines=int(model_config.get("upstream_turbines", 5)),
        leading_segments=int(model_config.get("leading_segments", 3)),
        dropout=float(model_config.get("dropout", 0.1)),
        candidate_periods=(
            None
            if model_config.get("candidate_periods") is None
            else [int(value) for value in model_config["candidate_periods"]]
        ),
        period_top_k=(
            None
            if model_config.get("period_top_k") is None
            else int(model_config["period_top_k"])
        ),
    )


def _resolve_config_path(config: dict[str, Any], path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return resolve_project_path(candidate)


def _load_static_coordinates_from_config(
    config: dict[str, Any],
    turbines: list[str],
) -> np.ndarray | None:
    static_path = config.get("data", {}).get("static_metadata_path")
    if static_path is None:
        return None
    path = _resolve_config_path(config, static_path)
    if not path.exists():
        return None
    rows: dict[str, tuple[float, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            title = str(row.get("Alternative Title", "")).strip()
            if not title.upper().startswith("T"):
                continue
            try:
                turbine_id = f"WT{int(title[1:]):02d}"
                latitude = float(str(row.get("Latitude", "")).strip())
                longitude = float(str(row.get("Longitude", "")).strip())
            except ValueError:
                continue
            rows[turbine_id] = (latitude, longitude)
    if any(turbine not in rows for turbine in turbines):
        return None
    geodetic = np.asarray([rows[turbine] for turbine in turbines], dtype=np.float32)
    lat = np.deg2rad(geodetic[:, 0])
    lon = np.deg2rad(geodetic[:, 1])
    origin_lat = float(lat.mean())
    origin_lon = float(lon.mean())
    earth_radius_m = 6_371_000.0
    x_coord = earth_radius_m * np.cos(origin_lat) * (lon - origin_lon)
    y_coord = earth_radius_m * (lat - origin_lat)
    return np.stack([x_coord, y_coord], axis=1).astype(np.float32)


def _model_coordinates(
    config: dict[str, Any],
    prepared: dict[str, Any] | None,
) -> np.ndarray | None:
    if prepared is None:
        return None
    coordinates = prepared.get("coordinates")
    if coordinates is not None:
        coordinates = np.asarray(coordinates, dtype=np.float32)
        if coordinates.ndim == 2 and np.isfinite(coordinates[:, :2]).all():
            return coordinates
    metadata = prepared.get("metadata", {})
    turbines = metadata.get("turbines", [])
    if isinstance(turbines, list) and turbines:
        return _load_static_coordinates_from_config(config, [str(item) for item in turbines])
    return None


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        num_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.input_projection = nn.Linear(num_features, d_model)
        self.position = nn.Parameter(torch.empty(1, sequence_length, d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )
        self.value_projection = nn.Linear(d_model, num_features)
        self.temporal_projection = nn.Linear(
            sequence_length, prediction_length
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, "
                f"got {x.shape[1]}"
            )
        encoded = self.input_projection(x) + self.position
        encoded = self.encoder(encoded)
        values = self.value_projection(encoded).transpose(1, 2)
        return self.temporal_projection(values).transpose(1, 2)


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left_padding, 0)))


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation
        )
        self.conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size, dilation
        )
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        output = self.dropout(F.gelu(self.norm1(self.conv1(x))))
        output = self.dropout(F.gelu(self.norm2(self.conv2(output))))
        return F.gelu(output + residual)


class TCN(nn.Module):
    def __init__(
        self,
        prediction_length: int,
        num_features: int,
        channels: int = 64,
        kernel_size: int = 3,
        levels: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        blocks = []
        in_channels = num_features
        for level in range(levels):
            blocks.append(
                TemporalBlock(
                    in_channels,
                    channels,
                    kernel_size,
                    dilation=2**level,
                    dropout=dropout,
                )
            )
            in_channels = channels
        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(
            channels, prediction_length * num_features
        )
        self.prediction_length = prediction_length
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.network(x.transpose(1, 2))
        output = self.head(encoded[:, :, -1])
        return output.reshape(
            x.shape[0], self.prediction_length, self.num_features
        )


class PatchTST(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        num_features: int,
        patch_length: int = 16,
        stride: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        revin: bool = True,
    ) -> None:
        super().__init__()
        if patch_length > sequence_length:
            raise ValueError("patch_length cannot exceed sequence_length")
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.num_features = num_features
        self.patch_length = patch_length
        self.stride = stride
        self.revin = revin
        self.num_patches = (
            sequence_length - patch_length
        ) // stride + 1
        self.patch_projection = nn.Linear(patch_length, d_model)
        self.position = nn.Parameter(
            torch.empty(1, self.num_patches, d_model)
        )
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )
        self.head = nn.Linear(
            self.num_patches * d_model, prediction_length
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, "
                f"got {x.shape[1]}"
            )
        if self.revin:
            mean = x.mean(dim=1, keepdim=True).detach()
            scale = torch.sqrt(
                x.var(dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x = (x - mean) / scale
        else:
            mean = scale = None

        patches = x.transpose(1, 2).unfold(
            dimension=2,
            size=self.patch_length,
            step=self.stride,
        )
        batch, channels, patch_count, patch_length = patches.shape
        patches = patches.reshape(
            batch * channels, patch_count, patch_length
        )
        encoded = self.patch_projection(patches) + self.position
        encoded = self.encoder(encoded)
        output = self.head(encoded.flatten(start_dim=1))
        output = output.reshape(
            batch, channels, self.prediction_length
        ).transpose(1, 2)
        if mean is not None and scale is not None:
            output = output * scale + mean
        return output


class ITransformer(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        num_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.use_norm = use_norm
        self.token_projection = nn.Linear(sequence_length, d_model)
        self.variable_embedding = nn.Parameter(
            torch.empty(1, num_features, d_model)
        )
        nn.init.normal_(self.variable_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )
        self.head = nn.Linear(d_model, prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, "
                f"got {x.shape[1]}"
            )
        if self.use_norm:
            mean = x.mean(dim=1, keepdim=True).detach()
            scale = torch.sqrt(
                x.var(dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x = (x - mean) / scale
        else:
            mean = scale = None

        tokens = self.token_projection(x.transpose(1, 2))
        tokens = tokens + self.variable_embedding
        encoded = self.encoder(tokens)
        output = self.head(encoded).transpose(1, 2)
        if mean is not None and scale is not None:
            output = output * scale + mean
        return output


class TimeMixerBlock(nn.Module):
    """A compact multi-scale temporal mixing block.

    Each scale mixes the temporal axis independently and is then resized back
    to the original lookback. The resulting multi-scale representation is
    followed by a channel-mixing feed-forward network. This keeps the model
    fully convolution-free and makes the number of blocks explicit.
    """

    def __init__(
        self,
        sequence_length: int,
        num_features: int,
        scale_factors: tuple[int, ...],
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.scale_factors = scale_factors
        self.temporal_mixers = nn.ModuleList(
            [
                nn.Linear(
                    max(sequence_length // factor, 1),
                    max(sequence_length // factor, 1),
                )
                for factor in scale_factors
            ]
        )
        self.temporal_norm = nn.LayerNorm(num_features)
        self.channel_norm = nn.LayerNorm(num_features)
        self.channel_mixing = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_features),
            nn.Dropout(dropout),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, time, features] -> [batch, features, time]
        values = self.temporal_norm(x).transpose(1, 2)
        mixed_scales: list[torch.Tensor] = []
        for factor, mixer in zip(self.scale_factors, self.temporal_mixers):
            if factor == 1:
                reduced = values
            else:
                reduced = F.avg_pool1d(
                    values,
                    kernel_size=factor,
                    stride=factor,
                    ceil_mode=False,
                )
            reduced = mixer(reduced)
            if reduced.shape[-1] != self.sequence_length:
                reduced = F.interpolate(
                    reduced,
                    size=self.sequence_length,
                    mode="linear",
                    align_corners=False,
                )
            mixed_scales.append(reduced)

        temporal = torch.stack(mixed_scales, dim=0).mean(dim=0)
        x = x + self.residual_dropout(temporal.transpose(1, 2))
        return x + self.channel_mixing(self.channel_norm(x))


class TimeMixer(nn.Module):
    """Lightweight multi-scale mixer for multivariate forecasting."""

    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        num_features: int,
        num_layers: int = 3,
        hidden_size: int = 128,
        scale_factors: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if sequence_length < 4:
            raise ValueError("TimeMixer requires a lookback of at least 4.")
        if num_layers < 1:
            raise ValueError("TimeMixer num_layers must be positive.")
        valid_scales = tuple(
            int(factor)
            for factor in scale_factors
            if int(factor) > 0 and sequence_length // int(factor) >= 1
        )
        if not valid_scales:
            raise ValueError("TimeMixer needs at least one valid scale factor.")
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.num_features = num_features
        self.use_norm = use_norm
        self.blocks = nn.ModuleList(
            [
                TimeMixerBlock(
                    sequence_length=sequence_length,
                    num_features=num_features,
                    scale_factors=valid_scales,
                    hidden_size=hidden_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.forecast = nn.Linear(sequence_length, prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, "
                f"got {x.shape[1]}"
            )
        if self.use_norm:
            mean = x.mean(dim=1, keepdim=True).detach()
            scale = torch.sqrt(
                x.var(dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x = (x - mean) / scale
        else:
            mean = scale = None

        for block in self.blocks:
            x = block(x)
        output = self.forecast(x.transpose(1, 2)).transpose(1, 2)
        if mean is not None and scale is not None:
            output = output * scale + mean
        return output


class LinearAttention(nn.Module):
    """Kernelized multi-head attention with linear sequence complexity."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        query = self.query(x).view(
            batch, tokens, self.nhead, self.head_dim
        )
        key = self.key(x).view(batch, tokens, self.nhead, self.head_dim)
        value = self.value(x).view(batch, tokens, self.nhead, self.head_dim)
        query = F.elu(query) + 1.0
        key = F.elu(key) + 1.0

        key_value = torch.einsum("bmhd,bmhe->bhde", key, value)
        key_sum = key.sum(dim=1)
        numerator = torch.einsum("bmhd,bhde->bmhe", query, key_value)
        denominator = torch.einsum(
            "bmhd,bhd->bmh", query, key_sum
        ).unsqueeze(-1)
        attended = numerator / denominator.clamp_min(1e-6)
        attended = attended.reshape(batch, tokens, self.d_model)
        return self.output(self.dropout(attended))


class TalkingHeadAttention(nn.Module):
    """Multi-head attention with learnable pre/post head mixing."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.pre_mix = nn.Linear(nhead, nhead, bias=False)
        self.post_mix = nn.Linear(nhead, nhead, bias=False)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        qkv = self.qkv(x).view(
            batch, tokens, 3, self.nhead, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        logits = torch.matmul(
            query, key.transpose(-2, -1)
        ) / (self.head_dim**0.5)
        logits = self.pre_mix(
            logits.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)
        weights = torch.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        attended = torch.matmul(weights, value)
        attended = self.post_mix(
            attended.permute(0, 2, 1, 3)
        ).permute(0, 2, 1, 3)
        attended = attended.transpose(1, 2).reshape(
            batch, tokens, self.d_model
        )
        return self.output(attended)


class PatchEncoderBlock(nn.Module):
    """One configurable patch encoder block."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        attention_type: str,
        use_gate: bool,
        dropout: float,
        dim_feedforward: int,
    ) -> None:
        super().__init__()
        normalized_type = attention_type.lower()
        if normalized_type == "linear":
            self.attention = LinearAttention(d_model, nhead, dropout)
        elif normalized_type == "standard":
            self.attention = nn.MultiheadAttention(
                d_model,
                nhead,
                dropout=dropout,
                batch_first=True,
            )
        elif normalized_type == "talking_head":
            self.attention = TalkingHeadAttention(d_model, nhead, dropout)
        else:
            raise ValueError(
                "attention_type must be one of: linear, standard, talking_head"
            )
        self.attention_type = normalized_type
        self.use_gate = use_gate
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model, d_model) if use_gate else None
        self.residual_scale = nn.Parameter(torch.tensor(-2.1972246))
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        if self.attention_type == "standard":
            attended, _ = self.attention(
                normalized,
                normalized,
                normalized,
                need_weights=False,
            )
        else:
            attended = self.attention(normalized)
        if self.use_gate:
            if self.gate is None:
                raise RuntimeError("Gate projection is missing.")
            attended = attended * torch.sigmoid(self.gate(normalized))
        x = x + torch.sigmoid(self.residual_scale) * attended
        return x + self.feed_forward(self.norm2(x))


class PatchSSMBlock(nn.Module):
    """A small diagonal state-space encoder over patch tokens."""

    def __init__(
        self,
        d_model: int,
        dropout: float,
        dim_feedforward: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(d_model, d_model)
        self.input_gate = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.log_decay = nn.Parameter(torch.full((d_model,), -1.5))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        projected = self.input_projection(normalized)
        gate = torch.sigmoid(self.input_gate(normalized))
        projected = projected * gate
        decay = torch.sigmoid(self.log_decay).view(1, -1)
        state = torch.zeros_like(projected[:, 0])
        outputs = []
        for index in range(projected.shape[1]):
            state = decay * state + (1.0 - decay) * projected[:, index]
            outputs.append(state)
        encoded = torch.stack(outputs, dim=1)
        x = x + self.dropout(self.output_projection(encoded))
        return x + self.feed_forward(self.norm2(x))


class PatchCommonResidualNet(nn.Module):
    """Patch-based common/residual forecaster with switchable encoders."""

    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        num_features: int,
        num_turbines: int,
        direction_representation: str = "scalar",
        speed_mean: np.ndarray | None = None,
        speed_std: np.ndarray | None = None,
        direction_mean: np.ndarray | None = None,
        direction_std: np.ndarray | None = None,
        coordinates: np.ndarray | None = None,
        patch_length: int = 12,
        stride: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        attention_type: str = "linear",
        use_gate: bool = True,
        use_graph: bool = True,
        graph_k: int = 4,
        graph_kernel_scale: float | None = None,
        graph_strength: float = 0.5,
        direction_convention: str = "from",
        graph_temperature: float = 0.35,
    ) -> None:
        super().__init__()
        representation = str(direction_representation).lower()
        expected_features = (
            3 * num_turbines
            if representation == "circular"
            else 2 * num_turbines
        )
        if representation not in {"scalar", "circular"}:
            raise ValueError(
                "direction_representation must be 'scalar' or 'circular'."
            )
        if num_features != expected_features:
            raise ValueError(
                f"Expected {expected_features} features for {representation}, "
                f"got {num_features}."
            )
        if patch_length < 1 or patch_length > sequence_length:
            raise ValueError("patch_length must lie within sequence_length.")
        if stride < 1:
            raise ValueError("stride must be positive.")

        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.num_features = num_features
        self.num_turbines = num_turbines
        self.direction_representation = representation
        self.patch_length = patch_length
        self.stride = stride
        self.num_patches = (sequence_length - patch_length) // stride + 1
        self.common_weights = nn.Parameter(torch.zeros(num_turbines))
        self.turbine_scale = nn.Parameter(torch.ones(num_turbines))
        self.turbine_bias = nn.Parameter(torch.zeros(num_turbines))

        self.register_buffer(
            "speed_mean",
            torch.tensor(
                np.zeros(num_turbines)
                if speed_mean is None
                else np.asarray(speed_mean, dtype=np.float32)
            ),
        )
        self.register_buffer(
            "speed_std",
            torch.tensor(
                np.ones(num_turbines)
                if speed_std is None
                else np.asarray(speed_std, dtype=np.float32)
            ),
        )
        self.register_buffer(
            "direction_mean",
            torch.tensor(
                np.zeros(num_turbines)
                if direction_mean is None
                else np.asarray(direction_mean, dtype=np.float32)
            ),
        )
        self.register_buffer(
            "direction_std",
            torch.tensor(
                np.ones(num_turbines)
                if direction_std is None
                else np.asarray(direction_std, dtype=np.float32)
            ),
        )

        patch_channels = 3 * num_turbines + 3
        self.patch_projection = nn.Linear(
            patch_length * patch_channels,
            d_model,
        )
        self.position = nn.Parameter(
            torch.empty(1, self.num_patches, d_model)
        )
        nn.init.normal_(self.position, std=0.02)
        if attention_type.lower() == "ssm":
            self.encoder = nn.ModuleList(
                [
                    PatchSSMBlock(
                        d_model=d_model,
                        dropout=dropout,
                        dim_feedforward=dim_feedforward,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.encoder = nn.ModuleList(
                [
                    PatchEncoderBlock(
                        d_model=d_model,
                        nhead=nhead,
                        attention_type=attention_type,
                        use_gate=use_gate,
                        dropout=dropout,
                        dim_feedforward=dim_feedforward,
                    )
                    for _ in range(num_layers)
                ]
            )
        self.final_norm = nn.LayerNorm(d_model)
        flattened_size = self.num_patches * d_model
        self.common_speed_head = nn.Linear(
            flattened_size, prediction_length
        )
        self.residual_speed_head = nn.Linear(
            flattened_size, prediction_length * num_turbines
        )
        self.common_direction_head = nn.Linear(
            flattened_size, prediction_length * 2
        )
        self.residual_direction_head = nn.Linear(
            flattened_size, prediction_length * num_turbines * 2
        )

        self.use_graph = bool(use_graph)
        if self.use_graph:
            self.speed_refiner = GraphLaplacianRefiner(
                num_turbines=num_turbines,
                coordinates=coordinates,
                k_neighbors=graph_k,
                kernel_scale=graph_kernel_scale,
                strength=graph_strength,
            )
            self.direction_refiner = DirectionalGraphRefiner(
                num_turbines=num_turbines,
                coordinates=coordinates,
                k_neighbors=graph_k,
                kernel_scale=graph_kernel_scale,
                direction_convention=direction_convention,
                temperature=graph_temperature,
            )

    def _make_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        speed = x[..., : self.num_turbines]
        if self.direction_representation == "scalar":
            angle = (
                x[..., self.num_turbines :]
                * self.direction_std.view(1, 1, -1)
                + self.direction_mean.view(1, 1, -1)
            )
            sine = torch.sin(angle)
            cosine = torch.cos(angle)
        else:
            sine = x[
                ..., self.num_turbines : 2 * self.num_turbines
            ]
            cosine = x[..., 2 * self.num_turbines :]
            norm = torch.sqrt(sine.square() + cosine.square() + 1e-6)
            sine = sine / norm
            cosine = cosine / norm

        weights = torch.softmax(self.common_weights, dim=0)
        common_speed = torch.einsum("bln,n->bl", speed, weights)
        direction_vectors = torch.stack([cosine, sine], dim=-1)
        common_direction = torch.einsum(
            "blnd,n->bld", direction_vectors, weights
        )
        common_direction = common_direction / torch.sqrt(
            common_direction.square().sum(dim=-1, keepdim=True) + 1e-6
        )
        residual_speed = speed - common_speed.unsqueeze(-1)
        residual_direction = (
            direction_vectors - common_direction.unsqueeze(2)
        )
        features = torch.cat(
            [
                common_speed.unsqueeze(-1),
                residual_speed,
                common_direction,
                residual_direction[..., 0],
                residual_direction[..., 1],
            ],
            dim=-1,
        )
        patches = features.unfold(
            dimension=1,
            size=self.patch_length,
            step=self.stride,
        )
        batch, patch_count, channels, patch_length = patches.shape
        if patch_count != self.num_patches:
            raise RuntimeError("Unexpected patch count.")
        return patches.permute(0, 1, 3, 2).reshape(
            batch, patch_count, patch_length * channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected_shape = (
            x.shape[0],
            self.sequence_length,
            self.num_features,
        )
        if x.ndim != 3 or tuple(x.shape) != expected_shape:
            raise ValueError(
                f"Expected [batch, {self.sequence_length}, {self.num_features}], "
                f"got {tuple(x.shape)}"
            )
        tokens = self.patch_projection(self._make_patch_features(x))
        tokens = tokens + self.position
        for block in self.encoder:
            tokens = block(tokens)
        flattened = self.final_norm(tokens).flatten(start_dim=1)

        common_speed = self.common_speed_head(flattened).unsqueeze(-1)
        residual_speed = self.residual_speed_head(flattened).view(
            x.shape[0], self.prediction_length, self.num_turbines
        )
        common_direction = self.common_direction_head(flattened).view(
            x.shape[0], self.prediction_length, 2
        )
        residual_direction = self.residual_direction_head(flattened).view(
            x.shape[0],
            self.prediction_length,
            self.num_turbines,
            2,
        )

        common_angle = torch.atan2(
            common_direction[..., 1],
            common_direction[..., 0],
        )
        if self.use_graph:
            residual_speed = self.speed_refiner(residual_speed)
            residual_direction = torch.stack(
                [
                    self.direction_refiner(
                        residual_direction[..., component],
                        common_angle,
                    )
                    for component in range(2)
                ],
                dim=-1,
            )
        speed_prediction = (
            common_speed * self.turbine_scale.view(1, 1, -1)
            + self.turbine_bias.view(1, 1, -1)
            + residual_speed
        )
        direction_prediction = (
            common_direction.unsqueeze(2) + residual_direction
        )
        direction_prediction = direction_prediction / torch.sqrt(
            direction_prediction.square().sum(dim=-1, keepdim=True) + 1e-6
        )
        if self.direction_representation == "circular":
            return torch.cat(
                [
                    speed_prediction,
                    direction_prediction[..., 1],
                    direction_prediction[..., 0],
                ],
                dim=-1,
            )
        angle = torch.remainder(
            torch.atan2(
                direction_prediction[..., 1],
                direction_prediction[..., 0],
            ),
            2.0 * torch.pi,
        )
        normalized_angle = (
            angle - self.direction_mean.view(1, 1, -1)
        ) / self.direction_std.view(1, 1, -1).clamp_min(1e-6)
        return torch.cat([speed_prediction, normalized_angle], dim=-1)


def _build_patch_common_variant(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None,
    default_attention_type: str,
    default_gate: bool,
    default_graph: bool,
) -> nn.Module:
    if prepared is None:
        raise ValueError("PatchCommonResidualNet requires prepared metadata.")
    data_config = config["data"]
    model_config = config.get("model", {})
    metadata = prepared["metadata"]
    return PatchCommonResidualNet(
        sequence_length=int(data_config["lookback"]),
        prediction_length=int(data_config["horizon"]),
        num_features=num_features,
        num_turbines=int(metadata["num_turbines"]),
        direction_representation=str(
            metadata.get(
                "direction_representation",
                data_config.get("direction_representation", "scalar"),
            )
        ),
        speed_mean=prepared.get("speed_mean"),
        speed_std=prepared.get("speed_std"),
        direction_mean=prepared.get("direction_mean"),
        direction_std=prepared.get("direction_std"),
        coordinates=_model_coordinates(config, prepared),
        patch_length=int(model_config.get("patch_length", 12)),
        stride=int(model_config.get("stride", 6)),
        d_model=int(model_config.get("d_model", 64)),
        nhead=int(model_config.get("nhead", 4)),
        num_layers=int(model_config.get("num_layers", 1)),
        dim_feedforward=int(model_config.get("dim_feedforward", 128)),
        dropout=float(model_config.get("dropout", 0.1)),
        attention_type=str(
            model_config.get("attention_type", default_attention_type)
        ),
        use_gate=bool(model_config.get("use_gate", default_gate)),
        use_graph=bool(model_config.get("use_graph", default_graph)),
        graph_k=int(model_config.get("graph_k", 4)),
        graph_kernel_scale=(
            None
            if model_config.get("graph_kernel_scale") is None
            else float(model_config["graph_kernel_scale"])
        ),
        graph_strength=float(model_config.get("graph_strength", 0.5)),
        direction_convention=str(
            model_config.get("direction_convention", "from")
        ),
        graph_temperature=float(
            model_config.get("graph_temperature", 0.35)
        ),
    )


@register_model("patch_linear")
def _build_patch_linear(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    return _build_patch_common_variant(
        config, num_features, prepared, "linear", False, False
    )


@register_model("patch_linear_gated")
def _build_patch_linear_gated(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    return _build_patch_common_variant(
        config, num_features, prepared, "linear", True, False
    )


@register_model("patch_gated")
def _build_patch_gated(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    return _build_patch_common_variant(
        config, num_features, prepared, "standard", True, True
    )


@register_model("patch_standard")
def _build_patch_standard(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    return _build_patch_common_variant(
        config, num_features, prepared, "standard", False, True
    )


@register_model("patch_talking_head")
def _build_patch_talking_head(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    return _build_patch_common_variant(
        config, num_features, prepared, "talking_head", False, True
    )


@register_model("patch_ssm")
def _build_patch_ssm(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    return _build_patch_common_variant(
        config, num_features, prepared, "ssm", False, True
    )


@register_model("dlinear")
def _build_dlinear(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    data_config = config["data"]
    model_config = config["model"]
    return DLinear(
        sequence_length=int(data_config["lookback"]),
        prediction_length=int(data_config["horizon"]),
        num_features=num_features,
        moving_average=int(model_config.get("moving_average", 25)),
        individual=bool(model_config.get("individual", True)),
    )


def _build_dlinear_geo_variant(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None,
    decomposition: str,
    default_use_graph: bool,
) -> nn.Module:
    if prepared is None:
        raise ValueError("DLinearGeo requires prepared data metadata.")
    data_config = config["data"]
    model_config = config.get("model", {})
    metadata = prepared["metadata"]
    use_graph = bool(model_config.get("use_graph", default_use_graph))
    return DLinearGeo(
        sequence_length=int(data_config["lookback"]),
        prediction_length=int(data_config["horizon"]),
        num_features=num_features,
        num_turbines=int(metadata["num_turbines"]),
        direction_representation=str(
            metadata.get(
                "direction_representation",
                data_config.get("direction_representation", "scalar"),
            )
        ),
        speed_mean=prepared.get("speed_mean"),
        speed_std=prepared.get("speed_std"),
        direction_mean=prepared.get("direction_mean"),
        direction_std=prepared.get("direction_std"),
        coordinates=_model_coordinates(config, prepared),
        decomposition=str(
            model_config.get("decomposition", decomposition)
        ),
        moving_average=int(model_config.get("moving_average", 25)),
        ema_alpha=float(model_config.get("ema_alpha", 0.15)),
        graph_k=int(model_config.get("graph_k", 4)),
        graph_kernel_scale=(
            None
            if model_config.get("graph_kernel_scale") is None
            else float(model_config["graph_kernel_scale"])
        ),
        graph_strength=float(model_config.get("graph_strength", 0.5)),
        direction_convention=str(
            model_config.get("direction_convention", "from")
        ),
        graph_temperature=float(
            model_config.get("graph_temperature", 0.35)
        ),
        use_graph=use_graph,
    )


@register_model("dlinear_geo")
def _build_dlinear_geo(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    return _build_dlinear_geo_variant(
        config, num_features, prepared, "moving_average", True
    )


@register_model("dlinear_ema_geo")
def _build_dlinear_ema_geo(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    return _build_dlinear_geo_variant(
        config, num_features, prepared, "ema", True
    )


@register_model("dlinear_ema")
def _build_dlinear_ema(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    return _build_dlinear_geo_variant(
        config, num_features, prepared, "ema", False
    )


@register_model("dlinear_mlp_geo")
def _build_dlinear_mlp_geo(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    if prepared is None:
        raise ValueError(
            "DLinearSharedResidualMLPGeo requires prepared data metadata."
        )
    data_config = config["data"]
    model_config = config.get("model", {})
    metadata = prepared["metadata"]
    return DLinearSharedResidualMLPGeo(
        sequence_length=int(data_config["lookback"]),
        prediction_length=int(data_config["horizon"]),
        num_features=num_features,
        num_turbines=int(metadata["num_turbines"]),
        direction_representation=str(
            metadata.get(
                "direction_representation",
                data_config.get("direction_representation", "scalar"),
            )
        ),
        speed_mean=prepared.get("speed_mean"),
        speed_std=prepared.get("speed_std"),
        direction_mean=prepared.get("direction_mean"),
        direction_std=prepared.get("direction_std"),
        coordinates=_model_coordinates(config, prepared),
        decomposition="moving_average",
        moving_average=int(model_config.get("moving_average", 25)),
        ema_alpha=float(model_config.get("ema_alpha", 0.15)),
        graph_k=int(model_config.get("graph_k", 4)),
        graph_kernel_scale=(
            None
            if model_config.get("graph_kernel_scale") is None
            else float(model_config["graph_kernel_scale"])
        ),
        graph_strength=float(model_config.get("graph_strength", 0.5)),
        direction_convention=str(
            model_config.get("direction_convention", "from")
        ),
        graph_temperature=float(
            model_config.get("graph_temperature", 0.35)
        ),
        use_graph=bool(model_config.get("use_graph", True)),
        residual_hidden_size=int(
            model_config.get("residual_hidden_size", 32)
        ),
        residual_dropout=float(
            model_config.get("residual_dropout", 0.05)
        ),
    )


def _build_phase_common_residual_geo(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None,
    default_residual_encoder: str = "mlp",
) -> nn.Module:
    if prepared is None:
        raise ValueError(
            "PhaseCommonResidualGeo requires prepared data metadata."
        )
    data_config = config["data"]
    model_config = config.get("model", {})
    metadata = prepared["metadata"]
    return PhaseCommonResidualGeo(
        sequence_length=int(data_config["lookback"]),
        prediction_length=int(data_config["horizon"]),
        num_features=num_features,
        num_turbines=int(metadata["num_turbines"]),
        direction_representation=str(
            metadata.get(
                "direction_representation",
                data_config.get("direction_representation", "scalar"),
            )
        ),
        speed_mean=prepared.get("speed_mean"),
        speed_std=prepared.get("speed_std"),
        direction_mean=prepared.get("direction_mean"),
        direction_std=prepared.get("direction_std"),
        coordinates=_model_coordinates(config, prepared),
        periods=[
            int(value)
            for value in model_config.get("screened_periods", [6, 12])
        ],
        common_hidden_size=int(
            model_config.get("common_hidden_size", 32)
        ),
        common_encoder=str(
            model_config.get("common_encoder", "screened")
        ),
        common_speed_encoder=(
            None
            if model_config.get("common_speed_encoder") is None
            else str(model_config["common_speed_encoder"])
        ),
        common_direction_encoder=(
            None
            if model_config.get("common_direction_encoder") is None
            else str(model_config["common_direction_encoder"])
        ),
        common_phase_d_model=int(
            model_config.get("common_phase_d_model", 16)
        ),
        common_phase_nhead=int(
            model_config.get("common_phase_nhead", 2)
        ),
        common_phase_routers=int(
            model_config.get("common_phase_routers", 4)
        ),
        residual_encoder=str(
            model_config.get(
                "residual_encoder",
                default_residual_encoder,
            )
        ),
        residual_hidden_size=int(
            model_config.get("residual_hidden_size", 32)
        ),
        residual_d_model=int(
            model_config.get("residual_d_model", 16)
        ),
        residual_nhead=int(
            model_config.get("residual_nhead", 2)
        ),
        residual_layers=int(
            model_config.get("residual_layers", 1)
        ),
        residual_ffn_size=int(
            model_config.get("residual_ffn_size", 64)
        ),
        residual_persistence=bool(
            model_config.get("residual_persistence", False)
        ),
        dropout=float(model_config.get("dropout", 0.1)),
        graph_k=int(model_config.get("graph_k", 4)),
        graph_kernel_scale=(
            None
            if model_config.get("graph_kernel_scale") is None
            else float(model_config["graph_kernel_scale"])
        ),
        graph_strength=float(model_config.get("graph_strength", 0.5)),
        direction_convention=str(
            model_config.get("direction_convention", "from")
        ),
        graph_temperature=float(
            model_config.get("graph_temperature", 0.35)
        ),
        use_graph=bool(model_config.get("use_graph", True)),
    )


@register_model("phase_common_residual_geo")
def _build_phase_common_residual_geo_default(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    return _build_phase_common_residual_geo(
        config,
        num_features,
        prepared,
        "mlp",
    )


@register_model("phase_common_residual_mlp")
def _build_phase_common_residual_mlp(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    return _build_phase_common_residual_geo(
        config,
        num_features,
        prepared,
        "mlp",
    )


@register_model("cscd_net")
@register_model("phaseformer_common_residual_mlp")
@register_model("phase_common_residual_phase_mlp")
def _build_cscd_net(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    config = {
        **config,
        "model": {
            **config.get("model", {}),
            "common_speed_encoder": "phase_attention",
            "common_direction_encoder": "screened",
            "residual_encoder": "mlp",
        },
    }
    return _build_phase_common_residual_geo(
        config,
        num_features,
        prepared,
        "mlp",
    )


@register_model("phase_common_residual_transformer")
def _build_phase_common_residual_transformer(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    return _build_phase_common_residual_geo(
        config,
        num_features,
        prepared,
        "transformer",
    )


@register_model("phase_common_residual_ssm")
def _build_phase_common_residual_ssm(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    return _build_phase_common_residual_geo(
        config,
        num_features,
        prepared,
        "ssm",
    )


@register_model("ema_freq_graph")
def _build_ema_freq_graph(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    data_config = config["data"]
    model_config = config["model"]
    return EmaFrequencyGraphNet(
        sequence_length=int(data_config["lookback"]),
        prediction_length=int(data_config["horizon"]),
        num_features=num_features,
        ema_alpha=float(model_config.get("ema_alpha", 0.15)),
        graph_k=int(model_config.get("graph_k", 4)),
        graph_kernel_scale=(
            None
            if model_config.get("graph_kernel_scale") is None
            else float(model_config["graph_kernel_scale"])
        ),
        graph_strength=float(model_config.get("graph_strength", 0.5)),
        coordinates=_model_coordinates(config, prepared),
        num_turbines=(
            int(prepared["metadata"]["num_turbines"])
            if prepared is not None
            else None
        ),
    )


@register_model("crd_linear")
@register_model("common_residual_graph")
def _build_common_residual_graph(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    if prepared is None:
        raise ValueError("CommonResidualGraphNet requires prepared data metadata.")
    data_config = config["data"]
    model_config = config.get("model", {})
    metadata = prepared["metadata"]
    return CommonResidualGraphNet(
        sequence_length=int(data_config["lookback"]),
        prediction_length=int(data_config["horizon"]),
        num_features=num_features,
        num_turbines=int(metadata["num_turbines"]),
        direction_representation=str(
            metadata.get(
                "direction_representation",
                data_config.get("direction_representation", "scalar"),
            )
        ),
        speed_mean=prepared.get("speed_mean"),
        speed_std=prepared.get("speed_std"),
        direction_mean=prepared.get("direction_mean"),
        direction_std=prepared.get("direction_std"),
        ema_alpha=float(model_config.get("ema_alpha", 0.15)),
        graph_k=int(model_config.get("graph_k", 4)),
        graph_kernel_scale=(
            None
            if model_config.get("graph_kernel_scale") is None
            else float(model_config["graph_kernel_scale"])
        ),
        direction_convention=str(
            model_config.get("direction_convention", "from")
        ),
        graph_temperature=float(
            model_config.get("graph_temperature", 0.35)
        ),
        coordinates=_model_coordinates(config, prepared),
    )


@register_model("transformer")
def _build_transformer(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    data_config = config["data"]
    model_config = config["model"]
    return TemporalTransformer(
        sequence_length=int(data_config["lookback"]),
        prediction_length=int(data_config["horizon"]),
        num_features=num_features,
        d_model=int(model_config.get("d_model", 128)),
        nhead=int(model_config.get("nhead", 8)),
        num_layers=int(model_config.get("num_layers", 3)),
        dim_feedforward=int(
            model_config.get("dim_feedforward", 256)
        ),
        dropout=float(model_config.get("dropout", 0.1)),
    )


@register_model("tcn")
def _build_tcn(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    model_config = config["model"]
    return TCN(
        prediction_length=int(config["data"]["horizon"]),
        num_features=num_features,
        channels=int(model_config.get("channels", 64)),
        kernel_size=int(model_config.get("kernel_size", 3)),
        levels=int(model_config.get("levels", 7)),
        dropout=float(model_config.get("dropout", 0.1)),
    )


@register_model("patchtst")
def _build_patchtst(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    data_config = config["data"]
    model_config = config["model"]
    return PatchTST(
        sequence_length=int(data_config["lookback"]),
        prediction_length=int(data_config["horizon"]),
        num_features=num_features,
        patch_length=int(model_config.get("patch_length", 16)),
        stride=int(model_config.get("stride", 8)),
        d_model=int(model_config.get("d_model", 128)),
        nhead=int(model_config.get("nhead", 8)),
        num_layers=int(model_config.get("num_layers", 3)),
        dim_feedforward=int(
            model_config.get("dim_feedforward", 256)
        ),
        dropout=float(model_config.get("dropout", 0.1)),
        revin=bool(model_config.get("revin", True)),
    )


@register_model("itransformer")
def _build_itransformer(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    data_config = config["data"]
    model_config = config["model"]
    return ITransformer(
        sequence_length=int(data_config["lookback"]),
        prediction_length=int(data_config["horizon"]),
        num_features=num_features,
        d_model=int(model_config.get("d_model", 128)),
        nhead=int(model_config.get("nhead", 8)),
        num_layers=int(model_config.get("num_layers", 3)),
        dim_feedforward=int(
            model_config.get("dim_feedforward", 256)
        ),
        dropout=float(model_config.get("dropout", 0.1)),
        use_norm=bool(model_config.get("use_norm", True)),
    )


@register_model("timemixer")
def _build_timemixer(
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    data_config = config["data"]
    model_config = config["model"]
    scales = tuple(
        int(value) for value in model_config.get("scale_factors", [1, 2, 4])
    )
    return TimeMixer(
        sequence_length=int(data_config["lookback"]),
        prediction_length=int(data_config["horizon"]),
        num_features=num_features,
        num_layers=int(model_config.get("num_layers", 3)),
        hidden_size=int(model_config.get("hidden_size", 128)),
        scale_factors=scales,
        dropout=float(model_config.get("dropout", 0.1)),
        use_norm=bool(model_config.get("use_norm", True)),
    )


def build_model(
    model_name: str,
    config: dict[str, Any],
    num_features: int,
    prepared: dict[str, Any] | None = None,
) -> nn.Module:
    normalized_name = model_name.strip().lower()
    builder = MODEL_REGISTRY.get(normalized_name)
    if builder is None:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unsupported model '{model_name}'. Available models: {available}"
        )
    return builder(config, num_features, prepared)
