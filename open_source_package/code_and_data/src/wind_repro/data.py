from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import resolve_project_path


COLUMN_ALIASES = {
    "timestamp": (
        "timestamp",
        "datetime",
        "date_time",
        "dateandtime",
        "time",
        "date",
        "ts",
    ),
    "turbine": ("turbine_id", "turbine", "wtg", "asset", "unit", "id"),
    "speed": (
        "wind_speed",
        "windspeed",
        "wind speed",
        "wind_speed_avg",
        "windspeedavg",
    ),
    "direction": (
        "wind_direction",
        "winddirection",
        "wind direction",
        "wind_direction_avg",
        "winddir",
    ),
    "x": ("x", "easting", "longitude", "lon"),
    "y": ("y", "northing", "latitude", "lat"),
}


def _normalized_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _natural_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _find_wide_wind_columns(
    columns: Iterable[str],
) -> list[tuple[str, str, str]]:
    columns = list(columns)
    speed_columns: list[tuple[str, str]] = []
    direction_columns: dict[str, str] = {}
    for column in columns:
        speed_match = re.fullmatch(
            r"wind[\s_-]*speed[\s_-]+(.+)", column, flags=re.IGNORECASE
        )
        if speed_match:
            speed_columns.append((speed_match.group(1), column))
            continue
        direction_match = re.fullmatch(
            r"wind[\s_-]*direction[\s_-]+(.+)", column, flags=re.IGNORECASE
        )
        if direction_match:
            direction_columns[direction_match.group(1)] = column

    pairs = []
    used_directions: set[str] = set()
    for turbine_id, speed_column in speed_columns:
        direction_column = direction_columns.get(turbine_id)
        if direction_column in used_directions:
            direction_column = None

        # Some dataset releases contain a duplicated direction header. Pandas
        # suffixes it with ".1", so recover the intended pair from adjacency.
        if direction_column is None:
            speed_index = columns.index(speed_column)
            if speed_index + 1 < len(columns):
                adjacent = columns[speed_index + 1]
                if (
                    adjacent not in used_directions
                    and re.fullmatch(
                        r"wind[\s_-]*direction[\s_-]+(.+)",
                        adjacent,
                        flags=re.IGNORECASE,
                    )
                ):
                    direction_column = adjacent

        if direction_column is not None:
            pairs.append((turbine_id, speed_column, direction_column))
            used_directions.add(direction_column)
    return sorted(pairs, key=lambda pair: _natural_key(pair[0]))


def _find_column(
    columns: Iterable[str], explicit: str | None, kind: str, required: bool = True
) -> str | None:
    columns = list(columns)
    if explicit:
        if explicit not in columns:
            raise ValueError(f"Configured {kind} column '{explicit}' was not found.")
        return explicit

    normalized = {_normalized_column(column): column for column in columns}
    for alias in COLUMN_ALIASES[kind]:
        match = normalized.get(_normalized_column(alias))
        if match:
            return match

    tokens = {
        "timestamp": ("time", "date"),
        "turbine": ("turbine", "wtg", "asset"),
        "speed": ("wind", "speed"),
        "direction": ("wind", "direction"),
        "x": ("easting", "longitude"),
        "y": ("northing", "latitude"),
    }[kind]
    for column in columns:
        normalized_name = _normalized_column(column)
        if all(token in normalized_name for token in tokens):
            return column
    if required:
        raise ValueError(f"Could not detect a {kind} column from: {columns}")
    return None


def _read_csv_file(path: Path, data_config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    timestamp_column = _find_column(
        frame.columns, data_config.get("timestamp_column"), "timestamp"
    )
    wide_columns = _find_wide_wind_columns(frame.columns)
    explicit_long_columns = any(
        data_config.get(key)
        for key in ("speed_column", "direction_column", "turbine_column")
    )
    if len(wide_columns) > 1 and not explicit_long_columns:
        timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce")
        wide_frames = []
        for turbine_id, speed_column, direction_column in wide_columns:
            wide_frames.append(
                pd.DataFrame(
                    {
                        "timestamp": timestamps,
                        "turbine_id": turbine_id,
                        "wind_speed": pd.to_numeric(
                            frame[speed_column], errors="coerce"
                        ),
                        "wind_direction": pd.to_numeric(
                            frame[direction_column], errors="coerce"
                        ),
                        "x": np.nan,
                        "y": np.nan,
                    }
                )
            )
        return pd.concat(wide_frames, ignore_index=True).dropna(
            subset=["timestamp"]
        )

    speed_column = _find_column(
        frame.columns, data_config.get("speed_column"), "speed"
    )
    direction_column = _find_column(
        frame.columns, data_config.get("direction_column"), "direction"
    )
    turbine_column = _find_column(
        frame.columns,
        data_config.get("turbine_column"),
        "turbine",
        required=False,
    )
    x_column = _find_column(
        frame.columns, data_config.get("x_column"), "x", required=False
    )
    y_column = _find_column(
        frame.columns, data_config.get("y_column"), "y", required=False
    )

    output = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame[timestamp_column], errors="coerce"),
            "turbine_id": (
                frame[turbine_column].astype(str)
                if turbine_column
                else pd.Series(path.stem, index=frame.index, dtype=str)
            ),
            "wind_speed": pd.to_numeric(frame[speed_column], errors="coerce"),
            "wind_direction": pd.to_numeric(
                frame[direction_column], errors="coerce"
            ),
        }
    )
    output["x"] = (
        pd.to_numeric(frame[x_column], errors="coerce") if x_column else np.nan
    )
    output["y"] = (
        pd.to_numeric(frame[y_column], errors="coerce") if y_column else np.nan
    )
    return output.dropna(subset=["timestamp"])


def _load_csv_files(
    input_path: Path, data_config: dict[str, Any]
) -> pd.DataFrame:
    files = (
        [input_path]
        if input_path.is_file()
        else sorted(input_path.rglob("*.csv"))
    )
    if not files:
        raise FileNotFoundError(f"No CSV files were found under {input_path}")
    frames = [_read_csv_file(path, data_config) for path in files]
    return pd.concat(frames, ignore_index=True)


RAW_FORMAT_LOADERS: dict[
    str, Callable[[Path, dict[str, Any]], pd.DataFrame]
] = {
    "csv": _load_csv_files,
}


def load_raw_frame(data_config: dict[str, Any]) -> pd.DataFrame:
    input_path = resolve_project_path(data_config["input_path"])
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input data does not exist: {input_path}. "
            "Run generate_synthetic.py or download_data.py first."
        )
    data_format = str(data_config.get("format", "csv")).lower()
    loader = RAW_FORMAT_LOADERS.get(data_format)
    if loader is None:
        available = ", ".join(sorted(RAW_FORMAT_LOADERS))
        raise ValueError(
            f"Unsupported data format '{data_format}'. Available: {available}"
        )
    return loader(input_path, data_config)


def _fft_denoise(values: np.ndarray, amplitude_quantile: float) -> np.ndarray:
    if not 0.0 <= amplitude_quantile <= 1.0:
        raise ValueError("FFT amplitude_quantile must be between 0 and 1.")
    if len(values) < 2:
        return values.astype(np.float32, copy=True)

    spectrum = np.fft.rfft(values, axis=0)
    amplitudes = np.abs(spectrum)
    threshold = np.quantile(amplitudes[1:], amplitude_quantile, axis=0)
    spectrum[1:] = np.where(amplitudes[1:] >= threshold, spectrum[1:], 0)
    return np.fft.irfft(spectrum, n=len(values), axis=0).astype(np.float32)


def _fft_denoise_by_segment(
    values: np.ndarray,
    segment_ids: np.ndarray,
    amplitude_quantile: float,
) -> np.ndarray:
    if len(values) != len(segment_ids):
        raise ValueError("segment_ids must have one entry per time-series row.")
    if len(values) == 0:
        return values.astype(np.float32, copy=True)

    output = np.empty_like(values, dtype=np.float32)
    boundaries = np.flatnonzero(np.diff(segment_ids) != 0) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(values)]))
    for start, end in zip(starts, ends):
        output[start:end] = _fft_denoise(
            values[start:end], amplitude_quantile
        )
    return output


def fft_denoise_window_batch(
    values: torch.Tensor, amplitude_quantile: float
) -> torch.Tensor:
    """Denoise independent [batch, time, feature] windows on their device."""
    if values.ndim != 3:
        raise ValueError(
            f"Expected [batch, time, features], got {tuple(values.shape)}"
        )
    if not 0.0 <= amplitude_quantile <= 1.0:
        raise ValueError("FFT amplitude_quantile must be between 0 and 1.")
    if values.shape[1] < 2:
        return values

    spectrum = torch.fft.rfft(values, dim=1)
    amplitudes = spectrum.abs()
    threshold = torch.quantile(
        amplitudes[:, 1:, :],
        amplitude_quantile,
        dim=1,
        keepdim=True,
    )
    filtered = spectrum.clone()
    filtered[:, 1:, :] = torch.where(
        amplitudes[:, 1:, :] >= threshold,
        spectrum[:, 1:, :],
        torch.zeros_like(spectrum[:, 1:, :]),
    )
    return torch.fft.irfft(filtered, n=values.shape[1], dim=1)


def prepare_arrays(config: dict[str, Any]) -> Path:
    data_config = config["data"]
    direction_representation = str(
        data_config.get("direction_representation", "scalar")
    ).lower()
    if direction_representation not in {"scalar", "circular"}:
        raise ValueError(
            "direction_representation must be either 'scalar' or 'circular'."
        )
    frame = load_raw_frame(data_config)
    start_timestamp = data_config.get("start_timestamp")
    end_timestamp = data_config.get("end_timestamp")
    if start_timestamp is not None or end_timestamp is not None:
        start = (
            pd.Timestamp(start_timestamp)
            if start_timestamp is not None
            else frame["timestamp"].min()
        )
        end = (
            pd.Timestamp(end_timestamp)
            if end_timestamp is not None
            else frame["timestamp"].max()
        )
        if start > end:
            raise ValueError(
                "data.start_timestamp must not be later than "
                "data.end_timestamp."
            )
        frame = frame.loc[
            frame["timestamp"].between(start, end, inclusive="both")
        ].copy()
        if frame.empty:
            raise ValueError(
                "No raw rows remain after applying the configured date range."
            )
    coordinate_metadata = frame.attrs.get("coordinate_metadata")
    frame = frame.sort_values(["timestamp", "turbine_id"]).drop_duplicates(
        ["timestamp", "turbine_id"], keep="last"
    )

    max_speed = float(data_config.get("max_speed", 25.0))
    frame.loc[
        (frame["wind_speed"] < 0) | (frame["wind_speed"] > max_speed), "wind_speed"
    ] = np.nan

    finite_direction = frame["wind_direction"].dropna()
    if not finite_direction.empty and finite_direction.quantile(0.99) > 2 * np.pi + 0.1:
        frame["wind_direction"] = np.deg2rad(frame["wind_direction"])
    frame.loc[
        (frame["wind_direction"] < 0) | (frame["wind_direction"] > 2 * np.pi),
        "wind_direction",
    ] = np.nan

    turbines = sorted(frame["turbine_id"].unique().tolist(), key=_natural_key)
    interval = f"{int(data_config.get('interval_minutes', 10))}min"
    speed = frame.pivot_table(
        index="timestamp", columns="turbine_id", values="wind_speed", aggfunc="mean"
    ).reindex(columns=turbines)
    direction = frame.pivot_table(
        index="timestamp",
        columns="turbine_id",
        values="wind_direction",
        aggfunc="mean",
    ).reindex(columns=turbines)

    full_index = pd.date_range(
        min(speed.index.min(), direction.index.min()),
        max(speed.index.max(), direction.index.max()),
        freq=interval,
    )
    # Reindex the complete 10-minute timeline and interpolate every missing
    # value, including values flagged as invalid by the quality rules above.
    # This follows the paper's stated preprocessing and keeps the sequence
    # usable for a single chronological windowing pass.
    speed = speed.reindex(full_index).interpolate(
        method="linear", limit_direction="both"
    )
    direction = direction.reindex(full_index)
    if direction_representation == "circular":
        direction_sin = np.sin(direction)
        direction_cos = np.cos(direction)
        direction_sin = direction_sin.interpolate(
            method="linear", limit_direction="both"
        )
        direction_cos = direction_cos.interpolate(
            method="linear", limit_direction="both"
        )
        direction = np.arctan2(direction_sin, direction_cos)
        direction = (direction + 2 * np.pi) % (2 * np.pi)
    else:
        direction = direction.interpolate(
            method="linear", limit_direction="both"
        )
    speed = speed.to_numpy(dtype=np.float32, copy=True)
    direction = direction.to_numpy(dtype=np.float32, copy=True)
    if direction_representation == "circular":
        direction_sin = direction_sin.to_numpy(dtype=np.float32, copy=True)
        direction_cos = direction_cos.to_numpy(dtype=np.float32, copy=True)
    timestamps = full_index
    expected_delta = pd.Timedelta(interval)
    segment_ids = np.zeros(len(timestamps), dtype=np.int32)
    if len(timestamps) > 1:
        gaps = np.asarray(timestamps[1:] - timestamps[:-1] != expected_delta)
        segment_ids[1:] = np.cumsum(gaps, dtype=np.int32)

    if len(speed) == 0:
        raise ValueError("No aligned complete timestamps remain after preprocessing.")

    split = data_config.get("split", [0.7, 0.1, 0.2])
    if len(split) != 3 or not np.isclose(sum(split), 1.0):
        raise ValueError("data.split must contain three values summing to 1.")
    train_end = int(len(speed) * float(split[0]))
    validation_end = train_end + int(len(speed) * float(split[1]))

    minimum = int(data_config["lookback"]) + int(data_config["horizon"])
    lengths = [train_end, validation_end - train_end, len(speed) - validation_end]
    if min(lengths) < minimum:
        raise ValueError(
            f"Each chronological split needs at least {minimum} rows; got {lengths}."
        )

    denoise_config = data_config.get("fft_denoise", {})
    denoise_enabled = bool(denoise_config.get("enabled", False))
    denoise_quantile = float(
        denoise_config.get("amplitude_quantile", 0.1)
    )
    denoise_scope = str(
        denoise_config.get("scope", "split_and_contiguous_segment")
    ).lower()
    available_denoise_scopes = {
        "split_and_contiguous_segment",
        "window",
        "full_sequence",
    }
    if denoise_scope not in available_denoise_scopes:
        available = ", ".join(sorted(available_denoise_scopes))
        raise ValueError(
            f"Unsupported FFT denoise scope '{denoise_scope}'. "
            f"Available: {available}"
        )
    denoise_apply_to = [
        str(value).lower()
        for value in denoise_config.get("apply_to", ["input"])
    ]
    invalid_targets = set(denoise_apply_to) - {"input", "target"}
    if invalid_targets:
        invalid = ", ".join(sorted(invalid_targets))
        raise ValueError(f"Unsupported FFT apply_to values: {invalid}")
    denoise_features = {
        str(value).lower()
        for value in denoise_config.get(
            "apply_features", ["speed", "direction"]
        )
    }
    invalid_features = denoise_features - {"speed", "direction"}
    if invalid_features:
        invalid = ", ".join(sorted(invalid_features))
        raise ValueError(f"Unsupported FFT apply_features values: {invalid}")

    if denoise_enabled and denoise_scope == "full_sequence":
        if "speed" in denoise_features:
            speed = _fft_denoise(speed, denoise_quantile)
        if "direction" in denoise_features:
            if direction_representation == "scalar":
                direction = _fft_denoise(direction, denoise_quantile)
            else:
                direction_sin = _fft_denoise(direction_sin, denoise_quantile)
                direction_cos = _fft_denoise(direction_cos, denoise_quantile)
                direction = np.arctan2(direction_sin, direction_cos)
                direction = (direction + 2 * np.pi) % (2 * np.pi)
    elif denoise_enabled and denoise_scope == "split_and_contiguous_segment":
        split_boundaries = (0, train_end, validation_end, len(speed))
        for start, end in zip(split_boundaries[:-1], split_boundaries[1:]):
            split_segments = segment_ids[start:end]
            if "speed" in denoise_features:
                speed[start:end] = _fft_denoise_by_segment(
                    speed[start:end], split_segments, denoise_quantile
                )
            if "direction" in denoise_features and direction_representation == "scalar":
                direction[start:end] = _fft_denoise_by_segment(
                    direction[start:end], split_segments, denoise_quantile
                )

    speed_mean = speed[:train_end].mean(axis=0)
    speed_std = speed[:train_end].std(axis=0)
    speed_std = np.where(speed_std < 1e-6, 1.0, speed_std)
    normalized_speed = (speed - speed_mean) / speed_std
    if direction_representation == "scalar":
        direction_mean = direction[:train_end].mean(axis=0)
        direction_std = direction[:train_end].std(axis=0)
        direction_std = np.where(direction_std < 1e-6, 1.0, direction_std)
        normalized_direction = (direction - direction_mean) / direction_std
    else:
        direction_mean = np.zeros(len(turbines), dtype=np.float32)
        direction_std = np.ones(len(turbines), dtype=np.float32)
        normalized_direction = direction

    coordinates = (
        frame.groupby("turbine_id")[["x", "y"]]
        .median()
        .reindex(turbines)
        .to_numpy(dtype=np.float32)
    )
    geodetic_coordinates = np.full(
        (len(turbines), 2), np.nan, dtype=np.float32
    )
    elevations = np.full(len(turbines), np.nan, dtype=np.float32)
    if {"latitude", "longitude"}.issubset(frame.columns):
        geodetic_coordinates = (
            frame.groupby("turbine_id")[["latitude", "longitude"]]
            .median()
            .reindex(turbines)
            .to_numpy(dtype=np.float32)
        )
    if "elevation" in frame.columns:
        elevations = (
            frame.groupby("turbine_id")["elevation"]
            .median()
            .reindex(turbines)
            .to_numpy(dtype=np.float32)
        )
    metadata = {
        "turbines": turbines,
        "num_turbines": len(turbines),
        "rows": len(speed),
        "lookback": int(data_config["lookback"]),
        "horizon": int(data_config["horizon"]),
        "direction_representation": direction_representation,
        "feature_layout": (
            "speed, direction_sin, direction_cos"
            if direction_representation == "circular"
            else "speed, direction"
        ),
        "interval_minutes": int(data_config.get("interval_minutes", 10)),
        "configured_start_timestamp": (
            str(start_timestamp) if start_timestamp is not None else None
        ),
        "configured_end_timestamp": (
            str(end_timestamp) if end_timestamp is not None else None
        ),
        "split_lengths": {
            "train": lengths[0],
            "validation": lengths[1],
            "test": lengths[2],
        },
        "split_ranges": {
            "train": {
                "start": str(timestamps[0]),
                "end": str(timestamps[train_end - 1]),
            },
            "validation": {
                "start": str(timestamps[train_end]),
                "end": str(timestamps[validation_end - 1]),
            },
            "test": {
                "start": str(timestamps[validation_end]),
                "end": str(timestamps[-1]),
            },
        },
        "timestamp_start": str(timestamps[0]),
        "timestamp_end": str(timestamps[-1]),
        "segments": {
            "total": int(segment_ids[-1]) + 1,
            "train": int(len(np.unique(segment_ids[:train_end]))),
            "validation": int(
                len(np.unique(segment_ids[train_end:validation_end]))
            ),
            "test": int(len(np.unique(segment_ids[validation_end:]))),
        },
        "fft_denoise": {
            "enabled": denoise_enabled,
            "amplitude_quantile": denoise_quantile,
            "scope": denoise_scope,
            "apply_features": sorted(denoise_features),
        },
        "preprocessing": {
            "invalid_speed_threshold_mps": max_speed,
            "interpolation": "linear_unlimited_bidirectional",
            "drop_incomplete_timestamps": False,
            "fft_applied_before_windowing": denoise_enabled
            and denoise_scope == "full_sequence",
        },
    }
    if coordinate_metadata is not None:
        metadata["coordinates"] = coordinate_metadata
    if denoise_scope in {"window", "full_sequence"}:
        metadata["fft_denoise"]["apply_to"] = (
            ["input", "target"]
            if denoise_scope == "full_sequence" and denoise_enabled
            else denoise_apply_to
        )

    output_path = resolve_project_path(data_config["prepared_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        train_speed=normalized_speed[:train_end],
        train_segment_ids=segment_ids[:train_end],
        validation_speed=normalized_speed[train_end:validation_end],
        validation_segment_ids=segment_ids[train_end:validation_end],
        test_speed=normalized_speed[validation_end:],
        test_segment_ids=segment_ids[validation_end:],
        speed_mean=speed_mean,
        speed_std=speed_std,
        coordinates=coordinates,
        geodetic_coordinates=geodetic_coordinates,
        elevations=elevations,
        metadata=np.array(json.dumps(metadata)),
    )
    if direction_representation == "scalar":
        with np.load(output_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays.update(
            train_direction=normalized_direction[:train_end],
            validation_direction=normalized_direction[train_end:validation_end],
            test_direction=normalized_direction[validation_end:],
            direction_mean=direction_mean,
            direction_std=direction_std,
        )
        np.savez_compressed(output_path, **arrays)
    else:
        with np.load(output_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays.update(
            train_direction_sin=direction_sin[:train_end],
            train_direction_cos=direction_cos[:train_end],
            validation_direction_sin=direction_sin[train_end:validation_end],
            validation_direction_cos=direction_cos[train_end:validation_end],
            test_direction_sin=direction_sin[validation_end:],
            test_direction_cos=direction_cos[validation_end:],
            direction_mean=direction_mean,
            direction_std=direction_std,
        )
        np.savez_compressed(output_path, **arrays)
    return output_path


class WindWindowDataset(Dataset):
    def __init__(
        self,
        speed: np.ndarray,
        direction: np.ndarray,
        lookback: int,
        horizon: int,
        segment_ids: np.ndarray | None = None,
        direction_cos: np.ndarray | None = None,
    ) -> None:
        if speed.shape != direction.shape:
            raise ValueError("Speed and direction arrays must have matching shapes.")
        if direction_cos is None:
            self.values = np.concatenate([speed, direction], axis=1).astype(
                np.float32
            )
        else:
            if direction_cos.shape != direction.shape:
                raise ValueError(
                    "Direction sine and cosine arrays must have matching shapes."
                )
            self.values = np.concatenate(
                [speed, direction, direction_cos], axis=1
            ).astype(np.float32)
        self.lookback = lookback
        self.horizon = horizon
        candidate_count = len(self.values) - lookback - horizon + 1
        if candidate_count <= 0:
            raise ValueError("The split is too short for the requested window sizes.")
        if segment_ids is None:
            self.starts = np.arange(candidate_count, dtype=np.int64)
        else:
            segment_ids = np.asarray(segment_ids)
            if len(segment_ids) != len(self.values):
                raise ValueError(
                    "segment_ids must have one entry per time-series row."
                )
            span = lookback + horizon
            candidates = np.arange(candidate_count, dtype=np.int64)
            self.starts = candidates[
                segment_ids[candidates] == segment_ids[candidates + span - 1]
            ]
        self.length = len(self.starts)
        if self.length <= 0:
            raise ValueError(
                "No windows remain after excluding samples that cross time gaps."
            )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[index])
        middle = start + self.lookback
        end = middle + self.horizon
        return (
            torch.from_numpy(self.values[start:middle]),
            torch.from_numpy(self.values[middle:end]),
        )


def load_prepared(path: str | Path) -> dict[str, Any]:
    prepared_path = resolve_project_path(path)
    with np.load(prepared_path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    result["metadata"] = json.loads(str(result["metadata"].item()))
    result["path"] = str(prepared_path)
    return result


def make_dataset(
    prepared: dict[str, Any], split: str, lookback: int, horizon: int
) -> WindWindowDataset:
    direction_representation = prepared["metadata"].get(
        "direction_representation", "scalar"
    )
    return WindWindowDataset(
        prepared[f"{split}_speed"],
        prepared[
            f"{split}_direction_sin"
            if direction_representation == "circular"
            else f"{split}_direction"
        ],
        lookback,
        horizon,
        prepared.get(f"{split}_segment_ids"),
        (
            prepared[f"{split}_direction_cos"]
            if direction_representation == "circular"
            else None
        ),
    )


def inverse_transform(
    values: np.ndarray, prepared: dict[str, Any]
) -> np.ndarray:
    num_turbines = int(prepared["metadata"]["num_turbines"])
    direction_representation = prepared["metadata"].get(
        "direction_representation", "scalar"
    )
    output = np.empty_like(values, dtype=np.float32)
    output[..., :num_turbines] = (
        values[..., :num_turbines] * prepared["speed_std"]
        + prepared["speed_mean"]
    )
    if direction_representation == "circular":
        sine = values[..., num_turbines : 2 * num_turbines]
        cosine = values[..., 2 * num_turbines : 3 * num_turbines]
        output = np.concatenate(
            [
                output[..., :num_turbines],
                np.arctan2(sine, cosine) % (2 * np.pi),
            ],
            axis=-1,
        ).astype(np.float32)
    else:
        output[..., num_turbines:] = (
            values[..., num_turbines:] * prepared["direction_std"]
            + prepared["direction_mean"]
        )
    return output
