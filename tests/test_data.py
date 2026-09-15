import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wind_repro.data import (
    WindWindowDataset,
    _fft_denoise_by_segment,
    _load_penmanshiel_static_coordinates,
    _penmanshiel_turbine_id,
    fft_denoise_window_batch,
    load_prepared,
    load_raw_frame,
    prepare_arrays,
)


def test_window_dataset_has_no_label_leakage() -> None:
    speed = np.arange(20, dtype=np.float32).reshape(-1, 1)
    direction = (100 + np.arange(20, dtype=np.float32)).reshape(-1, 1)
    dataset = WindWindowDataset(speed, direction, lookback=4, horizon=2)
    features, target = dataset[3]
    assert features[:, 0].tolist() == [3.0, 4.0, 5.0, 6.0]
    assert target[:, 0].tolist() == [7.0, 8.0]
    assert features[:, 1].tolist() == [103.0, 104.0, 105.0, 106.0]
    assert target[:, 1].tolist() == [107.0, 108.0]


def test_window_dataset_does_not_cross_time_segments() -> None:
    speed = np.arange(12, dtype=np.float32).reshape(-1, 1)
    direction = (100 + np.arange(12, dtype=np.float32)).reshape(-1, 1)
    segments = np.array([0] * 6 + [1] * 6)
    dataset = WindWindowDataset(
        speed,
        direction,
        lookback=3,
        horizon=2,
        segment_ids=segments,
    )
    assert len(dataset) == 4
    starts = [dataset[index][0][0, 0].item() for index in range(len(dataset))]
    assert starts == [0.0, 1.0, 6.0, 7.0]


def test_fft_denoise_does_not_mix_contiguous_segments() -> None:
    first = np.sin(np.linspace(0, 4 * np.pi, 64, dtype=np.float32))
    second = np.cos(np.linspace(0, 4 * np.pi, 64, dtype=np.float32))
    values = np.concatenate([first, second]).reshape(-1, 1)
    segments = np.array([0] * len(first) + [1] * len(second))

    baseline = _fft_denoise_by_segment(values, segments, 0.5)
    changed = values.copy()
    changed[len(first) :] *= 100.0
    changed_result = _fft_denoise_by_segment(changed, segments, 0.5)

    assert np.allclose(
        baseline[: len(first)], changed_result[: len(first)], atol=1e-6
    )
    assert not np.allclose(
        baseline[len(first) :], changed_result[len(first) :]
    )


def test_window_fft_denoises_input_and_target_independently() -> None:
    features = torch.tensor(
        [[[0.0], [1.0], [0.0], [-1.0], [0.0], [0.5], [0.0], [-0.5]]]
    )
    target = torch.tensor([[[10.0], [11.0], [9.0], [10.0]]])
    filtered_features = fft_denoise_window_batch(features, 0.5)
    filtered_target = fft_denoise_window_batch(target, 0.5)

    changed_target = target * 100.0
    changed_filtered_target = fft_denoise_window_batch(changed_target, 0.5)

    assert filtered_features.shape == features.shape
    assert filtered_target.shape == target.shape
    assert torch.allclose(
        filtered_features,
        fft_denoise_window_batch(features, 0.5),
    )
    assert not torch.allclose(filtered_target, changed_filtered_target)


def test_penmanshiel_turbine_id_parser() -> None:
    assert (
        _penmanshiel_turbine_id(
            "Turbine_Data_Penmanshiel_09_2018-01-01_-_2019-01-01.csv"
        )
        == "WT09"
    )


def test_load_penmanshiel_static_coordinates(tmp_path: Path) -> None:
    static_path = tmp_path / "Penmanshiel_WT_static.csv"
    pd.DataFrame(
        {
            "Alternative Title": ["T01", "T02"],
            "Latitude": [55.9000, 55.9010],
            "Longitude": [-2.3000, -2.2980],
            "Elevation (m)": [200.0, 210.0],
        }
    ).to_csv(static_path, index=False)

    static = _load_penmanshiel_static_coordinates(static_path)

    assert static["turbine_id"].tolist() == ["WT01", "WT02"]
    assert np.allclose(static[["x", "y"]].mean(axis=0), 0.0, atol=1e-6)
    assert static["elevation"].tolist() == [200.0, 210.0]
    assert static.attrs["coordinate_metadata"]["units"] == "metres"


def test_load_raw_frame_reads_penmanshiel_zip(tmp_path: Path) -> None:
    archive_path = tmp_path / "Penmanshiel_SCADA_2018_WT01-10_test.zip"
    csv_text = "\n".join(
        [
            "# export",
            "#",
            "# Turbine: Penmanshiel 01",
            "# Time zone: UTC",
            "#",
            "# Date and time,Wind speed (m/s),Wind direction (deg),Power (kW)",
            "2018-01-01 00:00:00,5.5,270,100",
            "2018-01-01 00:10:00,6.0,275,120",
        ]
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "Turbine_Data_Penmanshiel_01_2018_test.csv", csv_text
        )

    frame = load_raw_frame(
        {
            "format": "penmanshiel_zip",
            "input_path": str(tmp_path),
            "years": [2018],
            "timestamp_column": None,
            "speed_column": None,
            "direction_column": None,
        }
    )
    assert frame["turbine_id"].tolist() == ["WT01", "WT01"]
    assert frame["wind_speed"].tolist() == [5.5, 6.0]
    assert frame["wind_direction"].tolist() == [270, 275]


def test_load_raw_frame_attaches_penmanshiel_coordinates(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "Penmanshiel_SCADA_2018_WT01-10_test.zip"
    csv_text = "\n".join(
        [
            "# export",
            "# Date and time,Wind speed (m/s),Wind direction (deg)",
            "2018-01-01 00:00:00,5.5,270",
        ]
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "Turbine_Data_Penmanshiel_01_2018_test.csv", csv_text
        )
    pd.DataFrame(
        {
            "Alternative Title": ["T01"],
            "Latitude": [55.902502],
            "Longitude": [-2.306389],
            "Elevation (m)": [212.26],
        }
    ).to_csv(tmp_path / "Penmanshiel_WT_static.csv", index=False)

    frame = load_raw_frame(
        {
            "format": "penmanshiel_zip",
            "input_path": str(tmp_path),
            "years": [2018],
        }
    )

    assert frame.loc[0, "latitude"] == 55.902502
    assert frame.loc[0, "longitude"] == -2.306389
    assert frame.loc[0, "elevation"] == 212.26
    assert frame.loc[0, "x"] == 0.0
    assert frame.loc[0, "y"] == 0.0


def test_prepare_arrays_builds_chronological_splits(tmp_path: Path) -> None:
    timestamps = pd.date_range("2025-01-01", periods=180, freq="10min")
    rows = []
    for turbine in range(2):
        for index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "timestamp": timestamp,
                    "turbine_id": f"T{turbine}",
                    "wind_speed": 5.0 + turbine + index * 0.01,
                    "wind_direction": 1.0 + turbine * 0.1 + index * 0.001,
                    "x": turbine * 100.0,
                    "y": 0.0,
                }
            )
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "prepared.npz"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    config = {
        "data": {
            "input_path": str(input_path),
            "prepared_path": str(output_path),
            "timestamp_column": "timestamp",
            "turbine_column": "turbine_id",
            "speed_column": "wind_speed",
            "direction_column": "wind_direction",
            "x_column": "x",
            "y_column": "y",
            "interval_minutes": 10,
            "lookback": 8,
            "horizon": 2,
            "split": [0.7, 0.1, 0.2],
            "max_speed": 40.0,
            "max_gap_steps": 2,
            "fft_denoise": {"enabled": False},
        }
    }
    result = prepare_arrays(config)
    prepared = load_prepared(result)
    assert prepared["metadata"]["num_turbines"] == 2
    assert prepared["metadata"]["split_lengths"] == {
        "train": 125,
        "validation": 18,
        "test": 37,
    }
    assert prepared["metadata"]["split_ranges"] == {
        "train": {
            "start": "2025-01-01 00:00:00",
            "end": "2025-01-01 20:40:00",
        },
        "validation": {
            "start": "2025-01-01 20:50:00",
            "end": "2025-01-01 23:40:00",
        },
        "test": {
            "start": "2025-01-01 23:50:00",
            "end": "2025-01-02 05:50:00",
        },
    }
    assert np.allclose(prepared["train_speed"].mean(axis=0), 0.0, atol=1e-5)
    assert not np.allclose(prepared["test_speed"].mean(axis=0), 0.0, atol=1e-2)


def test_prepare_arrays_fft_denoise_isolated_by_split(tmp_path: Path) -> None:
    timestamps = pd.date_range("2025-01-01", periods=200, freq="10min")

    def write_input(path: Path, test_offset: float) -> None:
        rows = []
        for index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "timestamp": timestamp,
                    "turbine_id": "T0",
                    "wind_speed": (
                        5.0
                        + np.sin(index / 5.0)
                        + (test_offset if index >= 160 else 0.0)
                    ),
                    "wind_direction": 2.0 + 0.2 * np.cos(index / 7.0),
                    "x": 0.0,
                    "y": 0.0,
                }
            )
        pd.DataFrame(rows).to_csv(path, index=False)

    def prepare(input_path: Path, output_path: Path) -> dict:
        return load_prepared(
            prepare_arrays(
                {
                    "data": {
                        "input_path": str(input_path),
                        "prepared_path": str(output_path),
                        "timestamp_column": "timestamp",
                        "turbine_column": "turbine_id",
                        "speed_column": "wind_speed",
                        "direction_column": "wind_direction",
                        "x_column": "x",
                        "y_column": "y",
                        "interval_minutes": 10,
                        "lookback": 8,
                        "horizon": 2,
                        "split": [0.7, 0.1, 0.2],
                        "max_speed": 1000.0,
                        "max_gap_steps": 2,
                        "fft_denoise": {
                            "enabled": True,
                            "amplitude_quantile": 0.5,
                        },
                    }
                }
            )
        )

    first_input = tmp_path / "first.csv"
    second_input = tmp_path / "second.csv"
    write_input(first_input, test_offset=0.0)
    write_input(second_input, test_offset=100.0)
    first = prepare(first_input, tmp_path / "first.npz")
    second = prepare(second_input, tmp_path / "second.npz")

    assert np.allclose(first["train_speed"], second["train_speed"], atol=1e-6)
    assert np.allclose(
        first["validation_speed"], second["validation_speed"], atol=1e-6
    )
    assert not np.allclose(first["test_speed"], second["test_speed"])
    assert first["metadata"]["fft_denoise"] == {
        "enabled": True,
        "amplitude_quantile": 0.5,
        "scope": "split_and_contiguous_segment",
        "apply_features": ["direction", "speed"],
    }


def test_prepare_arrays_defers_window_fft_until_batching(tmp_path: Path) -> None:
    timestamps = pd.date_range("2025-01-01", periods=100, freq="10min")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "turbine_id": "T0",
            "wind_speed": 5.0 + np.sin(np.arange(100) / 3.0),
            "wind_direction": 2.0 + np.cos(np.arange(100) / 5.0),
            "x": 0.0,
            "y": 0.0,
        }
    )
    input_path = tmp_path / "window_fft.csv"
    output_path = tmp_path / "window_fft.npz"
    frame.to_csv(input_path, index=False)

    prepared = load_prepared(
        prepare_arrays(
            {
                "data": {
                    "input_path": str(input_path),
                    "prepared_path": str(output_path),
                    "timestamp_column": "timestamp",
                    "turbine_column": "turbine_id",
                    "speed_column": "wind_speed",
                    "direction_column": "wind_direction",
                    "x_column": "x",
                    "y_column": "y",
                    "interval_minutes": 10,
                    "lookback": 8,
                    "horizon": 2,
                    "split": [0.7, 0.1, 0.2],
                    "max_speed": 40.0,
                    "max_gap_steps": 2,
                    "fft_denoise": {
                        "enabled": True,
                        "scope": "window",
                        "apply_to": ["input", "target"],
                        "amplitude_quantile": 0.5,
                    },
                }
            }
        )
    )

    assert prepared["metadata"]["fft_denoise"] == {
        "enabled": True,
        "amplitude_quantile": 0.5,
        "scope": "window",
        "apply_to": ["input", "target"],
        "apply_features": ["direction", "speed"],
    }
    assert np.allclose(prepared["train_speed"].mean(axis=0), 0.0, atol=1e-5)


def test_load_raw_frame_expands_wide_turbine_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "wide.csv"
    pd.DataFrame(
        {
            "ts": ["2025/1/1 0:00", "2025/1/1 0:10"],
            "wind_speed_0": [5.0, 5.5],
            "wind_direction_0": [1.0, 1.1],
            "wind_speed_1": [6.0, 6.5],
            "wind_direction_1": [2.0, 2.1],
            "wind_speed_2": [7.0, 7.5],
            "wind_direction_1.1": [3.0, 3.1],
        }
    ).to_csv(input_path, index=False)

    frame = load_raw_frame(
        {
            "input_path": str(input_path),
            "timestamp_column": None,
            "turbine_column": None,
            "speed_column": None,
            "direction_column": None,
            "x_column": None,
            "y_column": None,
        }
    )

    assert len(frame) == 6
    assert frame["turbine_id"].unique().tolist() == ["0", "1", "2"]
    assert frame.groupby("turbine_id")["wind_speed"].apply(list).to_dict() == {
        "0": [5.0, 5.5],
        "1": [6.0, 6.5],
        "2": [7.0, 7.5],
    }
    assert frame.loc[frame["turbine_id"] == "2", "wind_direction"].tolist() == [
        3.0,
        3.1,
    ]
