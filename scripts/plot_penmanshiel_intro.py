from __future__ import annotations

import argparse
import json
import math
import os
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import _bootstrap
from wind_repro.config import load_config, resolve_project_path
from wind_repro.data import load_raw_frame


FIGURE_NAMES = {
    1: "01_common_field",
    11: "01_1_frequency_wavelet",
    2: "02_similarity",
    3: "03_low_rank",
    4: "04_dynamic_sync",
    5: "05_embedding",
    7: "07_direction_geometry",
    8: "08_phase_tokens",
}


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _save_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _figure_dir(root: Path, number: int) -> tuple[Path, Path, Path, Path]:
    folder = root / FIGURE_NAMES[number]
    data = folder / "data"
    png = folder / "png"
    pdf = folder / "pdf"
    folder.mkdir(parents=True, exist_ok=True)
    data.mkdir(exist_ok=True)
    png.mkdir(exist_ok=True)
    return folder, data, png, pdf


def _save_figure(
    fig: plt.Figure,
    png_base: Path,
    pdf_base: Path | None = None,
    tight_rect: tuple[float, float, float, float] | None = None,
    apply_tight_layout: bool = True,
) -> None:
    if apply_tight_layout:
        if tight_rect is None:
            fig.tight_layout()
        else:
            fig.tight_layout(rect=tight_rect)
    fig.savefig(png_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _natural_turbines(values: Iterable[str]) -> list[str]:
    return sorted(
        {str(value) for value in values},
        key=lambda value: (0, int(value[2:])) if value.upper().startswith("WT") and value[2:].isdigit() else (1, value),
    )


def _prepare_frame(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data_config = dict(config["data"])
    raw = load_raw_frame(data_config)
    raw = raw.sort_values(["timestamp", "turbine_id"]).drop_duplicates(
        ["timestamp", "turbine_id"], keep="last"
    )

    raw["wind_speed"] = pd.to_numeric(raw["wind_speed"], errors="coerce")
    raw.loc[(raw["wind_speed"] < 0) | (raw["wind_speed"] > 25), "wind_speed"] = np.nan
    raw["wind_direction"] = pd.to_numeric(raw["wind_direction"], errors="coerce")

    turbines = _natural_turbines(raw["turbine_id"])
    interval_minutes = int(data_config.get("interval_minutes", 10))
    frequency = f"{interval_minutes}min"

    speed = raw.pivot_table(
        index="timestamp",
        columns="turbine_id",
        values="wind_speed",
        aggfunc="mean",
    ).reindex(columns=turbines)
    direction = raw.pivot_table(
        index="timestamp",
        columns="turbine_id",
        values="wind_direction",
        aggfunc="mean",
    ).reindex(columns=turbines)
    full_index = pd.date_range(
        min(speed.index.min(), direction.index.min()),
        max(speed.index.max(), direction.index.max()),
        freq=frequency,
    )
    speed = speed.reindex(full_index)
    direction = direction.reindex(full_index)
    speed = speed.interpolate(method="linear", limit_direction="both")

    direction_values = direction.to_numpy(dtype=np.float64)
    direction_finite = direction_values[np.isfinite(direction_values)]
    direction_in_degrees = bool(
        direction_finite.size and np.nanpercentile(direction_finite, 99) > 2 * np.pi + 0.1
    )
    if direction_in_degrees:
        direction_values = np.deg2rad(direction_values)
    direction_values = (direction_values + 2 * np.pi) % (2 * np.pi)
    direction_sin = pd.DataFrame(
        np.sin(direction_values), index=full_index, columns=turbines
    ).interpolate(method="linear", limit_direction="both")
    direction_cos = pd.DataFrame(
        np.cos(direction_values), index=full_index, columns=turbines
    ).interpolate(method="linear", limit_direction="both")
    direction_values = np.arctan2(
        direction_sin.to_numpy(), direction_cos.to_numpy()
    )
    direction_values = (direction_values + 2 * np.pi) % (2 * np.pi)
    direction = pd.DataFrame(direction_values, index=full_index, columns=turbines)

    static = raw.reindex(
        columns=["turbine_id", "latitude", "longitude", "x", "y", "elevation"]
    ).drop_duplicates("turbine_id")
    static = static.set_index("turbine_id").reindex(turbines).reset_index()

    metadata = {
        "dataset": data_config.get("dataset_name", "Penmanshiel"),
        "years": data_config.get("years"),
        "interval_minutes": interval_minutes,
        "rows": len(full_index),
        "num_turbines": len(turbines),
        "turbines": turbines,
        "timestamp_start": full_index.min(),
        "timestamp_end": full_index.max(),
        "direction_source_was_degrees": direction_in_degrees,
        "coordinate_columns": ["latitude", "longitude", "x", "y", "elevation"],
    }
    speed.attrs["metadata"] = metadata
    direction.attrs["metadata"] = metadata
    return speed, direction, static


def _select_window_index(
    speed: pd.DataFrame, samples_per_day: int, days: int = 14
) -> tuple[slice, dict]:
    """Choose a representative complete window for visual frequency analysis."""
    window = max(samples_per_day * days, 2)
    if len(speed) <= window:
        return slice(0, len(speed)), {
            "selection": "full_sequence_shorter_than_requested_window"
        }

    values = speed.to_numpy(dtype=np.float64)
    stride = max(samples_per_day * 7, 1)
    best_start = 0
    best_score = (-1, -np.inf, -np.inf)
    for start in range(0, len(speed) - window + 1, stride):
        block = values[start : start + window]
        std = np.nanstd(block, axis=0)
        active = std > 0.25
        score = (
            int(active.sum()),
            float(np.nanmean(std[active])) if active.any() else 0.0,
            float(np.nanmean(np.nanstd(block, axis=1))),
        )
        if score > best_score:
            best_start = start
            best_score = score
    return slice(best_start, best_start + window), {
        "selection": "maximum_active_turbines_then_variability",
        "start_row": int(best_start),
        "window_rows": int(window),
        "active_turbines_std_gt_0.25": int(best_score[0]),
        "mean_active_turbine_std": float(best_score[1]),
    }


def _decode_cached_timestamps(values: np.ndarray) -> pd.DatetimeIndex:
    """Decode caches created with ns/us/ms/s datetime integer storage."""
    values = np.asarray(values)
    magnitude = float(np.nanmedian(np.abs(values.astype(np.float64))))
    if magnitude >= 1e17:
        unit = "ns"
    elif magnitude >= 1e14:
        unit = "us"
    elif magnitude >= 1e11:
        unit = "ms"
    else:
        unit = "s"
    return pd.to_datetime(values, unit=unit)


def _off_diagonal(values: np.ndarray) -> np.ndarray:
    mask = ~np.eye(values.shape[0], dtype=bool)
    return values[mask]


def _common_and_residual(speed: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    common = speed.mean(axis=1)
    residual = speed.sub(common, axis=0)
    return common, residual


def _fft_power_profiles(
    window: pd.DataFrame, common_window: pd.Series, interval_minutes: int
) -> pd.DataFrame:
    """Return normalized one-sided power spectra in period-hour coordinates."""
    values = np.column_stack(
        [window.to_numpy(dtype=np.float64), common_window.to_numpy(dtype=np.float64)]
    )
    values = values - values.mean(axis=0, keepdims=True)
    spectrum = np.abs(np.fft.rfft(values, axis=0)) ** 2
    frequencies = np.fft.rfftfreq(len(window), d=interval_minutes / 60.0)
    valid = frequencies > 0
    periods = 1.0 / frequencies[valid]
    power = spectrum[valid]
    power = power / np.maximum(power.sum(axis=0, keepdims=True), 1e-12)
    order = np.argsort(periods)
    columns = list(window.columns) + ["common_field"]
    output = pd.DataFrame(
        power[order],
        columns=columns,
    )
    output.insert(0, "period_hours", periods[order])
    return output


def _wavelet_scale_energy(
    window: pd.DataFrame, common_window: pd.Series, interval_minutes: int
) -> pd.DataFrame:
    """Return normalized Morlet CWT energy over interpretable time scales."""
    sampling_hours = interval_minutes / 60.0
    target_periods = np.geomspace(1.0, min(24.0 * 7.0, len(window) * sampling_hours / 3.0), 36)
    wavelet = "morl"
    central_frequency = pywt.central_frequency(wavelet)
    scales = central_frequency * target_periods / sampling_hours
    scales = np.maximum(1.0, np.unique(np.round(scales, decimals=4)))
    periods = pywt.scale2frequency(wavelet, scales) ** -1 * sampling_hours
    values = np.column_stack(
        [window.to_numpy(dtype=np.float64), common_window.to_numpy(dtype=np.float64)]
    )
    values = values - values.mean(axis=0, keepdims=True)
    energy = []
    for column in range(values.shape[1]):
        coefficients, _ = pywt.cwt(
            values[:, column],
            scales,
            wavelet,
        )
        profile = np.mean(np.abs(coefficients) ** 2, axis=1)
        energy.append(profile / max(float(profile.sum()), 1e-12))
    output = pd.DataFrame(
        np.column_stack(energy),
        columns=list(window.columns) + ["common_field"],
    )
    output.insert(0, "period_hours", periods)
    return output.sort_values("period_hours").reset_index(drop=True)


def _profile_similarity(table: pd.DataFrame, profile_name: str) -> pd.DataFrame:
    profiles = table.drop(columns=["period_hours"]).to_numpy(dtype=np.float64).T
    similarity = np.corrcoef(profiles)
    names = list(table.columns[1:])
    return pd.DataFrame(similarity, index=names, columns=names).rename_axis(
        profile_name
    ).reset_index()


def _plot_figure_1(root: Path, speed: pd.DataFrame, common: pd.Series) -> dict:
    folder, data_dir, png_dir, pdf_dir = _figure_dir(root, 1)
    interval = int(speed.attrs["metadata"]["interval_minutes"])
    per_day = int(round(24 * 60 / interval))
    selected, window_metadata = _select_window_index(speed, per_day, days=14)
    window = speed.iloc[selected]
    common_window = common.iloc[selected]
    output = window.copy()
    output.insert(0, "timestamp", window.index)
    output["common_field"] = common_window.to_numpy()
    for turbine in speed.columns:
        output[f"residual_{turbine}"] = (
            window[turbine].to_numpy() - common_window.to_numpy()
        )
    _save_csv(data_dir / "figure1_common_field_window.csv", output.reset_index(drop=True))

    corr = speed.corr()
    _save_csv(
        data_dir / "figure1_pearson_correlation.csv",
        corr.rename_axis("turbine").reset_index(),
    )
    pca = PCA().fit(speed.to_numpy(dtype=np.float64))
    pca_table = pd.DataFrame(
        {
            "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    )
    _save_csv(data_dir / "figure1_pca_explained_variance.csv", pca_table)

    fft_profiles = _fft_power_profiles(window, common_window, interval)
    wavelet_profiles = _wavelet_scale_energy(window, common_window, interval)
    _save_csv(data_dir / "figure1_fft_power_spectrum.csv", fft_profiles)
    _save_csv(data_dir / "figure1_wavelet_scale_energy.csv", wavelet_profiles)
    _save_csv(
        data_dir / "figure1_fft_profile_similarity.csv",
        _profile_similarity(fft_profiles, "profile"),
    )
    _save_csv(
        data_dir / "figure1_wavelet_profile_similarity.csv",
        _profile_similarity(wavelet_profiles, "profile"),
    )
    _save_json(
        data_dir / "figure1_frequency_analysis_parameters.json",
        {
            "window_days": 14,
            "interval_minutes": interval,
            **window_metadata,
            "window_start": window.index.min(),
            "window_end": window.index.max(),
            "fft_power": "mean-centered one-sided rFFT power normalized by total power",
            "wavelet": "Morlet CWT with normalized mean scale energy",
            "wavelet_period_range_hours": [
                float(wavelet_profiles["period_hours"].min()),
                float(wavelet_profiles["period_hours"].max()),
            ],
        },
    )

    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    axes[0, 0].plot(
        window.index,
        window.to_numpy(),
        linewidth=0.65,
        alpha=0.75,
    )
    axes[0, 0].plot(
        common_window.index,
        common_window.to_numpy(),
        color="black",
        linewidth=1.5,
        label="common field",
    )
    axes[0, 0].set_title("(a) Aligned wind-speed traces")
    axes[0, 0].set_ylabel("Wind speed (m/s)")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].tick_params(axis="x", rotation=25)

    image = axes[0, 1].imshow(
        corr.to_numpy(),
        vmin=0.85,
        vmax=1.0,
        cmap="Blues",
        interpolation="nearest",
    )
    axes[0, 1].set_title("(b) Pearson correlation")
    axes[0, 1].set_xticks(range(len(corr)), corr.columns, rotation=45)
    axes[0, 1].set_yticks(range(len(corr)), corr.index)
    # Reserve a dedicated right margin so the colorbar is visibly outside
    # the heatmap panel rather than occupying its plotting area.
    colorbar_axis = fig.add_axes([0.925, 0.695, 0.014, 0.19])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Pearson r", rotation=90, labelpad=8)

    axes[1, 0].plot(
        pca_table["component"],
        pca_table["cumulative_explained_variance_ratio"],
        marker="o",
        markersize=3,
    )
    axes[1, 0].set_ylim(0, 1.02)
    axes[1, 0].set_xlabel("Number of components")
    axes[1, 0].set_ylabel("Cumulative explained variance")
    axes[1, 0].set_title("(c) Low-dimensional structure")

    residual = window.sub(common_window, axis=0)
    axes[1, 1].plot(
        common_window.index,
        common_window.to_numpy(),
        color="black",
        linewidth=1.5,
        label="common field",
    )
    axes[1, 1].plot(
        residual.index,
        residual.to_numpy(),
        linewidth=0.55,
        alpha=0.65,
    )
    axes[1, 1].axhline(0, color="0.6", linewidth=0.7)
    axes[1, 1].set_title("(d) Common field and turbine residuals")
    axes[1, 1].set_ylabel("Speed / residual (m/s)")
    axes[1, 1].tick_params(axis="x", rotation=25)
    
    for turbine in window.columns:
        axes[2, 0].plot(
            fft_profiles["period_hours"],
            fft_profiles[turbine],
            color="0.45",
            linewidth=0.7,
            alpha=0.45,
        )
    axes[2, 0].plot(
        fft_profiles["period_hours"],
        fft_profiles["common_field"],
        color="#c44e52",
        linewidth=2.0,
        label="common field",
    )
    axes[2, 0].set_xscale("log")
    axes[2, 0].set_xlabel("Period (hours, log scale)")
    axes[2, 0].set_ylabel("Normalized power")
    axes[2, 0].set_title("(e) Shared frequency-domain patterns")
    axes[2, 0].legend(frameon=False)
    axes[2, 0].grid(alpha=0.2, linewidth=0.5)

    for turbine in window.columns:
        axes[2, 1].plot(
            wavelet_profiles["period_hours"],
            wavelet_profiles[turbine],
            color="0.45",
            linewidth=0.7,
            alpha=0.45,
        )
    axes[2, 1].plot(
        wavelet_profiles["period_hours"],
        wavelet_profiles["common_field"],
        color="#4c72b0",
        linewidth=2.0,
        label="common field",
    )
    axes[2, 1].set_xscale("log")
    axes[2, 1].set_xlabel("Wavelet period (hours, log scale)")
    axes[2, 1].set_ylabel("Normalized scale energy")
    axes[2, 1].set_title("(f) Shared time-scale patterns")
    axes[2, 1].legend(frameon=False)
    axes[2, 1].grid(alpha=0.2, linewidth=0.5)
    _save_figure(
        fig,
        png_dir / "figure1_shared_farm_dynamics",
        pdf_dir / "figure1_shared_farm_dynamics",
        tight_rect=(0.0, 0.0, 0.90, 1.0),
    )
    return {
        "mean_pairwise_correlation": float(np.nanmean(_off_diagonal(corr.to_numpy()))),
        "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
        "pc3_cumulative_explained_variance": float(
            np.cumsum(pca.explained_variance_ratio_)[2]
        ),
        "mean_fft_profile_correlation": float(
            np.nanmean(
                _off_diagonal(
                    _profile_similarity(fft_profiles, "profile")
                    .drop(columns=["profile"])
                    .to_numpy()
                )
            )
        ),
        "mean_wavelet_profile_correlation": float(
            np.nanmean(
                _off_diagonal(
                    _profile_similarity(wavelet_profiles, "profile")
                    .drop(columns=["profile"])
                    .to_numpy()
                )
            )
        ),
    }


def _plot_figure_1_1(
    root: Path,
    speed: pd.DataFrame,
    common: pd.Series,
    top_k: int = 6,
    figure_title_prefix: str | None = None,
    selected_turbines_override: list[str] | None = None,
    frequency_max: float = 0.3,
    line_cmap: str = "tab10",
    wavelet_cmap: str = "viridis",
) -> dict:
    """Plot per-turbine FFT and Morlet scalograms for the most similar turbines."""
    folder, data_dir, png_dir, _ = _figure_dir(root, 11)
    interval = int(speed.attrs["metadata"]["interval_minutes"])
    per_day = int(round(24 * 60 / interval))
    selected, window_metadata = _select_window_index(speed, per_day, days=14)
    window = speed.iloc[selected]
    common_window = common.iloc[selected]

    similarity = speed.corrwith(common).sort_values(ascending=False)
    if selected_turbines_override is not None:
        missing = [
            turbine
            for turbine in selected_turbines_override
            if turbine not in speed.columns
        ]
        if missing:
            raise ValueError(
                f"Requested turbines are not available in this dataset: {missing}"
            )
        selected_turbines = list(selected_turbines_override)
    else:
        selected_turbines = [
            name for name in similarity.index if np.isfinite(similarity[name])
        ][: max(1, int(top_k))]
        if len(selected_turbines) < 3:
            selected_turbines = list(
                speed.columns[: min(max(3, int(top_k)), len(speed.columns))]
            )

    selection_table = pd.DataFrame(
        {
            "rank": np.arange(1, len(selected_turbines) + 1),
            "turbine": selected_turbines,
            "correlation_with_common_field": [
                float(similarity.get(turbine, np.nan)) for turbine in selected_turbines
            ],
        }
    )
    _save_csv(data_dir / "figure1_1_selected_turbines.csv", selection_table)

    sampling_hours = interval / 60.0
    n = len(window)
    frequencies = np.fft.rfftfreq(n, d=sampling_hours)[1:]
    target_periods = np.geomspace(
        1.0, min(12.0, n * sampling_hours / 3.0), 40
    )
    wavelet = "morl"
    scales = pywt.central_frequency(wavelet) * target_periods / sampling_hours
    scales = np.maximum(1.0, np.unique(np.round(scales, decimals=4)))
    periods = pywt.scale2frequency(wavelet, scales) ** -1 * sampling_hours
    periods_order = np.argsort(periods)
    periods = periods[periods_order]
    time_hours = np.arange(n, dtype=np.float64) * sampling_hours
    scalograms = []

    for turbine in selected_turbines:
        values = window[turbine].to_numpy(dtype=np.float64)
        centered = values - np.nanmean(values)
        fft_amplitude = np.abs(np.fft.rfft(centered))[1:]
        fft_amplitude = fft_amplitude / max(float(fft_amplitude.max()), 1e-12)
        _save_csv(
            data_dir / f"figure1_1_fft_{turbine}.csv",
            pd.DataFrame(
                {
                    "frequency_cycles_per_hour": frequencies,
                    "normalized_amplitude": fft_amplitude,
                }
            ),
        )

        standardized = (centered - np.mean(centered)) / max(
            float(np.std(centered)), 1e-12
        )
        coefficients, _ = pywt.cwt(standardized, scales, wavelet)
        power = np.abs(coefficients) ** 2
        power = power[periods_order]
        power = power / max(float(np.nanpercentile(power, 99)), 1e-12)
        power = np.clip(power, 0.0, 1.0)
        scalograms.append(power)
        wavelet_frame = pd.DataFrame(power, columns=time_hours)
        wavelet_frame.insert(0, "period_hours", periods)
        _save_csv(
            data_dir / f"figure1_1_wavelet_{turbine}.csv",
            wavelet_frame,
        )

    scalograms_array = np.stack(scalograms, axis=0)
    display_scalograms = np.log10(np.maximum(scalograms_array, 1e-5))
    _save_npz(
        data_dir / "figure1_1_wavelet_scalograms.npz",
        turbines=np.asarray(selected_turbines),
        time_hours=time_hours,
        period_hours=periods,
        normalized_power=scalograms_array.astype(np.float32),
    )
    _save_json(
        data_dir / "figure1_1_parameters.json",
        {
            "figure": "Figure 1.1",
            "analysis_window": window_metadata,
            "window_start": window.index.min(),
            "window_end": window.index.max(),
            "interval_minutes": interval,
            "selected_turbines": selected_turbines,
            "selection_rule": f"top {len(selected_turbines)} Pearson correlations with the common field",
            "frequency_axis_max_cycles_per_hour": float(frequency_max),
            "line_colormap": line_cmap,
            "wavelet_colormap": wavelet_cmap,
            "fft": "mean-centered one-sided rFFT amplitude normalized by channel maximum",
            "wavelet": "Morlet CWT power after per-channel standardization and 99th-percentile normalization",
            "wavelet_display": "log10 transform applied only for visual contrast",
            "wavelet_period_range_hours": [
                float(periods.min()),
                float(periods.max()),
            ],
        },
    )

    colors = plt.get_cmap(line_cmap)(
        np.linspace(0.05, 0.85, len(selected_turbines))
    )
    fig, axes = plt.subplots(
        len(selected_turbines),
        2,
        figsize=(7.2, 2.45 * len(selected_turbines)),
        gridspec_kw={"width_ratios": [0.78, 1.0], "wspace": 0.28, "hspace": 0.46},
        squeeze=False,
    )
    for row, turbine in enumerate(selected_turbines):
        color = colors[row]
        fft_table = pd.read_csv(data_dir / f"figure1_1_fft_{turbine}.csv")
        axes[row, 0].plot(
            fft_table["frequency_cycles_per_hour"],
            fft_table["normalized_amplitude"],
            color=color,
            linewidth=1.25,
        )
        title_prefix = f"{figure_title_prefix} | " if figure_title_prefix else ""
        axes[row, 0].set_title(
            f"{title_prefix}{turbine}  |  Frequency spectrum", loc="left"
        )
        axes[row, 0].set_ylabel("Amplitude")
        axes[row, 0].set_xlim(0.0, frequency_max)
        axes[row, 0].set_xticks(np.linspace(0.0, frequency_max, 4))
        axes[row, 0].set_ylim(0, 1.05)
        axes[row, 0].grid(alpha=0.2, linewidth=0.5)

        image = axes[row, 1].imshow(
            display_scalograms[row],
            aspect="auto",
            origin="lower",
            extent=[time_hours.min(), time_hours.max(), periods.min(), periods.max()],
            cmap=wavelet_cmap,
            vmin=-2.5,
            vmax=0.0,
            interpolation="nearest",
        )
        axes[row, 1].set_title(
            f"{title_prefix}{turbine}  |  Wavelet transform", loc="left"
        )
        axes[row, 1].set_ylabel("Period (hours)")
        axes[row, 1].set_yscale("log")
        axes[row, 1].set_ylim(periods.min(), periods.max())
        if row == len(selected_turbines) - 1:
            axes[row, 0].set_xlabel("Frequency (cycles/hour)")
            axes[row, 1].set_xlabel("Time from window start (hours)")
    # Use a dedicated right-side axis so the wavelet colorbar stays outside
    # the subplot grid and does not consume space inside any scalogram.
    colorbar_axis = fig.add_axes([0.910, 0.12, 0.018, 0.76])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("log10 normalized wavelet power")
    _save_figure(
        fig,
        png_dir / "figure1_1_frequency_wavelet",
        tight_rect=(0.0, 0.0, 0.88, 1.0),
    )
    return {
            "selected_turbines": selected_turbines,
            "mean_selected_correlation": float(
                selection_table["correlation_with_common_field"].mean()
            ),
        "window_start": window.index.min(),
        "window_end": window.index.max(),
    }


def _plot_figure_2(root: Path, speed: pd.DataFrame) -> dict:
    folder, data_dir, png_dir, _ = _figure_dir(root, 2)
    arrays = speed.to_numpy(dtype=np.float64)
    z = StandardScaler().fit_transform(arrays)
    pearson = np.corrcoef(arrays.T)
    spearman = speed.rank().corr().to_numpy()
    cosine = (z.T @ z) / np.maximum(
        np.linalg.norm(z, axis=0)[:, None] * np.linalg.norm(z, axis=0)[None, :],
        1e-12,
    )
    distance = 1.0 - cosine
    turbines = list(speed.columns)
    for name, matrix in [
        ("pearson", pearson),
        ("spearman", spearman),
        ("cosine_similarity_centered", cosine),
        ("cosine_distance_centered", distance),
    ]:
        frame = pd.DataFrame(matrix, index=turbines, columns=turbines)
        _save_csv(data_dir / f"figure2_{name}.csv", frame.rename_axis("turbine").reset_index())
    pairs = []
    for i in range(len(turbines)):
        for j in range(i + 1, len(turbines)):
            pairs.append(
                {
                    "turbine_a": turbines[i],
                    "turbine_b": turbines[j],
                    "pearson": pearson[i, j],
                    "spearman": spearman[i, j],
                    "cosine_similarity_centered": cosine[i, j],
                    "cosine_distance_centered": distance[i, j],
                }
            )
    _save_csv(data_dir / "figure2_pairwise_similarity_long.csv", pd.DataFrame(pairs))

    matrices = [
        ("(a) Pearson r", pearson, "similarity"),
        ("(b) Spearman rho", spearman, "similarity"),
        ("(c) Centered cosine", cosine, "similarity"),
        ("(d) Centered cosine distance", distance, "distance"),
    ]
    similarity_values = np.concatenate(
        [_off_diagonal(pearson), _off_diagonal(spearman), _off_diagonal(cosine)]
    )
    similarity_vmin = max(0.0, math.floor(float(np.nanmin(similarity_values)) * 100) / 100)
    distance_vmax = math.ceil(float(np.nanmax(_off_diagonal(distance))) * 100) / 100

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.8, 6.8),
    )
    similarity_image = None
    distance_image = None
    for index, (axis, (title, matrix, scale)) in enumerate(zip(axes.flat, matrices)):
        if scale == "similarity":
            image = axis.imshow(
                matrix,
                vmin=similarity_vmin,
                vmax=1.0,
                cmap="YlGnBu",
                interpolation="nearest",
            )
            similarity_image = image
        else:
            image = axis.imshow(
                matrix,
                vmin=0.0,
                vmax=max(distance_vmax, 0.01),
                cmap="YlOrRd",
                interpolation="nearest",
            )
            distance_image = image
        axis.set_title(title, loc="left", fontsize=11, pad=6)
        axis.set_xticks(range(len(turbines)))
        axis.set_yticks(range(len(turbines)))
        if index // 2 == 1:
            axis.set_xticklabels(turbines, rotation=45, ha="right")
        else:
            axis.set_xticklabels([])
        if index % 2 == 0:
            axis.set_yticklabels(turbines)
        else:
            axis.set_yticklabels([])
        axis.set_xticks(np.arange(-0.5, len(turbines), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(turbines), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.35)
        axis.tick_params(axis="both", which="major", labelsize=8, length=0)
        axis.tick_params(axis="both", which="minor", length=0)
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("0.25")
    if similarity_image is not None:
        colorbar_axis = fig.add_axes([0.885, 0.56, 0.020, 0.34])
        colorbar = fig.colorbar(
            similarity_image,
            cax=colorbar_axis,
        )
        colorbar.set_label("Similarity", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
    if distance_image is not None:
        colorbar_axis = fig.add_axes([0.885, 0.12, 0.020, 0.34])
        colorbar = fig.colorbar(
            distance_image,
            cax=colorbar_axis,
        )
        colorbar.set_label("Distance", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
    _save_figure(
        fig,
        png_dir / "figure2_similarity_matrices",
        (root / FIGURE_NAMES[2] / "pdf" / "figure2_similarity_matrices"),
        tight_rect=(0.0, 0.0, 0.86, 1.0),
    )
    summary = {
        "mean_off_diagonal_pearson": float(np.nanmean(_off_diagonal(pearson))),
        "mean_off_diagonal_spearman": float(np.nanmean(_off_diagonal(spearman))),
        "mean_off_diagonal_centered_cosine": float(np.nanmean(_off_diagonal(cosine))),
        "mean_off_diagonal_centered_cosine_distance": float(
            np.nanmean(_off_diagonal(distance))
        ),
    }
    _save_json(data_dir / "figure2_summary.json", summary)
    return summary


def _plot_figure_3(root: Path, speed: pd.DataFrame) -> dict:
    folder, data_dir, png_dir, _ = _figure_dir(root, 3)
    matrix = speed.to_numpy(dtype=np.float64).T
    matrix = matrix - matrix.mean(axis=1, keepdims=True)
    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    variance = singular_values**2
    ratios = variance / variance.sum()
    explained = pd.DataFrame(
        {
            "component": np.arange(1, len(ratios) + 1),
            "singular_value": singular_values,
            "explained_variance_ratio": ratios,
            "cumulative_explained_variance_ratio": np.cumsum(ratios),
        }
    )
    _save_csv(data_dir / "figure3_singular_values.csv", explained)
    turbine_scores = pd.DataFrame(
        u[:, :3],
        columns=["pc1_score", "pc2_score", "pc3_score"],
    )
    turbine_scores.insert(0, "turbine", speed.columns)
    _save_csv(data_dir / "figure3_turbine_pc_scores.csv", turbine_scores)
    reconstruction_rows = []
    for k in range(1, min(8, len(singular_values)) + 1):
        reconstruction = (u[:, :k] * singular_values[:k]) @ vt[:k]
        reconstruction_rows.append(
            {
                "num_components": k,
                "relative_frobenius_reconstruction_error": float(
                    np.linalg.norm(matrix - reconstruction) / np.linalg.norm(matrix)
                ),
                "cumulative_explained_variance_ratio": float(np.sum(ratios[:k])),
            }
        )
    reconstruction_table = pd.DataFrame(reconstruction_rows)
    _save_csv(data_dir / "figure3_reconstruction_error.csv", reconstruction_table)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].semilogy(
        explained["component"], explained["singular_value"], marker="o", markersize=3
    )
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Singular value")
    axes[0].set_title("(a) Singular-value spectrum")
    axes[1].plot(
        explained["component"],
        explained["cumulative_explained_variance_ratio"],
        marker="o",
        markersize=3,
    )
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative explained variance")
    axes[1].set_title("(b) Low-rank energy")
    axes[2].scatter(
        turbine_scores["pc1_score"],
        turbine_scores["pc2_score"],
        s=45,
        c=np.arange(len(turbines := speed.columns)),
        cmap="tab20",
    )
    for _, row in turbine_scores.iterrows():
        axes[2].annotate(row["turbine"], (row["pc1_score"], row["pc2_score"]), fontsize=8)
    axes[2].set_xlabel("PC1 score")
    axes[2].set_ylabel("PC2 score")
    axes[2].set_title("(c) Turbine coordinates in PCA space")
    _save_figure(
        fig,
        png_dir / "figure3_low_rank_structure",
        (root / FIGURE_NAMES[3] / "pdf" / "figure3_low_rank_structure"),
    )
    return {
        "pc1_explained_variance": float(ratios[0]),
        "pc3_cumulative_explained_variance": float(np.sum(ratios[:3])),
        "rank_for_95_percent": int(np.searchsorted(np.cumsum(ratios), 0.95) + 1),
    }


def _corr_at_lag(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    if lag > 0:
        left, right = a[:-lag], b[lag:]
    elif lag < 0:
        left, right = a[-lag:], b[:lag]
    else:
        left, right = a, b
    if len(left) < 3:
        return float("nan")
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else float("nan")


def _rolling_sync(speed: pd.DataFrame, common: pd.Series, window: int, stride: int) -> pd.DataFrame:
    rows = []
    values = speed.to_numpy(dtype=np.float64)
    common_values = common.to_numpy(dtype=np.float64)
    for end in range(window, len(speed) + 1, stride):
        start = end - window
        block = values[start:end]
        corr = np.corrcoef(block.T)
        rows.append(
            {
                "timestamp": speed.index[end - 1],
                "mean_pairwise_correlation": float(np.nanmean(_off_diagonal(corr))),
                "mean_common_field_correlation": float(
                    np.nanmean(
                        [
                            _corr_at_lag(block[:, i], common_values[start:end], 0)
                            for i in range(block.shape[1])
                        ]
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_figure_4(root: Path, speed: pd.DataFrame, common: pd.Series) -> dict:
    folder, data_dir, png_dir, _ = _figure_dir(root, 4)
    interval = int(speed.attrs["metadata"]["interval_minutes"])
    per_day = int(round(24 * 60 / interval))
    rolling = _rolling_sync(speed, common, window=7 * per_day, stride=per_day)
    _save_csv(data_dir / "figure4_rolling_synchrony.csv", rolling)

    reference = speed.columns[0]
    lags = range(-3 * per_day, 3 * per_day + 1, max(1, per_day // 12))
    lag_rows = []
    ref_values = speed[reference].to_numpy(dtype=np.float64)
    for turbine in speed.columns[1:]:
        values = speed[turbine].to_numpy(dtype=np.float64)
        correlations = [_corr_at_lag(ref_values, values, lag) for lag in lags]
        best = int(np.nanargmax(correlations))
        for lag, corr in zip(lags, correlations):
            lag_rows.append(
                {
                    "reference_turbine": reference,
                    "turbine": turbine,
                    "lag_steps": lag,
                    "lag_hours": lag * interval / 60.0,
                    "correlation": corr,
                    "is_best_lag": lag == list(lags)[best],
                }
            )
    lag_table = pd.DataFrame(lag_rows)
    _save_csv(data_dir / "figure4_lagged_correlation.csv", lag_table)
    best_lags = lag_table[lag_table["is_best_lag"]].copy()
    _save_csv(data_dir / "figure4_best_lags.csv", best_lags)

    fig_a, axis = plt.subplots(figsize=(8.5, 4.3))
    axis.plot(
        pd.to_datetime(rolling["timestamp"]),
        rolling["mean_pairwise_correlation"],
        label="mean pairwise",
    )
    axis.plot(
        pd.to_datetime(rolling["timestamp"]),
        rolling["mean_common_field_correlation"],
        label="turbine-common",
    )
    axis.set_xlim(pd.Timestamp("2017-01-01"), pd.Timestamp("2023-01-01"))
    axis.set_ylim(0.6, 1.0)
    axis.set_title("(a) Rolling 7-day synchrony")
    axis.set_ylabel("Correlation")
    axis.set_xlabel("Date")
    axis.legend(frameon=False)
    axis.tick_params(axis="x", rotation=25)
    _save_figure(
        fig_a,
        png_dir / "figure4a_rolling_7day_synchrony",
        (root / FIGURE_NAMES[4] / "pdf" / "figure4a_rolling_7day_synchrony"),
    )

    lag_turbines = list(speed.columns[1:9])
    fig_b, axes = plt.subplots(
        2,
        4,
        figsize=(14, 6.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for axis, turbine in zip(axes.flat, lag_turbines):
        subset = lag_table[lag_table["turbine"] == turbine]
        axis.plot(
            subset["lag_hours"],
            subset["correlation"],
            color="#4c72b0",
            alpha=0.85,
            linewidth=1.0,
        )
        axis.axvline(0, color="black", linewidth=0.7)
        axis.set_title(turbine)
        axis.set_xlim(-72, 72)
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.2, linewidth=0.5)
    for axis in axes[1, :]:
        axis.set_xlabel("Lag relative to WT01 (hours)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Correlation")
    fig_b.suptitle("(b) Cross-turbine lag correlation", y=1.01)
    _save_figure(
        fig_b,
        png_dir / "figure4b_lagged_correlation_8_turbines",
        (root / FIGURE_NAMES[4] / "pdf" / "figure4b_lagged_correlation_8_turbines"),
    )
    return {
        "rolling_mean_pairwise_correlation": float(
            rolling["mean_pairwise_correlation"].mean()
        ),
        "best_lag_mean_hours": float(best_lags["lag_hours"].mean()),
        "best_lag_std_hours": float(best_lags["lag_hours"].std(ddof=0)),
    }


def _window_matrix(values: np.ndarray, window: int, stride: int, limit: int) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0, max(0, len(values) - window + 1), stride)
    if len(starts) > limit:
        positions = np.linspace(0, len(starts) - 1, limit).astype(int)
        starts = starts[positions]
    windows = np.stack([values[start : start + window] for start in starts], axis=0)
    return windows, starts


def _plot_figure_5(root: Path, speed: pd.DataFrame) -> dict:
    folder, data_dir, png_dir, _ = _figure_dir(root, 5)
    values = speed.to_numpy(dtype=np.float64)
    window_length = min(72, len(values) // 4)
    all_windows = []
    labels = []
    starts = []
    for turbine_index, turbine in enumerate(speed.columns):
        windows, local_starts = _window_matrix(
            values[:, turbine_index], window_length, window_length, limit=180
        )
        all_windows.append(windows)
        labels.extend([turbine] * len(windows))
        starts.extend(local_starts.tolist())
    matrix = np.concatenate(all_windows, axis=0)
    matrix = StandardScaler().fit_transform(matrix)
    reduced = PCA(n_components=min(20, matrix.shape[1], matrix.shape[0])).fit_transform(matrix)
    perplexity = min(30, max(5, (len(reduced) - 1) // 3))
    embedding_method = "tsne"
    try:
        embedding = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=2026,
            n_jobs=1,
        ).fit_transform(reduced)
    except Exception as error:
        # Some Windows installations have an incompatible threadpoolctl/BLAS
        # combination. PCA keeps the figure reproducible and records the
        # fallback explicitly instead of preventing all other figures.
        print(f"tsne_fallback={type(error).__name__}: {error}")
        embedding_method = "pca_fallback"
        embedding = PCA(n_components=2, random_state=2026).fit_transform(reduced)
    embedding_table = pd.DataFrame(
        {
            "sample_id": np.arange(len(embedding)),
            "turbine": labels,
            "window_start": [
                speed.index[int(start)] for start in starts
            ],
            "window_mean_speed": matrix.mean(axis=1),
            "window_std_speed": matrix.std(axis=1),
            "tsne_1": embedding[:, 0],
            "tsne_2": embedding[:, 1],
        }
    )
    _save_csv(data_dir / "figure5_window_embedding.csv", embedding_table)
    _save_json(
        data_dir / "figure5_embedding_method.json",
        {"method": embedding_method, "perplexity_requested": perplexity},
    )
    _save_npz(
        data_dir / "figure5_window_samples.npz",
        windows=matrix.astype(np.float32),
        embedding=embedding.astype(np.float32),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    turbine_codes = pd.Categorical(embedding_table["turbine"]).codes
    axes[0].scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=turbine_codes,
        cmap="tab20",
        s=8,
        alpha=0.65,
    )
    axes[0].set_title(f"(a) Window embedding by {embedding_method}, turbine color")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    axes[1].scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=embedding_table["window_mean_speed"],
        cmap="viridis",
        s=8,
        alpha=0.65,
    )
    axes[1].set_title(f"(b) Window embedding by {embedding_method}, speed color")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    _save_figure(
        fig,
        png_dir / "figure5_turbine_window_embedding",
        (root / FIGURE_NAMES[5] / "pdf" / "figure5_turbine_window_embedding"),
    )
    return {
        "window_length_steps": int(window_length),
        "num_windows": int(len(matrix)),
        "tsne_perplexity": float(perplexity),
        "embedding_method": embedding_method,
    }


def _plot_figure_7(root: Path, direction: pd.DataFrame) -> dict:
    folder, data_dir, png_dir, _ = _figure_dir(root, 7)
    degrees = np.rad2deg(direction.to_numpy(dtype=np.float64)) % 360.0
    sample_count = min(5000, degrees.size)
    flat = degrees.reshape(-1)
    sample_indices = np.linspace(0, len(flat) - 1, sample_count).astype(int)
    sampled = flat[sample_indices]
    circle = pd.DataFrame(
        {
            "sample_id": np.arange(sample_count),
            "direction_degrees": sampled,
            "sin_theta": np.sin(np.deg2rad(sampled)),
            "cos_theta": np.cos(np.deg2rad(sampled)),
        }
    )
    _save_csv(data_dir / "figure7_direction_unit_circle_samples.csv", circle)

    crossing = pd.DataFrame(
        {
            "step": np.arange(5),
            "true_direction_degrees": [359.0, 0.0, 1.0, 2.0, 3.0],
            "linear_interpolation_degrees": [359.0, 269.5, 180.0, 90.5, 3.0],
            "circular_interpolation_degrees": [359.0, 359.5, 1.0, 2.0, 3.0],
        }
    )
    _save_csv(data_dir / "figure7_angle_wraparound_example.csv", crossing)
    errors = pd.DataFrame(
        {
            "true_degrees": [1.0, 359.0, 10.0, 350.0],
            "prediction_degrees": [359.0, 1.0, 350.0, 10.0],
        }
    )
    errors["scalar_abs_error_degrees"] = np.abs(
        errors["true_degrees"] - errors["prediction_degrees"]
    )
    errors["circular_abs_error_degrees"] = np.minimum(
        errors["scalar_abs_error_degrees"],
        360.0 - errors["scalar_abs_error_degrees"],
    )
    _save_csv(data_dir / "figure7_scalar_vs_circular_error.csv", errors)

    theta = np.linspace(0, 2 * np.pi, 400)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].scatter(circle["cos_theta"], circle["sin_theta"], s=5, alpha=0.25)
    axes[0].plot(np.cos(theta), np.sin(theta), color="black", linewidth=0.8)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("cos(theta)")
    axes[0].set_ylabel("sin(theta)")
    axes[0].set_title("(a) Direction on the unit circle")
    axes[1].plot(crossing["step"], crossing["true_direction_degrees"], marker="o", label="true")
    axes[1].plot(
        crossing["step"],
        crossing["linear_interpolation_degrees"],
        marker="o",
        label="scalar interpolation",
    )
    axes[1].plot(
        crossing["step"],
        crossing["circular_interpolation_degrees"],
        marker="o",
        label="circular interpolation",
    )
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Direction (deg)")
    axes[1].set_title("(b) 359 deg to 1 deg")
    axes[1].legend(frameon=False, fontsize=8)
    x = np.arange(len(errors))
    axes[2].bar(x - 0.18, errors["scalar_abs_error_degrees"], width=0.36, label="scalar")
    axes[2].bar(x + 0.18, errors["circular_abs_error_degrees"], width=0.36, label="circular")
    axes[2].set_xticks(x, [f"case {i + 1}" for i in x])
    axes[2].set_ylabel("Absolute error (deg)")
    axes[2].set_title("(c) Error metric comparison")
    axes[2].legend(frameon=False)
    _save_figure(
        fig,
        png_dir / "figure7_direction_geometry",
        (root / FIGURE_NAMES[7] / "pdf" / "figure7_direction_geometry"),
    )
    return {
        "sample_count": int(sample_count),
        "mean_scalar_error_degrees": float(errors["scalar_abs_error_degrees"].mean()),
        "mean_circular_error_degrees": float(errors["circular_abs_error_degrees"].mean()),
    }


def _screen_period(common: np.ndarray, candidate_periods: list[int]) -> tuple[int, pd.DataFrame]:
    centered = common - np.nanmean(common)
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(len(centered))
    rows = []
    for period in candidate_periods:
        target_frequency = 1.0 / period
        index = int(np.argmin(np.abs(frequencies - target_frequency)))
        rows.append(
            {
                "period_steps": period,
                "period_hours": period / 6.0,
                "fft_bin": index,
                "power": float(spectrum[index]),
            }
        )
    table = pd.DataFrame(rows)
    selected = int(table.sort_values("power", ascending=False).iloc[0]["period_steps"])
    table["selected"] = table["period_steps"] == selected
    return selected, table


def _phase_tokens(
    windows: np.ndarray, period: int, patch_length: int
) -> tuple[np.ndarray, np.ndarray]:
    usable = (windows.shape[1] // period) * period
    folded = windows[:, :usable].reshape(windows.shape[0], usable // period, period)
    phase = folded.mean(axis=1)
    if period != patch_length:
        old_grid = np.linspace(0.0, 1.0, period)
        new_grid = np.linspace(0.0, 1.0, patch_length)
        phase = np.stack(
            [np.interp(new_grid, old_grid, row) for row in phase],
            axis=0,
        )
    patch_count = windows.shape[1] // patch_length
    patches = windows[:, : patch_count * patch_length].reshape(
        windows.shape[0], patch_count, patch_length
    )
    return patches.reshape(-1, patch_length), phase


def _rbf_mmd(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.concatenate([a, b], axis=0)
    distances = np.sqrt(
        np.maximum(
            ((pooled[:, None, :] - pooled[None, :, :]) ** 2).sum(axis=2),
            0.0,
        )
    )
    bandwidth = float(np.median(distances[distances > 0])) if np.any(distances > 0) else 1.0
    gamma = 1.0 / max(2.0 * bandwidth * bandwidth, 1e-12)
    kaa = np.exp(-gamma * ((a[:, None, :] - a[None, :, :]) ** 2).sum(axis=2)).mean()
    kbb = np.exp(-gamma * ((b[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)).mean()
    kab = np.exp(-gamma * ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)).mean()
    return float(kaa + kbb - 2.0 * kab)


def _embed_tokens(tokens: np.ndarray, labels: list[str], random_state: int = 2026) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(tokens)
    pca_dim = min(8, scaled.shape[0] - 1, scaled.shape[1])
    reduced = PCA(n_components=max(2, pca_dim), random_state=random_state).fit_transform(scaled)
    projection = PCA(n_components=2, random_state=random_state).fit_transform(scaled)
    return pd.DataFrame(
        {
            "representation": labels,
            "token_id": np.arange(len(tokens)),
            "embedding_1": projection[:, 0],
            "embedding_2": projection[:, 1],
            "reduced_features": list(reduced[:, : min(8, reduced.shape[1])]),
        }
    )


def _plot_figure_8(root: Path, common: pd.Series) -> dict:
    folder, data_dir, png_dir, _ = _figure_dir(root, 8)
    values = common.to_numpy(dtype=np.float64)
    interval = int(common.attrs["metadata"]["interval_minutes"])
    per_day = int(round(24 * 60 / interval))
    window_length = min(288, len(values) // 8)
    windows, starts = _window_matrix(values, window_length, window_length, limit=96)
    candidate_periods = [max(12, per_day // 4), max(24, per_day // 2), per_day, 2 * per_day]
    candidate_periods = sorted(set(p for p in candidate_periods if p <= window_length))
    selected_period, period_table = _screen_period(values, candidate_periods)
    _save_csv(data_dir / "figure8_screened_periods.csv", period_table)

    patch_tokens, phase_tokens = _phase_tokens(windows, selected_period, patch_length=12)
    patch_per_window = patch_tokens.reshape(len(windows), -1, patch_tokens.shape[-1])
    token_matrix = np.vstack([patch_tokens, phase_tokens])
    token_projection = PCA(n_components=2, random_state=2026).fit_transform(
        StandardScaler().fit_transform(token_matrix)
    )
    patch_projection = token_projection[: len(patch_tokens)]
    phase_projection = token_projection[len(patch_tokens) :]
    token_table = pd.concat(
        [
            pd.DataFrame(
                {
                    "representation": "patch",
                    "token_id": np.arange(len(patch_projection)),
                    "embedding_1": patch_projection[:, 0],
                    "embedding_2": patch_projection[:, 1],
                }
            ),
            pd.DataFrame(
                {
                    "representation": "phase",
                    "token_id": np.arange(len(phase_projection)),
                    "embedding_1": phase_projection[:, 0],
                    "embedding_2": phase_projection[:, 1],
                }
            ),
        ],
        ignore_index=True,
    )
    _save_csv(data_dir / "figure8_patch_phase_token_embeddings.csv", token_table)
    _save_npz(
        data_dir / "figure8_tokens.npz",
        windows=windows.astype(np.float32),
        patch_tokens=patch_tokens.astype(np.float32),
        phase_tokens=phase_tokens.astype(np.float32),
        starts=starts.astype(np.int64),
    )

    # Compare token distributions across chronological blocks after mapping
    # both representations to the same feature dimension.
    groups = np.array_split(np.arange(len(windows)), 6)
    block_labels = [f"block_{i + 1}" for i in range(len(groups))]
    patch_block = []
    phase_block = []
    for block_id, indices in enumerate(groups):
        patch_samples = patch_per_window[indices].reshape(-1, patch_tokens.shape[-1])
        phase_samples = phase_tokens[indices]
        patch_samples = StandardScaler().fit_transform(patch_samples)
        phase_samples = StandardScaler().fit_transform(phase_samples)
        patch_block.append(patch_samples[:80])
        phase_block.append(phase_samples[:80])
    patch_mmd = np.zeros((len(groups), len(groups)))
    phase_mmd = np.zeros_like(patch_mmd)
    for i in range(len(groups)):
        for j in range(len(groups)):
            patch_mmd[i, j] = _rbf_mmd(patch_block[i], patch_block[j])
            phase_mmd[i, j] = _rbf_mmd(phase_block[i], phase_block[j])
    _save_csv(
        data_dir / "figure8_patch_mmd.csv",
        pd.DataFrame(patch_mmd, index=block_labels, columns=block_labels).rename_axis("block").reset_index(),
    )
    _save_csv(
        data_dir / "figure8_phase_mmd.csv",
        pd.DataFrame(phase_mmd, index=block_labels, columns=block_labels).rename_axis("block").reset_index(),
    )

    patch_block_drift = np.array(
        [np.mean(np.delete(patch_mmd[index], index)) for index in range(len(block_labels))]
    )
    phase_block_drift = np.array(
        [np.mean(np.delete(phase_mmd[index], index)) for index in range(len(block_labels))]
    )
    drift_ratio = float(patch_block_drift.mean() / max(phase_block_drift.mean(), 1e-12))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    x = np.arange(len(block_labels))
    for index in x:
        axes[0].plot(
            [index, index],
            [phase_block_drift[index], patch_block_drift[index]],
            color="0.78",
            linewidth=1.0,
            zorder=1,
        )
    axes[0].plot(
        x,
        patch_block_drift,
        color="#4e79a7",
        marker="o",
        markersize=6.0,
        linewidth=1.8,
        label="patch tokens",
        zorder=3,
    )
    axes[0].plot(
        x,
        phase_block_drift,
        color="#f28e2b",
        marker="D",
        markersize=6.0,
        linewidth=1.8,
        label="phase tokens",
        zorder=4,
    )
    axes[0].set_title("(a) Cross-block distribution drift", fontsize=16)
    axes[0].set_xlabel("Chronological block", fontsize=14)
    axes[0].set_ylabel("Mean MMD to other blocks", fontsize=14)
    axes[0].set_xticks(x, block_labels, rotation=35, ha="right")
    axes[0].set_yscale("log")
    axes[0].set_ylim(
        max(float(phase_block_drift.min()) * 0.72, 1e-4),
        float(patch_block_drift.max()) * 1.42,
    )
    axes[0].tick_params(labelsize=12)
    axes[0].grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=11, loc="upper left")
    axes[0].text(
        0.5,
        0.05,
        f"mean drift ratio = {drift_ratio:.1f}x",
        transform=axes[0].transAxes,
        fontsize=11,
        color="0.25",
        ha="center",
    )
    for spine in ["top", "right"]:
        axes[0].spines[spine].set_visible(False)

    # Save the two token spaces separately as standalone figures for paper layout.
    for projection, title, filename, color, marker, size, alpha in [
        (
            patch_projection,
            "Patch-token representation in the shared PCA space",
            "figure8_patch_token_space",
            "#006d9c",
            "o",
            18,
            0.5,
        ),
        (
            phase_projection,
            "Phase-token representation in the shared PCA space",
            "figure8_phase_token_space",
            "#f28e2b",
            "D",
            48,
            0.9,
        ),
    ]:
        token_fig, token_axis = plt.subplots(figsize=(6.2, 5.2))
        token_axis.scatter(
            projection[:, 0],
            projection[:, 1],
            s=size,
            alpha=alpha,
            color=color,
            edgecolors="white" if marker == "D" else "none",
            linewidths=0.4 if marker == "D" else 0.0,
            marker=marker,
        )
        token_axis.set_title(title, fontsize=17)
        token_axis.set_xlabel("PCA 1", fontsize=14)
        token_axis.set_ylabel("PCA 2", fontsize=14)
        token_axis.tick_params(labelsize=12)
        _save_figure(token_fig, png_dir / filename)

    for axis, matrix, title, value_format in [
        (axes[1], patch_mmd, "(b) Patch-token MMD", "{:.3f}"),
        (axes[2], phase_mmd, "(c) Phase-token MMD", "{:.4f}"),
    ]:
        image = axis.imshow(matrix, cmap="GnBu", vmin=0.0, vmax=float(np.max(matrix)))
        axis.set_title(title, fontsize=16)
        axis.set_xticks(range(len(block_labels)), block_labels, rotation=45, ha="right")
        axis.set_yticks(range(len(block_labels)), block_labels)
        axis.tick_params(labelsize=11)
        threshold = float(np.max(matrix)) * 0.55
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                text_color = "#073b4c" if matrix[row, col] < threshold else "white"
                axis.text(
                    col,
                    row,
                    value_format.format(matrix[row, col]),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=text_color,
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04).ax.tick_params(
            labelsize=10
        )
    # Match panel (a)'s plot height to the heatmaps after colorbar layout.
    fig.tight_layout()
    panel_a_box = axes[0].get_position()
    heatmap_box = axes[1].get_position()
    panel_scale = 0.95
    panel_width = panel_a_box.width * panel_scale
    panel_height = heatmap_box.height * panel_scale
    axes[0].set_position(
        [
            panel_a_box.x0 + (panel_a_box.width - panel_width) / 2,
            heatmap_box.y0 + (heatmap_box.height - panel_height) / 2,
            panel_width,
            panel_height,
        ]
    )
    _save_figure(
        fig,
        png_dir / "figure8_phase_tokens_vs_patch_tokens",
        (root / FIGURE_NAMES[8] / "pdf" / "figure8_phase_tokens_vs_patch_tokens"),
        apply_tight_layout=False,
    )
    return {
        "window_length_steps": int(window_length),
        "selected_period_steps": int(selected_period),
        "selected_period_hours": float(selected_period * interval / 60.0),
        "mean_off_diagonal_patch_mmd": float(np.mean(_off_diagonal(patch_mmd))),
        "mean_off_diagonal_phase_mmd": float(np.mean(_off_diagonal(phase_mmd))),
    }


def _load_or_prepare_plot_data(
    config: dict,
    output_root: Path,
    cache_stem: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_root.mkdir(parents=True, exist_ok=True)
    aligned_cache = output_root / f"aligned_{cache_stem}_arrays.npz"
    static_cache = output_root / "shared_static_coordinates.csv"
    summary_cache = output_root / "dataset_summary.json"
    if aligned_cache.exists() and static_cache.exists() and summary_cache.exists():
        print(f"loading_aligned_cache={cache_stem}")
        cache = np.load(aligned_cache, allow_pickle=False)
        metadata = json.loads(summary_cache.read_text(encoding="utf-8"))
        timestamps = _decode_cached_timestamps(cache["timestamps"])
        turbines = list(metadata["turbines"])
        speed = pd.DataFrame(cache["speed"], index=timestamps, columns=turbines)
        direction = pd.DataFrame(cache["direction"], index=timestamps, columns=turbines)
        speed.attrs["metadata"] = metadata
        direction.attrs["metadata"] = metadata
        static = pd.read_csv(static_cache)
    else:
        print(f"loading_raw_data={cache_stem}")
        speed, direction, static = _prepare_frame(config)
        metadata = dict(speed.attrs["metadata"])
        metadata["output_root"] = str(output_root.resolve())
        metadata["common_field_definition"] = (
            "mean wind speed across available turbines at each timestamp"
        )
        metadata["residual_definition"] = (
            "turbine wind speed minus common field"
        )
        _save_json(output_root / "dataset_summary.json", metadata)
        _save_csv(output_root / "shared_static_coordinates.csv", static)
        common_cache, residual_cache = _common_and_residual(speed)
        _save_npz(
            aligned_cache,
            timestamps=speed.index.values.astype("datetime64[ns]").view("int64"),
            speed=speed.to_numpy(dtype=np.float32),
            direction=direction.to_numpy(dtype=np.float32),
            common_field=common_cache.to_numpy(dtype=np.float32),
            residual=residual_cache.to_numpy(dtype=np.float32),
        )
    return speed, direction, static


def run(
    config_path: str | Path,
    output_root: str | Path,
    shanxi_config_path: str | Path | None = None,
) -> None:
    config = load_config(config_path)
    output_root = Path(output_root)
    speed, direction, static = _load_or_prepare_plot_data(
        config, output_root, cache_stem="penmanshiel"
    )
    common, residual = _common_and_residual(speed)
    common.attrs["metadata"] = speed.attrs["metadata"]
    residual.attrs["metadata"] = speed.attrs["metadata"]

    summaries = {
        "figure1": _plot_figure_1(output_root, speed, common),
        "figure1_1": _plot_figure_1_1(
            output_root,
            speed,
            common,
            selected_turbines_override=["WT02", "WT04", "WT06"],
        ),
        "figure2": _plot_figure_2(output_root, speed),
        "figure3": _plot_figure_3(output_root, speed),
        "figure4": _plot_figure_4(output_root, speed, common),
        "figure5": _plot_figure_5(output_root, speed),
        "figure7": _plot_figure_7(output_root, direction),
        "figure8": _plot_figure_8(output_root, common),
    }
    _save_json(output_root / "all_figure_summaries.json", summaries)
    print(f"figures={output_root.resolve()}")
    print(json.dumps(summaries, ensure_ascii=False, indent=2, default=_json_default))

    if shanxi_config_path is not None:
        shanxi_config = load_config(shanxi_config_path)
        shanxi_root = output_root.parent / "shanxi_intro"
        shanxi_speed, _, _ = _load_or_prepare_plot_data(
            shanxi_config, shanxi_root, cache_stem="shanxi"
        )
        shanxi_common, _ = _common_and_residual(shanxi_speed)
        shanxi_common.attrs["metadata"] = shanxi_speed.attrs["metadata"]
        shanxi_summary = {
            "figure1_1": _plot_figure_1_1(
                shanxi_root,
                shanxi_speed,
                shanxi_common,
                top_k=3,
                figure_title_prefix="Shanxi",
                frequency_max=0.15,
                line_cmap="Dark2",
                wavelet_cmap="plasma",
            )
        }
        _save_json(shanxi_root / "all_figure_summaries.json", shanxi_summary)
        print(f"figures={shanxi_root.resolve()}")
        print(
            json.dumps(
                shanxi_summary,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Penmanshiel introduction figures and reusable plotting data."
    )
    parser.add_argument(
        "--config",
        default="configs/final/penmanshiel_extended_benchmark.yaml",
        help="YAML config containing the Penmanshiel raw-data path.",
    )
    parser.add_argument("--output-root", default="figures/penmanshiel_intro")
    parser.add_argument(
        "--shanxi-config",
        default=None,
        help="Optional Shanxi config. If given, also generate its Figure 1.1.",
    )
    args = parser.parse_args()
    try:
        run(args.config, args.output_root, args.shanxi_config)
    except KeyboardInterrupt:
        print("plotting_interrupted", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
