from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DATASETS = {
    "shanxi": PROCESSED / "shanxi_h24_interp_circular.npz",
    "penmanshiel": PROCESSED / "penmanshiel_2016_2022_h24_full_interp_fft.npz",
}
PERIODS_HOURS = (24, 12, 8, 6, 4)
NOISE_LEVEL = 0.1
DEFAULT_SEED = 2026


def _metadata(value: np.ndarray) -> dict:
    raw = value.item() if value.shape == () else value
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, dict):
        return raw
    raise TypeError(f"Unsupported metadata value: {type(raw)!r}")


def _load_processed(path: Path) -> tuple[np.ndarray, np.ndarray, dict, int]:
    cache = np.load(path, allow_pickle=False)
    metadata = _metadata(cache["metadata"])
    speed = cache["test_speed"].astype(np.float64)
    direction_representation = metadata.get(
        "direction_representation", "scalar"
    )

    if direction_representation == "circular":
        sine = cache["test_direction_sin"].astype(np.float64)
        cosine = cache["test_direction_cos"].astype(np.float64)
        direction = np.arctan2(sine, cosine) % (2.0 * np.pi)
    else:
        direction = (
            cache["test_direction"].astype(np.float64)
            * cache["direction_std"].astype(np.float64)[None, :]
            + cache["direction_mean"].astype(np.float64)[None, :]
        )

    interval_minutes = int(metadata.get("interval_minutes", 10))
    return speed, direction, metadata, interval_minutes


def _windowed_period_amplitude(
    values: np.ndarray,
    periods_hours: tuple[int, ...],
    interval_minutes: int,
    window_hours: int = 48,
    stride_hours: int = 24,
) -> np.ndarray:
    window_steps = int(round(window_hours * 60 / interval_minutes))
    stride_steps = int(round(stride_hours * 60 / interval_minutes))
    if len(values) < window_steps:
        raise ValueError(
            f"Need at least {window_steps} rows, got {len(values)}."
        )

    starts = np.arange(0, len(values) - window_steps + 1, stride_steps)
    windows = np.stack(
        [values[start : start + window_steps] for start in starts], axis=0
    )
    windows = windows - windows.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(windows, axis=1)
    amplitudes = []
    for period_hours in periods_hours:
        period_steps = int(round(period_hours * 60 / interval_minutes))
        if window_steps % period_steps != 0:
            raise ValueError(
                f"Period {period_hours} h is not an integer FFT bin "
                f"for a {window_hours} h window."
            )
        bin_index = window_steps // period_steps
        amplitude = 2.0 * np.abs(spectrum[:, bin_index, :]) / window_steps
        amplitudes.append(amplitude.mean())
    return np.asarray(amplitudes, dtype=np.float64)


def _direction_vector(direction: np.ndarray) -> np.ndarray:
    return np.stack([np.sin(direction), np.cos(direction)], axis=-1)


def _joint_features(
    speed_standardized: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            speed_standardized[..., None],
            _direction_vector(direction),
        ],
        axis=-1,
    )


def _joint_period_amplitude(
    speed_standardized: np.ndarray,
    direction: np.ndarray,
    periods_hours: tuple[int, ...],
    interval_minutes: int,
) -> np.ndarray:
    features = _joint_features(speed_standardized, direction)
    window_hours = 48
    stride_hours = 24
    window_steps = int(round(window_hours * 60 / interval_minutes))
    stride_steps = int(round(stride_hours * 60 / interval_minutes))
    starts = np.arange(0, len(features) - window_steps + 1, stride_steps)
    windows = np.stack(
        [features[start : start + window_steps] for start in starts], axis=0
    )
    windows = windows - windows.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(windows, axis=1)
    outputs = []
    for period_hours in periods_hours:
        period_steps = int(round(period_hours * 60 / interval_minutes))
        bin_index = window_steps // period_steps
        amplitude = 2.0 * np.abs(spectrum[:, bin_index, :]) / window_steps
        outputs.append(np.sqrt(np.mean(np.square(amplitude), axis=-1)).mean())
    return np.asarray(outputs, dtype=np.float64)


def _make_plot_table(
    speed: np.ndarray,
    direction: np.ndarray,
    speed_mean: np.ndarray,
    speed_std: np.ndarray,
    interval_minutes: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # The stored speed is standardized. This is also the space used by the
    # robustness experiment, so the perturbation level is exactly 0.1.
    speed_noisy = speed + NOISE_LEVEL * rng.standard_normal(speed.shape)
    direction_noisy = (
        direction
        + NOISE_LEVEL
        * rng.standard_normal(direction.shape)
    ) % (2.0 * np.pi)

    speed_clean = _windowed_period_amplitude(
        speed,
        PERIODS_HOURS,
        interval_minutes,
    )
    speed_perturbed = _windowed_period_amplitude(
        speed_noisy,
        PERIODS_HOURS,
        interval_minutes,
    )
    direction_clean = _windowed_period_amplitude(
        _direction_vector(direction).reshape(len(direction), -1),
        PERIODS_HOURS,
        interval_minutes,
    )
    direction_perturbed = _windowed_period_amplitude(
        _direction_vector(direction_noisy).reshape(len(direction_noisy), -1),
        PERIODS_HOURS,
        interval_minutes,
    )
    # Rescale the direction-vector spectra to a comparable visual range.
    direction_scale = max(float(direction_clean.mean()), 1e-12)
    direction_clean = direction_clean / direction_scale
    direction_perturbed = direction_perturbed / direction_scale
    joint_clean = _joint_period_amplitude(
        speed,
        direction,
        PERIODS_HOURS,
        interval_minutes,
    )
    joint_perturbed = _joint_period_amplitude(
        speed_noisy,
        direction_noisy,
        PERIODS_HOURS,
        interval_minutes,
    )

    rows = []
    panels = [
        ("speed", "Wind-speed noise", speed_clean, speed_perturbed),
        (
            "direction",
            "Wind-direction noise",
            direction_clean,
            direction_perturbed,
        ),
        ("combined", "Combined input noise", joint_clean, joint_perturbed),
    ]
    for panel, _, clean, noisy in panels:
        for period, clean_value, noisy_value in zip(
            PERIODS_HOURS, clean, noisy
        ):
            rows.append(
                {
                    "panel": panel,
                    "period_hours": period,
                    "clean_spectral_magnitude": float(clean_value),
                    "noisy_spectral_magnitude": float(noisy_value),
                    "relative_change_pct": float(
                        (noisy_value - clean_value)
                        / max(abs(clean_value), 1e-12)
                        * 100.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _draw_panel(axis: plt.Axes, table: pd.DataFrame, panel: str, title: str, color: str) -> None:
    subset = table[table["panel"] == panel].sort_values(
        "period_hours", ascending=False
    )
    x = np.arange(len(subset))
    width = 0.34
    clean = subset["clean_spectral_magnitude"].to_numpy()
    noisy = subset["noisy_spectral_magnitude"].to_numpy()

    axis.bar(
        x - width / 2,
        clean,
        width,
        label="Clean samples",
        color="#5B7DB1",
        alpha=0.42,
        edgecolor="#5B7DB1",
        linewidth=1.5,
    )
    axis.bar(
        x + width / 2,
        noisy,
        width,
        label="Noisy samples",
        color=color,
        alpha=0.40,
        edgecolor=color,
        linewidth=1.5,
    )
    axis.set_title(title, fontsize=18, pad=14)
    axis.set_xticks(x, [str(int(value)) for value in subset["period_hours"]])
    axis.set_xlabel("Period (hours)", fontsize=14, labelpad=9)
    axis.grid(axis="y", linestyle=":", linewidth=0.9, color="#cfd6dd")
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=12)
    axis.legend(
        loc="upper right",
        fontsize=10.5,
        frameon=True,
        facecolor="white",
        edgecolor="#c8cdd2",
        framealpha=0.92,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_noise_interference(
    dataset: str,
    data_path: Path,
    output_path: Path,
    data_output_path: Path,
    seed: int,
) -> None:
    speed, direction, metadata, interval_minutes = _load_processed(data_path)
    speed_mean = np.asarray([], dtype=np.float64)
    speed_std = np.asarray([], dtype=np.float64)
    table = _make_plot_table(
        speed,
        direction,
        speed_mean,
        speed_std,
        interval_minutes,
        seed,
    )
    data_output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(data_output_path, index=False, encoding="utf-8-sig")

    colors = {
        "speed": "#E5B341",
        "direction": "#45B7A5",
        "combined": "#E28E53",
    }
    titles = {
        "speed": "(a) Wind-speed noise",
        "direction": "(b) Wind-direction noise",
        "combined": "(c) Combined input noise",
    }
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.8, 5.8),
        sharey=True,
        dpi=260,
    )
    fig.patch.set_facecolor("white")
    for axis, panel in zip(axes, ("speed", "direction", "combined")):
        _draw_panel(axis, table, panel, titles[panel], colors[panel])
    axes[0].set_ylabel("Spectral magnitude", fontsize=15, labelpad=10)
    fig.suptitle(
        f"Noise interference across characteristic periods ({dataset.title()})",
        fontsize=20,
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"figure={output_path}")
    print(f"data={data_output_path}")
    print(f"dataset={dataset}")
    print(f"test_rows={len(speed)}")
    print(f"turbines={speed.shape[1]}")
    print(f"noise_level={NOISE_LEVEL}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot clean versus noisy period spectra from processed wind data."
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="shanxi")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--data-output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    data_path = args.data or DATASETS[args.dataset]
    output_path = args.output or (
        ROOT / "figures" / f"noise_interference_period_spectrum_{args.dataset}.png"
    )
    data_output_path = args.data_output or (
        ROOT / "figures" / f"noise_interference_period_spectrum_{args.dataset}.csv"
    )
    plot_noise_interference(
        args.dataset,
        data_path,
        output_path,
        data_output_path,
        args.seed,
    )


if __name__ == "__main__":
    main()
