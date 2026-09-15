from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wind_repro.data import _penmanshiel_turbine_id, _read_penmanshiel_entry  # noqa: E402


ARCHIVE = (
    ROOT
    / "data"
    / "raw"
    / "penmanshiel"
    / "Penmanshiel_SCADA_2022_WT11-15_4463.zip"
)
TURBINES = [f"WT{index:02d}" for index in range(11, 16)]
WINDOW_START = pd.Timestamp("2022-03-27 09:30:00")
WINDOW_END = pd.Timestamp("2022-03-27 13:30:00")
OUTPUT = (
    ROOT
    / "figures"
    / "penmanshiel_intro"
    / "07_direction_geometry"
    / "png"
    / "figure7_wind_direction_across_turbines.png"
)


def _load_turbine_data() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        entries = sorted(
            name
            for name in archive.namelist()
            if Path(name).name.startswith("Turbine_Data_Penmanshiel_")
            and name.lower().endswith(".csv")
        )
        for entry in entries:
            turbine_id = _penmanshiel_turbine_id(entry)
            if turbine_id not in TURBINES:
                continue
            frame = _read_penmanshiel_entry(
                archive,
                entry,
                {
                    "timestamp_column": None,
                    "speed_column": None,
                    "direction_column": None,
                },
            )
            frame = frame.sort_values("timestamp").drop_duplicates(
                "timestamp", keep="last"
            )
            direction = pd.to_numeric(frame["wind_direction"], errors="coerce")
            if (
                not direction.dropna().empty
                and direction.dropna().quantile(0.99) <= 2 * np.pi + 0.1
            ):
                direction = np.rad2deg(direction)
            direction[(direction < 0) | (direction > 360)] = np.nan
            current = pd.DataFrame(
                {
                    f"{turbine_id}_direction": direction.to_numpy(),
                    f"{turbine_id}_speed": pd.to_numeric(
                        frame["wind_speed"], errors="coerce"
                    ).to_numpy(),
                },
                index=frame["timestamp"],
            )
            frames.append(current)

    data = pd.concat(frames, axis=1).sort_index()
    full_index = pd.date_range(data.index.min(), data.index.max(), freq="10min")
    return data.reindex(full_index)


def _plot_wrapped(
    axis: plt.Axes,
    x_values: pd.Index,
    values: pd.Series,
    *,
    color: str,
    linewidth: float = 1.8,
) -> None:
    numeric = values.to_numpy(dtype=np.float64)
    start = 0
    for index in range(1, len(numeric) + 1):
        split = index == len(numeric)
        if index < len(numeric):
            left = numeric[index - 1]
            right = numeric[index]
            split = (
                not np.isfinite(left)
                or not np.isfinite(right)
                or abs(right - left) > 180
            )
        if split:
            if index > start:
                axis.plot(
                    x_values[start:index],
                    numeric[start:index],
                    color=color,
                    linewidth=linewidth,
                    solid_capstyle="round",
                )
            start = index


def _circular_mean(values: pd.DataFrame) -> pd.Series:
    radians = np.deg2rad(values.to_numpy(dtype=np.float64))
    valid = np.isfinite(radians)
    vectors = np.where(valid, np.exp(1j * radians), np.nan + 1j * np.nan)
    mean_vector = np.nanmean(vectors, axis=1)
    return pd.Series(
        (np.rad2deg(np.angle(mean_vector)) + 360) % 360,
        index=values.index,
        name="farm_mean_direction",
    )


def _relative_direction(values: pd.DataFrame, mean_direction: pd.Series) -> pd.DataFrame:
    relative = values.subtract(mean_direction, axis=0)
    return (relative + 180) % 360 - 180


def main() -> None:
    data = _load_turbine_data()
    window = data.loc[WINDOW_START:WINDOW_END]
    direction_columns = [f"{turbine}_direction" for turbine in TURBINES]
    speed_columns = [f"{turbine}_speed" for turbine in TURBINES]
    directions = window[direction_columns].copy()
    directions.columns = TURBINES
    speeds = window[speed_columns].copy()
    speeds.columns = TURBINES
    mean_direction = _circular_mean(directions)
    relative_direction = _relative_direction(directions, mean_direction)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9.2,
            "axes.titlesize": 11,
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.8,
        }
    )
    colors = {
        "WT11": "#0072B2",
        "WT12": "#D55E00",
        "WT13": "#009E73",
        "WT14": "#CC79A7",
        "WT15": "#E69F00",
    }

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.7, 6.7),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 0.9, 1.0]},
    )
    fig.patch.set_facecolor("white")

    for axis in axes:
        axis.grid(axis="y", color="0.9", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("0.25")
        axis.spines["bottom"].set_color("0.25")
        axis.tick_params(length=3, width=0.75)
        axis.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    direction_axis, speed_axis, relative_axis = axes
    for turbine in TURBINES:
        _plot_wrapped(
            direction_axis,
            directions.index,
            directions[turbine],
            color=colors[turbine],
        )
        speed_axis.plot(
            speeds.index,
            speeds[turbine],
            color=colors[turbine],
            linewidth=1.45,
            solid_capstyle="round",
        )
        relative_axis.plot(
            relative_direction.index,
            relative_direction[turbine],
            color=colors[turbine],
            linewidth=1.55,
            solid_capstyle="round",
        )

    direction_axis.set_ylim(0, 360)
    direction_axis.set_yticks([0, 90, 180, 270, 360])
    direction_axis.axhline(0, color="0.55", linewidth=0.7, linestyle=":")
    direction_axis.axhline(360, color="0.55", linewidth=0.7, linestyle=":")
    direction_axis.set_ylabel("Direction (deg)")
    direction_axis.set_title("(a) Wind direction", loc="center", pad=5)

    speed_axis.set_ylim(bottom=0)
    speed_axis.set_ylabel("Wind speed (m/s)")
    speed_axis.set_title("(b) Wind speed", loc="center", pad=5)

    relative_axis.axhline(0, color="0.3", linewidth=0.8)
    relative_axis.set_ylim(-180, 180)
    relative_axis.set_yticks([-180, -90, 0, 90, 180])
    relative_axis.set_ylabel("Relative direction (deg)")
    relative_axis.set_xlabel("Time on 2022-03-27")
    relative_axis.set_title(
        "(c) Relative direction to farm mean",
        loc="center",
        pad=5,
    )

    handles = [
        mlines.Line2D(
            [],
            [],
            color=colors[turbine],
            linewidth=2,
            label=turbine,
        )
        for turbine in TURBINES
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=5,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.1, right=0.985, top=0.92, bottom=0.09, hspace=0.34)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
