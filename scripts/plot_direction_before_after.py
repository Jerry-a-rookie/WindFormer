from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wind_repro.data import _penmanshiel_turbine_id, _read_penmanshiel_entry  # noqa: E402


TURBINE_ID = "WT11"
ARCHIVE = ROOT / "data" / "raw" / "penmanshiel" / "Penmanshiel_SCADA_2022_WT11-15_4463.zip"
WINDOW_START = pd.Timestamp("2022-07-28 07:00:00")
WINDOW_END = pd.Timestamp("2022-07-28 10:50:00")
OUTPUT = (
    ROOT
    / "figures"
    / "penmanshiel_intro"
    / "07_direction_geometry"
    / "png"
    / "figure7_wt11_direction_before_after.png"
)


def _load_direction_series() -> pd.Series:
    with zipfile.ZipFile(ARCHIVE) as archive:
        entries = sorted(
            name
            for name in archive.namelist()
            if Path(name).name.startswith("Turbine_Data_Penmanshiel_")
            and name.lower().endswith(".csv")
        )
        entry = next(
            name for name in entries if _penmanshiel_turbine_id(name) == TURBINE_ID
        )
        frame = _read_penmanshiel_entry(
            archive,
            entry,
            {
                "timestamp_column": None,
                "speed_column": None,
                "direction_column": None,
            },
        )

    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    direction = pd.to_numeric(frame["wind_direction"], errors="coerce")
    if not direction.dropna().empty and direction.dropna().quantile(0.99) <= 2 * np.pi + 0.1:
        direction = np.rad2deg(direction)
    direction[(direction < 0) | (direction > 360)] = np.nan
    series = pd.Series(direction.to_numpy(), index=frame["timestamp"], name="raw_direction")
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="10min")
    return series.reindex(full_index)


def _circular_interpolate(direction_degrees: pd.Series) -> pd.Series:
    radians = np.deg2rad(direction_degrees)
    sine = pd.Series(np.sin(radians), index=direction_degrees.index).interpolate(
        method="linear", limit_direction="both"
    )
    cosine = pd.Series(np.cos(radians), index=direction_degrees.index).interpolate(
        method="linear", limit_direction="both"
    )
    circular = (np.rad2deg(np.arctan2(sine, cosine)) + 360.0) % 360.0
    return pd.Series(circular, index=direction_degrees.index, name="circular")


def _plot_wrapped_line(
    axis: plt.Axes,
    x_values: pd.Index,
    y_values: pd.Series,
    *,
    color: str,
    label: str,
    linewidth: float,
) -> None:
    values = y_values.to_numpy(dtype=np.float64)
    start = 0
    label_pending = True
    for index in range(1, len(values) + 1):
        split = index == len(values)
        if index < len(values) and np.isfinite(values[index - 1]) and np.isfinite(values[index]):
            split = abs(values[index] - values[index - 1]) > 180.0
        if split:
            segment_label = label if label_pending else None
            axis.plot(
                x_values[start:index],
                values[start:index],
                color=color,
                linewidth=linewidth,
                label=segment_label,
            )
            label_pending = False
            start = index


def _clock_xy(degrees: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(degrees)
    return np.sin(theta), np.cos(theta)


def _short_clock_arc(start: float, end: float, samples: int = 80) -> np.ndarray:
    delta = ((end - start + 180.0) % 360.0) - 180.0
    return (np.linspace(start, start + delta, samples) + 360.0) % 360.0


def _long_clock_arc(start: float, end: float, samples: int = 120) -> np.ndarray:
    delta = ((end - start + 180.0) % 360.0) - 180.0
    if delta >= 0:
        delta -= 360.0
    else:
        delta += 360.0
    return (np.linspace(start, start + delta, samples) + 360.0) % 360.0


def _plot_clock_arc(
    axis: plt.Axes,
    degrees: np.ndarray,
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    label: str,
) -> None:
    x_values, y_values = _clock_xy(degrees)
    axis.plot(
        x_values,
        y_values,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
    )


def _add_clock_arrow(axis: plt.Axes, degree: float, color: str) -> None:
    start = np.array(_clock_xy(degree - 3.0))
    end = np.array(_clock_xy(degree + 3.0))
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": 1.15,
            "mutation_scale": 10,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=6,
    )


def _setup_clock_axis(axis: plt.Axes) -> None:
    face = plt.Circle((0, 0), 1.0, facecolor="#fbfbfb", edgecolor="0.64", linewidth=1.0)
    axis.add_patch(face)
    for degree in range(0, 360, 30):
        outer = np.array(_clock_xy(degree))
        inner = np.array(_clock_xy(degree)) * (0.88 if degree % 90 == 0 else 0.93)
        axis.plot(
            [inner[0], outer[0]],
            [inner[1], outer[1]],
            color="0.72",
            linewidth=0.9 if degree % 90 == 0 else 0.55,
        )
    for degree, label, ha, va in [
        (0, "0/360", "center", "bottom"),
        (90, "90", "left", "center"),
        (180, "180", "center", "top"),
        (270, "270", "right", "center"),
    ]:
        x_label, y_label = _clock_xy(degree)
        axis.text(
            x_label * 1.14,
            y_label * 1.14,
            label,
            ha=ha,
            va=va,
            color="0.32",
            fontsize=7.8,
        )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-1.28, 1.28)
    axis.set_ylim(-1.23, 1.23)
    axis.axis("off")


def main() -> None:
    raw = _load_direction_series()
    scalar = raw.interpolate(method="linear", limit_direction="both")
    circular = _circular_interpolate(raw)

    window_raw = raw.loc[WINDOW_START:WINDOW_END]
    window_scalar = scalar.loc[WINDOW_START:WINDOW_END]
    window_circular = circular.loc[WINDOW_START:WINDOW_END]
    window_scalar_filled = window_scalar.where(window_raw.isna())
    window_circular_filled = window_circular.where(window_raw.isna())

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9.2,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.6,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "legend.fontsize": 8.2,
        }
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(9.9, 3.05),
        gridspec_kw={"width_ratios": [1.08, 1.08, 0.82]},
    )
    colors = {
        "raw": "#333333",
        "before": "#bd3f2a",
        "after": "#2467a6",
        "gap": "#e9cf79",
        "gap_edge": "#d4a620",
    }

    for axis in axes[:2]:
        axis.axvspan(
            pd.Timestamp("2022-07-28 09:50:00"),
            pd.Timestamp("2022-07-28 10:00:00"),
            color=colors["gap"],
            alpha=0.14,
            lw=0,
        )
        axis.axvline(
            pd.Timestamp("2022-07-28 09:50:00"),
            color=colors["gap_edge"],
            linewidth=0.65,
            alpha=0.36,
        )
        axis.axvline(
            pd.Timestamp("2022-07-28 10:00:00"),
            color=colors["gap_edge"],
            linewidth=0.65,
            alpha=0.36,
        )
        axis.grid(axis="y", color="0.9", linewidth=0.58)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("0.2")
        axis.spines["bottom"].set_color("0.2")
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axis.tick_params(axis="x", rotation=30, length=3.2, width=0.8)
        axis.tick_params(axis="y", length=3.2, width=0.8)
        axis.set_ylim(0, 360)
        axis.set_yticks([0, 90, 180, 270, 360])
        axis.set_xlabel("Time on 2022-07-28", labelpad=3)

    axes[0].plot(
        window_scalar.index,
        window_scalar,
        color=colors["before"],
        linestyle="--",
        linewidth=1.85,
        label="linear",
    )
    axes[0].scatter(
        window_raw.index,
        window_raw,
        color=colors["raw"],
        s=16,
        alpha=0.78,
        label="observed",
        zorder=3,
    )
    axes[0].scatter(
        window_scalar_filled.index,
        window_scalar_filled,
        facecolors="white",
        edgecolors=colors["before"],
        linewidths=1.45,
        s=42,
        marker="D",
        label="filled gap",
        zorder=4,
    )
    axes[0].set_title("(a) Scalar interpolation", loc="center")
    axes[0].set_ylabel("Direction (deg)")
    axes[0].legend(frameon=False, loc="upper left", handlelength=1.55, borderaxespad=0.25)

    _plot_wrapped_line(
        axes[1],
        window_circular.index,
        window_circular,
        color=colors["after"],
        linewidth=1.95,
        label="circular",
    )
    axes[1].scatter(
        window_raw.index,
        window_raw,
        color=colors["raw"],
        s=16,
        alpha=0.78,
        label="observed",
        zorder=3,
    )
    axes[1].scatter(
        window_circular_filled.index,
        window_circular_filled,
        facecolors="white",
        edgecolors=colors["after"],
        linewidths=1.45,
        s=42,
        marker="D",
        label="filled gap",
        zorder=4,
    )
    axes[1].axhline(360, color="0.6", linewidth=0.7, linestyle=":")
    axes[1].axhline(0, color="0.6", linewidth=0.7, linestyle=":")
    axes[1].set_title("(b) Circular interpolation", loc="center")
    axes[1].legend(frameon=False, loc="upper left", handlelength=1.55, borderaxespad=0.25)

    clock_axis = axes[2]
    _setup_clock_axis(clock_axis)
    observed_points = window_raw.dropna()
    gap_start = pd.Timestamp("2022-07-28 09:40:00")
    gap_end = pd.Timestamp("2022-07-28 10:10:00")
    anchor_before = float(observed_points.loc[gap_start])
    anchor_after = float(observed_points.loc[gap_end])
    scalar_path = np.array(
        [
            anchor_before,
            *window_scalar_filled.dropna().to_numpy(dtype=np.float64),
            anchor_after,
        ]
    )
    circular_path = np.array(
        [
            anchor_before,
            *window_circular_filled.dropna().to_numpy(dtype=np.float64),
            anchor_after,
        ]
    )
    _plot_clock_arc(
        clock_axis,
        _long_clock_arc(anchor_before, anchor_after),
        color=colors["before"],
        linestyle="--",
        linewidth=1.9,
        label="scalar",
    )
    _plot_clock_arc(
        clock_axis,
        _short_clock_arc(anchor_before, anchor_after),
        color=colors["after"],
        linestyle="-",
        linewidth=2.05,
        label="circular",
    )
    _add_clock_arrow(clock_axis, 42, colors["after"])
    _add_clock_arrow(clock_axis, 138, colors["before"])
    for values, color, marker, size, label in [
        (np.array([anchor_before, anchor_after]), colors["raw"], "o", 24, "observed"),
        (scalar_path[1:-1], colors["before"], "D", 30, "linear fill"),
        (circular_path[1:-1], colors["after"], "D", 30, "circular fill"),
    ]:
        x_values, y_values = _clock_xy(values)
        clock_axis.scatter(
            x_values,
            y_values,
            s=size,
            marker=marker,
            facecolors="white" if marker == "D" else color,
            edgecolors=color,
            linewidths=1.35,
            zorder=5,
            label=label,
        )
    clock_axis.set_title("(c) Clockwise geometry", loc="center", pad=7)
    clock_axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.3),
        ncol=2,
        handlelength=1.25,
        columnspacing=0.65,
        borderaxespad=0,
    )

    fig.tight_layout(w_pad=1.15)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
