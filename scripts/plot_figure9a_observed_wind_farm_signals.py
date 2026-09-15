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
DEFAULT_CACHE = ROOT / "figures" / "penmanshiel_intro" / "aligned_penmanshiel_arrays.npz"
DEFAULT_METADATA = ROOT / "figures" / "penmanshiel_intro" / "dataset_summary.json"
DEFAULT_OUTPUT = (
    ROOT
    / "figures"
    / "penmanshiel_intro"
    / "09_motivation"
    / "figure9a_observed_wind_farm_signals.png"
)
DEFAULT_DATA_OUTPUT = (
    ROOT
    / "figures"
    / "penmanshiel_intro"
    / "09_motivation"
    / "figure9a_observed_wind_farm_signals.csv"
)


def _decode_timestamps(values: np.ndarray) -> pd.DatetimeIndex:
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


def _select_window(
    speed: pd.DataFrame, interval_minutes: int, hours: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Select a representative contiguous window with visible farm dynamics."""
    window_steps = int(round(hours * 60 / interval_minutes))
    if len(speed) <= window_steps:
        selected = speed.copy()
        return selected, selected.mean(axis=1)

    values = speed.to_numpy(dtype=np.float64)
    common = np.nanmean(values, axis=1)
    stride = max(window_steps // 4, 1)
    best_start = 0
    best_score = -np.inf

    for start in range(0, len(speed) - window_steps + 1, stride):
        block = common[start : start + window_steps]
        if not np.isfinite(block).all():
            continue
        variation = float(np.std(block))
        range_score = float(np.ptp(block))
        smooth_change = float(np.mean(np.abs(np.diff(block))))
        score = variation + 0.25 * range_score + 0.15 * smooth_change
        if score > best_score:
            best_start = start
            best_score = score

    selected = speed.iloc[best_start : best_start + window_steps].copy()
    return selected, selected.mean(axis=1)


def plot_figure9a(
    cache_path: Path,
    metadata_path: Path,
    output_path: Path,
    data_output_path: Path,
    start: str | None = None,
    hours: int = 36,
) -> None:
    cache = np.load(cache_path, allow_pickle=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    timestamps = _decode_timestamps(cache["timestamps"])
    turbines = list(metadata["turbines"])
    speed = pd.DataFrame(cache["speed"], index=timestamps, columns=turbines)
    interval_minutes = int(metadata.get("interval_minutes", 10))

    selected_turbines = [
        turbine
        for turbine in ["WT04", "WT05", "WT06", "WT07", "WT08"]
        if turbine in speed.columns
    ]
    if len(selected_turbines) < 2:
        raise ValueError(f"Insufficient available turbines: {selected_turbines}")

    speed = speed[selected_turbines].interpolate(
        method="linear", limit_direction="both"
    )
    if start is not None:
        start_timestamp = pd.Timestamp(start)
        end_timestamp = start_timestamp + pd.Timedelta(hours=hours)
        selected = speed.loc[start_timestamp:end_timestamp].iloc[:-1]
        if len(selected) < 2:
            raise ValueError(f"No usable data found from start={start!r}")
        common = selected.mean(axis=1)
    else:
        selected, common = _select_window(speed, interval_minutes, hours)

    elapsed_hours = (
        (selected.index - selected.index[0]).total_seconds() / 3600.0
    )
    plot_data = selected.copy()
    plot_data.insert(0, "time_hours", elapsed_hours)
    plot_data.insert(0, "timestamp", selected.index)
    plot_data["common_field"] = common.to_numpy(dtype=np.float64)
    data_output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_data.to_csv(data_output_path, index=False)

    colors = {
        "WT04": "#C06C84",
        "WT05": "#7B6FB0",
        "WT06": "#E39A4A",
        "WT07": "#4C78A8",
        "WT08": "#59A14F",
    }

    # The original canvas was approximately 12.8 x 8 inches.
    # Keep the same height and reduce the x-direction canvas by half.
    fig, axis = plt.subplots(figsize=(6.4, 8.0), dpi=220)
    axis.set_facecolor("#fbfcfe")
    fig.patch.set_facecolor("white")

    for turbine in selected_turbines:
        axis.plot(
            elapsed_hours,
            selected[turbine].to_numpy(dtype=np.float64),
            color=colors.get(turbine, "#4C78A8"),
            linewidth=1.8,
            alpha=0.92,
            label=turbine,
        )
    axis.plot(
        elapsed_hours,
        common.to_numpy(dtype=np.float64),
        color="#111111",
        linewidth=3.6,
        label="Common field",
        zorder=5,
    )

    axis.set_title(
        "(a) Observed wind-farm signals",
        fontsize=22,
        fontweight="bold",
        pad=18,
    )
    axis.set_xlabel("Time (h)", fontsize=17, labelpad=10)
    axis.set_ylabel("Wind speed (m/s)", fontsize=17, labelpad=10)
    axis.set_xlim(float(elapsed_hours[0]), float(elapsed_hours[-1]))
    axis.set_xticks(np.arange(0, hours + 0.1, 6))
    axis.tick_params(axis="both", labelsize=13, colors="#111111")
    axis.grid(True, color="#dce3ea", linewidth=0.8, alpha=0.9)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#65717d")
    axis.spines["bottom"].set_color("#65717d")
    axis.legend(
        loc="lower right",
        ncol=2,
        fontsize=12.5,
        frameon=True,
        facecolor="white",
        edgecolor="#cbd5df",
        framealpha=0.94,
        borderpad=0.8,
        handlelength=2.2,
        columnspacing=1.0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"figure={output_path}")
    print(f"data={data_output_path}")
    print(f"window_start={selected.index[0]}")
    print(f"window_end={selected.index[-1]}")
    print(f"turbines={','.join(selected_turbines)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot only Figure 9(a): observed wind-farm signals."
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-output", type=Path, default=DEFAULT_DATA_OUTPUT)
    parser.add_argument(
        "--start",
        default=None,
        help="Optional fixed start timestamp, e.g. 2022-01-01 00:00:00.",
    )
    parser.add_argument("--hours", type=int, default=36)
    args = parser.parse_args()
    plot_figure9a(
        cache_path=args.cache,
        metadata_path=args.metadata,
        output_path=args.output,
        data_output_path=args.data_output,
        start=args.start,
        hours=args.hours,
    )


if __name__ == "__main__":
    main()
