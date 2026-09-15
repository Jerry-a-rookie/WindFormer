from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "Shanxi"
sys.path.insert(0, str(ROOT / "src"))

from wind_repro.data import _find_wide_wind_columns  # noqa: E402


TURBINES = [f"WT{i:02d}" for i in range(24)]
BLUE_GREEN = "#2A9D8F"
DEEP_TEAL = "#1F6F78"
NAVY_TEAL = "#173F5F"
THRESHOLD_BLUE = "#457B9D"
BLUE_GREEN_CMAP = LinearSegmentedColormap.from_list(
    "blue_green_direction",
    ["#EEF8F6", "#B7E4C7", "#52B69A", DEEP_TEAL, NAVY_TEAL],
    N=256,
)

FORMAL_OUTPUT = (
    ROOT / "正式文件" / "all_24_turbines_direction_difference_summary.png"
)
FIGURES_OUTPUT = (
    ROOT / "figures" / "all_24_turbines_direction_difference_summary.png"
)


def load_direction() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        frame = pd.read_csv(path)
        current = pd.DataFrame({"ts": pd.to_datetime(frame["ts"])})
        for turbine_id, _, direction_column in _find_wide_wind_columns(
            frame.columns
        ):
            if turbine_id.isdigit() and int(turbine_id) < len(TURBINES):
                current[f"WT{int(turbine_id):02d}"] = pd.to_numeric(
                    frame[direction_column], errors="coerce"
                )
        frames.append(current)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.sort_values("ts").drop_duplicates("ts").set_index("ts")
    full_index = pd.date_range(raw.index.min(), raw.index.max(), freq="10min")
    direction = raw.reindex(columns=TURBINES).reindex(full_index)

    # Interpolate on the unit circle so the 0/2-pi boundary remains continuous.
    sine = np.sin(direction).interpolate(method="linear", limit_direction="both")
    cosine = np.cos(direction).interpolate(
        method="linear", limit_direction="both"
    )
    norm = np.sqrt(sine**2 + cosine**2).clip(lower=1e-8)
    return pd.DataFrame(
        np.arctan2(sine / norm, cosine / norm),
        index=full_index,
        columns=TURBINES,
    )


def pairwise_distances(direction: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    radians = direction.to_numpy(dtype=float)
    matrix = np.zeros((len(TURBINES), len(TURBINES)), dtype=float)
    rows: list[tuple[str, str, float]] = []
    for left in range(len(TURBINES)):
        for right in range(left + 1, len(TURBINES)):
            distance = np.rad2deg(
                np.abs(
                    np.arctan2(
                        np.sin(radians[:, left] - radians[:, right]),
                        np.cos(radians[:, left] - radians[:, right]),
                    )
                )
            )
            mean_distance = float(np.mean(distance))
            matrix[left, right] = matrix[right, left] = mean_distance
            rows.append((TURBINES[left], TURBINES[right], mean_distance))
    return matrix, pd.DataFrame(
        rows,
        columns=["turbine_i", "turbine_j", "mean_angular_distance_deg"],
    )


def format_polar_axis(axis: plt.Axes) -> None:
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    axis.set_xticks(np.deg2rad([0, 90, 180, 270]))
    axis.set_xticklabels(["N", "E", "S", "W"], fontsize=15)
    axis.tick_params(axis="y", labelsize=14, pad=1)
    axis.grid(color="0.82", linewidth=0.55)


def draw_summary(direction: pd.DataFrame) -> None:
    matrix, pairwise = pairwise_distances(direction)
    radians = direction.to_numpy(dtype=float)
    deviations = np.rad2deg(
        np.abs(
            np.arctan2(
                np.sin(radians - np.arctan2(
                    np.sin(radians).mean(axis=1),
                    np.cos(radians).mean(axis=1),
                )[:, None]),
                np.cos(radians - np.arctan2(
                    np.sin(radians).mean(axis=1),
                    np.cos(radians).mean(axis=1),
                )[:, None]),
            )
        )
    )
    median_pairwise = float(pairwise["mean_angular_distance_deg"].median())
    share_over_90 = float(
        (pairwise["mean_angular_distance_deg"] >= 90.0).mean()
    )
    median_deviation = float(np.median(deviations.mean(axis=1)))

    display_turbines = TURBINES[::2]
    fig = plt.figure(figsize=(20, 16), dpi=180)
    outer = fig.add_gridspec(
        2,
        2,
        height_ratios=[2.35, 1.45],
        width_ratios=[1.25, 0.90],
        hspace=0.28,
        wspace=0.16,
    )
    # Keep the existing 3 x 4 placement, but reduce gaps so the roses grow.
    rose_grid = outer[0, :].subgridspec(3, 4, hspace=0.54, wspace=0.12)

    sector_edges = np.arange(0.0, 360.0 + 22.5, 22.5)
    sector_centers = (sector_edges[:-1] + sector_edges[1:]) / 2.0
    sector_widths = np.deg2rad(np.diff(sector_edges))
    rose_axes = [
        fig.add_subplot(rose_grid[row, col], projection="polar")
        for row in range(3)
        for col in range(4)
    ]
    for axis, turbine in zip(rose_axes, display_turbines):
        degrees = np.rad2deg(direction[turbine].to_numpy()) % 360.0
        counts, _ = np.histogram(degrees, bins=sector_edges)
        frequency = counts / max(counts.sum(), 1)
        local_rmax = max(
            0.15,
            np.ceil(float(np.max(frequency)) * 1.12 / 0.05) * 0.05,
        )
        axis.bar(
            np.deg2rad(sector_centers),
            frequency,
            width=sector_widths,
            color=BLUE_GREEN,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.92,
        )
        format_polar_axis(axis)
        axis.set_ylim(0, local_rmax)
        axis.set_yticks([local_rmax / 2, local_rmax])
        axis.set_yticklabels(
            [f"{local_rmax / 2:.2f}", f"{local_rmax:.2f}"],
            fontsize=14,
        )
        axis.set_title(turbine, fontsize=17, pad=5)

    histogram_axis = fig.add_subplot(outer[1, 0])
    histogram_axis.hist(
        pairwise["mean_angular_distance_deg"],
        bins=np.arange(0, 181, 10),
        color=BLUE_GREEN,
        edgecolor="white",
        linewidth=0.6,
    )
    histogram_axis.axvline(
        median_pairwise,
        color=NAVY_TEAL,
        linewidth=2,
        label=f"Median = {median_pairwise:.1f} deg",
    )
    histogram_axis.axvline(
        90,
        color=THRESHOLD_BLUE,
        linestyle="--",
        linewidth=1.2,
        label=f">=90 deg: {share_over_90:.0%} of pairs",
    )
    histogram_axis.set_xlim(0, 180)
    histogram_axis.set_xlabel(
        "Mean angular difference between turbine pairs (deg)",
        fontsize=20,
    )
    histogram_axis.set_ylabel("Number of turbine pairs", fontsize=20)
    histogram_axis.set_title(
        "(b) Pairwise mean direction-difference distribution",
        fontsize=23,
        pad=12,
    )
    histogram_axis.grid(axis="y", color="0.88", linewidth=0.7)
    histogram_axis.tick_params(axis="both", labelsize=18)
    histogram_axis.legend(frameon=False, loc="upper left", fontsize=18)

    heatmap_axis = fig.add_subplot(outer[1, 1])
    image = heatmap_axis.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        vmin=0,
        vmax=180,
        cmap=BLUE_GREEN_CMAP,
    )
    heatmap_axis.set_xticks(np.arange(len(TURBINES)))
    heatmap_axis.set_yticks(np.arange(len(TURBINES)))
    heatmap_axis.set_xticklabels(
        TURBINES,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=12,
    )
    heatmap_axis.set_yticklabels(TURBINES, fontsize=12)
    heatmap_axis.set_xlabel("Turbine", fontsize=18, labelpad=8)
    heatmap_axis.set_ylabel("Turbine", fontsize=18, labelpad=8)
    heatmap_axis.set_title(
        "(c) Pairwise mean angular-distance matrix",
        fontsize=23,
        pad=12,
    )
    heatmap_axis.grid(False)
    heatmap_colorbar = fig.colorbar(
        image,
        ax=heatmap_axis,
        pad=0.02,
        fraction=0.045,
    )
    heatmap_colorbar.set_label(
        "Mean angular distance (deg)",
        fontsize=20,
        labelpad=10,
    )
    heatmap_colorbar.ax.tick_params(labelsize=16)

    fig.text(
        0.5,
        0.965,
        "(a) Representative turbine wind roses",
        ha="center",
        fontsize=26,
        color="black",
    )
    fig.text(
        0.5,
        0.012,
        (
            "Representative turbines shown in (a); all 24 turbines used in (b)-(c).\n"
            f"Median pairwise mean angular distance: {median_pairwise:.1f} deg | "
            f"Pairs with mean distance >=90 deg: {share_over_90:.0%} | "
            f"Median within-farm mean deviation: {median_deviation:.1f} deg"
        ),
        ha="center",
        fontsize=17,
        linespacing=1.35,
        color="0.35",
    )

    FORMAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FORMAL_OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES_OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(FORMAL_OUTPUT)
    print(FIGURES_OUTPUT)


if __name__ == "__main__":
    draw_summary(load_direction())
