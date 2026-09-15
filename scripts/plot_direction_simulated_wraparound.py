from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "figures"
    / "penmanshiel_intro"
    / "07_direction_geometry"
    / "png"
    / "figure7_simulated_direction_wraparound.png"
)


def _circular_interpolate(values: pd.Series) -> pd.Series:
    radians = np.deg2rad(values)
    sine = pd.Series(np.sin(radians), index=values.index).interpolate()
    cosine = pd.Series(np.cos(radians), index=values.index).interpolate()
    return pd.Series((np.rad2deg(np.arctan2(sine, cosine)) + 360) % 360, index=values.index)


def _angle_xy(degrees: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radians = np.deg2rad(np.asarray(degrees, dtype=np.float64))
    return np.cos(radians), np.sin(radians)


def _plot_angle_arc(
    axis: plt.Axes,
    degrees: np.ndarray,
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    label: str,
) -> None:
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(degrees)))
    samples = []
    for start, end in zip(unwrapped[:-1], unwrapped[1:]):
        samples.append(np.linspace(start, end, 80, endpoint=False))
    samples.append(np.array([unwrapped[-1]]))
    x_values, y_values = _angle_xy(np.concatenate(samples))
    axis.plot(
        x_values,
        y_values,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
    )


def main() -> None:
    steps = np.arange(5)
    raw = pd.Series([350.0, np.nan, np.nan, np.nan, 10.0], index=steps)
    scalar = raw.interpolate(method="linear")
    circular = _circular_interpolate(raw)

    scalar_filled = scalar.where(raw.isna())
    circular_filled = circular.where(raw.isna())

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    colors = {
        "raw": "#262626",
        "scalar": "#c23b22",
        "circular": "#2563a9",
        "gap": "#f1c232",
    }

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.4, 3.55),
        gridspec_kw={"width_ratios": [1.08, 1.08, 0.86], "wspace": 0.36},
    )

    for axis in axes[:2]:
        axis.axvspan(0.5, 3.5, color=colors["gap"], alpha=0.16, lw=0)
        axis.set_xlim(-0.2, 4.2)
        axis.set_ylim(0, 360)
        axis.set_xticks(steps)
        axis.set_yticks([0, 90, 180, 270, 360])
        axis.set_xlabel("Time step")
        axis.set_ylabel("Direction (deg)")
        axis.grid(axis="y", color="0.88", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].plot(
        steps,
        scalar,
        color=colors["scalar"],
        linestyle="--",
        linewidth=2.2,
        label="scalar fill",
    )
    axes[0].scatter(steps, raw, color=colors["raw"], s=34, label="observed", zorder=4)
    axes[0].scatter(
        steps,
        scalar_filled,
        facecolors="white",
        edgecolors=colors["scalar"],
        linewidths=1.7,
        s=64,
        marker="D",
        label="filled",
        zorder=5,
    )
    axes[0].set_title("(a) Scalar interpolation", loc="center")
    axes[0].legend(frameon=False, loc="center left", fontsize=10)
    for step, value in scalar_filled.dropna().items():
        axes[0].text(step, value + 13, f"{value:.0f} deg", ha="center", fontsize=9, color=colors["scalar"])

    axes[1].plot(
        steps[:2],
        circular.iloc[:2],
        color=colors["circular"],
        linewidth=2.2,
        label="circular fill",
    )
    axes[1].plot(
        steps[2:],
        circular.iloc[2:],
        color=colors["circular"],
        linewidth=2.2,
    )
    axes[1].scatter(steps, raw, color=colors["raw"], s=34, label="observed", zorder=4)
    axes[1].scatter(
        steps,
        circular_filled,
        facecolors="white",
        edgecolors=colors["circular"],
        linewidths=1.7,
        s=64,
        marker="D",
        label="filled",
        zorder=5,
    )
    axes[1].axhline(360, color="0.55", linewidth=0.8, linestyle=":")
    axes[1].axhline(0, color="0.55", linewidth=0.8, linestyle=":")
    axes[1].text(
        2.2,
        306,
        "short path through\n360/0 deg",
        ha="center",
        fontsize=9,
        color="0.35",
    )
    axes[1].set_title("(b) Circular interpolation", loc="center")
    axes[1].legend(frameon=False, loc="center left", fontsize=10)
    for step, value in circular_filled.dropna().items():
        label_y = 18 if value < 12 else value - 48
        axes[1].text(step, label_y, f"{value:.0f} deg", ha="center", fontsize=9, color=colors["circular"])

    theta = np.linspace(0, 2 * np.pi, 400)
    axes[2].plot(np.cos(theta), np.sin(theta), color="0.25", linewidth=0.9)
    axes[2].axhline(0, color="0.88", linewidth=0.6)
    axes[2].axvline(0, color="0.88", linewidth=0.6)
    axes[2].set_aspect("equal")
    axes[2].set_xlim(-1.12, 1.12)
    axes[2].set_ylim(-1.12, 1.12)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    for spine in axes[2].spines.values():
        spine.set_visible(False)
    axes[2].set_title("(c) Paths on the unit circle", loc="center")
    _plot_angle_arc(
        axes[2],
        scalar.to_numpy(),
        color=colors["scalar"],
        linestyle="--",
        linewidth=2.1,
        label="scalar path",
    )
    circular_path = np.array([350.0, 355.0, 360.0, 365.0, 370.0])
    _plot_angle_arc(
        axes[2],
        circular_path,
        color=colors["circular"],
        linestyle="-",
        linewidth=2.4,
        label="circular path",
    )
    raw_x, raw_y = _angle_xy(raw.dropna().to_numpy())
    scalar_x, scalar_y = _angle_xy(scalar_filled.dropna().to_numpy())
    circular_x, circular_y = _angle_xy(circular_filled.dropna().to_numpy())
    axes[2].scatter(raw_x, raw_y, color=colors["raw"], s=34, zorder=5)
    axes[2].scatter(
        scalar_x,
        scalar_y,
        facecolors="white",
        edgecolors=colors["scalar"],
        linewidths=1.6,
        s=54,
        marker="D",
        zorder=6,
    )
    axes[2].scatter(
        circular_x,
        circular_y,
        facecolors="white",
        edgecolors=colors["circular"],
        linewidths=1.6,
        s=54,
        marker="D",
        zorder=7,
    )
    axes[2].text(1.04, 0.04, "0/360", fontsize=9, color="0.35")
    axes[2].text(-1.05, 0.04, "180", fontsize=9, color="0.35", ha="right")
    axes[2].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.2), fontsize=9)

    fig.suptitle("Simulated wind direction wraparound example", y=0.99, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
