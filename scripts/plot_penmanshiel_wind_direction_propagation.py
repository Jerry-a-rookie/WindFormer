"""Plot dominant wind-direction propagation over Penmanshiel terrain.

The arrows show the downwind propagation direction. The source SCADA direction
is treated as a meteorological "coming from" direction and rotated by 180
degrees for the displayed propagation arrows.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib import patheffects

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))

from plot_penmanshiel_terrain_map import (  # noqa: E402
    TRAIN_TURBINES,
    add_scale_bar,
    assemble_grid,
    hillshade,
    wgs84_to_bng,
)


STATIC_CSV = ROOT / "data" / "raw" / "penmanshiel" / "Penmanshiel_WT_static.csv"
PREPARED = ROOT / "data" / "processed" / "penmanshiel_2016_2022_h24_full_interp_fft.npz"
FIG_DIR = ROOT / "figures"


def read_turbines() -> list[dict[str, float | str]]:
    turbines: list[dict[str, float | str]] = []
    with STATIC_CSV.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("Latitude") or not row.get("Longitude"):
                continue
            easting, northing = wgs84_to_bng(
                float(row["Latitude"]), float(row["Longitude"])
            )
            turbines.append(
                {
                    "id": row["Alternative Title"],
                    "easting": easting,
                    "northing": northing,
                }
            )
    return turbines


def circular_direction_summary() -> tuple[list[str], np.ndarray, np.ndarray]:
    with np.load(PREPARED, allow_pickle=True) as archive:
        metadata = __import__("json").loads(str(archive["metadata"].item()))
        names = list(metadata["turbines"])
        direction_mean = archive["direction_mean"].astype(float)
        direction_std = archive["direction_std"].astype(float)
        direction_parts = [
            archive["train_direction"],
            archive["validation_direction"],
            archive["test_direction"],
        ]
        speed_parts = [
            archive["train_speed"],
            archive["validation_speed"],
            archive["test_speed"],
        ]
        direction = np.concatenate(direction_parts, axis=0).astype(float)
        speed = np.concatenate(speed_parts, axis=0).astype(float)

    direction = direction * direction_std[None, :] + direction_mean[None, :]
    # Load speed normalization separately because the np.load context is closed.
    with np.load(PREPARED, allow_pickle=True) as archive:
        speed_mean = archive["speed_mean"].astype(float)
        speed_std = archive["speed_std"].astype(float)
    speed = speed * speed_std[None, :] + speed_mean[None, :]

    mean_from = np.zeros(direction.shape[1], dtype=float)
    mean_speed = np.nanmean(speed, axis=0)
    for index in range(direction.shape[1]):
        valid = np.isfinite(direction[:, index])
        mean_from[index] = np.arctan2(
            np.nanmean(np.sin(direction[valid, index])),
            np.nanmean(np.cos(direction[valid, index])),
        ) % (2.0 * np.pi)
    return names, mean_from, mean_speed


def main() -> None:
    turbines = read_turbines()
    names, direction_from, mean_speed = circular_direction_summary()
    direction_by_name = dict(zip(names, direction_from))
    speed_by_name = dict(zip(names, mean_speed))

    elevation, x, y, cellsize = assemble_grid()
    tx = np.asarray([item["easting"] for item in turbines], dtype=float)
    ty = np.asarray([item["northing"] for item in turbines], dtype=float)
    buffer_m = 1800.0
    xmin, xmax = tx.min() - buffer_m, tx.max() + buffer_m
    ymin, ymax = ty.min() - buffer_m, ty.max() + buffer_m
    xmask = (x >= xmin) & (x <= xmax)
    ymask = (y >= ymin) & (y <= ymax)
    clipped = elevation[np.ix_(ymask, xmask)]
    xc, yc = x[xmask], y[ymask]
    X, Y = np.meshgrid(xc, yc)

    fig, ax = plt.subplots(figsize=(10.6, 8.0), constrained_layout=True)
    image = ax.pcolormesh(
        X,
        Y,
        clipped,
        shading="auto",
        cmap="terrain",
        vmin=np.nanpercentile(clipped, 2),
        vmax=np.nanpercentile(clipped, 98),
    )
    ax.imshow(
        hillshade(clipped, cellsize),
        extent=(xc.min(), xc.max(), yc.min(), yc.max()),
        origin="lower",
        cmap="gray",
        alpha=0.24,
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    levels = np.arange(
        np.floor(np.nanmin(clipped) / 20) * 20,
        np.ceil(np.nanmax(clipped) / 20) * 20 + 1,
        20,
    )
    ax.contour(X, Y, clipped, levels=levels, colors="#594638", linewidths=0.35, alpha=0.55)

    arrow_length = 650.0
    for item in turbines:
        name = str(item["id"])
        data_name = name.replace("T", "WT", 1)
        angle_from = direction_by_name[data_name]
        angle_to = (angle_from + np.pi) % (2.0 * np.pi)
        dx = arrow_length * np.sin(angle_to)
        dy = arrow_length * np.cos(angle_to)
        is_train = name in TRAIN_TURBINES
        marker = "^" if is_train else "o"
        color = "#D55E00" if is_train else "#0072B2"

        ax.scatter(
            [item["easting"]],
            [item["northing"]],
            s=96 if is_train else 78,
            marker=marker,
            color=color,
            edgecolors="white",
            linewidths=1.25,
            zorder=6,
        )
        ax.annotate(
            "",
            xy=(float(item["easting"]) + dx, float(item["northing"]) + dy),
            xytext=(float(item["easting"]), float(item["northing"])),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#202020",
                "lw": 1.7,
                "mutation_scale": 15,
                "shrinkA": 3,
                "shrinkB": 0,
            },
            zorder=5,
        )
        ax.text(
            float(item["easting"]) + 75,
            float(item["northing"]) + 48,
            f"{name}\n{np.rad2deg(angle_from):.0f}° from",
            fontsize=7.6,
            color="#17202A",
            fontweight="bold",
            zorder=7,
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            path_effects=[patheffects.withStroke(linewidth=1.6, foreground="white")],
        )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    colorbar.set_label("Elevation (m)")
    ax.set_xlabel("Easting (m), EPSG:27700")
    ax.set_ylabel("Northing (m), EPSG:27700")
    ax.set_title(
        "Penmanshiel dominant wind-direction propagation",
        pad=10,
        fontweight="bold",
    )
    ax.text(
        0.015,
        0.02,
        "Arrows show downwind propagation; labels report meteorological direction from which wind arrives.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#202020",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="white", alpha=0.22, linewidth=0.55)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="^", color="w", markerfacecolor="#D55E00", markeredgecolor="white", markersize=10, linestyle="None", label="Training turbine"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#0072B2", markeredgecolor="white", markersize=9, linestyle="None", label="Zero-shot prediction turbine"),
            Line2D([0], [0], color="#202020", lw=1.7, marker=">", markersize=7, label="Downwind propagation direction"),
        ],
        loc="upper right",
        frameon=True,
        framealpha=0.92,
    )
    add_scale_bar(ax, length_m=1000.0)
    ax.annotate(
        "N",
        xy=(0.95, 0.92),
        xytext=(0.95, 0.78),
        xycoords="axes fraction",
        ha="center",
        va="center",
        arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": "black"},
        fontsize=12,
        fontweight="bold",
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIG_DIR / "penmanshiel_wind_direction_propagation.png"
    svg_path = FIG_DIR / "penmanshiel_wind_direction_propagation.svg"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")
    print(
        "Mean wind speed range: "
        f"{float(np.nanmin(list(speed_by_name.values()))):.2f} to "
        f"{float(np.nanmax(list(speed_by_name.values()))):.2f} m/s"
    )


if __name__ == "__main__":
    main()
