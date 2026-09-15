"""Plot a 3D Penmanshiel terrain surface with the zero-shot turbine split."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))

from plot_penmanshiel_terrain_map import (  # noqa: E402
    assemble_grid,
    wgs84_to_bng,
)


STATIC_CSV = ROOT / "data" / "raw" / "penmanshiel" / "Penmanshiel_WT_static.csv"
FIG_DIR = ROOT / "figures"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 16,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


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
                    "elevation": float(row["Elevation (m)"]),
                    "hub_height": float(row["Hub Height (m)"]),
                }
            )
    return turbines


def nearest_dem_elevation(
    easting: float,
    northing: float,
    elevation: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    ix = int(np.abs(x - easting).argmin())
    iy = int(np.abs(y - northing).argmin())
    value = float(elevation[iy, ix])
    return value if np.isfinite(value) else float(np.nanmedian(elevation))


def add_flag_marker(
    ax,
    x: float,
    y: float,
    ground_z: float,
    item_id: str,
    color: str,
    hub_height: float,
    terrain_max: float,
) -> float:
    """Add a prominent flagpole marker above the terrain surface."""
    pole_top = max(
        ground_z + max(hub_height + 28.0, 125.0),
        terrain_max + 92.0,
    )
    flag_width = 125.0
    flag_height = 38.0
    # Orient each flag face toward the selected 3D camera azimuth.
    camera_azimuth = np.deg2rad(-58.0)
    flag_dx = -np.sin(camera_azimuth) * flag_width
    flag_dy = np.cos(camera_azimuth) * flag_width

    ax.plot(
        [x, x],
        [y, y],
        [ground_z, pole_top],
        color="#2B2B2B",
        linewidth=1.9,
        solid_capstyle="round",
        zorder=12,
    )
    ax.plot(
        [x, x],
        [y, y],
        [ground_z, pole_top],
        color=color,
        linewidth=0.9,
        solid_capstyle="round",
        zorder=13,
    )

    flag_vertices = [
        (x, y, pole_top),
        (x + flag_dx, y + flag_dy, pole_top),
        (x + flag_dx, y + flag_dy, pole_top - flag_height),
        (x, y, pole_top - flag_height),
    ]
    ax.add_collection3d(
        Poly3DCollection(
            [flag_vertices],
            facecolors=color,
            edgecolors="white",
            linewidths=0.9,
            alpha=0.98,
            zorder=14,
        )
    )
    turbine_number = int(item_id[1:]) if item_id[1:].isdigit() else 0
    lateral_offset = 24.0 + 22.0 * (turbine_number % 3)
    vertical_offset = (-8.0, 5.0, 14.0)[turbine_number % 3]
    ax.text(
        x + flag_dx + (-flag_dy / flag_width) * lateral_offset,
        y + flag_dy + (flag_dx / flag_width) * lateral_offset,
        pole_top + vertical_offset,
        item_id,
        fontsize=7.4,
        fontweight="bold",
        color="#17202A",
        bbox={
            "boxstyle": "round,pad=0.12",
            "facecolor": "white",
            "edgecolor": "#D6DDE3",
            "linewidth": 0.35,
            "alpha": 0.92,
        },
        zorder=15,
    )
    return pole_top


def main() -> None:
    turbines = read_turbines()
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

    finite = clipped[np.isfinite(clipped)]
    zmin, zmax = float(np.min(finite)), float(np.max(finite))
    ls = LightSource(azdeg=315, altdeg=42)
    colors = ls.shade(
        clipped,
        cmap=plt.get_cmap("terrain"),
        norm=Normalize(vmin=np.percentile(finite, 2), vmax=np.percentile(finite, 98)),
        vert_exag=1.35,
        dx=cellsize,
        dy=cellsize,
        blend_mode="soft",
    )

    fig = plt.figure(figsize=(11.6, 8.3), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False
    ax.plot_surface(
        X,
        Y,
        clipped,
        facecolors=colors,
        linewidth=0,
        antialiased=True,
        shade=False,
        rstride=1,
        cstride=1,
        zorder=1,
    )

    contour_levels = np.arange(np.floor(zmin / 20) * 20, np.ceil(zmax / 20) * 20 + 1, 20)
    ax.contour(
        X,
        Y,
        clipped,
        levels=contour_levels,
        zdir="z",
        offset=zmin - 8,
        colors="#594638",
        linewidths=0.45,
        alpha=0.55,
        zorder=2,
    )

    turbine_handle = Patch(
        facecolor="#0072B2",
        edgecolor="white",
        label="Wind turbine",
    )
    flag_handle = Line2D(
        [0],
        [0],
        color="#2B2B2B",
        linewidth=2.6,
        marker="|",
        markerfacecolor="#2B2B2B",
        markeredgecolor="#2B2B2B",
        markersize=12,
        label="Flagpole marker (height enlarged)",
    )

    for item in turbines:
        ground_z = nearest_dem_elevation(
            float(item["easting"]),
            float(item["northing"]),
            elevation,
            x,
            y,
        )
        add_flag_marker(
            ax=ax,
            x=float(item["easting"]),
            y=float(item["northing"]),
            ground_z=ground_z,
            item_id=str(item["id"]),
            color="#0072B2",
            hub_height=float(item["hub_height"]),
            terrain_max=zmax,
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin - 8, zmax + 155)
    ax.set_box_aspect((xmax - xmin, ymax - ymin, 0.28 * (xmax - xmin)))
    ax.view_init(elev=31, azim=-58)
    ax.set_xlabel("Easting (m), EPSG:27700", labelpad=8)
    ax.set_ylabel("Northing (m), EPSG:27700", labelpad=8)
    ax.set_zlabel("Height (m)", labelpad=5)
    ax.set_title(
        "Penmanshiel 3D terrain and turbine split",
        pad=10,
        fontweight="bold",
    )
    ax.legend(
        handles=[turbine_handle, flag_handle],
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        frameon=True,
        framealpha=0.94,
        borderpad=0.65,
        fontsize=9,
    )
    ax.grid(color="#AEB8C2", alpha=0.28, linewidth=0.55)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.96, 0.97, 0.98, 1.0))
        axis.pane.set_edgecolor("#D4DCE3")
    ax.text2D(
        0.02,
        0.02,
        "Flagpole heights are visually enlarged; turbine coordinates are unchanged.",
        transform=ax.transAxes,
        fontsize=7.6,
        color="#202020",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#D6DDE3",
            "alpha": 0.88,
        },
    )

    scalar = plt.cm.ScalarMappable(
        norm=Normalize(vmin=np.percentile(finite, 2), vmax=np.percentile(finite, 98)),
        cmap="terrain",
    )
    scalar.set_array([])
    colorbar = fig.colorbar(
        scalar,
        ax=ax,
        orientation="horizontal",
        fraction=0.025,
        pad=0.10,
        shrink=0.52,
        aspect=50,
    )
    colorbar.set_label("Terrain elevation (m)")
    colorbar.ax.tick_params(labelsize=8.5, length=3)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIG_DIR / "penmanshiel_terrain_3d_zero_shot.png"
    svg_path = FIG_DIR / "penmanshiel_terrain_3d_zero_shot.svg"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")


if __name__ == "__main__":
    main()
