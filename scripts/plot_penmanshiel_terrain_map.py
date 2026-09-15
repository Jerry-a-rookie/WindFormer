"""Plot a publication-ready terrain map for the Penmanshiel wind farm.

Inputs:
  - OS Terrain 50 ASCII grids in data/raw/penmanshiel/terrain/
  - Penmanshiel turbine metadata CSV

Outputs:
  - figures/penmanshiel_terrain_map.png
  - figures/penmanshiel_terrain_map.svg
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as path_effects


ROOT = Path(__file__).resolve().parents[1]
TERRAIN_DIR = ROOT / "data" / "raw" / "penmanshiel" / "terrain"
STATIC_CSV = ROOT / "data" / "raw" / "penmanshiel" / "Penmanshiel_WT_static.csv"
FIG_DIR = ROOT / "figures"

# Default zero-shot split used by the open-source experiment code
# (seed=2026): source turbines are used for training and target turbines
# are held out for zero-shot prediction.
TRAIN_TURBINES = {"T01", "T04", "T10", "T11"}


def wgs84_to_bng(latitude_deg: float, longitude_deg: float) -> tuple[float, float]:
    """Convert WGS84 latitude/longitude to British National Grid EPSG:27700."""
    lat = math.radians(latitude_deg)
    lon = math.radians(longitude_deg)

    # WGS84 geocentric coordinates.
    a_wgs = 6378137.0
    b_wgs = 6356752.3141
    e2_wgs = 1.0 - (b_wgs * b_wgs) / (a_wgs * a_wgs)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    nu_wgs = a_wgs / math.sqrt(1.0 - e2_wgs * sin_lat * sin_lat)
    x = nu_wgs * cos_lat * math.cos(lon)
    y = nu_wgs * cos_lat * math.sin(lon)
    z = (b_wgs * b_wgs / (a_wgs * a_wgs) * nu_wgs) * sin_lat

    # Helmert transform from WGS84 to OSGB36.
    tx, ty, tz = 446.448, -125.157, 542.060
    scale = -20.4894e-6
    rx = math.radians(0.1502 / 3600.0)
    ry = math.radians(0.2470 / 3600.0)
    rz = math.radians(0.8421 / 3600.0)
    x2 = tx + (1.0 + scale) * x - rz * y + ry * z
    y2 = ty + rz * x + (1.0 + scale) * y - rx * z
    z2 = tz - ry * x + rx * y + (1.0 + scale) * z

    # OSGB36 geodetic coordinates.
    a = 6377563.396
    b = 6356256.909
    e2 = 1.0 - (b * b) / (a * a)
    p = math.sqrt(x2 * x2 + y2 * y2)
    lat2 = math.atan2(z2, p * (1.0 - e2))
    for _ in range(10):
        nu = a / math.sqrt(1.0 - e2 * math.sin(lat2) ** 2)
        lat2 = math.atan2(z2 + e2 * nu * math.sin(lat2), p)
    lon2 = math.atan2(y2, x2)

    # British National Grid Transverse Mercator.
    f0 = 0.9996012717
    lat0 = math.radians(49.0)
    lon0 = math.radians(-2.0)
    e0, n0 = 400000.0, -100000.0
    n = (a - b) / (a + b)
    sin_lat = math.sin(lat2)
    cos_lat = math.cos(lat2)
    tan_lat = math.tan(lat2)
    nu = a * f0 / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    rho = a * f0 * (1.0 - e2) / (1.0 - e2 * sin_lat * sin_lat) ** 1.5
    eta2 = nu / rho - 1.0
    m = b * f0 * (
        (1 + n + 5 / 4 * n**2 + 5 / 4 * n**3) * (lat2 - lat0)
        - (3 * n + 3 * n**2 + 21 / 8 * n**3)
        * math.sin(lat2 - lat0)
        * math.cos(lat2 + lat0)
        + (15 / 8 * n**2 + 15 / 8 * n**3)
        * math.sin(2 * (lat2 - lat0))
        * math.cos(2 * (lat2 + lat0))
        - 35 / 24 * n**3
        * math.sin(3 * (lat2 - lat0))
        * math.cos(3 * (lat2 + lat0))
    )
    i = m + n0
    ii = nu / 2 * sin_lat * cos_lat
    iii = nu / 24 * sin_lat * cos_lat**3 * (5 - tan_lat**2 + 9 * eta2)
    iiia = nu / 720 * sin_lat * cos_lat**5 * (
        61 - 58 * tan_lat**2 + tan_lat**4
    )
    iv = nu * cos_lat
    v = nu / 6 * cos_lat**3 * (nu / rho - tan_lat**2)
    vi = nu / 120 * cos_lat**5 * (
        5 - 18 * tan_lat**2 + tan_lat**4 + 14 * eta2 - 58 * tan_lat**2 * eta2
    )
    d_lon = lon2 - lon0
    easting = e0 + iv * d_lon + v * d_lon**3 + vi * d_lon**5
    northing = i + ii * d_lon**2 + iii * d_lon**4 + iiia * d_lon**6
    return easting, northing


def read_ascii_grid(path: Path) -> tuple[np.ndarray, float, float, float]:
    with path.open(encoding="ascii") as handle:
        headers = {}
        first_data_line = None
        for _ in range(6):
            line = handle.readline()
            parts = line.split()
            if len(parts) != 2 or parts[0].lower() not in {
                "ncols",
                "nrows",
                "xllcorner",
                "yllcorner",
                "cellsize",
                "nodata_value",
            }:
                first_data_line = line
                break
            headers[parts[0].lower()] = float(parts[1])
        if first_data_line is None:
            first_data_line = handle.readline()
        values = np.loadtxt(handle, dtype=float)
        if first_data_line.strip():
            values = np.vstack([np.fromstring(first_data_line, sep=" "), values])

    nrows = int(headers["nrows"])
    ncols = int(headers["ncols"])
    if values.shape != (nrows, ncols):
        raise ValueError(f"Unexpected grid shape in {path}: {values.shape}")
    values[values <= -1.999] = np.nan
    # ASCII Grid starts at the upper-left row. Flip to increasing northing.
    values = values[::-1, :]
    return values, headers["xllcorner"], headers["yllcorner"], headers["cellsize"]


def read_turbines() -> list[dict[str, float | str]]:
    turbines = []
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
                    "metadata_elevation": float(row["Elevation (m)"]),
                }
            )
    return turbines


def assemble_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    grids = []
    for name in ("NT76", "NT86"):
        values, x0, y0, cellsize = read_ascii_grid(TERRAIN_DIR / f"{name}.asc")
        x = x0 + (np.arange(values.shape[1]) + 0.5) * cellsize
        y = y0 + (np.arange(values.shape[0]) + 0.5) * cellsize
        grids.append((values, x, y, cellsize))

    # Both tiles have matching 50 m spacing and adjacent easting ranges.
    left, right = grids
    elevation = np.concatenate([left[0], right[0]], axis=1)
    x = np.concatenate([left[1], right[1]])
    y = left[2]
    return elevation, x, y, left[3]


def hillshade(elevation: np.ndarray, cellsize: float) -> np.ndarray:
    filled = np.nan_to_num(elevation, nan=np.nanmedian(elevation))
    dy, dx = np.gradient(filled, cellsize, cellsize)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(dx * dx + dy * dy))
    aspect = np.arctan2(-dx, dy)
    altitude = math.radians(45.0)
    azimuth = math.radians(315.0)
    shade = (
        np.sin(altitude) * np.sin(slope)
        + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
    )
    return (shade - shade.min()) / (shade.max() - shade.min())


def add_scale_bar(ax, length_m: float = 1000.0) -> None:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0 = xlim[0] + 0.06 * (xlim[1] - xlim[0])
    y0 = ylim[0] + 0.06 * (ylim[1] - ylim[0])
    ax.plot([x0, x0 + length_m], [y0, y0], color="black", lw=2.2, solid_capstyle="butt")
    ax.plot([x0, x0], [y0 - 35, y0 + 35], color="black", lw=1.2)
    ax.plot([x0 + length_m, x0 + length_m], [y0 - 35, y0 + 35], color="black", lw=1.2)
    ax.text(x0 + length_m / 2, y0 + 90, f"{int(length_m / 1000)} km", ha="center", va="bottom")


def main() -> None:
    turbines = read_turbines()
    elevation, x, y, cellsize = assemble_grid()
    tx = np.array([item["easting"] for item in turbines])
    ty = np.array([item["northing"] for item in turbines])
    buffer_m = 2000.0
    xmin, xmax = tx.min() - buffer_m, tx.max() + buffer_m
    ymin, ymax = ty.min() - buffer_m, ty.max() + buffer_m

    xmask = (x >= xmin) & (x <= xmax)
    ymask = (y >= ymin) & (y <= ymax)
    clipped = elevation[np.ix_(ymask, xmask)]
    xc = x[xmask]
    yc = y[ymask]
    if clipped.size == 0 or np.isnan(clipped).all():
        raise RuntimeError("The requested map extent does not overlap the DEM.")

    shade = hillshade(clipped, cellsize)
    X, Y = np.meshgrid(xc, yc)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(7.6, 5.5), constrained_layout=True)
    levels = np.arange(
        math.floor(np.nanmin(clipped) / 20) * 20,
        math.ceil(np.nanmax(clipped) / 20) * 20 + 1,
        20,
    )
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
        shade,
        extent=(xc.min(), xc.max(), yc.min(), yc.max()),
        origin="lower",
        cmap="gray",
        alpha=0.24,
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    ax.contour(X, Y, clipped, levels=levels, colors="#4b3b2a", linewidths=0.34, alpha=0.6)
    train_mask = np.array([item["id"] in TRAIN_TURBINES for item in turbines])
    target_mask = ~train_mask
    ax.scatter(
        tx[target_mask],
        ty[target_mask],
        s=96,
        marker="o",
        c="#0072B2",
        edgecolors="white",
        linewidths=1.4,
        zorder=5,
        label="Zero-shot prediction turbine",
    )
    ax.scatter(
        tx[train_mask],
        ty[train_mask],
        s=118,
        marker="^",
        c="#D55E00",
        edgecolors="white",
        linewidths=1.4,
        zorder=6,
        label="Training turbine",
    )
    for item in turbines:
        ax.text(
            item["easting"] + 85,
            item["northing"] + 55,
            item["id"],
            fontsize=8.2,
            color="#17202A",
            fontweight="bold",
            zorder=7,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
            path_effects=[
                path_effects.withStroke(linewidth=1.8, foreground="white")
            ],
        )

    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Elevation (m)")
    ax.set_xlabel("Easting (m), EPSG:27700")
    ax.set_ylabel("Northing (m), EPSG:27700")
    ax.set_title(
        "Penmanshiel terrain and zero-shot turbine split",
        pad=10,
        fontweight="bold",
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="white", alpha=0.22, linewidth=0.55)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
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
    add_scale_bar(ax)

    png_path = FIG_DIR / "penmanshiel_terrain_map.png"
    svg_path = FIG_DIR / "penmanshiel_terrain_map.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")
    print(
        f"Mapped {len(turbines)} turbines; extent "
        f"{xmin:.0f}:{xmax:.0f} E, {ymin:.0f}:{ymax:.0f} N"
    )
    print(
        f"DEM elevation range in map: "
        f"{np.nanmin(clipped):.1f} to {np.nanmax(clipped):.1f} m"
    )


if __name__ == "__main__":
    main()
