"""Plot a six-panel 3D hyperparameter sensitivity figure for WindFormer.

The production workflow is CSV-driven:

    python scripts/plot_cscd_hyperparameter_sensitivity.py \
        --data figures/cscd_hyperparameter_sensitivity/data/hyperparameter_results.csv \
        --out-prefix figures/cscd_hyperparameter_sensitivity/final/cscd_hyperparameter_sensitivity

For a layout-only preview, use the explicit demo flag:

    python scripts/plot_cscd_hyperparameter_sensitivity.py --demo

Demo values are illustrative and must not be reported as experimental results.
The CSV stores one row per parameter combination and should contain:

    panel, x_value, y_value, speed_mae, direction_mae, seed, status

The three panel families are:
    A/D: common_phase_d_model x common_phase_routers
    B/E: lookback x common_phase_d_model
    C/F: direction_patch_length x direction_layers
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    ROOT
    / "figures"
    / "cscd_hyperparameter_sensitivity"
    / "data"
    / "hyperparameter_results.csv"
)
DEFAULT_FINAL_PREFIX = (
    ROOT
    / "figures"
    / "cscd_hyperparameter_sensitivity"
    / "final"
    / "cscd_hyperparameter_sensitivity"
)
DEFAULT_PREVIEW_PREFIX = (
    ROOT
    / "figures"
    / "cscd_hyperparameter_sensitivity"
    / "preview"
    / "cscd_hyperparameter_sensitivity_preview"
)


@dataclass(frozen=True)
class PanelSpec:
    panel: str
    key: str
    x_param: str
    y_param: str
    x_label: str
    y_label: str
    x_values: tuple[int, ...]
    y_values: tuple[int, ...]
    title: str


PANEL_SPECS = (
    PanelSpec(
        "A",
        "common_representation",
        "common_phase_d_model",
        "common_phase_routers",
        "d_model",
        "routers",
        (16, 32, 64, 128),
        (1, 2, 4, 8),
        "Common phase representation",
    ),
    PanelSpec(
        "B",
        "lookback_representation",
        "lookback",
        "common_phase_d_model",
        "lookback (steps)",
        "d_model",
        (144, 288, 432, 576),
        (32, 64, 96, 128),
        "History length and representation",
    ),
    PanelSpec(
        "C",
        "direction_patch",
        "direction_patch_length",
        "direction_layers",
        "patch length",
        "layers",
        (6, 12, 24, 36),
        (1, 2, 3, 4),
        "Circular-direction patch encoder",
    ),
)

REQUIRED_COLUMNS = {
    "panel",
    "x_value",
    "y_value",
    "speed_mae",
    "direction_mae",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    data = frame.copy()
    data["panel"] = data["panel"].astype(str).str.upper()
    for column in ("x_value", "y_value", "speed_mae", "direction_mae"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError(
            "CSV contains missing or non-numeric values in required columns. "
            "Fill speed_mae and direction_mae before plotting formal results."
        )

    for spec in PANEL_SPECS:
        subset = data[data["panel"] == spec.panel]
        expected = {
            (float(x_value), float(y_value))
            for x_value in spec.x_values
            for y_value in spec.y_values
        }
        observed = set(zip(subset["x_value"], subset["y_value"]))
        if observed != expected:
            missing_pairs = sorted(expected - observed)
            extra_pairs = sorted(observed - expected)
            raise ValueError(
                f"Panel {spec.panel} does not match its expected grid. "
                f"Missing={missing_pairs}; extra={extra_pairs}"
            )
        if len(subset) != len(expected):
            raise ValueError(f"Panel {spec.panel} contains duplicate parameter pairs.")

    return data


def build_demo_frame() -> pd.DataFrame:
    """Build deterministic layout-preview data, never used by the formal workflow."""
    rows: list[dict[str, float | int | str]] = []
    for spec in PANEL_SPECS:
        for x_value in spec.x_values:
            for y_value in spec.y_values:
                if spec.panel == "A":
                    x_term = abs(np.log2(x_value / 64.0))
                    y_term = abs(np.log2(y_value / 4.0))
                    interaction = 0.004 * x_term * y_term
                elif spec.panel == "B":
                    x_term = abs(x_value - 288) / 288
                    y_term = abs(y_value - 64) / 64
                    interaction = 0.008 * x_term * y_term
                else:
                    x_term = abs(x_value - 12) / 12
                    y_term = abs(y_value - 2) / 2
                    interaction = 0.007 * x_term * y_term

                speed_mae = 0.7874 + 0.040 * x_term + 0.030 * y_term + interaction
                direction_mae = 0.3393 + 0.055 * x_term + 0.045 * y_term + interaction
                rows.append(
                    {
                        "panel": spec.panel,
                        "x_value": x_value,
                        "y_value": y_value,
                        "speed_mae": round(speed_mae, 5),
                        "direction_mae": round(direction_mae, 5),
                        "seed": "demo",
                        "status": "illustrative_demo",
                    }
                )
    return pd.DataFrame(rows)


def metric_matrix(
    data: pd.DataFrame, spec: PanelSpec, metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = data[data["panel"] == spec.panel].copy()
    matrix = (
        subset.pivot(index="y_value", columns="x_value", values=metric)
        .reindex(index=spec.y_values, columns=spec.x_values)
        .to_numpy(dtype=float)
    )
    x_grid, y_grid = np.meshgrid(
        np.arange(len(spec.x_values), dtype=float),
        np.arange(len(spec.y_values), dtype=float),
    )
    return x_grid, y_grid, matrix


def bar_colors(values: np.ndarray, cmap_name: str) -> list[tuple[float, ...]]:
    norm = Normalize(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
    cmap = mpl.colormaps.get_cmap(cmap_name)
    return [cmap(float(norm(value))) for value in values.ravel()]


def draw_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    spec: PanelSpec,
    metric: str,
    cmap_name: str,
    panel_letter: str,
    zlim: tuple[float, float],
) -> None:
    x_grid, y_grid, values = metric_matrix(data, spec, metric)
    flat_values = values.ravel()
    dx, dy = 0.50, 0.50
    dz0 = np.zeros_like(flat_values)
    colors = bar_colors(values, cmap_name)

    ax.bar3d(
        x_grid.ravel() - dx / 2,
        y_grid.ravel() - dy / 2,
        dz0,
        dx,
        dy,
        flat_values,
        color=colors,
        edgecolor=(0.25, 0.25, 0.25, 0.65),
        linewidth=0.35,
        shade=False,
        zsort="min",
    )

    best_index = int(np.nanargmin(values))
    best_y, best_x = np.unravel_index(best_index, values.shape)
    best_value = values[best_y, best_x]
    ax.bar3d(
        [best_x - dx / 2],
        [best_y - dy / 2],
        [0],
        [dx],
        [dy],
        [best_value],
        color="#f6c453",
        edgecolor="#1f2933",
        linewidth=1.25,
        shade=False,
        zsort="min",
    )

    ax.set_xticks(np.arange(len(spec.x_values)))
    ax.set_xticklabels([str(value) for value in spec.x_values])
    ax.set_yticks(np.arange(len(spec.y_values)))
    ax.set_yticklabels([str(value) for value in spec.y_values])
    ax.set_xlabel(spec.x_label, labelpad=2)
    ax.set_ylabel(spec.y_label, labelpad=2)
    ax.set_title(f"{panel_letter}  {spec.title}", loc="left", pad=1.0, fontsize=8.2)
    ax.set_proj_type("ortho")
    ax.view_init(elev=24, azim=-52)
    ax.set_box_aspect((1.12, 0.98, 0.78))
    ax.set_zlim(*zlim)
    ax.set_zticks(np.linspace(0.0, zlim[1], 4))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((0.55, 0.55, 0.55, 0.35))
    ax.tick_params(axis="both", which="major", pad=0, labelsize=6.0)
    ax.tick_params(axis="z", which="major", pad=1, labelsize=5.8)


def save_figure(fig: plt.Figure, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.tiff", dpi=600, bbox_inches="tight")


def write_demo_source(output_prefix: Path, data: pd.DataFrame) -> Path:
    demo_path = output_prefix.with_name(f"{output_prefix.name}_data.csv")
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(demo_path, index=False)
    return demo_path


def plot(data: pd.DataFrame, output_prefix: Path, demo: bool) -> None:
    configure_matplotlib()
    fig = plt.figure(figsize=(13.8, 7.55), constrained_layout=False)
    grid = fig.add_gridspec(
        nrows=2,
        ncols=3,
        left=0.052,
        right=0.992,
        bottom=0.105,
        top=0.905 if demo else 0.925,
        wspace=0.005,
        hspace=0.015,
    )

    metric_specs = (
        ("speed_mae", "Blues"),
        ("direction_mae", "Oranges"),
    )
    for row, (metric, cmap_name) in enumerate(metric_specs):
        metric_values = pd.concat(
            [
                data.loc[data["panel"] == spec.panel, metric]
                for spec in PANEL_SPECS
            ],
            ignore_index=True,
        )
        zlim = (0.0, float(metric_values.max()) * 1.08)
        for column, spec in enumerate(PANEL_SPECS):
            ax = fig.add_subplot(grid[row, column], projection="3d")
            panel_letter = chr(ord("A") + column + row * 3)
            draw_panel(
                ax,
                data,
                spec,
                metric,
                cmap_name,
                panel_letter,
                zlim,
            )

    fig.text(
        0.014,
        0.695,
        "Speed MAE",
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
    )
    fig.text(
        0.014,
        0.305,
        "Direction MAE (rad)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
    )
    fig.text(
        0.985,
        0.018,
        "Gold edge: lowest validation metric in each panel.",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="#6b4f00",
    )
    if demo:
        fig.suptitle(
            "WindFormer hyperparameter sensitivity: layout preview only",
            x=0.5,
            y=0.965,
            fontsize=11,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.035,
            "Illustrative demo values; replace with repeated-seed validation results before manuscript use.",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color="#8a3b12",
        )
    else:
        fig.suptitle(
            "WindFormer hyperparameter sensitivity",
            x=0.5,
            y=0.972,
            fontsize=11,
            fontweight="bold",
        )

    save_figure(fig, output_prefix)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="CSV containing real hyperparameter scan results.",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=None,
        help="Output prefix without extension for SVG/PDF/PNG.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate an explicitly labelled illustrative layout preview.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo:
        data = build_demo_frame()
        output_prefix = args.out_prefix or DEFAULT_PREVIEW_PREFIX
        demo_path = write_demo_source(output_prefix, data)
        print(f"Using illustrative demo data: {demo_path}")
    else:
        if not args.data.exists():
            raise FileNotFoundError(
                f"Real result CSV not found: {args.data}. "
                "Fill the template or rerun with --demo for a layout preview."
            )
        data = validate_frame(pd.read_csv(args.data))
        output_prefix = args.out_prefix or DEFAULT_FINAL_PREFIX
        print(f"Using real experiment data: {args.data}")

    plot(data, output_prefix, demo=args.demo)
    print(f"Saved: {output_prefix}.svg")
    print(f"Saved: {output_prefix}.pdf")
    print(f"Saved: {output_prefix}.png")
    print(f"Saved: {output_prefix}.tiff")


if __name__ == "__main__":
    main()
