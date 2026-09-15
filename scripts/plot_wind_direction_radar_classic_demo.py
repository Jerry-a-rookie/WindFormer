from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "figures" / "model_comparison" / "wind_direction_radar"
DATA_DIR = OUTPUT_ROOT / "data"
PNG_DIR = OUTPUT_ROOT / "png"


def build_demo_data() -> list[dict[str, object]]:
    # Schematic unwrapped direction coordinates for visual explanation.
    true_direction = np.array([120.0, 135.0, 150.0, 165.0, 180.0, 195.0, 210.0, 225.0])
    signed_offset = np.array([0.20, -0.40, 0.60, -0.30, 0.50, -0.50, 0.45, -0.41])
    predicted_direction = true_direction + signed_offset
    mae = float(np.mean(np.abs(predicted_direction - true_direction)))

    rows = []
    for index, (true_value, predicted_value) in enumerate(
        zip(true_direction, predicted_direction), start=1
    ):
        rows.append(
            {
                "coordinate": f"t{index}",
                "true_direction_deg": round(float(true_value), 4),
                "predicted_direction_deg": round(float(predicted_value), 4),
                "absolute_error_deg": round(abs(float(predicted_value - true_value)), 4),
                "data_status": "simulated",
            }
        )
    rows.append(
        {
            "coordinate": "MAE",
            "true_direction_deg": "",
            "predicted_direction_deg": "",
            "absolute_error_deg": round(mae, 4),
            "data_status": "simulated_summary",
        }
    )
    return rows


def save_data(rows: list[dict[str, object]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "wind_direction_radar_classic_demo.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_radar(rows: list[dict[str, object]]) -> Path:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    plot_rows = rows[:-1]
    labels = [str(row["coordinate"]) for row in plot_rows]
    true_values = [float(row["true_direction_deg"]) for row in plot_rows]
    predicted_values = [float(row["predicted_direction_deg"]) for row in plot_rows]
    mae = float(rows[-1]["absolute_error_deg"])

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    true_values += true_values[:1]
    predicted_values += predicted_values[:1]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 18,
            "legend.fontsize": 12,
        }
    )
    fig, ax = plt.subplots(figsize=(8.4, 8.0), subplot_kw={"polar": True})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#ffffff")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, fontweight="bold", color="#1f2937")

    # A zoomed radial range makes the small 0.42-degree deviation visible.
    ax.set_ylim(115, 230)
    ax.set_yticks([120, 150, 180, 210, 225])
    ax.set_yticklabels(["120°", "150°", "180°", "210°", "225°"], color="#64748b", fontsize=9)
    ax.set_rlabel_position(18)
    ax.grid(color="#cbd5e1", linewidth=0.9, alpha=0.9)
    ax.spines["polar"].set_color("#94a3b8")
    ax.spines["polar"].set_linewidth(1.0)

    ax.plot(
        angles,
        true_values,
        color="#dc2626",
        linewidth=2.8,
        marker="o",
        markersize=6,
        label="True direction",
        zorder=4,
    )
    ax.plot(
        angles,
        predicted_values,
        color="#2563eb",
        linewidth=2.8,
        marker="o",
        markersize=6,
        label="Prediction (MAE = 0.42°)",
        zorder=5,
    )
    ax.fill(angles, true_values, color="#dc2626", alpha=0.045, zorder=1)
    ax.fill(angles, predicted_values, color="#2563eb", alpha=0.045, zorder=2)

    ax.set_title(
        "Wind-Direction Prediction Deviation",
        pad=28,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.5,
        0.035,
        "Schematic example: red = ground truth, blue = prediction; radial axis is locally zoomed.",
        ha="center",
        color="#475569",
        fontsize=10,
    )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1.08),
        frameon=False,
        labelspacing=1.0,
    )

    fig.tight_layout(rect=[0.0, 0.07, 0.84, 0.98])
    output_path = PNG_DIR / "wind_direction_radar_classic_demo.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    rows = build_demo_data()
    save_data(rows)
    output_path = plot_radar(rows)
    print(f"saved={output_path}")
    print(f"mae={rows[-1]['absolute_error_deg']} deg")


if __name__ == "__main__":
    main()
