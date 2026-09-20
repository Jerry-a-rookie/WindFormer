from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "figures" / "model_comparison" / "wind_direction_radar"
DATA_DIR = OUTPUT_ROOT / "data"
PNG_DIR = OUTPUT_ROOT / "png"


def build_demo_data() -> list[dict[str, object]]:
    # Simulated circular MAE values in degrees. Lower values are better.
    values = {
        "WindFormer": [4.2, 5.0, 5.9, 7.1],
        "Transformer": [4.8, 5.7, 6.8, 8.2],
        "PatchTST": [5.1, 6.2, 7.2, 8.8],
        "iTransformer": [4.5, 5.5, 6.5, 7.9],
        "TimeMixer": [5.4, 6.4, 7.5, 9.1],
        "DLinear": [6.1, 7.1, 8.2, 9.8],
    }
    rows = []
    horizons = [6, 12, 18, 24]
    for model, errors in values.items():
        for horizon, error in zip(horizons, errors):
            rows.append(
                {
                    "model": model,
                    "horizon": horizon,
                    "circular_mae_deg": error,
                    "data_status": "simulated",
                }
            )
    return rows


def save_data(rows: list[dict[str, object]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CSV_PATH = DATA_DIR / "wind_direction_radar_demo.csv"
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "status": "simulated_demo",
        "metric": "circular_mae_deg",
        "unit": "degree",
        "direction": "lower_is_better",
        "horizons": [6, 12, 18, 24],
        "note": "Replace circular_mae_deg with real evaluation results before manuscript use.",
    }
    (DATA_DIR / "wind_direction_radar_demo_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def plot_radar(rows: list[dict[str, object]]) -> Path:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    horizons = [6, 12, 18, 24]
    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    table = {
        model: [
            float(next(row["circular_mae_deg"] for row in rows if row["model"] == model and row["horizon"] == horizon))
            for horizon in horizons
        ]
        for model in models
    }

    angles = np.linspace(0, 2 * np.pi, len(horizons), endpoint=False).tolist()
    angles += angles[:1]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )
    fig, ax = plt.subplots(figsize=(8.6, 8.0), subplot_kw={"polar": True})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfcfd")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"H={h}" for h in horizons], color="#243447", fontweight="bold")
    ax.set_ylim(0, 12)
    ax.set_yticks([2, 4, 6, 8, 10, 12])
    ax.set_yticklabels([f"{value}°" for value in [2, 4, 6, 8, 10, 12]], color="#64748b", fontsize=9)
    ax.set_rlabel_position(18)
    ax.grid(color="#cbd5e1", linewidth=0.8, alpha=0.8)
    ax.spines["polar"].set_color("#94a3b8")
    ax.spines["polar"].set_linewidth(1.0)

    colors = ["#0f766e", "#2563eb", "#7c3aed", "#ea580c", "#0891b2", "#64748b"]
    for model, color in zip(models, colors):
        values = table[model] + table[model][:1]
        ax.plot(angles, values, color=color, linewidth=2.2, marker="o", markersize=5, label=model)
        ax.fill(angles, values, color=color, alpha=0.035)

    ax.set_title(
        "Simulated Wind-Direction Error Across Forecast Horizons",
        pad=28,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.5,
        0.035,
        "Circular MAE (degrees), lower is better; values are simulated for layout demonstration.",
        ha="center",
        color="#475569",
        fontsize=10,
    )
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.08, 1.08),
        frameon=False,
        labelspacing=0.8,
    )
    for text in legend.get_texts():
        text.set_color("#334155")

    fig.tight_layout(rect=[0.0, 0.07, 0.82, 0.98])
    output_path = PNG_DIR / "wind_direction_radar_demo.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    rows = build_demo_data()
    save_data(rows)
    output_path = plot_radar(rows)
    print(f"saved={output_path}")
    print(f"data={DATA_DIR / 'wind_direction_radar_demo.csv'}")


if __name__ == "__main__":
    main()
