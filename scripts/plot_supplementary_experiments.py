from __future__ import annotations

import csv
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"
SUMMARY_PATH = EXPERIMENTS_DIR / "model_comparison_summary.md"
EXTREME_PATH = EXPERIMENTS_DIR / "极端场景.md"
OUTPUT_DIRS = [ROOT / "figures", ROOT / "正式文件"]

MODEL_ORDER = [
    "CSCD-Net",
    "MST-Net",
    "PatchTST",
    "TimeMixer++",
    "GLPT",
    "DLinear",
    "TimesNet",
    "iTransformer",
]
DATASET_ORDER = ["Penmanshiel", "Shanxi"]
HORIZONS = [6, 12, 18]
NOISE_CONDITIONS = ["noise_0.1", "noise_0.2", "noise_0.3"]
FOCUS_DATASET = "Penmanshiel"
FOCUS_MODELS = ["CSCD-Net", "MST-Net", "DLinear", "TimesNet", "iTransformer"]


def display_model_name(model: str) -> str:
    """Return the public model name used in generated figures and tables."""
    return "WindFormer" if model == "CSCD-Net" else model

MODEL_COLORS = {
    "CSCD-Net": "#C76E5E",
    "MST-Net": "#4E79A7",
    "PatchTST": "#8F9F5A",
    "TimeMixer++": "#D49A4A",
    "GLPT": "#7B6FA8",
    "DLinear": "#9A7664",
    "TimesNet": "#5F9E8F",
    "iTransformer": "#7A8797",
}
MODEL_MARKERS = {
    "CSCD-Net": "o",
    "MST-Net": "s",
    "PatchTST": "^",
    "TimeMixer++": "D",
    "GLPT": "P",
    "DLinear": "X",
    "TimesNet": "v",
    "iTransformer": "h",
}
EXTREME_STAGE_LABELS = {
    "Shanxi": ["S1\n0-10", "S2\n10-20", "S3\n20-30", "S4\n30-40", "S5\n40+"],
    "Penmanshiel": ["S1\n0-15", "S2\n15-25", "S3\n25-35", "S4\n35-45", "S5\n45+"],
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "font.weight": "bold",
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.5,
            "axes.labelweight": "bold",
            "axes.linewidth": 1.2,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "grid.color": "#E0E5EC",
            "grid.linestyle": "-",
            "grid.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _clean_cell(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("`", "")
    cleaned = cleaned.replace("\u3000", " ")
    cleaned = cleaned.replace("(Ours)", "")
    cleaned = cleaned.replace("（本文）", "")
    return cleaned.strip()


def _parse_float(text: str) -> float:
    cleaned = _clean_cell(text).replace(",", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not match:
        raise ValueError(f"Could not parse a number from {text!r}")
    return float(match.group(0))


def _normalize_model_name(text: str) -> str:
    cleaned = _clean_cell(text)
    aliases = {
        "ITransformer": "iTransformer",
        "iTransformer": "iTransformer",
        "TimeMixer": "TimeMixer++",
    }
    return aliases.get(cleaned, cleaned)


def _is_separator_row(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{2,}:?", cell or "-") for cell in cells)


def _extract_markdown_table(path: Path, heading_prefix: str) -> tuple[list[str], list[dict[str, str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip().startswith(heading_prefix):
            start = index + 1
            break
    if start is None:
        raise ValueError(f"Could not find heading {heading_prefix!r} in {path}")

    table_lines: list[str] = []
    collecting = False
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("|"):
            table_lines.append(stripped)
            collecting = True
            continue
        if collecting:
            break

    if len(table_lines) < 2:
        raise ValueError(f"No markdown table found after heading {heading_prefix!r} in {path}")

    header = [_clean_cell(cell) for cell in table_lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for raw_line in table_lines[1:]:
        cells = [_clean_cell(cell) for cell in raw_line.strip("|").split("|")]
        if _is_separator_row(cells) or len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells)))
    return header, rows


def _load_detailed_table(
    heading_prefix: str,
) -> dict[int, dict[str, dict[str, float]]]:
    _, rows = _extract_markdown_table(SUMMARY_PATH, heading_prefix)
    values: dict[int, dict[str, dict[str, float]]] = {}
    for row in rows:
        horizon = int(_parse_float(row["Horizon"]))
        condition = _clean_cell(row["Condition"])
        values.setdefault(horizon, {}).setdefault(condition, {})
        for model in MODEL_ORDER:
            values[horizon][condition][model] = _parse_float(row[model])
    return values


def load_summary_tables() -> dict[str, dict[str, dict[int, dict[str, dict[str, float]]]]]:
    heading_map = {
        "full": {
            "Penmanshiel": {
                "speed": "### 4.1 Penmanshiel",
                "direction": "### 4.2 Penmanshiel",
            },
            "Shanxi": {
                "speed": "### 4.3 Shanxi",
                "direction": "### 4.4 Shanxi",
            },
        },
        "zero_shot": {
            "Penmanshiel": {
                "speed": "### 7.1 Penmanshiel",
                "direction": "### 7.2 Penmanshiel",
            },
            "Shanxi": {
                "speed": "### 7.3 Shanxi",
                "direction": "### 7.4 Shanxi",
            },
        },
    }

    values: dict[str, dict[str, dict[int, dict[str, dict[str, float]]]]] = {}
    for experiment, dataset_map in heading_map.items():
        values[experiment] = {}
        for dataset, metric_map in dataset_map.items():
            values[experiment][dataset] = {}
            for metric, heading in metric_map.items():
                values[experiment][dataset][metric] = _load_detailed_table(heading)
    return values


def load_noise_degradation(
    summary_values: dict[str, dict[str, dict[int, dict[str, dict[str, float]]]]]
) -> dict[str, dict[str, dict[str, list[float]]]]:
    values: dict[str, dict[str, dict[str, list[float]]]] = {}
    for dataset in DATASET_ORDER:
        values[dataset] = {}
        for metric in ["speed", "direction"]:
            values[dataset][metric] = {}
            metric_values = summary_values["full"][dataset][metric]
            for model in MODEL_ORDER:
                degradation: list[float] = []
                for condition in NOISE_CONDITIONS:
                    ratios = []
                    for horizon in HORIZONS:
                        normal = metric_values[horizon]["normal"][model]
                        noisy = metric_values[horizon][condition][model]
                        ratios.append((noisy / normal - 1.0) * 100.0)
                    degradation.append(float(np.mean(ratios)))
                values[dataset][metric][model] = degradation
    return values


def load_extreme_tables() -> dict[str, dict[str, dict[int, dict[str, list[float]]]]]:
    metric_tables = {
        "speed": _extract_markdown_table(EXTREME_PATH, "### 表 1"),
        "direction": _extract_markdown_table(EXTREME_PATH, "### 表 2"),
    }

    values: dict[str, dict[str, dict[int, dict[str, list[float]]]]] = {
        "speed": {},
        "direction": {},
    }
    for metric_name, (_, rows) in metric_tables.items():
        last_model = ""
        last_dataset = ""
        for row in rows:
            model = _normalize_model_name(row.get("模型名称", "")) or last_model
            dataset = _clean_cell(row.get("数据集", "")) or last_dataset
            horizon = int(_parse_float(row.get("步长", "")))
            if model:
                last_model = model
            if dataset:
                last_dataset = dataset
            if model not in MODEL_ORDER or dataset not in DATASET_ORDER or horizon not in HORIZONS:
                continue

            values[metric_name].setdefault(dataset, {}).setdefault(horizon, {})[model] = [
                _parse_float(row["阶段1 (基准)"]),
                _parse_float(row["阶段2 (轻微/初跃)"]),
                _parse_float(row["阶段3 (中度/震荡)"]),
                _parse_float(row["阶段4 (重度/二次)"]),
                _parse_float(row["阶段5 (极端恶化)"]),
            ]
    return values


def style_axis(axis: plt.Axes, *, grid_axis: str = "y") -> None:
    axis.grid(axis=grid_axis)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#9CA8B6")
    axis.spines["bottom"].set_color("#9CA8B6")
    axis.tick_params(colors="#26313F", width=1.05, length=3.8)
    for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
        tick_label.set_fontweight("bold")


def save_figure(fig: plt.Figure, stem: str) -> None:
    for output_dir in OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{stem}.png", dpi=360, bbox_inches="tight")
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")


def save_sparse_error_table(
    summary_values: dict[str, dict[str, dict[int, dict[str, dict[str, float]]]]]
) -> None:
    rows: list[dict[str, str | int | float]] = []
    condition_specs = [
        ("full", "normal", "Full-sample"),
        ("zero_shot", "normal", "Zero-shot"),
        ("zero_shot", "noise_0.3", "Zero-shot + sigma=0.3"),
    ]
    for dataset in [FOCUS_DATASET]:
        for metric in ["speed", "direction"]:
            for experiment_key, condition_key, condition_label in condition_specs:
                metric_values = summary_values[experiment_key][dataset][metric]
                for horizon in HORIZONS:
                    for model in FOCUS_MODELS:
                        rows.append(
                            {
                                "dataset": dataset,
                                "metric": metric,
                                "condition": condition_label,
                                "horizon": horizon,
                                "model": display_model_name(model),
                                "mae": metric_values[horizon][condition_key][model],
                            }
                        )

    for output_dir in OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "supplementary_sparse_deployment_zero_shot_values.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["dataset", "metric", "condition", "horizon", "model", "mae"],
            )
            writer.writeheader()
            writer.writerows(rows)


def plot_noise_robustness(
    noise_values: dict[str, dict[str, dict[str, list[float]]]]
) -> plt.Figure:
    noise_labels = [r"$\sigma=0.1$", r"$\sigma=0.2$", r"$\sigma=0.3$"]
    metric_specs = [
        ("speed", "Wind speed", "MAE increase (%)"),
        ("direction", "Wind direction", "MAE increase (%)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.7), dpi=360)
    bar_width = 0.095
    x = np.arange(len(noise_labels), dtype=float)

    for row_index, (metric_key, metric_title, y_label) in enumerate(metric_specs):
        for col_index, dataset in enumerate(DATASET_ORDER):
            axis = axes[row_index, col_index]
            for model_index, model in enumerate(MODEL_ORDER):
                offsets = (model_index - (len(MODEL_ORDER) - 1) / 2.0) * bar_width
                axis.bar(
                    x + offsets,
                    noise_values[dataset][metric_key][model],
                    width=bar_width,
                    color=MODEL_COLORS[model],
                    edgecolor="#FFFFFF",
                    linewidth=1.0,
                    alpha=0.98,
                    label=display_model_name(model),
                )
            axis.set_xticks(x)
            axis.set_xticklabels(noise_labels)
            axis.set_xlabel("Injected Gaussian noise")
            axis.set_ylabel(y_label)
            axis.set_title(
                f"({chr(97 + row_index * 2 + col_index)}) {dataset} - {metric_title}",
                pad=9,
            )
            style_axis(axis)
            upper = max(
                max(noise_values[dataset][metric_key][model]) for model in MODEL_ORDER
            )
            axis.set_ylim(0, upper * 1.18 if upper > 0 else 1)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=8,
        frameon=False,
        handlelength=1.0,
        columnspacing=0.78,
    )
    fig.suptitle("Sensor Perturbation Robustness", y=1.05, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def plot_extreme_volatility(
    extreme_values: dict[str, dict[str, dict[int, dict[str, list[float]]]]]
) -> plt.Figure:
    row_specs = [
        ("speed", FOCUS_DATASET, "Wind speed", "Wind speed MAE (m/s)"),
        ("direction", FOCUS_DATASET, "Wind direction", "Wind direction MAE (rad)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.9), dpi=360)

    for row_index, (metric_key, dataset, metric_title, y_label) in enumerate(row_specs):
        for col_index, horizon in enumerate(HORIZONS):
            axis = axes[row_index, col_index]
            labels = EXTREME_STAGE_LABELS[dataset]
            x = np.arange(len(labels), dtype=float)
            for model in FOCUS_MODELS:
                axis.plot(
                    x,
                    extreme_values[metric_key][dataset][horizon][model],
                    color=MODEL_COLORS[model],
                    linestyle="-",
                    marker=MODEL_MARKERS[model],
                    linewidth=2.65 if model == "CSCD-Net" else 2.2,
                    markersize=6.6,
                    markerfacecolor=MODEL_COLORS[model],
                    markeredgecolor="white",
                    markeredgewidth=0.85,
                    label=display_model_name(model),
                )
            axis.set_xticks(x)
            axis.set_xticklabels(labels)
            axis.set_xlabel("Extreme-event density stage")
            axis.set_ylabel(y_label)
            axis.set_title(
                f"({chr(97 + row_index * 3 + col_index)}) {dataset} - {metric_title}\n{horizon}-step",
                pad=8,
            )
            style_axis(axis)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=5,
        frameon=False,
        handlelength=1.45,
        columnspacing=0.85,
    )
    fig.suptitle("Extreme Volatility Stress Test", y=1.05, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def plot_sparse_deployment(
    summary_values: dict[str, dict[str, dict[int, dict[str, dict[str, float]]]]]
) -> plt.Figure:
    condition_specs = [
        ("full", "normal", "Full-sample"),
        ("zero_shot", "normal", "Zero-shot"),
        ("zero_shot", "noise_0.3", r"Zero-shot + $\sigma=0.3$"),
    ]
    row_specs = [
        (FOCUS_DATASET, "speed", "Wind speed MAE (m/s)", "Wind speed"),
        (FOCUS_DATASET, "direction", "Wind direction MAE (rad)", "Wind direction"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.9), dpi=360)
    x = np.asarray(HORIZONS, dtype=float)
    x_labels = [str(horizon) for horizon in HORIZONS]

    for row_index, (dataset, metric_key, y_label, metric_title) in enumerate(row_specs):
        for col_index, (experiment_key, condition_key, condition_title) in enumerate(condition_specs):
            axis = axes[row_index, col_index]
            metric_values = summary_values[experiment_key][dataset][metric_key]
            for model in FOCUS_MODELS:
                y = [metric_values[horizon][condition_key][model] for horizon in HORIZONS]
                axis.plot(
                    x,
                    y,
                    color=MODEL_COLORS[model],
                    linestyle="-",
                    marker=MODEL_MARKERS[model],
                    linewidth=2.65 if model == "CSCD-Net" else 2.2,
                    markersize=6.8,
                    markerfacecolor=MODEL_COLORS[model],
                    markeredgecolor="white",
                    markeredgewidth=0.85,
                    label=display_model_name(model),
                )
            axis.set_xticks(x)
            axis.set_xticklabels(x_labels)
            axis.set_xlabel("Forecast horizon")
            axis.set_ylabel(y_label)
            axis.set_title(
                f"({chr(97 + row_index * 3 + col_index)}) {dataset} - {metric_title}\n{condition_title}",
                pad=8,
            )
            style_axis(axis)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=5,
        frameon=False,
        handlelength=1.45,
        columnspacing=0.85,
    )
    fig.suptitle(
        "Full-sample and Zero-shot Deployment Evaluation",
        y=1.055,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def main() -> None:
    configure_style()
    summary_values = load_summary_tables()
    noise_values = load_noise_degradation(summary_values)
    extreme_values = load_extreme_tables()

    figures = {
        "supplementary_sensor_perturbation_robustness": plot_noise_robustness(noise_values),
        "supplementary_extreme_volatility_stress_test": plot_extreme_volatility(extreme_values),
        "supplementary_sparse_deployment_zero_shot": plot_sparse_deployment(summary_values),
    }
    save_sparse_error_table(summary_values)

    for stem, fig in figures.items():
        save_figure(fig, stem)
        plt.close(fig)
        print(stem)


if __name__ == "__main__":
    main()
