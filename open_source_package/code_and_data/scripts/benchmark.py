from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import _bootstrap
from wind_repro.config import load_config, project_root
from wind_repro.engine import evaluate_experiment, train_experiment
from wind_repro.reporting import make_report
from wind_repro.utils import read_json, write_json


REPORT_HORIZONS = (6, 12, 18)


def _deep_update(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


def _write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json({"runs": rows}, output_dir / "summary.json")
    if not rows:
        return

    fieldnames = [
        "model",
        "horizon",
        "status",
        "speed_mae",
        "speed_mse",
        "speed_rmse",
        "direction_mae_rad",
        "direction_mse_rad2",
        "direction_rmse_rad",
        "direction_mae_degrees",
        "direction_mse",
        "direction_rmse_degrees",
        "parameters",
        "training_seconds",
        "mean_epoch_seconds",
        "peak_gpu_memory_mb",
        "evaluation_seconds",
        "samples_per_second",
        "best_epoch",
        "run_dir",
        "error",
    ]
    with (output_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Benchmark Summary",
        "",
        "| Model | Horizon | Speed MAE | Speed RMSE | Direction MAE (rad) | "
        "Direction RMSE (rad) | Parameters | Train (s) | "
        "Peak GPU (MB) | Samples/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["status"] == "completed":
            lines.append(
                f"| {row['model']} | {row['horizon']} | "
                f"{row['speed_mae']:.6f} | "
                f"{row['speed_rmse']:.6f} | "
                f"{row['direction_mae_rad']:.6f} | "
                f"{row['direction_rmse_rad']:.6f} | "
                f"{row['parameters']:,} | "
                f"{row['training_seconds']:.2f} | "
                f"{row['peak_gpu_memory_mb']:.1f} | "
                f"{row['samples_per_second']:.1f} |"
            )
        else:
            lines.append(
                f"| {row['model']} | {row['horizon']} | failed | failed | "
                f"failed | failed | "
                f"- | - | - | - |"
            )
    (output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument(
        "--horizons",
        nargs="*",
        type=int,
        default=None,
        help="Independent forecast horizons to train and evaluate.",
    )
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    base_config = load_config(args.config)
    benchmark_config = base_config.get("benchmark", {})
    registered_runs = benchmark_config.get("models", {})
    models = args.models or list(registered_runs)
    if not models:
        raise ValueError("No benchmark models were configured or requested.")

    prefix = args.run_prefix or str(
        benchmark_config.get(
            "run_prefix", base_config["project"]["name"]
        )
    )
    benchmark_dir = project_root() / "experiments" / prefix
    rows: list[dict[str, Any]] = []

    configured_horizons = benchmark_config.get(
        "horizons", list(REPORT_HORIZONS)
    )
    horizons = sorted(
        set(
            args.horizons
            if args.horizons is not None
            else configured_horizons
        )
    )
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("All benchmark horizons must be positive integers.")

    for horizon in horizons:
        for model_name in models:
            if model_name not in registered_runs:
                raise ValueError(
                    f"Model '{model_name}' has no benchmark override."
                )
            run_path = project_root() / "experiments" / (
                f"{prefix}-h{horizon}-{model_name}"
            )
            if run_path.exists() and not args.skip_existing:
                raise FileExistsError(
                    f"Run directory already exists: {run_path}"
                )

            if (
                args.skip_existing
                and (run_path / "metrics.json").exists()
                and (run_path / "training_summary.json").exists()
            ):
                print(f"skipping_completed=h{horizon}-{model_name}")
            else:
                config = deepcopy(base_config)
                config.pop("benchmark", None)
                _deep_update(config, registered_runs[model_name])
                config["model"]["name"] = model_name
                config["project"]["name"] = f"{prefix}-h{horizon}-{model_name}"
                config["data"]["horizon"] = horizon
                config.setdefault("evaluation", {})["report_horizons"] = [
                    horizon
                ]
                print(
                    f"benchmark_model={model_name} horizon={horizon} "
                    f"run={run_path}"
                )
                try:
                    train_experiment(config, model_name, run_path)
                    evaluate_experiment(run_path)
                    make_report(run_path)
                except Exception as error:
                    failed_row = {
                        "model": model_name,
                        "horizon": horizon,
                        "status": "failed",
                        "speed_mae": "",
                        "speed_mse": "",
                        "speed_rmse": "",
                        "direction_mae_rad": "",
                        "direction_mse_rad2": "",
                        "direction_rmse_rad": "",
                        "direction_mae_degrees": "",
                        "direction_mse": "",
                        "direction_rmse_degrees": "",
                        "parameters": "",
                        "training_seconds": "",
                        "mean_epoch_seconds": "",
                        "peak_gpu_memory_mb": "",
                        "evaluation_seconds": "",
                        "samples_per_second": "",
                        "best_epoch": "",
                        "run_dir": str(run_path),
                        "error": repr(error),
                    }
                    rows.append(failed_row)
                    _write_summary(rows, benchmark_dir)
                    print(
                        f"benchmark_failed={model_name} horizon={horizon} "
                        f"error={error!r}"
                    )
                    continue

            metrics = read_json(run_path / "metrics.json")
            summary = read_json(run_path / "training_summary.json")
            rows.append(
                {
                    "model": model_name,
                    "horizon": horizon,
                    "status": "completed",
                    "speed_mae": metrics["speed"]["mae"],
                    "speed_mse": metrics["speed"]["mse"],
                    "speed_rmse": metrics["speed"]["rmse"],
                    "direction_mae_rad": metrics["direction"].get(
                        "mae_rad", metrics["direction"]["mae"]
                    ),
                    "direction_mse_rad2": metrics["direction"].get(
                        "mse_rad2", metrics["direction"]["mse"]
                    ),
                    "direction_rmse_rad": metrics["direction"].get(
                        "rmse_rad", metrics["direction"]["rmse"]
                    ),
                    "direction_mae_degrees": metrics["direction"][
                        "mae_degrees"
                    ],
                    "direction_mse": metrics["direction"]["mse"],
                    "direction_rmse_degrees": metrics["direction"][
                        "rmse_degrees"
                    ],
                    "parameters": summary["parameters"],
                    "training_seconds": summary["training_seconds"],
                    "mean_epoch_seconds": summary["mean_epoch_seconds"],
                    "peak_gpu_memory_mb": summary["peak_gpu_memory_mb"],
                    "evaluation_seconds": metrics["run"][
                        "evaluation_seconds"
                    ],
                    "samples_per_second": metrics["run"][
                        "samples_per_second"
                    ],
                    "best_epoch": metrics["run"]["checkpoint_epoch"],
                    "run_dir": str(run_path),
                    "error": "",
                }
            )
            _write_summary(rows, benchmark_dir)
            print(
                f"benchmark_completed={model_name} horizon={horizon}"
            )


if __name__ == "__main__":
    main()
