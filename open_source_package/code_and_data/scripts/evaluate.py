from __future__ import annotations

import argparse

import _bootstrap
from wind_repro.engine import evaluate_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    metrics = evaluate_experiment(args.run)
    print(
        f"speed_mae={metrics['speed']['mae']:.6f} "
        f"speed_mse={metrics['speed']['mse']:.6f} "
        f"direction_mae={metrics['direction']['mae']:.6f} "
        f"direction_mse={metrics['direction']['mse']:.6f} "
        f"direction_rmse={metrics['direction']['rmse']:.6f}"
    )
    for horizon, horizon_metrics in metrics.get("horizons", {}).items():
        print(
            f"horizon={horizon} "
            f"speed_mae={horizon_metrics['speed']['mae']:.6f} "
            f"speed_rmse={horizon_metrics['speed']['rmse']:.6f} "
            f"direction_mae="
            f"{horizon_metrics['direction']['mae']:.6f} "
            f"direction_mse="
            f"{horizon_metrics['direction']['mse']:.6f} "
            f"direction_rmse="
            f"{horizon_metrics['direction']['rmse']:.6f}"
        )


if __name__ == "__main__":
    main()
