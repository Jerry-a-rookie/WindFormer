from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import _bootstrap
from wind_repro.config import load_config, resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]
    synthetic = data_config.get("synthetic", {})
    timesteps = int(synthetic.get("timesteps", 480))
    turbines = int(synthetic.get("turbines", 4))
    interval = int(data_config.get("interval_minutes", 10))
    seed = int(config["project"].get("seed", 7))
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range("2025-01-01", periods=timesteps, freq=f"{interval}min")
    rows = []
    time = np.arange(timesteps)
    for turbine in range(turbines):
        phase = turbine * 0.35
        daily = np.sin(2 * np.pi * time / (24 * 60 / interval) + phase)
        short = 0.4 * np.sin(2 * np.pi * time / 18 + phase)
        speed = 8.0 + 2.0 * daily + short + rng.normal(0, 0.25, timesteps)
        direction = (
            np.pi
            + 0.6 * np.sin(2 * np.pi * time / (24 * 60 / interval) + phase)
            + rng.normal(0, 0.05, timesteps)
        ) % (2 * np.pi)
        for index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "timestamp": timestamp,
                    "turbine_id": f"T{turbine + 1:02d}",
                    "wind_speed": speed[index],
                    "wind_direction": direction[index],
                    "x": float(turbine % 2) * 500.0,
                    "y": float(turbine // 2) * 500.0,
                }
            )

    output = resolve_project_path(data_config["input_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"generated={output} rows={len(rows)}")


if __name__ == "__main__":
    main()
