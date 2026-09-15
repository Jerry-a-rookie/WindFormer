from __future__ import annotations

import argparse

import _bootstrap
from wind_repro.config import load_config, project_root
from wind_repro.engine import train_experiment
from wind_repro.utils import make_run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="dlinear")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config["model"]["name"] = args.model
    run_id = args.run_id or make_run_id(args.model)
    run_path = project_root() / "experiments" / run_id
    result = train_experiment(config, args.model, run_path)
    print(f"run={result}")


if __name__ == "__main__":
    main()
