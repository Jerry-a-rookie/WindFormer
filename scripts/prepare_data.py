from __future__ import annotations

import argparse

import _bootstrap
from wind_repro.config import load_config
from wind_repro.data import load_prepared, prepare_arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    output = prepare_arrays(config)
    metadata = load_prepared(output)["metadata"]
    print(f"prepared={output}")
    print(
        f"rows={metadata['rows']} turbines={metadata['num_turbines']} "
        f"splits={metadata['split_lengths']}"
    )
    if "segments" in metadata:
        print(f"segments={metadata['segments']}")


if __name__ == "__main__":
    main()
