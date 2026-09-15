from __future__ import annotations

import argparse
import sys

import benchmark


DEFAULT_MODELS = [
    "patchtst",
    "itransformer",
    "timemixer",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the non-paper model comparison benchmark."
    )
    parser.add_argument(
        "--config",
        default="configs/final/shanxi_benchmark.yaml",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
    )
    parser.add_argument(
        "--horizons",
        nargs="*",
        type=int,
        default=None,
    )
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    forwarded = [
        "--config",
        args.config,
        "--models",
        *args.models,
    ]
    if args.horizons is not None:
        forwarded.extend(["--horizons", *[str(item) for item in args.horizons]])
    if args.run_prefix is not None:
        forwarded.extend(["--run-prefix", args.run_prefix])
    if args.skip_existing:
        forwarded.append("--skip-existing")
    sys.argv = [sys.argv[0], *forwarded]
    benchmark.main()


if __name__ == "__main__":
    main()
