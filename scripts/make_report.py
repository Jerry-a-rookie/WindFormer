from __future__ import annotations

import argparse

import _bootstrap
from wind_repro.reporting import make_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    output = make_report(args.run)
    print(f"report={output}")


if __name__ == "__main__":
    main()
