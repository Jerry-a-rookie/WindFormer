from __future__ import annotations

import sys

import torch

import _bootstrap
from wind_repro.config import load_config
from wind_repro.data import load_prepared
from wind_repro.models import build_model


def main() -> None:
    config = load_config("configs/final/shanxi_benchmark.yaml")
    prepared = load_prepared(config["data"]["prepared_path"])
    if not torch.cuda.is_available():
        raise RuntimeError("The model smoke test requires a CUDA device.")

    device = torch.device("cuda")
    num_features = 3 * int(prepared["metadata"]["num_turbines"])
    features = torch.randn(
        2,
        int(config["data"]["lookback"]),
        num_features,
        device=device,
    )
    print(f"device={torch.cuda.get_device_name(0)} features={num_features}")

    with torch.no_grad():
        for horizon in (6, 24):
            config["data"]["horizon"] = horizon
            for model_name in (
                "phase_net",
                "patchtst",
                "itransformer",
                "timemixer",
            ):
                model = build_model(
                    model_name,
                    config,
                    num_features,
                    prepared,
                ).to(device)
                torch.cuda.synchronize()
                output = model(features)
                torch.cuda.synchronize()
                parameters = sum(
                    parameter.numel() for parameter in model.parameters()
                )
                print(
                    f"model={model_name} horizon={horizon} "
                    f"shape={tuple(output.shape)} parameters={parameters}"
                )


if __name__ == "__main__":
    main()
