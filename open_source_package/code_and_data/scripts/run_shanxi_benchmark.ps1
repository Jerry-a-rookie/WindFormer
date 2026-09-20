$ErrorActionPreference = "Stop"

python scripts/prepare_data.py `
  --config configs/final/shanxi_benchmark.yaml

python scripts/benchmark.py `
  --config configs/final/shanxi_benchmark.yaml `
  --models windformer itransformer `
  --horizons 6 12 18 `
  --run-prefix windformer-shanxi-2023-11-2024-05 `
  --skip-existing
