$ErrorActionPreference = "Stop"

python scripts/prepare_data.py `
  --config configs/final/shanxi_benchmark.yaml

python scripts/benchmark.py `
  --config configs/final/shanxi_benchmark.yaml `
  --models windformer patchtst itransformer timemixer `
  --horizons 6 12 18 24 `
  --run-prefix windformer-shanxi-2023-11-2024-05 `
  --skip-existing
