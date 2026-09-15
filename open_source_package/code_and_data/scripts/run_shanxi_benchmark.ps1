$ErrorActionPreference = "Stop"

python scripts/prepare_data.py `
  --config configs/final/shanxi_benchmark.yaml

python scripts/benchmark.py `
  --config configs/final/shanxi_benchmark.yaml `
  --models cscd_net itransformer `
  --horizons 6 12 18 `
  --run-prefix cscd-net-shanxi-2023-11-2024-05 `
  --skip-existing
