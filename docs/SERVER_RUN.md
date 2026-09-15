# Server Run Guide

This guide uses the final extended Penmanshiel protocol:

- Data: complete available 2016-2022 timeline reconstructed at 10-minute
  intervals.
- Turbines: WT01, WT02, WT04-WT15, 14 turbines in total.
- Sampling: 10 minutes.
- Input: 288 historical steps.
- Independent forecast experiments: 6, 12, 18, and 24 future steps,
  corresponding to 1, 2, 3, and 4 hours at the 10-minute sampling interval.
- Split: chronological 70/10/20.
- Quality processing: invalid values are linearly interpolated without row
  deletion; full-sequence FFT is applied before window construction to both
  historical inputs and future targets.
- Direction: scalar angle for the standard baselines; the common-residual
  graph model uses circular direction evaluation and vector-consistency loss.
- Training: batch size 128 and learning rate 1e-4.

## Prepare the Cache

Copy the complete `data/raw/penmanshiel` directory, including all 2016-2022
ZIP archives and `Penmanshiel_WT_static.csv`, to the server. Then run:

```powershell
python scripts/prepare_data.py --config configs/final/penmanshiel_extended_benchmark.yaml
```

The prepared cache is written to (the same normalized sequence cache is reused
by all four horizon experiments):

```text
data/processed/penmanshiel_2016_2022_h24_full_interp_fft.npz
```

## Run One Model

For the common-residual graph model:

```powershell
python scripts/train.py --config configs/final/penmanshiel_extended_common_residual_graph.yaml --model common_residual_graph --run-id extended-common-residual-graph
python scripts/evaluate.py --run experiments/extended-common-residual-graph
python scripts/make_report.py --run experiments/extended-common-residual-graph
```

For the paper-equivalent implementation:

```powershell
python scripts/train.py --config configs/final/penmanshiel_extended_mstnet.yaml --model mstnet --run-id extended-mstnet
python scripts/evaluate.py --run experiments/extended-mstnet
python scripts/make_report.py --run experiments/extended-mstnet
```

## Run the Completed Benchmark

The legacy benchmark contains DLinear, TCN, Transformer, EMA-Frequency-Graph,
MST-Net, and Common-Residual-Graph. The final non-paper comparison is
documented below and excludes the original-paper MST-Net implementation.

```powershell
python scripts/benchmark.py --config configs/final/penmanshiel_extended_benchmark.yaml --skip-existing
```

Results are written under:

```text
experiments/penmanshiel-extended-2016-2022-h24-full-interp-fft/
```

Each run is stored separately, for example:

```text
experiments/penmanshiel-extended-2016-2022-h24-full-interp-fft-h6-transformer/
experiments/penmanshiel-extended-2016-2022-h24-full-interp-fft-h12-transformer/
experiments/penmanshiel-extended-2016-2022-h24-full-interp-fft-h18-transformer/
experiments/penmanshiel-extended-2016-2022-h24-full-interp-fft-h24-transformer/
```

Use `--models transformer` to run one model at all four horizons, or combine
it with `--horizons 6 12` to run selected horizons only.

## Run the Non-paper Model Comparison

Use the patch-attention configuration to compare the conventional baselines
and the new model family:

```powershell
python scripts/compare_models.py `
  --config configs/final/penmanshiel_patch_attention_benchmark.yaml `
  --models dlinear tcn transformer itransformer patchtst `
  patch_linear patch_linear_gated patch_gated patch_standard `
  patch_talking_head patch_ssm `
  --horizons 6 12 18 24
```

## Server Checks

Before a long run:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -m pytest -q
```

Each run saves its resolved configuration, checkpoint, metrics, timing,
environment, predictions, plots, and Markdown report. The current benchmark
uses one seed and is suitable for engineering comparison. The final paper
should repeat the primary models with at least three, preferably five, seeds.
