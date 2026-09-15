# Extending the Experiment Framework

## Add a Forecasting Model

Models receive a tensor shaped `[batch, lookback, features]` and must return
`[batch, horizon, features]`. Register a builder in
`src/wind_repro/models.py`:

```python
@register_model("my_model")
def _build_my_model(config: dict, num_features: int) -> nn.Module:
    return MyModel(
        lookback=int(config["data"]["lookback"]),
        horizon=int(config["data"]["horizon"]),
        num_features=num_features,
        **config["model"],
    )
```

The existing training, checkpoint, evaluation, prediction, and report code
does not need to change. Select the implementation with:

```powershell
python scripts/train.py --config configs/my_model.yaml --model my_model
```

Model-specific settings belong under the `model` section of its YAML file.
Keep dataset, split, and training settings in their existing sections so runs
remain comparable.

For a multi-model benchmark, add model-specific overrides under
`benchmark.models` and run:

```powershell
python scripts/benchmark.py --config configs/my_benchmark.yaml
```

The runner trains, evaluates, and reports each registered model sequentially,
then writes `summary.csv`, `summary.json`, and `summary.md` under the benchmark
experiment directory.

## Add a Raw Dataset Format

A raw-data adapter returns a long pandas DataFrame with these columns:

- `timestamp`
- `turbine_id`
- `wind_speed`
- `wind_direction`
- `x`
- `y`

Implement a loader with the signature:

```python
def load_my_format(input_path: Path, data_config: dict) -> pd.DataFrame:
    ...
```

Then add it to `RAW_FORMAT_LOADERS` in `src/wind_repro/data.py` and select it
with `data.format` in YAML. The common preparation stage handles validation,
alignment, limited missing-value interpolation, chronological splitting,
normalization, segment-safe windowing, and NPZ caching.

## Reproducible Runs

Every run directory contains the resolved YAML configuration, best checkpoint,
training history, environment details, data metadata, predictions, metrics,
plots, and report. Use a unique `--run-id` rather than overwriting an existing
experiment when comparing models or seeds.
