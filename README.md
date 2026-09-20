# WindFormer: Structure-Aware Wind Resource Forecasting

<p align="center">
  <img src="assets/architecture.png" width="820" alt="WindFormer architecture">
</p>

<p align="center"><em>Shared wind-speed dynamics meet circular wind-direction geometry.</em></p>

> Reproducibility code, experiment protocols, and manuscript assets for
> short-term multi-turbine wind resource forecasting.

<p align="center">
  <img src="assets/phase_representation.png" width="820" alt="Patch and phase representations">
</p>

This repository contains the reproducibility code and manuscript assets for
the WindFormer wind resource forecasting study. The current benchmark compares:

- WindFormer
- PatchTST
- iTransformer
- TimeMixer

All models use the same data split, loss, batch size, and evaluation
interface. WindFormer uses a phase-aware common-speed branch, a shared
turbine-residual branch, circular wind-direction representation, and no
geographic graph correction in the current final configuration.

## Method at a glance

| Branch | Representation | Main operation | Role |
| --- | --- | --- | --- |
| Wind speed | Common field + turbine residuals | Phase-Attention | Capture shared evolution and recover local deviations |
| Wind direction | 2-D sine/cosine unit vectors | Shared Patch-Attention | Preserve circular continuity and turbine-specific patterns |
| Joint objective | Speed, direction-vector and wind-vector losses | Multi-objective training | Couple physical quantities without merging their representations |

<p align="center">
  <img src="assets/frequency_wavelet_evidence.png" width="820" alt="Wind-speed frequency and wavelet evidence">
</p>

<p align="center"><em>Representative turbines exhibit shared multi-scale frequency and time-frequency structure.</em></p>

## Project map

| Path | Contents |
| --- | --- |
| `submission_package/` | Latest manuscript PDF, LaTeX source, figures and supplementary material |
| `open_source_package/code_and_data/` | Minimal reproducibility code, configurations, scripts and public-data manifest |
| `configs/`, `src/`, `scripts/` | Full development pipeline and experiment utilities |
| `tests/`, `docs/` | Smoke tests and reproducibility notes |

Legacy manuscript conversions, local experiment outputs, downloaded datasets,
and intermediate figure arrays are intentionally kept out of the public Git
history. The two directories above are the release boundary: one for the
formal submission and one for the reproducibility code.

## Dataset

The open-source release under `open_source_package/code_and_data` does not
include local copies of the datasets. The experiments use public datasets;
the corresponding raw files must be obtained from their original sources
before running the pipeline. If the surrounding working directory still
contains a `data/` directory, treat it as local experimental data and do not
redistribute it with the code release.

### Shanxi Wind Turbines Dataset

The Shanxi benchmark uses the public **Shanxi Wind Turbines Dataset**:

- GitHub: <https://github.com/lou-yimin/Shanxi-Wind-Turbines-Dataset>
- Expected local directory:

```text
data/raw/Shanxi/
```

The loader reads the wide-table columns `ts`,
`wind_speed_0 ... wind_speed_23`, and
`wind_direction_0 ... wind_direction_23`.

The benchmark configuration covers 24 turbines, 10-minute observations from
2023-11-01 to 2024-05-31. Please follow the original repository's license,
terms of use, and data-download instructions.

### Penmanshiel Wind Farm Dataset

The extended experiments use the public Penmanshiel wind-farm SCADA dataset
(14 turbines, 10-minute observations). The raw files are not redistributed
here; obtain them from the original public Zenodo record and place them
under the path expected by the selected Penmanshiel configuration:

- Zenodo record: <https://doi.org/10.5281/zenodo.5946808>
- Expected local directory: `data/raw/penmanshiel/`

The dataset description and citation metadata are recorded in
`data_manifest.yaml` and the manuscript's `references.bib`.

### Data and citation policy

This project provides code, configurations, and experiment protocols only.
It does not claim ownership of either dataset. When using the benchmark,
please cite the original dataset or paper and retain the original attribution
and license notice. Dataset checksums are intentionally not provided because
the open-source release does not redistribute the raw data files.

The current preprocessing protocol is:

- 10-minute sampling;
- 288 historical steps (48 hours);
- linear interpolation for missing/invalid values;
- circular wind-direction representation using sine and cosine;
- chronological 70%/10%/20% split;
- no geographic or graph correction;
- four independent forecast experiments: 6, 12, 18, and 24 steps.

## Run

Prepare the shared cache:

```powershell
python scripts/prepare_data.py --config configs/final/shanxi_benchmark.yaml
```

Run only WindFormer at one horizon:

```powershell
python scripts/train.py `
  --config configs/final/shanxi_benchmark.yaml `
  --model windformer `
  --run-id shanxi-windformer-h6
python scripts/evaluate.py --run experiments/shanxi-windformer-h6
```

Run all 16 experiments:

```powershell
python scripts/benchmark.py `
  --config configs/final/shanxi_benchmark.yaml `
  --models windformer patchtst itransformer timemixer `
  --horizons 6 12 18 24
```

The same workflow can be started with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_shanxi_benchmark.ps1
```

Results are written under:

```text
experiments/windformer-shanxi-2023-11-2024-05/
```

Each model/horizon run contains its resolved configuration, checkpoint,
training history, predictions, metrics, and report.

## Tests

```powershell
python -m pytest -q
```

The model registry and training engine are shared, so adding another model
only requires a model class and a registered builder in
`src/wind_repro/models.py`.
