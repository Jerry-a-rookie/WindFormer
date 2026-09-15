from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_speed(path: Path) -> tuple[np.ndarray, dict]:
    data = np.load(path, allow_pickle=True)
    metadata = json.loads(str(data["metadata"]))
    speed_mean = data["speed_mean"]
    speed_std = data["speed_std"]
    speed = np.concatenate(
        [
            data["train_speed"] * speed_std + speed_mean,
            data["validation_speed"] * speed_std + speed_mean,
            data["test_speed"] * speed_std + speed_mean,
        ],
        axis=0,
    )
    return speed.astype(np.float64, copy=False), metadata


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    series = pd.Series(values)
    return (
        series.rolling(window=window, center=True, min_periods=max(2, window // 4))
        .mean()
        .bfill()
        .ffill()
        .to_numpy(dtype=np.float64)
    )


def _autocorr(values: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= len(values):
        return float("nan")
    centered = values - values.mean()
    left = centered[:-lag]
    right = centered[lag:]
    denom = np.sqrt(np.dot(left, left) * np.dot(right, right))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(left, right) / denom)


def _periodogram(values: np.ndarray, interval_minutes: int) -> pd.DataFrame:
    centered = values - values.mean()
    spectrum = np.fft.rfft(centered)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(len(centered), d=interval_minutes / 60.0)
    valid = frequencies > 0
    periods_hours = 1.0 / frequencies[valid]
    power = power[valid]
    order = np.argsort(power)[::-1]
    rows = []
    total_power = float(power.sum())
    for index in order[:12]:
        rows.append(
            {
                "period_hours": float(periods_hours[index]),
                "period_days": float(periods_hours[index] / 24.0),
                "power_share": float(power[index] / max(total_power, 1e-12)),
            }
        )
    return pd.DataFrame(rows)


def _save_figures(
    output_dir: Path,
    common: np.ndarray,
    residual: np.ndarray,
    interval_minutes: int,
    turbines: list[str],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    samples_per_day = int(round(24 * 60 / interval_minutes))
    common_daily = np.array(
        [common[i::samples_per_day].mean() for i in range(samples_per_day)]
    )
    residual_std = residual.std(axis=0)

    fig, axis = plt.subplots(figsize=(9, 3.2), dpi=160)
    days = np.arange(min(len(common), samples_per_day * 21)) / samples_per_day
    axis.plot(days, common[: len(days)], linewidth=0.8, label="common field")
    axis.plot(
        days,
        _moving_average(common[: len(days)], samples_per_day),
        linewidth=1.4,
        label="1-day rolling mean",
    )
    axis.set_xlabel("Day")
    axis.set_ylabel("Wind speed (m/s)")
    axis.set_title("Common field, first 21 valid days")
    axis.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "common_field_first_21_days.png"
    fig.savefig(path)
    plt.close(fig)
    paths["common_21_days"] = path

    fig, axis = plt.subplots(figsize=(8, 3.2), dpi=160)
    hours = np.arange(samples_per_day) * interval_minutes / 60.0
    axis.plot(hours, common_daily, linewidth=1.4)
    axis.set_xlabel("Hour of day")
    axis.set_ylabel("Mean wind speed (m/s)")
    axis.set_title("Average daily profile of common field")
    fig.tight_layout()
    path = output_dir / "common_field_daily_profile.png"
    fig.savefig(path)
    plt.close(fig)
    paths["daily_profile"] = path

    fig, axis = plt.subplots(figsize=(8, 3.2), dpi=160)
    axis.bar(turbines, residual_std)
    axis.set_xlabel("Turbine")
    axis.set_ylabel("Residual std (m/s)")
    axis.set_title("Turbine residual magnitude")
    axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    path = output_dir / "residual_std_by_turbine.png"
    fig.savefig(path)
    plt.close(fig)
    paths["residual_std"] = path

    return paths


def run(prepared_path: Path, output_path: Path) -> None:
    speed, metadata = _load_speed(prepared_path)
    interval_minutes = int(metadata.get("interval_minutes", 10))
    turbines = list(metadata["turbines"])
    samples_per_day = int(round(24 * 60 / interval_minutes))
    samples_per_week = 7 * samples_per_day
    lookback = int(metadata.get("lookback", 288))

    common = speed.mean(axis=1)
    residual = speed - common[:, None]

    raw_entry_var = float(speed.var())
    common_var = float(common.var())
    residual_entry_var = float(residual.var())
    common_variance_share = common_var / max(raw_entry_var, 1e-12)
    residual_variance_share = residual_entry_var / max(raw_entry_var, 1e-12)

    raw_turbine_std = speed.std(axis=0)
    residual_turbine_std = residual.std(axis=0)
    residual_abs_mean = np.abs(residual).mean(axis=0)
    residual_ratio = residual_turbine_std / np.maximum(raw_turbine_std, 1e-12)

    raw_corr = pd.DataFrame(speed, columns=turbines).corr().to_numpy()
    residual_corr = pd.DataFrame(residual, columns=turbines).corr().to_numpy()
    mask = ~np.eye(len(turbines), dtype=bool)
    raw_mean_corr = float(np.nanmean(raw_corr[mask]))
    residual_mean_corr = float(np.nanmean(residual_corr[mask]))

    trend_48h = _moving_average(common, lookback)
    trend_7d = _moving_average(common, samples_per_week)
    trend_48h_share = float(trend_48h.var() / max(common_var, 1e-12))
    trend_7d_share = float(trend_7d.var() / max(common_var, 1e-12))

    daily_profile = np.array(
        [common[i::samples_per_day].mean() for i in range(samples_per_day)]
    )
    daily_profile_full = np.resize(daily_profile, len(common))
    daily_r2 = float(
        1.0 - np.var(common - daily_profile_full) / max(common_var, 1e-12)
    )
    daily_amp = float(daily_profile.max() - daily_profile.min())

    autocorr_rows = []
    for label, lag in [
        ("1 hour", 6),
        ("6 hours", 36),
        ("24 hours", samples_per_day),
        ("48 hours", 2 * samples_per_day),
        ("7 days", samples_per_week),
    ]:
        autocorr_rows.append({"lag": label, "autocorr": _autocorr(common, lag)})

    spectrum = _periodogram(common, interval_minutes)
    residual_table = pd.DataFrame(
        {
            "turbine": turbines,
            "raw_std_ms": raw_turbine_std,
            "residual_std_ms": residual_turbine_std,
            "residual_abs_mean_ms": residual_abs_mean,
            "residual_std_to_raw_std": residual_ratio,
        }
    )

    figure_dir = output_path.parent / "figures" / output_path.stem
    figures = _save_figures(figure_dir, common, residual, interval_minutes, turbines)
    residual_csv = output_path.with_suffix(".residual_by_turbine.csv")
    spectrum_csv = output_path.with_suffix(".spectrum_top_periods.csv")
    residual_table.to_csv(residual_csv, index=False)
    spectrum.to_csv(spectrum_csv, index=False)

    split_ranges = metadata.get("split_ranges", {})
    report = f"""# Common-Residual Diagnostic

Prepared data: `{prepared_path.as_posix()}`

Time span: `{metadata.get("timestamp_start")}` to `{metadata.get("timestamp_end")}`

Rows: `{metadata.get("rows")}`; turbines: `{len(turbines)}`; interval: `{interval_minutes}` minutes.

## Main Findings

- Common field variance share: `{common_variance_share:.4f}`.
- Residual variance share: `{residual_variance_share:.4f}`.
- Mean raw inter-turbine correlation: `{raw_mean_corr:.4f}`.
- Mean residual inter-turbine correlation: `{residual_mean_corr:.4f}`.
- Mean residual std: `{residual_turbine_std.mean():.4f}` m/s.
- Mean residual absolute deviation: `{residual_abs_mean.mean():.4f}` m/s.
- Mean residual std / raw turbine std: `{residual_ratio.mean():.4f}`.

Interpretation: most wind-speed variance is farm-level common variation, while turbine-specific residuals are much smaller but still non-negligible.

## Common Field Trend And Seasonality

- 48-hour rolling trend variance / common variance: `{trend_48h_share:.4f}`.
- 7-day rolling trend variance / common variance: `{trend_7d_share:.4f}`.
- Average daily profile amplitude: `{daily_amp:.4f}` m/s.
- Daily phase profile R2: `{daily_r2:.4f}`.

Autocorrelation:

| Lag | Autocorrelation |
| --- | ---: |
{chr(10).join(f"| {row['lag']} | {row['autocorr']:.4f} |" for row in autocorr_rows)}

Top FFT periods of the common field:

| Rank | Period (hours) | Period (days) | Power share |
| ---: | ---: | ---: | ---: |
{chr(10).join(f"| {i + 1} | {row.period_hours:.2f} | {row.period_days:.2f} | {row.power_share:.4f} |" for i, row in spectrum.iterrows())}

Interpretation: the common field has strong persistence and slow variation. A decomposition-style temporal backbone is justified, but the daily mean profile alone explains only a small part of variance; the more important component is low-frequency trend/regime variation rather than a clean deterministic daily seasonality.

## Residual Magnitude By Turbine

| Turbine | Raw std (m/s) | Residual std (m/s) | Mean abs residual (m/s) | Residual/raw std |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(f"| {row.turbine} | {row.raw_std_ms:.4f} | {row.residual_std_ms:.4f} | {row.residual_abs_mean_ms:.4f} | {row.residual_std_to_raw_std:.4f} |" for row in residual_table.itertuples())}

## Figures

- Common field first 21 valid days: `{figures['common_21_days'].as_posix()}`
- Average daily profile: `{figures['daily_profile'].as_posix()}`
- Residual std by turbine: `{figures['residual_std'].as_posix()}`

## Split Ranges

```json
{json.dumps(split_ranges, ensure_ascii=False, indent=2)}
```

## Modeling Implication

The data support a common-residual architecture:

1. Forecast the common field as the dominant farm-level signal.
2. Forecast smaller turbine residuals separately.
3. Apply spatial refinement on residuals, where true local differences are less masked by farm-wide co-movement.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(output_path)
    print(residual_csv)
    print(spectrum_csv)
    for path in figures.values():
        print(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("data/processed/penmanshiel_2016_2022_h6_window_fft.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/common_residual_analysis_2026-07-29.md"),
    )
    args = parser.parse_args()
    run(args.prepared, args.output)


if __name__ == "__main__":
    main()
