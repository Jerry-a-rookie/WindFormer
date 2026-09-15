from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import _bootstrap
from wind_repro.config import load_config, project_root, resolve_project_path
from wind_repro.data import load_prepared


def _common_field(speed: np.ndarray) -> np.ndarray:
    values = np.asarray(speed, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected [time, turbines], got {values.shape}")
    return np.nanmean(values, axis=1)


def _detrend(values: np.ndarray) -> np.ndarray:
    time = np.linspace(-1.0, 1.0, len(values), dtype=np.float64)
    coefficients = np.polyfit(time, values, deg=1)
    return values - np.polyval(coefficients, time)


def _spectral_score(values: np.ndarray, period: int) -> float:
    detrended = _detrend(values)
    spectrum = np.abs(np.fft.rfft(detrended))
    if len(spectrum) <= 1:
        return 0.0
    index = max(1, min(len(spectrum) - 1, round(len(values) / period)))
    left = max(1, index - 1)
    right = min(len(spectrum), index + 2)
    score = float(np.max(spectrum[left:right]))
    return score / max(float(np.sum(spectrum[1:])), 1e-12)


def _autocorrelation_score(values: np.ndarray, period: int) -> float:
    if period >= len(values):
        return 0.0
    left = values[:-period] - values[:-period].mean()
    right = values[period:] - values[period:].mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / max(float(denominator), 1e-12))


def _stability_score(values: np.ndarray, period: int, blocks: int = 4) -> float:
    block_size = len(values) // blocks
    if block_size < max(period * 2, 4):
        return 0.0
    scores = []
    for block in range(blocks):
        start = block * block_size
        end = start + block_size
        scores.append(_autocorrelation_score(values[start:end], period))
    scores = np.asarray(scores, dtype=np.float64)
    return float(np.mean(scores) - 0.5 * np.std(scores))


def _seasonal_naive_mae(
    values: np.ndarray,
    period: int,
    horizon: int,
    max_windows: int = 20000,
) -> float:
    if len(values) <= period + horizon:
        return float("inf")
    starts = np.arange(period, len(values) - horizon + 1)
    if len(starts) > max_windows:
        starts = np.linspace(
            starts[0],
            starts[-1],
            max_windows,
            dtype=np.int64,
        )
    errors = []
    for start in starts:
        errors.append(
            np.mean(
                np.abs(
                    values[start : start + horizon]
                    - values[start - period : start - period + horizon]
                )
            )
        )
    return float(np.mean(errors))


def _rank_normalize(values: list[float], descending: bool) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(-array if descending else array)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(order), dtype=np.float64)
    if len(order) <= 1:
        return np.ones_like(ranks)
    return 1.0 - ranks / (len(order) - 1)


def select_periods(
    prepared_path: str | Path,
    candidate_periods: list[int],
    horizon: int,
    top_k: int,
) -> dict[str, Any]:
    prepared = load_prepared(prepared_path)
    train = _common_field(prepared["train_speed"])
    validation = _common_field(prepared["validation_speed"])
    candidate_periods = list(dict.fromkeys(int(value) for value in candidate_periods))
    if not candidate_periods:
        raise ValueError("At least one candidate period is required.")

    spectral = [_spectral_score(train, period) for period in candidate_periods]
    autocorrelation = [
        _autocorrelation_score(train, period) for period in candidate_periods
    ]
    stability = [
        _stability_score(train, period) for period in candidate_periods
    ]
    validation_mae = [
        _seasonal_naive_mae(validation, period, horizon)
        for period in candidate_periods
    ]

    combined = (
        0.35 * _rank_normalize(spectral, descending=True)
        + 0.25 * _rank_normalize(autocorrelation, descending=True)
        + 0.15 * _rank_normalize(stability, descending=True)
        + 0.25 * _rank_normalize(validation_mae, descending=False)
    )
    order = np.argsort(-combined)
    selected = [
        candidate_periods[int(index)]
        for index in order[: min(max(top_k, 1), len(order))]
    ]
    rows = []
    for index, period in enumerate(candidate_periods):
        rows.append(
            {
                "period": period,
                "hours": period / 6.0,
                "spectral_score": spectral[index],
                "autocorrelation": autocorrelation[index],
                "stability_score": stability[index],
                "validation_seasonal_naive_mae": validation_mae[index],
                "combined_score": float(combined[index]),
            }
        )
    rows.sort(key=lambda row: row["combined_score"], reverse=True)
    return {
        "prepared_path": str(prepared_path),
        "interval_minutes": 10,
        "lookback": int(prepared["metadata"]["lookback"]),
        "horizon": int(horizon),
        "candidate_periods": candidate_periods,
        "selected_periods": selected,
        "method": {
            "spectral_weight": 0.35,
            "autocorrelation_weight": 0.25,
            "stability_weight": 0.15,
            "validation_naive_weight": 0.25,
            "validation_is_used_for_selection": True,
        },
        "scores": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Screen candidate periods using train statistics and validation "
            "seasonal-naive error."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output path. Defaults to reports/period_screening_h6.json.",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        type=int,
        default=[6, 12, 18, 24, 36, 48, 72, 96],
    )
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]
    prepared_path = resolve_project_path(data_config["prepared_path"])
    horizon = int(args.horizon or data_config.get("horizon", 6))
    result = select_periods(
        prepared_path,
        args.candidates,
        horizon,
        args.top_k,
    )
    output = (
        project_root() / "reports" / "period_screening_h6.json"
        if args.output is None
        else resolve_project_path(args.output)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"period_screening={output}")
    print(
        "selected_periods="
        + ",".join(
            f"{period}({period / 6.0:.1f}h)"
            for period in result["selected_periods"]
        )
    )
    for row in result["scores"]:
        print(
            f"period={row['period']} "
            f"hours={row['hours']:.1f} "
            f"combined={row['combined_score']:.4f} "
            f"validation_naive_mae={row['validation_seasonal_naive_mae']:.6f}"
        )


if __name__ == "__main__":
    main()
