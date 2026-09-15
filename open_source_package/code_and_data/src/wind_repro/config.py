from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config = deepcopy(config)
    config["_config_path"] = str(config_path)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate

