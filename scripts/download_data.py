from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

import yaml

import _bootstrap
from wind_repro.config import project_root, resolve_project_path


def _download_file(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    print(f"downloading={url}")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(output)


def _download_git(entry: dict) -> None:
    target = resolve_project_path(entry["target_dir"])
    if target.exists() and any(target.iterdir()):
        print(f"skip_existing={target}")
        return
    if shutil.which("git") is None:
        raise RuntimeError("Git is required to download the Shanxi repository.")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", entry["url"], str(target)], check=True)


def _download_zenodo(entry: dict) -> None:
    record_id = str(entry["record_id"])
    api_url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(api_url) as response:
        metadata = json.load(response)
    target = resolve_project_path(entry["target_dir"])
    target.mkdir(parents=True, exist_ok=True)
    files = metadata.get("files", [])
    if not files:
        raise RuntimeError(f"Zenodo record {record_id} contains no files.")
    for item in files:
        output = target / item["key"]
        if output.exists() and output.stat().st_size == int(item.get("size", -1)):
            print(f"skip_existing={output}")
            continue
        _download_file(item["links"]["self"], output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["shanxi", "penmanshiel"])
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    manifest_path = project_root() / "data_manifest.yaml"
    with manifest_path.open("r", encoding="utf-8") as handle:
        datasets = yaml.safe_load(handle)["datasets"]
    if args.list_only or not args.dataset:
        for key, entry in datasets.items():
            print(f"{key}: {entry['url']} -> {entry['target_dir']}")
        return

    entry = datasets[args.dataset]
    if entry["source_type"] == "git":
        _download_git(entry)
    elif entry["source_type"] == "zenodo":
        _download_zenodo(entry)
    else:
        raise ValueError(f"Unsupported source type: {entry['source_type']}")


if __name__ == "__main__":
    main()
