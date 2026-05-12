"""YAML configuration loading and command-line override helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Config root must be a mapping in {config_path}")
    payload["_config_path"] = str(config_path)
    return payload


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save a configuration dictionary as YAML."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge updates into a copy of base."""

    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def set_by_path(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    """Set a nested configuration value using dot notation."""

    parts = dotted_path.split(".")
    cursor: dict[str, Any] = config
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def parse_override(override: str) -> tuple[str, Any]:
    """Parse a key=value command-line override."""

    if "=" not in override:
        raise ValueError(f"Override must be key=value, got {override!r}")
    key, raw_value = override.split("=", 1)
    if not key:
        raise ValueError("Override key cannot be empty")
    return key, yaml.safe_load(raw_value)


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply a list of dotted key=value overrides to a config copy."""

    updated = copy.deepcopy(config)
    for override in overrides or []:
        key, value = parse_override(override)
        set_by_path(updated, key, value)
    return updated


def apply_named_overrides(config: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    """Apply common flat CLI names to their nested config locations."""

    mapping = {
        "data_dir": "data.data_dir",
        "split_index": "data.split_index",
        "batch_size": "train.batch_size",
        "lr": "train.lr",
        "epochs": "train.num_epochs",
        "resume_from": "train.resume_from",
        "output_dir": "train.output_dir",
        "run_name": "train.run_name",
    }
    updated = copy.deepcopy(config)
    for arg_name, dotted_path in mapping.items():
        value = values.get(arg_name)
        if value is not None:
            set_by_path(updated, dotted_path, value)
    return updated
