from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_campaign_config(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    extends = raw.get("extends")
    if not extends:
        return raw
    base_path = Path(extends)
    if not base_path.is_absolute():
        base_path = target.parent / base_path
    base = load_campaign_config(base_path)
    return _deep_merge(base, raw)
