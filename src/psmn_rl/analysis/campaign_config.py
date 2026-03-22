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
    if extends:
        base_path = Path(extends)
        if not base_path.is_absolute():
            base_path = target.parent / base_path
        raw = _deep_merge(load_campaign_config(base_path), raw)

    candidate_subset = raw.get("candidate_subset")
    if candidate_subset is not None:
        candidates = raw.get("candidates", {})
        ordered_candidates: dict[str, Any] = {}
        missing = [str(name) for name in candidate_subset if str(name) not in candidates]
        if missing:
            raise KeyError(f"campaign candidate_subset references unknown candidates: {missing}")
        for name in candidate_subset:
            ordered_candidates[str(name)] = copy.deepcopy(candidates[str(name)])
        raw["candidates"] = ordered_candidates
    return raw
