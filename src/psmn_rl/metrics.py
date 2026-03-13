from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import torch


def scalarize_metrics(metrics: dict[str, float | int | torch.Tensor]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                continue
            result[key] = float(value.detach().float().mean().item())
        else:
            result[key] = float(value)
    return result


def reduce_path_statistics(counts: torch.Tensor, top_ns: Iterable[int] = (10, 20, 50)) -> dict[str, float]:
    total = float(counts.sum().item())
    if total <= 0:
        return {"path_entropy": 0.0, **{f"path_coverage_top_{n}": 0.0 for n in top_ns}}
    probs = counts[counts > 0].float() / total
    entropy = float((-(probs * probs.log())).sum().item())
    sorted_probs = torch.sort(probs, descending=True).values
    stats = {"path_entropy": entropy}
    for top_n in top_ns:
        stats[f"path_coverage_top_{top_n}"] = float(sorted_probs[:top_n].sum().item())
    return stats


@dataclass
class RunningMean:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


@dataclass
class MetricAggregator:
    storage: dict[str, RunningMean] = field(default_factory=lambda: defaultdict(RunningMean))

    def update(self, metrics: dict[str, float | int | torch.Tensor], n: int = 1) -> None:
        for key, value in scalarize_metrics(metrics).items():
            self.storage[key].update(value, n=n)

    def compute(self) -> dict[str, float]:
        return {key: tracker.mean for key, tracker in self.storage.items()}
