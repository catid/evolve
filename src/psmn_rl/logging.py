from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter

from psmn_rl.config import ExperimentConfig, dump_config


LOGGER = logging.getLogger("psmn_rl")


class RunLogger:
    def __init__(self, config: ExperimentConfig, enabled: bool = True) -> None:
        self.enabled = enabled
        self.output_dir = Path(config.logging.output_dir)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.summary_path = self.output_dir / "summary.md"
        self.writer: SummaryWriter | None = None
        if not self.enabled:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dump_config(config, self.output_dir / "resolved_config.yaml")
        if config.logging.tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "tb"))
        env_meta = {
            "pid": os.getpid(),
            "started_at": time.time(),
        }
        self._append_json({"type": "run_meta", **env_meta})

    def _append_json(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def log(self, step: int, metrics: dict[str, float]) -> None:
        if not self.enabled:
            return
        payload = {"type": "scalar", "step": int(step), **metrics}
        self._append_json(payload)
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

    def write_summary(self, text: str) -> None:
        if not self.enabled:
            return
        self.summary_path.write_text(text)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


def configure_logging(is_main_process: bool = True) -> None:
    level = logging.INFO if is_main_process else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
