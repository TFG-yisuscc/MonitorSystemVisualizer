"""Locate standard files inside a run directory."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RunPaths:
    """Resolved file paths within a run directory."""

    root: Path
    resumen: Path
    prompt_metrics: Path
    hw_metrics: Path | None  # None if TYPE_0 (no hardware monitoring)
    meta: Path | None        # None if no meta.yaml provided


def discover_run_paths(run_dir: Path) -> RunPaths:
    """Locate the standard files inside a run directory.

    Raises:
        FileNotFoundError: if resumen.json or *_prompt_metrics_*.jsonl is missing.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    resumen = run_dir / "resumen.json"
    if not resumen.is_file():
        raise FileNotFoundError(f"Missing resumen.json in {run_dir}")

    prompt_files = sorted(run_dir.glob("*_prompt_metrics_*.jsonl"))
    if not prompt_files:
        raise FileNotFoundError(f"No *_prompt_metrics_*.jsonl in {run_dir}")
    if len(prompt_files) > 1:
        logger.warning(
            "Multiple prompt_metrics files in %s, using first: %s",
            run_dir, prompt_files[0].name,
        )
    prompt_metrics = prompt_files[0]

    hw_files = sorted(run_dir.glob("*_hw_metrics_*.jsonl"))
    if len(hw_files) > 1:
        logger.warning(
            "Multiple hw_metrics files in %s, using first: %s",
            run_dir, hw_files[0].name,
        )
    hw_metrics = hw_files[0] if hw_files else None

    meta_file = run_dir / "meta.yaml"
    meta = meta_file if meta_file.is_file() else None

    return RunPaths(
        root=run_dir,
        resumen=resumen,
        prompt_metrics=prompt_metrics,
        hw_metrics=hw_metrics,
        meta=meta,
    )
