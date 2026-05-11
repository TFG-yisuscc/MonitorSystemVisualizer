"""Public loading API: load_run() and load_collection()."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from monitorviz.models import Run, RunMeta

from ._discovery import discover_run_paths
from ._parsers import (
    _try_parse_meta_from_annotations,
    parse_hw_metrics,
    parse_meta,
    parse_prompt_metrics,
    parse_resumen,
)

if TYPE_CHECKING:
    from monitorviz.transforms.collection import RunCollection

logger = logging.getLogger(__name__)


def load_run(run_dir: Path | str) -> Run:
    """Load a single run from a directory.

    Steps:
      1. Discover standard files (resumen, prompt_metrics, optional hw, optional meta).
      2. Parse the resumen, including hardware_period_s from og_config_json.
      3. Parse prompt_metrics with the resumen's model_path_or_name as fallback
         when individual prompts have model="".
      4. Parse hw_metrics if present (TYPE_1, TYPE_2). TYPE_0 has none.
      5. Parse meta.yaml if present.
      6. Build the Run object. run_id = directory name.

    Raises:
        FileNotFoundError: if resumen.json or prompt_metrics is missing.
    """
    run_dir = Path(run_dir)
    paths = discover_run_paths(run_dir)
    run_id = run_dir.name

    summary = parse_resumen(paths.resumen)
    prompts = parse_prompt_metrics(
        paths.prompt_metrics,
        fallback_model=summary.model_path_or_name,
        run_id=run_id,
    )
    hw_samples = parse_hw_metrics(paths.hw_metrics) if paths.hw_metrics else []

    # Meta priority: meta.yaml > annotations JSON > empty defaults
    meta_from_annotations = _try_parse_meta_from_annotations(summary.raw_resumen)
    if paths.meta:
        file_dict = parse_meta(paths.meta).model_dump(exclude_unset=True)
        meta_from_annotations.update({k: v for k, v in file_dict.items() if v is not None})
    meta = RunMeta(**meta_from_annotations) if meta_from_annotations else RunMeta()

    return Run(
        run_id=run_id,
        summary=summary,
        meta=meta,
        prompts=prompts,
        hw_samples=hw_samples,
    )


def load_collection(
    root: Path | str,
    skip_errors: bool = True,
) -> RunCollection:
    """Load every run found in subdirectories of ``root``.

    With skip_errors=True (default), a run that fails to load logs a warning
    and is skipped; the rest still load. With skip_errors=False, the first
    failure raises.
    """
    # Local import to avoid circular dependency with transforms.collection
    from monitorviz.transforms.collection import RunCollection

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Collection root does not exist: {root}")

    runs: list[Run] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        try:
            runs.append(load_run(sub))
        except Exception as e:
            if skip_errors:
                logger.warning("Skipping run %s: %s", sub.name, e)
            else:
                raise
    logger.info("Loaded %d runs from %s", len(runs), root)
    return RunCollection(runs=runs)
