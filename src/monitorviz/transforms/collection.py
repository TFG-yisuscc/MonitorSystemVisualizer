"""RunCollection: aggregates multiple runs and exposes cross-run tables."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from monitorviz.models import Run

from .aggregations import (
    _mbu_pct,
    _model_size_bytes,
    hw_freq_long,
    hw_metrics_to_df,
    model_display_label,
    prompt_metrics_to_df,
    tokens_to_df,
)
from .aggregations import (
    power_per_phase_df as _ppf,
)

logger = logging.getLogger(__name__)


@dataclass
class RunCollection:
    """Collection of Run objects with DataFrame-producing helpers.

    All DataFrames carry the experimental factor columns, so notebooks
    can directly use ``hue``, ``col``, ``row`` in seaborn or ``groupby``
    in pandas without further joins.
    """

    runs: list[Run] = field(default_factory=list)

    # --- container protocol ----------------------------------------------

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self):
        return iter(self.runs)

    # --- selection -------------------------------------------------------

    def filter(self, **factors: Any) -> RunCollection:
        """Return a new collection with runs matching all given factors.

        Factor names can be any attribute exposed in the factor block
        (engine, model_short, test_type, fan, accelerator, ...). Equality
        is exact. Unknown factor names raise ValueError.
        """
        valid_keys = {
            "run_id", "engine", "model_short", "test_type",
            "batch_size", "context_size", "seed", "temperature",
            "hardware_period_s", "fan", "accelerator",
        }
        unknown = set(factors) - valid_keys
        if unknown:
            raise ValueError(f"Unknown filter keys: {unknown}")

        def _matches(run: Run) -> bool:
            return all(_factor_value(run, k) == v for k, v in factors.items())

        return RunCollection(runs=[r for r in self.runs if _matches(r)])

    # --- per-event tables ------------------------------------------------

    def prompt_metrics_df(self) -> pd.DataFrame:
        """Concatenate prompt metrics across all runs."""
        if not self.runs:
            return pd.DataFrame()
        return pd.concat(
            [prompt_metrics_to_df(r) for r in self.runs],
            ignore_index=True,
        )

    def hw_metrics_df(self) -> pd.DataFrame:
        """Concatenate hardware samples across runs that have hw data.

        Logs a warning naming runs that were skipped (TYPE_0).
        """
        with_hw = [r for r in self.runs if r.has_hardware_data]
        without = [r.run_id for r in self.runs if not r.has_hardware_data]
        if without:
            logger.warning(
                "hw_metrics_df: skipping %d runs without hardware data: %s",
                len(without), without,
            )
        if not with_hw:
            return pd.DataFrame()
        return pd.concat(
            [hw_metrics_to_df(r) for r in with_hw],
            ignore_index=True,
        )

    def power_per_phase_df(self) -> pd.DataFrame:
        """Concatenate per-phase power data across all runs with hw."""
        with_hw = [r for r in self.runs if r.has_hardware_data]
        if not with_hw:
            return pd.DataFrame()
        return pd.concat(
            [_ppf(r) for r in with_hw],
            ignore_index=True,
        )

    def hw_freq_long_df(self) -> pd.DataFrame:
        """Long-format per-core frequency, concatenated across runs with hw."""
        with_hw = [r for r in self.runs if r.has_hardware_data]
        if not with_hw:
            return pd.DataFrame()
        return pd.concat(
            [hw_freq_long(r) for r in with_hw],
            ignore_index=True,
        )

    def tokens_df(self, include_llama: bool = True) -> pd.DataFrame:
        """Concatenate per-token logprobs across all runs.

        With include_llama=False, LLAMA tokens are dropped (no token text).
        """
        if not self.runs:
            return pd.DataFrame()
        return pd.concat(
            [tokens_to_df(r, include_llama=include_llama) for r in self.runs],
            ignore_index=True,
        )

    # --- run-level summary ----------------------------------------------

    def summary_df(self) -> pd.DataFrame:
        """One row per run with experimental factors and aggregated metrics.

        This is the primary table for cross-run comparative plots and tables.
        """
        if not self.runs:
            return pd.DataFrame()

        from monitorviz.viz.style import TARGET_PEAK_MEMORY_BANDWIDTH_GBs

        rows: list[dict[str, Any]] = []
        for run in self.runs:
            row: dict[str, Any] = _factor_block(run)
            row.update(_prompt_aggregates(run))
            row.update(_hw_aggregates(run))

            tps = row.get("tokens_per_s_mean")
            pwr = row.get("power_mean_w")
            row["energy_per_token_j"] = pwr / tps if (tps and pwr and tps > 0) else np.nan

            mi = run.summary.model_info
            tpot = row.get("tpot_mean_s")
            seq_len = run.summary.context_size
            model_bytes = _model_size_bytes(mi)
            row["model_size_gb"] = model_bytes / 1e9 if model_bytes is not None else np.nan
            row["mbu_pct"] = _mbu_pct(
                mi,
                tpot,
                seq_length=seq_len,
                batch_size=1,
                peak_bandwidth_gbs=TARGET_PEAK_MEMORY_BANDWIDTH_GBs,
            )

            rows.append(row)
        return pd.DataFrame(rows)

    def experiment_matrix(self) -> pd.DataFrame:
        """Coverage table: count of runs per (model_short x engine x fan x accelerator).

        Useful in the dissertation as evidence of which configurations
        have been measured. Missing combinations show as 0 or NaN.
        """
        if not self.runs:
            return pd.DataFrame()
        df = pd.DataFrame(
            [
                {
                    "model_short": r.model_short,
                    "engine": r.summary.inference_engine,
                    "fan": r.meta.fan,
                    "accelerator": r.meta.accelerator,
                }
                for r in self.runs
            ]
        )
        return (
            df.groupby(["model_short", "engine", "fan", "accelerator"], dropna=False)
              .size()
              .rename("n_runs")
              .reset_index()
        )


# ---------------------------------------------------------------------------
# Helpers (module-level so the dataclass body stays compact)
# ---------------------------------------------------------------------------

def _factor_value(run: Run, key: str) -> Any:
    """Resolve a factor value from a Run, looking up summary or meta as needed."""
    if key == "run_id":
        return run.run_id
    if key == "engine":
        return run.summary.inference_engine
    if key == "model_short":
        return run.model_short
    if key == "fan":
        return run.meta.fan
    if key == "accelerator":
        return run.meta.accelerator
    return getattr(run.summary, key, None)


def _factor_block(run: Run) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "engine": run.summary.inference_engine,
        "model_short": run.model_short,
        "model_label": model_display_label(run.model_short),
        "test_type": run.summary.test_type,
        "batch_size": run.summary.batch_size,
        "context_size": run.summary.context_size,
        "seed": run.summary.seed,
        "temperature": run.summary.temperature,
        "hardware_period_s": run.summary.hardware_period_s,
        "fan": run.meta.fan,
        "accelerator": run.meta.accelerator,
    }


def _prompt_aggregates(run: Run) -> dict[str, Any]:
    """Compute per-run aggregates over prompts. Missing data → NaN."""
    n_prompts = len(run.prompts)
    n_empty = sum(1 for p in run.prompts if p.is_empty_generation)

    tps = pd.Series([p.tokens_per_second for p in run.prompts], dtype="float64")
    wps = pd.Series([p.words_per_second for p in run.prompts], dtype="float64")
    ppl = pd.Series([p.perplexity for p in run.prompts], dtype="float64")
    lat_ms = pd.Series(
        [p.total_duration_ns / 1e6 if not p.is_empty_generation else np.nan
         for p in run.prompts],
        dtype="float64",
    )
    ttft_ms = pd.Series(
        [p.time_to_first_token_ns / 1e6 if not p.is_empty_generation else np.nan
         for p in run.prompts],
        dtype="float64",
    )
    tpot_series = pd.Series(
        [
            p.eval_duration_ns / p.eval_count / 1e9
            if not p.is_empty_generation and p.eval_count > 0
            else np.nan
            for p in run.prompts
        ],
        dtype="float64",
    )

    log_ppl = np.log(ppl.dropna())
    ppl_geomean = float(np.exp(log_ppl.mean())) if len(log_ppl) else np.nan

    return {
        "n_prompts": n_prompts,
        "n_empty_generations": n_empty,
        "tokens_per_s_mean": float(tps.mean()) if tps.notna().any() else np.nan,
        "tokens_per_s_p50": float(tps.median()) if tps.notna().any() else np.nan,
        "tokens_per_s_p95": float(tps.quantile(0.95)) if tps.notna().any() else np.nan,
        "words_per_s_mean": float(wps.mean()) if wps.notna().any() else np.nan,
        "perplexity_geomean": ppl_geomean,
        "perplexity_p50": float(ppl.median()) if ppl.notna().any() else np.nan,
        "latency_ms_mean": float(lat_ms.mean()) if lat_ms.notna().any() else np.nan,
        "latency_ms_p95": float(lat_ms.quantile(0.95)) if lat_ms.notna().any() else np.nan,
        "ttft_ms_mean": float(ttft_ms.mean()) if ttft_ms.notna().any() else np.nan,
        "tpot_mean_s": float(tpot_series.mean()) if tpot_series.notna().any() else np.nan,
    }


def _hw_aggregates(run: Run) -> dict[str, Any]:
    """Compute per-run aggregates over hardware samples. NaN if no hw data."""
    if not run.has_hardware_data:
        return {
            "temp_max_c": np.nan,
            "temp_mean_c": np.nan,
            "power_mean_w": np.nan,
            "power_max_w": np.nan,
            "throttled_ratio": np.nan,
            "throttled_any_ever": np.nan,
        }

    temps = np.array([s.temperature_c for s in run.hw_samples], dtype="float64")
    pwrs = np.array([s.internal_power_w for s in run.hw_samples], dtype="float64")
    active = np.array([s.throttling.any_active for s in run.hw_samples], dtype="bool")
    ever = any(s.throttling.any_ever_occurred for s in run.hw_samples)
    return {
        "temp_max_c": float(np.max(temps)),
        "temp_mean_c": float(np.mean(temps)),
        "power_mean_w": float(np.mean(pwrs)),
        "power_max_w": float(np.max(pwrs)),
        "throttled_ratio": float(np.mean(active)),
        "throttled_any_ever": bool(ever),
    }
