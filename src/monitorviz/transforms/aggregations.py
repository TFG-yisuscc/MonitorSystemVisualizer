"""Pure functions that turn a Run into pandas DataFrames.

The functions here do not depend on RunCollection. They are the building
blocks RunCollection uses to assemble cross-run tables.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from monitorviz.models import HwSample, Run

# --- factor columns --------------------------------------------------------

_FACTOR_COLUMNS: list[str] = [
    "run_id",
    "engine",
    "model_short",
    "model_label",
    "test_type",
    "batch_size",
    "context_size",
    "seed",
    "temperature",
    "hardware_period_s",
    "fan",
    "accelerator",
]

# Keys are substrings matched case-insensitively; first match wins.
_MODEL_DISPLAY_LABELS: list[tuple[str, str]] = [
    ("granite", "granite"),
    ("gemma", "gemma"),
    ("ministral", "ministral"),
    ("qwen", "qwen"),
    ("tinyllama", "tinyllama"),
]


def model_display_label(model_short: str) -> str:
    """Return a short one-word display label for use in plot axes and legends.

    Falls back to the first 10 characters of model_short if no pattern matches.
    """
    lower = model_short.lower()
    for key, label in _MODEL_DISPLAY_LABELS:
        if key.lower() in lower:
            return label
    return model_short[:10]


def _factor_dict(run: Run) -> dict[str, Any]:
    """Build the factor block that every row inherits from its run."""
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


# --- prompt metrics --------------------------------------------------------

def prompt_metrics_to_df(run: Run) -> pd.DataFrame:
    """One row per prompt of a single run.

    Columns: factor block + raw fields from PromptMetric + derived metrics
    (perplexity, tokens_per_second, words_per_second, time_to_first_token_ms,
    is_empty_generation, t_rel_s).
    """
    factors = _factor_dict(run)
    run_start_ns = run.summary.timestamp_run_start_ns

    rows: list[dict[str, Any]] = []
    for p in run.prompts:
        row = {
            **factors,
            "prompt_id": p.prompt_id,
            "model": p.model,
            "prob_type": p.prob_type,
            "start_timestamp_ns": p.start_timestamp_ns,
            "finish_timestamp_ns": p.finish_timestamp_ns,
            "total_duration_ns": p.total_duration_ns,
            "prompt_eval_duration_ns": p.prompt_eval_duration_ns,
            "eval_duration_ns": p.eval_duration_ns,
            "load_duration_ns": p.load_duration_ns,
            "prompt_eval_count": p.prompt_eval_count,
            "eval_count": p.eval_count,
            "n_words_answer": len(p.answer.split()),
            "is_empty_generation": p.is_empty_generation,
            "perplexity": p.perplexity,
            "tokens_per_second": p.tokens_per_second,
            "words_per_second": p.words_per_second,
            "time_to_first_token_ms": p.time_to_first_token_ns / 1e6,
            "latency_ms": p.total_duration_ns / 1e6,
            "t_rel_s": (p.start_timestamp_ns - run_start_ns) / 1e9,
        }
        rows.append(row)

    return pd.DataFrame(rows)


# --- tokens (long) ---------------------------------------------------------

def tokens_to_df(run: Run, include_llama: bool = True) -> pd.DataFrame:
    """Long format: one row per (prompt, token).

    LLAMA tokens have token="". With include_llama=False the LLAMA rows
    are dropped, useful for analyses that need token text.
    """
    factors = _factor_dict(run)
    rows: list[dict[str, Any]] = []
    for p in run.prompts:
        if p.is_empty_generation:
            continue
        if not include_llama and p.engine == "LLAMA":
            continue
        for idx, t in enumerate(p.tokens):
            rows.append(
                {
                    **factors,
                    "prompt_id": p.prompt_id,
                    "token_idx": idx,
                    "token": t.token,
                    "logprob": t.logprob,
                }
            )
    return pd.DataFrame(rows)


# --- hw metrics (wide) -----------------------------------------------------

_THROTTLE_FIELDS: list[str] = [
    "under_voltage",
    "under_voltage_occurred",
    "freq_capped",
    "freq_capped_occurred",
    "throttled",
    "throttled_occurred",
    "soft_throttled",
    "soft_throttled_occurred",
]


def _expand_hw_sample(s: HwSample, run_start_ns: int) -> dict[str, Any]:
    """Turn one HwSample into a flat dict, expanding throttling and freq stats."""
    freq = s.frequency_ghz
    row = {
        "timestamp_ms": s.timestamp_ms,
        "t_rel_s": (s.timestamp_ms * 1_000_000 - run_start_ns) / 1e9,
        "temperature_c": s.temperature_c,
        "fan_rpm": s.fan_rpm,
        "voltage_v": s.voltage_v,
        "internal_power_w": s.internal_power_w,
        "cpu_usage_pct": s.cpu_usage_pct,
        "mem_used_bytes": s.mem_used_bytes,
        "mem_total_bytes": s.mem_total_bytes,
        "mem_pct": s.mem_pct,
        "swap_used_bytes": s.swap_used_bytes,
        "swap_total_bytes": s.swap_total_bytes,
        "swap_pct": s.swap_pct,
        "freq_mean_ghz": float(np.mean(freq)) if freq else math.nan,
        "freq_max_ghz": float(np.max(freq)) if freq else math.nan,
        "freq_min_ghz": float(np.min(freq)) if freq else math.nan,
        "n_cores": len(freq),
    }
    for field in _THROTTLE_FIELDS:
        row[f"throt_{field}"] = getattr(s.throttling, field)
    row["throt_any_active"] = s.throttling.any_active
    row["throt_any_ever_occurred"] = s.throttling.any_ever_occurred
    return row


def hw_metrics_to_df(run: Run) -> pd.DataFrame:
    """One row per hardware sample of a single run. Wide format with
    aggregated frequency stats. Empty DataFrame if run has no hw data."""
    if not run.has_hardware_data:
        return pd.DataFrame()

    factors = _factor_dict(run)
    run_start_ns = run.summary.timestamp_run_start_ns
    rows = [{**factors, **_expand_hw_sample(s, run_start_ns)} for s in run.hw_samples]
    return pd.DataFrame(rows)


# --- hw frequency (long) ---------------------------------------------------

_GGUF_BPW: dict[str, float] = {
    "Q2_K":   2.63,
    "Q3_K_S": 3.00,
    "Q3_K_M": 3.35,
    "Q3_K_L": 3.60,
    "Q4_0":   4.00,
    "Q4_1":   4.50,
    "Q4_K_S": 4.25,
    "Q4_K_M": 4.50,
    "Q5_0":   5.00,
    "Q5_1":   5.50,
    "Q5_K_S": 5.25,
    "Q5_K_M": 5.50,
    "Q6_K":   6.50,
    "Q8_0":   8.00,
    "F16":   16.00,
    "BF16":  16.00,
    "F32":   32.00,
}


def _model_size_bytes(model_info: dict | None) -> float | None:
    """Model size in bytes. Uses bits_per_weight if available,
    otherwise infers from quantization string via lookup table."""
    if model_info is None:
        return None
    n_params = model_info.get("n_params")
    if n_params is None:
        return None
    bpw = model_info.get("bits_per_weight")
    if bpw is None:
        quant = (model_info.get("quantization") or "").upper()
        bpw = _GGUF_BPW.get(quant)
    if bpw is None:
        return None
    try:
        return int(n_params) * float(bpw) / 8.0
    except (ValueError, TypeError):
        return None


def _kv_cache_bytes(
    model_info: dict | None,
    seq_length: int,
    batch_size: int = 1,
) -> float | None:
    """KV cache size in bytes for a given sequence length.

    Formula from ELIB paper (Chen et al., 2025):
        KV_Cache = seq_length x n_layers x n_kv_heads x head_dim x 2 (K+V) x 2 (fp16 bytes)

    head_dim = embedding_length / n_heads
    """
    if model_info is None:
        return None
    n_layers = model_info.get("n_layers")
    n_kv_heads = model_info.get("n_kv_heads")
    embedding_length = model_info.get("embedding_length")
    n_heads = model_info.get("n_heads")
    if any(v is None for v in [n_layers, n_kv_heads, embedding_length, n_heads]):
        return None
    try:
        n_layers = int(n_layers)
        n_kv_heads = int(n_kv_heads)
        embedding_length = int(embedding_length)
        n_heads = int(n_heads)
    except (ValueError, TypeError):
        return None
    head_dim = embedding_length / n_heads
    return batch_size * seq_length * head_dim * n_layers * n_kv_heads * 2 * 2


def _mbu_pct(
    model_info: dict | None,
    tpot_s: float | None,
    seq_length: int,
    batch_size: int = 1,
    peak_bandwidth_gbs: float = 34.1,
) -> float | None:
    """Model Bandwidth Utilization (%) as defined in ELIB paper.

    Args:
        tpot_s: Time Per Output Token in seconds (eval_duration_s / eval_count).
        seq_length: Average sequence length during decode.
        batch_size: Batch size used during inference.
        peak_bandwidth_gbs: Peak memory bandwidth of the target device in GB/s.
    """
    if tpot_s is None or tpot_s <= 0 or model_info is None:
        return None
    model_bytes = _model_size_bytes(model_info)
    kv_bytes = _kv_cache_bytes(model_info, seq_length, batch_size)
    if model_bytes is None or kv_bytes is None:
        return None
    achieved_gbs = (model_bytes + kv_bytes) / tpot_s / 1e9
    return min(achieved_gbs / peak_bandwidth_gbs * 100, 100.0)


def _flops_gflops(
    model_info: dict | None,
    tokens_per_s: float | None,
) -> float | None:
    """Estimated decode FLOPS in GFLOPS.

    Approximation: 2 x n_params x tokens_per_s (standard LLM literature estimate).
    Does not account for attention KV cache overhead.
    """
    if model_info is None or tokens_per_s is None or tokens_per_s <= 0:
        return None
    n_params = model_info.get("n_params")
    if n_params is None:
        return None
    try:
        return 2.0 * int(n_params) * tokens_per_s / 1e9
    except (ValueError, TypeError):
        return None


def power_per_phase_df(run: Run) -> pd.DataFrame:
    """Compute mean power during each inference phase (load, prefill, decode).

    Aligns hw_metrics timestamps (milliseconds) with prompt_metrics phase
    windows (nanoseconds). Only phases long enough to contain at least one
    hw sample produce a reliable estimate; load is typically <300 ms and
    will usually contain 0 samples (returned as NaN).

    Returns a long-format DataFrame with columns:
        run_id, model_short, model_label, engine, prompt_id,
        phase (str), phase_start_s (float), phase_end_s (float),
        n_hw_samples (int), power_mean_w (float or NaN).
    """
    if not run.has_hardware_data:
        return pd.DataFrame()

    factors = _factor_dict(run)
    run_start_ns = run.summary.timestamp_run_start_ns

    hw_df = hw_metrics_to_df(run)
    if hw_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for p in run.prompts:
        if p.is_empty_generation:
            continue

        load_start = (p.start_timestamp_ns - run_start_ns) / 1e9
        load_end   = load_start + p.load_duration_ns / 1e9
        pre_start  = load_end
        pre_end    = pre_start + p.prompt_eval_duration_ns / 1e9
        dec_start  = pre_end
        dec_end    = dec_start + p.eval_duration_ns / 1e9

        phases = [
            ("load",    load_start, load_end),
            ("prefill", pre_start,  pre_end),
            ("decode",  dec_start,  dec_end),
        ]

        for phase_name, t_start, t_end in phases:
            mask = (hw_df["t_rel_s"] >= t_start) & (hw_df["t_rel_s"] < t_end)
            samples = hw_df.loc[mask, "internal_power_w"]
            rows.append({
                **factors,
                "prompt_id":     p.prompt_id,
                "phase":         phase_name,
                "phase_start_s": t_start,
                "phase_end_s":   t_end,
                "phase_dur_s":   t_end - t_start,
                "n_hw_samples":  len(samples),
                "power_mean_w":  float(samples.mean()) if len(samples) > 0 else float("nan"),
            })

    return pd.DataFrame(rows)


def hw_freq_long(run: Run) -> pd.DataFrame:
    """Long format for per-core frequency analysis.

    One row per (sample, core_id). Useful for seaborn with hue=core_id.
    """
    if not run.has_hardware_data:
        return pd.DataFrame()

    factors = _factor_dict(run)
    run_start_ns = run.summary.timestamp_run_start_ns
    rows: list[dict[str, Any]] = []
    for s in run.hw_samples:
        t_rel_s = (s.timestamp_ms * 1_000_000 - run_start_ns) / 1e9
        for core_id, freq in enumerate(s.frequency_ghz):
            rows.append(
                {
                    **factors,
                    "timestamp_ms": s.timestamp_ms,
                    "t_rel_s": t_rel_s,
                    "core_id": core_id,
                    "freq_ghz": freq,
                }
            )
    return pd.DataFrame(rows)
