"""Low-level parsers for individual files. Not part of the public API."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import yaml

from monitorviz.models import (
    HwSample,
    PromptMetric,
    RunMeta,
    RunSummary,
    ThrottlingFlags,
    TokenProb,
)

logger = logging.getLogger(__name__)


# --- resumen.json ----------------------------------------------------------

def parse_resumen(path: Path) -> RunSummary:
    """Read resumen.json and build a RunSummary.

    Parses og_config_json (a JSON-encoded string nested inside the resumen)
    to extract hardware_period_s. Maps the typo 'anotations' to 'annotations'.
    """
    with open(path) as f:
        raw = json.load(f)

    og_config_str = raw.get("og_config_json", "{}")
    try:
        og_config = json.loads(og_config_str)
    except json.JSONDecodeError as e:
        logger.warning("Could not parse og_config_json in %s: %s", path, e)
        og_config = {}

    hardware_period_s = float(og_config.get("hardware_period", 0.0))
    raw_annot = raw.get("annotations", raw.get("anotations", ""))
    if isinstance(raw_annot, dict):
        annotations_str = json.dumps(raw_annot)
        model_info = raw_annot.get("model_info")
    elif isinstance(raw_annot, str):
        annotations_str = raw_annot
        try:
            parsed = json.loads(raw_annot)
            model_info = parsed.get("model_info") if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, TypeError):
            model_info = None
    else:
        annotations_str = ""
        model_info = None
    annotations = annotations_str

    return RunSummary(
        timestamp_run_start_ns=raw["timestamp_run_start"],
        timestamp_run_end_ns=raw["timestamp_run_end"],
        inference_engine=raw["inference_engine"],
        model_path_or_name=raw["model_path_or_name"],
        test_type=raw["test_type"],
        batch_size=raw["batch_size"],
        context_size=raw["context_size"],
        num_prompts=raw["num_prompts"],
        seed=raw["seed"],
        temperature=raw["temperature"],
        hardware_period_s=hardware_period_s,
        annotations=annotations,
        model_info=model_info,
        total_kwh=raw.get("total_kwh"),
        watt_min=raw.get("watt_min"),
        watt_max=raw.get("watt_max"),
        raw_resumen=raw,
        raw_og_config=og_config,
    )


# --- tokenProb normalization ----------------------------------------------

_PROB_TYPE_MAP: dict[int | str, int] = {
    "LOG_PROBABILITY": 0,
    "log_probability": 0,
    0: 0,
    "0": 0,
    "PROBABILITY": 1,
    "probability": 1,
    1: 1,
    "1": 1,
}


def _parse_token_prob(entry: dict | float | str) -> TokenProb:
    """Parse a single tokenProb entry.

    OLLAMA format: {"token": "hello", "logprob": -0.5, "bytes": [...]}
    LLAMA format:  raw logprob value, no token text
    """
    if isinstance(entry, dict):
        return TokenProb(**entry)
    try:
        return TokenProb(token="", logprob=float(entry))
    except (ValueError, TypeError):
        return TokenProb(token="", logprob=float("nan"))


def normalize_token_prob(
    raw_token_prob: str | list,
    prob_type: int | str,
    run_id: str,
    prompt_id: int,
) -> list[TokenProb]:
    """Convert tokenProb from either engine format to a uniform list[TokenProb].

    Accepts either the raw JSON string (typical) or an already-parsed list
    (defensive: tolerate future binary changes).

    - prob_type 0 / "LOG_PROBABILITY" (OLLAMA): list of dicts {token, logprob, bytes}.
    - prob_type 1 / "PROBABILITY" (LLAMA): list of strings with raw probabilities.
      Convert with math.log(p), guarding against p <= 0 and p > 1.
    """
    prob_type_int = _PROB_TYPE_MAP.get(prob_type, -1)
    if prob_type_int == -1:
        logger.warning(
            "Unknown probType=%r for run=%s prompt=%s, treating as OLLAMA (0)",
            prob_type, run_id, prompt_id,
        )
        prob_type_int = 0

    if isinstance(raw_token_prob, str):
        try:
            parsed = json.loads(raw_token_prob)
        except json.JSONDecodeError as e:
            logger.warning(
                "tokenProb is not valid JSON for run=%s prompt=%s: %s",
                run_id, prompt_id, e,
            )
            return []
    else:
        parsed = raw_token_prob

    if not isinstance(parsed, list):
        logger.warning(
            "tokenProb did not parse to a list for run=%s prompt=%s",
            run_id, prompt_id,
        )
        return []

    tokens: list[TokenProb] = []

    if prob_type_int == 0:
        for item in parsed:
            if not isinstance(item, dict):
                logger.warning(
                    "Expected dict in OLLAMA tokenProb, got %r (run=%s prompt=%s)",
                    type(item).__name__, run_id, prompt_id,
                )
                continue
            tokens.append(_parse_token_prob(item))
    elif prob_type_int == 1:
        _invalid_prob_count = 0
        for item in parsed:
            try:
                p = float(item)
            except (TypeError, ValueError):
                logger.warning(
                    "Could not parse LLAMA prob %r (run=%s prompt=%s)",
                    item, run_id, prompt_id,
                )
                continue
            if p <= 0:
                _invalid_prob_count += 1
                logprob = float("-inf")
            elif p > 1:
                logger.warning(
                    "Probability >1 (%s) in LLAMA tokenProb (run=%s prompt=%s)",
                    p, run_id, prompt_id,
                )
                logprob = 0.0
            else:
                logprob = math.log(p)
            tokens.append(TokenProb(token="", logprob=logprob))
        if _invalid_prob_count:
            logger.warning(
                "Non-positive prob en LLAMA tokenProb: %d tokens omitidos "
                "(run=%s, prompt=%d). Run posiblemente corrupto.",
                _invalid_prob_count, run_id, prompt_id,
            )

    return tokens


def _try_parse_meta_from_annotations(raw: dict) -> dict:
    """Extract meta fields from the annotations field of resumen.json.

    Handles:
    - New binary: annotations is a native JSON dict.
    - Old binary: anotations is a JSON-encoded string.
    - Very old: plain text string (ignored).

    Excludes model_info (handled separately in parse_resumen).
    """
    raw_annot = raw.get("annotations", raw.get("anotations"))
    if raw_annot is None:
        return {}

    if isinstance(raw_annot, dict):
        data = raw_annot
    elif isinstance(raw_annot, str):
        try:
            data = json.loads(raw_annot)
            if not isinstance(data, dict):
                return {}
        except (json.JSONDecodeError, TypeError):
            return {}
    else:
        return {}

    return {k: v for k, v in data.items() if k != "model_info"}


# --- prompt_metrics --------------------------------------------------------

def parse_prompt_metrics(
    path: Path,
    fallback_model: str,
    run_id: str,
) -> list[PromptMetric]:
    """Parse a prompt_metrics JSONL file. One line = one PromptMetric."""
    prompts: list[PromptMetric] = []
    with open(path) as f:
        for line_num, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                d = json.loads(raw_line)
            except json.JSONDecodeError as e:
                logger.error(
                    "Malformed JSON in %s line %d: %s",
                    path, line_num, e,
                )
                continue

            prompt_id = d.get("prompt_id", line_num - 1)
            raw_prob_type = d.get("probType", 0)
            tokens = normalize_token_prob(
                d.get("tokenProb", "[]"),
                prob_type=raw_prob_type,
                run_id=run_id,
                prompt_id=prompt_id,
            )

            eval_count = int(d.get("eval_count", 0))
            diff = abs(len(tokens) - eval_count)
            if diff >= 2:
                logger.warning(
                    "tokenProb size mismatch in run=%s prompt=%s: "
                    "got %d tokens, eval_count=%d",
                    run_id, prompt_id, len(tokens), eval_count,
                )

            model = d.get("model", "") or fallback_model

            prompts.append(
                PromptMetric(
                    engine=d["engine"],
                    probType=raw_prob_type,
                    model=model,
                    prompt_id=prompt_id,
                    start_timestamp_ns=int(d.get("start_timestamp_ns", 0)),
                    finish_timestamp_ns=int(d.get("finish_timestamp_ns", 0)),
                    total_duration_ns=int(d.get("total_duration_ns", 0)),
                    prompt_eval_duration_ns=int(d.get("prompt_eval_duration_ns", 0)),
                    eval_duration_ns=int(d.get("eval_duration_ns", 0)),
                    load_duration_ns=int(d.get("load_duration_ns", 0)),
                    prompt_eval_count=int(d.get("prompt_eval_count", 0)),
                    eval_count=eval_count,
                    answer=d.get("answer", ""),
                    tokens=tokens,
                )
            )
    return prompts


# --- hw_metrics ------------------------------------------------------------

_HW_FIELD_MAP: dict[str, str] = {
    "temperature_": "temperature_c",
    "fan_speed_": "fan_rpm",
    "voltage_": "voltage_v",
    "internalpower_": "internal_power_w",
    "frequency_": "frequency_ghz",
    "cpu_usage_": "cpu_usage_pct",
    "cpu_ticks_": "cpu_ticks",
    "mem_used_": "mem_used_bytes",
    "mem_total_": "mem_total_bytes",
    "mem_percent_": "mem_pct",
    "swap_used_": "swap_used_bytes",
    "swap_total_": "swap_total_bytes",
    "swap_percent_": "swap_pct",
    "timestamp_": "timestamp_ms",
}


def _map_hw_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Rename C++-style keys to model-style keys, ignoring engine_."""
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k == "engine_":
            continue
        new_key = _HW_FIELD_MAP.get(k, k)
        out[new_key] = v
    return out


def parse_hw_metrics(path: Path) -> list[HwSample]:
    """Parse a hw_metrics JSONL file. One line = one HwSample."""
    samples: list[HwSample] = []
    with open(path) as f:
        for line_num, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                d = json.loads(raw_line)
            except json.JSONDecodeError as e:
                logger.error(
                    "Malformed JSON in %s line %d: %s",
                    path, line_num, e,
                )
                continue

            mapped = _map_hw_dict(d)
            throttling_raw = mapped.pop("throttling_", None) or mapped.pop(
                "throttling", None
            )
            if throttling_raw is None:
                logger.warning("Missing throttling in %s line %d", path, line_num)
                continue

            try:
                samples.append(
                    HwSample(
                        **mapped,
                        throttling=ThrottlingFlags(**throttling_raw),
                    )
                )
            except Exception as e:
                logger.error(
                    "Could not build HwSample from %s line %d: %s",
                    path, line_num, e,
                )
    return samples


# --- meta.yaml -------------------------------------------------------------

def parse_meta(path: Path) -> RunMeta:
    """Read meta.yaml. On any error returns RunMeta() with a warning logged."""
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            logger.warning("meta.yaml in %s is not a mapping, ignoring", path)
            return RunMeta()
        return RunMeta(**data)
    except Exception as e:
        logger.warning("Could not parse meta.yaml in %s: %s", path, e)
        return RunMeta()
