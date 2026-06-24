"""Figure of Merit (FoM) calculations for E0–E5 experimental series.

Two FoM variants are provided:

* FoM_full  — composite of 4 normalised axes; used for series E0–E4.
* FoM_red   — reduced 2-axis version for series E5 (intra-model normalisation
              vs the CPU baseline of each model).

A usability criterion flags runs whose throughput falls below the minimum
human-readable rate (3 words/s converted to tokens/s per model).
"""

from __future__ import annotations

import math
import re
from typing import Final

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reference configuration for FoM_full normalisation
# ---------------------------------------------------------------------------

FOM_REF_MODEL_SUBSTR: Final[str] = "llama3.2_3b"   # matched in model_short
FOM_REF_N_PARAMS: Final[float] = 3e9               # llama3.2:3b parameter count
FOM_REF_ENGINE: Final[str] = "LLAMA"
FOM_REF_QUANTIZATION: Final[str] = "Q4_K_M"
FOM_REF_CONTEXT: Final[int] = 4096
FOM_REF_BATCH: Final[int] = 512
FOM_REF_SEED: Final[int] = 42
FOM_REF_TEMPERATURE: Final[float] = 0.0

# Minimum human-readable rate used for the usability criterion.
USABILITY_MIN_WORDS_PER_S: Final[float] = 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _geomean(values: list[float]) -> float:
    """Geometric mean of a list of positive finite values."""
    if not values:
        return math.nan
    log_sum = sum(math.log(v) for v in values if v > 0 and math.isfinite(v))
    n = sum(1 for v in values if v > 0 and math.isfinite(v))
    return math.exp(log_sum / n) if n else math.nan


def _find_reference_row(df: pd.DataFrame) -> pd.Series | None:
    """Return the reference row from df, or None if not found.

    Matches model_short after stripping separators (``-``, ``_``, ``.``, ``:``)
    so that e.g. ``"Llama-3.2-3B-Instruct-Q4_K_M"`` and ``"llama3.2_3b:q4_k_m"``
    both match the constant ``FOM_REF_MODEL_SUBSTR = "llama3.2_3b"``.
    """
    _substr_norm = re.sub(r"[-_:.]", "", FOM_REF_MODEL_SUBSTR.lower())
    _model_norm = df["model_short"].str.lower().str.replace(r"[-_:.]", "", regex=True)
    mask = (
        _model_norm.str.contains(_substr_norm, regex=False)
        & (df["engine"].str.upper() == FOM_REF_ENGINE)
        & (df["context_size"] == FOM_REF_CONTEXT)
        & (df["batch_size"] == FOM_REF_BATCH)
        & (df["seed"] == FOM_REF_SEED)
        & (df["temperature"] == FOM_REF_TEMPERATURE)
    )
    hits = df[mask]
    if hits.empty:
        return None
    if len(hits) > 1:
        if "quantization" in hits.columns:
            q_hits = hits[hits["quantization"].str.upper() == FOM_REF_QUANTIZATION]
            if not q_hits.empty:
                return q_hits.iloc[0]
        else:
            # quantization is embedded in model_short (e.g. "Llama-3.2-3B-Instruct-Q4_K_M")
            q_hits = hits[hits["model_short"].str.upper().str.contains(
                FOM_REF_QUANTIZATION, regex=False
            )]
            if not q_hits.empty:
                return q_hits.iloc[0]
    return hits.iloc[0]


# ---------------------------------------------------------------------------
# FoM_full  (E0–E4)
# ---------------------------------------------------------------------------

def compute_fom_full(
    df: pd.DataFrame,
    n_params_col: str = "n_params",
    weights: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame:
    """Add FoM_full and its four normalised axes to a summary DataFrame.

    The input DataFrame must contain the columns produced by
    ``RunCollection.summary_df()``. A column ``n_params`` with the number of
    model parameters is required; if the column is absent the function tries
    ``model_size_gb * 1e9 / bpw * 8`` as a rough fallback but this is
    unreliable — prefer populating n_params from model_info.

    New columns added (in-place copy is returned):
        fom_T_norm      — (N * tokens_per_s_mean) / (N_ref * T_ref)
        fom_MBU_norm    — mbu_corr / MBU_corr_ref
        fom_eta_norm    — cpu_efficiency / eta_CPU_ref
        fom_eps_norm    — (N * eps) / (N_ref * eps_ref),  eps = tokens/J
        fom_full        — geometric mean of the four normed axes
        fom_full_w      — weighted geometric mean (if weights provided)

    Args:
        weights: Four non-negative weights (w_T, w_MBU, w_eta, w_eps).
            If None, equal weights (1,1,1,1) are used (equivalent to fom_full).
    """
    out = df.copy()

    # --- Resolve N_params per row -----------------------------------------
    if n_params_col in out.columns:
        N = out[n_params_col].astype(float)
    else:
        # Rough fallback: model_size_gb is in bytes already normalised
        N = out.get("model_size_gb", pd.Series(np.nan, index=out.index)) * 1e9
    out["_N"] = N

    # eps = tokens per joule
    out["_eps"] = 1.0 / out["energy_per_token_j"].replace(0, np.nan)

    # --- Reference row ----------------------------------------------------
    ref = _find_reference_row(out)
    if ref is None:
        out["fom_T_norm"] = np.nan
        out["fom_MBU_norm"] = np.nan
        out["fom_eta_norm"] = np.nan
        out["fom_eps_norm"] = np.nan
        out["fom_full"] = np.nan
        if weights is not None:
            out["fom_full_w"] = np.nan
        out.drop(columns=["_N", "_eps"], inplace=True)
        return out

    N_ref = float(ref["_N"]) if not np.isnan(float(ref["_N"])) else FOM_REF_N_PARAMS
    T_ref = float(ref["tokens_per_s_mean"])
    MBU_ref = float(ref["mbu_corr"])
    eta_ref = float(ref["cpu_efficiency"])
    eps_ref = float(ref["_eps"])

    # --- Normalised axes --------------------------------------------------
    out["fom_T_norm"] = (out["_N"] * out["tokens_per_s_mean"]) / (N_ref * T_ref)
    out["fom_MBU_norm"] = out["mbu_corr"] / MBU_ref
    out["fom_eta_norm"] = out["cpu_efficiency"] / eta_ref
    out["fom_eps_norm"] = (out["_N"] * out["_eps"]) / (N_ref * eps_ref)

    axes = ["fom_T_norm", "fom_MBU_norm", "fom_eta_norm", "fom_eps_norm"]

    def _row_fom(row: pd.Series, w: tuple[float, ...]) -> float:
        vals = [row[c] for c in axes]
        if any(not np.isfinite(v) or v <= 0 for v in vals):
            return math.nan
        log_sum = sum(wi * math.log(v) for wi, v in zip(w, vals))
        return math.exp(log_sum / sum(w))

    w_eq = (1.0, 1.0, 1.0, 1.0)
    out["fom_full"] = out.apply(_row_fom, axis=1, w=w_eq)

    if weights is not None:
        out["fom_full_w"] = out.apply(_row_fom, axis=1, w=weights)

    out.drop(columns=["_N", "_eps"], inplace=True)
    return out


# ---------------------------------------------------------------------------
# FoM_red  (E5 — intra-model normalisation vs CPU baseline)
# ---------------------------------------------------------------------------

def compute_fom_red(
    df: pd.DataFrame,
    cpu_engine: str = "LLAMA",
    n_params_col: str = "n_params",
) -> pd.DataFrame:
    """Add FoM_red and its two normalised axes to a summary DataFrame.

    FoM_red = sqrt(T5_norm * eps5_norm)

    Normalisation is intra-model, relative to the CPU run for each model:
        T5_norm   = tokens_per_s_mean / T_cpu_m
        eps5_norm = eps / eps_cpu_m   (eps = tokens/J)

    The CPU baseline is identified as the run with ``engine == cpu_engine``
    for each (model_short, context_size, batch_size, seed, temperature) group.
    If no CPU baseline is found for a model, fom_red is NaN for all rows of
    that model.

    New columns added:
        fom_T5_norm     — T / T_cpu_m
        fom_eps5_norm   — eps / eps_cpu_m
        fom_red         — sqrt(fom_T5_norm * fom_eps5_norm)
    """
    out = df.copy()
    out["_eps"] = 1.0 / out["energy_per_token_j"].replace(0, np.nan)

    group_keys = ["model_short", "context_size", "batch_size", "seed", "temperature"]

    fom_T5 = np.full(len(out), np.nan)
    fom_eps5 = np.full(len(out), np.nan)

    for _, grp in out.groupby(group_keys, dropna=False):
        cpu_rows = grp[grp["engine"].str.upper() == cpu_engine.upper()]
        if cpu_rows.empty:
            continue
        T_cpu = float(cpu_rows["tokens_per_s_mean"].mean())
        eps_cpu = float(cpu_rows["_eps"].mean())
        if not (np.isfinite(T_cpu) and T_cpu > 0):
            continue
        if not (np.isfinite(eps_cpu) and eps_cpu > 0):
            continue
        idx = grp.index
        fom_T5[out.index.get_indexer(idx)] = out.loc[idx, "tokens_per_s_mean"] / T_cpu
        fom_eps5[out.index.get_indexer(idx)] = out.loc[idx, "_eps"] / eps_cpu

    out["fom_T5_norm"] = fom_T5
    out["fom_eps5_norm"] = fom_eps5
    out["fom_red"] = np.where(
        np.isfinite(fom_T5) & np.isfinite(fom_eps5) & (fom_T5 > 0) & (fom_eps5 > 0),
        np.sqrt(fom_T5 * fom_eps5),
        np.nan,
    )

    out.drop(columns=["_eps"], inplace=True)
    return out


# ---------------------------------------------------------------------------
# Usability criterion
# ---------------------------------------------------------------------------

def compute_usability(
    summary_df: pd.DataFrame,
    prompt_df: pd.DataFrame,
    min_words_per_s: float = USABILITY_MIN_WORDS_PER_S,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add usability columns to summary_df and return per-model token/word factors.

    Empirically measures the tokens-per-word ratio per model from prompt_df
    (which must contain ``eval_count`` and ``n_words_answer``), then derives
    the human-readable throughput threshold T_hum (tokens/s) and marks each
    run as usable or not.

    Args:
        summary_df: DataFrame from RunCollection.summary_df().
        prompt_df:  DataFrame from RunCollection.prompt_metrics_df(), used to
                    compute the per-model token/word ratio.
        min_words_per_s: Minimum legible generation speed in words/s.

    Returns:
        (summary_out, tok_per_word_df) where:
        - summary_out has new columns: tokens_per_word, T_hum, usable.
        - tok_per_word_df is a per-model summary with columns:
              model_short, tokens_per_word_mean, tokens_per_word_median,
              T_hum (tokens/s).
    """
    # Compute tokens-per-word ratio from prompts (ignore empty generations).
    valid = prompt_df[
        ~prompt_df["is_empty_generation"]
        & (prompt_df["n_words_answer"] > 0)
        & (prompt_df["eval_count"] > 0)
    ].copy()
    valid["_tpw"] = valid["eval_count"] / valid["n_words_answer"]

    tpw_stats = (
        valid.groupby("model_short")["_tpw"]
        .agg(tokens_per_word_mean="mean", tokens_per_word_median="median")
        .reset_index()
    )
    tpw_stats["T_hum"] = tpw_stats["tokens_per_word_mean"] * min_words_per_s

    # Global (all-model) fallback for runs where the model has no word data.
    global_tpw = valid["_tpw"].mean() if not valid.empty else 1.0
    global_T_hum = global_tpw * min_words_per_s

    out = summary_df.merge(
        tpw_stats[["model_short", "tokens_per_word_mean", "T_hum"]],
        on="model_short",
        how="left",
    )
    out["tokens_per_word"] = out["tokens_per_word_mean"].fillna(global_tpw)
    out["T_hum"] = out["T_hum"].fillna(global_T_hum)
    out["usable"] = out["tokens_per_s_mean"] >= out["T_hum"]
    out.drop(columns=["tokens_per_word_mean"], inplace=True)

    return out, tpw_stats
