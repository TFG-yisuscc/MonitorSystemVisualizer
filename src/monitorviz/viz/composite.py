"""High-level composite plots that combine primitives into ready-to-show figures.

Notebooks should prefer these for the canonical visualizations of the project.
For ad-hoc tweaks, drop down to ``primitives``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from monitorviz.models.prompt import TokenProb
    from monitorviz.models.run import Run

from .primitives import (
    plot_dual_axis,
    plot_freq_per_core,
    plot_hw_line,
    plot_metric_bars,
    plot_pareto,
    plot_phase_strip,
    plot_prompt_lines,
    plot_prompt_phases,
    plot_prompt_spans,
    plot_throttle_markers,
)
from .style import COLORS

_VALID_ANNOTATIONS = {"strip", "phases", "spans", "lines", "none"}

_HW_DEFAULT_LABELS: dict[str, str] = {
    "temperature_c": "Temperatura (°C)",
    "internal_power_w": "Potencia (W)",
    "freq_mean_ghz": "Freq. media (GHz)",
    "cpu_usage_pct": "CPU (%)",
    "voltage_v": "Voltaje (V)",
    "fan_rpm": "Fan (RPM)",
    "mem_pct": "Memoria (%)",
    "swap_pct": "Swap (%)",
}

_HW_DEFAULT_COLORS: dict[str, str] = {
    "temperature_c": COLORS["temperature"],
    "internal_power_w": COLORS["power"],
    "freq_mean_ghz": COLORS["frequency"],
    "cpu_usage_pct": COLORS["cpu"],
    "voltage_v": COLORS["power"],
    "fan_rpm": COLORS["cpu"],
    "mem_pct": COLORS["memory"],
    "swap_pct": COLORS["swap"],
}


# --- Internal layout helper -----------------------------------------------

def _build_timeline_figure(
    hw_df: pd.DataFrame,
    prompt_df: pd.DataFrame | None,
    metrics: tuple[str, ...],
    metric_labels: dict[str, str],
    palette_for_metric: dict[str, str],
    *,
    show_throttle: bool,
    prompt_annotation: str,
    figsize: tuple[float, float],
    hardware_period_s: float = 0.5,
) -> Figure:
    """Internal: lay out a strip (optional) + N hw metric panels sharing X."""
    n_hw = len(metrics)
    use_strip = (
        prompt_annotation == "strip"
        and prompt_df is not None
        and not prompt_df.empty
    )

    ax_strip = None
    if use_strip:
        from matplotlib.gridspec import GridSpec
        # Use constrained_layout for GridSpec figures so tight_layout() does
        # not conflict with the manually-managed hspace ratios.
        fig = plt.figure(figsize=figsize, layout="constrained")
        gs = GridSpec(
            nrows=n_hw + 1, ncols=1,
            height_ratios=[0.6] + [3.0] * n_hw,
            hspace=0.15,
            figure=fig,
        )
        ax_strip = fig.add_subplot(gs[0])
        axes = [
            fig.add_subplot(gs[i + 1], sharex=ax_strip)
            for i in range(n_hw)
        ]
        plot_phase_strip(ax_strip, prompt_df)
        ax_strip.tick_params(axis="x", labelbottom=False)
    else:
        fig, axes = plt.subplots(n_hw, 1, figsize=figsize, sharex=True)
        if n_hw == 1:
            axes = [axes]

    for ax, metric in zip(axes, metrics, strict=False):
        color = palette_for_metric.get(metric, "C0")
        plot_hw_line(ax, hw_df, metric=metric, color=color)
        ax.set_xlabel("")
        ax.set_ylabel(metric_labels.get(metric, metric))

    # Hide x tick labels on intermediate panels (strip is already hidden above)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)

    if (
        prompt_df is not None
        and not prompt_df.empty
        and prompt_annotation in {"phases", "spans", "lines"}
    ):
        for ax in axes:
            if prompt_annotation == "phases":
                plot_prompt_phases(ax, prompt_df)
            elif prompt_annotation == "spans":
                plot_prompt_spans(ax, prompt_df)
            elif prompt_annotation == "lines":
                plot_prompt_lines(ax, prompt_df)

    # Throttle markers on every hw panel (but NOT the strip)
    if show_throttle and "throt_any_active" in hw_df.columns:
        for ax in axes:
            plot_throttle_markers(ax, hw_df, hardware_period_s=hardware_period_s)

    # Format x-axis: show in minutes if total duration > 1800s
    t_max = hw_df["t_rel_s"].max() if not hw_df.empty else 0
    if t_max > 1800:
        x_label = "Tiempo desde inicio del run (min)"
        for ax in axes:
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x/60:.0f}")
            )
        if ax_strip is not None:
            ax_strip.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x/60:.0f}")
            )
    else:
        x_label = "Tiempo desde inicio del run (s)"

    axes[-1].set_xlabel(x_label)
    return fig


# --- Hardware timeline (the flagship plot of the project) -----------------

def hw_timeline(
    hw_df: pd.DataFrame,
    prompt_df: pd.DataFrame | None = None,
    *,
    metrics: tuple[str, ...] = (
        "temperature_c", "internal_power_w", "freq_mean_ghz", "cpu_usage_pct"
    ),
    metric_labels: dict[str, str] | None = None,
    show_throttle: bool = True,
    prompt_annotation: str = "strip",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    hardware_period_s: float = 0.5,
) -> Figure:
    """Multi-panel hardware timeline with optional prompt annotations and
    throttling markers.

    Parameters
    ----------
    prompt_annotation : "strip" | "phases" | "spans" | "lines" | "none"
        How to overlay the prompt windows:
        - "strip": dedicated phase strip on top of the hw panels (default).
          Each prompt's load/prefill/decode are colored rectangles in their
          own panel, with duration labels and a legend.
        - "phases": colored sub-bands inside the top of every hw panel.
        - "spans": thin orange band on top of each hw panel.
        - "lines": vertical start/end lines.
        - "none": don't annotate prompts.
    """
    if prompt_annotation not in _VALID_ANNOTATIONS:
        raise ValueError(f"prompt_annotation must be one of {_VALID_ANNOTATIONS}")

    labels = {**_HW_DEFAULT_LABELS, **(metric_labels or {})}
    use_strip = (
        prompt_annotation == "strip"
        and prompt_df is not None
        and not prompt_df.empty
    )
    if figsize is None:
        figsize = (14, (1.0 if use_strip else 0) + 3.0 * len(metrics))

    fig = _build_timeline_figure(
        hw_df, prompt_df, metrics, labels, _HW_DEFAULT_COLORS,
        show_throttle=show_throttle,
        prompt_annotation=prompt_annotation,
        figsize=figsize,
        hardware_period_s=hardware_period_s,
    )
    if title:
        # With constrained_layout (strip mode), let the engine place the title.
        # Without it, nudge above the axes with y=1.02.
        fig.suptitle(title) if use_strip else fig.suptitle(title, y=1.02)
    if not use_strip:
        fig.tight_layout()
    return fig


def hw_freq_panel(
    freq_long_df: pd.DataFrame,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (13, 4),
) -> Figure:
    """Single panel showing per-core frequency over time."""
    fig, ax = plt.subplots(figsize=figsize)
    plot_freq_per_core(ax, freq_long_df)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


# --- Inference distributions ----------------------------------------------

def inference_summary_panel(
    prompt_df: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = (
        "latency_ms", "tokens_per_second", "perplexity",
    ),
    metric_labels: dict[str, str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Side-by-side bar plots of per-prompt metrics."""
    default_labels = {
        "latency_ms": "Latencia (s)",
        "tokens_per_second": "tokens/s",
        "words_per_second": "palabras/s",
        "perplexity": "Perplejidad",
        "time_to_first_token_ms": "TTFT (ms)",
    }
    labels = {**default_labels, **(metric_labels or {})}

    n = len(metrics)
    if figsize is None:
        figsize = (4.5 * n, 4)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics, strict=False):
        # Convert latency from ms to seconds for display
        if m == "latency_ms":
            df_to_plot = prompt_df.copy()
            df_to_plot["latency_ms"] = df_to_plot["latency_ms"] / 1000.0
            plot_metric_bars(ax, df_to_plot, metric=m)
        else:
            plot_metric_bars(ax, prompt_df, metric=m)
        ax.set_ylabel(labels.get(m, m))
        ax.set_title(labels.get(m, m))

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def logprob_panel(
    token_probs: list[TokenProb],
    prompt_id: int | None = None,
    *,
    figsize: tuple[float, float] = (14, 4),
    max_tokens: int = 80,
    title: str | None = None,
) -> Figure:
    """Log-probability per token.

    Works for both OLLAMA (token text available) and LLAMA (index only).
    When token text is available, labels the X axis with token strings.
    When not available (LLAMA), labels with token position index.
    """
    if not token_probs:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Sin datos de tokenProb",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    tps = token_probs[:max_tokens]
    logprobs = [tp.logprob for tp in tps]

    has_text = any(tp.token.strip() for tp in tps)

    fig, ax = plt.subplots(figsize=figsize)

    if has_text:
        labels = [
            tp.token.replace("\n", "↵").replace(" ", "·")[:12]
            for tp in tps
        ]
        x = range(len(tps))
        ax.bar(x, logprobs, color="tab:blue", alpha=0.7)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Token")
    else:
        x = list(range(len(tps)))
        ax.plot(x, logprobs, color="tab:orange", lw=1.5, marker=".", ms=4)
        ax.fill_between(x, logprobs, alpha=0.15, color="tab:orange")
        ax.set_xlabel("Posición del token (sin texto disponible — motor LLAMA)")

    ax.set_ylabel("Log-probabilidad")
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    default_title = "Log-probabilidad por token"
    if prompt_id is not None:
        default_title += f" — prompt {prompt_id}"
    ax.set_title(title or default_title)

    fig.tight_layout()
    return fig


# --- Cross-run comparisons (for the comparativa notebook) -----------------

def pareto_panel(
    summary_df: pd.DataFrame,
    *,
    x: str = "power_mean_w",
    y: str = "tokens_per_s_mean",
    hue: str = "model_short",
    style: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> Figure:
    """Scatter of two run-level metrics. Useful for energy-vs-throughput
    Pareto frontiers and similar comparisons."""
    fig, ax = plt.subplots(figsize=figsize)
    plot_pareto(ax, summary_df, x=x, y=y, hue=hue, style=style)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


# --- Memory and resource panels -------------------------------------------

def memory_timeline(
    hw_df: pd.DataFrame,
    prompt_df: pd.DataFrame | None = None,
    *,
    show_swap: bool = True,
    prompt_annotation: str = "strip",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    hardware_period_s: float = 0.5,
) -> Figure:
    """RAM (and optionally SWAP) usage over time."""
    if prompt_annotation not in _VALID_ANNOTATIONS:
        raise ValueError(f"prompt_annotation must be one of {_VALID_ANNOTATIONS}")

    metrics = ("mem_pct", "swap_pct") if show_swap else ("mem_pct",)
    metric_labels = {"mem_pct": "Memoria (%)", "swap_pct": "Swap (%)"}
    palette_for_metric = {
        "mem_pct": COLORS["memory"],
        "swap_pct": COLORS["swap"],
    }
    use_strip = (
        prompt_annotation == "strip"
        and prompt_df is not None
        and not prompt_df.empty
    )
    if figsize is None:
        figsize = (14, (1.0 if use_strip else 0) + 3.0 * len(metrics))

    fig = _build_timeline_figure(
        hw_df, prompt_df, metrics, metric_labels, palette_for_metric,
        show_throttle=False,
        prompt_annotation=prompt_annotation,
        figsize=figsize,
        hardware_period_s=hardware_period_s,
    )
    if title:
        fig.suptitle(title) if use_strip else fig.suptitle(title, y=1.02)
    if not use_strip:
        fig.tight_layout()
    return fig


def cpu_memory_dual(
    hw_df: pd.DataFrame,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (14, 5),
) -> Figure:
    """CPU% and memory% on the same time axis with twin y-axes.

    Useful to spot whether memory pressure precedes CPU saturation or vice versa.
    """
    fig, ax = plt.subplots(figsize=figsize)
    plot_dual_axis(
        ax, hw_df,
        x="t_rel_s",
        y_left="cpu_usage_pct",
        y_right="mem_pct",
        color_left=COLORS["cpu"],
        color_right=COLORS["memory"],
        label_left="CPU (%)",
        label_right="Memoria (%)",
    )
    ax.set_xlabel("Tiempo desde inicio del run (s)")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def temp_power_dual(
    hw_df: pd.DataFrame,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (14, 5),
) -> Figure:
    """Temperature and power on the same time axis with twin y-axes."""
    fig, ax = plt.subplots(figsize=figsize)
    plot_dual_axis(
        ax, hw_df,
        x="t_rel_s",
        y_left="temperature_c",
        y_right="internal_power_w",
        color_left=COLORS["temperature"],
        color_right=COLORS["power"],
        label_left="Temperatura (°C)",
        label_right="Potencia (W)",
    )
    ax.set_xlabel("Tiempo desde inicio del run (s)")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


# --- Internal phase-band helper -------------------------------------------

_PHASE_COLORS = {
    "load":    COLORS["phase_load"],
    "prefill": COLORS["phase_prefill"],
    "decode":  COLORS["phase_decode"],
}


def _add_phase_strip(ax: Axes, run: Run) -> None:
    """Overlay colored axvspan bands for each inference phase on an axes."""
    run_start_ns = run.summary.timestamp_run_start_ns
    for p in run.prompts:
        if p.is_empty_generation:
            continue
        load_start = (p.start_timestamp_ns - run_start_ns) / 1e9
        prefill_start = load_start + p.load_duration_ns / 1e9
        decode_start = prefill_start + p.prompt_eval_duration_ns / 1e9
        decode_end = decode_start + p.eval_duration_ns / 1e9
        for t0, t1, phase in [
            (load_start, prefill_start, "load"),
            (prefill_start, decode_start, "prefill"),
            (decode_start, decode_end, "decode"),
        ]:
            ax.axvspan(t0, t1, alpha=0.12, color=_PHASE_COLORS[phase], zorder=0)


# --- New composites (Benoit-Cattin 2020 / TFG Fig. 2.16) ------------------

def cpu_memory_dual_phases(
    hw_df: pd.DataFrame,
    run: Run,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (14, 5),
) -> Figure:
    """RAM (MB) and CPU (%) over time with vertical phase delimiters.

    Variant B of Fig. 2.16 (TFG parcial). Adds dashed vertical lines at
    load→prefill and prefill→decode transitions for each prompt.
    """
    mem_mb = hw_df["mem_used_bytes"] / 1e6

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    ax1.plot(hw_df["t_rel_s"], mem_mb,
             color=COLORS["memory"], lw=1.5, label="RAM (MB)")
    ax1.set_ylabel("RAM (MB)", color=COLORS["memory"])
    ax1.tick_params(axis="y", labelcolor=COLORS["memory"])

    ax2.plot(hw_df["t_rel_s"], hw_df["cpu_usage_pct"],
             color=COLORS["cpu"], lw=1.0, alpha=0.8, label="CPU (%)")
    ax2.set_ylabel("CPU (%)", color=COLORS["cpu"])
    ax2.tick_params(axis="y", labelcolor=COLORS["cpu"])
    ax2.set_ylim(0, 105)
    ax2.grid(False)

    run_start_ns = run.summary.timestamp_run_start_ns
    first = True
    for p in run.prompts:
        if p.is_empty_generation:
            continue
        load_start = (p.start_timestamp_ns - run_start_ns) / 1e9
        prefill_start = load_start + p.load_duration_ns / 1e9
        decode_start = prefill_start + p.prompt_eval_duration_ns / 1e9
        decode_end = decode_start + p.eval_duration_ns / 1e9
        for t, color, label in [
            (prefill_start, _PHASE_COLORS["prefill"], "prefill" if first else None),
            (decode_start,  _PHASE_COLORS["decode"],  "decode"  if first else None),
            (decode_end,    "gray",                   "fin"     if first else None),
        ]:
            ax1.axvline(t, color=color, ls="--", lw=1.2, alpha=0.8, label=label)
        first = False

    ax1.set_xlabel("Tiempo desde inicio del run (s)")
    ax1.set_title(title or f"RAM y CPU (fases) — {run.model_short}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    fig.tight_layout()
    return fig


def temp_freq_dual(
    hw_df: pd.DataFrame,
    run: Run,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (14, 5),
    throttle_line: float = 80.0,
) -> Figure:
    """CPU temperature (°C) and mean CPU frequency (GHz) over time.

    Dual Y-axis layout. Equivalent to Benoit-Cattin et al. (2020) Fig. 2(i).
    Adds a horizontal dashed line at throttle_line °C and a phase strip.
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    ax1.plot(hw_df["t_rel_s"], hw_df["temperature_c"],
             color=COLORS["temperature"], lw=1.5, label="Temp (°C)")
    ax1.axhline(throttle_line, color=COLORS["temperature"], ls="--", lw=1.0,
                alpha=0.6, label=f"{throttle_line} °C")
    ax1.set_ylabel("Temperatura (°C)", color=COLORS["temperature"])
    ax1.tick_params(axis="y", labelcolor=COLORS["temperature"])

    freq_cols = [c for c in hw_df.columns
                 if "freq" in c.lower() and "ghz" in c.lower()
                 and "min" not in c.lower() and "max" not in c.lower()]
    freq_col = "freq_mean_ghz" if "freq_mean_ghz" in hw_df.columns else (
        freq_cols[0] if freq_cols else None
    )

    if freq_col:
        freq_series = hw_df[freq_col].round(3)
        ax2.plot(hw_df["t_rel_s"], freq_series,
                 color=COLORS["frequency"], lw=1.2, alpha=0.8, label="Freq (GHz)")
        ax2.axhline(2.4, color=COLORS["frequency"], ls="--", lw=0.8,
                    alpha=0.5, label="2.4 GHz nom.")
        ax2.set_ylabel("Frecuencia (GHz)", color=COLORS["frequency"])
        ax2.tick_params(axis="y", labelcolor=COLORS["frequency"])
        ax2.set_ylim(bottom=max(0.0, freq_series.min() - 0.1), top=2.5)
        ax2.grid(False)
    else:
        ax2.set_visible(False)

    _add_phase_strip(ax1, run)

    ax1.set_xlabel("Tiempo desde inicio del run (s)")
    ax1.set_title(title or f"Temperatura y frecuencia — {run.model_short}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = (ax2.get_legend_handles_labels() if freq_col else ([], []))
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    fig.tight_layout()
    return fig


def hw_distributions_panel(
    hw_df: pd.DataFrame,
    run: Run,
    *,
    figsize: tuple[float, float] = (16, 4),
    title: str | None = None,
    bins: int = 30,
) -> Figure:
    """Four histograms of key hw metrics during inference.

    Layout: [CPU Temp] [CPU Freq] [CPU %] [Power]
    Equivalent to Benoit-Cattin et al. (2020) Fig. 2(iii).
    Uses only samples where cpu_usage_pct > 10 to exclude idle periods.
    """
    active = hw_df[hw_df["cpu_usage_pct"] > 10].copy()
    if active.empty:
        active = hw_df.copy()

    freq_col = "freq_mean_ghz" if "freq_mean_ghz" in active.columns else next(
        (c for c in active.columns if "freq" in c and "ghz" in c), None
    )

    metrics: list[tuple[str, str, str]] = []
    if "temperature_c" in active.columns:
        metrics.append(("temperature_c",   "Temperatura (°C)", COLORS["temperature"]))
    if freq_col:
        metrics.append((freq_col,           "Frecuencia (GHz)", COLORS["frequency"]))
    if "cpu_usage_pct" in active.columns:
        metrics.append(("cpu_usage_pct",    "CPU (%)",          COLORS["cpu"]))
    if "internal_power_w" in active.columns:
        metrics.append(("internal_power_w", "Potencia (W)",     COLORS["power"]))

    n = len(metrics)
    if n == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, (col, xlabel, color) in zip(axes, metrics, strict=False):
        data = active[col].dropna()
        ax.hist(data, bins=bins, color=color, alpha=0.7, edgecolor="white")
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x = np.linspace(data.min(), data.max(), 200)
            ax_kde = ax.twinx()
            ax_kde.plot(x, kde(x), color=color, lw=2)
            ax_kde.set_yticks([])
            ax_kde.grid(False)
        except Exception:
            pass
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Muestras")
        ax.set_title(f"mean={data.mean():.1f}  std={data.std():.1f}")

    fig.suptitle(
        title or f"Distribución de métricas hw — {run.model_short}",
        y=1.02,
    )
    fig.tight_layout()
    return fig


# --- New visualization functions for parametric analysis and advanced insights ---

def sensitivity_curve(
    summary_df: pd.DataFrame,
    param_col: str,
    metrics: list[tuple[str, str]],
    *,
    group_col: str | None = None,
    figsize: tuple[float, float] = (14, 5),
    title: str | None = None,
    param_label: str | None = None,
    log_x: bool = False,
) -> Figure:
    """Sensitivity analysis curve: metric vs experimental parameter.

    Args:
        summary_df: aggregated summary, one row per run.
        param_col: column name of the experimental parameter (X axis),
                   e.g. "context_size", "batch_size", "bits_per_weight".
        metrics: list of (column, label) tuples — one subplot per metric.
        group_col: optional column to draw separate lines (e.g., "model_label"
                   when comparing several models in the same experiment).
        log_x: use logarithmic X axis (useful for batch_size or context_size).

    Returns:
        Matplotlib Figure with one subplot per metric.
    """
    # Handle empty dataframe or missing param_col
    if summary_df.empty or param_col not in summary_df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Sin datos para esta sensibilidad",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    data = summary_df.dropna(subset=[param_col]).copy()
    if data.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Sin datos para esta sensibilidad",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    n = len(metrics)
    ncols = min(n, 2)
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize[0], figsize[1] * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax, (col, ylabel) in zip(axes_flat, metrics):
        if col not in data.columns:
            ax.text(0.5, 0.5, f"Sin columna {col}",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        sub = data.dropna(subset=[col])
        if sub.empty:
            ax.text(0.5, 0.5, f"Sin datos para {col}",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        if group_col and group_col in sub.columns:
            groups = sorted(sub[group_col].dropna().unique())
            for grp in groups:
                gsub = sub[sub[group_col] == grp].sort_values(param_col)
                ax.plot(gsub[param_col], gsub[col],
                        "o-", lw=1.8, ms=8, label=str(grp))
            ax.legend(loc="best", fontsize=9)
        else:
            sub_sorted = sub.sort_values(param_col)
            ax.plot(sub_sorted[param_col], sub_sorted[col],
                    "o-", lw=1.8, ms=8, color="tab:blue")

        ax.set_xlabel(param_label or param_col)
        ax.set_ylabel(ylabel)
        if log_x:
            ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if title:
        fig.suptitle(title, y=1.01, fontsize=13)
    fig.tight_layout()
    return fig


def pareto_panel_multi(
    summary_df: pd.DataFrame,
    pairs: list[tuple[str, str, str, str, bool, bool]],
    *,
    label_col: str = "model_label",
    figsize: tuple[float, float] = (7, 5),
    title: str | None = None,
) -> Figure:
    """Multiple Pareto trade-off scatter plots.

    Args:
        pairs: list of (x_col, y_col, x_label, y_label,
                        x_lower_is_better, y_lower_is_better) tuples.
        label_col: column to annotate each point.
    """
    n = len(pairs)
    ncols = min(n, 2)
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, squeeze=False,
        figsize=(figsize[0] * ncols, figsize[1] * nrows),
    )
    axes_flat = axes.flatten()

    for ax, (xc, yc, xl, yl, x_lower, y_lower) in zip(axes_flat, pairs):
        data = summary_df.dropna(subset=[xc, yc]).copy()
        if data.empty:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        ax.scatter(data[xc], data[yc], s=120, alpha=0.85,
                   color="tab:blue", edgecolor="black", zorder=3)
        for _, row in data.iterrows():
            ax.annotate(str(row[label_col]),
                        (row[xc], row[yc]),
                        xytext=(6, 4), textcoords="offset points",
                        fontsize=9)

        # Compute Pareto front
        # For each point, check if any other dominates it
        pts = data[[xc, yc]].to_numpy()
        n_pts = len(pts)
        is_dominated = np.zeros(n_pts, dtype=bool)
        for i in range(n_pts):
            for j in range(n_pts):
                if i == j:
                    continue
                x_better = ((pts[j, 0] <= pts[i, 0]) if x_lower
                            else (pts[j, 0] >= pts[i, 0]))
                y_better = ((pts[j, 1] <= pts[i, 1]) if y_lower
                            else (pts[j, 1] >= pts[i, 1]))
                strict = (
                    (pts[j, 0] != pts[i, 0]) or (pts[j, 1] != pts[i, 1])
                )
                if x_better and y_better and strict:
                    is_dominated[i] = True
                    break

        front = data[~is_dominated].sort_values(xc)
        ax.plot(front[xc], front[yc],
                color="tab:red", lw=1.5, ls="--",
                label="Frontera Pareto", zorder=2)

        ax.set_xlabel(xl + (" ↓ mejor" if x_lower else " ↑ mejor"))
        ax.set_ylabel(yl + (" ↓ mejor" if y_lower else " ↑ mejor"))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if title:
        fig.suptitle(title, y=1.01, fontsize=13)
    fig.tight_layout()
    return fig


def throughput_over_time(
    run: Run,
    *,
    figsize: tuple[float, float] = (12, 5),
    max_prompts: int = 5,
    title: str | None = None,
) -> Figure:
    """Cumulative tokens generated over time, one curve per prompt.

    For each non-empty prompt, plots cumulative token count vs elapsed
    decode time. Slope = instantaneous tokens/s. A flattening curve
    indicates throttling or KV cache slowdown.

    Args:
        run: Run object with prompts and tokens.
        max_prompts: limit number of prompts plotted to avoid clutter.
    """
    fig, ax = plt.subplots(figsize=figsize)

    plotted = 0
    for p in run.prompts:
        if p.is_empty_generation or not p.tokens:
            continue
        if plotted >= max_prompts:
            break
        n_tokens = len(p.tokens)
        eval_s = p.eval_duration_ns / 1e9
        if eval_s <= 0 or n_tokens == 0:
            continue
        # Assume uniform distribution of tokens over decode time (approx)
        t = np.linspace(0, eval_s, n_tokens + 1)
        cum = np.arange(0, n_tokens + 1)
        ax.plot(t, cum, lw=1.8, marker=".", ms=3,
                label=f"prompt {p.prompt_id} ({n_tokens} tok)")
        plotted += 1

    ax.set_xlabel("Tiempo de decode (s)")
    ax.set_ylabel("Tokens acumulados generados")
    ax.set_title(title or f"Generación acumulada — {run.model_short}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def phase_breakdown_stacked(
    summary_df: pd.DataFrame,
    *,
    label_col: str = "model_label",
    figsize: tuple[float, float] = (12, 5),
    title: str | None = None,
    normalized: bool = True,
) -> Figure:
    """Stacked bar of load/prefill/decode time per model.

    Args:
        normalized: if True, bars sum to 100%. If False, absolute seconds.
    """
    cols = ["load_s_mean", "prefill_s_mean", "decode_s_mean"]
    available = [c for c in cols if c in summary_df.columns]
    if not available:
        # Fallback: derive from durations
        df = summary_df.copy()
        df["load_s_mean"] = df.get("load_duration_ns_mean", 0) / 1e9
        df["prefill_s_mean"] = df.get(
            "prompt_eval_duration_ns_mean", 0) / 1e9
        df["decode_s_mean"] = df.get("eval_duration_ns_mean", 0) / 1e9
        available = cols
    else:
        df = summary_df.copy()

    data = df.dropna(subset=available).copy()
    if data.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Sin datos de fases",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    if normalized:
        total = data[available].sum(axis=1)
        for c in available:
            data[c] = data[c] / total * 100
        ylabel = "Proporción de tiempo (%)"
    else:
        ylabel = "Tiempo (s)"

    phase_labels = {"load_s_mean": "Load",
                    "prefill_s_mean": "Prefill",
                    "decode_s_mean": "Decode"}
    colors = {"load_s_mean": "#bcbd22",
              "prefill_s_mean": "#17becf",
              "decode_s_mean": "#ff7f0e"}

    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(data))
    for c in available:
        ax.bar(data[label_col], data[c],
               bottom=bottom, label=phase_labels[c], color=colors[c])
        bottom += data[c].to_numpy()

    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title or
                 ("Desglose de fases (proporción)" if normalized
                  else "Desglose de fases (absoluto)"))
    ax.legend(title="Fase", loc="upper right")
    if normalized:
        ax.set_ylim(0, 105)
    fig.tight_layout()
    return fig


def throttling_heatmap(
    hw_df: pd.DataFrame,
    runs: list[Run],
    *,
    bins: int = 60,
    figsize: tuple[float, float] = (12, 5),
    title: str | None = None,
) -> Figure:
    """Heatmap of throttling events across runs.

    Each row is a run, each column is a time bin (1/bins of total run duration).
    Cell color: 1.0 if any throttling flag was active in that bin, 0.0 otherwise.
    """
    rows = []
    labels = []
    for r in runs:
        sub = hw_df[hw_df["run_id"] == r.run_id]
        if sub.empty:
            continue
        # Compute "any throttling active" boolean per sample
        throttle_cols = [c for c in sub.columns
                         if c.startswith("throt_")
                         and not c.endswith("_occurred")]
        if not throttle_cols:
            continue
        sub = sub.copy()
        sub["any_throt"] = sub[throttle_cols].any(axis=1).astype(int)
        # Bin by time
        t_max = sub["t_rel_s"].max()
        bin_edges = np.linspace(0, t_max, bins + 1)
        idx = np.digitize(sub["t_rel_s"], bin_edges) - 1
        idx = np.clip(idx, 0, bins - 1)
        row = np.zeros(bins)
        for i, v in zip(idx, sub["any_throt"]):
            row[i] = max(row[i], v)
        rows.append(row)
        fan_tag = "fan" if r.meta.fan else "no-fan"
        labels.append(f"{getattr(r, 'model_label', r.model_short)} ({fan_tag})")

    if not rows:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Sin datos de throttling",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    matrix = np.array(rows)
    fig, ax = plt.subplots(figsize=(figsize[0],
                                    max(3, 0.4 * len(labels))))
    im = ax.imshow(matrix, aspect="auto", cmap="Reds",
                   interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticks([0, bins // 2, bins - 1])
    ax.set_xticklabels(["inicio", "mitad", "fin"], fontsize=9)
    ax.set_xlabel("Tiempo del run (normalizado)")
    ax.set_title(title or "Eventos de throttling por run")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Throttling activo", fontsize=9)
    fig.tight_layout()
    return fig


def correlation_heatmap(
    summary_df: pd.DataFrame,
    *,
    cols: list[str] | None = None,
    method: str = "pearson",
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Pearson correlation heatmap between key performance metrics.

    Args:
        cols: list of column names; if None, uses a sensible default set.
        method: "pearson", "spearman", or "kendall".
        figsize: (width, height) in inches; auto-sized if None.
    """
    try:
        import seaborn as sns
    except ImportError:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5,
                "Seaborn no instalado.\n"
                "Instala con: uv pip install seaborn",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    _ALIASES = {
        "tokens_per_s_mean":   "tok/s",
        "ttft_ms_mean":        "TTFT",
        "latency_ms_mean":     "latencia",
        "perplexity_geomean":  "PPL",
        "temp_max_c":          "temp_máx",
        "temp_mean_c":         "temp_med",
        "power_mean_w":        "W_med",
        "power_max_w":         "W_máx",
        "energy_per_token_j":  "J/tok",
        "mbu_corr":            "MBU_corr",
        "throttled_ratio":     "throttle",
    }
    default_cols = [
        "tokens_per_s_mean", "ttft_ms_mean", "latency_ms_mean",
        "perplexity_geomean", "temp_max_c", "temp_mean_c",
        "power_mean_w", "power_max_w", "energy_per_token_j",
        "mbu_corr", "throttled_ratio",
    ]
    use_cols = cols or [c for c in default_cols if c in summary_df.columns]
    data = summary_df[use_cols].dropna(how="all")
    if data.empty or len(data) < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sin datos suficientes (≥2 runs)",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    corr = data.corr(method=method)
    corr.columns = [_ALIASES.get(c, c) for c in corr.columns]
    corr.index = [_ALIASES.get(c, c) for c in corr.index]

    n = len(corr)
    auto_size = max(10, n * 0.9)
    fs = figsize or (auto_size, auto_size * 0.85)

    fig, ax = plt.subplots(figsize=fs)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1, vmax=1, center=0,
        square=True,
        linewidths=0.4,
        annot_kws={"size": max(7, 10 - n // 3)},
        ax=ax,
        cbar_kws={"shrink": 0.7, "label": f"r ({method})"},
    )
    ax.set_title(
        title or f"Correlación {method} entre métricas (n={len(data)} runs)",
        pad=14, fontsize=12,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    fig.tight_layout()
    return fig


