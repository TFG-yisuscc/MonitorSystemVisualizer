"""Low-level plotting primitives. Each draws on a given Axes.

Convention: every function takes ``ax`` as the first parameter and returns it.
This makes them composable: notebooks can lay out their own grid of axes
and dispatch primitives onto each cell.
"""

from __future__ import annotations

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from .style import COLORS, MODEL_PALETTE


def _palette_for_hue(df: pd.DataFrame, hue: str | None) -> list[str] | None:
    """Return a truncated MODEL_PALETTE sized to the number of unique hue levels.

    Avoids the seaborn warning about having more palette entries than groups.
    Returns None when hue is None or exceeds palette length (seaborn picks its own).
    """
    if hue is None:
        return None
    n = df[hue].nunique()
    return MODEL_PALETTE[:n] if n <= len(MODEL_PALETTE) else None


# --- Hardware time series --------------------------------------------------

def plot_hw_line(
    ax: Axes,
    hw_df: pd.DataFrame,
    metric: str,
    *,
    hue: str | None = None,
    color: str | None = None,
    label: str | None = None,
) -> Axes:
    """Plot a hw metric vs t_rel_s on the given Axes.

    If ``hue`` is given (e.g. "model_short", "run_id"), draws one line per
    group with the project palette. Otherwise draws a single line with
    ``color`` (or sensible default).

    Caller is responsible for setting the x-axis label.
    """
    if hue is not None:
        sns.lineplot(
            data=hw_df, x="t_rel_s", y=metric,
            hue=hue, palette=_palette_for_hue(hw_df, hue), ax=ax,
        )
    else:
        ax.plot(
            hw_df["t_rel_s"], hw_df[metric],
            color=color or "C0", label=label,
        )
    return ax


def plot_freq_per_core(ax: Axes, freq_long_df: pd.DataFrame) -> Axes:
    """Per-core frequency with a line per core_id. Expects long format."""
    sns.lineplot(
        data=freq_long_df, x="t_rel_s", y="freq_ghz",
        hue="core_id", palette="tab10", ax=ax,
    )
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Frecuencia (GHz)")
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("Core")
    return ax


# --- Annotations on top of timelines --------------------------------------

def plot_prompt_spans(
    ax: Axes,
    prompt_df: pd.DataFrame,
    *,
    alpha: float = 0.10,
    color: str | None = None,
    annotate_ids: bool = False,
) -> Axes:
    """Shade thin horizontal bands at the top of the Axes for each prompt window.

    Less invasive than full-height bands. Uses the top 8% of the y-range so
    the main data lines stay visually clean.
    """
    if prompt_df.empty:
        return ax
    color = color or COLORS["prompt_span"]
    ymin, ymax = ax.get_ylim()
    band_height = (ymax - ymin) * 0.08
    band_bottom = ymax - band_height
    for _, row in prompt_df.iterrows():
        start = row["t_rel_s"]
        end = start + row["latency_ms"] / 1000.0
        ax.fill_between(
            [start, end],
            band_bottom, ymax,
            alpha=alpha + 0.2, color=color, zorder=1,
        )
        if annotate_ids:
            ax.text(
                (start + end) / 2, ymax,
                f"#{int(row['prompt_id'])}",
                ha="center", va="bottom", fontsize=9, color=color,
            )
    ax.set_ylim(ymin, ymax)
    return ax


def plot_prompt_lines(
    ax: Axes,
    prompt_df: pd.DataFrame,
    *,
    color: str | None = None,
    alpha: float = 0.4,
) -> Axes:
    """Draw thin vertical lines at the start and end of each prompt window.

    Minimal annotation: useful when the data lines themselves are dense and
    any shading would obscure them.
    """
    if prompt_df.empty:
        return ax
    color = color or COLORS["prompt_span"]
    for _, row in prompt_df.iterrows():
        start = row["t_rel_s"]
        end = start + row["latency_ms"] / 1000.0
        ax.axvline(start, color=color, ls="-", lw=1.2, alpha=alpha, zorder=1)
        ax.axvline(end, color=color, ls="--", lw=1.0, alpha=alpha, zorder=1)
    return ax


def plot_prompt_phases(
    ax: Axes,
    prompt_df: pd.DataFrame,
    *,
    band_position: str = "top",
    band_fraction: float = 0.10,
    alpha: float = 0.7,
) -> Axes:
    """Annotate prompt phases (load → prefill → decode) as colored sub-bands.

    Draws the phase bands in the top (or bottom) ``band_fraction`` of the
    y-range. Meant for overlaying phase info inside existing data panels.
    For a dedicated strip, use ``plot_phase_strip`` instead.

    Phases with zero duration are skipped.

    Required columns in ``prompt_df``:
      - t_rel_s
      - load_duration_ns
      - prompt_eval_duration_ns
      - eval_duration_ns
    """
    if prompt_df.empty:
        return ax
    if band_position not in {"top", "bottom"}:
        raise ValueError("band_position must be 'top' or 'bottom'")

    ymin, ymax = ax.get_ylim()
    band_height = (ymax - ymin) * band_fraction
    if band_position == "top":
        bottom = ymax - band_height
        top = ymax
    else:
        bottom = ymin
        top = ymin + band_height

    phase_specs = [
        ("load_duration_ns", COLORS["phase_load"]),
        ("prompt_eval_duration_ns", COLORS["phase_prefill"]),
        ("eval_duration_ns", COLORS["phase_decode"]),
    ]

    for _, row in prompt_df.iterrows():
        cursor = row["t_rel_s"]
        for col, color in phase_specs:
            dur_s = float(row[col]) / 1e9
            if dur_s <= 0:
                cursor += dur_s
                continue
            ax.fill_between(
                [cursor, cursor + dur_s],
                bottom, top,
                alpha=alpha, color=color, zorder=2, linewidth=0,
            )
            cursor += dur_s
    ax.set_ylim(ymin, ymax)
    return ax


def plot_phase_strip(
    ax: Axes,
    prompt_df: pd.DataFrame,
    *,
    annotate_durations: bool = True,
    annotate_prompt_ids: bool = True,
    min_visible_width_s: float = 0.0,
) -> Axes:
    """Render the load/prefill/decode phases of every prompt as colored bands
    that fill the entire height of the given Axes.

    Designed as a small dedicated strip above a hardware timeline, sharing
    the x-axis with it. Each band carries:
      - Its phase name ("load" / "prefill" / "decode") at the bottom.
      - Its duration ("12.0s") above the phase name, when wide enough.
      - The prompt_id ("#0", "#1") above the phase block.

    Phases with zero duration are skipped.

    Required columns in ``prompt_df``:
      - prompt_id
      - t_rel_s
      - load_duration_ns
      - prompt_eval_duration_ns
      - eval_duration_ns

    Parameters
    ----------
    annotate_durations : bool
        Show the duration as text inside the band when there's room.
    annotate_prompt_ids : bool
        Show "#0", "#1", ... above each prompt block.
    min_visible_width_s : float
        If > 0, force phases narrower than this to render at this width.
        Defaults to 0 (honest, no inflation).
    """
    if prompt_df.empty:
        return ax

    phase_specs = [
        ("load_duration_ns", COLORS["phase_load"], "load"),
        ("prompt_eval_duration_ns", COLORS["phase_prefill"], "prefill"),
        ("eval_duration_ns", COLORS["phase_decode"], "decode"),
    ]

    for _, row in prompt_df.iterrows():
        cursor = float(row["t_rel_s"])
        prompt_start = cursor
        prompt_id = int(row["prompt_id"])

        for col, color, label in phase_specs:
            dur_s = float(row[col]) / 1e9
            if dur_s <= 0:
                continue
            visible_w = max(dur_s, min_visible_width_s)
            ax.fill_between(
                [cursor, cursor + visible_w],
                0, 1,
                alpha=0.85, color=color, zorder=2, linewidth=0,
                transform=ax.get_xaxis_transform(),
            )
            band_center_x = cursor + visible_w / 2
            if visible_w >= 4.0:
                if annotate_durations:
                    ax.text(
                        band_center_x, 0.65,
                        f"{dur_s:.1f}s",
                        ha="center", va="center",
                        fontsize=9, color="black",
                        transform=ax.get_xaxis_transform(),
                    )
                ax.text(
                    band_center_x, 0.30,
                    label,
                    ha="center", va="center",
                    fontsize=8, color="black", style="italic",
                    transform=ax.get_xaxis_transform(),
                )
            elif visible_w >= 2.0:
                ax.text(
                    band_center_x, 0.5,
                    label,
                    ha="center", va="center",
                    fontsize=8, color="black",
                    transform=ax.get_xaxis_transform(),
                )
            # Else: too narrow — color alone conveys the information.
            cursor += dur_s

        if annotate_prompt_ids:
            ax.text(
                (prompt_start + cursor) / 2, 1.20,
                f"#{prompt_id}",
                ha="center", va="bottom",
                fontsize=10, color="#444444", fontweight="bold",
                transform=ax.get_xaxis_transform(),
            )

    ax.set_yticks([])
    ax.set_ylabel("Fases", rotation=0, ha="right", va="center", labelpad=15)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    return ax


def plot_throttle_markers(
    ax: Axes,
    hw_df: pd.DataFrame,
    *,
    color: str | None = None,
    alpha: float = 0.7,
) -> Axes:
    """Draw vertical lines at every sample where throttling was active."""
    color = color or COLORS["throttle"]
    active = hw_df[hw_df["throt_any_active"]]
    for t in active["t_rel_s"]:
        ax.axvline(t, color=color, lw=1.0, alpha=alpha, zorder=3)
    return ax


# --- Distributions of per-prompt metrics ----------------------------------

def plot_metric_bars(
    ax: Axes,
    prompt_df: pd.DataFrame,
    metric: str,
    *,
    hue: str | None = None,
    color: str | None = None,
) -> Axes:
    """Bar plot of a per-prompt metric (one bar per prompt_id)."""
    if hue is not None:
        sns.barplot(
            data=prompt_df, x="prompt_id", y=metric,
            hue=hue, palette=_palette_for_hue(prompt_df, hue), ax=ax,
        )
    else:
        sns.barplot(
            data=prompt_df, x="prompt_id", y=metric,
            color=color or "C0", ax=ax,
        )
    ax.set_xlabel("prompt_id")
    return ax


def plot_metric_distribution(
    ax: Axes,
    prompt_df: pd.DataFrame,
    metric: str,
    *,
    kind: str = "box",
    hue: str | None = None,
) -> Axes:
    """Distribution of a metric across prompts.

    ``kind`` in {"box", "violin", "ecdf"}.
    """
    palette = _palette_for_hue(prompt_df, hue)
    if kind == "box":
        sns.boxplot(
            data=prompt_df, x=hue, y=metric,
            hue=hue, palette=palette,
            legend=False, ax=ax,
        )
    elif kind == "violin":
        sns.violinplot(
            data=prompt_df, x=hue, y=metric,
            hue=hue, palette=palette,
            legend=False, ax=ax,
        )
    elif kind == "ecdf":
        sns.ecdfplot(
            data=prompt_df, x=metric,
            hue=hue, palette=palette,
            ax=ax,
        )
    else:
        raise ValueError(f"Unknown kind={kind!r}, expected box|violin|ecdf")
    return ax


# --- Tokens ----------------------------------------------------------------

def plot_logprob_by_position(
    ax: Axes,
    tokens_df: pd.DataFrame,
    *,
    hue: str = "prompt_id",
    alpha: float = 0.6,
) -> Axes:
    """Per-token logprob vs position in the response.

    By default colors by prompt_id (intra-run analysis). For cross-run
    analyses pass ``hue="run_id"`` or ``hue="model_short"``.
    """
    sns.lineplot(
        data=tokens_df, x="token_idx", y="logprob",
        hue=hue, palette=_palette_for_hue(tokens_df, hue), ax=ax, alpha=alpha,
    )
    ax.set_xlabel("posición del token")
    ax.set_ylabel("log-prob")
    ax.axhline(0, ls=":", color=COLORS["reference_line"], alpha=0.5, zorder=0)
    return ax


# --- Cross-run scatters ---------------------------------------------------

def plot_pareto(
    ax: Axes,
    summary_df: pd.DataFrame,
    *,
    x: str = "power_mean_w",
    y: str = "tokens_per_s_mean",
    hue: str = "model_short",
    style: str | None = None,
) -> Axes:
    """Scatter of two run-level metrics, suitable for Pareto-style charts."""
    sns.scatterplot(
        data=summary_df, x=x, y=y,
        hue=hue, style=style,
        palette=_palette_for_hue(summary_df, hue), ax=ax, s=80,
    )
    return ax


# --- Dual-axis correlation ------------------------------------------------

def plot_dual_axis(
    ax: Axes,
    df: pd.DataFrame,
    *,
    x: str,
    y_left: str,
    y_right: str,
    color_left: str | None = None,
    color_right: str | None = None,
    label_left: str | None = None,
    label_right: str | None = None,
) -> tuple[Axes, Axes]:
    """Plot two metrics on the same Axes with independent y-scales.

    Returns (ax_left, ax_right). The right Axes is created with twinx().
    Useful for correlating CPU% with memory%, temperature with power, etc.
    """
    color_left = color_left or "C0"
    color_right = color_right or "C3"
    ax.plot(df[x], df[y_left], color=color_left, label=label_left or y_left)
    ax.set_ylabel(label_left or y_left, color=color_left)
    ax.tick_params(axis="y", labelcolor=color_left)

    ax2 = ax.twinx()
    ax2.plot(df[x], df[y_right], color=color_right, label=label_right or y_right)
    ax2.set_ylabel(label_right or y_right, color=color_right)
    ax2.tick_params(axis="y", labelcolor=color_right)
    ax2.grid(False)

    return ax, ax2
