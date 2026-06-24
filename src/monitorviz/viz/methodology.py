"""Publication-quality methodology figures for the TFG dissertation.

Three figures:
  1. plot_monitor_architecture() — MonitorSystem block diagram.
  2. plot_fom_diagram()          — FoM conceptual diagram (FoM_full & FoM_red).
  3. plot_experiment_matrix()    — Visual experimental matrix (E0–E5).

All functions return a (fig, axes) tuple and optionally save to disk.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

_C_INFERENCE = "#4878d0"
_C_MONITOR   = "#ee854a"
_C_ARTIFACT  = "#6acc65"
_C_SYNC      = "#d65f5f"
_C_NEUTRAL   = "#e8e8e8"

_C_AXIS_T    = "#4878d0"
_C_AXIS_MBU  = "#ee854a"
_C_AXIS_ETA  = "#6acc65"
_C_AXIS_EPS  = "#956cb4"
_C_FOM_FULL  = "#d65f5f"
_C_FOM_RED   = "#2e75b6"


def _save(fig: plt.Figure, path, fmt) -> None:
    if path is None:
        return
    fmts = (fmt,) if isinstance(fmt, str) else fmt
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    for f in fmts:
        fig.savefig(p.with_suffix(f".{f}"), format=f, bbox_inches="tight", dpi=150)


def _box(ax, x, y, w, h, text, color, alpha=0.22, fs=9, **kw):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.03",
        linewidth=1.4,
        edgecolor=color,
        facecolor=color,
        alpha=alpha,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fs, color="black", **kw)


def _arrow(ax, x0, y0, x1, y1, color="#555555", lw=1.3, style="->"):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                        connectionstyle="arc3,rad=0.0"),
    )


# ---------------------------------------------------------------------------
# Figure 1 — MonitorSystem architecture
# ---------------------------------------------------------------------------

def plot_monitor_architecture(
    save_path=None,
    fmt="pdf",
) -> tuple[plt.Figure, plt.Axes]:
    """Block diagram of the MonitorSystem C++ binary."""
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.suptitle("Arquitectura de MonitorSystem (C++)", y=0.97,
                 fontsize=14, fontweight="bold")

    LX, RX = 3.2, 9.8      # column centers
    BW = 4.2                # box width
    BH = 0.70               # box height
    FS = 9

    # ── Inference thread ────────────────────────────────────────────────────
    ax.text(LX, 8.5, "Hilo de inferencia", ha="center", fontsize=11,
            fontweight="bold", color=_C_INFERENCE)

    steps_l = [
        (8.0, "Carga de prompts (JSONL)"),
        (6.9, "Motor de inferencia\nllama.cpp  /  Ollama  /  hailo-ollama"),
        (5.65, "Prompt N → eval_count, tokens\ntiempos (ns), tokenProb (logprobs)"),
        (4.4, "¿Más prompts?"),
        (3.1, "prompt_metrics_*.jsonl\n(una línea por prompt → append)"),
    ]
    for y, label in steps_l:
        _box(ax, LX, y, BW, BH, label, _C_INFERENCE, fs=FS)

    for (y0, _), (y1, _) in zip(steps_l, steps_l[1:]):
        if y0 > 4.4:
            _arrow(ax, LX, y0 - BH / 2, LX, y1 + BH / 2, _C_INFERENCE)

    # "sí" loop-back arrow
    _arrow(ax, LX, 4.4 - BH / 2, LX, 3.1 + BH / 2, _C_INFERENCE)
    for x0, x1, y in [(LX - BW / 2, 1.1, 4.4), (1.1, 1.1, 8.0)]:
        ax.plot([x0, x1], [y, y], color=_C_INFERENCE, lw=1.3)
    ax.plot([1.1, 1.1], [8.0, 4.4], color=_C_INFERENCE, lw=1.3)
    _arrow(ax, 1.1, 8.0, LX - BW / 2, 8.0, _C_INFERENCE)
    ax.text(0.75, 6.2, "sí", ha="center", fontsize=9,
            color=_C_INFERENCE, style="italic")

    # ── Monitor thread ───────────────────────────────────────────────────────
    ax.text(RX, 8.5, "Hilo de monitorización", ha="center", fontsize=11,
            fontweight="bold", color=_C_MONITOR)

    steps_r = [
        (8.0, f"Timer periódico (hardware_period ≈ 0.25 s)"),
        (6.9, "Lectura sensores BCM2712\nT°,  freq/core,  V,  P,  RAM,  CPU%,  throttle"),
        (5.65, "timestamp compartido (ms)\nsync con hilo inferencia"),
        (4.4, "hw_metrics_*.jsonl\n(una línea por muestra → append)"),
    ]
    for y, label in steps_r:
        c = _C_SYNC if "sync" in label else _C_MONITOR
        _box(ax, RX, y, BW, BH, label, c, fs=FS)

    for (y0, _), (y1, _) in zip(steps_r, steps_r[1:]):
        _arrow(ax, RX, y0 - BH / 2, RX, y1 + BH / 2, _C_MONITOR)

    # Sync arrow between columns
    ax.annotate(
        "", xy=(RX - BW / 2, 5.65), xytext=(LX + BW / 2, 5.65),
        arrowprops=dict(arrowstyle="<->", color=_C_SYNC, lw=1.4,
                        linestyle="dashed"),
    )
    ax.text(6.5, 5.82, "sync ts", ha="center", fontsize=8,
            color=_C_SYNC, style="italic")

    # ── Output artefacts ─────────────────────────────────────────────────────
    ax.text(6.5, 2.25, "Artefactos generados por ejecución",
            ha="center", fontsize=10, fontweight="bold")

    arts = [
        (2.5,  "prompt_metrics_*.jsonl"),
        (6.5,  "hw_metrics_*.jsonl"),
        (10.5, "resumen.json"),
    ]
    for xc, label in arts:
        _box(ax, xc, 1.55, 3.4, 0.75, label, _C_ARTIFACT, alpha=0.30, fs=9)

    _arrow(ax, LX, 3.1 - BH / 2, 2.5, 1.55 + 0.375, _C_ARTIFACT)
    _arrow(ax, RX, 4.4 - BH / 2, 6.5, 1.55 + 0.375, _C_ARTIFACT)
    _arrow(ax, 6.5, 4.0, 10.5, 1.55 + 0.375, "#888888")
    ax.text(8.7, 3.2, "al finalizar\nel run", ha="center", fontsize=8,
            color="#888888", style="italic")

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor=_C_INFERENCE, alpha=0.4, label="Inferencia"),
        mpatches.Patch(facecolor=_C_MONITOR,   alpha=0.4, label="Monitorización"),
        mpatches.Patch(facecolor=_C_ARTIFACT,  alpha=0.4, label="Artefactos"),
        mpatches.Patch(facecolor=_C_SYNC,      alpha=0.4, label="Sincronización"),
    ]
    ax.legend(handles=legend_items, loc="lower right",
              frameon=True, fontsize=8, ncol=2, framealpha=0.9)

    fig.tight_layout()
    _save(fig, save_path, fmt)
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 2 — FoM conceptual diagram
# ---------------------------------------------------------------------------

def plot_fom_diagram(
    save_path=None,
    fmt="pdf",
) -> tuple[plt.Figure, plt.Axes]:
    """Conceptual diagram of FoM_full and FoM_red."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    fig.suptitle("Figuras de Mérito (FoM)", y=0.99,
                 fontsize=14, fontweight="bold")

    FS = 9
    BH = 0.62

    # ── FoM_full (left, x 0–8.5) ────────────────────────────────────────────
    ax.text(4.0, 7.0, "FoM_full  —  series E0–E4", ha="center",
            fontsize=11, fontweight="bold", color=_C_FOM_FULL)

    axes_full = [
        (r"$T_{norm}$ = N · T / (N$_{ref}$ · T$_{ref}$)",          _C_AXIS_T,   5.5),
        (r"$MBU_{norm}$ = MBU$_{corr}$ / MBU$_{ref}$",             _C_AXIS_MBU, 4.6),
        (r"$\eta_{norm}$ = $\eta_{CPU}$ / $\eta_{ref}$",            _C_AXIS_ETA, 3.7),
        (r"$\varepsilon_{norm}$ = N · ε / (N$_{ref}$ · ε$_{ref}$)", _C_AXIS_EPS, 2.8),
    ]
    AX_W = 4.8
    AX_CX = 2.6
    GEO_CX = 6.0
    GEO_W = 1.4
    GEO_H = 3.6
    RES_CX = 7.6
    RES_W = 1.4

    for label, color, yc in axes_full:
        _box(ax, AX_CX, yc, AX_W, BH, label, color, fs=FS)
        _arrow(ax, AX_CX + AX_W / 2, yc, GEO_CX - GEO_W / 2, yc, color)

    # Geomean box
    geo_y = (axes_full[0][2] + axes_full[-1][2]) / 2
    _box(ax, GEO_CX, geo_y, GEO_W, GEO_H,
         "Media\ngeo-\nmétrica\n$^{1/4}$",
         _C_FOM_FULL, alpha=0.28, fs=9, fontweight="bold")
    _arrow(ax, GEO_CX + GEO_W / 2, geo_y, RES_CX - RES_W / 2, geo_y,
           _C_FOM_FULL, lw=1.8)
    _box(ax, RES_CX, geo_y, RES_W, 0.72,
         "FoM_full", _C_FOM_FULL, alpha=0.50, fs=10, fontweight="bold")

    # Weighted note
    ax.text(4.0, 2.05,
            r"Variante ponderada: $\prod_i x_i^{w_i}$  (pesos $w_i$ opcionales)",
            ha="center", fontsize=8, color="#666666", style="italic")

    # Formula
    ax.text(4.0, 6.3,
            r"FoM_full $= (T_{norm} \cdot MBU_{norm}"
            r" \cdot \eta_{norm} \cdot \varepsilon_{norm})^{1/4}$",
            ha="center", fontsize=10, color=_C_FOM_FULL)

    # ── Divider ─────────────────────────────────────────────────────────────
    ax.axvline(8.9, ymin=0.08, ymax=0.97, color="#cccccc", lw=1.2, ls="--")

    # ── FoM_red (right, x 8.9–14) ───────────────────────────────────────────
    ax.text(11.5, 7.0, "FoM_red  —  serie E5", ha="center",
            fontsize=11, fontweight="bold", color=_C_FOM_RED)

    axes_red = [
        (r"$T_{5,norm}$ = T / T$_{CPU,m}$",           _C_AXIS_T,   5.2),
        (r"$\varepsilon_{5,norm}$ = ε / ε$_{CPU,m}$", _C_AXIS_EPS, 4.1),
    ]
    RAX_W = 3.6
    RAX_CX = 10.6
    RGEO_CX = 12.7
    RGEO_W = 1.2
    RGEO_H = 1.9
    RRES_CX = RGEO_CX   # result box below geomean
    RRES_Y  = 2.8

    geo_ry = (axes_red[0][2] + axes_red[-1][2]) / 2

    for label, color, yc in axes_red:
        _box(ax, RAX_CX, yc, RAX_W, BH, label, color, fs=FS)
        _arrow(ax, RAX_CX + RAX_W / 2, yc, RGEO_CX - RGEO_W / 2, yc, color)

    _box(ax, RGEO_CX, geo_ry, RGEO_W, RGEO_H,
         "Media\ngeo-\n$^{1/2}$",
         _C_FOM_RED, alpha=0.28, fs=9, fontweight="bold")
    _arrow(ax, RGEO_CX, geo_ry - RGEO_H / 2, RGEO_CX, RRES_Y + 0.38,
           _C_FOM_RED, lw=1.8)
    _box(ax, RRES_CX, RRES_Y, 2.0, 0.72,
         "FoM_red", _C_FOM_RED, alpha=0.50, fs=10, fontweight="bold")

    # Normalisation note
    ax.text(11.5, 6.3,
            r"FoM_red $= \sqrt{T_{5,norm} \cdot \varepsilon_{5,norm}}$",
            ha="center", fontsize=10, color=_C_FOM_RED)
    ax.text(11.5, 2.05,
            "Normalización intra-modelo\nvs. run en CPU (mismo modelo)",
            ha="center", fontsize=8, color="#666666", style="italic")

    # ── Usability criterion ──────────────────────────────────────────────────
    _box(ax, 7.0, 0.85, 13.0, 0.82,
         "Criterio de usabilidad:    usable = (T ≥ T_hum)\n"
         r"    T_hum = (tokens/palabra)$_m$  ×  3 palabras/s    "
         "(medido empíricamente por modelo)",
         "#888888", alpha=0.12, fs=9)

    fig.tight_layout()
    _save(fig, save_path, fmt)
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 3 — Visual experimental matrix
# ---------------------------------------------------------------------------

def plot_experiment_matrix(
    data=None,
    save_path=None,
    fmt="pdf",
) -> tuple[plt.Figure, plt.Axes]:
    """Visual experimental matrix table for E0–E5.

    ``data`` accepts two formats:
    * ``list[list[str]]`` — raw rows; up to 4 columns are mapped to
      [Serie, Variable independiente, Configuraciones evaluadas, Métricas].
    * ``list[dict]`` — dicts with any key subset of the default columns.
    * ``None`` — uses the built-in placeholder data.
    """
    _DEFAULT = [
        {"Serie": "E0", "Variable independiente": "Línea base (ventilador ON/OFF)",
         "Configuraciones evaluadas": "4 modelos × {OLLAMA, LLAMA} × {FAN, NOFAN}",
         "Métricas clave": "T, MBU_corr, η, ε, PPL, FoM_full"},
        {"Serie": "E1", "Variable independiente": "Cuantización",
         "Configuraciones evaluadas": "Todos × {Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q8_0}",
         "Métricas clave": "T, MBU_corr, PPL"},
        {"Serie": "E2", "Variable independiente": "Tamaño de contexto",
         "Configuraciones evaluadas": "Llama-3.2-1B × ctx ∈ {512, 1024, 2048, 4096, 5120}",
         "Métricas clave": "T, TTFT, MBU_corr, FoM_full"},
        {"Serie": "E3", "Variable independiente": "Batch size",
         "Configuraciones evaluadas": "Llama-3.2-1B × batch ∈ {128, 256, 512, 1024, 2048}",
         "Métricas clave": "T, η, ε, FoM_full"},
        {"Serie": "E4", "Variable independiente": "Motor de inferencia",
         "Configuraciones evaluadas": "Todos × {OLLAMA, LLAMA.cpp}",
         "Métricas clave": "T, MBU_corr, FoM_full"},
        {"Serie": "E5", "Variable independiente": "Acelerador hardware (Hailo-8L)",
         "Configuraciones evaluadas": "Modelos seleccionados × {CPU, Hailo-8L}",
         "Métricas clave": "FoM_red"},
    ]

    col_labels = [
        "Serie",
        "Variable independiente",
        "Configuraciones evaluadas",
        "Métricas clave",
    ]
    col_widths = [0.06, 0.22, 0.45, 0.27]

    rows = data if data is not None else _DEFAULT

    # Accept both list[dict] and list[list]
    if rows and isinstance(rows[0], dict):
        table_data = [[str(r.get(c, "")) for c in col_labels] for r in rows]
    else:
        # list[list] — pad/trim to 4 columns
        table_data = [(list(r) + ["", "", "", ""])[:4] for r in rows]
        table_data = [[str(v) for v in row] for row in table_data]

    n_rows = len(table_data)
    n_cols = len(col_labels)

    fig_h = max(2.5, 0.55 * n_rows + 1.0)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    ax.axis("off")
    fig.suptitle("Matriz experimental (E0–E5)", y=1.02,
                 fontsize=13, fontweight="bold")

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.0)

    # Header row
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor("#2e3f5c")
        cell.set_text_props(color="white", fontweight="bold", ha="left")

    # Row colors
    _ROW_COLORS = {0: "#d4e6f1", n_rows - 1: "#fdebd0"}
    for i in range(n_rows):
        fc = _ROW_COLORS.get(i, "#f9f9f9" if i % 2 == 0 else "#ffffff")
        for j in range(n_cols):
            cell = tbl[i + 1, j]
            cell.set_facecolor(fc)
            cell.PAD = 0.06

    # Highlight E0 "Serie" cell in FoM color
    tbl[1, 0].set_text_props(fontweight="bold", color=_C_FOM_FULL)
    # Highlight E5 "Serie" cell
    tbl[n_rows, 0].set_text_props(fontweight="bold", color=_C_FOM_RED)

    fig.tight_layout()
    _save(fig, save_path, fmt)
    return fig, ax
