"""Visualization style: theme setup, color palette, axis formatters."""

from __future__ import annotations

from typing import Final

import matplotlib as mpl
import seaborn as sns

# --- Color palette ---------------------------------------------------------

ENGINE_COLORS: Final[dict[str, str]] = {
    "OLLAMA": "#1f77b4",
    "LLAMA": "#ff7f0e",
}

MODEL_PALETTE: Final[list[str]] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]

COLORS: Final[dict[str, str]] = {
    "temperature": "#d62728",
    "power": "#1f77b4",
    "frequency": "#2ca02c",
    "cpu": "#9467bd",
    "memory": "#8c564b",
    "swap": "#e377c2",
    "throttle": "#e41a1c",
    "prompt_span": "#ff7f0e",
    "reference_line": "#7f7f7f",
    # Phase colors for prompt-stage annotations
    "phase_load": "#bcbd22",      # olive yellow — model load
    "phase_prefill": "#17becf",   # cyan — prompt prefill
    "phase_decode": "#ff7f0e",    # orange — token generation
}


def setup_style(context: str = "talk") -> None:
    """Configure seaborn + matplotlib defaults for the project.

    Default context "talk" gives larger fonts well-suited for figures that
    will end up in a written dissertation. Use "notebook" for compact
    in-notebook exploration, "paper" for two-column publications.
    """
    sns.set_theme(
        context=context,
        style="whitegrid",
        palette=MODEL_PALETTE,
    )
    mpl.rcParams.update(
        {
            "figure.figsize": (12, 6),
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.titlesize": "large",
            "axes.titleweight": "bold",
            "axes.labelsize": "medium",
            "legend.fontsize": "medium",
            "legend.frameon": False,
            "xtick.labelsize": "small",
            "ytick.labelsize": "small",
            "lines.linewidth": 1.8,
        }
    )


def get_engine_color(engine: str) -> str:
    """Return the canonical color for an inference engine name."""
    return ENGINE_COLORS.get(engine, "#7f7f7f")


# --- Hardware constants ---------------------------------------------------

# Raspberry Pi 5 (BCM2712, LPDDR4X-4267, 64-bit bus)
# Peak memory bandwidth = 4267 MT/s x 8 bytes x 1 channel = 34.1 GB/s
# Source: BCM2712 datasheet / RPi5 official specs
RPI5_PEAK_MEMORY_BANDWIDTH_GBs: Final[float] = 34.1

# Default target platform for MBU calculations.
# Change to the actual peak bandwidth of your device if different.
TARGET_PEAK_MEMORY_BANDWIDTH_GBs: Final[float] = RPI5_PEAK_MEMORY_BANDWIDTH_GBs

# RPi5 BCM2712 — frecuencia nominal y número de cores
RPI5_NOMINAL_FREQ_GHZ: Final[float] = 2.4
RPI5_NUM_CORES: Final[int] = 4

# Defaults para cálculos de trabajo CPU efectivo
TARGET_NOMINAL_FREQ_GHZ: Final[float] = RPI5_NOMINAL_FREQ_GHZ
TARGET_NUM_CORES: Final[int] = RPI5_NUM_CORES


def format_seconds(value: float, _pos: int | None = None) -> str:
    """Axis formatter: turn raw seconds into '12.3 s' style strings."""
    if abs(value) >= 60:
        m, s = divmod(value, 60)
        return f"{int(m):d}m{int(s):02d}s"
    return f"{value:.1f}s"
