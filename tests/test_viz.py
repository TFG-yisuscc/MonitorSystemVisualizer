"""Smoke tests for viz: build figures, never assert pixels."""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Use non-interactive backend so tests don't pop up windows
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from monitorviz.io import load_collection, load_run
from monitorviz.transforms import (
    hw_freq_long,
    hw_metrics_to_df,
    prompt_metrics_to_df,
    tokens_to_df,
)
from monitorviz.viz import (
    COLORS,
    ENGINE_COLORS,
    MODEL_PALETTE,
    cpu_memory_dual,
    cpu_memory_dual_phases,
    get_engine_color,
    hw_distributions_panel,
    hw_freq_panel,
    hw_timeline,
    inference_summary_panel,
    logprob_panel,
    memory_timeline,
    pareto_panel,
    plot_dual_axis,
    plot_freq_per_core,
    plot_hw_line,
    plot_logprob_by_position,
    plot_metric_bars,
    plot_metric_distribution,
    plot_pareto,
    plot_phase_strip,
    plot_prompt_lines,
    plot_prompt_phases,
    plot_prompt_spans,
    plot_throttle_markers,
    setup_style,
    temp_freq_dual,
    temp_power_dual,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def ollama_run():
    return load_run(FIXTURES / "run_ollama_type1")


@pytest.fixture(scope="module")
def coll():
    return load_collection(FIXTURES)


@pytest.fixture(autouse=True)
def close_figures():
    """Always close figures after each test to avoid memory bloat."""
    yield
    plt.close("all")


# --- style ----------------------------------------------------------------

class TestStyle:
    def test_setup_style_runs(self):
        setup_style()  # no exception

    def test_engine_color_known(self):
        assert get_engine_color("OLLAMA") == ENGINE_COLORS["OLLAMA"]

    def test_engine_color_unknown_falls_back(self):
        assert get_engine_color("FOO") == "#7f7f7f"

    def test_palettes_non_empty(self):
        assert len(MODEL_PALETTE) >= 4
        assert "temperature" in COLORS


# --- primitives -----------------------------------------------------------

class TestPrimitives:
    def test_plot_hw_line(self, ollama_run):
        df = hw_metrics_to_df(ollama_run)
        _, ax = plt.subplots()
        result = plot_hw_line(ax, df, metric="temperature_c")
        assert isinstance(result, Axes)

    def test_plot_hw_line_with_hue(self, coll):
        df = coll.hw_metrics_df()
        _, ax = plt.subplots()
        plot_hw_line(ax, df, metric="temperature_c", hue="run_id")

    def test_plot_freq_per_core(self, ollama_run):
        df = hw_freq_long(ollama_run)
        _, ax = plt.subplots()
        plot_freq_per_core(ax, df)

    def test_plot_prompt_spans(self, ollama_run):
        prompt_df = prompt_metrics_to_df(ollama_run)
        hw_df = hw_metrics_to_df(ollama_run)
        _, ax = plt.subplots()
        plot_hw_line(ax, hw_df, metric="temperature_c")
        plot_prompt_spans(ax, prompt_df)

    def test_plot_prompt_spans_handles_empty(self):
        _, ax = plt.subplots()
        plot_prompt_spans(ax, pd.DataFrame())  # must not raise

    def test_plot_throttle_markers(self, ollama_run):
        hw_df = hw_metrics_to_df(ollama_run)
        _, ax = plt.subplots()
        plot_hw_line(ax, hw_df, metric="temperature_c")
        plot_throttle_markers(ax, hw_df)

    def test_plot_metric_bars(self, ollama_run):
        df = prompt_metrics_to_df(ollama_run)
        _, ax = plt.subplots()
        plot_metric_bars(ax, df, metric="latency_ms")

    @pytest.mark.parametrize("kind", ["box", "violin", "ecdf"])
    def test_plot_metric_distribution(self, coll, kind):
        df = coll.prompt_metrics_df()
        _, ax = plt.subplots()
        plot_metric_distribution(ax, df, metric="latency_ms", kind=kind, hue="engine")

    def test_plot_metric_distribution_invalid_kind_raises(self, ollama_run):
        df = prompt_metrics_to_df(ollama_run)
        _, ax = plt.subplots()
        with pytest.raises(ValueError):
            plot_metric_distribution(ax, df, metric="latency_ms", kind="invalid")

    def test_plot_logprob_by_position(self, ollama_run):
        df = tokens_to_df(ollama_run)
        _, ax = plt.subplots()
        plot_logprob_by_position(ax, df)

    def test_plot_pareto(self, coll):
        summary = coll.summary_df()
        _, ax = plt.subplots()
        plot_pareto(
            ax, summary,
            x="power_mean_w", y="tokens_per_s_mean", hue="model_short",
        )


# --- composite ------------------------------------------------------------

class TestComposite:
    def test_hw_timeline_basic(self, ollama_run):
        hw_df = hw_metrics_to_df(ollama_run)
        prompt_df = prompt_metrics_to_df(ollama_run)
        fig = hw_timeline(hw_df, prompt_df, title="test")
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 5  # 4 hw panels + 1 phase strip

    def test_hw_timeline_custom_metrics(self, ollama_run):
        hw_df = hw_metrics_to_df(ollama_run)
        fig = hw_timeline(
            hw_df, metrics=("temperature_c", "internal_power_w")
        )
        assert len(fig.axes) == 2

    def test_hw_timeline_no_prompts(self, ollama_run):
        hw_df = hw_metrics_to_df(ollama_run)
        fig = hw_timeline(hw_df)  # prompt_df is None
        assert isinstance(fig, Figure)

    def test_hw_freq_panel(self, ollama_run):
        df = hw_freq_long(ollama_run)
        fig = hw_freq_panel(df, title="freq")
        assert isinstance(fig, Figure)

    def test_inference_summary_panel(self, ollama_run):
        df = prompt_metrics_to_df(ollama_run)
        fig = inference_summary_panel(df)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 3  # default 3 metrics

    def test_inference_summary_panel_custom(self, ollama_run):
        df = prompt_metrics_to_df(ollama_run)
        fig = inference_summary_panel(df, metrics=("latency_ms",))
        assert len(fig.axes) == 1

    def test_logprob_panel(self, ollama_run):
        p = next(pr for pr in ollama_run.prompts if not pr.is_empty_generation)
        fig = logprob_panel(p.tokens)
        assert isinstance(fig, Figure)

    def test_pareto_panel(self, coll):
        df = coll.summary_df()
        fig = pareto_panel(df, title="Pareto")
        assert isinstance(fig, Figure)


# --- prompt annotations ---------------------------------------------------

class TestPromptAnnotations:
    @pytest.mark.parametrize("annot", ["strip", "phases", "spans", "lines", "none"])
    def test_hw_timeline_with_each_annotation(self, ollama_run, annot):
        hw_df = hw_metrics_to_df(ollama_run)
        prompt_df = prompt_metrics_to_df(ollama_run)
        fig = hw_timeline(hw_df, prompt_df, prompt_annotation=annot)
        assert isinstance(fig, Figure)

    def test_hw_timeline_invalid_annotation_raises(self, ollama_run):
        hw_df = hw_metrics_to_df(ollama_run)
        prompt_df = prompt_metrics_to_df(ollama_run)
        with pytest.raises(ValueError):
            hw_timeline(hw_df, prompt_df, prompt_annotation="bogus")

    def test_plot_prompt_phases_skips_zero_duration(self):
        """Phases with duration 0 must not raise (LLAMA load=0 case)."""
        df = pd.DataFrame([
            {
                "prompt_id": 0, "t_rel_s": 0.0,
                "load_duration_ns": 0,
                "prompt_eval_duration_ns": 1_000_000_000,
                "eval_duration_ns": 2_000_000_000,
                "latency_ms": 3000,
            },
        ])
        _, ax = plt.subplots()
        ax.plot([0, 5], [0, 1])
        plot_prompt_phases(ax, df)  # must not raise

    def test_plot_prompt_phases_invalid_position(self, ollama_run):
        prompt_df = prompt_metrics_to_df(ollama_run)
        _, ax = plt.subplots()
        with pytest.raises(ValueError):
            plot_prompt_phases(ax, prompt_df, band_position="middle")

    def test_plot_prompt_lines(self, ollama_run):
        prompt_df = prompt_metrics_to_df(ollama_run)
        hw_df = hw_metrics_to_df(ollama_run)
        _, ax = plt.subplots()
        plot_hw_line(ax, hw_df, metric="temperature_c")
        plot_prompt_lines(ax, prompt_df)


# --- memory and dual-axis -------------------------------------------------

class TestMemoryAndDualAxis:
    def test_memory_timeline(self, ollama_run):
        hw_df = hw_metrics_to_df(ollama_run)
        prompt_df = prompt_metrics_to_df(ollama_run)
        fig = memory_timeline(hw_df, prompt_df, title="mem")
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 3  # 2 mem panels + 1 phase strip

    def test_memory_timeline_no_swap(self, ollama_run):
        hw_df = hw_metrics_to_df(ollama_run)
        fig = memory_timeline(hw_df, show_swap=False)
        assert len(fig.axes) == 1

    def test_cpu_memory_dual(self, ollama_run):
        hw_df = hw_metrics_to_df(ollama_run)
        fig = cpu_memory_dual(hw_df, title="cpu vs mem")
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2

    def test_temp_power_dual(self, ollama_run):
        hw_df = hw_metrics_to_df(ollama_run)
        fig = temp_power_dual(hw_df)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2

    def test_plot_dual_axis_returns_pair(self, ollama_run):
        hw_df = hw_metrics_to_df(ollama_run)
        _, ax = plt.subplots()
        a, b = plot_dual_axis(
            ax, hw_df,
            x="t_rel_s", y_left="cpu_usage_pct", y_right="mem_pct",
        )
        assert a is ax
        assert b is not ax


# --- phase strip ----------------------------------------------------------

class TestPhaseStrip:
    def test_strip_default_in_hw_timeline(self, ollama_run):
        hw_df = hw_metrics_to_df(ollama_run)
        prompt_df = prompt_metrics_to_df(ollama_run)
        fig = hw_timeline(hw_df, prompt_df)  # default = "strip"
        assert len(fig.axes) == 5  # 4 hw panels + 1 phase strip

    def test_strip_in_memory_timeline(self, ollama_run):
        hw_df = hw_metrics_to_df(ollama_run)
        prompt_df = prompt_metrics_to_df(ollama_run)
        fig = memory_timeline(hw_df, prompt_df)  # default = "strip"
        assert len(fig.axes) == 3  # 2 mem panels + 1 strip

    def test_no_strip_when_no_prompts(self, ollama_run):
        hw_df = hw_metrics_to_df(ollama_run)
        fig = hw_timeline(hw_df, prompt_annotation="strip")
        assert len(fig.axes) == 4  # only hw panels, no prompt_df

    def test_plot_phase_strip_isolated(self, ollama_run):
        prompt_df = prompt_metrics_to_df(ollama_run)
        _, ax = plt.subplots(figsize=(10, 1))
        result = plot_phase_strip(ax, prompt_df)
        assert result is ax

    def test_plot_phase_strip_handles_empty(self):
        _, ax = plt.subplots()
        plot_phase_strip(ax, pd.DataFrame())  # must not raise

    def test_plot_phase_strip_with_zero_load(self):
        """Simulate LLAMA-style data where load_duration_ns == 0."""
        df = pd.DataFrame([
            {
                "prompt_id": 0, "t_rel_s": 0.0,
                "load_duration_ns": 0,
                "prompt_eval_duration_ns": 1_000_000_000,
                "eval_duration_ns": 5_000_000_000,
            },
        ])
        _, ax = plt.subplots()
        plot_phase_strip(ax, df)  # must not raise

    def test_strip_has_no_legend(self, ollama_run):
        """Phase labels are inside the bands, not in a floating legend."""
        prompt_df = prompt_metrics_to_df(ollama_run)
        _, ax = plt.subplots(figsize=(10, 1))
        plot_phase_strip(ax, prompt_df)
        assert ax.get_legend() is None

    def test_strip_throttle_markers_applied(self, ollama_run):
        """When show_throttle=True, the strip also receives throttle markers."""
        hw_df = hw_metrics_to_df(ollama_run)
        prompt_df = prompt_metrics_to_df(ollama_run)
        fig = hw_timeline(hw_df, prompt_df, show_throttle=True)
        strip_ax = fig.axes[0]
        xd = [ln.get_xdata() for ln in strip_ax.lines]
        n_vlines = sum(
            1 for x in xd
            if len(x) == 2 and x[0] == x[1]
        )
        assert n_vlines >= 1  # fixture has 1 throttling sample

    def test_xticks_hidden_on_intermediate_panels(self, ollama_run):
        """Only the last hw panel shows x tick labels."""
        hw_df = hw_metrics_to_df(ollama_run)
        prompt_df = prompt_metrics_to_df(ollama_run)
        fig = hw_timeline(hw_df, prompt_df)  # default = "strip"
        # axes[0]=strip, axes[1:-1]=intermediate hw, axes[-1]=last hw
        for ax in fig.axes[1:-1]:
            labels = ax.xaxis.get_majorticklabels()
            assert all(not lbl.get_visible() for lbl in labels) or len(labels) == 0


# --- new composites (Benoit-Cattin 2020 / TFG Fig. 2.16) -----------------

class TestNewComposites:
    def test_cpu_memory_dual_phases_returns_figure(self, ollama_run):
        hw_df = hw_metrics_to_df(ollama_run)
        if hw_df.empty:
            pytest.skip("no hw data in fixture")
        fig = cpu_memory_dual_phases(hw_df, ollama_run)
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_temp_freq_dual_returns_figure(self, ollama_run):
        hw_df = hw_metrics_to_df(ollama_run)
        if hw_df.empty:
            pytest.skip("no hw data in fixture")
        fig = temp_freq_dual(hw_df, ollama_run)
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_hw_distributions_panel_returns_figure(self, ollama_run):
        hw_df = hw_metrics_to_df(ollama_run)
        if hw_df.empty:
            pytest.skip("no hw data in fixture")
        fig = hw_distributions_panel(hw_df, ollama_run)
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_cpu_memory_dual_phases_has_twin_axes(self, ollama_run):
        hw_df = hw_metrics_to_df(ollama_run)
        if hw_df.empty:
            pytest.skip("no hw data in fixture")
        fig = cpu_memory_dual_phases(hw_df, ollama_run)
        assert len(fig.axes) == 2

    def test_temp_freq_dual_has_twin_axes(self, ollama_run):
        hw_df = hw_metrics_to_df(ollama_run)
        if hw_df.empty:
            pytest.skip("no hw data in fixture")
        fig = temp_freq_dual(hw_df, ollama_run)
        assert len(fig.axes) == 2

    def test_hw_distributions_panel_title_override(self, ollama_run):
        hw_df = hw_metrics_to_df(ollama_run)
        if hw_df.empty:
            pytest.skip("no hw data in fixture")
        fig = hw_distributions_panel(hw_df, ollama_run, title="mi título")
        assert fig.texts[0].get_text() == "mi título"
        plt.close("all")


# --- New visualization functions (parametric analysis and insights) --------

class TestNewCompositesExtra:
    def test_sensitivity_curve_empty(self):
        from monitorviz.viz import sensitivity_curve
        df = pd.DataFrame()
        fig = sensitivity_curve(df, "batch_size",
                                [("tokens_per_s_mean", "T")])
        assert fig is not None
        plt.close("all")

    def test_sensitivity_curve_with_data(self, coll):
        from monitorviz.viz import sensitivity_curve
        s = coll.summary_df()
        if "context_size" not in s.columns:
            pytest.skip("no context_size column")
        fig = sensitivity_curve(
            s, "context_size",
            [("tokens_per_s_mean", "Throughput (tok/s)")]
        )
        assert fig is not None
        plt.close("all")

    def test_pareto_multi_runs(self, coll):
        from monitorviz.viz import pareto_panel_multi
        s = coll.summary_df()
        if len(s) < 2:
            pytest.skip("requires ≥2 runs")
        pairs = [
            ("temp_max_c", "tokens_per_s_mean",
             "Temp máx", "tok/s", True, False),
        ]
        fig = pareto_panel_multi(s, pairs)
        assert fig is not None
        plt.close("all")

    def test_throughput_over_time(self, ollama_run):
        from monitorviz.viz import throughput_over_time
        fig = throughput_over_time(ollama_run)
        assert fig is not None
        plt.close("all")

    def test_phase_breakdown_stacked(self, coll):
        from monitorviz.viz import phase_breakdown_stacked
        s = coll.summary_df()
        fig = phase_breakdown_stacked(s)
        assert fig is not None
        plt.close("all")

    def test_phase_breakdown_stacked_absolute(self, coll):
        from monitorviz.viz import phase_breakdown_stacked
        s = coll.summary_df()
        fig = phase_breakdown_stacked(s, normalized=False)
        assert fig is not None
        plt.close("all")

    def test_throttling_heatmap(self, coll):
        from monitorviz.viz import throttling_heatmap
        hw = coll.hw_metrics_df()
        runs = [r for r in coll.runs if r.has_hardware_data]
        if not runs:
            pytest.skip("no hw fixture")
        fig = throttling_heatmap(hw, runs)
        assert fig is not None
        plt.close("all")

    def test_correlation_heatmap(self, coll):
        from monitorviz.viz import correlation_heatmap
        s = coll.summary_df()
        fig = correlation_heatmap(s)
        assert fig is not None
        plt.close("all")

    def test_correlation_heatmap_insufficient_data(self):
        from monitorviz.viz import correlation_heatmap
        df = pd.DataFrame({"a": [1.0]})
        fig = correlation_heatmap(df)  # only 1 row, must not raise
        assert fig is not None
        plt.close("all")
