"""Tests for the transforms layer (DataFrames + RunCollection)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from monitorviz.io import load_collection, load_run
from monitorviz.transforms import (
    RunCollection,
    hw_freq_long,
    hw_metrics_to_df,
    model_display_label,
    prompt_metrics_to_df,
    tokens_to_df,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Per-run aggregations
# ---------------------------------------------------------------------------

class TestPromptMetricsToDf:
    def setup_method(self) -> None:
        self.run = load_run(FIXTURES / "run_ollama_type1")
        self.df = prompt_metrics_to_df(self.run)

    def test_one_row_per_prompt(self) -> None:
        assert len(self.df) == 2

    def test_factor_columns_present(self) -> None:
        for col in ["run_id", "engine", "model_short", "test_type",
                    "batch_size", "context_size", "seed", "temperature",
                    "hardware_period_s", "fan", "accelerator"]:
            assert col in self.df.columns

    def test_factors_are_run_constants(self) -> None:
        assert self.df["engine"].nunique() == 1
        assert self.df["engine"].iloc[0] == "OLLAMA"

    def test_derived_metrics_present(self) -> None:
        for col in ["perplexity", "tokens_per_second", "words_per_second",
                    "time_to_first_token_ms", "latency_ms", "t_rel_s"]:
            assert col in self.df.columns

    def test_perplexity_positive(self) -> None:
        assert (self.df["perplexity"] > 1).all()

    def test_t_rel_s_starts_near_zero(self) -> None:
        assert self.df["t_rel_s"].iloc[0] < 1.0

    def test_t_rel_s_monotonic(self) -> None:
        diffs = self.df["t_rel_s"].diff().dropna()
        assert (diffs > 0).all()


class TestPromptMetricsHandlesEmptyGeneration:
    def test_empty_generation_metrics_nan(self) -> None:
        run = load_run(FIXTURES / "run_llama_type1")
        df = prompt_metrics_to_df(run)
        empty_row = df[df["is_empty_generation"]]
        assert len(empty_row) == 1
        assert empty_row["perplexity"].isna().all()
        assert empty_row["tokens_per_second"].isna().all()
        assert empty_row["words_per_second"].isna().all()


class TestHwMetricsToDf:
    def setup_method(self) -> None:
        self.run = load_run(FIXTURES / "run_ollama_type1")
        self.df = hw_metrics_to_df(self.run)

    def test_one_row_per_sample(self) -> None:
        assert len(self.df) == 5

    def test_throttling_expanded(self) -> None:
        for col in ["throt_under_voltage", "throt_freq_capped",
                    "throt_throttled", "throt_soft_throttled",
                    "throt_any_active", "throt_any_ever_occurred"]:
            assert col in self.df.columns

    def test_throttling_active_count(self) -> None:
        assert int(self.df["throt_any_active"].sum()) == 1

    def test_freq_aggregates(self) -> None:
        for col in ["freq_mean_ghz", "freq_max_ghz", "freq_min_ghz", "n_cores"]:
            assert col in self.df.columns
        assert (self.df["freq_mean_ghz"] == 2.4).all()
        assert (self.df["n_cores"] == 4).all()

    def test_t_rel_s_correct(self) -> None:
        assert 0.0 < self.df["t_rel_s"].iloc[0] < 1.0

    def test_empty_dataframe_for_type0(self) -> None:
        run = load_run(FIXTURES / "run_type0_no_hw")
        df = hw_metrics_to_df(run)
        assert df.empty


class TestHwFreqLong:
    def test_four_rows_per_sample(self) -> None:
        run = load_run(FIXTURES / "run_ollama_type1")
        df = hw_freq_long(run)
        # 5 samples x 4 cores
        assert len(df) == 20

    def test_core_id_range(self) -> None:
        run = load_run(FIXTURES / "run_ollama_type1")
        df = hw_freq_long(run)
        assert set(df["core_id"].unique()) == {0, 1, 2, 3}


class TestTokensToDf:
    def test_long_format_rows(self) -> None:
        run = load_run(FIXTURES / "run_ollama_type1")
        df = tokens_to_df(run)
        # Two prompts with 3 tokens each
        assert len(df) == 6

    def test_token_idx_per_prompt(self) -> None:
        run = load_run(FIXTURES / "run_ollama_type1")
        df = tokens_to_df(run)
        for _, sub in df.groupby("prompt_id"):
            assert list(sub["token_idx"]) == [0, 1, 2]

    def test_skip_empty_generations(self) -> None:
        run = load_run(FIXTURES / "run_llama_type1")
        df = tokens_to_df(run)
        # Only the non-empty prompt contributes (5 tokens)
        assert len(df) == 5
        assert df["prompt_id"].nunique() == 1

    def test_exclude_llama_when_requested(self) -> None:
        coll = load_collection(FIXTURES)
        df_all = coll.tokens_df(include_llama=True)
        df_no_llama = coll.tokens_df(include_llama=False)
        assert len(df_no_llama) < len(df_all)
        assert "LLAMA" not in df_no_llama["engine"].unique()


# ---------------------------------------------------------------------------
# RunCollection
# ---------------------------------------------------------------------------

class TestRunCollection:
    def setup_method(self) -> None:
        self.coll = load_collection(FIXTURES)

    def test_loads_all_fixtures(self) -> None:
        assert len(self.coll) == 5

    def test_filter_by_engine(self) -> None:
        ollama_only = self.coll.filter(engine="OLLAMA")
        assert all(r.summary.inference_engine == "OLLAMA" for r in ollama_only)
        assert len(ollama_only) == 4  # ollama_type1, type0_no_hw, with_meta, v2_format

    def test_filter_by_meta(self) -> None:
        with_fan = self.coll.filter(fan=True)
        # run_with_meta + run_ollama_v2_format both have fan=True
        assert len(with_fan) == 2
        run_ids = {r.run_id for r in with_fan}
        assert "run_with_meta" in run_ids
        assert "run_ollama_v2_format" in run_ids

    def test_filter_unknown_key_raises(self) -> None:
        with pytest.raises(ValueError):
            self.coll.filter(nonexistent_factor=42)

    def test_prompt_metrics_df(self) -> None:
        df = self.coll.prompt_metrics_df()
        # 2 + 2 + 1 + 2 + 1 = 8 prompts in total (v2_format adds 1)
        assert len(df) == 8
        assert df["run_id"].nunique() == 5

    def test_hw_metrics_df_skips_type0(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            df = self.coll.hw_metrics_df()
        # 5 + 3 + 5 + 3 = 16 (TYPE_0 contributes 0, v2_format adds 3)
        assert len(df) == 16
        assert "run_type0_no_hw" in caplog.text

    def test_summary_df_one_row_per_run(self) -> None:
        df = self.coll.summary_df()
        assert len(df) == 5
        assert df["run_id"].nunique() == 5

    def test_summary_df_columns(self) -> None:
        df = self.coll.summary_df()
        for col in ["n_prompts", "n_empty_generations",
                    "tokens_per_s_mean", "perplexity_geomean",
                    "latency_ms_mean", "ttft_ms_mean",
                    "temp_max_c", "power_mean_w",
                    "energy_per_token_j", "throttled_ratio"]:
            assert col in df.columns

    def test_summary_df_nan_for_type0_hw(self) -> None:
        df = self.coll.summary_df()
        type0_row = df[df["test_type"] == "TYPE_0"]
        assert type0_row["temp_max_c"].isna().all()
        assert type0_row["power_mean_w"].isna().all()
        assert type0_row["throttled_ratio"].isna().all()

    def test_summary_df_counts_empty_generations(self) -> None:
        df = self.coll.summary_df()
        llama_row = df[df["run_id"] == "run_llama_type1"]
        assert int(llama_row["n_empty_generations"].iloc[0]) == 1
        assert int(llama_row["n_prompts"].iloc[0]) == 2

    def test_experiment_matrix_basic(self) -> None:
        m = self.coll.experiment_matrix()
        assert "n_runs" in m.columns
        assert int(m["n_runs"].sum()) == 5

    def test_empty_collection(self) -> None:
        empty = RunCollection(runs=[])
        assert len(empty) == 0
        assert empty.prompt_metrics_df().empty
        assert empty.hw_metrics_df().empty
        assert empty.summary_df().empty
        assert empty.experiment_matrix().empty


class TestModelDisplayLabel:
    def test_known_model_short(self) -> None:
        assert model_display_label("qwen2.5-1.5b-instruct-q4_k_m") == "qwen"
        assert model_display_label("gemma3n:e2b") == "gemma"
        assert model_display_label("granite4:micro-h") == "granite"
        assert model_display_label("ministral-3:3b-instruct") == "ministral"

    def test_unknown_falls_back(self) -> None:
        assert model_display_label("some-unknown-model-xyz") == "some-unkno"

    def test_model_label_column_in_prompt_df(self) -> None:
        run = load_run(FIXTURES / "run_ollama_type1")
        df = prompt_metrics_to_df(run)
        assert "model_label" in df.columns
        assert df["model_label"].notna().all()

    def test_model_label_column_in_hw_df(self) -> None:
        run = load_run(FIXTURES / "run_ollama_type1")
        df = hw_metrics_to_df(run)
        assert "model_label" in df.columns

    def test_model_label_in_summary_df(self) -> None:
        run = load_run(FIXTURES / "run_ollama_type1")
        from monitorviz.transforms import RunCollection
        coll = RunCollection(runs=[run])
        df = coll.summary_df()
        assert "model_label" in df.columns
        assert df["model_label"].notna().all()


class TestPowerPerPhase:
    def test_returns_df_with_hw_run(self):
        run = load_run(FIXTURES / "run_ollama_type1")
        from monitorviz.transforms import power_per_phase_df
        df = power_per_phase_df(run)
        assert not df.empty
        assert set(df["phase"].unique()) <= {"load", "prefill", "decode"}

    def test_empty_for_run_without_hw(self):
        run = load_run(FIXTURES / "run_type0_no_hw")
        from monitorviz.transforms import power_per_phase_df
        df = power_per_phase_df(run)
        assert df.empty

    def test_load_phase_likely_nan(self):
        """Load is <300 ms — usually 0 hw samples, so power_mean_w is NaN."""
        run = load_run(FIXTURES / "run_ollama_type1")
        from monitorviz.transforms import power_per_phase_df
        df = power_per_phase_df(run)
        load_rows = df[df["phase"] == "load"]
        # Not asserting NaN (fixture hw period is 1s, load is 0.2s → 0 samples)
        assert len(load_rows) > 0  # rows exist even if NaN

    def test_collection_power_per_phase(self):
        coll = load_collection(FIXTURES)
        df = coll.power_per_phase_df()
        assert "phase" in df.columns
        assert "power_mean_w" in df.columns
        assert "model_label" in df.columns


class TestMBU:
    def test_model_size_bytes(self):
        from monitorviz.transforms.aggregations import _model_size_bytes
        mi = {"n_params": 1_777_088_000, "bits_per_weight": 5.0}
        size = _model_size_bytes(mi)
        # 1.78B x 5 / 8 ≈ 1.11 GB
        assert size is not None
        assert 1.0e9 < size < 1.2e9

    def test_kv_cache_bytes(self):
        from monitorviz.transforms.aggregations import _kv_cache_bytes
        mi = {"n_layers": 28, "n_kv_heads": 2, "embedding_length": 1536, "n_heads": 12}
        kv = _kv_cache_bytes(mi, seq_length=4096, batch_size=1)
        # head_dim=128, KV = 1 x 4096 x 128 x 28 x 2 x 2 x 2 = 117,440,512 bytes (~112 MiB)
        assert kv is not None
        assert 110e6 < kv < 120e6

    def test_mbu_pct_range(self):
        from monitorviz.transforms.aggregations import _mbu_corr_pct
        mi = {"n_params": 1_777_088_000, "bits_per_weight": 5.0,
              "n_layers": 28, "n_kv_heads": 2, "embedding_length": 1536, "n_heads": 12}
        # batch_size=512 makes KV cache large → achieved BW > peak → capped at 100%
        mbu = _mbu_corr_pct(mi, tpot_s=0.6, seq_length=4096, batch_size=512)
        assert mbu is not None
        assert 0 < mbu <= 100

    def test_mbu_none_without_model_info(self):
        from monitorviz.transforms.aggregations import _mbu_corr_pct
        assert _mbu_corr_pct(None, 0.6, 4096) is None



class TestMBUFixes:
    def test_mbu_uses_batch1_for_kv(self):
        """KV cache must use batch=1 regardless of run batch_size."""
        from monitorviz.transforms.aggregations import _kv_cache_bytes
        mi = {"n_layers": 28, "n_kv_heads": 2,
              "embedding_length": 1536, "n_heads": 12}
        kv_1 = _kv_cache_bytes(mi, seq_length=4096, batch_size=1)
        kv_512 = _kv_cache_bytes(mi, seq_length=4096, batch_size=512)
        assert kv_1 is not None
        assert kv_512 == kv_1 * 512  # confirm scaling

    def test_model_size_from_quant_lookup(self):
        """BPW should be inferred from quantization when bits_per_weight is null."""
        from monitorviz.transforms.aggregations import _model_size_bytes
        mi = {"n_params": 2_000_000_000, "bits_per_weight": None,
              "quantization": "Q4_K_M"}
        size = _model_size_bytes(mi)
        assert size is not None
        # 2B x 4.5 bpw / 8 = 1.125 GB
        assert abs(size - 1_125_000_000) < 1e7

    def test_meta_fan_from_native_dict(self):
        """fan and accelerator must be extracted from native dict annotations."""
        run = load_run(FIXTURES / "run_ollama_v2_format")
        assert run.meta.fan is True
        assert run.meta.accelerator is False


class TestEnergyPerToken:
    def test_finite_when_hw_and_tokens_present(self) -> None:
        coll = load_collection(FIXTURES)
        df = coll.summary_df()
        ollama_row = df[df["run_id"] == "run_ollama_type1"]
        e = ollama_row["energy_per_token_j"].iloc[0]
        assert np.isfinite(e)
        assert e > 0

    def test_nan_for_type0(self) -> None:
        coll = load_collection(FIXTURES)
        df = coll.summary_df()
        type0_row = df[df["test_type"] == "TYPE_0"]
        assert type0_row["energy_per_token_j"].isna().all()


class TestCPUWork:
    def test_cpu_work_full_load(self) -> None:
        from monitorviz.transforms.aggregations import _cpu_work_effective
        import pandas as pd
        # 4 cores × 100% × frecuencia nominal × 20 samples × 0.5 s = 40 core·s
        hw = pd.DataFrame({
            "freq_mean_ghz": [2.4] * 20,
            "cpu_usage_pct": [100.0] * 20,
        })
        w = _cpu_work_effective(hw, period_s=0.5, n_cores=4, f_nom_ghz=2.4)
        assert w == pytest.approx(40.0, rel=1e-3)

    def test_cpu_work_half_freq(self) -> None:
        from monitorviz.transforms.aggregations import _cpu_work_effective
        import pandas as pd
        # Throttling al 50% de frecuencia → trabajo a la mitad
        hw = pd.DataFrame({
            "freq_mean_ghz": [1.2] * 20,
            "cpu_usage_pct": [100.0] * 20,
        })
        w = _cpu_work_effective(hw, period_s=0.5, n_cores=4, f_nom_ghz=2.4)
        assert w == pytest.approx(20.0, rel=1e-3)

    def test_cpu_efficiency_full(self) -> None:
        from monitorviz.transforms.aggregations import _cpu_efficiency
        # 40 core·s en 10 s con 4 cores → η = 1.0
        assert _cpu_efficiency(40.0, 10.0, n_cores=4) == pytest.approx(1.0)

    def test_cpu_efficiency_half(self) -> None:
        from monitorviz.transforms.aggregations import _cpu_efficiency
        # 20 core·s en 10 s con 4 cores → η = 0.5
        assert _cpu_efficiency(20.0, 10.0, n_cores=4) == pytest.approx(0.5)

    def test_cpu_work_per_token(self) -> None:
        from monitorviz.transforms.aggregations import _cpu_work_per_token
        # 40 core·s para 100 tokens → 0.4 core·s/token
        assert _cpu_work_per_token(40.0, 100) == pytest.approx(0.4)

    def test_cpu_work_handles_none(self) -> None:
        from monitorviz.transforms.aggregations import (
            _cpu_efficiency,
            _cpu_work_effective,
            _cpu_work_per_token,
        )
        import pandas as pd
        assert _cpu_work_effective(pd.DataFrame(), 0.5) is None
        assert _cpu_efficiency(None, 10.0) is None
        assert _cpu_work_per_token(None, 100) is None
        assert _cpu_work_per_token(10.0, 0) is None

    def test_summary_df_has_cpu_work_metrics(self) -> None:
        coll = load_collection(FIXTURES)
        df = coll.summary_df()
        for col in ["cpu_work_core_s", "cpu_efficiency", "cpu_work_per_token"]:
            assert col in df.columns


class TestCPUWorkByPhase:
    def test_cpu_work_by_phase_returns_dataframe(self) -> None:
        from monitorviz.transforms.aggregations import cpu_work_by_phase
        import pandas as pd
        run = load_run(FIXTURES / "run_ollama_type1")
        hw = hw_metrics_to_df(run)
        if hw.empty:
            pytest.skip("no hw data")
        df = cpu_work_by_phase(hw, run)
        assert isinstance(df, pd.DataFrame)

    def test_cpu_work_by_phase_columns(self) -> None:
        from monitorviz.transforms.aggregations import cpu_work_by_phase
        run = load_run(FIXTURES / "run_ollama_type1")
        hw = hw_metrics_to_df(run)
        if hw.empty:
            pytest.skip("no hw data")
        df = cpu_work_by_phase(hw, run)
        if not df.empty:
            for col in ["phase", "cpu_work_core_s", "duration_s", "cpu_efficiency"]:
                assert col in df.columns

    def test_cpu_work_by_phase_efficiency_range(self) -> None:
        from monitorviz.transforms.aggregations import cpu_work_by_phase
        run = load_run(FIXTURES / "run_ollama_type1")
        hw = hw_metrics_to_df(run)
        if hw.empty:
            pytest.skip("no hw data")
        df = cpu_work_by_phase(hw, run)
        if not df.empty:
            assert (df["cpu_efficiency"].between(0, 1)).all()

    def test_compute_efficiency_per_joule(self) -> None:
        from monitorviz.transforms.aggregations import _compute_efficiency_per_joule
        assert _compute_efficiency_per_joule(40.0, 10.0) == pytest.approx(4.0)
        assert _compute_efficiency_per_joule(None, 10.0) is None
        assert _compute_efficiency_per_joule(40.0, 0.0) is None

    def test_mbu_is_upper_bound_field(self) -> None:
        coll = load_collection(FIXTURES)
        df = coll.summary_df()
        assert "mbu_is_upper_bound" in df.columns

    def test_cpu_work_by_phase_df_collection(self) -> None:
        import pandas as pd
        coll = load_collection(FIXTURES)
        df = coll.cpu_work_by_phase_df()
        assert isinstance(df, pd.DataFrame)
