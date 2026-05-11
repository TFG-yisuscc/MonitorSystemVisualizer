"""Tests for the loader. Each fixture exercises a specific aspect of the schema."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest

from monitorviz.io import load_collection, load_run

FIXTURES = Path(__file__).parent / "fixtures"


class TestOllamaRun:
    def setup_method(self) -> None:
        self.run = load_run(FIXTURES / "run_ollama_type1")

    def test_run_id_matches_directory(self) -> None:
        assert self.run.run_id == "run_ollama_type1"

    def test_summary_basics(self) -> None:
        assert self.run.summary.inference_engine == "OLLAMA"
        assert self.run.summary.test_type == "TYPE_1"
        # hardware_period extracted from og_config_json
        assert self.run.summary.hardware_period_s == 0.5

    def test_two_prompts_loaded(self) -> None:
        assert len(self.run.prompts) == 2

    def test_first_prompt_off_by_one_tolerated(self) -> None:
        # Fixture has eval_count=4 but tokenProb has 3 elements.
        # The loader must accept this without error.
        p = self.run.prompts[0]
        assert p.eval_count == 4
        assert len(p.tokens) == 3

    def test_first_prompt_token_data(self) -> None:
        p = self.run.prompts[0]
        assert p.tokens[0].token == "#"
        assert p.tokens[0].logprob < 0
        # logprob from fixture is -0.6993
        assert -0.7 < p.tokens[0].logprob < -0.6

    def test_perplexity_computed(self) -> None:
        p = self.run.prompts[0]
        assert p.perplexity is not None
        assert p.perplexity > 1.0  # always > 1 for negative logprobs

    def test_hw_samples_loaded(self) -> None:
        assert len(self.run.hw_samples) == 5
        assert self.run.has_hardware_data

    def test_hw_field_renaming(self) -> None:
        s = self.run.hw_samples[0]
        assert s.temperature_c == 55.65
        assert s.fan_rpm == 2121.0
        assert s.internal_power_w == 2.94
        assert len(s.frequency_ghz) == 4

    def test_hw_timestamp_in_milliseconds(self) -> None:
        s = self.run.hw_samples[0]
        # 13 digits → milliseconds, not nanoseconds
        assert 1e12 < s.timestamp_ms < 1e14

    def test_throttling_detected(self) -> None:
        # Fixture has soft_throttled=True only on sample index 2
        active = [s for s in self.run.hw_samples if s.throttling.any_active]
        assert len(active) == 1

    def test_no_meta_yaml(self) -> None:
        # This fixture has no meta.yaml → defaults
        assert self.run.meta.fan is None
        assert self.run.meta.accelerator is None


class TestLlamaRun:
    def setup_method(self) -> None:
        self.run = load_run(FIXTURES / "run_llama_type1")

    def test_engine(self) -> None:
        assert self.run.summary.inference_engine == "LLAMA"

    def test_first_prompt_is_empty_generation(self) -> None:
        # Legitimate empty: base model decided not to generate
        p = self.run.prompts[0]
        assert p.is_empty_generation
        assert p.perplexity is None
        assert p.tokens_per_second is None
        assert p.words_per_second is None

    def test_second_prompt_normalized(self) -> None:
        p = self.run.prompts[1]
        assert not p.is_empty_generation
        # Fixture has 5 raw probs
        assert len(p.tokens) == 5
        # LLAMA does not provide token text
        assert all(t.token == "" for t in p.tokens)
        # All converted to negative logprobs
        assert all(t.logprob < 0 for t in p.tokens)

    def test_model_fallback_from_summary(self) -> None:
        # JSONL has model="", loader must fall back to summary.model_path_or_name
        p = self.run.prompts[1]
        assert p.model == "/home/user/tinytest.gguf"

    def test_model_short(self) -> None:
        # LLAMA: basename of GGUF without extension
        assert self.run.model_short == "tinytest"


class TestType0:
    def test_no_hardware_data(self) -> None:
        run = load_run(FIXTURES / "run_type0_no_hw")
        assert not run.has_hardware_data
        assert run.summary.test_type == "TYPE_0"
        assert len(run.prompts) == 1
        assert len(run.hw_samples) == 0


class TestMeta:
    def test_meta_yaml_loaded(self) -> None:
        run = load_run(FIXTURES / "run_with_meta")
        assert run.meta.fan is True
        assert run.meta.accelerator is False
        assert run.meta.ambient_temperature_c == 22.5
        assert run.meta.notes == "Fixture con meta para tests"


class TestCollection:
    def test_load_all_fixtures(self) -> None:
        coll = load_collection(FIXTURES)
        # 4 old + 1 new (v2_format) = 5 runs
        assert len(coll) == 5

    def test_skip_errors(self, tmp_path: Path) -> None:
        # Build a collection with one broken run + one valid run
        broken = tmp_path / "broken"
        broken.mkdir()
        (broken / "resumen.json").write_text("{ malformed json")

        good_src = FIXTURES / "run_type0_no_hw"
        shutil.copytree(good_src, tmp_path / "good")

        coll = load_collection(tmp_path, skip_errors=True)
        assert len(coll) == 1  # only the good one survives

    def test_no_skip_errors_raises(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken"
        broken.mkdir()
        (broken / "resumen.json").write_text("{ malformed")
        with pytest.raises(Exception):  # noqa: B017
            load_collection(tmp_path, skip_errors=False)


class TestV2Format:
    """Tests for the new binary format: string probType + JSON annotations."""

    def setup_method(self) -> None:
        self.run = load_run(FIXTURES / "run_ollama_v2_format")

    def test_loads_without_error(self) -> None:
        assert self.run.run_id == "run_ollama_v2_format"

    def test_string_prob_type_preserved(self) -> None:
        p = self.run.prompts[0]
        assert p.prob_type == "LOG_PROBABILITY"

    def test_string_prob_type_normalized(self) -> None:
        p = self.run.prompts[0]
        assert p.prob_type_normalized == 0

    def test_tokens_parsed_correctly(self) -> None:
        p = self.run.prompts[0]
        assert len(p.tokens) == 2
        assert p.tokens[0].token == "H"
        assert p.tokens[0].logprob < 0

    def test_meta_from_annotations_json(self) -> None:
        assert self.run.meta.fan is True
        assert self.run.meta.accelerator is False

    def test_meta_other_field(self) -> None:
        assert self.run.meta.other == "none"

    def test_throttling_occurred_spelling(self) -> None:
        s = self.run.hw_samples[0]
        # New fixture uses 'occurred' — Python field must be accessible
        assert s.throttling.under_voltage_occurred is False
        assert s.throttling.freq_capped_occurred is False


class TestV2BackwardsCompat:
    """Confirm old format (int probType, legacy typo in throttling keys) still works."""

    def test_old_ollama_still_loads(self) -> None:
        run = load_run(FIXTURES / "run_ollama_type1")
        p = run.prompts[0]
        assert p.prob_type == 0
        assert p.prob_type_normalized == 0

    def test_old_llama_still_loads(self) -> None:
        run = load_run(FIXTURES / "run_llama_type1")
        p = run.prompts[1]
        assert p.prob_type == 1
        assert p.prob_type_normalized == 1

    def test_old_meta_yaml_still_works(self) -> None:
        run = load_run(FIXTURES / "run_with_meta")
        assert run.meta.fan is True
        assert run.meta.accelerator is False

    def test_collection_mixes_old_and_new(self) -> None:
        coll = load_collection(FIXTURES)
        # 4 old + 1 new = 5 fixture runs
        assert len(coll) == 5


class TestLogging:
    def test_off_by_one_of_one_does_not_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The fixture has eval_count=4 with 3 tokens (diff=1). Should NOT log
        a warning about size mismatch."""
        with caplog.at_level(logging.WARNING):
            load_run(FIXTURES / "run_ollama_type1")
        relevant = [
            r for r in caplog.records if "tokenProb size mismatch" in r.message
        ]
        assert len(relevant) == 0
