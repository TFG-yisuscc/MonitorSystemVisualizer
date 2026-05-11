from monitorviz.models import PromptMetric, ThrottlingFlags, TokenProb


def test_empty_generation_has_no_metrics():
    """Prompts with eval_count<=1 and empty answer return None for derived metrics."""
    p = PromptMetric(
        engine="LLAMA",
        probType=1,
        model="test",
        prompt_id=0,
        start_timestamp_ns=0,
        finish_timestamp_ns=0,
        total_duration_ns=0,
        prompt_eval_duration_ns=0,
        eval_duration_ns=0,
        load_duration_ns=0,
        prompt_eval_count=10,
        eval_count=1,
        answer="",
        tokens=[],
    )
    assert p.is_empty_generation is True
    assert p.perplexity is None
    assert p.tokens_per_second is None
    assert p.words_per_second is None


def test_normal_prompt_computes_metrics():
    """A normal prompt with 3 tokens computes perplexity > 1."""
    tokens = [
        TokenProb(token="A", logprob=-0.5),
        TokenProb(token="B", logprob=-0.3),
        TokenProb(token="C", logprob=-0.1),
    ]
    p = PromptMetric(
        engine="OLLAMA",
        probType=0,
        model="test",
        prompt_id=0,
        start_timestamp_ns=1_000_000_000,
        finish_timestamp_ns=2_000_000_000,
        total_duration_ns=1_000_000_000,
        prompt_eval_duration_ns=200_000_000,
        eval_duration_ns=800_000_000,
        load_duration_ns=50_000_000,
        prompt_eval_count=10,
        eval_count=3,
        answer="A B C",
        tokens=tokens,
    )
    assert p.is_empty_generation is False
    assert p.perplexity is not None
    assert p.perplexity > 1.0  # always >1 for negative logprobs
    assert p.tokens_per_second == 3 / 0.8  # 3.75
    assert p.words_per_second == 3 / 0.8
    assert p.time_to_first_token_ns == 250_000_000


def test_throttling_active_detection():
    """any_active is True when any live flag is set."""
    flags = ThrottlingFlags(
        under_voltage=False,
        under_voltage_occurred=False,
        freq_capped=True,
        freq_capped_occurred=True,
        throttled=False,
        throttled_occurred=False,
        soft_throttled=False,
        soft_throttled_occurred=False,
    )
    assert flags.any_active is True
    assert flags.any_ever_occurred is True
