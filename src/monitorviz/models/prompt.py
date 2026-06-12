import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


class TokenProb(BaseModel):
    """Log-probability of a single generated token.

    OLLAMA provides token text, logprob, and optional byte offsets.
    LLAMA provides only the raw probability; we convert to logprob and
    leave token as empty string (no text available).
    """

    token: str = ""
    logprob: float
    bytes: list[int] | None = None


class PromptMetric(BaseModel):
    """A single inference event (one prompt -> one response)."""

    engine: Literal["OLLAMA", "LLAMA", "HAILO_OLLAMA"]
    prob_type: int | str = Field(alias="probType")
    model: str
    prompt_id: int
    start_timestamp_ns: int
    finish_timestamp_ns: int
    total_duration_ns: int
    prompt_eval_duration_ns: int
    eval_duration_ns: int
    load_duration_ns: int
    prompt_eval_count: int
    eval_count: int
    answer: str
    tokens: list[TokenProb] = []

    model_config = ConfigDict(populate_by_name=True)

    @property
    def prob_type_normalized(self) -> int:
        """Normalize prob_type to int for backwards compat.

        New binary: "LOG_PROBABILITY" → 0, "PROBABILITY" → 1.
        Old binary: 0 → 0, 1 → 1.
        """
        _MAP: dict[int | str, int] = {
            "LOG_PROBABILITY": 0,
            "log_probability": 0,
            0: 0,
            "0": 0,
            "PROBABILITY": 1,
            "probability": 1,
            1: 1,
            "1": 1,
        }
        return _MAP.get(self.prob_type, 0)

    @property
    def _effective_eval_duration_ns(self) -> int:
        """Eval duration, falling back to total_duration_ns for HAILO_OLLAMA.

        Hailo reports eval_duration_ns=0; the wall-clock decode time is
        captured in total_duration_ns instead.
        """
        if self.eval_duration_ns > 0:
            return self.eval_duration_ns
        if self.engine == "HAILO_OLLAMA" and self.total_duration_ns > 0:
            return self.total_duration_ns
        return 0

    @computed_field
    @property
    def is_empty_generation(self) -> bool:
        """True when the model returned no meaningful generation.

        Happens for non-instruction-tuned base models that decide the
        prompt is already complete. Not an error.
        """
        return (
            self.eval_count <= 1
            or self._effective_eval_duration_ns <= 0
            or not self.answer.strip()
        )

    @computed_field
    @property
    def time_to_first_token_ns(self) -> int:
        return self.load_duration_ns + self.prompt_eval_duration_ns

    @computed_field
    @property
    def perplexity(self) -> float | None:
        """Perplexity over generated tokens. None if empty generation."""
        if self.is_empty_generation or not self.tokens:
            return None
        logprobs = [t.logprob for t in self.tokens if math.isfinite(t.logprob)]
        if not logprobs:
            return None
        return math.exp(-sum(logprobs) / len(logprobs))

    @computed_field
    @property
    def tokens_per_second(self) -> float | None:
        if self.is_empty_generation:
            return None
        return self.eval_count / (self._effective_eval_duration_ns / 1e9)

    @computed_field
    @property
    def words_per_second(self) -> float | None:
        if self.is_empty_generation:
            return None
        n_words = len(self.answer.split())
        if n_words == 0:
            return None
        return n_words / (self._effective_eval_duration_ns / 1e9)
