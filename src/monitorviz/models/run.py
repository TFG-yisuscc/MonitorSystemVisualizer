from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .hw import HwSample
from .meta import RunMeta
from .prompt import PromptMetric


class RunSummary(BaseModel):
    """Parsed contents of resumen.json, plus hardware_period_s extracted
    from the nested og_config_json string."""

    timestamp_run_start_ns: int
    timestamp_run_end_ns: int
    inference_engine: Literal["OLLAMA", "LLAMA", "HAILO_OLLAMA"]
    model_path_or_name: str
    test_type: Literal["TYPE_0", "TYPE_1", "TYPE_2"]
    batch_size: int
    context_size: int
    num_prompts: int
    seed: int
    temperature: float
    hardware_period_s: float
    annotations: str = ""
    model_info: dict | None = None
    # External power-meter fields (HAILO_OLLAMA only): measured from power-on
    # (5-min idle baseline + full inference run) by an external smart-plug.
    total_kwh: float | None = None   # total energy in kWh
    watt_min: float | None = None    # minimum system power (W), used as idle baseline
    watt_max: float | None = None    # peak system power (W) during the run
    raw_resumen: dict
    raw_og_config: dict


class Run(BaseModel):
    """A complete run: metadata + per-prompt metrics + (optional) hw timeline."""

    run_id: str
    summary: RunSummary
    meta: RunMeta = Field(default_factory=RunMeta)
    prompts: list[PromptMetric]
    hw_samples: list[HwSample] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def has_hardware_data(self) -> bool:
        return len(self.hw_samples) > 0

    @property
    def model_short(self) -> str:
        """Human-readable model name for plot legends.

        OLLAMA names are used as-is (e.g. 'granite4:micro-h').
        LLAMA model paths are reduced to the GGUF basename without
        extension (e.g. '/home/user/foo.gguf' -> 'foo').
        """
        name = self.summary.model_path_or_name
        if self.summary.inference_engine == "LLAMA":
            return Path(name).stem.replace(".gguf", "")
        return name  # OLLAMA and HAILO_OLLAMA use the model name as-is
