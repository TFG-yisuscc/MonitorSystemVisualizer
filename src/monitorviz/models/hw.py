from pydantic import AliasChoices, BaseModel, ConfigDict, Field, computed_field


class ThrottlingFlags(BaseModel):
    """Throttling flags from vcgencmd.

    Accepts both the corrected spelling ('occurred') introduced in MonitorSystem
    v2 and the legacy typo ('ocurred') present in runs recorded with older
    binaries. The Python field names use the correct spelling.
    """

    model_config = ConfigDict(populate_by_name=True)

    under_voltage: bool
    under_voltage_occurred: bool = Field(
        validation_alias=AliasChoices("under_voltage_occurred", "under_voltage_ocurred")
    )
    freq_capped: bool
    freq_capped_occurred: bool = Field(
        validation_alias=AliasChoices("freq_capped_occurred", "freq_capped_ocurred")
    )
    throttled: bool
    throttled_occurred: bool = Field(
        validation_alias=AliasChoices("throttled_occurred", "throttled_ocurred")
    )
    soft_throttled: bool
    soft_throttled_occurred: bool = Field(
        validation_alias=AliasChoices("soft_throttled_occurred", "soft_throttled_ocurred")
    )

    @computed_field
    @property
    def any_active(self) -> bool:
        """True if any throttling state is currently active."""
        return (
            self.under_voltage
            or self.freq_capped
            or self.throttled
            or self.soft_throttled
        )

    @computed_field
    @property
    def any_ever_occurred(self) -> bool:
        """True if any throttling has ever occurred since system boot."""
        return any(
            [
                self.under_voltage_occurred,
                self.freq_capped_occurred,
                self.throttled_occurred,
                self.soft_throttled_occurred,
            ]
        )


class HwSample(BaseModel):
    """One hardware sample, taken every ~hardware_period seconds."""

    timestamp_ms: int
    temperature_c: float
    fan_rpm: float
    voltage_v: float
    internal_power_w: float
    frequency_ghz: list[float]
    cpu_usage_pct: float
    cpu_ticks: dict[str, int]
    mem_used_bytes: int
    mem_total_bytes: int
    mem_pct: float
    swap_used_bytes: int
    swap_total_bytes: int
    swap_pct: float
    throttling: ThrottlingFlags
