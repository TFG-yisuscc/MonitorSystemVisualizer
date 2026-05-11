from pydantic import BaseModel, ConfigDict


class RunMeta(BaseModel):
    """External metadata associated with a run, written manually by the
    experimenter as ``meta.yaml`` next to ``resumen.json``.

    Captures factors that the C++ binary doesn't know about: whether
    the fan was on, whether an external accelerator was attached, etc.
    """

    fan: bool | None = None
    accelerator: bool | None = None
    ambient_temperature_c: float | None = None
    notes: str | None = None
    other: str | None = None

    model_config = ConfigDict(extra="allow")
