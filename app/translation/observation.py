"""The one reading format every device adapter translates into, so the
models downstream never see a device's raw output directly.

Numbers in an Observation come only from deterministic, tested adapter
code (see as7265x.py) — never from an LLM. AI's role in this layer is
the fuzzy parts around the numbers (format detection, metadata, drafting
new adapters for human review), not the values themselves.
"""
from datetime import datetime

from pydantic import BaseModel, Field


class QualityCheck(BaseModel):
    passed: bool
    # Machine-readable reasons, e.g. "saturated:R" or "reflectance_out_of_range:L".
    # Any flag means the models are not run on this reading.
    flags: list[str] = Field(default_factory=list)


class Provenance(BaseModel):
    adapter: str
    adapter_version: str
    # Calibration inputs that produced the values, so a result can be
    # re-derived or audited later.
    details: dict = Field(default_factory=dict)


class Observation(BaseModel):
    device: str
    # Which trained model/band grid the values are shaped for
    # (matches app.config.Settings.model_name_by_sensor keys).
    sensor_type: str
    wavelengths_nm: list[float]
    values: list[float]
    units: str
    lat: float | None = None
    lon: float | None = None
    depth_m: float | None = None
    captured_at: datetime | None = None
    qc: QualityCheck
    provenance: Provenance
