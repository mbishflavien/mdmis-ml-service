from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    # Raman shift in cm^-1 and the matching intensity at each point. Field
    # names are deliberately generic (not "wavelengths"/"reflectance") so a
    # future non-Raman sensor type can reuse this shape if its readings are
    # also just an (x, y) spectrum.
    x_values: list[float] = Field(min_length=10)
    intensities: list[float] = Field(min_length=10)
    sensor_type: str = "lab"


class ConfidenceAlternative(BaseModel):
    mineral: str
    probability: float


class ClassifyResponse(BaseModel):
    mineral_type: str
    confidence_score: int
    confidence_alternatives: list[ConfidenceAlternative]
    # Grade (ore concentration %) isn't estimable from a mineral-ID
    # classifier alone — it needs an instrument-specific calibration curve
    # this v1 model doesn't have. Always null here rather than guessed;
    # the backend leaves MineralZone.grade_pct for a geologist/lab to fill.
    grade_pct: None = None
    model_version: str
