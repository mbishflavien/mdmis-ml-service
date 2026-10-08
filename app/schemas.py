from datetime import datetime

from pydantic import BaseModel, Field

from app.translation.observation import Observation


class ClassifyRequest(BaseModel):
    # Raman shift in cm^-1 and the matching intensity at each point. Field
    # names are deliberately generic (not "wavelengths"/"reflectance") so a
    # future non-Raman sensor type can reuse this shape if its readings are
    # also just an (x, y) spectrum.
    x_values: list[float] = Field(min_length=10)
    intensities: list[float] = Field(min_length=10)
    sensor_type: str = "lab"


class PathfinderRequest(BaseModel):
    # Same (x, y) spectrum shape as ClassifyRequest, but restricted to the
    # two reflectance sensors — the pathfinder model was never trained on
    # Raman shift, so "lab" isn't a valid sensor_type here.
    x_values: list[float] = Field(min_length=10)
    intensities: list[float] = Field(min_length=10)
    sensor_type: str = "sentinel2"


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


class AS7265xReadingRequest(BaseModel):
    # Each of sample/dark/white is either 18 numbers or one raw
    # SparkFun-style serial line ("123,456,...") as the device printed it.
    sample: list[float] | str
    dark: list[float] | str
    white: list[float] | str
    white_reference_reflectance: float = 0.99
    lat: float | None = None
    lon: float | None = None
    depth_m: float | None = None
    captured_at: datetime | None = None


class Sentinel2ReadingRequest(BaseModel):
    # Raw L2A digital numbers per band, exactly as stored in the product
    # (B02..B12, see app.sensor_bands.SENTINEL2_BANDS) — not reflectance.
    bands: dict[str, float]
    # From the scene metadata (STAC "s2:processing_baseline"), e.g. "05.10".
    # Decides whether the -1000 offset applies.
    processing_baseline: str
    scl: int | None = None
    lat: float | None = None
    lon: float | None = None
    captured_at: datetime | None = None


class ReadingPosition(BaseModel):
    lat: float
    lon: float
    depth_m: float = 0.0  # metres below the ground surface


class PlaceReadingsRequest(BaseModel):
    readings: list[ReadingPosition] = Field(min_length=1, max_length=10_000)


class BandRatiosOut(BaseModel):
    iron_oxide: float
    carbonate: float
    clay: float
    ndvi: float
    flags: list[str]


class PathfinderCategoryAlternative(BaseModel):
    category: str
    probability: float


class PathfinderResponse(BaseModel):
    category: str
    category_score: int
    category_alternatives: list[PathfinderCategoryAlternative]
    # Sum of the three gold-associated categories' probabilities
    # (excludes "background"). A screening score, not a confidence that
    # gold is present — see `caveat`.
    pathfinder_score: int
    caveat: str
    model_version: str


class ReadingResponse(BaseModel):
    observation: Observation
    # Both null when observation.qc.passed is false — the models are never
    # run on a reading that failed translation quality checks.
    mineral: ClassifyResponse | None = None
    pathfinder: PathfinderResponse | None = None
    # Sentinel-2 only: the SRS's rule-based alteration ratios.
    band_ratios: BandRatiosOut | None = None
