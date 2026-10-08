"""Sentinel-2 L2A digital numbers -> surface reflectance Observation.

L2A files don't store reflectance directly. Each band is an integer DN:
  reflectance = (DN + BOA_ADD_OFFSET) / 10000
where BOA_ADD_OFFSET is -1000 for scenes processed with baseline 04.00 or
later (ESA change of 25 Jan 2022) and 0 before that. The offset is not in
Planetary Computer's asset metadata (checked on a real baseline-05.10
scene), so it has to be applied here — scripts/fetch_sentinel2.py
originally divided by 10000 only, which made every recent reading ~0.1
too bright.

QC also uses the L2A Scene Classification layer (SCL) for the pixel, and
NDVI: clouds, shadows, water, snow and vegetation all make the mineral
models meaningless, so those readings are flagged and never sent to them.
"""
from datetime import datetime

from app.sensor_bands import SENTINEL2_BANDS, band_centers
from app.translation.observation import Observation, Provenance, QualityCheck

ADAPTER_VERSION = "1.0"
QUANTIFICATION_VALUE = 10000
_OFFSET_FROM_BASELINE = 4.0
_BOA_ADD_OFFSET = -1000

# ESA L2A Scene Classification values that make a pixel unusable for
# mineral work. 4 (vegetation) is handled by the NDVI check below too.
_BAD_SCL = {
    0: "no_data",
    1: "saturated_or_defective",
    2: "dark_area",
    3: "cloud_shadow",
    4: "vegetation",
    6: "water",
    8: "cloud_medium_probability",
    9: "cloud_high_probability",
    10: "thin_cirrus",
    11: "snow_or_ice",
}
_NDVI_VEGETATION = 0.2  # same cutoff as app/band_ratios.py
_REFLECTANCE_MIN = -0.02
_REFLECTANCE_MAX = 1.2


def boa_add_offset(processing_baseline: str) -> int:
    try:
        baseline = float(processing_baseline)
    except ValueError:
        raise ValueError(f"processing_baseline must look like '05.10', got {processing_baseline!r}.")
    return _BOA_ADD_OFFSET if baseline >= _OFFSET_FROM_BASELINE else 0


def translate(
    dn: dict[str, float],
    processing_baseline: str,
    scl: int | None = None,
    lat: float | None = None,
    lon: float | None = None,
    captured_at: datetime | None = None,
) -> Observation:
    missing = [b for b in SENTINEL2_BANDS if b not in dn]
    if missing:
        raise ValueError(f"Missing Sentinel-2 bands: {missing}. Need {list(SENTINEL2_BANDS)}.")

    offset = boa_add_offset(processing_baseline)
    flags: list[str] = []
    reflectance: dict[str, float] = {}
    for band in SENTINEL2_BANDS:
        if dn[band] == 0:  # 0 is L2A's no-data value
            flags.append(f"no_data:{band}")
        r = (dn[band] + offset) / QUANTIFICATION_VALUE
        if not _REFLECTANCE_MIN <= r <= _REFLECTANCE_MAX:
            flags.append(f"reflectance_out_of_range:{band}")
        reflectance[band] = r

    if scl is not None and scl in _BAD_SCL:
        flags.append(f"scene_class:{_BAD_SCL[scl]}")

    nir, red = reflectance["B08"], reflectance["B04"]
    ndvi = (nir - red) / (nir + red) if (nir + red) > 0 else 0.0
    if ndvi > _NDVI_VEGETATION:
        flags.append(f"vegetated_ndvi:{ndvi:.2f}")

    return Observation(
        device="sentinel2",
        sensor_type="sentinel2",
        wavelengths_nm=band_centers(SENTINEL2_BANDS),
        values=[reflectance[b] for b in SENTINEL2_BANDS],
        units="reflectance",
        lat=lat,
        lon=lon,
        captured_at=captured_at,
        qc=QualityCheck(passed=not flags, flags=flags),
        provenance=Provenance(
            adapter="sentinel2",
            adapter_version=ADAPTER_VERSION,
            details={
                "processing_baseline": processing_baseline,
                "boa_add_offset": offset,
                "quantification_value": QUANTIFICATION_VALUE,
                "scl": scl,
                "ndvi": round(ndvi, 4),
            },
        ),
    )
