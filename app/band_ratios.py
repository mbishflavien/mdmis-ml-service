"""Sentinel-2 band-ratio mineral alteration indices, per MDMIS_SRS_v2.0
REQ (Section on spectral preprocessing): "The system MUST apply spectral
band ratios as preprocessing features before classification. Minimum:
Iron Oxide (B11/B8), Carbonate (B11/B12), Clay (B11/B8A)."

These are standard remote-sensing lithology/alteration indices (not
specific to any one MINERAL_CHOICES class) — a ratio above its threshold
flags the *style* of alteration commonly associated with mineralisation,
which geologists use to prioritise where to send ground sensors. This is
deliberately rule-based, not a trained classifier: it needs zero labeled
data and is honest about being a coarse screening signal, not a mineral
identification.
"""
from dataclasses import dataclass


@dataclass
class BandRatioResult:
    iron_oxide: float
    carbonate: float
    clay: float
    ndvi: float
    vegetated: bool
    flags: list[str]


# Thresholds are the conventional starting points used in Sentinel-2
# lithological mapping literature — tune against real site data once
# available, not before (see README limitations).
_IRON_OXIDE_THRESHOLD = 1.1
_CARBONATE_THRESHOLD = 1.05
_CLAY_THRESHOLD = 1.1

# Verified against real data, not a guess: fetching real Sentinel-2 bands
# over the 10 seeded Rwandan sites (scripts/fetch_sentinel2.py) returned
# "carbonate_alteration" at all 10 — too uniform to be geology. Computing
# NDVI from the same bands showed why: those pixels are vegetated
# (NDVI 0.26-0.55), unlike the Cuprite, NV reference pixel (NDVI 0.07,
# bare ground) where the same formula correctly flagged its known
# iron-oxide/clay alteration zone. Vegetation canopy has its own B11/B12
# relationship (leaf water content) that these ratios were never designed
# to see through. NDVI > 0.2 is a conservative "likely vegetated, ratios
# unreliable here" cutoff — below it, treat flags as informative.
_NDVI_VEGETATION_THRESHOLD = 0.2


def compute_band_ratios(bands: dict[str, float]) -> BandRatioResult:
    """bands: {"B02": reflectance, "B03": ..., ...} — same keys as
    app.sensor_bands.SENTINEL2_BANDS. Raises KeyError if B04/B08/B8A/B11/B12
    are missing (all five are required for the ratios + the vegetation gate)."""
    iron_oxide = bands["B11"] / bands["B08"]
    carbonate = bands["B11"] / bands["B12"]
    clay = bands["B11"] / bands["B8A"]
    ndvi = (bands["B08"] - bands["B04"]) / (bands["B08"] + bands["B04"])
    vegetated = ndvi > _NDVI_VEGETATION_THRESHOLD

    flags = []
    if not vegetated:
        if iron_oxide > _IRON_OXIDE_THRESHOLD:
            flags.append("iron_oxide_alteration")
        if carbonate > _CARBONATE_THRESHOLD:
            flags.append("carbonate_alteration")
        if clay > _CLAY_THRESHOLD:
            flags.append("clay_alteration")

    return BandRatioResult(
        iron_oxide=iron_oxide, carbonate=carbonate, clay=clay,
        ndvi=ndvi, vegetated=vegetated, flags=flags,
    )
