"""Exact band definitions for the sensors actually in MDMIS_IoT_Budget.docx
(downloaded 2026-09-16) — replaces the earlier Raman-shift grid in
app/constants.py for anything that isn't the lab/Raman sensor type.

Every other module (band_ratios.py, model_sentinel2.py, model_as7265x.py,
parse_ecostress.py) resamples spectra onto one of these grids rather than
a generic continuum, so training data and live sensor output are always
shaped the same way.
"""

# Sentinel-2 L2A, center wavelengths in nm (ESA band definitions). Only the
# bands the SRS's band-ratio formulas need plus the common visible set —
# not all 13 bands, since B1/B9/B10 (coastal aerosol/water vapour/cirrus)
# aren't used for mineral discrimination.
SENTINEL2_BANDS: dict[str, float] = {
    "B02": 490.0,   # blue
    "B03": 560.0,   # green
    "B04": 665.0,   # red
    "B05": 705.0,   # red edge
    "B06": 740.0,   # red edge
    "B07": 783.0,   # red edge
    "B08": 842.0,   # NIR
    "B8A": 865.0,   # narrow NIR
    "B11": 1610.0,  # SWIR 1
    "B12": 2190.0,  # SWIR 2
}
# B01 (coastal aerosol), B09 (water vapour) and B10 (cirrus) are deliberately
# excluded: they're atmospheric-correction support bands, not meant for
# surface characterization, and B10 isn't even present in the L2A surface-
# reflectance product fetch_sentinel2.py pulls from — training on it would
# make the model depend on a band it will never see live.

# AS7265x breakout (AMS AS72651/2/3 triad), 18 channels, nm. Datasheet
# center wavelengths — this is the actual ground-level point spectrometer
# in the IoT budget ($70 line item), so live requests from it will carry
# exactly these 18 values.
AS7265X_BANDS: dict[str, float] = {
    "A": 410.0, "B": 435.0, "C": 460.0, "D": 485.0, "E": 510.0, "F": 535.0,
    "G": 560.0, "H": 585.0, "R": 610.0, "I": 645.0, "S": 680.0, "J": 705.0,
    "T": 730.0, "U": 760.0, "V": 810.0, "W": 860.0, "K": 900.0, "L": 940.0,
}


def band_centers(bands: dict[str, float]) -> list[float]:
    return [bands[k] for k in bands]
