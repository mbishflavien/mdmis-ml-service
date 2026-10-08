"""AS7265x raw counts -> calibrated reflectance Observation.

The chip outputs light *intensity* per channel, which depends on the
lamp, distance, angle and exposure — not reflectance, which is what the
sentinel2/as7265x models were trained on (USGS/ECOSTRESS lab spectra).
Feeding raw counts straight to the model gives confident nonsense.

The standard fix is reference-panel calibration, done in the field with
the same lamp/distance/gain as the sample:
  - dark:  a reading with the light blocked (sensor's own offset/noise)
  - white: a reading of a white reference panel of known reflectance
           (Spectralon ~0.99, PTFE ~0.95)
  reflectance = (sample - dark) / (white - dark) * panel_reflectance

Channel order is the SparkFun AS7265x Arduino library's print order
(A,B,C,D,E,F,G,H,R,I,S,J,T,U,V,W,K,L), the same order as
app.sensor_bands.AS7265X_BANDS.
"""
from datetime import datetime

import numpy as np

from app.sensor_bands import AS7265X_BANDS, band_centers
from app.translation.observation import Observation, Provenance, QualityCheck

ADAPTER_VERSION = "1.0"
CHANNELS = list(AS7265X_BANDS)
N_CHANNELS = len(CHANNELS)

# The chip's ADC is 16-bit; a raw value at the ceiling means the channel
# clipped and its true value is unknown.
_RAW_SATURATION = 65535
# Reflectance a little outside [0, 1] is normal noise/geometry; well
# outside it means the calibration readings don't match the sample's
# conditions. Starting points, to be tuned on real field readings.
_REFLECTANCE_MIN = -0.02
_REFLECTANCE_MAX = 1.2
_MIN_WHITE_MINUS_DARK = 1e-6


def parse_csv_line(line: str) -> list[float]:
    """One SparkFun-style serial line: 18 comma-separated numbers."""
    values = [float(v) for v in line.replace(";", ",").split(",") if v.strip()]
    if len(values) != N_CHANNELS:
        raise ValueError(f"Expected {N_CHANNELS} AS7265x channel values, got {len(values)}.")
    return values


def translate(
    sample: list[float],
    dark: list[float],
    white: list[float],
    white_reference_reflectance: float = 0.99,
    lat: float | None = None,
    lon: float | None = None,
    depth_m: float | None = None,
    captured_at: datetime | None = None,
) -> Observation:
    for name, arr in (("sample", sample), ("dark", dark), ("white", white)):
        if len(arr) != N_CHANNELS:
            raise ValueError(f"{name} must have {N_CHANNELS} values (channels {','.join(CHANNELS)}), got {len(arr)}.")
    if not 0 < white_reference_reflectance <= 1:
        raise ValueError("white_reference_reflectance must be in (0, 1].")

    s, d, w = (np.asarray(a, dtype=float) for a in (sample, dark, white))
    flags: list[str] = []

    for ch, sv, wv in zip(CHANNELS, s, w):
        if sv >= _RAW_SATURATION:
            flags.append(f"saturated_sample:{ch}")
        if wv >= _RAW_SATURATION:
            flags.append(f"saturated_white:{ch}")

    span = w - d
    reflectance = np.full(N_CHANNELS, np.nan)
    ok = span > _MIN_WHITE_MINUS_DARK
    for ch in np.array(CHANNELS)[~ok]:
        flags.append(f"white_not_above_dark:{ch}")
    reflectance[ok] = (s[ok] - d[ok]) / span[ok] * white_reference_reflectance

    for ch, r in zip(CHANNELS, reflectance):
        if not np.isnan(r) and not _REFLECTANCE_MIN <= r <= _REFLECTANCE_MAX:
            flags.append(f"reflectance_out_of_range:{ch}")

    return Observation(
        device="as7265x",
        sensor_type="as7265x",
        wavelengths_nm=band_centers(AS7265X_BANDS),
        # NaN isn't valid JSON; a failed channel is reported via qc.flags
        # and the models are never run on a reading that has any flag.
        values=[0.0 if np.isnan(r) else float(r) for r in reflectance],
        units="reflectance",
        lat=lat,
        lon=lon,
        depth_m=depth_m,
        captured_at=captured_at,
        qc=QualityCheck(passed=not flags, flags=flags),
        provenance=Provenance(
            adapter="as7265x",
            adapter_version=ADAPTER_VERSION,
            details={"method": "dark/white reference panel", "white_reference_reflectance": white_reference_reflectance},
        ),
    )
