"""Shared spectrum preprocessing — used by both the build/parse scripts
(training data) and app/model.py (live inference), so a request is
resampled exactly the way the training data was.

Two different resample functions, not one, because the two data sources
behave differently:
  - resample() / GRID: Raman shift (cm^-1). RRUFF's raw intensity units
    aren't comparable across instruments/laser sessions, so each spectrum
    is additionally min-max normalized to [0,1] — only the peak *shape*
    is meaningful.
  - resample_to_bands(): Sentinel-2/AS7265x reflectance (0-1, physically
    meaningful as an absolute value — a brighter vs. darker material is
    real signal). No extra normalization here: scripts/parse_usgs_splib.py
    didn't normalize when building the training data, and StandardScaler
    in the sklearn Pipeline already handles feature scaling. Normalizing
    here too would make live requests inconsistent with what was trained.
"""
import numpy as np

from app.constants import GRID_MAX_CM, GRID_MIN_CM, GRID_POINTS

GRID = np.linspace(GRID_MIN_CM, GRID_MAX_CM, GRID_POINTS)

# Tolerate a small scan-range shortfall at either edge — np.interp holds the
# boundary value flat there (a conservative clamp), not invented signal.
MAX_EDGE_GAP_CM = 50.0


class SpectrumRangeError(ValueError):
    pass


def resample(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    if xs.min() > GRID_MIN_CM + MAX_EDGE_GAP_CM or xs.max() < GRID_MAX_CM - MAX_EDGE_GAP_CM:
        raise SpectrumRangeError(
            f"Spectrum only covers {xs.min():.1f}-{xs.max():.1f} cm^-1; "
            f"need roughly {GRID_MIN_CM:.0f}-{GRID_MAX_CM:.0f} cm^-1 to classify."
        )
    resampled = np.interp(GRID, xs, ys)
    span = resampled.max() - resampled.min()
    if span < 1e-9:
        raise SpectrumRangeError("Spectrum is flat (no signal variation) — cannot classify.")
    return (resampled - resampled.min()) / span


def resample_to_bands(xs: np.ndarray, ys: np.ndarray, centers: list[float]) -> np.ndarray:
    """centers: target band wavelengths in nm (app.sensor_bands). Edge
    tolerance scales with the grid span instead of a fixed constant, since
    this is reused for both Sentinel-2 (490-2190nm span) and AS7265x
    (410-940nm span)."""
    lo, hi = min(centers), max(centers)
    tolerance = 0.1 * (hi - lo)
    if xs.min() > lo + tolerance or xs.max() < hi - tolerance:
        raise SpectrumRangeError(
            f"Spectrum only covers {xs.min():.1f}-{xs.max():.1f} nm; need roughly {lo:.0f}-{hi:.0f} nm to classify."
        )
    return np.interp(centers, xs, ys)
