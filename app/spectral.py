"""Shared spectrum preprocessing — used by both scripts/build_dataset.py
(training data) and app/model.py (live inference), so a request is
resampled/normalized exactly the way the training data was."""
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
