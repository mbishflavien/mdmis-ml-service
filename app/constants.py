"""Mirrors app/sites/models.py MINERAL_CHOICES in MDMIS_BACKEND-. Keep in sync
by hand — this service has no import path into the backend repo."""

MINERAL_CHOICES = (
    "cassiterite",
    "coltan",
    "wolframite",
    "gold",
    "beryl",
    "lithium",
    "cobalt",
    "copper",
    "gemstone",
    "unknown",
)

# Common wavenumber grid (cm^-1) every training/inference spectrum is
# resampled onto, so the classifier sees a fixed-length feature vector
# regardless of each RRUFF sample's original scan range/step.
GRID_MIN_CM = 150.0
GRID_MAX_CM = 1300.0
GRID_POINTS = 200
