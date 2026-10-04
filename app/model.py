import json

import joblib
import numpy as np

from app.config import settings
from app.sensor_bands import AS7265X_BANDS, SENTINEL2_BANDS, band_centers
from app.spectral import resample, resample_to_bands

_pipelines: dict[str, object] = {}
_metas: dict[str, dict] = {}

_BAND_CENTERS_BY_SENSOR = {
    "sentinel2": band_centers(SENTINEL2_BANDS),
    "as7265x": band_centers(AS7265X_BANDS),
}


def _load(sensor_type: str):
    if sensor_type not in _pipelines:
        _pipelines[sensor_type] = joblib.load(settings.model_path(sensor_type))
        _metas[sensor_type] = json.loads(settings.meta_path(sensor_type).read_text())
    return _pipelines[sensor_type], _metas[sensor_type]


def model_version(sensor_type: str = "lab") -> str:
    _, meta = _load(sensor_type)
    return meta["version"]


def classify(x_values: list[float], intensities: list[float], sensor_type: str = "lab") -> dict:
    """Returns {mineral_type, confidence_score (0-100 int), alternatives:
    [{mineral, probability (0-1 float)}, ...]}, sorted most-likely first.

    Raises app.spectral.SpectrumRangeError if the input doesn't cover
    enough of the trained range to classify. sensor_type picks both the
    model and the resampling grid — "lab" is Raman shift (cm^-1) with
    per-spectrum normalization, "sentinel2"/"as7265x" are wavelength (nm)
    reflectance resampled onto that sensor's exact band centers, no extra
    normalization (see app/spectral.py for why).
    """
    pipeline, _ = _load(sensor_type)
    xs, ys = np.asarray(x_values, dtype=float), np.asarray(intensities, dtype=float)

    if sensor_type in _BAND_CENTERS_BY_SENSOR:
        features = resample_to_bands(xs, ys, _BAND_CENTERS_BY_SENSOR[sensor_type])
    else:
        features = resample(xs, ys)

    probs = pipeline.predict_proba(features.reshape(1, -1))[0]
    classes = pipeline.classes_

    order = np.argsort(probs)[::-1]
    alternatives = [{"mineral": str(classes[i]), "probability": float(probs[i])} for i in order]

    return {
        "mineral_type": alternatives[0]["mineral"],
        "confidence_score": round(alternatives[0]["probability"] * 100),
        "alternatives": alternatives,
    }
