import json

import joblib
import numpy as np

from app.config import settings
from app.spectral import resample

_pipeline = None
_meta: dict | None = None


def _load():
    global _pipeline, _meta
    if _pipeline is None:
        _pipeline = joblib.load(settings.model_path)
        _meta = json.loads(settings.meta_path.read_text())
    return _pipeline, _meta


def model_version() -> str:
    _, meta = _load()
    return meta["version"]


def classify(x_values: list[float], intensities: list[float]) -> dict:
    """Returns {mineral_type, confidence_score (0-100 int), alternatives:
    [{mineral, probability (0-1 float)}, ...]}, sorted most-likely first.

    Raises app.spectral.SpectrumRangeError if the input doesn't cover
    enough of the trained wavenumber range to classify.
    """
    pipeline, _ = _load()
    features = resample(np.asarray(x_values, dtype=float), np.asarray(intensities, dtype=float))
    probs = pipeline.predict_proba(features.reshape(1, -1))[0]
    classes = pipeline.classes_

    order = np.argsort(probs)[::-1]
    alternatives = [{"mineral": str(classes[i]), "probability": float(probs[i])} for i in order]

    return {
        "mineral_type": alternatives[0]["mineral"],
        "confidence_score": round(alternatives[0]["probability"] * 100),
        "alternatives": alternatives,
    }
