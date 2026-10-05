"""Gold-pathfinder alteration indicator — deliberately NOT part of
app/model.py's mineral_type classification. This model never outputs
"gold": native gold has no diagnostic feature in Raman or VSWIR
reflectance (see README), so nothing here claims to detect it directly.
It outputs which alteration style (if any) a spectrum resembles —
iron_oxide_gossan, argillic_alteration, sulfide_pathfinder, or
background — the same indirect signal a geologist reads from an
outcrop before deciding where to actually sample for gold. See
scripts/pathfinder_mapping.py for the geological basis of each category.

sentinel2-trained is the only one with real validated signal so far —
see README "Gold pathfinder indicator" for why AS7265x's 410-940nm range
structurally can't see the ~2200nm clay/alunite feature that
argillic_alteration depends on most.
"""
import json

import joblib
import numpy as np

from app.config import settings
from app.sensor_bands import AS7265X_BANDS, SENTINEL2_BANDS, band_centers
from app.spectral import resample_to_bands

_pipelines: dict[str, object] = {}
_metas: dict[str, dict] = {}

_BAND_CENTERS_BY_SENSOR = {
    "sentinel2": band_centers(SENTINEL2_BANDS),
    "as7265x": band_centers(AS7265X_BANDS),
}

# Alteration categories that correlate with gold systems — summed into a
# single screening score. "background" is deliberately excluded (it's the
# complement, not a pathfinder signal).
_GOLD_ASSOCIATED_CATEGORIES = ("iron_oxide_gossan", "argillic_alteration", "sulfide_pathfinder")

# File-name prefix differs from the sensor_type key, matching the
# mineral_classifier_s2 convention already used in app/config.py.
_MODEL_NAME_BY_SENSOR = {"sentinel2": "pathfinder_classifier_s2", "as7265x": "pathfinder_classifier_as7265x"}


def _model_path(sensor_type: str, suffix: str) -> str:
    return str(settings.model_dir / f"{_MODEL_NAME_BY_SENSOR[sensor_type]}_v1.{suffix}")


def _load(sensor_type: str):
    if sensor_type not in _pipelines:
        _pipelines[sensor_type] = joblib.load(_model_path(sensor_type, "joblib"))
        _metas[sensor_type] = json.loads(open(_model_path(sensor_type, "meta.json")).read())
    return _pipelines[sensor_type], _metas[sensor_type]


def pathfinder_model_version(sensor_type: str = "sentinel2") -> str:
    _, meta = _load(sensor_type)
    return meta["version"]


def classify_pathfinder(x_values: list[float], intensities: list[float], sensor_type: str = "sentinel2") -> dict:
    """Returns {category, category_score (0-100), category_alternatives,
    pathfinder_score (0-100 — the three gold-associated categories summed),
    caveat}. Raises app.spectral.SpectrumRangeError on an out-of-range
    spectrum, KeyError if sensor_type isn't "sentinel2"/"as7265x"."""
    pipeline, _ = _load(sensor_type)
    xs, ys = np.asarray(x_values, dtype=float), np.asarray(intensities, dtype=float)
    features = resample_to_bands(xs, ys, _BAND_CENTERS_BY_SENSOR[sensor_type])

    probs = pipeline.predict_proba(features.reshape(1, -1))[0]
    classes = pipeline.classes_
    by_class = dict(zip(classes, probs))

    order = np.argsort(probs)[::-1]
    alternatives = [{"category": str(classes[i]), "probability": float(probs[i])} for i in order]
    pathfinder_score = round(sum(by_class.get(c, 0.0) for c in _GOLD_ASSOCIATED_CATEGORIES) * 100)

    return {
        "category": alternatives[0]["category"],
        "category_score": round(alternatives[0]["probability"] * 100),
        "category_alternatives": alternatives,
        "pathfinder_score": pathfinder_score,
        "caveat": (
            "This flags alteration chemistry associated with gold systems, not gold itself - "
            "native gold has no diagnostic Raman or VSWIR signature. Treat as a follow-up-worth "
            "flag for a geologist, not a detection."
        ),
    }
