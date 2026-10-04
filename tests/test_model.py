"""Round-trip checks against the actual trained artifacts for all three
sensor types. Requires each sensor's dataset + model to exist:
  - lab:       scripts/download_data.py + build_dataset.py + train.py
  - sentinel2/as7265x: scripts/parse_usgs_splib.py + train.py (see README)
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import model  # noqa: E402
from app.constants import GRID_MAX_CM, GRID_MIN_CM, GRID_POINTS, MINERAL_CHOICES  # noqa: E402
from app.sensor_bands import AS7265X_BANDS, SENTINEL2_BANDS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "lab": ROOT / "data" / "processed" / "spectral_dataset.csv",
    "sentinel2": ROOT / "data" / "processed" / "sentinel2_dataset.csv",
    "as7265x": ROOT / "data" / "processed" / "as7265x_dataset.csv",
}


def _first_row(sensor_type: str):
    with open(DATASETS[sensor_type]) as f:
        reader = csv.reader(f)
        next(reader)  # header
        return next(reader)


def _x_values(sensor_type: str) -> list[float]:
    if sensor_type == "lab":
        return list(np.linspace(GRID_MIN_CM, GRID_MAX_CM, GRID_POINTS))
    bands = SENTINEL2_BANDS if sensor_type == "sentinel2" else AS7265X_BANDS
    return list(bands.values())


def _check_well_formed_and_self_predicts(sensor_type: str):
    row = _first_row(sensor_type)
    label = row[0]
    features = [float(v) for v in row[3:]]
    xs = _x_values(sensor_type)

    result = model.classify(xs, features, sensor_type=sensor_type)

    assert result["mineral_type"] in MINERAL_CHOICES
    assert 0 <= result["confidence_score"] <= 100
    assert len(result["alternatives"]) > 0
    total_prob = sum(a["probability"] for a in result["alternatives"])
    assert abs(total_prob - 1.0) < 1e-6
    # Not a guaranteed match for every row (this is a real classifier, not
    # a lookup), but the first training-set row predicting its own label
    # back is a cheap sanity check that preprocessing/inference wiring
    # hasn't silently broken.
    assert result["mineral_type"] == label, f"{sensor_type}: expected {label}, got {result['mineral_type']}"


def test_lab_classifier():
    _check_well_formed_and_self_predicts("lab")


def test_sentinel2_classifier():
    _check_well_formed_and_self_predicts("sentinel2")


def test_as7265x_classifier():
    _check_well_formed_and_self_predicts("as7265x")


if __name__ == "__main__":
    test_lab_classifier()
    test_sentinel2_classifier()
    test_as7265x_classifier()
    print("OK")
