"""Round-trip check against the actual trained artifact — requires
models/mineral_classifier_v1.joblib to exist (run scripts/download_data.py
+ build_dataset.py + train.py first)."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import model  # noqa: E402
from app.constants import MINERAL_CHOICES  # noqa: E402

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "spectral_dataset.csv"


def _first_row():
    with open(DATA_PATH) as f:
        reader = csv.reader(f)
        next(reader)  # header
        return next(reader)


def test_classify_returns_well_formed_response():
    row = _first_row()
    features = [float(v) for v in row[3:]]
    import numpy as np
    from app.constants import GRID_MIN_CM, GRID_MAX_CM, GRID_POINTS

    xs = list(np.linspace(GRID_MIN_CM, GRID_MAX_CM, GRID_POINTS))

    result = model.classify(xs, features)

    assert result["mineral_type"] in MINERAL_CHOICES
    assert 0 <= result["confidence_score"] <= 100
    assert len(result["alternatives"]) > 0
    total_prob = sum(a["probability"] for a in result["alternatives"])
    assert abs(total_prob - 1.0) < 1e-6


def test_known_sample_predicts_its_own_label():
    row = _first_row()
    label = row[0]
    features = [float(v) for v in row[3:]]
    import numpy as np
    from app.constants import GRID_MIN_CM, GRID_MAX_CM, GRID_POINTS

    xs = list(np.linspace(GRID_MIN_CM, GRID_MAX_CM, GRID_POINTS))

    result = model.classify(xs, features)
    # Not a guaranteed match for every row (this is a real classifier, not
    # a lookup), but the first training-set row predicting its own label
    # back is a cheap sanity check that preprocessing/inference wiring
    # hasn't silently broken.
    assert result["mineral_type"] == label


if __name__ == "__main__":
    test_classify_returns_well_formed_response()
    test_known_sample_predicts_its_own_label()
    print("OK")
