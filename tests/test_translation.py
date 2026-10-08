"""AS7265x translation: raw counts -> reflectance -> same model output as
the known reflectance. Simulates what the chip would print for a real
training spectrum under a lamp, so it needs data/processed/as7265x_dataset.csv
and the as7265x model (see README Setup)."""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import model  # noqa: E402
from app.translation import as7265x  # noqa: E402

DATASET = Path(__file__).resolve().parent.parent / "data" / "processed" / "as7265x_dataset.csv"
PANEL = 0.99


def _real_reflectance() -> np.ndarray:
    with open(DATASET) as f:
        reader = csv.reader(f)
        next(reader)
        return np.asarray([float(v) for v in next(reader)[3:]])


def _simulate_device(reflectance: np.ndarray):
    """What the chip would print: a dark offset, a white panel reading that
    varies per channel (lamp spectrum + channel sensitivity differ), and
    the sample somewhere between them."""
    rng = np.random.default_rng(0)
    dark = 300 + rng.uniform(-20, 20, as7265x.N_CHANNELS)
    white = dark + rng.uniform(8000, 30000, as7265x.N_CHANNELS)
    sample = dark + reflectance / PANEL * (white - dark)
    return sample.tolist(), dark.tolist(), white.tolist()


def test_round_trip_recovers_reflectance_and_prediction():
    truth = _real_reflectance()
    sample, dark, white = _simulate_device(truth)

    obs = as7265x.translate(sample, dark, white, white_reference_reflectance=PANEL)

    assert obs.qc.passed, obs.qc.flags
    assert np.allclose(obs.values, truth, atol=1e-9)
    direct = model.classify(obs.wavelengths_nm, truth.tolist(), sensor_type="as7265x")
    via_device = model.classify(obs.wavelengths_nm, obs.values, sensor_type="as7265x")
    assert direct["mineral_type"] == via_device["mineral_type"]


def test_raw_counts_without_calibration_differ():
    # The reason this layer exists: feeding raw counts straight in is not
    # the same input the model was trained on.
    truth = _real_reflectance()
    sample, _, _ = _simulate_device(truth)
    assert not np.allclose(sample, truth)


def test_saturated_channel_is_flagged():
    sample, dark, white = _simulate_device(_real_reflectance())
    sample[3] = 65535
    obs = as7265x.translate(sample, dark, white, white_reference_reflectance=PANEL)
    assert not obs.qc.passed
    assert "saturated_sample:D" in obs.qc.flags


def test_white_not_above_dark_is_flagged():
    sample, dark, white = _simulate_device(_real_reflectance())
    white[0] = dark[0]
    obs = as7265x.translate(sample, dark, white, white_reference_reflectance=PANEL)
    assert "white_not_above_dark:A" in obs.qc.flags


def test_wrong_channel_count_is_rejected():
    try:
        as7265x.translate([1.0] * 17, [0.0] * 18, [2.0] * 18)
    except ValueError:
        return
    raise AssertionError("expected ValueError for 17 channels")


def test_parse_serial_line():
    line = ",".join(str(i) for i in range(18)) + ","  # trailing comma, as the example sketch prints
    assert as7265x.parse_csv_line(line) == [float(i) for i in range(18)]


S2_DATASET = Path(__file__).resolve().parent.parent / "data" / "processed" / "sentinel2_dataset.csv"


def _s2_real_reflectance() -> dict[str, float]:
    from app.sensor_bands import SENTINEL2_BANDS

    with open(S2_DATASET) as f:
        row = next(csv.DictReader(f))
    return {b: float(row[b]) for b in SENTINEL2_BANDS}


def _to_l2a_dn(reflectance: dict[str, float], offset: int) -> dict[str, float]:
    # What an L2A file stores for that reflectance: DN = reflectance*10000 - offset
    return {b: round(r * 10000 - offset) for b, r in reflectance.items()}


def test_s2_new_baseline_applies_offset_and_round_trips():
    from app.translation import sentinel2

    truth = _s2_real_reflectance()
    obs = sentinel2.translate(_to_l2a_dn(truth, -1000), "05.10", scl=5)
    assert obs.provenance.details["boa_add_offset"] == -1000
    assert np.allclose(obs.values, list(truth.values()), atol=1e-4)
    if obs.qc.passed:
        direct = model.classify(obs.wavelengths_nm, list(truth.values()), sensor_type="sentinel2")
        translated = model.classify(obs.wavelengths_nm, obs.values, sensor_type="sentinel2")
        assert direct["mineral_type"] == translated["mineral_type"]


def test_s2_old_baseline_has_no_offset():
    from app.translation import sentinel2

    truth = _s2_real_reflectance()
    obs = sentinel2.translate(_to_l2a_dn(truth, 0), "03.01")
    assert obs.provenance.details["boa_add_offset"] == 0
    assert np.allclose(obs.values, list(truth.values()), atol=1e-4)


def test_s2_ignoring_offset_reads_too_bright():
    # The bug this adapter fixes: dividing a new-baseline DN by 10000
    # without the offset reads 0.1 too bright in every band.
    truth = _s2_real_reflectance()
    dn = _to_l2a_dn(truth, -1000)
    naive = np.array([v / 10000 for v in dn.values()])
    assert np.allclose(naive - np.array(list(truth.values())), 0.1, atol=1e-4)


def test_s2_cloud_and_vegetation_are_flagged():
    from app.translation import sentinel2

    truth = _s2_real_reflectance()
    cloudy = sentinel2.translate(_to_l2a_dn(truth, -1000), "05.10", scl=9)
    assert "scene_class:cloud_high_probability" in cloudy.qc.flags

    leafy = dict(truth, B04=0.04, B08=0.40)  # typical vegetation: dark red, bright NIR
    veg = sentinel2.translate(_to_l2a_dn(leafy, -1000), "05.10")
    assert any(f.startswith("vegetated_ndvi") for f in veg.qc.flags)


def test_s2_missing_band_is_rejected():
    from app.translation import sentinel2

    dn = _to_l2a_dn(_s2_real_reflectance(), -1000)
    del dn["B11"]
    try:
        sentinel2.translate(dn, "05.10")
    except ValueError as e:
        assert "B11" in str(e)
        return
    raise AssertionError("expected ValueError for missing B11")


if __name__ == "__main__":
    test_s2_new_baseline_applies_offset_and_round_trips()
    test_s2_old_baseline_has_no_offset()
    test_s2_ignoring_offset_reads_too_bright()
    test_s2_cloud_and_vegetation_are_flagged()
    test_s2_missing_band_is_rejected()
    test_round_trip_recovers_reflectance_and_prediction()
    test_raw_counts_without_calibration_differ()
    test_saturated_channel_is_flagged()
    test_white_not_above_dark_is_flagged()
    test_wrong_channel_count_is_rejected()
    test_parse_serial_line()
    print("OK")
