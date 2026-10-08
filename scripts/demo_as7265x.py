"""Walk one AS7265x reading through the translation step by step.

No hardware needed: picks a real mineral spectrum (known reflectance) from
data/processed/as7265x_dataset.csv, simulates the raw counts the chip
would print for it under a lamp, then translates those counts back and
shows every intermediate number, plus what the models make of it.

Usage:
  python scripts/demo_as7265x.py                      # random mineral
  python scripts/demo_as7265x.py --mineral copper
  python scripts/demo_as7265x.py --saturate R         # watch QC block a clipped channel
  python scripts/demo_as7265x.py --json               # also print a request body for /readings/as7265x
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import model, pathfinder  # noqa: E402
from app.sensor_bands import AS7265X_BANDS  # noqa: E402
from app.translation import as7265x  # noqa: E402

DATASET = Path(__file__).resolve().parent.parent / "data" / "processed" / "as7265x_dataset.csv"


def pick_spectrum(mineral: str | None):
    with open(DATASET) as f:
        rows = list(csv.DictReader(f))
    if mineral:
        rows = [r for r in rows if r["label"] == mineral]
        if not rows:
            sys.exit(f"No '{mineral}' rows. Labels: {sorted({r['label'] for r in csv.DictReader(open(DATASET))})}")
    row = random.choice(rows)
    return row, np.array([float(row[ch]) for ch in AS7265X_BANDS])


def simulate_device(reflectance: np.ndarray, panel: float, seed: int):
    """What the chip would print. Each channel sees a different lamp
    brightness and has a different sensitivity, so the white panel gives a
    different count per channel; dark is the sensor's own small offset."""
    rng = np.random.default_rng(seed)
    dark = np.round(300 + rng.uniform(-20, 20, len(reflectance)))
    white = np.round(dark + rng.uniform(8000, 30000, len(reflectance)))
    sample = np.round(dark + reflectance / panel * (white - dark))
    return sample, dark, white


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mineral")
    p.add_argument("--panel", type=float, default=0.99, help="white panel reflectance (Spectralon ~0.99, PTFE ~0.95)")
    p.add_argument("--saturate", help="channel letter to force to 65535, e.g. R")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    random.seed(a.seed)

    row, truth = pick_spectrum(a.mineral)
    sample, dark, white = simulate_device(truth, a.panel, a.seed)
    if a.saturate:
        sample[list(AS7265X_BANDS).index(a.saturate.upper())] = 65535

    print(f"\nSpectrum: {row['mineral_name']}  (class '{row['label']}', source {row['source']})")
    print(f"White panel reflectance: {a.panel}\n")

    print("STEP 1 - what the sensor prints (raw counts, NOT reflectance):")
    print("  sample:", ",".join(f"{v:.0f}" for v in sample))
    print("  dark:  ", ",".join(f"{v:.0f}" for v in dark))
    print("  white: ", ",".join(f"{v:.0f}" for v in white))

    print("\nSTEP 2 - per channel: reflectance = (sample - dark) / (white - dark) x panel")
    print(f"  {'ch':2} {'nm':>4} {'sample-dark':>11} {'white-dark':>10} {'ratio':>7} {'x panel':>8} {'true':>7}")
    for i, (ch, nm) in enumerate(AS7265X_BANDS.items()):
        num, den = sample[i] - dark[i], white[i] - dark[i]
        print(f"  {ch:2} {nm:4.0f} {num:11.0f} {den:10.0f} {num/den:7.4f} {num/den*a.panel:8.4f} {truth[i]:7.4f}")
    print("  ('true' = the library's measured reflectance; small differences come from")
    print("   the sensor printing whole-number counts)")

    obs = as7265x.translate(sample.tolist(), dark.tolist(), white.tolist(), white_reference_reflectance=a.panel)

    print("\nSTEP 3 - quality check:", "PASSED" if obs.qc.passed else f"FAILED {obs.qc.flags}")
    if not obs.qc.passed:
        print("  -> the models are NOT run on this reading.")
        return

    print("\nSTEP 4 - what the model receives:")
    print("  wavelengths_nm:", [round(w) for w in obs.wavelengths_nm])
    print("  values:        ", [round(v, 4) for v in obs.values])

    m = model.classify(obs.wavelengths_nm, obs.values, sensor_type="as7265x")
    pf = pathfinder.classify_pathfinder(obs.wavelengths_nm, obs.values, sensor_type="as7265x")
    print("\nSTEP 5 - model output")
    print(f"  mineral:    {m['mineral_type']} ({m['confidence_score']}%)   true class: {row['label']}")
    print("  top 3:     ", ", ".join(f"{x['mineral']} {x['probability']:.2f}" for x in m["alternatives"][:3]))
    print(f"  pathfinder: {pf['category']} (score {pf['pathfinder_score']})")

    raw = model.classify(obs.wavelengths_nm, sample.tolist(), sensor_type="as7265x")
    print("\nFOR COMPARISON - raw counts fed straight in, skipping translation:")
    print(f"  mineral:    {raw['mineral_type']} ({raw['confidence_score']}%)  <- not meaningful: the model was")
    print("              trained on reflectance (0-1), these values are in the thousands")

    if a.json:
        body = {"sample": ",".join(f"{v:.0f}" for v in sample), "dark": ",".join(f"{v:.0f}" for v in dark),
                "white": ",".join(f"{v:.0f}" for v in white), "white_reference_reflectance": a.panel}
        print("\nRequest body for POST /readings/as7265x:")
        print(json.dumps(body))


if __name__ == "__main__":
    main()
