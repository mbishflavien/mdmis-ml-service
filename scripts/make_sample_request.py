"""Convenience for manual testing: prints (or saves) a ready-to-use
/classify request body built from a real spectrum in the training set, so
you don't have to hand-write 200 floats to try the API yourself.

Usage:
  python scripts/make_sample_request.py                  # random sample, prints JSON
  python scripts/make_sample_request.py --mineral gold    # a specific class
  python scripts/make_sample_request.py --mineral copper --out request.json
  python scripts/make_sample_request.py --list            # show available classes + counts
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.constants import GRID_MAX_CM, GRID_MIN_CM, GRID_POINTS  # noqa: E402

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "spectral_dataset.csv"


def load_rows() -> list[list[str]]:
    if not DATA_PATH.exists():
        print(f"{DATA_PATH} not found — run download_data.py + build_dataset.py first.")
        sys.exit(1)
    with open(DATA_PATH) as f:
        reader = csv.reader(f)
        next(reader)  # header
        return list(reader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mineral", help="e.g. cassiterite, beryl, unknown — omit for a random one")
    parser.add_argument("--out", help="write JSON here instead of printing it")
    parser.add_argument("--list", action="store_true", help="list available classes and sample counts")
    args = parser.parse_args()

    rows = load_rows()

    if args.list:
        counts = Counter(r[0] for r in rows)
        for label, n in sorted(counts.items()):
            print(f"  {label:12s} {n}")
        print("\n(note: 'gold' has 0 rows - see scripts/mineral_mapping.py for why)")
        return

    candidates = [r for r in rows if args.mineral is None or r[0] == args.mineral]
    if not candidates:
        print(f"No samples for '{args.mineral}'. Run with --list to see what's available.")
        sys.exit(1)

    row = random.choice(candidates)
    true_label = row[0]
    intensities = [float(v) for v in row[3:]]
    x_values = list(np.linspace(GRID_MIN_CM, GRID_MAX_CM, GRID_POINTS))

    body = {"x_values": x_values, "intensities": intensities, "sensor_type": "lab"}

    if args.out:
        Path(args.out).write_text(json.dumps(body))
        print(f"Wrote {args.out} (true label: {true_label})")
        print(f"Try: curl -X POST http://localhost:8100/classify "
              f"-H \"X-ML-Service-Key: <your SERVICE_API_KEY>\" -H \"Content-Type: application/json\" "
              f"-d @{args.out}")
    else:
        print(f"# true label: {true_label}")
        print(json.dumps(body))


if __name__ == "__main__":
    main()
