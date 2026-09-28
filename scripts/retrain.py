"""Incremental-retraining loop: pulls every lab_confirmed MineralZone (plus
the raw spectral reading its ScanSession stored) from the live backend,
appends them to the public seed dataset, retrains, and writes a new
model version.

This is "incremental" in the sense of a scheduled retrain-on-accumulated-
data loop, not per-sample online learning — RandomForestClassifier has no
partial_fit. That's a deliberate choice (see the plan): far more robust
than true online learning on a dataset this small, and easy to explain.

Usage: python scripts/retrain.py --new-version v2
"""
import argparse
import csv
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402
from app.spectral import SpectrumRangeError, resample  # noqa: E402
import numpy as np  # noqa: E402

SEED_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "spectral_dataset.csv"


def fetch_confirmed_zones() -> list[dict]:
    """Calls the backend's service-key-protected retrain-data endpoint.
    Expected to not exist until MDMIS_BACKEND-'s
    GET /api/scans/retrain-data route is added (see the backend-changes
    part of the plan) — this script fails loudly rather than silently
    training on stale data if that endpoint is missing."""
    resp = httpx.get(
        f"{settings.backend_url}/scans/retrain-data",
        headers={"X-ML-Service-Key": settings.service_api_key},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def append_confirmed_rows(records: list[dict]) -> int:
    if not records:
        return 0
    with open(SEED_DATA_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        written = 0
        for rec in records:
            try:
                features = resample(
                    np.asarray(rec["x_values"], dtype=float),
                    np.asarray(rec["intensities"], dtype=float),
                )
            except SpectrumRangeError as e:
                print(f"  skip zone {rec.get('zone_id')}: {e}")
                continue
            writer.writerow(
                [rec["mineral_type"], rec.get("zone_id", ""), "backend:lab_confirmed"]
                + [f"{v:.6f}" for v in features]
            )
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-version", required=True, help="e.g. v2")
    args = parser.parse_args()

    print(f"Pulling lab_confirmed zones from {settings.backend_url} ...")
    records = fetch_confirmed_zones()
    print(f"Got {len(records)} confirmed zone(s).")

    added = append_confirmed_rows(records)
    print(f"Appended {added} new training row(s) to {SEED_DATA_PATH}")

    print(f"Retraining as {args.new_version} ...")
    import subprocess

    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("train.py")), "--version", args.new_version],
        check=True,
    )


if __name__ == "__main__":
    main()
