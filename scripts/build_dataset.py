"""Cleans the raw RRUFF Raman zips into a fixed-width feature dataset.

For every RRUFF sample whose mineral name is in mineral_mapping.py:
  1. parse the "Raman_Data_Processed" text file (baseline already removed
     by RRUFF) into (wavenumber_cm-1, intensity) pairs
  2. resample onto the common grid in app.constants (linear interpolation)
  3. min-max normalize the intensities to [0, 1] (RRUFF intensity units
     aren't comparable across instruments/sessions, only shape matters)
  4. write one row per spectrum to data/processed/spectral_dataset.csv:
     label, rruff_id, source_zip, f0..f{GRID_POINTS-1}

Prints a per-class sample count at the end — this is where a real
coverage gap (e.g. "gold: 0 samples") is supposed to show up loudly
rather than be silently dropped.

Usage: python scripts/build_dataset.py
"""
import csv
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.constants import GRID_POINTS  # noqa: E402
from app.spectral import SpectrumRangeError, resample  # noqa: E402
from scripts.mineral_mapping import MINERAL_NAME_MAP  # noqa: E402

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "rruff_raman"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "spectral_dataset.csv"


def parse_spectrum(text: str) -> tuple[np.ndarray, np.ndarray] | None:
    xs, ys = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("##"):
            continue
        parts = line.split(",")
        if len(parts) != 2:
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        xs.append(x)
        ys.append(y)
    if len(xs) < 10:
        return None
    order = np.argsort(xs)
    return np.asarray(xs)[order], np.asarray(ys)[order]


def try_resample(xs: np.ndarray, ys: np.ndarray) -> np.ndarray | None:
    try:
        return resample(xs, ys)
    except SpectrumRangeError:
        return None


def iter_target_files(zip_path: Path):
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if "Raman_Data_Processed" not in name:
                continue
            mineral_name = name.split("__", 1)[0]
            label = MINERAL_NAME_MAP.get(mineral_name)
            if label is None:
                continue
            rruff_id = name.split("__")[1] if "__" in name else ""
            yield label, rruff_id, name, z.read(name).decode("utf-8", errors="replace")


def main() -> None:
    zips = sorted(RAW_DIR.glob("*.zip"))
    if not zips:
        print(f"No zips found in {RAW_DIR} — run scripts/download_data.py first.")
        sys.exit(1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {label: 0 for label in set(MINERAL_NAME_MAP.values())}
    rows_written = 0
    seen_rruff_ids: set[str] = set()  # de-dupe: the two tiers can overlap

    with open(OUT_PATH, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["label", "rruff_id", "source"] + [f"f{i}" for i in range(GRID_POINTS)])

        for zip_path in zips:
            print(f"[reading] {zip_path.name}")
            for label, rruff_id, source, text in iter_target_files(zip_path):
                dedupe_key = f"{label}:{rruff_id}:{source}"
                if dedupe_key in seen_rruff_ids:
                    continue
                seen_rruff_ids.add(dedupe_key)

                parsed = parse_spectrum(text)
                if parsed is None:
                    continue
                features = try_resample(*parsed)
                if features is None:
                    continue

                writer.writerow([label, rruff_id, source] + [f"{v:.6f}" for v in features])
                counts[label] += 1
                rows_written += 1

    print(f"\nWrote {rows_written} spectra to {OUT_PATH}\n")
    print("Per-class sample counts (MINERAL_CHOICES coverage):")
    for label in sorted(counts):
        flag = "  <-- too few for training, see mineral_mapping.py" if counts[label] < 5 else ""
        print(f"  {label:12s} {counts[label]:4d}{flag}")


if __name__ == "__main__":
    main()
