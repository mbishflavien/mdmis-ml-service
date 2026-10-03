"""Parses the ECOSTRESS Spectral Library "All Minerals" bulk download
(speclib.jpl.nasa.gov) into training rows resampled onto BOTH the
Sentinel-2 and AS7265x band grids (app.sensor_bands) — genuine VNIR-SWIR
reflectance spectra, unlike the earlier RRUFF Raman-shift data, so this
actually matches what your hyperspectral/multispectral sensors measure.

ECOSTRESS can't be bulk-downloaded by a script (its download page is a
JS shopping-cart UI with no underlying API — verified, not guessed: a
plone.restapi @search probe found no indexed spectrum content, and
POSTing the visible form fields to checkout-form didn't trigger a
download either). A human click-through is the fastest real path —
see README.md "Getting the ECOSTRESS data" for the 2-minute steps.

Expected input: the unzipped "All Minerals" bundle in data/raw/ecostress/
— each sample is a .spectrum.txt file: ~20 header lines starting with
"X:", "Y:", "Name:", etc., a blank line, then whitespace-separated
"wavelength_um  reflectance" pairs (wavelength in micrometres per the
ECOSTRESS format spec).

Usage: python scripts/parse_ecostress.py
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.sensor_bands import AS7265X_BANDS, SENTINEL2_BANDS, band_centers  # noqa: E402
from scripts.mineral_mapping import MINERAL_NAME_MAP  # noqa: E402

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "ecostress"
OUT_SENTINEL2 = Path(__file__).resolve().parent.parent / "data" / "processed" / "sentinel2_dataset.csv"
OUT_AS7265X = Path(__file__).resolve().parent.parent / "data" / "processed" / "as7265x_dataset.csv"

# ECOSTRESS "Name:" values are full mineral names (sometimes with locality/
# variety suffixes, e.g. "Quartz GDS73"), so this matches the mapping key
# as a prefix rather than requiring an exact match like the RRUFF filename
# parser did.
_NAME_RE = re.compile(r"^Name:\s*(.+)$", re.IGNORECASE)


def parse_spectrum_file(path: Path) -> tuple[str | None, np.ndarray, np.ndarray]:
    name = None
    xs, ys = [], []
    in_data = False
    for line in path.read_text(errors="replace").splitlines():
        if not in_data:
            m = _NAME_RE.match(line.strip())
            if m:
                name = m.group(1).strip()
            if line.strip() == "" and name is not None:
                in_data = True
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            wavelength_um, reflectance = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        xs.append(wavelength_um * 1000.0)  # um -> nm, to match sensor_bands.py
        ys.append(reflectance)
    return name, np.asarray(xs), np.asarray(ys)


def match_label(name: str) -> str | None:
    for key, label in MINERAL_NAME_MAP.items():
        if name.lower().startswith(key.lower()):
            return label
    return None


def resample_to_bands(xs: np.ndarray, ys: np.ndarray, bands: dict[str, float]) -> np.ndarray | None:
    centers = band_centers(bands)
    if xs.min() > min(centers) or xs.max() < max(centers):
        return None  # doesn't cover the sensor's range
    return np.interp(centers, xs, ys)


def main() -> None:
    files = sorted(RAW_DIR.glob("*.spectrum.txt")) or sorted(RAW_DIR.glob("*.txt"))
    if not files:
        print(f"No spectrum files found in {RAW_DIR}.")
        print("See README.md 'Getting the ECOSTRESS data' for how to get them there.")
        sys.exit(1)

    rows_s2, rows_as = [], []
    counts: dict[str, int] = {}
    unmatched = 0

    for path in files:
        name, xs, ys = parse_spectrum_file(path)
        if name is None or len(xs) < 5:
            continue
        label = match_label(name)
        if label is None:
            unmatched += 1
            continue

        s2 = resample_to_bands(xs, ys, SENTINEL2_BANDS)
        if s2 is not None:
            rows_s2.append([label, name, path.name] + list(s2))

        as7265x = resample_to_bands(xs, ys, AS7265X_BANDS)
        if as7265x is not None:
            rows_as.append([label, name, path.name] + list(as7265x))

        counts[label] = counts.get(label, 0) + 1

    for out_path, rows, bands in [(OUT_SENTINEL2, rows_s2, SENTINEL2_BANDS), (OUT_AS7265X, rows_as, AS7265X_BANDS)]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "mineral_name", "source"] + list(bands))
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {out_path}")

    print(f"\n{unmatched} files had no matching mineral_mapping.py entry (expected — most of")
    print("ECOSTRESS's 3104 minerals aren't in MINERAL_CHOICES).")
    print("\nPer-class sample counts:")
    for label in sorted(counts):
        flag = "  <-- too few for training" if counts[label] < 5 else ""
        print(f"  {label:12s} {counts[label]:4d}{flag}")


if __name__ == "__main__":
    main()
