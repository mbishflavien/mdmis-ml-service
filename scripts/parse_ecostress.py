"""Parses the ECOSTRESS Spectral Library order (manually downloaded via
the JPL order form, see README "Getting more training data") and APPENDS
it to the existing USGS splib07-derived Sentinel-2/AS7265x datasets —
more samples per class from a second, independent collection (ECOSTRESS
absorbed the old ASTER Spectral Library; different specimens than USGS
splib07), not a replacement.

Format (confirmed from the actual downloaded file, not guessed): each
"*.spectrum.txt" has a clean header (Name/Class/Wavelength Range/...),
a blank line, then tab-separated "wavelength_um  reflectance_percent"
rows. "Wavelength Range: VSWIR" entries are what we want (0.4-2.5um,
covers both sensors); "TIR" entries (thermal infrared, 2.5-15.4um) are
skipped — same reason RRUFF's infrared category was skipped, wrong
region for either sensor. Reflectance is a 0-100 percent, unlike USGS
splib07's 0-1 fraction — converted here to stay consistent with the
already-trained models' feature scale.

Usage: python scripts/parse_ecostress.py
"""
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.sensor_bands import AS7265X_BANDS, SENTINEL2_BANDS, band_centers  # noqa: E402
from app.spectral import SpectrumRangeError, resample_to_bands  # noqa: E402
from scripts.mineral_mapping import MINERAL_NAME_MAP  # noqa: E402

ZIP_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "ecostress" / "ecospeclib_all_minerals.zip"
OUT_SENTINEL2 = Path(__file__).resolve().parent.parent / "data" / "processed" / "sentinel2_dataset.csv"
OUT_AS7265X = Path(__file__).resolve().parent.parent / "data" / "processed" / "as7265x_dataset.csv"

_NAME_RE = re.compile(r"^Name:\s*(.+)$")
_RANGE_RE = re.compile(r"^Wavelength Range:\s*(\S+)$")


def match_label(name: str) -> str | None:
    for key, label in MINERAL_NAME_MAP.items():
        if name.lower().startswith(key.lower()):
            return label
    return None


def parse_spectrum(text: str) -> tuple[str | None, str | None, np.ndarray, np.ndarray]:
    lines = text.splitlines()
    name, wavelength_range = None, None
    data_start = None
    for i, line in enumerate(lines):
        m = _NAME_RE.match(line)
        if m:
            name = m.group(1).strip()
        m = _RANGE_RE.match(line)
        if m:
            wavelength_range = m.group(1).strip()
        if line.strip() == "" and name is not None:
            data_start = i + 1
            break
    if data_start is None:
        return name, wavelength_range, np.array([]), np.array([])

    xs, ys = [], []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            xs.append(float(parts[0]) * 1000.0)  # um -> nm
            ys.append(float(parts[1]) / 100.0)   # percent -> fraction
        except ValueError:
            continue
    return name, wavelength_range, np.asarray(xs), np.asarray(ys)


def build_rows(zf: zipfile.ZipFile, centers: list[float]) -> tuple[list, dict[str, int]]:
    rows = []
    counts: dict[str, int] = {}
    for info in zf.infolist():
        if not info.filename.endswith(".spectrum.txt") or ".vswir." not in info.filename:
            continue
        name, wavelength_range, xs, ys = parse_spectrum(zf.read(info.filename).decode(errors="replace"))
        if name is None or wavelength_range != "VSWIR" or len(xs) < 5:
            continue
        label = match_label(name)
        if label is None:
            continue
        try:
            features = resample_to_bands(xs, ys, centers)
        except SpectrumRangeError:
            continue
        rows.append([label, name, info.filename] + list(features))
        counts[label] = counts.get(label, 0) + 1
    return rows, counts


def append_to(out_path: Path, new_rows: list, columns: list[str]) -> None:
    new_df = pd.DataFrame(new_rows, columns=columns)
    if out_path.exists():
        existing = pd.read_csv(out_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(out_path, index=False)
    print(f"  {out_path.name}: {len(new_df)} new rows appended, {len(combined)} total")


def main() -> None:
    if not ZIP_PATH.exists():
        print(f"{ZIP_PATH} not found — see README 'Getting more training data'.")
        sys.exit(1)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        print("=== Sentinel-2 ===")
        s2_rows, s2_counts = build_rows(zf, band_centers(SENTINEL2_BANDS))
        for label in sorted(s2_counts):
            print(f"  {label:12s} +{s2_counts[label]}")
        append_to(OUT_SENTINEL2, s2_rows, ["label", "mineral_name", "source"] + list(SENTINEL2_BANDS))

        print("\n=== AS7265x ===")
        as_rows, as_counts = build_rows(zf, band_centers(AS7265X_BANDS))
        for label in sorted(as_counts):
            print(f"  {label:12s} +{as_counts[label]}")
        append_to(OUT_AS7265X, as_rows, ["label", "mineral_name", "source"] + list(AS7265X_BANDS))


if __name__ == "__main__":
    main()
