"""Parses the USGS Digital Spectral Library splib07 (downloaded manually
via ScienceBase — https://www.sciencebase.gov/catalog/item/586e8c88e4b0f5ce109fccae,
see README "Getting training data") into training rows for both real
sensors in MDMIS_IoT_Budget.docx:

  - Sentinel-2: reads data/raw/usgs_splib/ASCIIdata_splib07b_rsSentinel2/
    ChapterM_Minerals/ — USGS has ALREADY resampled these onto Sentinel-2's
    13 official bands, so this is used close to as-is (just dropping the
    3 bands fetch_sentinel2.py can't get live — see sensor_bands.py).
  - AS7265x: reads data/raw/usgs_splib/ASCIIdata_splib07a/ChapterM_Minerals/
    — the *measured* (non-resampled) library, since there's no USGS-made
    AS7265x convolution. Each spectrum file is a column of reflectance
    values that must be paired by row-index with the matching
    splib07a_Wavelengths_<INSTRUMENT>_*.txt file (same directory) — the
    instrument code is embedded in the spectrum's filename. Only ASD/BECK
    instrument samples are used: NIC4 starts at 1.12µm, past the AS7265x's
    0.94µm top band, so it can't cover that sensor's range at all.

Usage: python scripts/parse_usgs_splib.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.sensor_bands import AS7265X_BANDS, SENTINEL2_BANDS, band_centers  # noqa: E402
from scripts.mineral_mapping import MINERAL_NAME_MAP  # noqa: E402

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "usgs_splib"
SPLIB07A_MINERALS = RAW_DIR / "ASCIIdata_splib07a" / "ChapterM_Minerals"
SPLIB07B_S2_MINERALS = RAW_DIR / "ASCIIdata_splib07b_rsSentinel2" / "ChapterM_Minerals"
SPLIB07A_ROOT = RAW_DIR / "ASCIIdata_splib07a"

OUT_SENTINEL2 = Path(__file__).resolve().parent.parent / "data" / "processed" / "sentinel2_dataset.csv"
OUT_AS7265X = Path(__file__).resolve().parent.parent / "data" / "processed" / "as7265x_dataset.csv"

_RECORD_RE = re.compile(r"Record=\d+:\s*(\S+)")
_BAD_VALUE = -1.23e34  # USGS's no-data sentinel

# Each spectrum filename embeds one of these instrument codes; map it to the
# matching Wavelengths_*.txt file in the same splib07a root.
_INSTRUMENT_WAVELENGTH_FILES = {
    "ASD": SPLIB07A_ROOT / "splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt",
    "BECK": SPLIB07A_ROOT / "splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt",
}


def mineral_name_from_header(path: Path) -> str | None:
    first_line = path.read_text(errors="replace").splitlines()[0]
    m = _RECORD_RE.search(first_line)
    return m.group(1) if m else None


def match_label(name: str) -> str | None:
    for key, label in MINERAL_NAME_MAP.items():
        if name.lower() == key.lower():
            return label
    return None


def load_values(path: Path) -> np.ndarray:
    values = []
    for line in path.read_text(errors="replace").splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        values.append(float(line))
    return np.asarray(values)


def build_sentinel2_dataset() -> pd.DataFrame:
    # Official band order in the USGS Sentinel-2 wavelengths file is
    # B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12 (ascending wavelength) —
    # confirmed by inspecting the file's 13 values against ESA's published
    # band centers. We only keep the columns matching SENTINEL2_BANDS.
    full_order = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
    keep_idx = [full_order.index(b) for b in SENTINEL2_BANDS]

    rows = []
    counts: dict[str, int] = {}
    for path in sorted(SPLIB07B_S2_MINERALS.glob("*.txt")):
        name = mineral_name_from_header(path)
        if name is None:
            continue
        label = match_label(name)
        if label is None:
            continue
        values = load_values(path)
        if len(values) != 13 or np.any(values <= _BAD_VALUE):
            continue
        selected = values[keep_idx]
        rows.append([label, name, path.name] + list(selected))
        counts[label] = counts.get(label, 0) + 1

    df = pd.DataFrame(rows, columns=["label", "mineral_name", "source"] + list(SENTINEL2_BANDS))
    print("Sentinel-2 per-class sample counts:")
    for label in sorted(counts):
        print(f"  {label:12s} {counts[label]:4d}")
    return df


def build_as7265x_dataset() -> pd.DataFrame:
    wavelength_cache: dict[str, np.ndarray] = {}

    def wavelengths_for(instrument: str) -> np.ndarray | None:
        if instrument not in wavelength_cache:
            path = _INSTRUMENT_WAVELENGTH_FILES.get(instrument)
            if path is None or not path.exists():
                wavelength_cache[instrument] = None
            else:
                wavelength_cache[instrument] = load_values(path) * 1000.0  # microns -> nm
        return wavelength_cache[instrument]

    centers = band_centers(AS7265X_BANDS)
    rows = []
    counts: dict[str, int] = {}
    skipped_instrument = 0

    for path in sorted(SPLIB07A_MINERALS.glob("*.txt")):
        name = mineral_name_from_header(path)
        if name is None:
            continue
        label = match_label(name)
        if label is None:
            continue

        instrument = next((code for code in _INSTRUMENT_WAVELENGTH_FILES if code in path.stem), None)
        if instrument is None:
            skipped_instrument += 1
            continue
        xs = wavelengths_for(instrument)
        if xs is None:
            continue
        ys = load_values(path)
        if len(xs) != len(ys):
            continue
        valid = ys > _BAD_VALUE
        xs, ys = xs[valid], ys[valid]
        if len(xs) < 5:
            continue

        if xs.min() > min(centers) or xs.max() < max(centers):
            continue  # doesn't cover the AS7265x's 410-940nm range
        resampled = np.interp(centers, xs, ys)

        rows.append([label, name, path.name] + list(resampled))
        counts[label] = counts.get(label, 0) + 1

    print(f"\n({skipped_instrument} files used an instrument (NIC4) that doesn't cover 410-940nm, skipped)")
    df = pd.DataFrame(rows, columns=["label", "mineral_name", "source"] + list(AS7265X_BANDS))
    print("AS7265x per-class sample counts:")
    for label in sorted(counts):
        print(f"  {label:12s} {counts[label]:4d}")
    return df


def main() -> None:
    if not SPLIB07A_MINERALS.exists() or not SPLIB07B_S2_MINERALS.exists():
        print(f"Expected data under {RAW_DIR} — see README 'Getting training data'.")
        sys.exit(1)

    OUT_SENTINEL2.parent.mkdir(parents=True, exist_ok=True)

    print("=== Sentinel-2 (ASCIIdata_splib07b_rsSentinel2) ===")
    s2_df = build_sentinel2_dataset()
    s2_df.to_csv(OUT_SENTINEL2, index=False)
    print(f"Wrote {len(s2_df)} rows to {OUT_SENTINEL2}")

    print("\n=== AS7265x (ASCIIdata_splib07a, resampled) ===")
    as_df = build_as7265x_dataset()
    as_df.to_csv(OUT_AS7265X, index=False)
    print(f"Wrote {len(as_df)} rows to {OUT_AS7265X}")


if __name__ == "__main__":
    main()
