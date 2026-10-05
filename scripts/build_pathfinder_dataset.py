"""Builds the gold-pathfinder alteration-indicator datasets (Sentinel-2 +
AS7265x) from the same two sources already in data/raw/ — USGS splib07a
and the ECOSTRESS order — but labeled via pathfinder_mapping.py instead
of mineral_mapping.py. Combines both sources in one pass (unlike the main
mineral pipeline's two separate scripts) since this is a smaller, simpler
label set.

Usage: python scripts/build_pathfinder_dataset.py
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
from scripts.pathfinder_mapping import PATHFINDER_NAME_MAP  # noqa: E402

USGS_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw" / "usgs_splib" / "ASCIIdata_splib07a"
USGS_MINERALS = USGS_ROOT / "ChapterM_Minerals"
ECOSTRESS_ZIP = Path(__file__).resolve().parent.parent / "data" / "raw" / "ecostress" / "ecospeclib_all_minerals.zip"

OUT_SENTINEL2 = Path(__file__).resolve().parent.parent / "data" / "processed" / "pathfinder_sentinel2_dataset.csv"
OUT_AS7265X = Path(__file__).resolve().parent.parent / "data" / "processed" / "pathfinder_as7265x_dataset.csv"

_RECORD_RE = re.compile(r"Record=\d+:\s*(\S+)")
_NAME_RE = re.compile(r"^Name:\s*(.+)$")
_RANGE_RE = re.compile(r"^Wavelength Range:\s*(\S+)$")
_BAD_VALUE = -1.23e34

_INSTRUMENT_WAVELENGTH_FILES = {
    "ASD": USGS_ROOT / "splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt",
    "BECK": USGS_ROOT / "splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt",
}


def match_label(name: str) -> str | None:
    for key, label in PATHFINDER_NAME_MAP.items():
        if name.lower().startswith(key.lower()):
            return label
    return None


def load_values(path: Path) -> np.ndarray:
    values = []
    for line in path.read_text(errors="replace").splitlines()[1:]:
        line = line.strip()
        if line:
            values.append(float(line))
    return np.asarray(values)


def usgs_rows(centers: list[float]) -> list:
    wavelength_cache: dict[str, np.ndarray | None] = {}

    def wavelengths_for(instrument: str) -> np.ndarray | None:
        if instrument not in wavelength_cache:
            path = _INSTRUMENT_WAVELENGTH_FILES.get(instrument)
            wavelength_cache[instrument] = load_values(path) * 1000.0 if path and path.exists() else None
        return wavelength_cache[instrument]

    rows = []
    for path in sorted(USGS_MINERALS.glob("*.txt")):
        first_line = path.read_text(errors="replace").splitlines()[0]
        m = _RECORD_RE.search(first_line)
        if not m:
            continue
        label = match_label(m.group(1))
        if label is None:
            continue
        instrument = next((code for code in _INSTRUMENT_WAVELENGTH_FILES if code in path.stem), None)
        if instrument is None:
            continue
        xs = wavelengths_for(instrument)
        if xs is None:
            continue
        ys = load_values(path)
        if len(xs) != len(ys):
            continue
        valid = ys > _BAD_VALUE
        xs, ys = xs[valid], ys[valid]
        try:
            features = resample_to_bands(xs, ys, centers)
        except SpectrumRangeError:
            continue
        rows.append([label, m.group(1), f"usgs:{path.name}"] + list(features))
    return rows


def ecostress_rows(centers: list[float]) -> list:
    if not ECOSTRESS_ZIP.exists():
        return []
    rows = []
    with zipfile.ZipFile(ECOSTRESS_ZIP) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".spectrum.txt") or ".vswir." not in info.filename:
                continue
            lines = zf.read(info.filename).decode(errors="replace").splitlines()
            name, wavelength_range, data_start = None, None, None
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
            if data_start is None or wavelength_range != "VSWIR":
                continue
            label = match_label(name)
            if label is None:
                continue
            xs, ys = [], []
            for line in lines[data_start:]:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        xs.append(float(parts[0]) * 1000.0)
                        ys.append(float(parts[1]) / 100.0)
                    except ValueError:
                        continue
            try:
                features = resample_to_bands(np.asarray(xs), np.asarray(ys), centers)
            except SpectrumRangeError:
                continue
            rows.append([label, name, f"ecostress:{info.filename}"] + list(features))
    return rows


def main() -> None:
    for out_path, bands in [(OUT_SENTINEL2, SENTINEL2_BANDS), (OUT_AS7265X, AS7265X_BANDS)]:
        centers = band_centers(bands)
        rows = usgs_rows(centers) + ecostress_rows(centers)
        df = pd.DataFrame(rows, columns=["label", "mineral_name", "source"] + list(bands))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"{out_path.name}: {len(df)} rows")
        print(df["label"].value_counts().to_string())
        print()


if __name__ == "__main__":
    main()
