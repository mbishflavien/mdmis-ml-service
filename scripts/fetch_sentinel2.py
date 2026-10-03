"""Pulls real Sentinel-2 L2A band reflectance at specific GPS points via
Microsoft's Planetary Computer STAC API — free, no account/API key needed
for anonymous reads. This is the $0-cost sensor already in
MDMIS_IoT_Budget.docx (Table "Data & Platform"), so unlike the other
sensors it needs no hardware and is usable immediately.

Pulls two sets of points:
  1. Cuprite, Nevada — a real mining district with a published USGS
     mineral map, used as an external sanity check (not blind trust —
     see README) since we don't have our own lab-confirmed ground truth
     yet.
  2. The 10 seeded Rwandan sites from MDMIS_BACKEND-'s app/seed.py —
     real coordinates, but `primary_mineral` there is demo/placeholder
     data, not field-verified. Treat the output for these as "what the
     real pipeline produces," not as training labels.

Usage: python scripts/fetch_sentinel2.py
Output: data/processed/sentinel2_points.csv (site, lat, lon, band values,
band ratios, cloud_cover, scene_date)
"""
import csv
import sys
from pathlib import Path

import planetary_computer
import pystac_client
import rasterio
from rasterio.warp import transform as warp_transform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.band_ratios import compute_band_ratios  # noqa: E402
from app.sensor_bands import SENTINEL2_BANDS  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "sentinel2_points.csv"

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# name, lat, lon, reference_label (None where we have no real ground truth)
POINTS = [
    ("Cuprite-NV-reference", 37.5597, -117.1561, "alunite_kaolinite_zone"),  # published USGS alteration zone
    ("RW-RTG-01", -1.7783, 30.0611, "cassiterite"),
    ("RW-GTB-02", -1.8642, 29.5231, "coltan"),
    ("RW-NYK-03", -1.7419, 30.0089, "wolframite"),
    ("RW-GFW-04", -1.9928, 29.4102, "wolframite"),
    ("RW-RWK-05", -2.1481, 30.5892, "gold"),
    ("RW-NMB-06", -1.6892, 29.7743, "coltan"),
    ("RW-BGR-07", -2.6889, 29.0031, "lithium"),
    ("RW-MSH-08", -1.9231, 30.3402, "cassiterite"),
    ("RW-KRG-09", -2.0031, 29.3781, "beryl"),
    ("RW-RTS-10", -1.9312, 29.3312, "gold"),
]


def find_least_cloudy_scene(catalog, lat: float, lon: float):
    # The STAC query/sortby extensions are slow (or hang) against this API
    # for some collections — pull a page of recent items and pick the
    # least-cloudy one client-side instead, which is fast and reliable.
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=[lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01],
        limit=20,
    )
    items = list(search.items())
    if not items:
        return None
    return min(items, key=lambda it: it.properties.get("eo:cloud_cover", 100))


def sample_bands(item, lat: float, lon: float) -> dict[str, float]:
    values = {}
    for band in SENTINEL2_BANDS:
        href = planetary_computer.sign(item.assets[band].href)
        with rasterio.open(href) as src:
            xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
            row, col = src.index(xs[0], ys[0])
            window = ((row, row + 1), (col, col + 1))
            value = src.read(1, window=window)[0, 0]
            # L2A surface reflectance is scaled by 10000 per ESA's product spec
            values[band] = float(value) / 10000.0
    return values


def main() -> None:
    catalog = pystac_client.Client.open(STAC_URL)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["site", "lat", "lon", "reference_label", "scene_date", "cloud_cover"]
            + list(SENTINEL2_BANDS)
            + ["iron_oxide_ratio", "carbonate_ratio", "clay_ratio", "flags"]
        )

        for name, lat, lon, label in POINTS:
            print(f"[fetching] {name} ({lat}, {lon})")
            item = find_least_cloudy_scene(catalog, lat, lon)
            if item is None:
                print(f"  no clear scene found for {name}, skipping")
                continue
            bands = sample_bands(item, lat, lon)
            ratios = compute_band_ratios(bands)
            writer.writerow(
                [name, lat, lon, label, item.datetime.date().isoformat(), item.properties.get("eo:cloud_cover")]
                + [bands[b] for b in SENTINEL2_BANDS]
                + [ratios.iron_oxide, ratios.carbonate, ratios.clay, ";".join(ratios.flags)]
            )
            print(f"  ratios: Fe2O3={ratios.iron_oxide:.3f} CO3={ratios.carbonate:.3f} "
                  f"clay={ratios.clay:.3f} flags={ratios.flags}")

    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
