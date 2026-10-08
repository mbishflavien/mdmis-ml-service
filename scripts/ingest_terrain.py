"""Ingest a terrain model for a site -> data/terrain/<site>/ (dem_grid.json
for the 3D block surface, plus the GeoTIFF used to place readings).

Two sources, same output:
  --dtm path.tif     a WebODM/OpenDroneMap DTM from a drone flight (the
                     real target), or any elevation GeoTIFF
  --copernicus       free Copernicus DEM GLO-30 (30 m) from Planetary
                     Computer — a coarse but real stand-in until the drone
                     flies; re-ingest with --dtm later and it replaces this

Usage:
  python scripts/ingest_terrain.py --site RW-RTG-01 --lat -1.7783 --lon 30.0611 --copernicus
  python scripts/ingest_terrain.py --site RW-RTG-01 --lat -1.7783 --lon 30.0611 --dtm odm_dtm.tif
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.translation.terrain import TerrainModel, bbox_around  # noqa: E402

TERRAIN_DIR = Path(__file__).resolve().parent.parent / "data" / "terrain"


def fetch_copernicus(min_lat: float, max_lat: float, min_lon: float, max_lon: float, out_path: Path) -> None:
    import planetary_computer
    import pystac_client
    import rasterio
    from rasterio.windows import from_bounds

    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    items = list(catalog.search(collections=["cop-dem-glo-30"], bbox=[min_lon, min_lat, max_lon, max_lat]).items())
    if not items:
        sys.exit("No Copernicus DEM tile covers this area.")
    if len(items) > 1:
        print(f"  note: area spans {len(items)} DEM tiles; using the one containing the site centre")
    clat, clon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2
    item = next((it for it in items if it.bbox[0] <= clon <= it.bbox[2] and it.bbox[1] <= clat <= it.bbox[3]), items[0])

    # Partial downloads happen on slow links (seen in testing: a tile cut
    # off at 0.6 of 2.5 MB). GDAL retries interrupted HTTP reads itself
    # when told to.
    with rasterio.Env(GDAL_HTTP_MAX_RETRY=5, GDAL_HTTP_RETRY_DELAY=3), \
            rasterio.open(planetary_computer.sign(item.assets["data"].href)) as src:
        window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform).round_offsets().round_lengths()
        data = src.read(1, window=window)
        profile = src.profile | {
            "driver": "GTiff", "height": data.shape[0], "width": data.shape[1],
            "transform": src.window_transform(window), "compress": "deflate",
        }
        profile.pop("blockxsize", None), profile.pop("blockysize", None), profile.pop("tiled", None)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)
    print(f"  Copernicus tile {item.id}, {data.shape[1]}x{data.shape[0]} px")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--site", required=True)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--radius-km", type=float, default=2.0, help="matches the frontend's default 2 km")
    p.add_argument("--grid-size", type=int, default=64)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dtm")
    src.add_argument("--copernicus", action="store_true")
    a = p.parse_args()

    site_dir = TERRAIN_DIR / a.site
    site_dir.mkdir(parents=True, exist_ok=True)
    min_lat, max_lat, min_lon, max_lon = bbox_around(a.lat, a.lon, a.radius_km)
    dtm_path = site_dir / "dtm.tif"

    print(f"[{a.site}] bbox lat {min_lat:.5f}..{max_lat:.5f}  lon {min_lon:.5f}..{max_lon:.5f}")
    if a.copernicus:
        # Fetch a margin beyond the box so edge interpolation has neighbours.
        m = 0.002
        fetch_copernicus(min_lat - m, max_lat + m, min_lon - m, max_lon + m, dtm_path)
        source = "copernicus-dem-glo-30"
    else:
        shutil.copyfile(a.dtm, dtm_path)
        source = f"dtm:{Path(a.dtm).name}"

    terrain = TerrainModel.from_geotiff(str(dtm_path), window_bounds_lonlat=(min_lon, min_lat, max_lon, max_lat))
    grid = terrain.to_dem_grid(min_lat, max_lat, min_lon, max_lon, a.grid_size)
    grid["source"] = source

    (site_dir / "dem_grid.json").write_text(json.dumps(grid))
    (site_dir / "meta.json").write_text(json.dumps({
        "site": a.site, "lat": a.lat, "lon": a.lon, "radius_km": a.radius_km, "source": source,
        "dtm": "dtm.tif", "crs": str(terrain.crs), "resolution_m": terrain.resolution_m,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    print(f"  elevation {grid['minElevation']}..{grid['maxElevation']} m, "
          f"source resolution {terrain.resolution_m:.1f} m, filled {grid['filledFraction']:.1%}")
    print(f"  wrote {site_dir / 'dem_grid.json'}")


if __name__ == "__main__":
    main()
