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
  python scripts/ingest_terrain.py --sites-json sites.json --copernicus   # many sites, one tile search
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


def find_copernicus_tiles(min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> list:
    """One catalog search for the whole area — slow from here (~90 s), so a
    batch of sites shares a single search instead of one per site."""
    import pystac_client

    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    return list(catalog.search(collections=["cop-dem-glo-30"], bbox=[min_lon, min_lat, max_lon, max_lat]).items())


def fetch_copernicus(min_lat: float, max_lat: float, min_lon: float, max_lon: float, out_path: Path,
                     tiles: list | None = None) -> None:
    """Writes the area to a GeoTIFF, stitching every 1x1 degree tile it
    touches. A site near a whole-degree line (e.g. RW-GFW-04 at lat -1.99,
    RW-BGR-07 at lon 29.00) spans two tiles; reading only one left half the
    block as gap-filled guesses."""
    import planetary_computer
    import rasterio
    from rasterio.merge import merge

    tiles = tiles if tiles is not None else find_copernicus_tiles(min_lat, max_lat, min_lon, max_lon)
    needed = [t for t in tiles if t.bbox[0] < max_lon and t.bbox[2] > min_lon
              and t.bbox[1] < max_lat and t.bbox[3] > min_lat]
    if not needed:
        raise RuntimeError("No Copernicus DEM tile covers this area.")

    # Partial downloads happen on slow links (seen in testing: a tile cut
    # off at 0.6 of 2.5 MB). GDAL retries interrupted HTTP reads itself
    # when told to.
    with rasterio.Env(GDAL_HTTP_MAX_RETRY=5, GDAL_HTTP_RETRY_DELAY=3):
        sources = [rasterio.open(planetary_computer.sign(t.assets["data"].href)) for t in needed]
        try:
            data, transform = merge(sources, bounds=(min_lon, min_lat, max_lon, max_lat))
            profile = {
                "driver": "GTiff", "height": data.shape[1], "width": data.shape[2], "count": 1,
                "dtype": data.dtype, "crs": sources[0].crs, "transform": transform,
                "nodata": sources[0].nodata, "compress": "deflate",
            }
        finally:
            for src in sources:
                src.close()
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data[0], 1)
    print(f"  {len(needed)} Copernicus tile(s) {[t.id.split('_DEM')[0][-14:] for t in needed]}, "
          f"{data.shape[2]}x{data.shape[1]} px")


def ingest_site(site: str, lat: float, lon: float, radius_km: float, grid_size: int,
                dtm: str | None = None, tiles: list | None = None) -> dict:
    site_dir = TERRAIN_DIR / site
    site_dir.mkdir(parents=True, exist_ok=True)
    min_lat, max_lat, min_lon, max_lon = bbox_around(lat, lon, radius_km)
    dtm_path = site_dir / "dtm.tif"

    print(f"[{site}] bbox lat {min_lat:.5f}..{max_lat:.5f}  lon {min_lon:.5f}..{max_lon:.5f}", flush=True)
    if dtm is None:
        # Fetch a margin beyond the box so edge interpolation has neighbours.
        m = 0.002
        fetch_copernicus(min_lat - m, max_lat + m, min_lon - m, max_lon + m, dtm_path, tiles)
        source = "copernicus-dem-glo-30"
    else:
        shutil.copyfile(dtm, dtm_path)
        source = f"dtm:{Path(dtm).name}"

    terrain = TerrainModel.from_geotiff(str(dtm_path), window_bounds_lonlat=(min_lon, min_lat, max_lon, max_lat))
    grid = terrain.to_dem_grid(min_lat, max_lat, min_lon, max_lon, grid_size)
    grid["source"] = source

    (site_dir / "dem_grid.json").write_text(json.dumps(grid))
    (site_dir / "meta.json").write_text(json.dumps({
        "site": site, "lat": lat, "lon": lon, "radius_km": radius_km, "source": source,
        "dtm": "dtm.tif", "crs": str(terrain.crs), "resolution_m": terrain.resolution_m,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    print(f"  elevation {grid['minElevation']}..{grid['maxElevation']} m, "
          f"source resolution {terrain.resolution_m:.1f} m, filled {grid['filledFraction']:.1%}", flush=True)
    return grid


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--site")
    p.add_argument("--lat", type=float)
    p.add_argument("--lon", type=float)
    p.add_argument("--sites-json", help='batch: file with [{"id":..,"lat":..,"lon":..}, ...] (Copernicus only)')
    p.add_argument("--radius-km", type=float, default=2.0, help="matches the frontend's default 2 km")
    p.add_argument("--grid-size", type=int, default=64)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dtm")
    src.add_argument("--copernicus", action="store_true")
    a = p.parse_args()

    if a.sites_json:
        if a.dtm:
            sys.exit("--sites-json is for --copernicus; a drone DTM belongs to one site (use --site).")
        sites = json.loads(Path(a.sites_json).read_text())
        boxes = [bbox_around(s["lat"], s["lon"], a.radius_km + 0.5) for s in sites]
        print(f"Searching Copernicus tiles for {len(sites)} sites (one search)...", flush=True)
        tiles = find_copernicus_tiles(min(b[0] for b in boxes), max(b[1] for b in boxes),
                                      min(b[2] for b in boxes), max(b[3] for b in boxes))
        print(f"  {len(tiles)} tiles found", flush=True)
        failed = []
        for s in sites:
            try:
                ingest_site(s["id"], s["lat"], s["lon"], a.radius_km, a.grid_size, tiles=tiles)
            except Exception as e:  # one bad site shouldn't lose the rest of the batch
                print(f"  FAILED {s['id']}: {e}", flush=True)
                failed.append(s["id"])
        print(f"\nDone: {len(sites) - len(failed)}/{len(sites)} sites" + (f", failed: {failed}" if failed else ""))
        sys.exit(1 if failed else 0)

    if a.site is None or a.lat is None or a.lon is None:
        sys.exit("Give --site, --lat and --lon (or --sites-json for a batch).")
    ingest_site(a.site, a.lat, a.lon, a.radius_km, a.grid_size, dtm=a.dtm)


if __name__ == "__main__":
    main()
