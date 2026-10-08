"""Terrain ingestion: a synthetic tilted-plane DTM in UTM 35S (the kind of
projected GeoTIFF WebODM produces) with heights known analytically, so
every sampled and placed value can be checked exactly."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.translation.terrain import TerrainModel, bbox_around  # noqa: E402

SITE_LAT, SITE_LON = -1.7783, 30.0611  # RW-RTG-01
CRS = "EPSG:32735"  # UTM zone 35S
BASE, SLOPE_E, SLOPE_N = 1500.0, 0.05, 0.02  # m, m per m east, m per m north
PIXEL = 5.0


def _make_dtm(path: Path, nodata_hole: bool = False):
    (cx,), (cy,) = warp_transform("EPSG:4326", CRS, [SITE_LON], [SITE_LAT])
    size = 1200  # 6 km at 5 m/px, covers a 2 km-radius site box
    x0, y0 = cx - size * PIXEL / 2, cy + size * PIXEL / 2
    cols, rows = np.meshgrid(np.arange(size), np.arange(size))
    xs, ys = x0 + (cols + 0.5) * PIXEL, y0 - (rows + 0.5) * PIXEL  # pixel centres
    z = (BASE + SLOPE_E * (xs - cx) + SLOPE_N * (ys - cy)).astype("float32")
    if nodata_hole:
        z[590:610, 590:610] = -9999
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=1, dtype="float32",
                       crs=CRS, transform=from_origin(x0, y0, PIXEL, PIXEL), nodata=-9999) as dst:
        dst.write(z, 1)
    return cx, cy


def _expected(lats, lons, cx, cy):
    xs, ys = warp_transform("EPSG:4326", CRS, list(lons), list(lats))
    return BASE + SLOPE_E * (np.asarray(xs) - cx) + SLOPE_N * (np.asarray(ys) - cy)


def test_elevation_matches_plane_and_grid_orientation():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "dtm.tif"
        cx, cy = _make_dtm(path)
        t = TerrainModel.from_geotiff(str(path))

        lats = [SITE_LAT, SITE_LAT + 0.005, SITE_LAT - 0.007]
        lons = [SITE_LON, SITE_LON - 0.004, SITE_LON + 0.006]
        assert np.allclose(t.elevation_at(lats, lons), _expected(lats, lons, cx, cy), atol=1e-3)

        min_lat, max_lat, min_lon, max_lon = bbox_around(SITE_LAT, SITE_LON, 2.0)
        g = t.to_dem_grid(min_lat, max_lat, min_lon, max_lon, grid_size=16)
        e = np.array(g["elevations"])
        assert e.shape == (16, 16)
        assert g["filledFraction"] == 0
        # Frontend convention: row 0 = south (minLat), col 0 = west (minLon).
        # Height rises to the north and east, so:
        assert e[0].mean() < e[-1].mean()
        assert e[:, 0].mean() < e[:, -1].mean()
        corner = _expected([min_lat], [min_lon], cx, cy)[0]
        assert abs(e[0][0] - corner) < 0.01


def test_place_subtracts_depth_and_rejects_outside_points():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "dtm.tif"
        cx, cy = _make_dtm(path)
        t = TerrainModel.from_geotiff(str(path))
        placed = t.place([
            {"lat": SITE_LAT, "lon": SITE_LON, "depth_m": 12.0},
            {"lat": SITE_LAT + 0.5, "lon": SITE_LON, "depth_m": 3.0},  # ~55 km away, outside
        ])
        surface = _expected([SITE_LAT], [SITE_LON], cx, cy)[0]
        assert abs(placed[0]["surface_elevation_m"] - surface) < 0.01
        assert abs(placed[0]["elevation_m"] - (surface - 12.0)) < 0.01
        assert "error" in placed[1]


def test_nodata_hole_is_filled_and_reported():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "dtm.tif"
        _make_dtm(path, nodata_hole=True)
        t = TerrainModel.from_geotiff(str(path))
        lat0, lat1, lon0, lon1 = bbox_around(SITE_LAT, SITE_LON, 0.2)  # small box around the hole
        g = t.to_dem_grid(lat0, lat1, lon0, lon1, grid_size=32)
        assert g["filledFraction"] > 0
        assert all(np.isfinite(v) for row in g["elevations"] for v in row)


def test_invalid_site_id_is_rejected():
    from fastapi.testclient import TestClient

    from app.main import app

    c = TestClient(app)
    r = c.get("/terrain/bad.id", headers={"X-ML-Service-Key": "change-me-to-a-long-random-string"})
    assert r.status_code == 400


if __name__ == "__main__":
    test_elevation_matches_plane_and_grid_orientation()
    test_place_subtracts_depth_and_rejects_outside_points()
    test_nodata_hole_is_filled_and_reported()
    test_invalid_site_id_is_rejected()
    print("OK")
