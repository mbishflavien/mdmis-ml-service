"""Terrain model (DTM GeoTIFF) -> the surface of the 3D block, and the
frame every subsurface reading is placed into.

Input is any elevation GeoTIFF: a WebODM/OpenDroneMap DTM from the drone
(usually in a UTM projection, cm-to-dm resolution), or a public DEM such
as Copernicus GLO-30 (30 m) as a stand-in until flights happen. Both are
just "a grid of heights in some map projection", so one loader serves
both — the drone DTM simply replaces the coarse one for the same site.

Output matches the frontend's existing DemGrid type
(frontend/lib/dem-fetcher.ts) exactly: square grid, endpoints included,
elevations[row][col] with row 0 = minLat (south) and col 0 = minLon
(west), so lib/use-dem-terrain.ts can render it without changes.

Placement: a reading at (lat, lon) with depth_m below the surface sits at
  elevation_m = surface_elevation(lat, lon) - depth_m

Caveat to carry into the UI: drone-derived DTMs without surveyed ground
control points are accurate *relative to themselves* (shapes, slopes,
depths) but their absolute height can be off by metres to tens of metres
— DJI Mini GPS/barometric altitude isn't survey-grade.
"""
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from scipy import ndimage


@dataclass
class TerrainModel:
    elevation: np.ndarray  # metres, NaN where the source had no data
    transform: rasterio.Affine
    crs: object
    source: str
    resolution_m: float

    @classmethod
    def from_geotiff(cls, path: str, window_bounds_lonlat: tuple[float, float, float, float] | None = None) -> "TerrainModel":
        """window_bounds_lonlat=(min_lon, min_lat, max_lon, max_lat) reads only
        that area — DTMs and public DEM tiles can be far bigger than a site."""
        with rasterio.open(path) as src:
            window = None
            if window_bounds_lonlat is not None:
                min_lon, min_lat, max_lon, max_lat = window_bounds_lonlat
                xs, ys = warp_transform("EPSG:4326", src.crs, [min_lon, max_lon, min_lon, max_lon],
                                        [min_lat, min_lat, max_lat, max_lat])
                window = rasterio.windows.from_bounds(min(xs), min(ys), max(xs), max(ys), src.transform)
                # Pad 2 px: interpolating a point at the very edge of the box
                # needs the pixel centres just outside it, otherwise edge
                # points read as no-data and get reported as "filled".
                window = rasterio.windows.Window(window.col_off - 2, window.row_off - 2,
                                                 window.width + 4, window.height + 4)
                window = window.round_offsets().round_lengths()
                window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            data = src.read(1, window=window, masked=True).astype(float).filled(np.nan)
            transform = src.window_transform(window) if window is not None else src.transform
            if src.crs.is_geographic:
                res_m = abs(src.res[0]) * 111_320  # degrees -> approx metres at the equator
            else:
                res_m = abs(src.res[0])
            return cls(elevation=data, transform=transform, crs=src.crs, source=str(path), resolution_m=float(res_m))

    def elevation_at(self, lats: list[float], lons: list[float]) -> np.ndarray:
        """Bilinear-interpolated surface elevation at each lat/lon; NaN
        outside the model or where it has no data."""
        xs, ys = warp_transform("EPSG:4326", self.crs, list(lons), list(lats))
        cols, rows = ~self.transform * (np.asarray(xs), np.asarray(ys))
        # Affine maps to pixel corners; shift so integer coords are pixel centres.
        cols, rows = np.asarray(cols) - 0.5, np.asarray(rows) - 0.5
        h, w = self.elevation.shape
        out = np.full(len(cols), np.nan)
        inside = (cols >= 0) & (rows >= 0) & (cols <= w - 1) & (rows <= h - 1)
        c, r = cols[inside], rows[inside]
        c0, r0 = np.floor(c).astype(int), np.floor(r).astype(int)
        c1, r1 = np.minimum(c0 + 1, w - 1), np.minimum(r0 + 1, h - 1)
        fc, fr = c - c0, r - r0
        e = self.elevation
        out[inside] = (e[r0, c0] * (1 - fc) * (1 - fr) + e[r0, c1] * fc * (1 - fr)
                       + e[r1, c0] * (1 - fc) * fr + e[r1, c1] * fc * fr)
        return out

    def to_dem_grid(self, min_lat: float, max_lat: float, min_lon: float, max_lon: float, grid_size: int = 64) -> dict:
        """Frontend DemGrid. Holes (no-data, or outside the model) are filled
        from the nearest valid cell and reported in `filledFraction`, so the
        mesh renders but nobody mistakes a filled area for a measurement."""
        lats = np.linspace(min_lat, max_lat, grid_size)
        lons = np.linspace(min_lon, max_lon, grid_size)
        lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")  # row = lat, col = lon
        grid = self.elevation_at(lat_g.ravel().tolist(), lon_g.ravel().tolist()).reshape(grid_size, grid_size)

        missing = np.isnan(grid)
        if missing.all():
            raise ValueError("Requested area doesn't overlap the terrain model.")
        filled_fraction = float(missing.mean())
        if missing.any():
            _, (ri, ci) = ndimage.distance_transform_edt(missing, return_indices=True)
            grid = grid[ri, ci]

        return {
            "gridSize": grid_size,
            "minLat": min_lat,
            "maxLat": max_lat,
            "minLon": min_lon,
            "maxLon": max_lon,
            "elevations": np.round(grid, 2).tolist(),
            "minElevation": float(np.round(grid.min(), 2)),
            "maxElevation": float(np.round(grid.max(), 2)),
            # Not in the frontend type; extra keys are ignored there.
            "filledFraction": round(filled_fraction, 4),
            "sourceResolutionM": round(self.resolution_m, 3),
            "source": self.source,
        }

    def place(self, readings: list[dict]) -> list[dict]:
        """readings: [{lat, lon, depth_m}] -> each with surface_elevation_m and
        elevation_m (= surface - depth), or an `error` if outside the model."""
        if not readings:
            return []
        surface = self.elevation_at([r["lat"] for r in readings], [r["lon"] for r in readings])
        out = []
        for r, s in zip(readings, surface):
            if np.isnan(s):
                out.append({**r, "error": "outside terrain model or no data at this point"})
            else:
                out.append({**r, "surface_elevation_m": round(float(s), 2),
                            "elevation_m": round(float(s) - float(r.get("depth_m") or 0.0), 2)})
        return out


def bbox_around(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    """(min_lat, max_lat, min_lon, max_lon) — identical formula to the
    frontend's getSiteBoundingBox, so both compute the same box."""
    dlat = radius_km / 111
    dlon = radius_km / (111 * np.cos(np.radians(lat)))
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon
