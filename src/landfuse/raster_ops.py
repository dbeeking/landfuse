"""Input checks, tile iteration, and feature extraction."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import reproject, transform_bounds

from .contract import LandFuseError, Scene, MODEL_FEATURES


def validate_scene(naip: rasterio.io.DatasetReader, terrain: rasterio.io.DatasetReader, mode: str,
                   *, max_pixels: int = 20_000_000) -> dict:
    if naip.crs is None or terrain.crs is None:
        raise LandFuseError("Both rasters require an explicit CRS")
    if naip.count != 4:
        raise LandFuseError("Four-band multispectral aerial imagery is required (R,G,B,NIR); RGB map tiles are insufficient")
    if any(dtype != "uint8" for dtype in naip.dtypes):
        raise LandFuseError("Imagery must contain four 8-bit raw NAIP bands; unsupported pixel type")
    if not naip.crs.is_projected or abs(naip.crs.linear_units_factor[1] - 1.0) > 0.0001:
        raise LandFuseError("Imagery must use a projected metric CRS")
    if naip.width < 1 or naip.height < 1 or naip.width * naip.height > max_pixels:
        raise LandFuseError(f"Image exceeds safety limit of {max_pixels:,} pixels; request a smaller AOI")
    if not naip.transform.is_rectilinear or naip.res[0] <= 0 or naip.res[1] <= 0:
        raise LandFuseError("Image pixels must form a north-up, rectilinear grid")
    if mode not in ("dem", "chm"):
        raise LandFuseError("Terrain mode must be dem or chm")
    if terrain.count != 1:
        raise LandFuseError("Terrain input must be a single-band GeoTIFF")
    bbox = transform_bounds(terrain.crs, naip.crs, *terrain.bounds, densify_pts=21)
    overlap_x = min(naip.bounds.right, bbox[2]) - max(naip.bounds.left, bbox[0])
    overlap_y = min(naip.bounds.top, bbox[3]) - max(naip.bounds.bottom, bbox[1])
    if overlap_x <= 0 or overlap_y <= 0:
        raise LandFuseError("Imagery and terrain rasters do not overlap geographically")
    image_area = (naip.bounds.right - naip.bounds.left) * (naip.bounds.top - naip.bounds.bottom)
    return {
        "crs": naip.crs.to_string(),
        "image_pixel_size": [float(abs(naip.res[0])), float(abs(naip.res[1]))],
        "terrain_pixel_size_native": [float(abs(terrain.res[0])), float(abs(terrain.res[1]))],
        "image_shape": [naip.height, naip.width],
        "terrain_mode": mode,
        "features": MODEL_FEATURES,
        "terrain_coverage_fraction": round(float(overlap_x * overlap_y / image_area), 6),
    }


def windows(ds, tile: int = 512):
    for r in range(0, ds.height, tile):
        for c in range(0, ds.width, tile):
            yield Window(c, r, min(tile, ds.width - c), min(tile, ds.height - r))


def _gradient(z: np.ndarray, step: float, axis: int) -> np.ndarray:
    """Central difference that uses one side next to nodata or a raster edge."""
    delta = np.diff(z, axis=axis) / step
    forward = np.full_like(z, np.nan)
    backward = np.full_like(z, np.nan)
    if axis == 0:
        forward[:-1, :] = delta
        backward[1:, :] = delta
    else:
        forward[:, :-1] = delta
        backward[:, 1:] = delta
    stacked = np.stack((forward, backward))
    count = np.isfinite(stacked).sum(axis=0)
    return np.nansum(stacked, axis=0) / np.maximum(count, 1)


def get_features(naip, terrain, mode: str, window: Window):
    """Return a [rows, columns, 8] feature array and validity mask for a tile.

    The terrain feature is slope in DEM mode and height above ground in CHM mode.
    """
    rgbn = naip.read([1, 2, 3, 4], window=window, masked=True).astype(np.float32)
    rgbn = rgbn.filled(np.nan)
    # A pixel that is zero in all four bands is outside the image footprint.
    valid = np.isfinite(rgbn).all(axis=0) & (rgbn > 0).any(axis=0)
    h, w = rgbn.shape[1:]
    # Read one extra pixel on each side so slope doesn't change at tile edges.
    pad = 1 if mode == "dem" else 0
    padded = Window(window.col_off - pad, window.row_off - pad, w + 2 * pad, h + 2 * pad)
    terrain_a = np.full((h + 2 * pad, w + 2 * pad), np.nan, dtype="float32")
    reproject(
        source=rasterio.band(terrain, 1), destination=terrain_a,
        src_transform=terrain.transform, src_crs=terrain.crs,
        dst_transform=naip.window_transform(padded), dst_crs=naip.crs,
        dst_nodata=np.nan, src_nodata=terrain.nodata,
        resampling=Resampling.bilinear if mode == "dem" else Resampling.nearest,
    )
    inner = terrain_a[pad:pad + h, pad:pad + w]
    valid &= np.isfinite(inner)
    if np.count_nonzero(valid) == 0:
        return np.zeros((h,w,8), dtype="float32"), valid
    if mode == "dem":
        # Slope comes from the DEM resampled to the image grid, so its detail
        # is limited by the DEM resolution.
        gy = _gradient(terrain_a, float(abs(naip.res[1])), 0)[pad:pad + h, pad:pad + w]
        gx = _gradient(terrain_a, float(abs(naip.res[0])), 1)[pad:pad + h, pad:pad + w]
        t = np.clip(np.sqrt(gx**2 + gy**2), 0.0, 1.0)
    else:
        t = np.clip(inner, 0.0, 60.0) / 60.0
    r,g,b,n = [np.clip(x / 255.0, 0.0, 1.0) for x in rgbn]
    eps = 1e-4
    ndvi = (n-r) / (n+r+eps)
    ndwi = (g-n) / (g+n+eps)
    bright = (r+g+b) / 3.0
    x = np.stack((r,g,b,n,ndvi,ndwi,bright,t),axis=-1).astype("float32")
    x[~np.isfinite(x)] = 0
    return x, valid


@contextmanager
def open_scene(scene: Scene, max_pixels=20_000_000):
    scene.check()
    with rasterio.open(scene.imagery) as naip, rasterio.open(scene.terrain) as terrain:
        metadata = validate_scene(naip,terrain,scene.terrain_mode,max_pixels=max_pixels)
        yield naip,terrain,metadata
