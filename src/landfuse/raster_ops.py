"""Input checks, tile iteration, and feature extraction."""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import reproject, transform_bounds

from .contract import LandFuseError, Scene, feature_names


def validate_scene(naip: rasterio.io.DatasetReader, terrain: rasterio.io.DatasetReader | None,
                   mode: str, *, max_pixels: int = 20_000_000) -> dict:
    """Check grid suitability; report truthfully whether terrain was supplied."""
    if naip.crs is None:
        raise LandFuseError("Imagery requires an explicit CRS")
    if mode not in ("imagery", "dem", "chm"):
        raise LandFuseError("Mode must be imagery, dem or chm")
    bands = naip.count
    if bands not in (3, 4) or (mode != "imagery" and bands != 4):
        raise LandFuseError("Four-band RGB-NIR required for terrain fusion; imagery-only accepts RGB or RGB-NIR")
    if any(dtype != "uint8" for dtype in naip.dtypes):
        raise LandFuseError("This release requires raw 8-bit aerial GeoTIFF bands, not rendered map tiles")
    if not naip.crs.is_projected or abs(naip.crs.linear_units_factor[1] - 1.0) > 0.0001:
        raise LandFuseError("Imagery must use a projected metric CRS")
    if naip.width < 1 or naip.height < 1 or naip.width * naip.height > max_pixels:
        raise LandFuseError(f"Image exceeds safety limit of {max_pixels:,} pixels; request a smaller AOI")
    if not naip.transform.is_rectilinear or naip.res[0] <= 0 or naip.res[1] <= 0:
        raise LandFuseError("Imagery must use a north-up, rectilinear grid")
    result = {
        "crs": naip.crs.to_string(),
        "image_pixel_size": [float(abs(naip.res[0])), float(abs(naip.res[1]))],
        "image_shape": [naip.height, naip.width],
        "imagery_bands": bands,
        "terrain_mode": mode,
        "features": feature_names(mode, bands),
        "terrain_pixel_size_native": None,
        "terrain_coverage_fraction": None,
        "height_estimation_available": mode == "chm",
    }
    if mode == "imagery":
        if terrain is not None:
            raise LandFuseError("Imagery-only mode must not include terrain")
        return result
    if terrain is None or terrain.crs is None or terrain.count != 1:
        raise LandFuseError("DEM/CHM must be a single-band raster with a CRS")
    bbox = transform_bounds(terrain.crs, naip.crs, *terrain.bounds, densify_pts=21)
    overlap_x = min(naip.bounds.right, bbox[2]) - max(naip.bounds.left, bbox[0])
    overlap_y = min(naip.bounds.top, bbox[3]) - max(naip.bounds.bottom, bbox[1])
    if overlap_x <= 0 or overlap_y <= 0:
        raise LandFuseError("Imagery and terrain rasters do not overlap geographically")
    image_area = (naip.bounds.right - naip.bounds.left) * (naip.bounds.top - naip.bounds.bottom)
    result["terrain_pixel_size_native"] = [float(abs(terrain.res[0])), float(abs(terrain.res[1]))]
    result["terrain_coverage_fraction"] = round(float(overlap_x * overlap_y / image_area), 6)
    return result


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
    """Return tile features and validity mask in mode/band-specific order."""
    bands = naip.count
    cube = naip.read(list(range(1, bands + 1)), window=window, masked=True).astype(np.float32)
    cube = cube.filled(np.nan)
    valid = np.isfinite(cube).all(axis=0) & (cube > 0).any(axis=0)
    h, w = cube.shape[1:]
    r, g, b = [np.clip(x / 255.0, 0.0, 1.0) for x in cube[:3]]
    channels = [r, g, b]
    if bands == 4:
        n = np.clip(cube[3] / 255.0, 0.0, 1.0)
        eps = 1e-4
        channels += [n, (n - r) / (n + r + eps), (g - n) / (g + n + eps)]
    channels.append((r + g + b) / 3.0)
    if mode != "imagery":
        # Use halo pixels to ensure DEM slope does not change at tile edges.
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
        if mode == "dem":
            gy = _gradient(terrain_a, float(abs(naip.res[1])), 0)[pad:pad + h, pad:pad + w]
            gx = _gradient(terrain_a, float(abs(naip.res[0])), 1)[pad:pad + h, pad:pad + w]
            channels.append(np.clip(np.sqrt(gx**2 + gy**2), 0.0, 1.0))
        else:
            channels.append(np.clip(inner, 0.0, 60.0) / 60.0)
    features = np.stack(channels, axis=-1).astype("float32")
    features[~np.isfinite(features)] = 0.0
    return features, valid


@contextmanager
def open_scene(scene: Scene, max_pixels=20_000_000):
    scene.check()
    with rasterio.open(scene.imagery) as naip:
        if scene.terrain is None:
            yield naip, None, validate_scene(naip, None, scene.terrain_mode, max_pixels=max_pixels)
        else:
            with rasterio.open(scene.terrain) as terrain:
                yield naip, terrain, validate_scene(naip, terrain, scene.terrain_mode, max_pixels=max_pixels)
