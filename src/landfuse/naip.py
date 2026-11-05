"""Download four-band NAIP clips from the Arizona ArcGIS ImageServer."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import math
import json

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.windows import Window, bounds as window_bounds
from rasterio.warp import reproject
from rasterio.enums import Resampling

from . import __version__
from .contract import LandFuseError, sha256, write_json

PINNED_IMAGE_SERVICE = "https://azgeo.az.gov/arcgis/rest/services/imagery/NAIP2023/ImageServer/exportImage"


def fetch_naip(output: Path, bbox: tuple[float,float,float,float], epsg: int,
               *, resolution_m: float = 0.30, max_pixels: int = 5_000_000,
               tile_pixels: int = 1024, timeout: int = 60) -> dict:
    """Download a clip as a local GeoTIFF and write a JSON sidecar.

    The clip is requested in tiles and mosaicked. Any response that is not a
    four-band georeferenced TIFF raises LandFuseError.
    """
    if output.exists(): raise LandFuseError("Refusing to overwrite an existing NAIP clip")
    crs=CRS.from_epsg(epsg)
    if not crs.is_projected or abs(crs.linear_units_factor[1] - 1.0) > 0.0001:
        raise LandFuseError("Provide a projected CRS with metre units (e.g. EPSG:26912)")
    xmin,ymin,xmax,ymax=bbox
    if not all(map(math.isfinite,bbox)) or not (xmin<xmax and ymin<ymax):
        raise LandFuseError("Invalid bounding box")
    if not (0.15 <= resolution_m <= 2.0):
        raise LandFuseError("Resolution must be between 0.15 and 2.0 m; smaller pixels do not improve native detail")
    w=int(math.ceil((xmax-xmin)/resolution_m))
    h=int(math.ceil((ymax-ymin)/resolution_m))
    if w*h > max_pixels or w*h < 1:
        raise LandFuseError(f"Clip is {w*h:,} pixels, over configured {max_pixels:,} limit")
    if tile_pixels < 64 or tile_pixels > 2048:
        raise LandFuseError("tile_pixels must be 64 to 2048")
    affine=from_origin(xmin,ymax,resolution_m,resolution_m)
    profile={"driver":"GTiff","count":4,"dtype":"uint8","crs":crs,
             "transform":affine,"width":w,"height":h,"compress":"deflate",
             "tiled":True,"blockxsize":256,"blockysize":256,"photometric":"MINISBLACK"}
    output.parent.mkdir(parents=True,exist_ok=True)
    nrequest=0
    try:
        with rasterio.open(output,"w",**profile) as dst:
            for row in range(0,h,tile_pixels):
                for col in range(0,w,tile_pixels):
                    win=Window(col,row,min(tile_pixels,w-col),min(tile_pixels,h-row))
                    b=window_bounds(win,affine)
                    args={"bbox":",".join(f"{v:.6f}" for v in b),"bboxSR":str(epsg),
                          "imageSR":str(epsg),"size":f"{int(win.width)},{int(win.height)}",
                          "format":"tiff","pixelType":"U8","bandIds":"0,1,2,3",
                          "interpolation":"RSP_NearestNeighbor","f":"image"}
                    req=Request(PINNED_IMAGE_SERVICE,
                                data=urlencode(args).encode("ascii"),
                                headers={"User-Agent":f"LandFuse/{__version__}","Accept":"image/tiff"},
                                method="POST")
                    try:
                        with urlopen(req,timeout=timeout) as resp:
                            raw=resp.read(24*1024*1024+1)
                    except Exception as exc:
                        raise LandFuseError(f"NAIP service fetch failed ({exc}). Try a smaller area or retry later") from exc
                    if len(raw)>24*1024*1024 or not raw[:4] in (b"II*\x00", b"MM\x00*"):
                        raise LandFuseError("NAIP service did not return a bounded GeoTIFF; refusing image/JSON fallback")
                    with MemoryFile(raw) as mem:
                        with mem.open() as src:
                            if src.count != 4 or src.crs is None or src.width<=0 or src.height<=0:
                                raise LandFuseError("NAIP response must contain 4 georeferenced bands")
                            if src.dtypes[0] != "uint8":
                                raise LandFuseError("Unexpected NAIP pixel type")
                            arr=np.empty((4,int(win.height),int(win.width)),dtype="uint8")
                            for band in range(1,5):
                                reproject(source=rasterio.band(src,band),
                                          destination=arr[band-1],
                                          src_transform=src.transform,src_crs=src.crs,
                                          dst_transform=dst.window_transform(win),dst_crs=dst.crs,
                                          resampling=Resampling.nearest,
                                          src_nodata=src.nodata)
                            dst.write(arr,window=win)
                    nrequest+=1
            dst.update_tags(PROVIDER="Arizona NAIP 2023 ImageServer",
                            SERVICE_URL=PINNED_IMAGE_SERVICE,
                            BAND_ORDER="R,G,B,NIR (requested; verify mosaic provenance)",
                            REPORTED_NATIVE_PIXEL_METRES="0.30",
                            IMAGE_YEAR="2023")
    except Exception:
        output.unlink(missing_ok=True)
        raise
    info={"source":"Arizona NAIP2023 ImageServer", "service_url":PINNED_IMAGE_SERVICE,
          "output":str(output),"sha256":sha256(output),"crs":crs.to_string(),
          "clip_bbox":list(bbox),"pixel_size_m":resolution_m,"size_px":[w,h],
          "requests":nrequest,"bands":"R,G,B,NIR (requested)",
          "quality":"RAW_BANDS_REQUESTED; PIXEL PROVENANCE NOT INDEPENDENTLY CHECKED"}
    write_json(str(output)+".json",info)
    return info
