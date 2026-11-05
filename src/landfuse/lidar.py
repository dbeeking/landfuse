"""Build a canopy height model from LiDAR using the PDAL command-line tool.

Input point clouds need a valid CRS and classified ground returns (ASPRS class 2).
"""
from __future__ import annotations

import json
from pathlib import Path
from shutil import which
import subprocess
import tempfile
from urllib.parse import urlsplit

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.fill import fillnodata
from rasterio.warp import reproject

from .contract import LandFuseError, sha256, write_json


def _is_usgs_ept(url: str) -> bool:
    u = urlsplit(url)
    if not u.path.endswith("/ept.json") or ".." in u.path:
        return False
    if u.hostname == "s3-us-west-2.amazonaws.com":
        return u.path.startswith("/usgs-lidar-public/")
    return u.hostname in {"usgs-lidar-public.s3.us-west-2.amazonaws.com",
                          "usgs-lidar-public.s3.amazonaws.com"}


def pdal_canopy_height(lidar: str, imagery: Path, output: Path,
                       *, resolution_m: float = 1.0, timeout_seconds: int = 1800,
                       ground_fill_m: float = 25.0) -> dict:
    if which("pdal") is None:
        raise LandFuseError("PDAL executable is required for LAS/LAZ/EPT: install PDAL first")
    if output.exists(): raise LandFuseError("Refusing to overwrite an existing CHM")
    remote=lidar.startswith("https://")
    if remote:
        # Remote sources are limited to public USGS 3DEP EPT hosts.
        if not _is_usgs_ept(lidar):
            raise LandFuseError("Remote EPT must use approved USGS public S3 endpoint")
    elif not Path(lidar).is_file() or Path(lidar).suffix.lower() not in (".las",".laz"):
        raise LandFuseError("Provide a local LAS/LAZ or approved USGS public EPT URL")
    if not 0.5 <= resolution_m <= 5.0:
        raise LandFuseError("LiDAR rasterization resolution must be 0.5 to 5 m")
    with rasterio.open(imagery) as image:
        if image.crs is None or not image.crs.is_projected:
            raise LandFuseError("NAIP raster must have a projected CRS before LiDAR preparation")
        x0,y0,x1,y1 = image.bounds
        bbox=f"([{x0},{x1}],[{y0},{y1}])"
        with tempfile.TemporaryDirectory(prefix="landfuse-lidar-") as tmp:
            paths={"ground":Path(tmp)/"dtm.tif","surface":Path(tmp)/"dsm.tif"}
            for kind,path in paths.items():
                reader = ({"type":"readers.ept","filename":lidar,
                           "bounds":bbox+"/"+image.crs.to_string()} if remote else
                          {"type":"readers.las","filename":str(Path(lidar).resolve())})
                filters=[{"type":"filters.reprojection","out_srs":image.crs.to_string()},
                         {"type":"filters.crop","bounds":bbox}]
                expression = "Classification == 2" if kind=="ground" else "Classification != 7 && Classification != 18"
                pipeline={"pipeline":[reader,*filters,
                    {"type":"filters.expression","expression":expression},
                    {"type":"writers.gdal","filename":str(path),
                     "resolution":resolution_m,"gdaldriver":"GTiff",
                     "output_type":"min" if kind=="ground" else "max",
                     "bounds":bbox,"data_type":"float","nodata":-9999}]}
                try:
                    p=subprocess.run(["pdal","pipeline","--stdin"],input=json.dumps(pipeline),
                                     text=True,capture_output=True,timeout=timeout_seconds,
                                     check=False)
                except subprocess.TimeoutExpired as exc:
                    raise LandFuseError("LiDAR preparation timed out") from exc
                if p.returncode or not path.is_file():
                    raise LandFuseError(f"PDAL {kind} surface creation failed; verify CRS and ground classification")
            with rasterio.open(paths["ground"]) as ground, rasterio.open(paths["surface"]) as surface:
                if ground.crs is None or surface.crs is None:
                    raise LandFuseError("PDAL outputs lacked CRS")
                height=surface.height; width=surface.width
                if height*width > 10_000_000:
                    raise LandFuseError("LiDAR CHM exceeds 10 million pixels")
                dsm=surface.read(1,masked=True).filled(np.nan).astype("float32")
                dtm=np.full(dsm.shape,np.nan,dtype="float32")
                reproject(rasterio.band(ground,1),dtm,
                          src_transform=ground.transform,src_crs=ground.crs,src_nodata=ground.nodata,
                          dst_transform=surface.transform,dst_crs=surface.crs,dst_nodata=np.nan,
                          resampling=Resampling.nearest)
                # Ground returns are sparse under canopy, so interpolate small gaps.
                has_ground=np.isfinite(dtm)
                if has_ground.any():
                    dtm=fillnodata(dtm,mask=has_ground.astype("uint8"),
                                   max_search_distance=ground_fill_m/abs(surface.res[0]))
                filled_ground=int(np.count_nonzero(np.isfinite(dtm)&~has_ground))
                valid=np.isfinite(dsm)&np.isfinite(dtm)
                if np.count_nonzero(valid) == 0:
                    raise LandFuseError("No overlapping valid ground and surface points; cannot calculate CHM")
                chm=np.where(valid,np.maximum(dsm-dtm,0.0),-9999.0).astype("float32")
                profile=surface.profile.copy()
                profile.update(driver="GTiff",count=1,dtype="float32",nodata=-9999,
                               compress="deflate",predictor=3,tiled=True,
                               blockxsize=256,blockysize=256)
                output.parent.mkdir(parents=True,exist_ok=True)
                with rasterio.open(output,"w",**profile) as out:
                    out.write(chm,1)
                    out.update_tags(LANDFUSE="height_above_ground_m",SOURCE="classified_lidar")
    info={"output":str(output),"sha256":sha256(output),
          "source_kind":"USGS EPT" if remote else "LAS/LAZ",
          "source":lidar if remote else Path(lidar).name,
          "valid_pixel_count":int(np.count_nonzero(valid)),"rasterization_m":resolution_m,
          "filled_ground_pixels":filled_ground,"ground_fill_m":ground_fill_m,
          "method":"DSM maximum Z minus DTM minimum ground-class Z with interpolated ground gaps; unvalidated heights"}
    write_json(str(output)+".json",info)
    return info
