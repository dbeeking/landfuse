"""End-to-end CHM test that runs only when PDAL and laspy are installed."""
import shutil

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from landfuse.lidar import pdal_canopy_height

laspy = pytest.importorskip("laspy")
pyproj = pytest.importorskip("pyproj")
pytestmark = pytest.mark.skipif(shutil.which("pdal") is None, reason="PDAL is not installed")

ORIGIN_X = 365186.0
ORIGIN_Y = 3823072.0


def write_points(path):
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 60, 7200)
    y = rng.uniform(0, 60, 7200)
    in_trees = (x > 20) & (x < 35) & (y > 20) & (y < 35)
    ground_x, ground_y = x[~in_trees], y[~in_trees]
    tree_x = rng.uniform(20, 35, 500)
    tree_y = rng.uniform(20, 35, 500)
    noise_x = rng.uniform(0, 60, 40)
    noise_y = rng.uniform(0, 60, 40)

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = [ORIGIN_X, ORIGIN_Y - 60, 1600]
    header.scales = [0.01, 0.01, 0.01]
    header.add_crs(pyproj.CRS.from_epsg(26912))
    points = laspy.LasData(header)
    points.x = np.r_[ground_x, tree_x, noise_x] + ORIGIN_X
    points.y = ORIGIN_Y - np.r_[ground_y, tree_y, noise_y]
    points.z = np.r_[1600 + 0.02 * ground_x, 1612 + 0.02 * tree_x, np.full(40, 1900.0)]
    points.classification = np.r_[
        np.full(ground_x.size, 2), np.full(tree_x.size, 5), np.full(40, 7)
    ].astype("uint8")
    points.write(str(path))


def test_chm_from_real_pdal_pipeline(tmp_path):
    image = tmp_path / "image.tif"
    with rasterio.open(
        image, "w", driver="GTiff", width=200, height=200, count=4, dtype="uint8",
        crs="EPSG:26912", transform=from_origin(ORIGIN_X, ORIGIN_Y, 0.3, 0.3),
        photometric="MINISBLACK",
    ) as dst:
        dst.write(np.full((4, 200, 200), 90, "uint8"))
    points = tmp_path / "points.las"
    write_points(points)

    info = pdal_canopy_height(str(points), image, tmp_path / "chm.tif")

    with rasterio.open(tmp_path / "chm.tif") as chm:
        heights = chm.read(1, masked=True)
    trees = heights[26:39, 21:34]
    assert trees.count() == trees.size
    assert float(np.ma.median(trees)) == pytest.approx(12.0, abs=1.0)
    assert float(np.ma.median(heights[:15, :15])) < 0.5
    assert float(heights.max()) < 20
    assert info["filled_ground_pixels"] > 0
