import json
import ssl
import types

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from landfuse import lidar, model
from landfuse.contract import LandFuseError, Scene
from landfuse.evaluate import evaluate
from landfuse.lidar import pdal_canopy_height
from landfuse.model import _split_limit, infer, train
from landfuse.naip import fetch_naip
from landfuse.raster_ops import get_features, open_scene

TRANSFORM = from_origin(365000, 3823000, 0.3, 0.3)


def write(path, arr, transform=TRANSFORM, nodata=None, **options):
    if arr.ndim == 2:
        arr = arr[None, ...]
    with rasterio.open(
        path, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1],
        count=arr.shape[0], dtype=arr.dtype, crs="EPSG:26912", nodata=nodata,
        transform=transform, **options,
    ) as dst:
        dst.write(arr)
    return path


def imagery(path, size, value=100):
    return write(path, np.full((4, size, size), value, "uint8"), photometric="MINISBLACK")


@pytest.mark.parametrize(
    ("counts", "limit", "expected"),
    [
        ([10, 10, 10], 9, [3, 3, 3]),
        ([1, 100, 100], 9, [1, 4, 4]),
        ([0, 5], 9, [0, 5]),
        ([100, 100], 1, [1, 0]),
    ],
)
def test_split_limit(counts, limit, expected):
    assert _split_limit(counts, limit) == expected


def test_training_samples_span_tiles(tmp_path, monkeypatch):
    size = 256
    rows, cols = np.indices((size, size))
    tile_id = rows // 64 * 4 + cols // 64
    bands = np.full((4, size, size), 60, "uint8")
    bands[0] = (tile_id * 10 + 20).astype("uint8")
    labels = np.full((size, size), 3, "uint8")
    labels[:, ::2] = 2
    image = write(tmp_path / "image.tif", bands, photometric="MINISBLACK")
    dem = write(tmp_path / "dem.tif", np.full((size, size), 1500, "float32"))
    label_path = write(tmp_path / "labels.tif", labels)
    captured = {}
    original_fit = model.RandomForestClassifier.fit

    def fit(self, X, y):
        captured["X"] = X
        return original_fit(self, X, y)

    monkeypatch.setattr(model.RandomForestClassifier, "fit", fit)
    manifest = train(Scene(image, dem, "dem"), label_path, tmp_path / "model.joblib",
                     tile=64, trees=2, per_class_limit=1000)
    tiles = np.unique(np.round((captured["X"][:, 0] * 255 - 20) / 10))
    assert tiles.size == 16
    assert manifest["train_class_tiles"]["low_vegetation"] == 16


def test_slope_has_no_tile_seam(tmp_path):
    size = 256
    y, x = np.mgrid[0:size, 0:size] * 0.3
    dem = (1500 + 0.002 * (x - 40) ** 2 + 0.003 * (y - 40) ** 2).astype("float32")
    image = imagery(tmp_path / "image.tif", size)
    terrain = write(tmp_path / "dem.tif", dem)
    with open_scene(Scene(image, terrain, "dem")) as (naip, dem_ds, _):
        full, _ = get_features(naip, dem_ds, "dem", Window(0, 0, size, size))
        left, _ = get_features(naip, dem_ds, "dem", Window(0, 0, 128, 128))
        right, _ = get_features(naip, dem_ds, "dem", Window(128, 0, 128, 128))
    assert np.abs(left[:, 127, 7] - full[:128, 127, 7]).max() < 1e-5
    assert np.abs(right[:, 0, 7] - full[:128, 128, 7]).max() < 1e-5


def test_slope_is_correct_beside_nodata(tmp_path):
    size = 200
    _, x = np.mgrid[0:size, 0:size] * 0.3
    dem = (1500 + 0.01 * x).astype("float32")
    dem[80:120, 80:120] = -9999
    image = imagery(tmp_path / "image.tif", size)
    terrain = write(tmp_path / "dem.tif", dem, nodata=-9999)
    with open_scene(Scene(image, terrain, "dem")) as (naip, dem_ds, _):
        features, valid = get_features(naip, dem_ds, "dem", Window(0, 0, size, size))
    assert not valid[90, 90]
    slopes = features[..., 7][valid]
    assert slopes == pytest.approx(0.01, abs=1e-3)


def test_zero_pixels_are_invalid(tmp_path):
    bands = np.full((4, 32, 32), 90, "uint8")
    bands[:, :, :8] = 0
    image = write(tmp_path / "image.tif", bands, photometric="MINISBLACK")
    dem = write(tmp_path / "dem.tif", np.full((32, 32), 1500, "float32"))
    with open_scene(Scene(image, dem, "dem")) as (naip, terrain, _):
        _, valid = get_features(naip, terrain, "dem", Window(0, 0, 32, 32))
    assert not valid[:, :8].any()
    assert valid[:, 8:].all()


@pytest.mark.parametrize(
    "url",
    [
        "https://s3-us-west-2.amazonaws.com/other-bucket/ept.json",
        "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/../other/ept.json",
        "https://example.com/usgs-lidar-public/ept.json",
    ],
)
def test_ept_url_must_be_usgs(url, tmp_path, monkeypatch):
    monkeypatch.setattr(lidar, "which", lambda name: "/usr/bin/pdal")
    image = imagery(tmp_path / "image.tif", 16)
    with pytest.raises(LandFuseError, match="approved USGS"):
        pdal_canopy_height(url, image, tmp_path / "chm.tif")


def test_chm_keeps_trees_without_ground_returns(tmp_path, monkeypatch):
    def fake_pdal(command, input=None, **kwargs):
        stages = json.loads(input)["pipeline"]
        surface = np.full((30, 30), 1500.0, "float32")
        if "== 2" in stages[-2]["expression"]:
            surface[10:20, 10:20] = -9999
        else:
            surface[10:20, 10:20] = 1512.0
        write(stages[-1]["filename"], surface, from_origin(365000, 3823000, 1, 1), nodata=-9999)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(lidar, "which", lambda name: "/usr/bin/pdal")
    monkeypatch.setattr(lidar.subprocess, "run", fake_pdal)
    image = imagery(tmp_path / "image.tif", 100)
    source = tmp_path / "points.laz"
    source.write_bytes(b"0")
    info = pdal_canopy_height(str(source), image, tmp_path / "chm.tif")
    with rasterio.open(tmp_path / "chm.tif") as chm:
        heights = chm.read(1, masked=True)
    assert heights[10:20, 10:20].count() == 100
    assert heights.max() == pytest.approx(12.0)
    assert info["filled_ground_pixels"] == 100


def test_failed_classification_leaves_no_output(tmp_path, monkeypatch):
    size = 600
    image = imagery(tmp_path / "image.tif", size)
    dem = write(tmp_path / "dem.tif", np.full((size, size), 1500, "float32"))
    labels = np.where(np.indices((size, size))[0] < 300, 3, 2).astype("uint8")
    label_path = write(tmp_path / "labels.tif", labels)
    scene = Scene(image, dem, "dem")
    train(scene, label_path, tmp_path / "model.joblib", trees=2, per_class_limit=100)
    original = model.RandomForestClassifier.predict
    calls = []

    def fail_on_second_tile(self, X):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("worker stopped")
        return original(self, X)

    monkeypatch.setattr(model.RandomForestClassifier, "predict", fail_on_second_tile)
    output = tmp_path / "classified.tif"
    with pytest.raises(RuntimeError):
        infer(scene, tmp_path / "model.joblib", output, trust_local_model=True)
    assert not output.exists()
    assert not (tmp_path / "classified.tif.partial").exists()


def test_evaluation_counts_missing_predictions(tmp_path):
    prediction = np.zeros((100, 100), "uint8")
    prediction[:5, :] = 2
    reference = np.full((100, 100), 2, "uint8")
    report = evaluate(write(tmp_path / "prediction.tif", prediction),
                      write(tmp_path / "reference.tif", reference),
                      tmp_path / "report.json")
    assert report["pixel_accuracy"] == pytest.approx(0.05)
    assert report["classified_pixel_accuracy"] == 1.0
    assert report["coverage"] == pytest.approx(0.05)
    assert report["per_class"]["bare_ground"]["recall"] == pytest.approx(0.05)
    assert report["unclassified_by_reference_class"]["bare_ground"] == 9500


def test_inspect_reports_terrain_coverage(tmp_path):
    image = imagery(tmp_path / "image.tif", 100)
    dem = write(tmp_path / "dem.tif", np.full((50, 50), 1500, "float32"))
    with open_scene(Scene(image, dem, "dem")) as (_, _, metadata):
        assert metadata["terrain_coverage_fraction"] == pytest.approx(0.25)


def test_fetch_error_includes_cause(tmp_path, monkeypatch):
    def fail(request, timeout=60):
        raise ssl.SSLCertVerificationError("certificate verify failed")

    monkeypatch.setattr("landfuse.naip.urlopen", fail)
    with pytest.raises(LandFuseError, match="certificate verify failed"):
        fetch_naip(tmp_path / "clip.tif", (365000, 3822970, 365030, 3823000), 26912)
