import json

import numpy as np
import pytest
import rasterio
import sklearn
from rasterio.transform import from_origin

from landfuse.cli import main
from landfuse.contract import LabeledScene, LandFuseError, Scene, load_scene_list
from landfuse.crossval import bootstrap_interval, cross_validate, near_blocks
from landfuse.model import collect_samples, fit_forest, infer, train, train_scenes


def write(path, arr, transform, nodata=None, **options):
    if arr.ndim == 2:
        arr = arr[None, ...]
    with rasterio.open(
        path, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1],
        count=arr.shape[0], dtype=arr.dtype, crs="EPSG:26912", nodata=nodata,
        transform=transform, **options,
    ) as dst:
        dst.write(arr)
    return path


def labeled_scene(folder, scene_id, group, origin_x=365000.0, size=128):
    """Write a scene whose top half is vegetation and bottom half is bare ground."""
    folder.mkdir(parents=True, exist_ok=True)
    transform = from_origin(origin_x, 3823000, 0.3, 0.3)
    rows = np.indices((size, size))[0]
    top = rows < size // 2
    bands = np.zeros((4, size, size), "uint8")
    bands[0] = np.where(top, 40, 170)
    bands[1] = np.where(top, 90, 120)
    bands[2] = 50
    bands[3] = np.where(top, 180, 70)
    image = write(folder / f"{scene_id}_image.tif", bands, transform, photometric="MINISBLACK")
    dem = write(folder / f"{scene_id}_dem.tif", np.full((size, size), 1500, "float32"), transform)
    labels = write(folder / f"{scene_id}_labels.tif", np.where(top, 3, 2).astype("uint8"), transform)
    return LabeledScene(Scene(image, dem, "dem"), labels, scene_id, group)


def scene_list(path, scenes, mode="dem"):
    path.write_text(json.dumps({
        "terrain_mode": mode,
        "scenes": [{
            "id": item.scene_id,
            "group": item.group,
            "imagery": item.scene.imagery.name,
            "terrain": item.scene.terrain.name,
            "labels": item.labels.name,
        } for item in scenes],
    }))
    return path


def test_load_scene_list_resolves_relative_paths(tmp_path):
    scenes = [labeled_scene(tmp_path, "a", "north"), labeled_scene(tmp_path, "b", "south")]
    loaded = load_scene_list(scene_list(tmp_path / "scenes.json", scenes))
    assert [item.scene_id for item in loaded] == ["a", "b"]
    assert [item.group for item in loaded] == ["north", "south"]
    assert loaded[0].scene.imagery == tmp_path / "a_image.tif"


def test_scene_list_rejects_duplicate_ids(tmp_path):
    item = labeled_scene(tmp_path, "a", "north")
    with pytest.raises(LandFuseError, match="Duplicate scene id"):
        load_scene_list(scene_list(tmp_path / "scenes.json", [item, item]))


def test_scene_list_reports_missing_file(tmp_path):
    item = labeled_scene(tmp_path, "a", "north")
    item.labels.unlink()
    with pytest.raises(LandFuseError, match="Missing input file"):
        load_scene_list(scene_list(tmp_path / "scenes.json", [item]))


def test_multi_scene_training_uses_every_scene(tmp_path):
    scenes = [labeled_scene(tmp_path, name, name, 365000.0 + i * 1000) for i, name in enumerate("abc")]
    manifest = train_scenes(scenes, tmp_path / "model.joblib", trees=3, per_class_limit=300, tile=64)
    assert [record["id"] for record in manifest["scenes"]] == ["a", "b", "c"]
    assert all(record["samples"] > 0 for record in manifest["scenes"])
    assert manifest["train_class_counts"] == {"bare_ground": 300, "low_vegetation": 300}
    assert manifest["sklearn_version"] == sklearn.__version__


def test_samples_keep_map_positions(tmp_path):
    item = labeled_scene(tmp_path, "a", "north")
    samples = collect_samples([item], per_class_limit=100, tile=64)
    with rasterio.open(item.labels) as labels:
        rows, cols = rasterio.transform.rowcol(labels.transform, samples.x, samples.y)
        values = labels.read(1)[np.array(rows), np.array(cols)]
    assert np.array_equal(values, samples.labels)


def test_group_holdout_keeps_groups_apart(tmp_path):
    scenes = [labeled_scene(tmp_path, "a", "north"), labeled_scene(tmp_path, "b", "north", 366000.0),
              labeled_scene(tmp_path, "c", "south", 367000.0)]
    report = cross_validate(scenes, tmp_path / "cv.json", trees=5, per_class_limit=200, tile=64)
    assert report["holdout_method"] == "group"
    assert [fold["held_out"] for fold in report["folds"]] == [["north"], ["south"]]
    assert report["overall_accuracy"] == pytest.approx(1.0)
    assert report["macro_f1"] == pytest.approx(1.0)
    assert (tmp_path / "cv.json").is_file()


def test_near_blocks_uses_buffer_distance():
    x = np.array([99.0, 95.0, 89.0, 150.0, 211.0])
    y = np.array([50.0, 50.0, 50.0, 50.0, 50.0])
    held_out = np.array([[1, 0]])
    near = near_blocks(x, y, held_out, 100.0, 10.0)
    assert near.tolist() == [True, True, False, True, False]


def test_block_holdout_removes_location_leakage(tmp_path):
    """A location proxy looks perfect on random pixels and fails on held-out blocks."""
    size = 400
    transform = from_origin(365000, 3823000, 1.0, 1.0)
    rows, cols = np.indices((size, size))
    cell = (rows // 50) * 8 + cols // 50
    patch_labels = np.random.default_rng(3).integers(2, 4, size=64)
    labels = patch_labels[cell].astype("uint8")
    bands = np.full((4, size, size), 80, "uint8")
    bands[0] = (cols * 255 / (size - 1)).astype("uint8")
    bands[1] = (rows * 255 / (size - 1)).astype("uint8")
    image = write(tmp_path / "image.tif", bands, transform, photometric="MINISBLACK")
    dem = write(tmp_path / "dem.tif", np.full((size, size), 1500, "float32"), transform)
    label_path = write(tmp_path / "labels.tif", labels, transform)
    item = LabeledScene(Scene(image, dem, "dem"), label_path, "a", "a")

    samples = collect_samples([item], per_class_limit=4000, tile=128)
    order = np.random.default_rng(1).permutation(samples.labels.size)
    half = order.size // 2
    forest = fit_forest(samples.features[order[:half]], samples.labels[order[:half]], trees=20)
    random_split = np.mean(forest.predict(samples.features[order[half:]]) == samples.labels[order[half:]])

    report = cross_validate([item], tmp_path / "cv.json", by="block", block_m=100, buffer_m=10,
                            per_class_limit=4000, tile=128, trees=20)
    assert report["holdout_method"] == "block"
    assert report["buffer_excluded_training_samples"] > 0
    assert random_split > 0.9
    assert report["overall_accuracy"] < random_split - 0.2


def test_block_holdout_needs_more_than_one_block(tmp_path):
    item = labeled_scene(tmp_path, "a", "north", size=32)
    with pytest.raises(LandFuseError, match="one block"):
        cross_validate([item], tmp_path / "cv.json", by="block", block_m=5000, buffer_m=0,
                       per_class_limit=50, tile=32, trees=2)


def test_block_holdout_rejects_buffer_over_block_size(tmp_path):
    item = labeled_scene(tmp_path, "a", "north", size=32)
    with pytest.raises(LandFuseError, match="buffer_m"):
        cross_validate([item], tmp_path / "cv.json", by="block", block_m=10, buffer_m=20,
                       per_class_limit=50, tile=32, trees=2)


def test_bootstrap_interval_needs_five_units():
    assert bootstrap_interval(np.array([1.0, 2.0]), np.array([2.0, 2.0])) is None
    interval = bootstrap_interval(np.array([8.0, 9.0, 7.0, 10.0, 6.0, 8.0]), np.full(6, 10.0))
    assert 0.6 <= interval[0] <= interval[1] <= 1.0


def test_classification_reports_coverage(tmp_path):
    size = 64
    transform = from_origin(365000, 3823000, 0.3, 0.3)
    bands = np.full((4, size, size), 90, "uint8")
    bands[3, :, 16:] = 180
    bands[0, :32, 16:] = 40
    bands[:, :, :16] = 0
    labels = np.where(np.indices((size, size))[0] < 32, 3, 2).astype("uint8")
    labels[:, :16] = 0
    image = write(tmp_path / "image.tif", bands, transform, photometric="MINISBLACK")
    dem = write(tmp_path / "dem.tif", np.full((size, size), 1500, "float32"), transform)
    label_path = write(tmp_path / "labels.tif", labels, transform)
    scene = Scene(image, dem, "dem")
    train(scene, label_path, tmp_path / "model.joblib", trees=3, per_class_limit=100)
    summary = infer(scene, tmp_path / "model.joblib", tmp_path / "out.tif", trust_local_model=True)
    coverage = summary["coverage"]
    assert coverage["total_pixels"] == size * size
    assert coverage["nodata_pixels"] == 16 * size
    assert coverage["coverage_fraction"] == pytest.approx(0.75)
    assert sum(coverage["class_pixels"].values()) == coverage["classified_pixels"]
    assert sum(coverage["class_area_m2"].values()) == pytest.approx(coverage["classified_pixels"] * 0.09)


def test_model_from_other_sklearn_version_is_rejected(tmp_path):
    item = labeled_scene(tmp_path, "a", "north")
    model = tmp_path / "model.joblib"
    train_scenes([item], model, trees=2, per_class_limit=50)
    sidecar = tmp_path / "model.joblib.json"
    manifest = json.loads(sidecar.read_text())
    manifest["sklearn_version"] = "0.0.1"
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(LandFuseError, match="scikit-learn 0.0.1"):
        infer(item.scene, model, tmp_path / "out.tif", trust_local_model=True)


def test_cli_trains_and_cross_validates_from_scene_list(tmp_path, capsys):
    scenes = [labeled_scene(tmp_path, "a", "north"), labeled_scene(tmp_path, "b", "south", 366000.0)]
    listing = scene_list(tmp_path / "scenes.json", scenes)
    assert main(["train", "--scenes", str(listing), "--output-model", str(tmp_path / "m.joblib"),
                 "--trees", "3", "--per-class-limit", "100"]) == 0
    assert main(["cross-validate", "--scenes", str(listing), "--output", str(tmp_path / "cv.json"),
                 "--trees", "3", "--per-class-limit", "100"]) == 0
    assert json.loads((tmp_path / "cv.json").read_text())["holdout_method"] == "group"
    capsys.readouterr()


def test_cli_train_rejects_mixed_inputs(tmp_path, capsys):
    item = labeled_scene(tmp_path, "a", "north")
    listing = scene_list(tmp_path / "scenes.json", [item])
    code = main(["train", "--scenes", str(listing), "--labels", str(item.labels),
                 "--output-model", str(tmp_path / "m.joblib")])
    assert code == 2
    assert "not both" in capsys.readouterr().err
