"""Regression tests for no-terrain aerial imagery classification."""
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from landfuse.cli import main
from landfuse.contract import LandFuseError, Scene, load_scene_list, feature_names
from landfuse.model import train, infer
from landfuse.raster_ops import open_scene


def write(path: Path, data: np.ndarray, *, transform=None):
    transform = transform or from_origin(500000, 4000000, 1, 1)
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[-2],
                       width=data.shape[-1], count=data.shape[0], dtype=str(data.dtype),
                       crs="EPSG:26912", transform=transform) as raster:
        raster.write(data)
    return path


@pytest.mark.parametrize("bands", [3, 4])
def test_imagery_only_end_to_end(tmp_path, bands):
    h = w = 64
    yy, xx = np.mgrid[:h, :w]
    raw = np.empty((bands, h, w), dtype="uint8")
    raw[0] = np.where(xx < w // 2, 30, 180)
    raw[1] = np.where(xx < w // 2, 140, 25)
    raw[2] = 40
    if bands == 4:
        raw[3] = np.where(xx < w // 2, 190, 10)
    raw[:, 0:4, 0:4] = 0  # no data (all zero bands)
    imagery = write(tmp_path / f"naip_{bands}band.tif", raw)
    lbl = np.where(xx < w // 2, 3, 2).astype('uint8')
    lbl[:4, :4] = 0
    labels = write(tmp_path / "labels.tif", lbl[np.newaxis])
    scene = Scene(imagery)

    with open_scene(scene) as (img, terrain, meta):
        assert terrain is None
        assert meta['imagery_bands'] == bands
        assert meta['terrain_mode'] == 'imagery'
        assert meta['height_estimation_available'] is False
        assert meta['terrain_coverage_fraction'] is None

    model = tmp_path / 'forest.joblib'
    manifest = train(scene, labels, model, trees=10)
    assert manifest['terrain_mode'] == 'imagery'
    assert manifest['imagery_bands'] == bands
    assert manifest['feature_names'] == feature_names('imagery', bands)
    assert manifest['scenes'][0]['terrain_sha256'] is None

    output = tmp_path / 'classified.tif'
    summary = infer(scene, model, output, trust_local_model=True)
    assert summary['terrain_sha256'] is None
    assert summary['coverage']['nodata_pixels'] == 16
    assert summary['coverage']['classified_pixels'] == h * w - 16
    assert summary['height_estimation_available'] is False
    assert 'does not measure' in summary['warning']
    with rasterio.open(output) as ds:
        assert ds.crs.to_epsg() == 26912
        assert ds.transform == from_origin(500000, 4000000, 1, 1)
        pixels = ds.read(1)
        assert np.all(pixels[:4, :4] == 0)
        assert set(np.unique(pixels)).issubset({0, 2, 3})

    with pytest.raises(LandFuseError, match='Refusing to overwrite'):
        infer(scene, model, output, trust_local_model=True)


def test_cli_naip_only_without_terrain(tmp_path):
    values = np.ones((4, 32, 32), dtype='uint8') * 25
    values[:, :, :16] = 180
    img = write(tmp_path / 'imagery.tif', values)
    labels = write(tmp_path / 'labels.tif',
                   np.where(np.indices((32, 32))[1] < 16, 2, 3).astype('uint8')[None])
    model = tmp_path / 'model.joblib'
    outfile = tmp_path / 'result.tif'
    assert main(['inspect', '--imagery', str(img)]) == 0
    assert main(['train', '--imagery', str(img), '--labels', str(labels),
                 '--output-model', str(model), '--trees', '10']) == 0
    assert main(['classify', '--imagery', str(img), '--model', str(model),
                 '--trust-local-model', '--output', str(outfile)]) == 0
    assert outfile.exists()


def test_scene_lists_no_terrain_and_mode_mismatch(tmp_path):
    raw = np.ones((4, 20, 20), dtype='uint8')
    img = write(tmp_path / 'image.tif', raw)
    lbl = write(tmp_path / 'labels.tif', np.ones((1, 20, 20), dtype='uint8'))
    listed = tmp_path / 'scenes.json'
    listed.write_text(json.dumps({'terrain_mode': 'imagery', 'scenes': [
        {'id': 'A', 'imagery': img.name, 'labels': lbl.name}]}))
    scenes = load_scene_list(listed)
    assert len(scenes) == 1 and scenes[0].scene.terrain is None
    listed.write_text(json.dumps({'terrain_mode': 'imagery', 'scenes': [
        {'id': 'A', 'imagery': img.name, 'terrain': 'dem.tif', 'labels': lbl.name}]}))
    with pytest.raises(LandFuseError, match='must not specify terrain'):
        load_scene_list(listed)


def test_mode_and_band_rejection(tmp_path):
    img4 = write(tmp_path / 'four.tif', np.ones((4, 32, 32), dtype='uint8'))
    img3 = write(tmp_path / 'three.tif', np.ones((3, 32, 32), dtype='uint8'))
    terrain = write(tmp_path / 'terrain.tif', np.ones((1, 32, 32), dtype='float32'))
    with pytest.raises(LandFuseError, match='must not include'):
        with open_scene(Scene(img4, terrain, 'imagery')): pass
    with pytest.raises(LandFuseError, match='requires --terrain'):
        with open_scene(Scene(img4, None, 'dem')): pass
    with pytest.raises(LandFuseError, match='Four-band'):
        with open_scene(Scene(img3, terrain, 'dem')): pass
    with pytest.raises(LandFuseError, match='mode'):
        with open_scene(Scene(img3, None, 'weird')): pass


def test_trained_rgb_model_rejects_rgbnir(tmp_path):
    image3 = write(tmp_path / 'rgb.tif', np.ones((3, 40, 40), dtype='uint8') * 100)
    image4 = write(tmp_path / 'rgbnir.tif', np.ones((4, 40, 40), dtype='uint8') * 100)
    label = np.where(np.indices((40, 40))[1] < 20, 2, 3).astype('uint8')[None]
    labels = write(tmp_path / 'labels.tif', label)
    model = tmp_path / 'model.joblib'
    train(Scene(image3), labels, model, trees=5)
    with pytest.raises(LandFuseError, match='bands or feature schema'):
        infer(Scene(image4), model, tmp_path / 'bad.tif', trust_local_model=True)
    assert not (tmp_path / 'bad.tif').exists()


def test_spatial_crossvalidation_scene_groups_without_elevation(tmp_path):
    from landfuse.crossval import cross_validate
    paths = []
    for number in range(2):
        rgbnir = np.ones((4, 32, 32), dtype='uint8') * 30
        rgbnir[0, :, :16] = 170
        rgbnir[1, :, 16:] = 170
        rgbnir[3, :, 16:] = 220
        labels_array = np.where(np.indices((32, 32))[1] < 16, 2, 3).astype('uint8')[None]
        img = write(tmp_path / f'scene{number}.tif', rgbnir,
                    transform=from_origin(500000 + number * 100, 4000000, 1, 1))
        labels = write(tmp_path / f'labels{number}.tif', labels_array,
                       transform=from_origin(500000 + number * 100, 4000000, 1, 1))
        paths.append({'id': str(number), 'group': str(number),
                      'imagery': img.name, 'labels': labels.name})
    scenes_file = tmp_path / 'scenes.json'
    scenes_file.write_text(json.dumps({'terrain_mode': 'imagery', 'scenes': paths}))
    result = cross_validate(load_scene_list(scenes_file), tmp_path / 'holdout.json',
                            by='group', trees=5, per_class_limit=100)
    assert result['terrain_mode'] == 'imagery'
    assert result['imagery_bands'] == 4
    assert result['scored_samples'] > 0
    assert all(scene['terrain_sha256'] is None for scene in result['scenes'])


def test_historical_four_band_terrain_model_stays_compatible(tmp_path):
    """Old manifests and bundles do not include imagery_bands; accept as four-band."""
    import joblib
    import sklearn
    from landfuse.contract import MODEL_FEATURES, SCHEMA, sha256
    from sklearn.ensemble import RandomForestClassifier

    img = np.ones((4, 32, 32), dtype='uint8') * 30
    img[0, :, 16:] = 190
    imagery = write(tmp_path / 'naip.tif', img)
    dem = write(tmp_path / 'dem.tif', np.ones((1, 32, 32), dtype='float32'))
    train_x = np.array([[0.1] * 8, [0.8] * 8], dtype='float32')
    train_y = np.array([2, 3], dtype='uint8')
    forest = RandomForestClassifier(n_estimators=5, random_state=42).fit(train_x, train_y)
    path = tmp_path / 'legacy.joblib'
    joblib.dump({'model': forest, 'terrain_mode': 'dem', 'schema': SCHEMA,
                 'features': MODEL_FEATURES}, path)
    manifest = {'terrain_mode': 'dem', 'schema': SCHEMA, 'feature_names': MODEL_FEATURES,
                'model_sha256': sha256(path), 'sklearn_version': sklearn.__version__}
    Path(str(path) + '.json').write_text(json.dumps(manifest))
    result = infer(Scene(imagery, dem, 'dem'), path, tmp_path / 'legacy_result.tif',
                   trust_local_model=True)
    assert result['terrain_mode'] == 'dem'
    assert result['imagery_bands'] == 4
    assert result['height_estimation_available'] is False
