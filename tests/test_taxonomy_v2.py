"""Versioned class-taxonomy regression tests: never reinterpret legacy pixel codes."""
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from landfuse.cli import main
from landfuse.contract import CLASSES, CLASSES_V2, SCHEMA, SCHEMA_V2, LandFuseError, Scene
from landfuse.model import train, infer
from landfuse.evaluate import evaluate
from landfuse.crossval import cross_validate
from landfuse.contract import LabeledScene


def raster(path, data, tags=None):
    with rasterio.open(path, 'w', driver='GTiff', width=data.shape[-1], height=data.shape[-2],
                       count=data.shape[0], dtype=data.dtype,
                       crs='EPSG:26912', transform=from_origin(430000, 3880000, .3, .3)) as dst:
        dst.write(data)
        if tags:
            dst.update_tags(**tags)
    return path


def scene_labels(tmp_path, code_left=7, code_right=4):
    yy, xx = np.indices((48, 48))
    raw = np.zeros((4, 48, 48), dtype=np.uint8)
    raw[0] = np.where(xx < 24, 210, 40)
    raw[1] = np.where(xx < 24, 30, 160)
    raw[2] = 45
    raw[3] = np.where(xx < 24, 20, 220)
    lbl = np.where(xx < 24, code_left, code_right).astype('uint8')
    return Scene(raster(tmp_path/'naip.tif',raw)), raster(tmp_path/'labels.tif',lbl[None])


def test_v2_schema_train_infer_evaluate_roundtrip(tmp_path):
    sc, labels = scene_labels(tmp_path)
    model=tmp_path/'model.joblib'
    meta=train(sc, labels, model, class_schema='v2', trees=10)
    assert meta['schema'] == SCHEMA_V2
    assert meta['classes'] == CLASSES_V2
    assert 'unpaved_road_track' in meta['train_class_counts']
    output=tmp_path/'classified.tif'
    result=infer(sc,model,output,trust_local_model=True)
    assert result['schema'] == SCHEMA_V2
    assert result['classes'] == CLASSES_V2
    with rasterio.open(output) as ds:
        assert ds.tags()['LANDFUSE_SCHEMA'] == SCHEMA_V2
        assert set(np.unique(ds.read(1))) <= {0, 4, 7}
    report=evaluate(output,labels,tmp_path/'evaluation.json')
    assert report['classifier_schema'] == SCHEMA_V2
    assert 'unpaved_road_track' in report['per_class']


def test_cli_explicit_v2_option(tmp_path):
    sc, labels = scene_labels(tmp_path)
    model=tmp_path/'cli.joblib'
    assert main(['train','--imagery',str(sc.imagery),'--labels',str(labels),
                 '--class-schema','v2','--trees','5','--output-model',str(model)]) == 0
    assert json.load(open(str(model)+'.json'))['schema'] == SCHEMA_V2


def test_v1_remains_default_and_rejects_v2_codes(tmp_path):
    sc,labels=scene_labels(tmp_path)
    assert CLASSES[2]=='bare_ground'
    with pytest.raises(LandFuseError,match='not defined by'):
        train(sc,labels,tmp_path/'v1.joblib',trees=5)
    assert not (tmp_path/'v1.joblib').exists()


def test_v2_sidecar_tampering_and_invalid_reference_rejected(tmp_path):
    sc, labels=scene_labels(tmp_path)
    model=tmp_path/'model.joblib'
    train(sc,labels,model,class_schema='v2',trees=5)
    manifest_path=tmp_path/'model.joblib.json'
    meta=json.loads(manifest_path.read_text())
    meta['classes']['2']='bare_ground'
    manifest_path.write_text(json.dumps(meta))
    with pytest.raises(LandFuseError,match='taxonomy mismatch'):
        infer(sc,model,tmp_path/'bad.tif',trust_local_model=True)
    assert not (tmp_path/'bad.tif').exists()
    meta['classes']['2']='rock_bedrock'
    manifest_path.write_text(json.dumps(meta))
    infer(sc,model,tmp_path/'good.tif',trust_local_model=True)
    invalid=raster(tmp_path/'invalid.tif',np.ones((1,48,48),dtype='uint8')*8)
    with pytest.raises(LandFuseError,match='codes'):
        evaluate(tmp_path/'good.tif', invalid,tmp_path/'bad_evaluation.json')


def test_v2_crossval_group_reports_schema(tmp_path):
    scene_list=[]
    for index in range(2):
        folder=tmp_path/str(index)
        folder.mkdir()
        sc,lbl=scene_labels(folder)
        scene_list.append(LabeledScene(sc,lbl,f'scene_{index}',f'group_{index}'))
    result=cross_validate(scene_list,tmp_path/'cv.json',by='group',per_class_limit=70,trees=5,
                          class_schema='v2')
    assert result['classifier_schema']==SCHEMA_V2
    assert 'unpaved_road_track' in result['per_class']
