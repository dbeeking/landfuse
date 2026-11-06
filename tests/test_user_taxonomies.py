"""Schema-driven training, evaluation, provenance, CLI, adversarial and legacy tests."""
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from landfuse.cli import main
from landfuse.contract import LandFuseError, Scene, LabeledScene
from landfuse.crossval import cross_validate
from landfuse.evaluate import evaluate
from landfuse.model import train, infer
from landfuse.schema import (load_class_schema, parse_schema, schema_from_raster_tags,
                             schema_from_canonical, canonical_json)

SCHEMA = {
    "id": "example.rock_and_grass", "version": "1.0.0",
    "description": "Reference-backed land cover for testing",
    "classes": [
        {"code": 2, "name": "rock", "description": "Exposed rock", "color": "#887766"},
        {"code": 9, "name": "vegetation", "color": "#22aA33"},
        {"code": 40, "name": "tree", "parent": "vegetation"},
    ],
}


def make_raster(path, data, tags=None, *, nodata=None, transform=None):
    with rasterio.open(path, 'w', driver='GTiff', width=data.shape[-1], height=data.shape[-2],
                       count=data.shape[0], dtype=data.dtype, crs='EPSG:26912',
                       transform=transform or from_origin(500000, 3800000, .3, .3),
                       nodata=nodata) as dst:
        dst.write(data)
        if tags:
            dst.update_tags(**tags)
    return path


def make_scene(tmp, *, bands=4, codes=(2, 40), size=48, tags=None, nodata=None):
    _, col = np.indices((size, size))
    raw = np.full((bands, size, size), 45, dtype=np.uint8)
    raw[0] = np.where(col < size//2, 180, 20)
    raw[1] = np.where(col < size//2, 25, 160)
    if bands == 4:
        raw[3] = np.where(col < size//2, 20, 220)
    labels = np.where(col < size//2, codes[0], codes[1]).astype('uint8')
    return (Scene(make_raster(tmp/'naip.tif', raw)),
            make_raster(tmp/'reference.tif', labels[None], tags=tags, nodata=nodata))


def write_schema(tmp, fmt='json', obj=None):
    path = tmp/('taxonomy.'+fmt)
    obj = SCHEMA if obj is None else obj
    if fmt == 'json':
        path.write_text(json.dumps(obj))
    else:
        import yaml
        path.write_text(yaml.safe_dump(obj))
    return path


def test_custom_canonical_document_hash_independent_of_file_format(tmp_path):
    j = load_class_schema(write_schema(tmp_path, 'json'))
    y = load_class_schema(write_schema(tmp_path, 'yaml'))
    assert j.digest == y.digest == parse_schema(SCHEMA).digest
    assert j.document['classes'] == sorted(j.document['classes'], key=lambda x: x['code'])
    assert j.classes == {0:'nodata',2:'rock',9:'vegetation',40:'tree'}
    assert j.document['classes'][1]['color'] == '#22aa33'
    assert schema_from_canonical(j.document).digest == j.digest
    assert j.trainable_codes == (2, 9, 40)


@pytest.mark.parametrize('mapping', [
    {1:'water', 2:'rocks'}, {'1':'water','2':'rocks'},
    {'1': {'name':'water'},'2':{'name':'rocks'}},
])
def test_dict_class_syntax(tmp_path, mapping):
    d={'name':'example.mapping','version':'2.0','classes':mapping}
    assert parse_schema(d).classes == {0:'nodata',1:'water',2:'rocks'}


@pytest.mark.parametrize('changes', [
    {'id':'',}, {'id':' bad'}, {'id':'wrong/slash'}, {'id':'bad\nname'},
    {'version':''}, {'version':'!!'}, {'version':' '*3},
    {'classes':None}, {'classes':[]}, {'classes':'water'},
    {'unexpected':1}, {'format':'unknown'},
    {'classes':[{'code':0,'name':'water'}]},
    {'classes':[{'code':-1,'name':'water'}]},
    {'classes':[{'code':1.5,'name':'water'}]},
    {'classes':[{'code':True,'name':'water'}]},
    {'classes':[{'code':None,'name':'water'}]},
    {'classes':[{'code':1,'name':'Water'}]},
    {'classes':[{'code':1,'name':'bad spaces'}]},
    {'classes':[{'code':1,'name':'nodata'}]},
    {'classes':[{'code':1,'name':'water','color':'red'}]},
    {'classes':[{'code':1,'name':'water','color':'#12345'}]},
    {'classes':[{'code':1,'name':'water','description':''}]},
    {'classes':[{'code':1,'name':'water','extra':1}]},
    {'classes':[{'code':1,'name':'water','parent':True}]},
    {'classes':[{'code':1,'name':'water','parent':'nonexistent'}]},
    {'classes':[{'code':1,'name':'water','parent':1}]},
    {'classes':[{'code':1,'name':'water','parent':0}]},
    {'classes':[{'code':1,'name':'water'}, {'code':1,'name':'rock'}]},
    {'classes':[{'code':1,'name':'water'}, {'code':2,'name':'water'}]},
    {'classes':[{'code':1,'name':'water','parent':'rock'}, {'code':2,'name':'rock','parent':'water'}]},
    {'classes':[{'code':1,'name':'water','parent':'rock'}, {'code':2,'name':'rock','parent':'soil'}, {'code':3,'name':'soil','parent':'water'}]},
    {'classes':[{'code':0,'name':'nodata','color':'#000000'},{'code':1,'name':'water'}]},
    {'classes':[{'code':65535,'name':'water'}, {'code':65536,'name':'bad'}]},
    {'description':''}, {'description':'x'*1001},
])
def test_invalid_schema_configurations(changes):
    s={'id':'test.invalid','version':'1.0', 'classes':[{'code':1,'name':'water'}]}
    s.update(changes)
    with pytest.raises(LandFuseError):
        parse_schema(s)


@pytest.mark.parametrize('which', [
    ('code',2, 3), ('name','rock','stones'), ('version','1.0.0','1.0.1'),
    ('description','Reference-backed land cover for testing','different'),
])
def test_changed_schema_changes_digest(which):
    field, before, after=which
    d=copy.deepcopy(SCHEMA)
    if field in {'version','description'}:
        d[field]=after
    else:
        d['classes'][0][field]=after
    assert parse_schema(d).digest != parse_schema(SCHEMA).digest


@pytest.mark.parametrize('path_suffix', ['.yaml','.yml','.json'])
def test_schema_read_supported_suffixes(tmp_path,path_suffix):
    p=tmp_path/('foo'+path_suffix)
    if path_suffix=='.json':
        p.write_text(json.dumps(SCHEMA))
    else:
        import yaml
        p.write_text(yaml.safe_dump(SCHEMA))
    assert load_class_schema(str(p)).classes[2]=='rock'


@pytest.mark.parametrize('bad', ['does-not-exist', 'nope.yaml', 'random.version.v900'])
def test_missing_custom_file_does_not_fallback_to_v1(bad):
    with pytest.raises(LandFuseError):
        load_class_schema(bad)


def test_filename_extension_and_size_limits(tmp_path):
    p=tmp_path/'wrong.txt';p.write_text(json.dumps(SCHEMA))
    with pytest.raises(LandFuseError,match='must end'):
        load_class_schema(p)
    p=tmp_path/'big.json';p.write_bytes(b' '*65537)
    with pytest.raises(LandFuseError,match='64 KiB'):
        load_class_schema(p)


def test_reject_custom_redefinition_of_legacy_schema(tmp_path):
    s=copy.deepcopy(SCHEMA);s['id']='landfuse.classification.v1'
    with pytest.raises(LandFuseError,match='must not reuse'):
        load_class_schema(write_schema(tmp_path,obj=s))
    legacy=load_class_schema('v1')
    tampered=copy.deepcopy(legacy.document)
    tampered['classes'][0]['name']='water_something_else'
    with pytest.raises(LandFuseError,match='cannot be redefined'):
        schema_from_canonical(tampered)


def test_order_independent_hash_and_hash_pins_parent(tmp_path):
    s=copy.deepcopy(SCHEMA);s['classes'].reverse()
    a=load_class_schema(write_schema(tmp_path,obj=s))
    b=parse_schema(SCHEMA)
    assert a.digest==b.digest
    s['classes'][0].pop('parent',None)
    assert parse_schema(s).digest!=b.digest


@pytest.mark.parametrize('bands', [3,4])
def test_train_infer_and_evaluate_custom_sparse_codes(tmp_path,bands):
    p=write_schema(tmp_path)
    sc,lbl=make_scene(tmp_path,bands=bands)
    model=tmp_path/'custom.joblib'
    meta=train(sc,lbl,model,trees=9,per_class_limit=100,class_schema=str(p))
    expected=load_class_schema(p)
    assert meta['schema_sha256']==expected.digest
    assert meta['schema_document']==expected.document
    assert meta['train_class_counts'].keys()=={'rock','tree'}
    bundle=joblib.load(model)
    assert bundle['schema_sha256']==expected.digest
    assert bundle['schema_document']==expected.document
    out=tmp_path/'classification.tif'
    result=infer(sc,model,out,trust_local_model=True,class_schema=str(p))
    assert result['coverage']['classified_pixels']==48*48
    assert result['coverage']['class_pixels']['rock']+result['coverage']['class_pixels']['tree']==48*48
    assert result['coverage']['class_pixels']['vegetation']==0
    with rasterio.open(out) as image:
        assert image.tags()['LANDFUSE_SCHEMA_SHA256']==expected.digest
        assert schema_from_raster_tags(image.tags()).classes==expected.classes
        assert set(np.unique(image.read(1)))<= {2,40}
        assert image.nodata==0
    with pytest.raises(LandFuseError,match='Untagged custom reference'):
        evaluate(out,lbl,tmp_path/'eval_bad.json')
    score=evaluate(out,lbl,tmp_path/'eval_ok.json',class_schema=str(p))
    assert score['classifier_schema_sha256']==expected.digest
    assert score['per_class']['tree']['support']>0
    assert score['per_class']['vegetation']['support']==0
    assert score['pixel_accuracy']>0.9
    assert not (tmp_path/'eval_bad.json').exists()


def test_custom_scene_list_group_crossvalidation(tmp_path):
    manifest=[]
    for n in range(3):
        sub=tmp_path/f'scene{n}';sub.mkdir()
        sc,lbl=make_scene(sub)
        manifest.append(LabeledScene(sc,lbl,f'scene{n}',f'group{n}'))
    taxonomy=write_schema(tmp_path)
    score=cross_validate(manifest,tmp_path/'crossval.json',by='group',trees=6,
                         per_class_limit=90,class_schema=str(taxonomy))
    assert score['classifier_schema_sha256']==load_class_schema(taxonomy).digest
    assert score['held_out_units']==3
    assert 'rock' in score['per_class']


def test_cli_schema_train_classify_evaluate(tmp_path,capsys):
    taxonomy=write_schema(tmp_path,'yaml')
    scene,lbl=make_scene(tmp_path)
    assert main(['schema','--class-schema',str(taxonomy)])==0
    assert 'example.rock_and_grass' in capsys.readouterr().out
    model=tmp_path/'cli.joblib';out=tmp_path/'cli.tif'
    assert main(['train','--imagery',str(scene.imagery),'--labels',str(lbl),
                 '--class-schema',str(taxonomy),'--trees','5','--output-model',str(model)])==0
    assert main(['classify','--imagery',str(scene.imagery),'--model',str(model),
                 '--class-schema',str(taxonomy),'--trust-local-model','--output',str(out)])==0
    assert main(['evaluate','--prediction',str(out),'--reference',str(lbl),
                 '--class-schema',str(taxonomy),'--output',str(tmp_path/'evaluation.json')])==0


def test_cli_rejects_model_other_schema(tmp_path):
    taxonomy=write_schema(tmp_path)
    sc,lbl=make_scene(tmp_path)
    model=tmp_path/'forest.joblib';out=tmp_path/'bad.tif'
    train(sc,lbl,model,class_schema=str(taxonomy),trees=5)
    assert main(['classify','--imagery',str(sc.imagery),'--model',str(model),
                 '--class-schema','v2','--trust-local-model','--output',str(out)])==2
    assert not out.exists()


@pytest.mark.parametrize('mutate', ['rename','version','digest','remove_document','classes'])
def test_tampered_sidecar_fails_closed(tmp_path,mutate):
    p=write_schema(tmp_path);sc,lbl=make_scene(tmp_path)
    model=tmp_path/'model.joblib';train(sc,lbl,model,trees=5,class_schema=str(p))
    side=Path(str(model)+'.json');meta=json.loads(side.read_text())
    if mutate=='rename': meta['schema_document']['classes'][0]['name']='other_rock'
    if mutate=='version':meta['schema_document']['version']='99'
    if mutate=='digest':meta['schema_sha256']='f'*64
    if mutate=='remove_document':del meta['schema_document']
    if mutate=='classes':meta['classes']['2']='wrong_class'
    side.write_text(json.dumps(meta))
    with pytest.raises(LandFuseError):
        infer(sc,model,tmp_path/'invalid.tif',trust_local_model=True)
    assert not (tmp_path/'invalid.tif').exists()


def test_modified_model_fails_before_joblib_load(tmp_path,monkeypatch):
    p=write_schema(tmp_path);sc,lbl=make_scene(tmp_path)
    model=tmp_path/'model.joblib';train(sc,lbl,model,trees=5,class_schema=str(p))
    with open(model,'ab') as fd:fd.write(b'tampered')
    called=[]
    monkeypatch.setattr(joblib,'load',lambda *a,**k: called.append(True))
    with pytest.raises(LandFuseError,match='SHA256 mismatch'):
        infer(sc,model,tmp_path/'bad.tif',trust_local_model=True)
    assert called==[]


def test_no_deserialization_without_explicit_trust(tmp_path,monkeypatch):
    p=write_schema(tmp_path);sc,lbl=make_scene(tmp_path)
    model=tmp_path/'model.joblib';train(sc,lbl,model,trees=5,class_schema=str(p))
    called=[]
    monkeypatch.setattr(joblib,'load',lambda *a,**k: called.append(True))
    with pytest.raises(LandFuseError,match='trusted local'):
        infer(sc,model,tmp_path/'bad.tif')
    assert called==[]


def test_reject_unverified_labels_schema_tag(tmp_path):
    p=write_schema(tmp_path)
    legacy=load_class_schema('v1')
    sc,lbl=make_scene(tmp_path,tags={'LANDFUSE_SCHEMA':legacy.identifier})
    with pytest.raises(LandFuseError,match='Training label raster class schema mismatch'):
        train(sc,lbl,tmp_path/'bad.joblib',trees=5,class_schema=str(p))


def test_reject_nodata_collision_with_custom_class(tmp_path):
    p=write_schema(tmp_path)
    sc,lbl=make_scene(tmp_path,nodata=40)
    with pytest.raises(LandFuseError,match='nodata value conflicts'):
        train(sc,lbl,tmp_path/'bad.joblib',trees=5,class_schema=str(p))


@pytest.mark.parametrize('invalid_code',[1,7,41,255])
def test_reject_unknown_class_code_in_training(tmp_path,invalid_code):
    p=write_schema(tmp_path)
    sc,lbl=make_scene(tmp_path,codes=(invalid_code,40))
    with pytest.raises(LandFuseError,match='not defined'):
        train(sc,lbl,tmp_path/'bad.joblib',trees=5,class_schema=str(p))


def test_sparse_highest_class_code_255_roundtrip(tmp_path):
    data={'id':'example.high','version':'1','classes':{2:'soil',255:'buildings'}}
    p=write_schema(tmp_path,obj=data)
    sc,lbl=make_scene(tmp_path,codes=(2,255))
    model=tmp_path/'high.joblib';out=tmp_path/'high.tif'
    train(sc,lbl,model,trees=5,class_schema=str(p))
    infer(sc,model,out,trust_local_model=True)
    with rasterio.open(out) as ds:
        assert set(np.unique(ds.read(1))) <= {2,255}
    score=evaluate(out,lbl,tmp_path/'evaluation.json',class_schema=str(p))
    assert score['evaluated_pixels']==2304


@pytest.mark.parametrize('tamper', ['digest','id','json','names'])
def test_tampered_raster_schema_tags_are_rejected(tmp_path,tamper):
    p=write_schema(tmp_path);sc,lbl=make_scene(tmp_path)
    model=tmp_path/'forest.joblib';out=tmp_path/'classification.tif'
    train(sc,lbl,model,trees=5,class_schema=str(p))
    infer(sc,model,out,trust_local_model=True)
    with rasterio.open(out,'r+') as raster:
        if tamper=='digest': raster.update_tags(LANDFUSE_SCHEMA_SHA256='f'*64)
        if tamper=='id': raster.update_tags(LANDFUSE_SCHEMA='other.taxonomy')
        if tamper=='json': raster.update_tags(LANDFUSE_SCHEMA_JSON='{"invalid":true}')
        if tamper=='names': raster.update_tags(LANDFUSE_SCHEMA_JSON='definitely not JSON')
    with pytest.raises(LandFuseError):
        evaluate(out,lbl,tmp_path/'invalid.json',class_schema=str(p))


def test_mismatched_reference_tags_are_rejected_even_when_codes_same(tmp_path):
    p=write_schema(tmp_path);sc,lbl=make_scene(tmp_path)
    model=tmp_path/'model.joblib';out=tmp_path/'classified.tif'
    train(sc,lbl,model,trees=5,class_schema=str(p));infer(sc,model,out,trust_local_model=True)
    other={'id':'another.same_numeric_codes','version':'1','classes':{2:'other',40:'other2'}}
    other_s=parse_schema(other)
    with rasterio.open(lbl,'r+') as ds:
        ds.update_tags(LANDFUSE_SCHEMA=other_s.identifier,LANDFUSE_SCHEMA_SHA256=other_s.digest,
                       LANDFUSE_SCHEMA_JSON=canonical_json(other_s.document))
    with pytest.raises(LandFuseError,match='Independent reference class schema mismatch'):
        evaluate(out,lbl,tmp_path/'invalid.json',class_schema=str(p))


def test_hash_does_not_depend_on_file_path(tmp_path):
    p=write_schema(tmp_path)
    q=tmp_path/'copy.json';q.write_bytes(p.read_bytes())
    assert load_class_schema(p).digest == load_class_schema(q).digest


def test_schema_with_parent_metadata_does_not_auto_generate_parent_label(tmp_path):
    p=write_schema(tmp_path)
    sc,lbl=make_scene(tmp_path)
    model=tmp_path/'model.joblib';out=tmp_path/'classified.tif'
    train(sc,lbl,model,trees=6,class_schema=str(p));summary=infer(sc,model,out,trust_local_model=True)
    assert summary['classes'][9]=='vegetation'
    assert summary['coverage']['class_pixels']['vegetation']==0  # parent is metadata, not inferred


def test_old_v1_model_classifies_and_legacy_v2_model_classifies(tmp_path):
    for version,codes in [('v1',(2,3)),('v2',(7,4))]:
        folder=tmp_path/version;folder.mkdir()
        sc,lbl=make_scene(folder,codes=codes)
        model=folder/'forest.joblib';out=folder/'classified.tif'
        train(sc,lbl,model,trees=5,class_schema=version)
        # Emulate models created before taxonomy digest/provenance was added.
        data=joblib.load(model)
        data.pop('schema_sha256');data.pop('schema_document')
        joblib.dump(data,model,compress=3)
        m=Path(str(model)+'.json');meta=json.loads(m.read_text())
        meta.pop('schema_sha256');meta.pop('schema_document')
        from landfuse.contract import sha256
        meta['model_sha256']=sha256(model)
        m.write_text(json.dumps(meta))
        res=infer(sc,model,out,trust_local_model=True)
        assert res['schema']=='landfuse.classification.'+version


@pytest.mark.parametrize('high_code',[256,512,4096,65535])
def test_uint16_schema_class_codes_and_geotiff(tmp_path,high_code):
    obj={'id':'high.codes','version':'2','classes':{1:'water',str(high_code):'specific_class'}}
    p=write_schema(tmp_path,obj=obj)
    sc,lbl=make_scene(tmp_path,codes=(1,2))
    # Replace uint8 fixture with a uint16 reference containing the high code.
    arr=np.where(np.indices((48,48))[1]<24,1,high_code).astype('uint16')
    make_raster(lbl,arr[None])
    model=tmp_path/'model.joblib';out=tmp_path/'out.tif'
    meta=train(sc,lbl,model,class_schema=str(p),trees=6)
    assert meta['classes'][high_code]=='specific_class'
    result=infer(sc,model,out,trust_local_model=True)
    with rasterio.open(out) as ds:
        assert ds.dtypes[0]=='uint16'
        assert set(np.unique(ds.read(1))) <= {1,high_code}
    assert result['coverage']['classified_pixels']==48*48
    score=evaluate(out,lbl,tmp_path/'score.json',class_schema=str(p))
    assert score['evaluated_pixels']==48*48


def test_reject_codes_above_uint16_limit():
    with pytest.raises(LandFuseError,match='out of range'):
        parse_schema({'id':'test.big','version':'1','classes':{65536:'too_high'}})


def test_raster_class_names_tag_tampering_rejected(tmp_path):
    p=write_schema(tmp_path);sc,lbl=make_scene(tmp_path)
    model=tmp_path/'model.joblib';out=tmp_path/'out.tif'
    train(sc,lbl,model,trees=5,class_schema=str(p))
    infer(sc,model,out,trust_local_model=True)
    with rasterio.open(out,'r+') as raster:
        raster.update_tags(LANDFUSE_CLASS_NAMES='{"2":"different"}')
    with pytest.raises(LandFuseError,match='class names'):
        evaluate(out,lbl,tmp_path/'score.json',class_schema=str(p))


def test_numpy_dtype_unchanged_for_builtin_v1(tmp_path):
    sc,lbl=make_scene(tmp_path,codes=(2,3))
    model=tmp_path/'v1.joblib';out=tmp_path/'v1.tif'
    train(sc,lbl,model,trees=5)
    infer(sc,model,out,trust_local_model=True)
    with rasterio.open(out) as raster:
        assert raster.dtypes[0]=='uint8'

@pytest.mark.parametrize('content,ext', [
    ('{"id":"x.one","id":"x.two","version":"1","classes":{"1":"soil"}}', '.json'),
    ('{"id":"x.one","version":"1","classes":{"1":"soil","1":"rock"}}', '.json'),
    ('id: x.one\nid: x.two\nversion: "1"\nclasses:\n  1: soil\n', '.yaml'),
    ('id: x.one\nversion: "1"\nclasses:\n  1: soil\n  1: rock\n', '.yaml'),
])
def test_duplicate_json_yaml_keys_are_rejected(tmp_path,content,ext):
    schema=tmp_path/('schema'+ext);schema.write_text(content)
    with pytest.raises(LandFuseError,match='Duplicate'):
        load_class_schema(schema)

@pytest.mark.parametrize('mode', ['dem','chm'])
def test_custom_schema_both_terrain_modes(tmp_path,mode):
    schema=write_schema(tmp_path)
    sc,lbl=make_scene(tmp_path)
    terrain_data=np.ones((1,48,48),dtype=np.float32)*2
    terrain=make_raster(tmp_path/'terrain.tif',terrain_data)
    scene=Scene(sc.imagery,terrain,mode)
    model=tmp_path/'terrain_model.joblib';out=tmp_path/'classified.tif'
    meta=train(scene,lbl,model,class_schema=str(schema),trees=5)
    assert meta['terrain_mode']==mode
    res=infer(scene,model,out,trust_local_model=True,class_schema=str(schema))
    assert res['schema_sha256']==load_class_schema(schema).digest
    assert res['terrain_mode']==mode


@pytest.mark.parametrize('n',[2,3,8,16,32,64,128,200])
def test_many_unrelated_classes_schema_is_supported(n):
    schema={'id':f'example.catalog{n}', 'version':'1',
            'classes':[{'code':c*2, 'name':f'class_{c}'} for c in range(1,n+1)]}
    obj=parse_schema(schema)
    assert len(obj.trainable_codes)==n
    assert obj.trainable_codes[0]==2
    assert obj.trainable_codes[-1]==n*2
    assert len(obj.digest)==64


def test_class_schema_model_bundle_mismatch_even_with_rehashed_model(tmp_path):
    schema=write_schema(tmp_path);sc,lbl=make_scene(tmp_path)
    model=tmp_path/'forest.joblib';out=tmp_path/'invalid.tif'
    train(sc,lbl,model,class_schema=str(schema),trees=5)
    data=joblib.load(model)
    data['schema_document']['classes'][0]['name']='counterfeit'
    joblib.dump(data,model,compress=3)
    meta_path=Path(str(model)+'.json')
    meta=json.loads(meta_path.read_text())
    from landfuse.contract import sha256
    meta['model_sha256']=sha256(model)
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(LandFuseError,match='Serialized model classification schema/hash mismatch'):
        infer(sc,model,out,trust_local_model=True)
    assert not out.exists()


def test_schema_same_codes_different_semantics_is_rejected_at_inference(tmp_path):
    schema=write_schema(tmp_path);sc,lbl=make_scene(tmp_path)
    model=tmp_path/'forest.joblib';out=tmp_path/'bad.tif'
    train(sc,lbl,model,class_schema=str(schema),trees=5)
    other=copy.deepcopy(SCHEMA)
    other['classes'][0]['name']='stone'
    other_path=tmp_path/'other.json';other_path.write_text(json.dumps(other))
    with pytest.raises(LandFuseError,match='Requested class schema does not match'):
        infer(sc,model,out,trust_local_model=True,class_schema=str(other_path))


def test_predicted_codes_validated_on_inference(tmp_path,monkeypatch):
    schema=write_schema(tmp_path);sc,lbl=make_scene(tmp_path)
    model=tmp_path/'forest.joblib';out=tmp_path/'invalid.tif'
    train(sc,lbl,model,class_schema=str(schema),trees=5)
    original_load=joblib.load
    def mutate(path):
        data=original_load(path)
        class FakeModel:
            def predict(self, array):
                return np.full(len(array),3,dtype=np.uint16)
        data['model']=FakeModel()
        return data
    monkeypatch.setattr(joblib,'load',mutate)
    with pytest.raises(LandFuseError,match='predicted codes outside'):
        infer(sc,model,out,trust_local_model=True)
    assert not out.exists()
    assert not (tmp_path/'invalid.tif.partial').exists()
