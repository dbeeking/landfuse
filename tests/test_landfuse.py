import json
from pathlib import Path
from io import BytesIO
from urllib.parse import parse_qs

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import from_origin, from_bounds

from landfuse.contract import LandFuseError, Scene
from landfuse.raster_ops import open_scene
from landfuse.model import infer, train
from landfuse.naip import fetch_naip
from landfuse.lidar import pdal_canopy_height
from landfuse.cli import main


def write(path, arr, transform=None, crs="EPSG:26912",nodata=None):
    if arr.ndim == 2: arr=arr[None,...]
    with rasterio.open(path,"w",driver="GTiff",width=arr.shape[2],height=arr.shape[1],
                       count=arr.shape[0],dtype=arr.dtype,crs=crs,
                       nodata=nodata,transform=transform or from_origin(430000,3880000,0.3,0.3)) as dst:
        dst.write(arr)
    return path


@pytest.fixture
def inputs(tmp_path):
    rng=np.random.default_rng(19)
    naip=np.zeros((4,48,48),dtype="uint8")
    naip[0]=np.where(np.indices((48,48))[0]<24,35,160)
    naip[1]=np.where(np.indices((48,48))[0]<24,90,70)
    naip[2]=40
    naip[3]=np.where(np.indices((48,48))[0]<24,170,45)
    dem=np.linspace(1500,1510,16*16,dtype="float32").reshape(16,16)
    labels=np.where(np.indices((48,48))[0]<24,3,2).astype("uint8")
    a=write(tmp_path/"naip.tif",naip)
    b=write(tmp_path/"dem.tif",dem,from_origin(430000,3880000,0.9,0.9))
    c=write(tmp_path/"labels.tif",labels)
    return Scene(a,b,"dem"),c


def test_inspect_modes(inputs):
    scene,_=inputs
    with open_scene(scene) as (_a,_b,metadata):
        assert metadata["image_pixel_size"][0] == pytest.approx(.3)
        assert metadata["terrain_pixel_size_native"][0] == pytest.approx(.9)
        assert metadata["terrain_mode"]=="dem"


def test_model_training_and_classification(inputs,tmp_path):
    scene,labels=inputs
    model=tmp_path/"landfuse.joblib"
    meta=train(scene,labels,model,trees=10,per_class_limit=250)
    assert model.exists() and meta["validated_accuracy"] is None
    assert set(meta["train_class_counts"])=={"bare_ground","low_vegetation"}
    output=tmp_path/"classified.tif"
    with pytest.raises(LandFuseError,match="trusted local"):
        infer(scene,model,output)
    result=infer(scene,model,output,trust_local_model=True)
    assert result["validation"]=="NOT_INDEPENDENTLY_VALIDATED"
    with rasterio.open(output) as ds:
        assert ds.crs==CRS.from_epsg(26912)
        assert ds.count==1 and ds.nodata == 0
        assert ds.shape==(48,48)
        assert set(np.unique(ds.read(1))) <= {0,2,3}
    assert Path(str(output)+".json").is_file()


def test_hash_tamper_fails(inputs,tmp_path):
    scene,labels=inputs
    model=tmp_path/"m.joblib"
    train(scene,labels,model,trees=5,per_class_limit=80)
    model.write_bytes(model.read_bytes()+b"TAMPERED")
    with pytest.raises(LandFuseError,match="SHA256 mismatch"):
        infer(scene,model,tmp_path/"out.tif",trust_local_model=True)


def test_mode_mismatch_fails(inputs,tmp_path):
    scene,labels=inputs;model=tmp_path/"m.joblib"
    train(scene,labels,model,trees=5,per_class_limit=80)
    with pytest.raises(LandFuseError,match="incompatible"):
        infer(Scene(scene.imagery,scene.terrain,"chm"),model,tmp_path/"out.tif",trust_local_model=True)


def test_reject_rgb_and_different_crs(inputs,tmp_path):
    scene,_=inputs
    a=write(tmp_path/"rgb.tif",np.zeros((3,48,48),dtype="uint8"))
    with pytest.raises(LandFuseError,match="Four-band"):
        with open_scene(Scene(a,scene.terrain,"dem")):pass
    b=write(tmp_path/"other.tif",np.ones((12,12),dtype="float32"),from_origin(300000,4000000,1,1))
    with pytest.raises(LandFuseError,match="do not overlap"):
        with open_scene(Scene(scene.imagery,b,"dem")):pass


def test_reject_unprojected_imagery(inputs,tmp_path):
    scene,_=inputs
    a=write(tmp_path/"latlon.tif",np.zeros((4,5,5),dtype="uint8"),from_origin(-112,34,0.00001,0.00001),crs="EPSG:4326")
    with pytest.raises(LandFuseError,match="projected metric CRS"):
        with open_scene(Scene(a,scene.terrain,"dem")):pass


def test_training_labels_alignment(inputs,tmp_path):
    scene,_=inputs
    bad=write(tmp_path/"bad_labels.tif",np.zeros((48,48),dtype="uint8"),from_origin(430001,3880000,0.3,0.3))
    with pytest.raises(LandFuseError,match="exactly aligned"):
        train(scene,bad,tmp_path/"model.joblib")


def test_lidar_requires_pdal(inputs,tmp_path,monkeypatch):
    scene,_=inputs
    monkeypatch.setattr("landfuse.lidar.which",lambda name:None)
    with pytest.raises(LandFuseError,match="PDAL executable"):
        pdal_canopy_height("none.laz",scene.imagery,tmp_path/"height.tif")


def test_bounded_naip_input_limit(tmp_path):
    with pytest.raises(LandFuseError,match="limit"):
        fetch_naip(tmp_path/"big.tif",(0,0,10000,10000),26912)
    with pytest.raises(LandFuseError,match="metre"):
        fetch_naip(tmp_path/"bad.tif",(-113,34,-112,35),4326)


def test_fetch_naip_mocked_4_band_geotiff(tmp_path,monkeypatch):
    class Resp:
        def __init__(self,data):self.data=data
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,n):return self.data
    def mocked(req,timeout=60):
        params=parse_qs(req.data.decode())
        assert params["bandIds"][0]=="0,1,2,3"
        bbox=[float(x) for x in params["bbox"][0].split(",")]
        w,h=map(int,params["size"][0].split(","))
        arr=np.zeros((4,h,w),dtype="uint8");arr[3]=200
        with MemoryFile() as mem:
            with mem.open(driver="GTiff",width=w,height=h,count=4,dtype="uint8",
                          crs="EPSG:26912",transform=from_bounds(*bbox,w,h)) as ds:
                ds.write(arr)
            return Resp(mem.read())
    monkeypatch.setattr("landfuse.naip.urlopen",mocked)
    out=tmp_path/"clip.tif"
    result=fetch_naip(out,(430000,3879970,430030,3880000),26912,tile_pixels=64)
    assert result["requests"]==4
    with rasterio.open(out) as ds:
        assert ds.count==4 and ds.width==100 and ds.height==100
        assert ds.read(4).min()==200


def test_reject_3_band_naip_response(tmp_path,monkeypatch):
    class Resp:
        def __init__(self,data):self.data=data
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def read(self,n):return self.data
    def mocked(req,timeout=60):
        bbox=list(map(float,parse_qs(req.data.decode())["bbox"][0].split(",")))
        w,h=map(int,parse_qs(req.data.decode())["size"][0].split(","))
        with MemoryFile() as mem:
            with mem.open(driver="GTiff",width=w,height=h,count=3,dtype="uint8",
                          crs="EPSG:26912",transform=from_bounds(*bbox,w,h)) as ds:
                ds.write(np.zeros((3,h,w),dtype="uint8"))
            return Resp(mem.read())
    monkeypatch.setattr("landfuse.naip.urlopen",mocked)
    out=tmp_path/"bad.tif"
    with pytest.raises(LandFuseError,match="4 georeferenced bands"):
        fetch_naip(out,(430000,3879997,430003,3880000),26912)
    assert not out.exists()


def test_cli_input_failure(inputs):
    scene,_=inputs
    assert main(["inspect","--imagery",str(scene.imagery),"--terrain",str(scene.terrain),"--terrain-mode","dem"])==0
    assert main(["classify","--imagery","notfound.tif","--terrain",str(scene.terrain),
                 "--terrain-mode","dem","--model","missing.joblib","--output","out.tif"])==2


def test_evaluation_metrics(tmp_path):
    from landfuse.evaluate import evaluate
    pred=np.full((10,10),3,dtype="uint8"); pred[:4,:]=2
    ref=pred.copy(); ref[:2,:]=3
    a=write(tmp_path/"pred.tif",pred)
    b=write(tmp_path/"ref.tif",ref)
    out=tmp_path/"metrics.json"
    r=evaluate(a,b,out)
    assert r["evaluated_pixels"]==100
    assert r["pixel_accuracy"]==pytest.approx(0.8)
    assert len(r["confusion_matrix"])==6
    assert r["reference_independence"]=="USER_ATTESTATION_REQUIRED"
    assert out.is_file()


def test_evaluation_misalignment_fails(tmp_path):
    from landfuse.evaluate import evaluate
    a=write(tmp_path/"a.tif",np.full((10,10),1,dtype="uint8"))
    b=write(tmp_path/"b.tif",np.full((10,10),1,dtype="uint8"),from_origin(440000,3880000,.3,.3))
    with pytest.raises(LandFuseError,match="exactly co-registered"):
        evaluate(a,b,tmp_path/"eval.json")
