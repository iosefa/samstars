"""Physical canopy rules stay consistent from seed selection through prompting."""
import json
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from samstars._prompts import sample_seed_prompts, validate_canopy_height
from samstars.features import seed_generation as seeds
from samstars.segmentation import engine, proposals
from samstars.training.chips import build_training_chips
from samstars.training.sam_training import Chips
from test_window_preprocessing import make_tile, Predictor


def raster(path, data):
    with rasterio.open(path, 'w', driver='GTiff', width=data.shape[1], height=data.shape[0],
                       count=1, dtype='float32', crs='EPSG:32605',
                       transform=from_origin(0,5,1,1), nodata=-9999) as dst:
        dst.write(np.asarray(data,np.float32),1)
    return path


@pytest.mark.parametrize('cutoff',[0,1,2,3])
def test_cutoff_is_inclusive_foreground_and_exclusive_background(cutoff):
    chm=np.array([[cutoff,cutoff+.2],[cutoff-.1,np.nan]])
    points, labels, _=sample_seed_prompts(chm,(0,0),.5,fg_extra=10,min_canopy_height=cutoff)
    values=chm[points[:,1].astype(int),points[:,0].astype(int)]
    assert np.all(values[labels==1]>=cutoff)
    assert np.all(values[labels==0]<cutoff)
    assert len(points)==3
    with pytest.raises(ValueError,match='below'):
        sample_seed_prompts(chm,(0,1),.5,min_canopy_height=cutoff)


@pytest.mark.parametrize('value',[True,-1,float('nan'),float('inf'),'2'])
def test_invalid_cutoffs_are_rejected(value):
    with pytest.raises(ValueError,match='height'):
        validate_canopy_height(value)


def test_chm_sampling_uses_the_containing_pixel_including_edges(tmp_path):
    data=np.arange(25,dtype=np.float32).reshape(5,5)
    path=raster(tmp_path/'chm.tif',data)
    got=seeds._sample_raster_values(path,np.array([1.5,3.5,4.5]),np.array([3.5,1.5,.5]))
    np.testing.assert_array_equal(got,[6,18,24])


def test_short_canopy_peak_uses_raw_height_not_density_cutoff(tmp_path):
    chm=np.zeros((5,5),np.float32); chm[3,3]=1
    params=seeds.SeedParams(chm_raster=raster(tmp_path/'chm.tif',chm),
        den_raster=raster(tmp_path/'den.tif',np.zeros_like(chm)),
        cost_raster=raster(tmp_path/'cost.tif',np.zeros_like(chm)),
        d_min=4.5,min_canopy_height=1,min_dist_px=1,gauss_sigma=1,rule_version=2)
    result=seeds.canonical_seeds(params)
    assert len(result)==1
    assert result.geometry.iloc[0].equals(Point(3.5,1.5))


def test_grouping_uses_chm_units_and_retains_an_actual_canopy_pixel(tmp_path,monkeypatch):
    chm=np.zeros((5,5),np.float32); chm[1,1]=2; chm[1,3]=2
    params=seeds.SeedParams(chm_raster=raster(tmp_path/'chm.tif',chm),
        den_raster=raster(tmp_path/'den.tif',chm),cost_raster=raster(tmp_path/'cost.tif',np.zeros_like(chm)),
        min_canopy_height=1,merge_radius=3,dz_merge=1,rule_version=2)
    def peaks(path,origin,*args,**kwargs):
        point=Point(1.5,3.5) if origin=='chm' else Point(3.5,3.5)
        return gpd.GeoDataFrame({'height':[2 if origin=='chm' else 1000],'origin':[origin]},
                               geometry=[point],crs='EPSG:32605')
    monkeypatch.setattr(seeds,'detect_peaks',peaks)
    result=seeds.canonical_seeds(params)
    assert len(result)==1
    assert result.geometry.iloc[0].equals(Point(1.5,3.5))
    assert result.geometry.iloc[0].x!=2.5  # Averaging would place the seed in ground.


def test_short_crown_prompts_match_during_training_and_prediction(tmp_path):
    tile=make_tile(tmp_path,shape=(32,32))
    with rasterio.open(tile/'chm.tif','r+') as dst:
        chm=dst.read(1); chm=np.where(chm>3,1.5,0).astype(np.float32); dst.write(chm,1)
    chips=tmp_path/'chips'
    build_training_chips(tile.parent,chips,chip_size=32,shared_windows=True,min_canopy_height=1)
    prompt=json.loads(next((chips/'prompts').glob('*.json')).read_text())
    x,y=prompt['point']
    with rasterio.open(tile/'img.tif') as src:
        point=Point(src.transform*(x+.5,y+.5))
    gpd.GeoDataFrame(geometry=[point],crs='EPSG:32605').to_file(tile/'seeds.gpkg')
    data=Chips(chips,img_size=32,pipeline_mode='simple',min_canopy_height=1,fg_extra=3)
    _,_,coords,labels,box=data[0]
    predictor=Predictor()
    proposals.process_chip(dict(img_path=str(tile/'img.tif'),tile=tile.name,chip_id=0,window=[0,0,32,32]),
        predictor,'cpu',pipeline_mode='simple',min_canopy_height=1,fg_extra=3,smooth_sigma=0)
    call=predictor.calls[0]
    np.testing.assert_array_equal(coords[labels>=0],call['point_coords'][0])
    np.testing.assert_array_equal(labels[labels>=0],call['point_labels'][0])
    np.testing.assert_array_equal(box,call['boxes'][0])
    native=coords[labels>=0].numpy()*32/1024
    heights=chm[native[:,1].astype(int),native[:,0].astype(int)]
    assert np.all(heights[labels[labels>=0].numpy()==1]>=1)
    assert np.all(heights[labels[labels>=0].numpy()==0]<1)


def test_cached_proposals_depend_on_the_canopy_rule(tmp_path,monkeypatch):
    tile=make_tile(tmp_path,shape=(32,32))
    checkpoint=tmp_path/'decoder.pth'; checkpoint.touch()
    model=SimpleNamespace(); model.to=lambda device:model; model.eval=lambda:model
    monkeypatch.setitem(proposals.sam_model_registry,'vit_l',lambda checkpoint:model)
    monkeypatch.setattr(proposals,'SamPredictor',lambda model:Predictor())
    args=dict(tiles_dir=tile.parent,out_dir=tmp_path/'out',sam_ckpt=checkpoint,
              pipeline_mode='simple',run_stardist=False)
    engine.run_segmentation(**args,min_canopy_height=1)
    with pytest.raises(ValueError,match='Cached outputs'):
        engine.run_segmentation(**args,min_canopy_height=2)
