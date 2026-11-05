# LandFuse

LandFuse classifies land cover from high-resolution aerial imagery alone (NAIP-only mode), or combines four-band aerial imagery with either a bare-earth DEM or a LiDAR-derived canopy height model. It writes a georeferenced classified GeoTIFF and a JSON sidecar with input hashes and coverage. It runs as a standalone command-line tool.


## Inputs

1. An aerial imagery GeoTIFF with four raw bands in **R, G, B, NIR** order. Map tiles and screenshots won't work.
2. Either a bare-earth DEM GeoTIFF (`--terrain-mode dem`) or a LiDAR-derived canopy height GeoTIFF (`--terrain-mode chm`). `prepare-lidar` builds the canopy height raster from LAS, LAZ, or a public EPT source using PDAL.

Training also needs a labeled class raster aligned to the imagery. Classification uses the two inputs above plus a trained model.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
pytest -q
landfuse --help
```

PDAL is a separate command-line dependency and is only needed for `prepare-lidar`. Pip doesn't install it.

## NAIP-only classification (no DEM or LiDAR)

**New in v0.2.0:** georeferenced raw 8-bit NAIP RGB or RGB-NIR GeoTIFFs work without terrain.
Input classes remain `water`, `bare_ground`, `low_vegetation`, `tall_vegetation`,
`built_surface`, `road_or_path`; code `0` is nodata/unclassified. The classifier
uses only spectral features and **does not measure plant height**: distinguishing
low from tall vegetation requires independently labeled training examples, and
may be unreliable without LiDAR. A trained model is NOT included.

```bash
landfuse inspect --imagery /data/naip_4band.tif
landfuse train --imagery /data/naip_4band.tif \
  --labels /data/training_labels.tif --output-model /data/models/naip_only.joblib
landfuse classify --imagery /data/naip_4band.tif \
  --model /data/models/naip_only.joblib --trust-local-model \
  --output /data/results/naip_landcover.tif
```

Train with independently prepared per-pixel labels (0 unlabeled, 1–6 class codes).
For robust accuracy checks, create a scene list with `"terrain_mode":"imagery"`
(no `terrain` keys) and run spatial `cross-validate`. A model trained for RGB cannot
classify RGB-NIR imagery, and imagery-only models cannot be used in DEM/CHM mode.
Inputs must be local georeferenced, north-up, metric CRS GeoTIFFs, not map tile PNGs.

## Workflows

### One scene with a bare-earth DEM

```bash
landfuse inspect \
  --imagery /data/naip_yavapai_4band.tif \
  --terrain /data/3dep_dem_clip.tif --terrain-mode dem

# Label pixels 1 to 6 by class. Use 0 for unlabeled pixels.
landfuse train \
  --imagery /data/naip_yavapai_4band.tif \
  --terrain /data/3dep_dem_clip.tif --terrain-mode dem \
  --labels /data/validated_training_labels.tif \
  --output-model /data/models/yavapai_dem_rf.joblib

landfuse classify \
  --imagery /data/naip_yavapai_4band.tif \
  --terrain /data/3dep_dem_clip.tif --terrain-mode dem \
  --model /data/models/yavapai_dem_rf.joblib \
  --trust-local-model --output /data/results/landcover_dem.tif
```

The `classify` sidecar reports total, classified, and nodata pixels, the coverage fraction, and the pixels and area for each class.

### Several scenes

A single scene can't represent a county, so list several in a JSON file. Relative paths are resolved from the file's folder. Scenes that share a `group` are held out together during validation. If `group` is omitted, each scene is its own group.

```json
{
  "terrain_mode": "dem",
  "scenes": [
    {
      "id": "prescott",
      "group": "central_highlands",
      "imagery": "prescott_naip.tif",
      "terrain": "prescott_dem.tif",
      "labels": "prescott_labels.tif"
    },
    {
      "id": "congress",
      "group": "desert",
      "imagery": "congress_naip.tif",
      "terrain": "congress_dem.tif",
      "labels": "congress_labels.tif"
    }
  ]
}
```

```bash
landfuse train --scenes /data/scenes.json --output-model /data/models/yavapai_dem_rf.joblib
```

### NAIP and LiDAR

```bash
# Requires PDAL. The LiDAR needs a valid CRS and classified ground returns
# (ASPRS class 2), and it must cover the imagery extent.
landfuse prepare-lidar \
  --lidar /data/yavapai_ground_classified.laz \
  --imagery /data/naip_yavapai_4band.tif \
  --output /data/height_chm.tif --resolution 1.0

# DEM and CHM modes need separate models. Rasterize the training labels
# to the NAIP grid first.
landfuse train --imagery /data/naip_yavapai_4band.tif \
  --terrain /data/height_chm.tif --terrain-mode chm \
  --labels /data/validated_training_labels.tif \
  --output-model /data/models/yavapai_lidar_rf.joblib

landfuse classify --imagery /data/naip_yavapai_4band.tif \
  --terrain /data/height_chm.tif --terrain-mode chm \
  --model /data/models/yavapai_lidar_rf.joblib \
  --trust-local-model --output /data/results/landcover_lidar.tif
```

You can also stream 3DEP LiDAR with `--lidar https://s3-us-west-2.amazonaws.com/usgs-lidar-public/<USGS-project>/ept.json`. Remote sources are limited to the USGS public bucket, and only the imagery extent is requested. Coverage varies, so check it before running.

### Download NAIP pixels

Use a bbox in a projected CRS with metre units.

```bash
landfuse fetch-naip \
  --bbox XMIN YMIN XMAX YMAX --epsg 26912 \
  --output /data/naip_yavapai_4band.tif
```

`fetch-naip` requests four-band TIFFs from the 2023 ImageServer and mosaics them into a local GeoTIFF. It stops if the service returns a three-band image, a JSON error, or a response without georeferencing. The default download limit is 5 million pixels, and each scene is limited to 20 million pixels. The provider can throttle or change its service.

## Output classes

| Pixel | Class |
|---:|---|
| 0 | Nodata or no terrain evidence |
| 1 | Water |
| 2 | Bare ground |
| 3 | Low vegetation |
| 4 | Tall vegetation |
| 5 | Built surface |
| 6 | Road or path |

Results depend on accurate, spatially representative training labels. Separating buildings from roads usually needs texture or context features, and this version uses only spectral and terrain features. DEM mode doesn't provide canopy or object heights. The model isn't calibrated or approved for site safety decisions.

## Validation

Nearby pixels look alike, so random pixel splits overstate accuracy. `cross-validate` holds out whole areas instead:

```bash
landfuse cross-validate --scenes /data/scenes.json --output /data/results/crossval.json
```

- With two or more groups, it trains on all but one group and scores the group left out.
- With one group, or with `--by block`, it splits the labeled pixels into map blocks. Set the size with `--block-m` and the folds with `--folds`. Training pixels within `--buffer-m` of a held-out block are excluded.

The report includes overall and per-fold accuracy, macro F1, per-class metrics, and a confusion matrix. It adds a 95% interval when there are at least five held-out units. These figures describe the labeled pixels, not area-weighted accuracy across a whole map.

To score a finished map against separate reference labels:

```bash
landfuse evaluate \
  --prediction /data/results/landcover_lidar.tif \
  --reference /data/heldout_reference_labels.tif \
  --output /data/results/heldout_evaluation.json
```

`pixel_accuracy` counts unclassified reference pixels as errors. `classified_pixel_accuracy` covers classified pixels only, and `coverage` shows how much of the reference was classified. LandFuse can't verify that your reference labels are independent, and it doesn't apply an approval threshold.

## Limitations

- Model and output sidecars record SHA-256 hashes and the source raster's resolution and CRS. Hashes show which files were used, not whether the results are accurate.
- Band order isn't detected automatically. Confirm R, G, B, NIR with the provider, and check values before relying on NDVI.
- NAIP is not calibrated reflectance, so a model may not transfer between flight lines or years.
- The CHM is a first-pass difference between DSM and DTM surfaces. Ground gaps up to 25 m are interpolated, and the sidecar records how many pixels were filled. Check point density, vertical datums, vegetation season, and acquisition dates.
- DEM slope is limited by the DEM's native resolution, even when output pixels are 0.30 m.
- Models use Python serialization. The CLI requires `--trust-local-model` and a matching SHA-256 sidecar, but a matching hash doesn't make a malicious model safe. Models must be loaded with the scikit-learn version that trained them.
- Outputs are tiled GeoTIFFs, not validated COGs. Use `gdal_translate -of COG` if you need a COG.

## Tests

`pytest -q` uses synthetic imagery, terrain, and labels. It covers alignment, sampling, holdout logic, provenance, model tampering, output CRS and shape, coverage, and failure handling. It doesn't measure field accuracy or run live NAIP downloads.

The PDAL integration test runs only when `pdal`, `laspy`, and `pyproj` are installed:

```bash
python -m pip install laspy pyproj
pytest -q tests/test_pdal_integration.py
```

## Project layout

- `src/landfuse/`: source code
- `tests/`: regression tests
- `examples/`: model card template

## License

LandFuse is licensed under Apache-2.0. The license covers this code, not upstream aerial imagery or LiDAR data.
