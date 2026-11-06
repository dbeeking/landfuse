# Changelog

## 0.3.0 (candidate) — Configurable taxonomies

- Accept user-defined JSON/YAML land-cover schemas, including custom class codes and metadata.
- Pin schema identity/hash in models and classified GeoTIFF metadata; reject schema mismatches.
- Keep six-class `v1` as the default; retain the opt-in seven-class `v2` taxonomy.
- Add schema inspection, custom taxonomy training/evaluation and regression tests.
- This release does not include pretrained models, reclassified Yavapai labels, or PlaceRules deployment.


## 0.2.0 — NAIP-only classification

- Added imagery-only classification using raw 8-bit NAIP RGB or RGB-NIR GeoTIFFs.
- DEM and LiDAR-derived CHM remain optional modes with compatible feature ordering.
- Mode/band-specific model checks prevent incompatible model inference.
- Updated scene lists, input inspection, training, spatial cross-validation, and sidecars.
- Explicitly reports that imagery-only mode cannot measure vegetation heights.
- Does not include a pretrained classifier or claim independently validated accuracy.


## 0.1.0

### Added

- `landfuse train --scenes` trains one model from a JSON list of labeled scenes.
- `landfuse cross-validate` estimates accuracy by holding out scene groups or map blocks, with a buffer around held-out blocks.
- `classify` sidecars now report coverage, nodata pixels, and pixels and area by class.
- Model manifests record the scikit-learn version, and `classify` stops if the installed version differs.
- `--version`, `--seed`, `--trees`, and `--per-class-limit` options.

### Changed

- `evaluate` now counts unclassified reference pixels as errors in `pixel_accuracy` and per-class recall. Accuracy among classified pixels is reported separately as `classified_pixel_accuracy`, alongside `coverage`.
- `ignored_unlabeled_or_nodata` is now `ignored_unlabeled_pixels`.
- `inspect` reports `terrain_coverage_fraction` instead of `overlap_valid`.
- Model manifests list per-scene hashes under `scenes`.
- `fetch-naip` requests nearest-neighbor resampling.
- The minimum setuptools version is now 77.

### Fixed

- Training samples were taken from the first tiles visited. Each class is now spread across every tile that contains it.
- Pixels with no ground return dropped out of the CHM, including trees. Ground gaps up to 25 m are now interpolated, and the sidecar records how many pixels were filled.
- Slope was wrong beside DEM nodata and changed at tile edges.
- Pixels that are zero in all four bands were classified as land.
- A failed `classify` left a partial GeoTIFF that blocked reruns.
- The EPT allowlist accepted any bucket on `s3-us-west-2.amazonaws.com`.
- Download and file errors now show the underlying cause.

