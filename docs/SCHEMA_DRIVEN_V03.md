# User-defined classification schemas (LandFuse v0.3.0)

LandFuse can train, predict, cross-validate and evaluate with a user-selected
**JSON or YAML land-cover taxonomy**, rather than the six built-in classes.
The default **remains** the six-class `v1` schema for existing users.
`v2` (the seven-class dryland taxonomy) and custom files are opt-in.

Example (see `examples/landcover_example.yaml` and `examples/forestry_taxonomy.json`):

```yaml
id: sample.landcover
version: 1.0.0
classes:
  1: water
  2: bare_soil
  9: vegetation
  1000: structure
```

- Code `0` is reserved for NoData/unclassified, never trained.
- Class codes can be noncontiguous integers from **1 to 65535**. The output
  GeoTIFF uses `uint8` when possible, otherwise `uint16`.
- Each schema must have a stable `id` and `version`. Class names are unique
  lowercase identifiers. The list-of-objects form also supports optional
  `description`, `color`, and descriptive `parent` metadata.
- Parent/child relations are metadata, **not** a hierarchical inference engine.
- Canonical schema bytes and SHA-256 are pinned to model and raster metadata.
  Model/schema mismatches and unknown label codes stop execution.
- An arbitrary schema **does not** make an arbitrary category detectable from
  imagery. Collect and independently verify training labels for each class.

```bash
landfuse schema --class-schema examples/landcover_example.yaml
landfuse train --imagery /data/naip.tif --labels /data/labels.tif \
  --class-schema examples/landcover_example.yaml \
  --output-model /data/model.joblib
landfuse classify --imagery /data/naip.tif --model /data/model.joblib \
  --trust-local-model --output /data/landcover.tif
landfuse evaluate --prediction /data/landcover.tif \
  --reference /data/independent_holdout.tif \
  --class-schema examples/landcover_example.yaml \
  --output /data/evaluation.json
```

`classify` reads the trained model's schema automatically. Supplying
`--class-schema` during classification explicitly checks for an exact match.
Existing legacy v1/v2 models can still be read when their model hashes,
feature contracts and class mappings match.

**Migration caution:** Pixel code meanings may differ between taxonomies. Existing labels and predictions must not be silently reinterpreted under a new schema. Downstream consumers must support the schema-aware class mappings before using new model outputs.

**Reliability:** This is a library implementation, not a pretrained model,
real-world classification validation, or approval for site-safety decisions.
Do not conflate dry-wash drainage hazards with visible land cover.
