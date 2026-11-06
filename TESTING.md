# LandFuse v0.3.0 schema-driven test evidence

This is a local development candidate, not a verified PlaceRules model.

Run `python -m pip install -e '.[test]'` followed by `pytest -q`.

Scope:
- v0.2.0 regression suite, imagery-only and DEM/CHM modes, scene validation, legacy v1/v2 taxonomy behavior;
- arbitrary JSON and YAML classes, sparse class codes 1–65535, uint8/uint16 output;
- full train → model → classification → GeoTIFF → independent evaluation → cross-validation paths;
- model/sidecar/raster taxonomy integrity, out-of-contract labels, and schema changes;
- JSON/YAML duplicate-key rejection, bounded size, class code collision, cycles and parents;
- CLI, old serialized model compatibility, and explicit deserialization trust guard.

Some tests use synthetic imagery and labels. Passing tests does NOT establish
real-world classification accuracy or independent validation.
