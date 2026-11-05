# Contributing

1. Open an issue that includes the sample CRS, resolution, band order, point classification, and expected behavior. Don't post private data or credentials.
2. Add regression tests that use small synthetic rasters or openly redistributable data.
3. Run `python -m pytest -q` and note any live tests you skipped. Don't present results from synthetic data as real-world accuracy.
4. Describe how your change affects alignment, model reproducibility, and memory use.
5. Keep APIs usable by any GIS client, and don't include private application code or data.

Contributions are licensed under Apache-2.0.
