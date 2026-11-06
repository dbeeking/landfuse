"""Train and run the Random Forest land-cover classifier."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os

import joblib
import numpy as np
import rasterio
import sklearn
from sklearn.ensemble import RandomForestClassifier

from .contract import (
    CLASSES,
    LABEL_CLASSES,
    MODEL_FEATURES,
    feature_names,
    SCHEMA,
    resolve_class_schema,
    LabeledScene,
    LandFuseError,
    Scene,
    sha256,
    write_json,
)
from .raster_ops import get_features, open_scene, windows
from .schema import load_class_schema, schema_from_canonical, schema_from_raster_tags


@dataclass
class Samples:
    """Labeled training pixels with their map positions and source scene."""

    features: np.ndarray
    labels: np.ndarray
    x: np.ndarray
    y: np.ndarray
    scene_index: np.ndarray
    class_tiles: dict[str, int]
    crs: list[str]
    alignment: list[dict]


def _assert_training_label_alignment(naip, labels):
    if labels.count != 1 or labels.crs != naip.crs or labels.width != naip.width or labels.height != naip.height:
        raise LandFuseError("Training labels must be single-band, same CRS and dimensions as NAIP")
    if not labels.transform.almost_equals(naip.transform):
        raise LandFuseError("Training labels must be exactly aligned with aerial imagery")


def _split_limit(counts: list[int], limit: int) -> list[int]:
    """Split a class sample limit across tiles, capped by what each tile holds."""
    quotas = [0] * len(counts)
    remaining = limit
    open_tiles = [i for i, count in enumerate(counts) if count]
    while remaining and open_tiles:
        share = max(remaining // len(open_tiles), 1)
        still_open = []
        for i in open_tiles:
            take = min(share, counts[i] - quotas[i], remaining)
            quotas[i] += take
            remaining -= take
            if quotas[i] < counts[i]:
                still_open.append(i)
        open_tiles = still_open
    return quotas


def collect_samples(scenes: list[LabeledScene], *, seed: int = 42, per_class_limit: int = 20000,
                    tile: int = 512, class_schema: str = "v1") -> Samples:
    """Sample each class evenly across every tile of every scene that contains it."""
    if per_class_limit < 20 or per_class_limit > 100000:
        raise LandFuseError("per_class_limit must be between 20 and 100000")
    if not scenes:
        raise LandFuseError("At least one labeled scene is required")
    if len({item.scene.terrain_mode for item in scenes}) != 1:
        raise LandFuseError("All training scenes must use the same input mode")
    taxonomy = load_class_schema(class_schema)
    schema_id, classes = taxonomy.identifier, taxonomy.classes
    label_codes = list(taxonomy.trainable_codes)
    rng = np.random.default_rng(seed)
    tiles = []
    tile_counts = {code: [] for code in label_codes}
    crs = []
    alignment = []
    for index, item in enumerate(scenes):
        with open_scene(item.scene) as (naip, _, metadata), rasterio.open(item.labels) as labels:
            _assert_training_label_alignment(naip, labels)
            if labels.tags().get("LANDFUSE_SCHEMA"):
                tagged_schema = schema_from_raster_tags(labels.tags())
                if tagged_schema.digest != taxonomy.digest:
                    raise LandFuseError("Training label raster class schema mismatch")
            if labels.nodata is not None and int(labels.nodata) in label_codes:
                raise LandFuseError("Training label nodata value conflicts with a class code")
            crs.append(naip.crs.to_string())
            alignment.append(metadata)
            if len({m["imagery_bands"] for m in alignment}) != 1:
                raise LandFuseError("All training scenes must have identical imagery band counts")
            for window in windows(naip, tile):
                values = labels.read(1, window=window)
                valid_codes = set(classes) | ({int(labels.nodata)} if labels.nodata is not None else set())
                if np.any(~np.isin(values, list(valid_codes))):
                    raise LandFuseError(f"Training labels contain codes not defined by {schema_id}")
                tiles.append((index, window))
                for code in label_codes:
                    tile_counts[code].append(int(np.count_nonzero(values == code)))

    # Random order keeps the remainder of each class limit from favoring one area.
    order = rng.permutation(len(tiles))
    tiles = [tiles[i] for i in order]
    quotas = {code: _split_limit([tile_counts[code][i] for i in order], per_class_limit)
              for code in label_codes}

    features, targets, xs, ys, sources = [], [], [], [], []
    class_tiles = {}
    for index, item in enumerate(scenes):
        with open_scene(item.scene) as (naip, terrain, _), rasterio.open(item.labels) as labels:
            for position, (tile_scene, window) in enumerate(tiles):
                if tile_scene != index or not any(quotas[code][position] for code in label_codes):
                    continue
                cube, good = get_features(naip, terrain, item.scene.terrain_mode, window)
                values = labels.read(1, window=window)
                transform = naip.window_transform(window)
                flat = cube.reshape(-1, cube.shape[-1])
                for code in label_codes:
                    ids = np.flatnonzero((values == code) & good)
                    take = min(quotas[code][position], ids.size)
                    if not take:
                        continue
                    chosen = rng.choice(ids, size=take, replace=False)
                    rows, cols = np.unravel_index(chosen, values.shape)
                    map_x = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
                    map_y = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
                    features.append(flat[chosen])
                    targets.append(np.full(take, code, dtype="uint16" if max(classes)>255 else "uint8"))
                    xs.append(np.asarray(map_x, dtype="float64"))
                    ys.append(np.asarray(map_y, dtype="float64"))
                    sources.append(np.full(take, index, dtype="int32"))
                    class_tiles[classes[code]] = class_tiles.get(classes[code], 0) + 1
    if len(class_tiles) < 2:
        raise LandFuseError("At least two distinct labeled classes with valid input pixels are needed")
    return Samples(np.vstack(features), np.concatenate(targets), np.concatenate(xs), np.concatenate(ys),
                   np.concatenate(sources), class_tiles, crs, alignment)


def fit_forest(features: np.ndarray, labels: np.ndarray, *, seed: int = 42, trees: int = 80):
    forest = RandomForestClassifier(n_estimators=trees, min_samples_leaf=3, max_features="sqrt",
                                    class_weight="balanced_subsample", random_state=seed, n_jobs=2)
    return forest.fit(features, labels)


def train_scenes(scenes: list[LabeledScene], output_model: Path, *, seed: int = 42,
                 per_class_limit: int = 20000, tile: int = 512, trees: int = 80,
                 class_schema: str = "v1") -> dict:
    """Train one model from one or more labeled scenes."""
    if output_model.exists():
        raise LandFuseError("Refusing to overwrite an existing model file")
    taxonomy = load_class_schema(class_schema)
    schema_id, classes = taxonomy.identifier, taxonomy.classes
    samples = collect_samples(scenes, seed=seed, per_class_limit=per_class_limit, tile=tile,
                              class_schema=class_schema)
    forest = fit_forest(samples.features, samples.labels, seed=seed, trees=trees)
    terrain_mode = scenes[0].scene.terrain_mode
    bands = int(samples.alignment[0]["imagery_bands"])
    names = feature_names(terrain_mode, bands)
    if any(item["imagery_bands"] != bands for item in samples.alignment):
        raise LandFuseError("Training scenes must use the same imagery band count")
    output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": forest, "terrain_mode": terrain_mode, "schema": schema_id,
                 "schema_sha256": taxonomy.digest, "schema_document": taxonomy.document,
                 "imagery_bands": bands, "features": names}, output_model, compress=3)
    codes, counts = np.unique(samples.labels, return_counts=True)
    scene_records = []
    for index, item in enumerate(scenes):
        in_scene = samples.scene_index == index
        scene_records.append({
            "id": item.scene_id,
            "group": item.group,
            "imagery_sha256": sha256(item.scene.imagery),
            "terrain_sha256": sha256(item.scene.terrain) if item.scene.terrain else None,
            "label_sha256": sha256(item.labels),
            "samples": int(np.count_nonzero(in_scene)),
            "source_alignment": samples.alignment[index],
        })
    manifest = {
        "schema": schema_id,
        "schema_sha256": taxonomy.digest,
        "schema_document": taxonomy.document,
        "trained_with": "user_supplied_labels",
        "validated_accuracy": None,
        "note": "Training fit is not accuracy. Run cross-validate with spatially separate scenes or blocks",
        "train_class_counts": {classes[int(code)]: int(count) for code, count in zip(codes, counts)},
        "train_class_tiles": samples.class_tiles,
        "feature_names": names,
        "imagery_bands": bands,
        "terrain_mode": terrain_mode,
        "scenes": scene_records,
        "model_sha256": sha256(output_model),
        "sklearn_version": sklearn.__version__,
        "classes": classes,
    }
    write_json(str(output_model) + ".json", manifest)
    return manifest


def train(scene: Scene, labels_path: Path, output_model: Path, *, seed: int = 42,
          per_class_limit: int = 20000, tile: int = 512, trees: int = 80,
          class_schema: str = "v1") -> dict:
    """Train a model from a single labeled scene."""
    item = LabeledScene(scene, labels_path, "scene", "scene")
    return train_scenes([item], output_model, seed=seed, per_class_limit=per_class_limit,
                        tile=tile, trees=trees, class_schema=class_schema)


def infer(scene: Scene, model_path: Path, output_path: Path, *, tile: int = 512,
          trust_local_model: bool = False, class_schema: str | None = None) -> dict:
    if not trust_local_model:
        raise LandFuseError("Models use Python serialization: load only trusted local models, with --trust-local-model")
    manifest_path = Path(str(model_path) + ".json")
    if not manifest_path.is_file():
        raise LandFuseError("Model sidecar manifest is missing")
    meta = json.loads(manifest_path.read_text())
    if meta.get("model_sha256") != sha256(model_path):
        raise LandFuseError("Model integrity check failed: SHA256 mismatch")
    if "schema_document" in meta:
        taxonomy = schema_from_canonical(meta["schema_document"])
        if meta.get("schema") != taxonomy.identifier or meta.get("schema_sha256") != taxonomy.digest:
            raise LandFuseError("Model sidecar classification schema/hash mismatch")
    else:
        # Legacy models are allowed ONLY for the two fixed built-in schemas.
        taxonomy = load_class_schema(meta.get("schema", ""))
        if not taxonomy.built_in:
            raise LandFuseError("Custom models require embedded immutable taxonomy metadata")
    schema_id, classes = taxonomy.identifier, taxonomy.classes
    if class_schema is not None and load_class_schema(class_schema).digest != taxonomy.digest:
        raise LandFuseError("Requested class schema does not match trained model taxonomy")
    if meta.get("terrain_mode") != scene.terrain_mode:
        raise LandFuseError("Model terrain mode incompatible with scene")
    # A sidecar can never silently reinterpret the integer class codes.
    manifest_classes = {int(code): name for code, name in meta.get("classes", {}).items()}
    if (manifest_classes and manifest_classes != classes) or (not manifest_classes and schema_id != SCHEMA):
        raise LandFuseError("Model sidecar class taxonomy mismatch")
    trained_with = meta.get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        raise LandFuseError(f"Model was trained with scikit-learn {trained_with}, but {sklearn.__version__} is installed")
    # Verify the scene and feature contract BEFORE loading Python-serialized model data.
    with open_scene(scene) as (_, _, source_checks):
        bands = int(source_checks["imagery_bands"])
        names = feature_names(scene.terrain_mode, bands)
    if meta.get("feature_names") != names or meta.get("imagery_bands", 4) != bands:
        raise LandFuseError("Model imagery bands or feature schema incompatible with scene")
    bundle = joblib.load(model_path)
    if (bundle.get("features") != names or bundle.get("schema") != schema_id
            or bundle.get("terrain_mode") != scene.terrain_mode
            or bundle.get("imagery_bands", 4) != bands):
        raise LandFuseError("Serialized model feature/schema contract mismatch")
    if "schema_document" in meta:
        if (bundle.get("schema_sha256") != taxonomy.digest
                or bundle.get("schema_document") != taxonomy.document):
            raise LandFuseError("Serialized model classification schema/hash mismatch")
    else:
        if bundle.get("schema_sha256") or bundle.get("schema_document"):
            raise LandFuseError("Legacy model classification metadata mismatch")
    if output_path.exists():
        raise LandFuseError("Refusing to overwrite an existing classification")
    pixel_dtype = "uint16" if max(classes) > 255 else "uint8"
    class_pixels = np.zeros(max(classes) + 1, dtype="int64")
    with open_scene(scene) as (naip, terrain, source):
        profile = naip.profile.copy()
        profile.update(driver="GTiff", height=naip.height, width=naip.width, count=1,
                       dtype=pixel_dtype, nodata=0, compress="deflate", predictor=2,
                       tiled=True, blockxsize=256, blockysize=256, interleave="band")
        pixel_area = float(abs(naip.res[0] * naip.res[1]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial = output_path.with_name(output_path.name + ".partial")
        try:
            with rasterio.open(partial, "w", **profile) as dst:
                dst.update_tags(LANDFUSE_SCHEMA=schema_id, LANDFUSE_CLASS_NAMES=json.dumps(classes),
                                LANDFUSE_SCHEMA_SHA256=taxonomy.digest,
                                LANDFUSE_SCHEMA_JSON=json.dumps(taxonomy.document, sort_keys=True, separators=(",", ":")),
                                LANDFUSE_MODE=scene.terrain_mode, LANDFUSE_QUALITY="UNVALIDATED")
                for window in windows(naip, tile):
                    cube, valid = get_features(naip, terrain, scene.terrain_mode, window)
                    result = np.zeros(valid.shape, dtype=pixel_dtype)
                    if np.any(valid):
                        predicted = bundle["model"].predict(cube[valid])
                        if not np.all(np.isin(predicted, taxonomy.trainable_codes)):
                            raise LandFuseError("Model predicted codes outside declared class schema")
                        result[valid] = predicted.astype(pixel_dtype)
                    dst.write(result, 1, window=window)
                    class_pixels += np.bincount(result.ravel(), minlength=max(classes) + 1)[:max(classes) + 1]
            os.replace(partial, output_path)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
    total = int(class_pixels.sum())
    classified = total - int(class_pixels[0])
    summary = {
        "schema": schema_id,
        "schema_sha256": taxonomy.digest,
        "schema_document": taxonomy.document,
        "imagery_sha256": sha256(scene.imagery),
        "terrain_sha256": sha256(scene.terrain) if scene.terrain else None,
        "model_sha256": sha256(model_path),
        "classified_raster_sha256": sha256(output_path),
        "terrain_mode": scene.terrain_mode,
        "imagery_bands": bands,
        "feature_names": names,
        "height_estimation_available": scene.terrain_mode == "chm",
        "classes": classes,
        "source_alignment": source,
        "coverage": {
            "total_pixels": total,
            "classified_pixels": classified,
            "nodata_pixels": int(class_pixels[0]),
            "coverage_fraction": classified / total if total else 0.0,
            "class_pixels": {classes[code]: int(class_pixels[code]) for code in classes if code != 0},
            "class_area_m2": {classes[code]: float(class_pixels[code] * pixel_area) for code in classes if code != 0},
        },
        "validation": "NOT_INDEPENDENTLY_VALIDATED",
        "warning": ("Imagery-only classification does not measure canopy/object heights. "
                    if scene.terrain_mode == "imagery" else
                    "Bare-earth DEM does not measure canopy height. " if scene.terrain_mode == "dem" else "") +
                   "Not a survey, flood assessment, or verified building-footprint map",
    }
    write_json(str(output_path) + ".json", summary)
    return summary
