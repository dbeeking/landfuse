"""Estimate accuracy by holding out whole scene groups or map blocks."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .contract import CLASSES, SCHEMA, LabeledScene, LandFuseError, sha256, write_json
from .evaluate import class_metrics
from .model import collect_samples, fit_forest


def block_ids(x: np.ndarray, y: np.ndarray, block_m: float) -> np.ndarray:
    """Return integer block columns and rows for map coordinates."""
    return np.stack((np.floor(x / block_m), np.floor(y / block_m)), axis=1).astype("int64")


def near_blocks(x: np.ndarray, y: np.ndarray, blocks: np.ndarray, block_m: float, buffer_m: float) -> np.ndarray:
    """Flag points within buffer_m of any listed block, using a square buffer."""
    near = np.zeros(x.shape, dtype=bool)
    if not len(blocks):
        return near
    held_out = {(int(col), int(row)) for col, row in blocks}
    for dx in (-buffer_m, 0.0, buffer_m):
        for dy in (-buffer_m, 0.0, buffer_m):
            cells = block_ids(x + dx, y + dy, block_m)
            near |= np.fromiter(((int(col), int(row)) in held_out for col, row in cells), bool, len(cells))
    return near


def bootstrap_interval(correct: np.ndarray, total: np.ndarray, *, seed: int = 42,
                       rounds: int = 2000) -> list[float] | None:
    """Return a 95% interval by resampling held-out units, or None with too few units."""
    usable = total > 0
    correct = correct[usable]
    total = total[usable]
    if correct.size < 5:
        return None
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, correct.size, size=(rounds, correct.size))
    rates = correct[picks].sum(axis=1) / total[picks].sum(axis=1)
    return [float(value) for value in np.percentile(rates, [2.5, 97.5])]


def cross_validate(scenes: list[LabeledScene], output: Path, *, by: str = "auto", folds: int = 5,
                   block_m: float = 250.0, buffer_m: float = 30.0, seed: int = 42,
                   per_class_limit: int = 20000, tile: int = 512, trees: int = 80) -> dict:
    """Train on some areas, score on the held-out areas, and write a JSON report."""
    if output.exists():
        raise LandFuseError("Refusing to overwrite an existing cross-validation report")
    if by not in ("auto", "group", "block"):
        raise LandFuseError("Holdout method must be auto, group, or block")
    if folds < 2:
        raise LandFuseError("Cross-validation needs at least 2 folds")
    if block_m <= 0 or buffer_m < 0 or buffer_m > block_m:
        raise LandFuseError("block_m must be positive and buffer_m must be between 0 and block_m")

    samples = collect_samples(scenes, seed=seed, per_class_limit=per_class_limit, tile=tile)
    groups = np.array([item.group for item in scenes])[samples.scene_index]
    distinct_groups = sorted(set(groups.tolist()))
    method = by if by != "auto" else ("group" if len(distinct_groups) >= 2 else "block")
    rng = np.random.default_rng(seed)

    if method == "group":
        if len(distinct_groups) < 2:
            raise LandFuseError("Group holdout needs labeled samples in at least two groups")
        units = np.searchsorted(distinct_groups, groups)
        unit_names = distinct_groups
        fold_of_unit = np.arange(len(distinct_groups))
        blocks = None
    else:
        if len(set(samples.crs)) != 1:
            raise LandFuseError("Block holdout needs every scene in the same CRS")
        cells = block_ids(samples.x, samples.y, block_m)
        blocks, units = np.unique(cells, axis=0, return_inverse=True)
        units = units.reshape(-1)
        unit_names = [f"{col},{row}" for col, row in blocks.tolist()]
        if len(blocks) < 2:
            raise LandFuseError("Labeled pixels fall in one block; use a smaller block size or more scenes")
        fold_of_unit = rng.permutation(len(blocks)) % min(folds, len(blocks))

    predicted = np.zeros(samples.labels.shape, dtype="uint8")
    fold_records = []
    excluded = 0
    for fold in range(int(fold_of_unit.max()) + 1):
        held_out_units = np.flatnonzero(fold_of_unit == fold)
        test = np.isin(units, held_out_units)
        train = ~test
        if method == "block" and buffer_m > 0:
            near = near_blocks(samples.x, samples.y, blocks[held_out_units], block_m, buffer_m) & train
            excluded += int(np.count_nonzero(near))
            train &= ~near
        record = {
            "fold": fold,
            "held_out": [unit_names[int(unit)] for unit in held_out_units],
            "train_samples": int(np.count_nonzero(train)),
            "test_samples": int(np.count_nonzero(test)),
        }
        train_classes = np.unique(samples.labels[train])
        if train_classes.size < 2 or not test.any():
            record["skipped"] = "Fold needs a test set and at least two training classes"
            fold_records.append(record)
            continue
        forest = fit_forest(samples.features[train], samples.labels[train], seed=seed, trees=trees)
        predicted[test] = forest.predict(samples.features[test]).astype("uint8")
        record["accuracy"] = float(np.mean(predicted[test] == samples.labels[test]))
        record["classes_missing_from_training"] = [
            CLASSES[int(code)] for code in np.setdiff1d(samples.labels[test], train_classes)
        ]
        fold_records.append(record)

    scored = predicted > 0
    if not scored.any():
        raise LandFuseError("No fold could be scored; add labeled classes to more areas")
    truth = samples.labels[scored].astype("int32")
    guess = predicted[scored].astype("int32")
    unit_count = len(unit_names)
    correct = np.bincount(units[scored], weights=(truth == guess).astype("float64"), minlength=unit_count)
    total = np.bincount(units[scored], minlength=unit_count).astype("float64")
    accuracies = [record["accuracy"] for record in fold_records if "accuracy" in record]
    report = {
        "schema": "landfuse.crossval.v1",
        "classifier_schema": SCHEMA,
        "holdout_method": method,
        "terrain_mode": scenes[0].scene.terrain_mode,
        "block_m": block_m if method == "block" else None,
        "buffer_m": buffer_m if method == "block" else None,
        "scenes": [{
            "id": item.scene_id,
            "group": item.group,
            "imagery_sha256": sha256(item.scene.imagery),
            "terrain_sha256": sha256(item.scene.terrain),
            "label_sha256": sha256(item.labels),
        } for item in scenes],
        "samples": int(samples.labels.size),
        "scored_samples": int(scored.sum()),
        "held_out_units": unit_count,
        "buffer_excluded_training_samples": excluded,
        "overall_accuracy": float(np.mean(truth == guess)),
        "overall_accuracy_ci95": bootstrap_interval(correct, total, seed=seed),
        "fold_accuracy_mean": float(np.mean(accuracies)),
        "fold_accuracy_min": float(np.min(accuracies)),
        "fold_accuracy_max": float(np.max(accuracies)),
        "folds": fold_records,
        "warning": "Sample-based estimate from labeled pixels, not a map-wide area-weighted accuracy",
    }
    report.update(class_metrics(truth, guess))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report
