"""Score a classified raster against reference labels."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .contract import CLASSES, LABEL_CLASSES, SCHEMA, resolve_class_schema, LandFuseError, sha256, write_json
from .schema import load_class_schema, schema_from_raster_tags


def class_metrics(reference: np.ndarray, predicted: np.ndarray, *, classes: dict[int, str] | None = None) -> dict:
    """Return confusion and per-class metrics. A prediction of 0 counts as a miss."""
    classes = classes or CLASSES
    label_codes = [code for code in classes if code != 0]
    names = [classes[code] for code in label_codes]
    classified = predicted > 0
    matrix = confusion_matrix(reference[classified], predicted[classified], labels=label_codes)
    report = classification_report(reference, predicted, labels=label_codes, target_names=names,
                                   output_dict=True, zero_division=0)
    present = [name for name in names if report[name]["support"] > 0]
    return {
        "confusion_matrix_labels": names,
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_class": report,
        "macro_f1": float(np.mean([report[name]["f1-score"] for name in present])),
        "unclassified_by_reference_class": {
            classes[code]: int(np.count_nonzero((reference == code) & ~classified))
            for code in label_codes
        },
    }


def evaluate(prediction: Path, reference: Path, output: Path, *, max_pixels: int = 20_000_000,
             class_schema: str | None = None) -> dict:
    if output.exists():
        raise LandFuseError("Refusing to overwrite an existing evaluation report")
    with rasterio.open(prediction) as pred, rasterio.open(reference) as ref:
        if pred.count != 1 or ref.count != 1:
            raise LandFuseError("Prediction and reference must each have one band")
        if pred.crs is None or pred.crs != ref.crs or pred.shape != ref.shape or not pred.transform.almost_equals(ref.transform):
            raise LandFuseError("Independent reference labels must be exactly co-registered with classification")
        if pred.width * pred.height > max_pixels:
            raise LandFuseError("Evaluation scene exceeds pixel limit; split into bounded independent scenes")
        taxonomy = schema_from_raster_tags(pred.tags()) if pred.tags().get("LANDFUSE_SCHEMA") else load_class_schema("v1")
        schema_id, classes = taxonomy.identifier, taxonomy.classes
        if class_schema is not None and load_class_schema(class_schema).digest != taxonomy.digest:
            raise LandFuseError("Evaluation requested class schema does not match prediction")
        if ref.tags().get("LANDFUSE_SCHEMA"):
            reference_taxonomy = schema_from_raster_tags(ref.tags())
            if reference_taxonomy.digest != taxonomy.digest:
                raise LandFuseError("Independent reference class schema mismatch")
        elif not taxonomy.built_in and class_schema is None:
            raise LandFuseError("Untagged custom reference needs explicit --class-schema agreement")
        y_pred = pred.read(1).reshape(-1)
        y_ref = ref.read(1).reshape(-1)
    valid = [code for code in classes if code != 0]
    if np.any((y_ref != 0) & ~np.isin(y_ref, valid)):
        raise LandFuseError("Reference label codes do not match prediction class schema")
    if np.any((y_pred != 0) & ~np.isin(y_pred, valid)):
        raise LandFuseError("Prediction codes do not match its embedded class schema")
    labeled = np.isin(y_ref, valid)
    classified = labeled & np.isin(y_pred, valid)
    if np.count_nonzero(classified) < 20:
        raise LandFuseError("At least 20 labeled pixels with valid classification are required")
    truth = y_ref[labeled].astype("int32")
    guess = np.where(classified[labeled], y_pred[labeled], 0).astype("int32")
    report = {
        "schema": "landfuse.evaluation.v1",
        "classifier_schema": schema_id,
        "classifier_schema_sha256": taxonomy.digest,
        "classifier_schema_document": taxonomy.document,
        "prediction_sha256": sha256(prediction),
        "reference_sha256": sha256(reference),
        "evaluated_pixels": int(labeled.sum()),
        "classified_pixels": int(classified.sum()),
        "ignored_unlabeled_pixels": int(labeled.size - labeled.sum()),
        "coverage": float(classified.sum() / labeled.sum()),
        "pixel_accuracy": float(accuracy_score(truth, guess)),
        "classified_pixel_accuracy": float(np.mean(truth[guess > 0] == guess[guess > 0])),
        "reference_independence": "USER_ATTESTATION_REQUIRED",
        "warning": "Spatially independent labels and adequate samples per class required before operational use",
    }
    report.update(class_metrics(truth, guess, classes=classes))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report
