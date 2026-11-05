"""Score a classified raster against reference labels."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .contract import CLASSES, LABEL_CLASSES, SCHEMA, LandFuseError, sha256, write_json


def class_metrics(reference: np.ndarray, predicted: np.ndarray) -> dict:
    """Return confusion and per-class metrics. A prediction of 0 counts as a miss."""
    names = [CLASSES[code] for code in LABEL_CLASSES]
    classified = predicted > 0
    matrix = confusion_matrix(reference[classified], predicted[classified], labels=LABEL_CLASSES)
    report = classification_report(reference, predicted, labels=LABEL_CLASSES, target_names=names,
                                   output_dict=True, zero_division=0)
    present = [name for name in names if report[name]["support"] > 0]
    return {
        "confusion_matrix_labels": names,
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_class": report,
        "macro_f1": float(np.mean([report[name]["f1-score"] for name in present])),
        "unclassified_by_reference_class": {
            CLASSES[code]: int(np.count_nonzero((reference == code) & ~classified))
            for code in LABEL_CLASSES
        },
    }


def evaluate(prediction: Path, reference: Path, output: Path, *, max_pixels: int = 20_000_000) -> dict:
    if output.exists():
        raise LandFuseError("Refusing to overwrite an existing evaluation report")
    with rasterio.open(prediction) as pred, rasterio.open(reference) as ref:
        if pred.count != 1 or ref.count != 1:
            raise LandFuseError("Prediction and reference must each have one band")
        if pred.crs is None or pred.crs != ref.crs or pred.shape != ref.shape or not pred.transform.almost_equals(ref.transform):
            raise LandFuseError("Independent reference labels must be exactly co-registered with classification")
        if pred.width * pred.height > max_pixels:
            raise LandFuseError("Evaluation scene exceeds pixel limit; split into bounded independent scenes")
        y_pred = pred.read(1).reshape(-1)
        y_ref = ref.read(1).reshape(-1)
    labeled = (y_ref > 0) & (y_ref <= 6)
    classified = labeled & (y_pred > 0) & (y_pred <= 6)
    if np.count_nonzero(classified) < 20:
        raise LandFuseError("At least 20 labeled pixels with valid classification are required")
    truth = y_ref[labeled].astype("int32")
    guess = np.where(classified[labeled], y_pred[labeled], 0).astype("int32")
    report = {
        "schema": "landfuse.evaluation.v1",
        "classifier_schema": SCHEMA,
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
    report.update(class_metrics(truth, guess))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report
