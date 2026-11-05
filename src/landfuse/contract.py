"""Shared class codes, feature names, errors, and input types."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

# Pixel values written to classified rasters.
CLASSES = {
    0: "nodata",
    1: "water",
    2: "bare_ground",
    3: "low_vegetation",
    4: "tall_vegetation",
    5: "built_surface",
    6: "road_or_path",
}
LABEL_CLASSES = [code for code in CLASSES if code != 0]
MODEL_FEATURES = ["red", "green", "blue", "nir", "ndvi", "ndwi", "brightness", "terrain"]
SCHEMA = "landfuse.classification.v1"
MODES = ("imagery", "dem", "chm")


def feature_names(mode: str, bands: int) -> list[str]:
    """Exact serialized model feature order; retain historical 4-band DEM/CHM layout."""
    if mode not in MODES:
        raise LandFuseError("Unsupported mode: expected imagery, dem or chm")
    if bands not in (3, 4) or (mode != "imagery" and bands != 4):
        raise LandFuseError("DEM/CHM require RGB-NIR; imagery-only supports RGB or RGB-NIR")
    names = ["red", "green", "blue"]
    if bands == 4:
        names += ["nir", "ndvi", "ndwi"]
    names += ["brightness"]
    if mode != "imagery":
        names += ["terrain"]
    return names



class LandFuseError(ValueError):
    """Raised for invalid input or a failed processing step."""


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class Scene:
    imagery: Path
    terrain: Path | None = None
    terrain_mode: str = "imagery"

    def check(self) -> None:
        if self.terrain_mode not in MODES:
            raise LandFuseError("terrain mode must be 'imagery', 'dem' or 'chm'")
        if self.terrain_mode == "imagery" and self.terrain is not None:
            raise LandFuseError("Imagery-only mode must not include a terrain raster")
        if self.terrain_mode in ("dem", "chm") and self.terrain is None:
            raise LandFuseError("DEM/CHM mode requires --terrain")
        for path in (self.imagery, self.terrain):
            if path is not None and not Path(path).is_file():
                raise LandFuseError(f"Missing input file: {path}")


@dataclass(frozen=True)
class LabeledScene:
    """A scene with training labels. Scenes in one group are held out together."""

    scene: Scene
    labels: Path
    scene_id: str
    group: str


def load_scene_list(path: str | Path) -> list[LabeledScene]:
    """Read a JSON scene list. Relative paths are resolved from the list's folder."""
    path = Path(path)
    if not path.is_file():
        raise LandFuseError(f"Missing scene list: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise LandFuseError(f"Scene list is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LandFuseError("Scene list must be a JSON object with terrain_mode and scenes")
    mode = payload.get("terrain_mode")
    entries = payload.get("scenes")
    if mode not in MODES:
        raise LandFuseError("Scene list terrain_mode must be 'imagery', 'dem' or 'chm'")
    if not isinstance(entries, list) or not entries:
        raise LandFuseError("Scene list must contain at least one scene")

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else path.parent / candidate

    scenes = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise LandFuseError("Each scene must be a JSON object")
        required = ("id", "imagery", "labels") if mode == "imagery" else ("id", "imagery", "terrain", "labels")
        missing = [key for key in required if not entry.get(key)]
        if mode == "imagery" and entry.get("terrain") is not None:
            raise LandFuseError("Imagery-only scene must not specify terrain")
        if missing:
            raise LandFuseError(f"Scene is missing: {', '.join(missing)}")
        scene_id = str(entry["id"])
        if scene_id in seen:
            raise LandFuseError(f"Duplicate scene id: {scene_id}")
        seen.add(scene_id)
        labels = resolve(entry["labels"])
        if not labels.is_file():
            raise LandFuseError(f"Missing input file: {labels}")
        scene = Scene(resolve(entry["imagery"]), resolve(entry["terrain"]) if mode != "imagery" else None, mode)
        scene.check()
        scenes.append(LabeledScene(scene, labels, scene_id, str(entry.get("group") or scene_id)))
    return scenes
