"""Stable, user-defined classification taxonomy contract.

LandFuse accepts any reviewed land-cover taxonomy fitting a byte-valued raster.
The taxonomy is data: models, predictions, and evaluations pin its canonical digest.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
from typing import Any

from .contract import CLASSES, CLASSES_V2, SCHEMA_NAMES, LandFuseError

_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9._:-]{0,127}$")
_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,127}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
_MAX_FILE_BYTES = 64 * 1024
_MAX_DOCUMENT_BYTES = 32 * 1024  # embedded unchanged in GeoTIFF tags



def _unique_json_mapping(pairs: list[tuple]) -> dict:
    out = {}
    for key, value in pairs:
        if key in out:
            raise LandFuseError(f"Duplicate JSON object key: {key!r}")
        out[key] = value
    return out


def _load_yaml_without_duplicate_keys(content: str) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise LandFuseError("Install PyYAML to read .yaml schema files") from exc

    class UniqueLoader(yaml.SafeLoader):
        pass

    def no_duplicates(loader, node):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=True)
            if not isinstance(key, (int, str, bool)):
                raise LandFuseError("YAML class keys must be scalar strings or integers")
            if key in mapping:
                raise LandFuseError(f"Duplicate YAML key: {key!r}")
            mapping[key] = loader.construct_object(value_node, deep=True)
        return mapping

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, no_duplicates)
    try:
        return yaml.load(content, Loader=UniqueLoader)
    except yaml.YAMLError as exc:
        raise LandFuseError(f"Invalid YAML schema: {exc}") from exc

def canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def document_sha256(document: dict) -> str:
    return hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ClassificationSchema:
    identifier: str
    version: str
    classes: dict[int, str]
    document: dict
    digest: str
    built_in: bool = False

    @property
    def trainable_codes(self) -> tuple[int, ...]:
        return tuple(code for code in sorted(self.classes) if code != 0)

    def as_dict(self) -> dict:
        return {
            "id": self.identifier, "version": self.version, "classes": self.classes,
            "sha256": self.digest, "document": self.document,
        }


def _positive_code(value: Any) -> int:
    if isinstance(value, bool) or (not isinstance(value, (int, str))) or not str(value).isdecimal():
        raise LandFuseError(f"Invalid class code {value!r}: expected an integer in [1,65535]")
    code = int(value)
    if code < 1 or code > 65535:
        raise LandFuseError(f"Class code {code} out of range [1,65535]; zero is reserved for nodata")
    return code


def _name(value: Any) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise LandFuseError(f"Invalid class name {value!r}: use lowercase letters, digits, underscores or hyphens")
    return value


def _string(value: Any, field: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise LandFuseError(f"{field} must be a nonempty string at most {limit} characters")
    return value.strip()


def parse_schema(document: dict, *, built_in: bool = False) -> ClassificationSchema:
    """Validate and normalize a JSON-compatible schema, rejecting silent ambiguities."""
    if not isinstance(document, dict):
        raise LandFuseError("Class schema must be a JSON/YAML mapping")
    allowed = {"id", "name", "version", "description", "classes", "format"}
    if set(document) - allowed:
        raise LandFuseError(f"Unexpected schema fields: {sorted(set(document) - allowed)}")
    if "format" in document and document["format"] != "landfuse.taxonomy.v1":
        raise LandFuseError("Unsupported taxonomy file format")
    ident = document.get("id", document.get("name"))
    if "id" in document and "name" in document and document["id"] != document["name"]:
        raise LandFuseError("Schema id and name disagree")
    if not isinstance(ident, str) or not _IDENTIFIER.fullmatch(ident):
        raise LandFuseError("Schema id/name must be a stable identifier (letters, digits, . _ : -)")
    version = document.get("version")
    if isinstance(version, (int, float)):
        version = str(version)
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise LandFuseError("Schema version must be a nonempty version string")
    description = document.get("description")
    if description is not None:
        description = _string(description, "Schema description", 1000)
    entries = document.get("classes")
    if isinstance(entries, dict):
        entries = [{"code": key, "name": val} if isinstance(val, str) else
                   {**val, "code": key} if isinstance(val, dict) and "code" not in val else
                   {"code": key, "name": val}
                   for key, val in entries.items()]
    if not isinstance(entries, list) or not entries:
        raise LandFuseError("Schema classes must be a nonempty list or code-to-name mapping")
    classes: dict[int, str] = {0: "nodata"}
    normalized: list[dict] = []
    names: set[str] = {"nodata"}
    for entry in entries:
        if not isinstance(entry, dict):
            raise LandFuseError("Each class must be a mapping with code and name")
        extras = set(entry) - {"code", "name", "description", "parent", "color"}
        if extras:
            raise LandFuseError(f"Unexpected class fields: {sorted(extras)}")
        raw_code = entry.get("code")
        if not isinstance(raw_code, bool) and str(raw_code) == "0":
            if entry.get("name") != "nodata" or len(entry) != 2:
                raise LandFuseError("Code 0 is reserved for nodata; it cannot be renamed or predicted")
            continue
        code = _positive_code(raw_code)
        name = _name(entry.get("name"))
        if code in classes or name in names:
            raise LandFuseError(f"Duplicate class code/name: {code}/{name}")
        names.add(name)
        classes[code] = name
        cleaned = {"code": code, "name": name}
        if "description" in entry:
            cleaned["description"] = _string(entry["description"], "Class description", 300)
        if "color" in entry:
            if not isinstance(entry["color"], str) or not _COLOR.fullmatch(entry["color"]):
                raise LandFuseError(f"Invalid class color for {name}: expected #RRGGBB")
            cleaned["color"] = entry["color"].lower()
        if "parent" in entry and entry["parent"] is not None:
            cleaned["parent"] = entry["parent"]
        normalized.append(cleaned)
    by_name = {name: code for code, name in classes.items()}
    parent_map: dict[int, int] = {}
    for entry in normalized:
        if "parent" not in entry:
            continue
        raw = entry["parent"]
        if isinstance(raw, bool):
            raise LandFuseError("Parent must be a class code or class name")
        if isinstance(raw, int) or (isinstance(raw, str) and raw.isdecimal()):
            parent = _positive_code(raw)
        elif isinstance(raw, str):
            if raw not in by_name:
                raise LandFuseError(f"Undefined parent class {raw!r}")
            parent = by_name[raw]
        else:
            raise LandFuseError("Parent must be a class code or class name")
        if parent not in classes or parent == entry["code"]:
            raise LandFuseError("Parent class missing or self-referential")
        entry["parent"] = parent
        parent_map[entry["code"]] = parent
    for code in parent_map:
        seen = {code}
        node = code
        while node in parent_map:
            node = parent_map[node]
            if node in seen:
                raise LandFuseError("Classification schema contains a parent cycle")
            seen.add(node)
    normalized.sort(key=lambda item: item["code"])
    canonical: dict = {"format": "landfuse.taxonomy.v1", "id": ident, "version": version, "classes": normalized}
    if description is not None:
        canonical["description"] = description
    if len(canonical_json(canonical).encode("utf-8")) > _MAX_DOCUMENT_BYTES:
        raise LandFuseError("Schema too large to embed in raster provenance")
    return ClassificationSchema(ident, version, classes, canonical, document_sha256(canonical), built_in)


def schema_from_canonical(document: dict) -> ClassificationSchema:
    result = parse_schema(document)
    if result.document != document:
        raise LandFuseError("Noncanonical schema document; refusing mismatched provenance")
    if result.identifier in SCHEMA_NAMES.values():
        legacy = load_class_schema(result.identifier)
        if legacy.digest != result.digest:
            raise LandFuseError("Built-in schema identifier cannot be redefined")
        return legacy
    return result


def load_class_schema(spec: str | Path | dict | ClassificationSchema = "v1") -> ClassificationSchema:
    if isinstance(spec, ClassificationSchema):
        return spec
    if isinstance(spec, dict):
        return parse_schema(spec)
    if not isinstance(spec, (str, Path)):
        raise LandFuseError("Class schema must be v1/v2, a JSON/YAML file, or a mapping")
    val = str(spec)
    for version, classes in (("v1", CLASSES), ("v2", CLASSES_V2)):
        ident = SCHEMA_NAMES[version]
        if val == version or val == ident:
            return parse_schema({"id": ident, "version": version,
                                 "classes": [{"code": k, "name": v} for k, v in classes.items() if k != 0]}, built_in=True)
    path = Path(val)
    if not path.is_file():
        raise LandFuseError(f"Unknown schema {val!r}; provide v1, v2, or an existing JSON/YAML file")
    if path.suffix.lower() not in (".json", ".yaml", ".yml"):
        raise LandFuseError("Schema file must end in .json, .yaml, or .yml")
    if path.stat().st_size > _MAX_FILE_BYTES:
        raise LandFuseError("Schema file exceeds 64 KiB limit")
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_mapping)
        else:
            payload = _load_yaml_without_duplicate_keys(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise LandFuseError(f"Cannot parse schema {path}: {exc}") from exc
    parsed = parse_schema(payload)
    if parsed.identifier in SCHEMA_NAMES.values():
        raise LandFuseError("Custom schema must not reuse a built-in v1/v2 identifier")
    return parsed


def schema_from_raster_tags(tags: dict) -> ClassificationSchema:
    ident = tags.get("LANDFUSE_SCHEMA", "")
    document = tags.get("LANDFUSE_SCHEMA_JSON")
    if document:
        try:
            parsed = schema_from_canonical(json.loads(document))
        except (json.JSONDecodeError, ValueError) as exc:
            raise LandFuseError("Invalid embedded classification schema") from exc
        if parsed.identifier != ident or tags.get("LANDFUSE_SCHEMA_SHA256") != parsed.digest:
            raise LandFuseError("Raster classification schema/hash mismatch")
        if tags.get("LANDFUSE_CLASS_NAMES"):
            try:
                names = {int(k): v for k, v in json.loads(tags["LANDFUSE_CLASS_NAMES"]).items()}
            except (ValueError, TypeError, AttributeError) as exc:
                raise LandFuseError("Invalid raster class names") from exc
            if names != parsed.classes:
                raise LandFuseError("Raster class names do not match embedded taxonomy")
        return parsed
    if not ident or ident not in SCHEMA_NAMES.values():
        raise LandFuseError("Raster lacks required custom classification schema provenance")
    # Backwards compatible with legacy v1/v2 rasters, which did not embed a digest.
    parsed = load_class_schema(ident)
    if tags.get("LANDFUSE_SCHEMA_SHA256") and tags["LANDFUSE_SCHEMA_SHA256"] != parsed.digest:
        raise LandFuseError("Raster built-in taxonomy digest mismatch")
    names = tags.get("LANDFUSE_CLASS_NAMES")
    if names:
        try:
            embedded_names = {int(code): name for code, name in json.loads(names).items()}
        except (ValueError, TypeError, AttributeError) as exc:
            raise LandFuseError("Invalid raster class names") from exc
        if embedded_names != parsed.classes:
            raise LandFuseError("Raster class names do not match declared schema")
    return parsed
