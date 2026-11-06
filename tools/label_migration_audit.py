#!/usr/bin/env python3
"""Read-only audit of v1 reviewed GeoJSON before creating v2 labels.

Does NOT assign v2 labels or modify sources. Proposed migration eligibility
is only a guide; original imagery needs expert review.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def audit(path: Path) -> dict:
    source = json.loads(path.read_text())
    if source.get("type") != "FeatureCollection":
        raise ValueError("Expected GeoJSON FeatureCollection")
    count = Counter()
    scenes = set()
    for feature in source["features"]:
        code = feature.get("properties", {}).get("class_code")
        if not isinstance(code, int) or code not in range(1, 7):
            raise ValueError(f"Invalid legacy class code: {code!r}")
        count[code] += 1
        scenes.add(feature.get("properties", {}).get("scene_id"))
    return {
        "source": str(path), "features": sum(count.values()),
        "distinct_scenes": len(scenes),
        "safe_to_propose_after_review": {"water_v1_1": count[1],
                                          "vegetation_v1_3_4": count[3]+count[4]},
        "requires_explicit_new_v2_classification": {
            "rock_or_earth_v1_2": count[2],
            "structure_or_other_built_v1_5": count[5],
            "paved_or_unpaved_road_v1_6": count[6]},
        "warning": "Read-only coverage audit. Nothing converted, approved, or published."
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("reviewed_v1_geojson",type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.reviewed_v1_geojson),indent=2))


if __name__ == "__main__":
    main()
