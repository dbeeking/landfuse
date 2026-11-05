"""LandFuse command-line interface."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

from . import __version__
from .contract import LandFuseError, Scene, load_scene_list
from .crossval import cross_validate
from .evaluate import evaluate
from .lidar import pdal_canopy_height
from .model import infer, train, train_scenes
from .naip import fetch_naip
from .raster_ops import open_scene


def add_scene(command, *, required=True):
    command.add_argument("--imagery", type=Path, required=required, help="Local 8-bit NAIP RGB or RGB-NIR GeoTIFF")
    command.add_argument("--terrain", type=Path, help="Optional bare-earth DEM or LiDAR-derived CHM GeoTIFF")
    command.add_argument("--terrain-mode", choices=["imagery", "dem", "chm"],
                         default=None, help="Default: imagery (NAIP only, no terrain)")


def add_training_options(command):
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--trees", type=int, default=80)
    command.add_argument("--per-class-limit", type=int, default=20000)


def parser():
    p = argparse.ArgumentParser(prog="landfuse", description="NAIP land-cover classification, optionally fused with DEM or LiDAR-derived CHM")
    p.add_argument("--version", action="version", version=f"landfuse {__version__}")
    commands = p.add_subparsers(dest="command", required=True)

    x = commands.add_parser("inspect", help="Check alignment and record source resolution")
    add_scene(x)

    x = commands.add_parser("train", help="Train a Random Forest from one labeled scene or a scene list")
    add_scene(x, required=False)
    x.add_argument("--labels", type=Path)
    x.add_argument("--scenes", type=Path, help="JSON scene list, used instead of the single-scene options")
    x.add_argument("--output-model", type=Path, required=True)
    add_training_options(x)

    x = commands.add_parser("classify", help="Write georeferenced classified GeoTIFF")
    add_scene(x)
    x.add_argument("--model", type=Path, required=True)
    x.add_argument("--output", type=Path, required=True)
    x.add_argument("--trust-local-model", action="store_true", help="Confirm user trusts this serialized Python model")

    x = commands.add_parser("cross-validate", help="Estimate accuracy by holding out scene groups or map blocks")
    x.add_argument("--scenes", type=Path, required=True, help="JSON scene list")
    x.add_argument("--output", type=Path, required=True)
    x.add_argument("--by", choices=["auto", "group", "block"], default="auto")
    x.add_argument("--folds", type=int, default=5, help="Number of folds for block holdout")
    x.add_argument("--block-m", type=float, default=250.0, help="Block size in metres")
    x.add_argument("--buffer-m", type=float, default=30.0, help="Training exclusion distance around held-out blocks")
    add_training_options(x)

    x = commands.add_parser("evaluate", help="Compare classified GeoTIFF with independent held-out reference labels")
    x.add_argument("--prediction", type=Path, required=True)
    x.add_argument("--reference", type=Path, required=True)
    x.add_argument("--output", type=Path, required=True)

    x = commands.add_parser("prepare-lidar", help="Convert ground-classified LAS/LAZ/EPT into CHM via PDAL")
    x.add_argument("--lidar", required=True)
    x.add_argument("--imagery", type=Path, required=True)
    x.add_argument("--output", type=Path, required=True)
    x.add_argument("--resolution", type=float, default=1.0)

    x = commands.add_parser("fetch-naip", help="Request bounded 4-band NAIP GeoTIFF from Arizona ImageServer")
    x.add_argument("--bbox", nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"), type=float, required=True)
    x.add_argument("--epsg", type=int, required=True, help="Projected metre CRS, e.g. 26912")
    x.add_argument("--output", type=Path, required=True)
    x.add_argument("--resolution", type=float, default=0.3)
    return p


def run_train(args):
    options = {"seed": args.seed, "trees": args.trees, "per_class_limit": args.per_class_limit}
    single = [args.imagery, args.terrain, args.terrain_mode, args.labels]
    if args.scenes:
        if any(value is not None for value in single):
            raise LandFuseError("Use --scenes or the single-scene options, not both")
        return train_scenes(load_scene_list(args.scenes), args.output_model, **options)
    if args.imagery is None or args.labels is None:
        raise LandFuseError("Provide --scenes or --imagery and --labels")
    return train(Scene(args.imagery, args.terrain, args.terrain_mode or "imagery"),
                 args.labels, args.output_model, **options)


def run(args):
    if args.command == "fetch-naip":
        return fetch_naip(args.output, tuple(args.bbox), args.epsg, resolution_m=args.resolution)
    if args.command == "evaluate":
        return evaluate(args.prediction, args.reference, args.output)
    if args.command == "prepare-lidar":
        return pdal_canopy_height(args.lidar, args.imagery, args.output, resolution_m=args.resolution)
    if args.command == "train":
        return run_train(args)
    if args.command == "cross-validate":
        return cross_validate(load_scene_list(args.scenes), args.output, by=args.by, folds=args.folds,
                              block_m=args.block_m, buffer_m=args.buffer_m, seed=args.seed,
                              per_class_limit=args.per_class_limit, trees=args.trees)
    scene = Scene(args.imagery, args.terrain, args.terrain_mode or "imagery")
    if args.command == "inspect":
        with open_scene(scene) as (_, _, metadata):
            return metadata
    return infer(scene, args.model, args.output, trust_local_model=args.trust_local_model)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        print(json.dumps(run(args), indent=2, sort_keys=True))
        return 0
    except LandFuseError as exc:
        print(f"LANDFUSE_STOP={exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"LANDFUSE_STOP=I/O or input processing error ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
