"""Main module for cricket object identification."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from image_splitter2 import (
    split_images_by_resolution,
    scale_and_crop_to_box,
    is_image_file,
    DEFAULT_EXTENSIONS,
)


def identify_object(image_path):
    """Placeholder: Identify cricket objects in an image."""
    pass


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_sample_source() -> Path:
    root = _project_root()
    return root / "resources" / "images" / "sample_images"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cricket-object-identifier",
        description="Utilities for cricket image operations",
    )

    sub = parser.add_subparsers(dest="command")

    p_split = sub.add_parser(
        "split", help="Split images into accepted/rejected based on resolution"
    )
    p_split.add_argument(
        "--source",
        type=str,
        default=str(_default_sample_source()),
        help="Source directory",
    )
    p_split.add_argument("--accepted-dir", type=str, default=None)
    p_split.add_argument("--rejected-dir", type=str, default=None)
    p_split.add_argument("--min-width", type=int, default=800)
    p_split.add_argument("--min-height", type=int, default=600)
    p_split.add_argument("--move", action="store_true")
    p_split.add_argument("--no-recursive", action="store_true")
    p_split.add_argument("--overwrite", action="store_true")
    p_split.add_argument("--dry-run", action="store_true")
    p_split.add_argument("--resize-accepted", action="store_true")
    p_split.add_argument("--resize-dest", type=str, default=None)

    p_resize = sub.add_parser("resize", help="Scale down images larger than a box")
    p_resize.add_argument("--source", type=str, default=str(_default_sample_source()))
    p_resize.add_argument("--dest", type=str, default=None)
    p_resize.add_argument("--max-width", type=int, default=800)
    p_resize.add_argument("--max-height", type=int, default=600)
    p_resize.add_argument("--overwrite", action="store_true")
    p_resize.add_argument("--no-recursive", action="store_true")

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "split":
        result = split_images_by_resolution(
            source_dir=Path(args.source),
            accepted_dir=Path(args.accepted_dir) if args.accepted_dir else None,
            rejected_dir=Path(args.rejected_dir) if args.rejected_dir else None,
            min_width=args.min_width,
            min_height=args.min_height,
            move=args.move,
            recursive=not args.no_recursive,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

        print(
            f"Processed: {result.processed} | Accepted: {result.accepted} | Rejected: {result.rejected}"
        )
        print(f"Accepted dir: {result.accepted_dir}")
        print(f"Rejected dir: {result.rejected_dir}")

        if args.resize_accepted:
            acc_dir = Path(result.accepted_dir)
            dest_dir = Path(args.resize_dest) if args.resize_dest else None

            count_total = 0
            count_resized = 0

            for p in acc_dir.glob("*"):
                if not is_image_file(p, DEFAULT_EXTENSIONS):
                    continue

                count_total += 1
                out_path = None

                if dest_dir is not None:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    out_path = dest_dir / p.name

                scale_and_crop_to_box(
                    source=p,
                    dest=out_path,
                    target_width=args.min_width,
                    target_height=args.min_height,
                    overwrite=args.overwrite,
                )
                count_resized += 1

            print(f"Resize (accepted): Checked: {count_total} | Resized: {count_resized}")
            if dest_dir is not None:
                print(f"Resized output dir: {dest_dir.resolve()}")
        return

    if args.command == "resize":
        src = Path(args.source)
        recursive = not args.no_recursive
        pattern = "**/*" if recursive else "*"

        count_total = 0
        count_resized = 0

        for p in src.glob(pattern):
            if not is_image_file(p, DEFAULT_EXTENSIONS):
                continue

            count_total += 1
            out_path = None

            if args.dest:
                dest_dir = Path(args.dest)
                dest_dir.mkdir(parents=True, exist_ok=True)
                out_path = dest_dir / p.name

            scale_and_crop_to_box(
                source=p,
                dest=out_path,
                target_width=args.max_width,
                target_height=args.max_height,
                overwrite=args.overwrite,
            )
            count_resized += 1

        print(f"Checked: {count_total} | Resized: {count_resized}")

        if args.dest:
            print(f"Output dir: {Path(args.dest).resolve()}")
        return

    print("Cricket Object Identifier - Ready to identify cricket objects!")


if __name__ == "__main__":
    main()
