from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image


DEFAULT_EXTENSIONS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)


@dataclass
class SplitResult:
    source_dir: Path
    accepted_dir: Path
    rejected_dir: Path
    min_width: int
    min_height: int
    processed: int
    accepted: int
    rejected: int
    skipped_non_images: int
    accepted_files: List[Path]
    rejected_files: List[Path]


def is_image_file(path: Path, allowed_extensions: Sequence[str] = DEFAULT_EXTENSIONS) -> bool:
    return path.is_file() and path.suffix.lower() in allowed_extensions


def get_image_size(path: Path) -> Optional[Tuple[int, int]]:
    try:
        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        return None


def meets_resolution(size: Tuple[int, int], min_width: int, min_height: int) -> bool:
    width, height = size
    return width >= min_width and height >= min_height


def ensure_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)


def _unique_destination(dest: Path, overwrite: bool) -> Path:
    if overwrite or not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    idx = 1
    while True:
        candidate = parent / f"{stem} ({idx}){suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def _iter_image_files(
    source_dir: Path,
    recursive: bool,
    allowed_extensions: Sequence[str],
) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for p in source_dir.glob(pattern):
        if is_image_file(p, allowed_extensions):
            yield p


def split_images_by_resolution(
    source_dir: Path | str,
    accepted_dir: Optional[Path | str] = None,
    rejected_dir: Optional[Path | str] = None,
    *,
    min_width: int = 800,
    min_height: int = 600,
    move: bool = False,
    recursive: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
    allowed_extensions: Sequence[str] = DEFAULT_EXTENSIONS,
) -> SplitResult:
    """
    Split images from `source_dir` into `accepted_dir` and `rejected_dir` based on resolution.

    Accepted if width >= min_width and height >= min_height.

    Args:
        source_dir: Directory containing images.
        accepted_dir: Output directory for accepted images. Defaults to `<source>/accepted`.
        rejected_dir: Output directory for rejected images. Defaults to `<source>/rejected`.
        min_width: Minimum width in pixels.
        min_height: Minimum height in pixels.
        move: Move files instead of copying.
        recursive: Recurse into subdirectories.
        overwrite: Overwrite existing files at destination.
        dry_run: If True, do not copy/move files; only compute result.
        allowed_extensions: Iterable of allowed file extensions (lowercase, with dots).

    Returns:
        SplitResult with counters and file lists.
    """
    src = Path(source_dir).resolve()
    if accepted_dir is None:
        accepted_dir = src / "accepted"
    if rejected_dir is None:
        rejected_dir = src / "rejected"

    acc_dir = Path(accepted_dir).resolve()
    rej_dir = Path(rejected_dir).resolve()

    ensure_dir(acc_dir)
    ensure_dir(rej_dir)

    processed = 0
    accepted = 0
    rejected = 0
    skipped_non_images = 0
    accepted_files: List[Path] = []
    rejected_files: List[Path] = []

    # Iterate candidates; we count only files that are image candidates by extension.
    for file_path in _iter_image_files(src, recursive, allowed_extensions):
        processed += 1
        size = get_image_size(file_path)
        if size is None:
            # unreadable image -> reject
            target_dir = rej_dir
            rejected += 1
            dest = _unique_destination(target_dir / file_path.name, overwrite)
            if not dry_run:
                ensure_dir(target_dir)
                if move:
                    shutil.move(str(file_path), str(dest))
                else:
                    shutil.copy2(str(file_path), str(dest))
            rejected_files.append(dest)
            continue

        if meets_resolution(size, min_width, min_height):
            target_dir = acc_dir
            accepted += 1
            dest = _unique_destination(target_dir / file_path.name, overwrite)
            if not dry_run:
                ensure_dir(target_dir)
                if move:
                    shutil.move(str(file_path), str(dest))
                else:
                    shutil.copy2(str(file_path), str(dest))
            accepted_files.append(dest)
        else:
            target_dir = rej_dir
            rejected += 1
            dest = _unique_destination(target_dir / file_path.name, overwrite)
            if not dry_run:
                ensure_dir(target_dir)
                if move:
                    shutil.move(str(file_path), str(dest))
                else:
                    shutil.copy2(str(file_path), str(dest))
            rejected_files.append(dest)

    return SplitResult(
        source_dir=src,
        accepted_dir=acc_dir,
        rejected_dir=rej_dir,
        min_width=min_width,
        min_height=min_height,
        processed=processed,
        accepted=accepted,
        rejected=rejected,
        skipped_non_images=skipped_non_images,
        accepted_files=accepted_files,
        rejected_files=rejected_files,
    )


def scale_down_to_box(
    source: Path | str,
    dest: Optional[Path | str] = None,
    *,
    max_width: int = 800,
    max_height: int = 600,
    overwrite: bool = False,
) -> Tuple[Path, bool]:
    """
    Scale down an image to fit within `max_width` x `max_height` if it's larger.

    - Preserves aspect ratio.
    - Does not upscale smaller images; returns unchanged.
    - If `dest` is None, writes next to source with suffix `-scaled` before extension.

    Returns:
        (output_path, resized_flag)
    """
    src_path = Path(source)
    out_path: Path
    if dest is None:
        out_path = src_path.with_name(f"{src_path.stem}-scaled{src_path.suffix}")
    else:
        out_path = Path(dest)

    with Image.open(src_path) as img:
        width, height = img.width, img.height
        if width <= max_width and height <= max_height:
            # No resize needed; copy if dest differs
            if out_path != src_path:
                if overwrite or not out_path.exists():
                    img.save(out_path)
            return out_path, False

        # Compute scale preserving aspect ratio to fit within box
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h)
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))

        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        # Preserve format
        if overwrite or not out_path.exists():
            resized.save(out_path, format=img.format)
        return out_path, True


__all__ = [
    "SplitResult",
    "split_images_by_resolution",
    "is_image_file",
    "get_image_size",
    "meets_resolution",
    "scale_down_to_box",
]
