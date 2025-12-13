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

    for file_path in _iter_image_files(src, recursive, allowed_extensions):
        processed += 1
        size = get_image_size(file_path)
        if size is None:
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


def scale_and_crop_to_box(
    source: Path | str,
    dest: Path | str | None = None,
    *,
    target_width: int = 800,
    target_height: int = 600,
    overwrite: bool = False,
) -> tuple[Path, bool]:
    """
    Scale and crop image to fit exactly target_width × target_height.
    
    - Preserves aspect ratio via letterboxing (adds borders if needed).
    - Converts RGBA → RGB (removes alpha channel).
    - If dest is None, writes next to source with suffix `-scaled` before extension.
    
    Returns:
        (output_path, was_modified_flag)
    """
    src_path = Path(source)
    out_path: Path
    
    if dest is None:
        out_path = src_path.with_name(f"{src_path.stem}-scaled{src_path.suffix}")
    else:
        out_path = Path(dest)

    with Image.open(src_path) as img:
        # ✅ FIX: Convert RGBA to RGB (remove alpha channel)
        if img.mode == 'RGBA':
            # Create white background
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])  # Paste with alpha as mask
            img = rgb_img
        elif img.mode != 'RGB':
            # Convert other modes (e.g., P, L) to RGB
            img = img.convert('RGB')

        width, height = img.width, img.height

        # Compute scale to fit within target box (preserve aspect ratio)
        scale = min(target_width / width, target_height / height)
        new_w = int(width * scale)
        new_h = int(height * scale)

        # Resize
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Create target canvas with white background
        canvas = Image.new('RGB', (target_width, target_height), (255, 255, 255))
        
        # Center the resized image on canvas
        offset_x = (target_width - new_w) // 2
        offset_y = (target_height - new_h) // 2
        canvas.paste(img_resized, (offset_x, offset_y))

        # Save
        if overwrite or not out_path.exists():
            # ✅ Force JPEG format for output (removes format detection issues)
            canvas.save(out_path, format='JPEG', quality=95)
        
        was_modified = (new_w != width or new_h != height)
        return out_path, was_modified


__all__ = [
    "SplitResult",
    "split_images_by_resolution",
    "is_image_file",
    "get_image_size",
    "meets_resolution",
    "scale_and_crop_to_box",
]
