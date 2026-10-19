#!/usr/bin/env python3
"""
Tiles images into a labelled contact sheet, for comparing renders side by side.

Reads PNG, JPG, BMP, TGA, Radiance HDR and PFM; HDR and PFM images are tonemapped with
1 - exp(-exposure * x) and sRGB encoded.

  image_grid.py out.png a.hdr b.hdr c.hdr
  image_grid.py out.png "spp 1=a.hdr" "spp 4=b.hdr" --columns 2 --exposure 2
  image_grid.py out.png sweep/*.pfm --title "path expressions"

A label defaults to the file name; "label=path" sets it explicitly, unless the whole item
is a file.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import lib  # noqa: F401

# isort: split
import merian.io

_BACKGROUND = (24, 24, 30)
_LABEL_COLOR = (200, 200, 215)
_LABEL_HEIGHT = 20
_PAD = 8
_TITLE_COLOR = (235, 235, 245)
_TITLE_HEIGHT = 26


def _load(path: str, exposure: float) -> Image.Image:
    suffix = Path(path).suffix.lower()
    if suffix in (".hdr", ".pfm"):
        linear = merian.io.load_image(path)
        linear = (
            np.repeat(linear, 3, axis=2) if linear.shape[2] == 1 else linear[..., :3]
        )
        display = 1.0 - np.exp(-np.maximum(linear, 0.0) * exposure)
        srgb = np.where(
            display <= 0.0031308,
            display * 12.92,
            1.055 * np.power(np.maximum(display, 1e-8), 1 / 2.4) - 0.055,
        )
        return Image.fromarray((np.clip(srgb, 0, 1) * 255).astype(np.uint8))
    return Image.open(path).convert("RGB")


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size)


def write_grid(
    entries: list[tuple[str, str]],
    out: str,
    columns: int = 0,
    width: int = 480,
    exposure: float = 1.0,
    title: str = "",
) -> tuple[int, int]:
    """Writes a contact sheet of (label, path) entries; returns its columns and rows."""
    loaded = []
    for label, path in entries:
        try:
            loaded.append((label, _load(path, exposure)))
        except (OSError, RuntimeError, ValueError) as error:
            print(f"skipping {path}: {error}", file=sys.stderr)
    if not loaded:
        raise ValueError("no images could be read")
    if width < 16:
        raise ValueError("width must be at least 16")

    aspect = loaded[0][1].height / loaded[0][1].width
    tile = (width, max(1, round(width * aspect)))
    for label, image in loaded[1:]:
        if abs((image.height / image.width) - aspect) > 0.01:
            print(
                f"{label}: aspect differs from the first image, stretching",
                file=sys.stderr,
            )
    columns = columns or min(
        range(1, len(loaded) + 1),
        key=lambda c: abs(
            math.log((c * tile[0]) / (math.ceil(len(loaded) / c) * tile[1]) / (16 / 9))
        ),
    )
    rows = math.ceil(len(loaded) / columns)

    title_height = _TITLE_HEIGHT if title else 0
    sheet = Image.new(
        "RGB",
        (
            columns * (tile[0] + _PAD) + _PAD,
            rows * (tile[1] + _LABEL_HEIGHT + _PAD) + _PAD + title_height,
        ),
        _BACKGROUND,
    )
    draw = ImageDraw.Draw(sheet)
    label_font = load_font(13)
    if title:
        draw.text((_PAD, _PAD), title, fill=_TITLE_COLOR, font=load_font(15))

    for index, (label, image) in enumerate(loaded):
        column, row = index % columns, index // columns
        x = _PAD + column * (tile[0] + _PAD)
        y = _PAD + title_height + row * (tile[1] + _LABEL_HEIGHT + _PAD)
        sheet.paste(image.resize(tile, Image.LANCZOS), (x, y))
        draw.text((x + 2, y + tile[1] + 3), label, fill=_LABEL_COLOR, font=label_font)

    sheet.save(out)
    return columns, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("out", help="contact sheet to write (.png)")
    parser.add_argument("images", nargs="+", help="image, or label=image")
    parser.add_argument("--columns", type=int, default=0, help="0 picks a ~16:9 grid")
    parser.add_argument("--width", type=int, default=480, help="tile width in pixels")
    parser.add_argument(
        "--exposure", type=float, default=1.0, help="for HDR and PFM inputs"
    )
    parser.add_argument("--title", default="")
    args = parser.parse_args(argv)

    entries = []
    for item in args.images:
        label, separator, path = item.rpartition("=")
        if not separator or Path(item).is_file():
            label, path = Path(item).stem, item
        entries.append((label, path))

    try:
        columns, rows = write_grid(
            entries, args.out, args.columns, args.width, args.exposure, args.title
        )
    except ValueError as error:
        sys.exit(str(error))
    print(f"{args.out}: {len(entries)} images, {columns}x{rows}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
