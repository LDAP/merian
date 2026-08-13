#!/usr/bin/env python3

"""
Tiles images into a labelled contact sheet, for comparing renders side by side.

Reads what merian writes: PNG/JPG/BMP/TGA through PIL, plus Radiance HDR and PFM natively.
Float images are tonemapped with 1 - exp(-exposure * x) and sRGB encoded.

  image_grid.py out.png a.hdr b.hdr c.hdr
  image_grid.py out.png "spp 1=a.hdr" "spp 4=b.hdr" --columns 2 --exposure 2
  image_grid.py out.png sweep/*.hdr --title "path expressions"

A label defaults to the file name; "label=path" sets it explicitly. The grid is chosen to
stay close to 16:9 unless --columns says otherwise.
"""

import argparse
import math
import pathlib
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LABEL_HEIGHT = 20
TITLE_HEIGHT = 26
PAD = 8
BACKGROUND = (24, 24, 30)
LABEL_COLOR = (200, 200, 215)
TITLE_COLOR = (235, 235, 245)


def load_hdr(path):
    """Radiance RGBE, new-style RLE scanlines."""
    with open(path, "rb") as file:
        if file.readline().strip() != b"#?RADIANCE":
            raise ValueError(f"{path}: not a Radiance file")
        while file.readline().strip() != b"":
            pass
        dims = file.readline().split()
        if len(dims) != 4 or dims[0] != b"-Y" or dims[2] != b"+X":
            raise ValueError(f"{path}: unsupported scanline order")
        height, width = int(dims[1]), int(dims[3])
        data = file.read()

    out = np.zeros((height, width, 3), np.float32)
    pos = 0
    for y in range(height):
        if data[pos] != 2 or data[pos + 1] != 2:
            raise ValueError(f"{path}: only RLE scanlines are supported")
        pos += 4
        channels = np.zeros((4, width), np.uint8)
        for channel in range(4):
            x = 0
            while x < width:
                count = data[pos]
                pos += 1
                if count > 128:  # a run of one value
                    channels[channel, x : x + count - 128] = data[pos]
                    pos += 1
                    x += count - 128
                else:
                    channels[channel, x : x + count] = np.frombuffer(
                        data[pos : pos + count], np.uint8
                    )
                    pos += count
                    x += count
        exponent = channels[3].astype(np.int32)
        scale = np.where(exponent == 0, 0.0, np.ldexp(1.0, exponent - 136))
        for channel in range(3):
            out[y, :, channel] = (channels[channel] + 0.5) * scale
    return out


def load_pfm(path):
    with open(path, "rb") as file:
        magic = file.readline().strip()
        channels = 3 if magic == b"PF" else 1 if magic == b"Pf" else 0
        if channels == 0:
            raise ValueError(f"{path}: bad PFM magic")
        width, height = (int(v) for v in file.readline().split())
        scale = float(file.readline())
        dtype = np.dtype(">f4" if scale > 0 else "<f4")  # a positive scale means big endian
        pixels = np.frombuffer(file.read(width * height * channels * 4), dtype)
    pixels = pixels.astype(np.float32).reshape(height, width, channels)
    pixels = np.flipud(pixels)  # PFM is bottom-up
    return np.repeat(pixels, 3, axis=2) if channels == 1 else pixels


def load(path, exposure):
    suffix = pathlib.Path(path).suffix.lower()
    if suffix in (".hdr", ".pfm"):
        linear = load_hdr(path) if suffix == ".hdr" else load_pfm(path)
        display = 1.0 - np.exp(-np.maximum(linear, 0.0) * exposure)
        srgb = np.where(
            display <= 0.0031308,
            display * 12.92,
            1.055 * np.power(np.maximum(display, 1e-8), 1 / 2.4) - 0.055,
        )
        return Image.fromarray((np.clip(srgb, 0, 1) * 255).astype(np.uint8))
    return Image.open(path).convert("RGB")


def font(size):
    for candidate in (
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if pathlib.Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", help="contact sheet to write (.png)")
    parser.add_argument("images", nargs="+", help="image, or label=image")
    parser.add_argument("--columns", type=int, default=0, help="0 picks a ~16:9 grid")
    parser.add_argument("--width", type=int, default=480, help="tile width in pixels")
    parser.add_argument("--exposure", type=float, default=1.0, help="for HDR and PFM inputs")
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    entries = []
    for item in args.images:
        # split at the last '=' so a label may contain one
        label, separator, path = item.rpartition("=")
        if not separator:
            label, path = pathlib.Path(item).stem, item
        try:
            entries.append((label, load(path, args.exposure)))
        except (OSError, ValueError, IndexError) as error:
            print(f"skipping {path}: {error}", file=sys.stderr)
    if not entries:
        sys.exit("no images could be read")

    if args.width < 16:
        sys.exit("--width must be at least 16")
    aspect = entries[0][1].height / entries[0][1].width
    tile = (args.width, max(1, round(args.width * aspect)))
    for label, image in entries[1:]:
        if abs((image.height / image.width) - aspect) > 0.01:
            print(f"{label}: aspect differs from the first image, stretching", file=sys.stderr)
    # the grid whose overall shape lands closest to 16:9
    columns = args.columns or min(
        range(1, len(entries) + 1),
        key=lambda c: abs(math.log((c * tile[0]) /
                                   (math.ceil(len(entries) / c) * tile[1]) / (16 / 9))),
    )
    rows = math.ceil(len(entries) / columns)

    title_height = TITLE_HEIGHT if args.title else 0
    sheet = Image.new(
        "RGB",
        (columns * (tile[0] + PAD) + PAD,
         rows * (tile[1] + LABEL_HEIGHT + PAD) + PAD + title_height),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(sheet)
    label_font = font(13)
    if args.title:
        draw.text((PAD, PAD), args.title, fill=TITLE_COLOR, font=font(15))

    for index, (label, image) in enumerate(entries):
        column, row = index % columns, index // columns
        x = PAD + column * (tile[0] + PAD)
        y = PAD + title_height + row * (tile[1] + LABEL_HEIGHT + PAD)
        sheet.paste(image.resize(tile, Image.LANCZOS), (x, y))
        draw.text((x + 2, y + tile[1] + 3), label, fill=LABEL_COLOR, font=label_font)

    sheet.save(args.out)
    print(f"{args.out}: {len(entries)} images, {columns}x{rows}")


if __name__ == "__main__":
    main()
