#!/usr/bin/env python3
"""Write a deterministic lat-long environment map (Radiance .hdr) for renderer evaluations:
a sky gradient plus a small bright sun for prominent, hard direct light.

  make_sun_env.py out.hdr [--width 2048] [--sun-intensity 400]
"""

import argparse

import numpy as np


def write_radiance_hdr(path, image):
    """RGBE, per the Radiance .hdr format stb_image reads."""
    height, width, _ = image.shape
    brightest = np.max(image, axis=2)
    exponent = np.zeros_like(brightest)
    mantissa = np.zeros_like(image)
    nonzero = brightest > 1e-32
    m, e = np.frexp(brightest[nonzero])
    exponent[nonzero] = e
    mantissa[nonzero] = image[nonzero] * (m / brightest[nonzero])[:, None]

    rgbe = np.zeros((height, width, 4), np.uint8)
    rgbe[..., :3] = np.clip(mantissa * 256.0, 0, 255).astype(np.uint8)
    rgbe[..., 3] = np.clip(exponent + 128, 0, 255).astype(np.uint8)

    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(f"-Y {height} +X {width}\n".encode())
        rgbe.tofile(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--sun-intensity", type=float, default=400.0)
    ap.add_argument("--sun-radius", type=float, default=0.045, help="radians")
    ap.add_argument("--sun-elevation", type=float, default=0.6, help="radians")
    ap.add_argument("--sun-azimuth", type=float, default=2.2, help="radians")
    args = ap.parse_args()

    width, height = args.width, args.width // 2
    # theta from +Y down, phi around
    theta = (np.arange(height) + 0.5) / height * np.pi
    phi = (np.arange(width) + 0.5) / width * 2.0 * np.pi
    theta, phi = np.meshgrid(theta, phi, indexing="ij")

    up = np.cos(theta)
    sky = np.stack(
        [
            0.25 + 0.35 * np.clip(up, 0, 1),
            0.38 + 0.45 * np.clip(up, 0, 1),
            0.62 + 0.60 * np.clip(up, 0, 1),
        ],
        axis=-1,
    ).astype(np.float32)
    ground = np.array([0.22, 0.20, 0.17], np.float32)
    image = np.where(up[..., None] > 0.0, sky, ground[None, None, :] * (1.0 + up[..., None]))

    direction = np.stack(
        [np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)], axis=-1
    )
    sun_theta = np.pi / 2.0 - args.sun_elevation
    sun_dir = np.array(
        [
            np.sin(sun_theta) * np.cos(args.sun_azimuth),
            np.cos(sun_theta),
            np.sin(sun_theta) * np.sin(args.sun_azimuth),
        ],
        np.float32,
    )
    cos_sun = np.clip(direction @ sun_dir, -1.0, 1.0)
    image = image + args.sun_intensity * (np.arccos(cos_sun) < args.sun_radius)[..., None]

    write_radiance_hdr(args.out, image.astype(np.float32))
    print(f"{args.out}: {width}x{height}, sun {args.sun_intensity}")


if __name__ == "__main__":
    main()
