#!/usr/bin/env python3
"""
Renders a scene into an image, headless.

  render_image.py scene.pbrt --samples 1024 --output image.pfm
  render_image.py scene.glb --renderer pt_mcpg --output image.hdr

The graph follows the scene's file type (gltf, glb, fbx, pbrt), a graph config (.json)
is the graph; --help lists its options.
"""

import argparse
import sys
from pathlib import Path

from lib import scene

# isort: split
import merian
import merian.graph as mg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    parser.add_argument("scene", type=Path, help="a scene or a graph config (.json)")
    parser.add_argument(
        "--output", type=Path, required=True, help="the image (PFM, HDR)"
    )
    parser.add_argument("--samples", type=int, default=256, help="iterations")
    parser.add_argument(
        "--capture",
        default=scene.ACCUMULATED,
        help="the output written (default: %(default)s)",
    )
    args, config = scene.parse(parser, argv)
    with mg.Graph(
        scene.context(), scene.headless(config.description(), args.capture)
    ) as graph:
        merian.io.save_image(args.output, scene.render(graph, args.samples))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
