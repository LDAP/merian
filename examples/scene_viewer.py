#!/usr/bin/env python3
"""
Shows a scene in a window.

  scene_viewer.py scene.glb --renderer pt_mcpg
  scene_viewer.py scene.pbrt --mixin imgui --max-iterations 600

The graph follows the scene's file type (gltf, glb, fbx, pbrt), a graph config (.json)
is the graph; --help lists its options. The options of graph_run.py apply too.
"""

import argparse
import sys
from pathlib import Path

from lib import scene

# isort: split
import graph_run
import merian.graph as mg


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    parser.add_argument("scene", type=Path, help="a scene or a graph config (.json)")
    known, rest = mg.config.parse_known(parser, argv)
    if known.scene is None or "--help" in argv:
        # Exits with the help or the usage error
        scene.parse(parser, argv)
    graph, scene_args = scene.graph_and_args(known.scene)
    return graph_run.main([str(graph), *scene_args, *rest])


if __name__ == "__main__":
    sys.exit(main())
