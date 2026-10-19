#!/usr/bin/env python3
"""
Renders an animation into a video, headless.

  render_animation.py graph.json --duration 4 --fps 30 --samples 64 --output animation.mp4 [args]
  render_animation.py scene.glb --duration 4 --output animation.mp4

The graph time holds still while the samples of a frame accumulate and advances by one
frame after; the Video Sink encodes the last iteration of each frame, what the window
would show.
"""

import argparse
import sys
from pathlib import Path

from lib import scene

# isort: split
import merian.graph as mg

_VIDEO = "script_video"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    parser.add_argument("scene", type=Path, help="a scene or a graph config (.json)")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="the video; the codec picks .mp4 or .mkv",
    )
    parser.add_argument(
        "--duration", type=float, required=True, help="seconds of animation"
    )
    parser.add_argument(
        "--start", type=float, default=0.0, help="graph time of the first frame"
    )
    parser.add_argument("--fps", type=float, default=30.0, help="frames per second")
    parser.add_argument("--samples", type=int, default=64, help="iterations per frame")
    args, config = scene.parse(parser, argv)

    description = config.description()
    node, _, output = scene.displayed(description).partition(".")
    description = scene.headless(description)
    description.add_node(
        "Video Sink",
        _VIDEO,
        {
            # A format string to the Video Sink
            "filename": str(args.output.with_suffix(""))
            .replace("{", "{{")
            .replace("}", "}}"),
            "framerate": args.fps,
            "trigger": "end of run",
            "enable": True,
        },
    )
    description.add_connection(node, _VIDEO, output, "src")

    frames = round(args.duration * args.fps)
    with mg.Graph(scene.context(), description) as graph:
        # The clear of the first frame needs the listeners of a connected graph
        graph.connect()
        if _VIDEO in graph.inactive_nodes:
            raise SystemExit(f"cannot encode: {graph.inactive_nodes[_VIDEO]}")
        for frame in range(frames):
            graph.send_event("clear")
            graph.run(iterations=args.samples, time=args.start + frame / args.fps)
            print(f"frame {frame + 1}/{frames}", end="\r", flush=True)
        graph.wait()
    print(f"\nwrote {args.output.with_suffix('')}.*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
