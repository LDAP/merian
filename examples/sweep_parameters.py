#!/usr/bin/env python3
"""
Renders every combination of the swept values and measures error and time against a
reference, headless.

  sweep_parameters.py scene.pbrt --reference ref.pfm --samples 256 \\
      --sweep "/nodes/render/properties/surface/samples per pixel=1,2,4" \\
      --sweep "/nodes/render/properties/surface/max path length=3,5" [graph options]

A sweep is a JSON pointer and its values; the graph options apply to every render. The
table goes to stdout and sweep.csv, with the squared error (relMSE as it is) times the
render time as the efficiency to compare at equal time.
"""

import argparse
import copy
import csv
import itertools
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
        "--reference", type=Path, required=True, help="see render_reference.py"
    )
    parser.add_argument(
        "--samples", type=int, default=256, help="iterations per render"
    )
    parser.add_argument(
        "--sweep",
        action="append",
        required=True,
        metavar="POINTER=V1,V2,...",
        help="a JSON pointer and its values; repeatable",
    )
    parser.add_argument("--metric", default="rmse", choices=scene.METRICS)
    parser.add_argument(
        "-o", "--out", type=Path, default=Path("sweep"), help="output directory"
    )
    args, base = scene.parse(parser, argv)
    sweeps = []
    for sweep in args.sweep:
        pointer, separator, values = sweep.partition("=")
        if not separator:
            parser.error(f"--sweep expects POINTER=V1,V2,..., got '{sweep}'")
        sweeps.append((pointer, values.split(",")))

    reference = scene.rgb(merian.io.load_image(args.reference))
    args.out.mkdir(parents=True, exist_ok=True)
    context = scene.context()
    rows = []
    squared = args.metric != "relmse"
    for values in itertools.product(*(choices for _, choices in sweeps)):
        config = copy.deepcopy(base)
        for (pointer, _), value in zip(sweeps, values):
            config.set_text(pointer, value)
        description = scene.headless(config.description(), scene.ACCUMULATED)
        with mg.Graph(context, description) as graph:
            graph.profiler.enabled = True
            image = scene.render(graph, args.samples)
            milliseconds = scene.render_ms(graph.profiler.report().gpu)
        name = "_".join(value.replace("/", "-") for value in values)
        merian.io.save_image(args.out / f"{name}.pfm", image)
        error = scene.metrics(reference, image)[args.metric]
        squared_error = error**2 if squared else error
        rows.append(
            [
                *values,
                f"{error:.4g}",
                f"{milliseconds:.3f}",
                f"{squared_error * milliseconds:.4g}",
            ]
        )
        print("  ".join(rows[-1]), flush=True)

    header = [Path(pointer).name for pointer, _ in sweeps]
    header += [
        args.metric,
        "ms/iteration",
        f"{args.metric}{'^2' if squared else ''} x ms",
    ]
    with open(args.out / "sweep.csv", "w", newline="") as file:
        csv.writer(file).writerows([header, *rows])
    best = min(rows, key=lambda row: float(row[-1]))
    print(
        f"wrote {args.out / 'sweep.csv'}; most efficient: "
        f"{', '.join(best[: len(sweeps)])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
