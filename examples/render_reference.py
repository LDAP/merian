#!/usr/bin/env python3
"""
Renders the reference image of a scene, resumable.

  render_reference.py scene.pbrt --samples 65536 --output reference.pfm [graph options]

After each batch <output>.npz is written: an interrupted render resumes from it, and a
finished one continues when asked for more samples. Batches render with distinct graph
seeds, none of them the default seed of the renders judged against the reference.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from lib import scene

# isort: split
import merian
import merian.graph as mg

_FIRST_SEED = 0x10000
_PLAIN_DENOISERS = {"none", "none_volume"}


def accumulate(
    output: Path,
    samples: int,
    batch: int,
    config: mg.config.Config,
    context: merian.Context | None = None,
) -> np.ndarray:
    """Returns the mean of samples iterations, checkpointed in output.npz."""
    denoisers = {
        name
        for group in config.preset_groups()
        if group.name == "denoiser"
        for name, path in group.choices.items()
        if path.resolve() in config.applied
    }
    if others := denoisers - _PLAIN_DENOISERS:
        raise SystemExit(
            f"a reference needs --denoiser {' or '.join(sorted(_PLAIN_DENOISERS))}, "
            f"which average plainly, not {', '.join(sorted(others))}"
        )
    config.set("/nodes/accumulate/properties/high precision", True)
    config.set("/nodes/accumulate/properties/inf history", True)
    signature = hashlib.sha256(
        json.dumps(config.json, sort_keys=True).encode()
    ).hexdigest()
    checkpoint = output.with_name(output.name + ".npz")
    total, done, batches = None, 0, 0
    if checkpoint.exists():
        with np.load(checkpoint) as state:
            if str(state["signature"]) != signature:
                raise SystemExit(
                    f"{checkpoint} belongs to another configuration, remove it"
                )
            total, done, batches = (
                state["total"],
                int(state["samples"]),
                int(state["batches"]),
            )
        print(f"resuming at {done} samples")

    description = scene.headless(config.description(), scene.ACCUMULATED)
    with mg.Graph(context or scene.context(), description) as graph:
        start = time.monotonic()
        rendered = 0
        while done < samples:
            iterations = min(batch, samples - done)
            # Reconnects, which starts the batch from scratch
            graph.set_properties({"graph_properties": {"seed": _FIRST_SEED + batches}})
            mean = scene.render(graph, iterations).astype(np.float64)
            total = mean * iterations if total is None else total + mean * iterations
            done, batches, rendered = (
                done + iterations,
                batches + 1,
                rendered + iterations,
            )
            temporary = checkpoint.with_name(checkpoint.name + ".tmp.npz")
            np.savez(
                temporary,
                total=total,
                samples=done,
                batches=batches,
                signature=signature,
            )
            temporary.replace(checkpoint)
            rate = rendered / (time.monotonic() - start)
            print(
                f"{done}/{samples} samples, {(samples - done) / rate:.0f} s left",
                flush=True,
            )
    return (total / done).astype(np.float32)


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
    parser.add_argument(
        "--samples", type=int, required=True, help="iterations in total"
    )
    parser.add_argument("--batch", type=int, default=4096, help="iterations per batch")
    args, config = scene.parse(parser, argv)
    merian.io.save_image(
        args.output, accumulate(args.output, args.samples, args.batch, config)
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
