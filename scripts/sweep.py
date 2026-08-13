#!/usr/bin/env python3

"""
Renders a merian graph once per value of a varied parameter and tiles the results.

Each run captures one image at a fixed iteration, so the variants are comparable. The value is
either a CLI option the graph declares, or a JSON pointer into the config for anything else.

  sweep.py examples/gltf.json --vary spp=1,2,4,16 -- scene.glb
  sweep.py examples/gltf.json --vary "/nodes/render/properties/max path length=2,3,5" -- scene.glb
  sweep.py examples/gltf.json --vary debug=off,fireflies -o dbg.png -- scene.glb --env-map e.hdr

Everything after -- is passed to merian-graph-run unchanged.
"""

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

CAPTURE_NODE = "sweep_capture"


def set_pointer(config, pointer, value):
    """Builds the nested dicts of a JSON pointer, RFC 6901 escapes included."""
    node = config
    parts = [part.replace("~1", "/").replace("~0", "~") for part in pointer.split("/")[1:]]
    for part in parts[:-1]:
        if not isinstance(node, dict):
            sys.exit(f"{pointer}: only object properties can be swept, not array elements")
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def parse_value(text):
    if text in ("true", "false"):
        return text == "true"
    for convert in (int, float):
        try:
            return convert(text)
        except ValueError:
            pass
    return text


def find_display_source(config):
    """The node output feeding the swapchain, i.e. what the window shows."""
    for name, node in config.get("nodes", {}).items():
        for connection in node.get("outputs", []) + node.get("$+$outputs", []):
            output, _, target = connection.partition("->")
            if target.endswith(".src") and target.split(".")[0] in ("blit", "output"):
                return f"{name}.{output}"
    return None


def capture_merge(source, path, frames, image_format):
    node, _, output = source.partition(".")
    return {
        "nodes": {
            CAPTURE_NODE: {
                "type": "Image Write",
                "enabled": True,
                "properties": {
                    "format": image_format.upper(),
                    "filename": str(path),
                    "iteration": frames,
                    # a fixed start makes every variant capture the same iteration
                    "advanced": {"start at run": 5, "exit at iteration": frames + 5},
                },
            },
            node: {"$+$outputs": [f"{output}->{CAPTURE_NODE}.src"]},
        }
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graph", help="graph config to run")
    parser.add_argument("--vary", required=True,
                        help="NAME=V1,V2,... where NAME is a graph CLI option or a JSON pointer")
    parser.add_argument("--separator", default=",",
                        help="between values, for values that contain a comma")
    parser.add_argument("-o", "--out", default="sweep.png", help="contact sheet to write")
    parser.add_argument("--frames", type=int, default=200, help="iteration to capture at")
    parser.add_argument("--format", default="hdr", choices=["hdr", "png", "jpg", "pfm"])
    parser.add_argument("--exposure", type=float, default=1.0, help="for the tonemap of HDR/PFM")
    parser.add_argument("--columns", type=int, default=0)
    parser.add_argument("--capture", default="",
                        help="NODE.OUTPUT to capture; defaults to what the window shows")
    parser.add_argument("--keep", default="", help="directory to keep the rendered images in")
    parser.add_argument("--binary", default="build/merian-graph-run")
    parser.add_argument("--timeout", type=float, default=600, help="seconds per run")
    parser.epilog = "arguments after -- are passed to merian-graph-run"
    # argparse would swallow the options after a REMAINDER positional, so split by hand
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    passthrough = argv[split + 1:]

    name, _, values_text = args.vary.partition("=")
    if not values_text:
        sys.exit("--vary needs NAME=V1,V2,...")
    values = values_text.split(args.separator)

    with open(args.graph) as file:
        config = json.load(file)
    source = args.capture or find_display_source(config)
    if not source:
        sys.exit("could not tell what the window shows, pass --capture NODE.OUTPUT")
    print(f"capturing {source} at iteration {args.frames}")

    keep = pathlib.Path(args.keep) if args.keep else None
    if keep:
        keep.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        directory = keep or pathlib.Path(scratch)
        entries = []
        for index, value in enumerate(values):
            stem = directory / f"{index:02d}"
            merges = [capture_merge(source, stem, args.frames, args.format)]
            command = [args.binary, args.graph]
            if name.startswith("/"):
                merges.append({})
                set_pointer(merges[-1], name, parse_value(value))
            else:
                command += [f"--{name}", value]

            merge_paths = []
            for number, merge in enumerate(merges):
                merge_path = directory / f"{index:02d}_merge{number}.json"
                merge_path.write_text(json.dumps(merge, indent=1))
                merge_paths += ["--merge", str(merge_path)]
            command += merge_paths + passthrough

            print(f"[{index + 1}/{len(values)}] {name}={value}", flush=True)
            log_path = directory / f"{index:02d}.log"
            image = stem.with_suffix("." + args.format)
            image.unlink(missing_ok=True)  # never pass off a previous run's image as this one
            status = None
            # straight to a file: a pipe would throttle the run, merian logs a lot per frame
            with open(log_path, "w") as log:
                try:
                    status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                            check=False, timeout=args.timeout).returncode
                except subprocess.TimeoutExpired:
                    print(f"  timed out after {args.timeout}s", file=sys.stderr)
                except OSError as error:
                    sys.exit(f"cannot run {args.binary}: {error}")
            if status not in (0, None):
                print(f"  exited with {status}, see {log_path}", file=sys.stderr)
            if not image.exists():
                tail = log_path.read_text()[-600:]
                print(f"  no image written, see {log_path}:\n{tail}", file=sys.stderr)
                continue
            entries.append(f"{name} = {value}={image}")

        if not entries:
            sys.exit("no variant produced an image")
        sheet = [sys.executable, str(pathlib.Path(__file__).parent / "image_grid.py"),
                 args.out, *entries, "--exposure", str(args.exposure),
                 "--title", f"{args.graph}: {name}"]
        if args.columns:
            sheet += ["--columns", str(args.columns)]
        subprocess.run(sheet, check=True)


if __name__ == "__main__":
    main()
