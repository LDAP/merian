#!/usr/bin/env python3
"""Generates merian graph configs for the Reduce bandwidth study.

Two source designs, because a uniform vkCmdClearColorImage can leave an image
fast-cleared / DCC-compressed, which would make the reads unrealistically cheap:

  direct : Color -> measured          (sources are transfer_dst|sampled, DCC possible <=64bpp)
  chain  : Color -> copy_i -> measured (sources are Reduce outputs => STORAGE => DCC off,
                                        which is what a path tracer's irradiance actually is)
"""
import json
import os

OUT = os.path.dirname(os.path.abspath(__file__))

ADD = {"initial value": "", "reduction": "accumulator + current_value"}


def graph(n_inputs, width, height, fmt, design):
    nodes = {}
    for i in range(n_inputs):
        src = f"color_{i}"
        if design == "direct":
            nodes[src] = {
                "type": "Color",
                "enabled": True,
                "properties": {"format": fmt, "extent": [width, height, 1]},
                "outputs": [f"out->measured.input_{i}"],
            }
        else:
            nodes[src] = {
                "type": "Color",
                "enabled": True,
                "properties": {"format": fmt, "extent": [width, height, 1]},
                "outputs": [f"out->copy_{i}.input_0"],
            }
            nodes[f"copy_{i}"] = {
                "type": "Reduce",
                "enabled": True,
                "properties": dict(ADD),
                "outputs": [f"out->measured.input_{i}"],
            }

    nodes["measured"] = {
        "type": "Reduce",
        "enabled": True,
        "properties": dict(ADD),
    }
    # "number inputs" is left at its default: setting it rebuilds input_connectors and drops
    # connections made against the previous set. Unconnected inputs cost nothing.
    return {"version": 4, "nodes": nodes}


def write(name, cfg):
    path = os.path.join(OUT, name + ".json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


F32 = "R32G32B32A32Sfloat"
F16 = "R16G16B16A16Sfloat"

RES = ((960, 540), (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160))

written = []
# exp 1: resolution scaling at 2 inputs -- locates the Infinity Cache cliff per format
for w, h in RES:
    written.append(write(f"res_{w}x{h}", graph(2, w, h, F32, "direct")))
for w, h in RES:
    written.append(write(f"res16_{w}x{h}", graph(2, w, h, F16, "direct")))
# exp 2: input-count scaling at 1080p (MALL-resident) and 1440p (DRAM-bound).
# slope = per-surface cost, intercept = fixed per-pass overhead.
for n in (1, 2, 3, 4, 5, 6):
    written.append(write(f"n{n}_1080", graph(n, 1920, 1080, F32, "direct")))
    written.append(write(f"n{n}_1440", graph(n, 2560, 1440, F32, "direct")))
# exp 3: source-design control. "direct" sources are transfer_dst|sampled and uniformly cleared,
# so they could be DCC-compressed / fast-cleared; "chain" sources are Reduce outputs (STORAGE,
# real content) which is what a path tracer's irradiance actually is.
for tag, w, h, fmt in (("1080_f32", 1920, 1080, F32), ("1440_f32", 2560, 1440, F32),
                       ("1440_f16", 2560, 1440, F16)):
    written.append(write(f"chain_{tag}", graph(2, w, h, fmt, "chain")))
    written.append(write(f"direct_{tag}", graph(2, w, h, fmt, "direct")))

for p in written:
    print(p)
