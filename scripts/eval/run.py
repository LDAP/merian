#!/usr/bin/env python3
"""Render + evaluate the dual-chain MCPG experiment end to end.

  run.py <eval_dir> <out_dir> [--path-length N] [--scenes a,b] [-- <render overrides>...]

Renders any missing reference (BSDF, 16 spp, decorrelated seed) and measurement runs
(1 spp, captures at powers of two; cold start and pre-trained protocols), then writes
metrics, labeled previews, image grids and convergence plots via plots.py.
Everything after `--` is forwarded to the measurement runs (e.g. --guiding-prob 0.6).
Finished runs (final capture present) are skipped, so the script is resumable.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import plots

REPO = Path(__file__).resolve().parents[2]

SCENES = {
    "cornell": {"file": "~/repos/bitterli-pbrt/cornell-box/scene-v4.pbrt"},
    "kitchen": {"file": "~/repos/bitterli-pbrt/kitchen/scene-v4.pbrt"},
    "sponza": {
        "file": "~/repos/glTF-Sample-Assets/Models/Sponza/glTF/Sponza.gltf",
        "args": ["--env-map", "{eval_dir}/sun_env.hdr",
                 "--merge", str(REPO / "examples/eval/sponza_camera.json")],
        # ~146 ms/frame at 16 spp; 8192 frames would run into the timeout
        "ref_frames": 4096,
    },
}

GRAPH_FOR_EXTENSION = {
    ".pbrt": "examples/pbrt.json",
    ".gltf": "examples/gltf.json",
    ".glb": "examples/gltf.json",
}

TECHNIQUE_ARGS = {
    "bsdf": ["--reference-mode", "true"],
    "old": [],
    "new": ["--dual-chain", "true"],
}
TRAIN_FRAMES = 1024
MEASURE_FRAMES = 4096
REF_SPP = 16
REF_SEED_OFFSET = 999983


def render(scene, out_files_dir, run_args, path_length, timeout=1800):
    scene_file = Path(scene["file"]).expanduser()
    graph = GRAPH_FOR_EXTENSION[scene_file.suffix]
    extra = [a.format(eval_dir=out_files_dir.parents[1]) for a in scene.get("args", [])]
    cmd = [
        str(REPO / "build/merian-graph-run"), str(REPO / graph), str(scene_file), *extra,
        "--renderer", "mcpg", "--merge", str(REPO / "examples/eval/mcpg_capture.json"),
        "--max-path-length", str(path_length), "--time-delta=10", "--validation=off",
        "--capture-file", f"{out_files_dir}/{{record_iteration:05}}", *run_args,
    ]
    out_files_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    errors = [l for l in result.stdout.splitlines() if "[error]" in l or "exception" in l]
    if errors or result.returncode not in (0, -15):
        sys.exit(f"render failed ({result.returncode}): {' '.join(cmd)}\n" + "\n".join(errors[:5]))


def ensure_reference(name, scene, eval_dir, path_length):
    ref_dir = eval_dir / name / "ref"
    if any(ref_dir.glob("*.pfm")):
        return
    frames = scene.get("ref_frames", 8192)
    print(f"[{name}] reference ({REF_SPP} spp x {frames})", flush=True)
    render(scene, ref_dir,
           ["--reference-mode", "true", "--seed-offset", str(REF_SEED_OFFSET),
            "--spp", str(REF_SPP), "--capture-iteration", str(frames),
            "--capture-power", "1", "--capture-quit", str(frames)], path_length)


def ensure_measurement(name, scene, eval_dir, protocol, technique, overrides, path_length):
    run_dir = eval_dir / name / f"{protocol}_{technique}"
    if (run_dir / f"{MEASURE_FRAMES:05}.pfm").exists():
        return
    print(f"[{name}] {protocol}_{technique}", flush=True)
    args = [*TECHNIQUE_ARGS[technique], "--spp", "1",
            "--capture-quit", str(MEASURE_FRAMES), *overrides]
    if protocol == "pre":
        args += ["--capture-enable", "false", "--capture-start", str(TRAIN_FRAMES)]
    render(scene, run_dir, args, path_length)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--path-length", type=int, default=3)
    ap.add_argument("--scenes", default=",".join(SCENES))
    args, overrides = ap.parse_known_args()
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]

    scenes = {n: SCENES[n] for n in args.scenes.split(",")}
    if "sponza" in scenes and not (args.eval_dir / "sun_env.hdr").exists():
        args.eval_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(Path(__file__).parent / "make_sun_env.py"),
                        str(args.eval_dir / "sun_env.hdr"), "--sun-elevation", "1.15"],
                       check=True)

    for name, scene in scenes.items():
        ensure_reference(name, scene, args.eval_dir, args.path_length)
        for protocol in ("cold", "pre"):
            for technique in TECHNIQUE_ARGS:
                ensure_measurement(name, scene, args.eval_dir, protocol, technique,
                                   overrides, args.path_length)

    plots.evaluate(args.eval_dir, args.out_dir)


if __name__ == "__main__":
    main()
