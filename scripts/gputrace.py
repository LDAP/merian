#!/usr/bin/env python3
"""
Capture an Nsight GPU trace of merian-graph-run and print per-node GPU times.

Drives the ngfx CLI so no Nsight UI is needed, then parses the auto-exported
tables and prints the median per node. Times are also given in Mcycles
(ms * GPC clock), which is the number to compare across runs -- this laptop's
GPU clock is not stable enough for raw milliseconds to be meaningful.

--sol adds the per-unit speed-of-light table and names the limiting unit per
node; the metric set and the 60 / 80 % triage boundaries are from NVIDIA's "The
Peak-Performance-Percentage Analysis Method for Optimizing Any GPU Workload".
--shaders splits each node by shader stage (raygen, closest_hit, compute, ...),
which is the only per-shader granularity the export carries.

Examples:
  scripts/gputrace.py subprojects/merian-plugin-quake/quake.json --frame 300 \\
      -- --renderer mcpg "-basedir C:/Users/me/.quakespasm -game ad +map ad_tears -nosound"
  scripts/gputrace.py graph.json --sol --shaders
  scripts/gputrace.py graph.json --filter render --filter "*svgf*" --sol
  scripts/gputrace.py graph.json --filter raygen --shaders
  scripts/gputrace.py graph.json --json before.json
  scripts/gputrace.py graph.json --baseline before.json
  scripts/gputrace.py --compare before.json after.json

Only top-level profiler scopes reach the export, so the rows are exactly the
graph's per-node labels; scopes a node opens inside its own process() do not
appear.
"""

import argparse
import fnmatch
import glob
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

NGFX_GLOB = "C:/Program Files/NVIDIA Corporation/Nsight Graphics */host/windows-desktop-nomad-x64/ngfx.exe"
# 2026.3.0's host hangs at "Searching for attachable processes" and never attaches.
PREFERRED_NSIGHT = ("2025.5.0",)
CLOCK_METRIC = "gpc__cycles_elapsed.avg.per_second"
# Per-frame clock drift above this makes the run not comparable to any other.
CLOCK_SPREAD_WARN = 0.02
# Above this a node's frames disagree enough that a small effect cannot be read off it.
NODE_SPREAD_WARN = 0.05
# The performance counters stay locked for a while after a session, so a retry needs a pause.
RETRY_BACKOFF_S = 15

# Per-unit speed-of-light, the "Top SOL" set from NVIDIA's peak-performance-percentage method.
# Blackwell exposes only the Top-Level Triage set, so there is no VRAM/DRAM row.
SOL_UNITS = (
    ("SM", "GPUTrace.sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("L1TEX", "GPUTrace.l1tex__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("L2", "GPUTrace.lts__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("Screen", "GPUTrace.ScreenPipe_throughput.avg.pct_of_peak_sustained_elapsed"),
    ("World", "GPUTrace.WorldPipe_throughput.avg.pct_of_peak_sustained_elapsed"),
    ("PCIe", "GPUTrace.pcie__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("SysL2", "GPUTrace.syslts__throughput.avg.pct_of_peak_sustained_elapsed"),
)
# Inside compute regimes these carry the profiler's own counter streaming, so they must not
# decide the top unit.
SOL_NOT_A_LIMITER = ("PCIe", "SysL2")
HIT_RATES = (
    ("L1 hit", "Top_Level_Triage.l1tex__t_sector_hit_rate.pct"),
    ("L2 hit", "Top_Level_Triage.lts__average_t_sector_hit_rate_srcnode_gpc_realtime.pct"),
)
# Warps active per shader stage, which is what splits a node into its shaders.
SHADER_PREFIX = "GPUTrace.PCSampler.tpc__warps_active_shader_"
SHADER_SUFFIX = ".avg.pct_of_peak_sustained_elapsed"
# Triage boundaries: above the first, take work off the unit; below the second, feed it more.
SOL_BOUND_HIGH = 80.0
SOL_BOUND_LOW = 60.0


def find_ngfx(explicit):
    if explicit:
        return Path(explicit)
    found = sorted(glob.glob(NGFX_GLOB))
    if not found:
        sys.exit(f"no ngfx.exe found; looked in {NGFX_GLOB}")
    for version in PREFERRED_NSIGHT:
        for path in found:
            if version in path:
                return Path(path)
    print(f"warning: none of {PREFERRED_NSIGHT} installed, using {found[-1]}", file=sys.stderr)
    return Path(found[-1])


def quote(token):
    return f'"{token}"' if " " in token else token


def build_command(args, out_dir, graph):
    exe = Path(args.exe or shutil.which("meson") or sys.exit("meson not on PATH; pass --exe"))
    build = Path(args.build).resolve()
    runner_exe = next((build / n for n in ("merian-graph-run.exe", "merian-graph-run")
                       if (build / n).exists()), None)
    if runner_exe is None:
        sys.exit(f"no merian-graph-run in {build}; pass --build")

    runner = ["devenv", f"./{runner_exe.name}", graph.as_posix()] + args.graph_args
    return [
        str(find_ngfx(args.ngfx)),
        "--activity", "GPU Trace Profiler",
        # Backslash paths here yield "Launch failure: Executable not found".
        "--exe", exe.as_posix(),
        "--dir", build.as_posix(),
        "--args", " ".join(quote(t) for t in runner),
        "--output-dir", str(out_dir),
        "--start-after-frames", str(args.frame),
        "--limit-to-frames", str(args.frames),
        "--auto-export",
    ]


def prepare_inputs(args, out_dir):
    """Copy the config out of the tree (the runner rewrites the config it is given) and
    materialize the overlays."""
    graph = Path(args.graph).resolve()
    if args.copy_config:
        copy = out_dir / graph.name
        shutil.copyfile(graph, copy)
        graph = copy

    if args.serialize:
        # With iterations in flight > 1 consecutive iterations overlap and per-node
        # attribution scatters; only safe to drop for frames long enough not to overlap.
        overlay = out_dir / "serialize.json"
        overlay.write_text(json.dumps({"graph_properties": {"iterations in flight": 1}}))
        args.graph_args = ["--merge", str(overlay)] + args.graph_args
    for merge in args.merge:
        args.graph_args = ["--merge", str(Path(merge).resolve())] + args.graph_args
    return graph


def kill_stray():
    subprocess.run(["taskkill", "/F", "/IM", "ngfx.exe"], capture_output=True, check=False)


def read_table(path):
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        parts = line.rstrip("\r").split("\t")
        if len(parts) < 2:
            continue
        values = []
        for cell in parts[1:]:
            try:
                values.append(float(cell))
            except ValueError:
                values = []
                break
        if values:
            rows[parts[0]] = values
    return rows


def summarize(values, clock_mhz):
    median = statistics.median(values)
    return {
        "ms": median,
        "ms_min": min(values),
        "ms_max": max(values),
        # The last frame of a capture is sometimes a >20 % outlier, hence median not mean.
        "spread": (max(values) - min(values)) / median if median else 0.0,
        "mcyc": median * clock_mhz / 1000.0,
    }


def read_regimes(path):
    """One row per debug label, columns are metric x frame with the metric name repeated."""
    if not path.exists():
        return {}
    lines = path.read_text(errors="replace").splitlines()
    if not lines:
        return {}
    columns = {}
    for index, metric in enumerate(lines[0].rstrip("\r").split("\t")[1:]):
        columns.setdefault(metric, []).append(index)

    regimes = {}
    for line in lines[1:]:
        cells = line.rstrip("\r").split("\t")
        values = {}
        for metric, indices in columns.items():
            frames = []
            for index in indices:
                try:
                    frames.append(float(cells[1 + index]))
                except (ValueError, IndexError):
                    pass
            if frames:
                values[metric] = statistics.median(frames)
        if values:
            regimes[cells[0]] = values
    return regimes


def sol_stats(regimes):
    stats = {}
    for name, metrics in regimes.items():
        units = {unit: metrics[metric] for unit, metric in SOL_UNITS if metric in metrics}
        limiters = {u: v for u, v in units.items() if u not in SOL_NOT_A_LIMITER}
        if not limiters:
            continue
        top = max(limiters, key=limiters.get)
        stats[name] = {
            "units": units,
            "top": top,
            "top_pct": limiters[top],
            "hit_rates": {label: metrics[m] for label, m in HIT_RATES if m in metrics},
        }
    return stats


def shader_stats(regimes):
    stats = {}
    for name, metrics in regimes.items():
        stages = {}
        for metric, value in metrics.items():
            if metric.startswith(SHADER_PREFIX) and metric.endswith(SHADER_SUFFIX) and value > 0:
                stages[metric[len(SHADER_PREFIX):-len(SHADER_SUFFIX)]] = value
        if stages:
            stats[name] = stages
    return stats


def verdict(top_pct):
    if top_pct > SOL_BOUND_HIGH:
        return "take work off it"
    if top_pct < SOL_BOUND_LOW:
        return "under-utilized, latency/occupancy bound"
    return "both: less work and more throughput"


def collect(out_dir):
    base = out_dir / "BASE"
    events = read_table(base / "D3DPERF_EVENTS.xls")
    if not events:
        return None
    clocks = read_table(base / "GPUTRACE_FRAME.xls").get(CLOCK_METRIC, [])
    if not clocks or min(clocks) <= 0:
        return None

    clock = statistics.median(clocks)
    frame = read_table(base / "FRAME.xls").get("GPU frame time", [])
    regimes = read_regimes(base / "GPUTRACE_REGIMES.xls")
    return {
        "clock_mhz": clock,
        "clock_spread": (max(clocks) - min(clocks)) / clock,
        "frames": len(next(iter(events.values()))),
        "frame_time": summarize(frame, clock) if frame else None,
        "nodes": {name: summarize(values, clock) for name, values in events.items()},
        "sol": sol_stats(regimes),
        "shaders": shader_stats(regimes),
    }


def capture(args):
    out_dir = Path(args.out) if args.out else Path(tempfile.gettempdir()) / "merian-gputrace"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)

    graph = prepare_inputs(args, out_dir)
    command = build_command(args, out_dir, graph)
    if args.dry_run:
        print(" ".join(quote(c) for c in command))
        return None

    for attempt in range(1, args.retries + 2):
        kill_stray()
        if attempt > 1:
            time.sleep(RETRY_BACKOFF_S)
        shutil.rmtree(out_dir / "BASE", ignore_errors=True)
        try:
            # ngfx hangs forever when the traced app dies early, so never run it unbounded.
            subprocess.run(command, timeout=args.timeout, check=False,
                           capture_output=not args.verbose)
        except subprocess.TimeoutExpired:
            print(f"attempt {attempt}: ngfx timed out after {args.timeout}s", file=sys.stderr)
        result = collect(out_dir)
        if result:
            result["out_dir"] = str(out_dir)
            result["frame"] = args.frame
            return result
        print(f"attempt {attempt}: no usable tables in {out_dir}", file=sys.stderr)
    kill_stray()
    sys.exit("capture failed; check that the app runs standalone and the GPU is idle")


def matches(name, patterns, extra=()):
    """A row is kept when its debug label or any of its shader stages matches a pattern."""
    if not patterns:
        return True
    haystack = [name.lower()] + [str(e).lower() for e in extra]
    for pattern in patterns:
        pattern = pattern.lower()
        for candidate in haystack:
            if fnmatch.fnmatch(candidate, pattern) or pattern in candidate:
                return True
    return False


def select(result, patterns):
    return [n for n in result["nodes"]
            if matches(n, patterns, result.get("shaders", {}).get(n, {}))]


def print_sol(result, names):
    units = [u for u, _ in SOL_UNITS
             if any(u in result["sol"].get(n, {}).get("units", {}) for n in names)]
    rows = [n for n in names if n in result["sol"]]
    if not rows:
        print("no per-regime metrics in this capture")
        return
    width = max(len(n) for n in rows)

    header = "".join(f"{u:>8}" for u in units)
    rates = [label for label, _ in HIT_RATES]
    print(f"{'node':<{width}}{header}{''.join(f'{r:>8}' for r in rates)}   top SOL")
    for name in sorted(rows, key=lambda n: -result["sol"][n]["top_pct"]):
        s = result["sol"][name]
        cells = "".join(f"{s['units'].get(u, 0.0):>8.1f}" for u in units)
        hits = "".join(f"{s['hit_rates'].get(r, float('nan')):>8.1f}" for r in rates)
        print(f"{name:<{width}}{cells}{hits}   {s['top']} {s['top_pct']:.0f}% "
              f"-> {verdict(s['top_pct'])}")
    if any(u in units for u in SOL_NOT_A_LIMITER):
        print(f"\n{'/'.join(SOL_NOT_A_LIMITER)} are shown but excluded from the top unit: inside "
              "compute regimes they mostly carry the profiler's own counter streaming.")


def print_shaders(result, names):
    rows = [n for n in names if result["shaders"].get(n)]
    if not rows:
        print("no per-shader metrics in this capture")
        return
    width = max(len(n) for n in rows)
    print(f"{'node':<{width}}  warps active per shader stage (% of peak sustained)")
    for name in rows:
        stages = result["shaders"][name]
        listed = "  ".join(f"{stage}={value:.1f}"
                           for stage, value in sorted(stages.items(), key=lambda kv: -kv[1]))
        print(f"{name:<{width}}  {listed}")


def print_report(result, names, top):
    print(f"{result['frames']} frames from frame {result['frame']}, "
          f"GPC clock {result['clock_mhz']:.1f} MHz "
          f"(spread {100 * result['clock_spread']:.1f}%)")
    if result["clock_spread"] > CLOCK_SPREAD_WARN:
        print("  warning: clocks moved during the capture, not comparable to other runs")
    if result["frame_time"]:
        f = result["frame_time"]
        print(f"GPU frame time: {f['ms']:.4f} ms / {f['mcyc']:.2f} Mcyc")
    print()

    nodes = sorted(((n, result["nodes"][n]) for n in names), key=lambda kv: -kv[1]["ms"])[:top]
    width = max((len(name) for name, _ in nodes), default=4)
    print(f"{'node':<{width}}  {'ms':>10}  {'Mcyc':>10}  {'spread':>7}")
    for name, s in nodes:
        flag = " !" if s["spread"] > NODE_SPREAD_WARN else ""
        print(f"{name:<{width}}  {s['ms']:>10.4f}  {s['mcyc']:>10.2f}  "
              f"{100 * s['spread']:>6.1f}%{flag}")


def print_comparison(before, after, patterns, top):
    print(f"clocks: {before['clock_mhz']:.1f} -> {after['clock_mhz']:.1f} MHz "
          "(comparison is on Mcycles, so this is informational)")
    if before["frame_time"] and after["frame_time"]:
        b, a = before["frame_time"]["mcyc"], after["frame_time"]["mcyc"]
        print(f"GPU frame time: {b:.2f} -> {a:.2f} Mcyc ({100 * (a - b) / b:+.1f}%)")
    print()

    names = [n for n in set(before["nodes"]) | set(after["nodes"])
             if matches(n, patterns, after.get("shaders", {}).get(n, {}))]
    names = sorted(names, key=lambda n: -after["nodes"].get(n, before["nodes"][n])["mcyc"])[:top]
    width = max((len(n) for n in names), default=4)
    print(f"{'node':<{width}}  {'before':>10}  {'after':>10}  {'delta':>8}")
    for name in names:
        b, a = before["nodes"].get(name), after["nodes"].get(name)
        if not b or not a:
            print(f"{name:<{width}}  {'-- only in ' + ('before' if b else 'after'):>32}")
            continue
        delta = 100 * (a["mcyc"] - b["mcyc"]) / b["mcyc"] if b["mcyc"] else 0.0
        # Anything inside the per-node spread of either run is not a result.
        noise = 100 * max(b["spread"], a["spread"])
        flag = "" if abs(delta) > noise else "  (within noise)"
        print(f"{name:<{width}}  {b['mcyc']:>10.2f}  {a['mcyc']:>10.2f}  {delta:>7.1f}%{flag}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graph", nargs="?", help="graph config passed to merian-graph-run")
    parser.add_argument("graph_args", nargs="*",
                        help="args forwarded to merian-graph-run; put them after --")
    parser.add_argument("--frame", type=int, default=300,
                        help="start capturing after this many frames (default: 300; quake/AD is "
                             "still warming up at 100 and the per-node spread shows it)")
    parser.add_argument("--frames", type=int, default=4,
                        help="number of frames to capture (default: 4)")
    parser.add_argument("--merge", action="append", default=[], metavar="FILE",
                        help="deep-merge a JSON overlay into the config (repeatable)")
    parser.add_argument("--serialize", action="store_true",
                        help="force iterations in flight = 1; required for scenes under ~5 ms")
    parser.add_argument("--build", default="build", help="merian build directory")
    parser.add_argument("--exe", help="launcher to trace (default: meson from PATH)")
    parser.add_argument("--ngfx", help="path to ngfx.exe (default: newest preferred install)")
    parser.add_argument("--out", help="output directory for the trace and tables")
    parser.add_argument("--timeout", type=int, default=300, help="per attempt, in seconds")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--top", type=int, default=25, help="rows to print")
    parser.add_argument("--filter", action="append", default=[], metavar="PATTERN",
                        help="keep only rows whose debug label or shader stage matches; "
                             "substring or glob, case-insensitive, repeatable")
    parser.add_argument("--sol", action="store_true",
                        help="per-unit speed-of-light and the limiting unit per node")
    parser.add_argument("--shaders", action="store_true",
                        help="warps active per shader stage, per node")
    parser.add_argument("--no-copy-config", dest="copy_config", action="store_false",
                        help="run the config in place, letting the runner rewrite it")
    parser.add_argument("--json", metavar="FILE", help="write the parsed stats")
    parser.add_argument("--baseline", metavar="FILE", help="compare this capture against a --json")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="compare two saved --json files without capturing")
    parser.add_argument("--dry-run", action="store_true", help="print the ngfx command and exit")
    parser.add_argument("--verbose", action="store_true", help="show ngfx output")
    args = parser.parse_args()

    if args.compare:
        before, after = (json.loads(Path(p).read_text()) for p in args.compare)
        print_comparison(before, after, args.filter, args.top)
        return
    if not args.graph:
        parser.error("a graph config is required unless --compare is given")

    result = capture(args)
    if result is None:
        return
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1))

    names = select(result, args.filter)
    if not names:
        sys.exit(f"nothing matches {args.filter}; have: {', '.join(sorted(result['nodes']))}")
    print_report(result, names, args.top)
    if args.sol:
        print()
        print_sol(result, names)
    if args.shaders:
        print()
        print_shaders(result, names)
    if args.baseline:
        print()
        print_comparison(json.loads(Path(args.baseline).read_text()), result,
                         args.filter, args.top)
    print(f"\ntrace: {result['out_dir']}")


if __name__ == "__main__":
    main()
