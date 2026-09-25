#!/usr/bin/env python3

"""
Compares rendering methods on a scene: an image grid and a convergence plot, headless.

A variant is a name and whatever overrides distinguish it - a renderer, a mixin, a graph option,
or a JSON pointer - so nothing has to be written to a file first:

  eval.py scene.pbrt -v pt --renderer pt -v mcpg --renderer mcpg
  eval.py scene.pbrt -v "p=0.3" --renderer mcpg --set "/nodes/render/properties/guiding prob=0.3" \\
                     -v "p=0.9" --renderer mcpg --set "/nodes/render/properties/guiding prob=0.9"
  eval.py scene.glb --graph examples/gltf.json -v pt --renderer pt --samples 1024

Each variant renders once, capturing the accumulated linear image at every power of two
(examples/mixins/capture.json). Metrics are computed here rather than on the GPU, so a finished
run can be re-plotted with a different metric without rendering again, and the captures double as
the grid. Runs whose last capture exists are skipped, so this is resumable.

The reference is rendered the same way, longer, unless --reference points at an image. Note that
a reference is itself an estimate: once a run approaches its noise the measured error stops
falling, so read the rate well above that, and give the reference several times the samples.
"""

import argparse
import base64
import json
import os
import shlex
import math
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import image_grid  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CAPTURE_MIXIN = ROOT / "examples/mixins/capture.json"
GRAPH_FOR_SUFFIX = {".pbrt": "examples/pbrt.json", ".gltf": "examples/gltf.json",
                    ".glb": "examples/gltf.json"}
COLORS = [(255, 190, 70), (110, 200, 255), (170, 255, 150), (255, 120, 120), (220, 160, 255),
          (255, 235, 130)]


def build_env():
    """meson puts each target's shared library in its own build subdirectory."""
    env = dict(os.environ)
    dll_dirs = sorted({str(p.parent) for p in (ROOT / "build").rglob("*.dll")})
    if dll_dirs:
        env["PATH"] = os.pathsep.join(dll_dirs + [env.get("PATH", "")])
    return env


def run_merian(graph, scene, out_dir, samples, extra, timeout=3600):
    """One render, capturing at every power of two up to `samples`, then quitting."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # a renderer variant replaces the render node wholesale, so it has to be selected before
    # any mixin that appends to it
    renderer = []
    rest = list(extra)
    if "--renderer" in rest:
        i = rest.index("--renderer")
        renderer = rest[i:i + 2]
        del rest[i:i + 2]
    extra = rest
    cmd = [str(ROOT / "build" / "merian-graph-run"), str(graph), str(scene), *renderer,
           "--merge", str(CAPTURE_MIXIN),
           "--capture-file", str(out_dir / "{record_iteration:05}").replace("\\", "/"),
           # a few frames of slack: the capture is written asynchronously and the exit
           # trigger fires before it lands
           "--capture-quit", str(samples + 8), "--validation=off", *extra]
    log = out_dir / "render.log"
    with open(log, "w") as f:
        f.write(" ".join(cmd) + "\n\n")
        f.flush()
        result = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                timeout=timeout, env=build_env())
    if result.returncode != 0:
        errors = [l for l in log.read_text(errors="ignore").splitlines() if "[error]" in l]
        sys.exit(f"render failed ({result.returncode}) in {out_dir}\n" + "\n".join(errors[:5]))
    return log


def captures(run_dir):
    """{samples: path} for the power-of-two captures of one run."""
    return {int(p.stem): p for p in sorted(run_dir.glob("*.pfm")) if p.stem.isdigit()}


def render_ms(log):
    """Average GPU time of the render node, so runs can also be compared at equal time."""
    gpu = log.read_text(errors="ignore").split("\nGPU:\n", 1)[-1]
    import re
    m = re.search(r"^\s+render \(Render.*?\):\s+([0-9.]+)", gpu, re.M)
    return float(m.group(1)) if m else float("nan")


def metrics(reference, image):
    d = np.nan_to_num(image - reference, nan=0.0, posinf=0.0, neginf=0.0)
    # relative to the reference's own brightness, so a dark surround cannot dominate; the
    # epsilon scales with the image, a constant swamps dim linear renders
    eps = max((0.05 * float(np.mean(reference))) ** 2, 1e-12)
    denom = reference * reference + eps
    return {"rmse": float(np.sqrt(np.mean(d * d))),
            "relmse": float(np.mean(d * d / denom)),
            "mae": float(np.mean(np.abs(d)))}


def fit_slope(points):
    pts = [(n, v) for n, v in points if n > 0 and v > 0]
    if len(pts) < 3:
        return float("nan")
    xs = [math.log10(n) for n, _ in pts]
    ys = [math.log10(v) for _, v in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else float("nan")


def plot(path, title, curves, xlabel, ideal, ideal_label):
    from PIL import Image, ImageDraw
    width, height = 900, 500
    pad_l, pad_r, pad_t, pad_b = 82, 230, 46, 56
    font, small = image_grid.load_font(15), image_grid.load_font(13)
    img = Image.new("RGB", (width, height), (22, 22, 26))
    d = ImageDraw.Draw(img)
    xs = [x for _, pts, _ in curves for x, _ in pts]
    ys = [y for _, pts, _ in curves for _, y in pts]
    x0, x1 = math.log10(min(xs)), math.log10(max(xs))
    if min(ys) <= 0.0:
        # comparing an image against itself gives exactly zero, which has no log
        print(f"  skipping the plot for {title}: a metric reached zero")
        return
    y0, y1 = math.log10(min(ys)) - 0.15, math.log10(max(ys)) + 0.15

    def sx(v):
        return pad_l + (width - pad_l - pad_r) * (math.log10(v) - x0) / max(x1 - x0, 1e-9)

    def sy(v):
        return pad_t + (height - pad_t - pad_b) * (y1 - math.log10(v)) / max(y1 - y0, 1e-9)

    for dec in range(math.floor(y0), math.ceil(y1) + 1):
        if y0 <= dec <= y1:
            y = sy(10.0 ** dec)
            d.line([(pad_l, y), (width - pad_r, y)], fill=(58, 58, 66))
            d.text((8, y - 8), f"1e{dec}", font=small, fill=(150, 150, 160))
    for dec in range(math.floor(x0), math.ceil(x1) + 1):
        if x0 <= dec <= x1:
            x = sx(10.0 ** dec)
            d.line([(x, pad_t), (x, height - pad_b)], fill=(58, 58, 66))
            d.text((x - 14, height - pad_b + 6), f"1e{dec}", font=small, fill=(150, 150, 160))

    ax, ay = curves[0][1][0]
    end = 10.0 ** x1
    d.line([(sx(ax), sy(ay)), (sx(end), sy(ay * (end / ax) ** ideal))], fill=(125, 125, 145))
    d.text((sx(end) - 92, sy(ay * (end / ax) ** ideal) - 18), ideal_label, font=small,
           fill=(145, 145, 165))

    for i, (name, pts, note) in enumerate(curves):
        color = COLORS[i % len(COLORS)]
        d.line([(sx(n), sy(v)) for n, v in pts], fill=color, width=2)
        for n, v in pts:
            d.ellipse([sx(n) - 2.5, sy(v) - 2.5, sx(n) + 2.5, sy(v) + 2.5], fill=color)
        d.text((width - pad_r + 12, pad_t + 8 + i * 62), name, font=font, fill=color)
        d.multiline_text((width - pad_r + 12, pad_t + 26 + i * 62), note, font=small,
                         fill=(170, 170, 180), spacing=3)

    d.rectangle([pad_l, pad_t, width - pad_r, height - pad_b], outline=(95, 95, 105))
    d.text((pad_l, 12), title, font=font, fill=(235, 235, 235))
    d.text((pad_l, height - 24), xlabel, font=small, fill=(160, 160, 170))
    img.save(path)
    print("wrote", path)



def report_html(path, title, sections, table, notes):
    """One self-contained page: it prints to PDF without needing its images alongside."""
    def embed(image):
        data = base64.b64encode(pathlib.Path(image).read_bytes()).decode()
        return f'<img src="data:image/png;base64,{data}">'

    rows = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in table[1:])
    head = "".join(f"<th>{c}</th>" for c in table[0])
    body = "".join(
        f"<section><h2>{heading}</h2>{''.join(embed(i) for i in images)}</section>"
        for heading, images in sections)
    note_items = "".join(f"<li>{n}</li>" for n in notes)
    pathlib.Path(path).write_text(f"""<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<style>
 body {{ font: 14px/1.5 system-ui, sans-serif; color: #16161a; background: #fff;
        margin: 32px auto; max-width: 1200px; }}
 h1 {{ font-size: 22px; margin: 0 0 4px; }}
 h2 {{ font-size: 16px; margin: 28px 0 8px; font-weight: 600; }}
 .meta {{ color: #6b6b73; margin-bottom: 20px; }}
 img {{ max-width: 100%; display: block; margin: 8px 0 16px;
        border: 1px solid #e2e2e6; border-radius: 4px; }}
 table {{ border-collapse: collapse; margin: 8px 0 20px; font-variant-numeric: tabular-nums; }}
 th, td {{ text-align: right; padding: 5px 14px; border-bottom: 1px solid #e6e6ea; }}
 th:first-child, td:first-child {{ text-align: left; }}
 th {{ font-weight: 600; color: #4a4a52; }}
 ul {{ color: #4a4a52; }}
 section {{ break-inside: avoid; }}
</style>
<h1>{title}</h1>
<div class="meta">{notes[0] if notes else ""}</div>
<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>
{body}
<h2>Reading this</h2>
<ul>{note_items}</ul>
""", encoding="utf-8")
    print("wrote", path)

def parse_variants(argv):
    """-v NAME followed by that variant's own options, until the next -v."""
    variants = []
    i = 0
    while i < len(argv):
        if argv[i] in ("-v", "--variant"):
            variants.append({"name": argv[i + 1], "args": [], "sets": []})
            i += 2
            continue
        if not variants:
            sys.exit(f"{argv[i]}: every option must follow a -v NAME")
        if argv[i] == "--set":
            variants[-1]["sets"].append(argv[i + 1])
            i += 2
            continue
        variants[-1]["args"].append(argv[i])
        i += 1
    return variants


def variant_extra(variant, out_dir):
    """CLI options plus, if the variant sets JSON pointers, a merge file holding them."""
    extra = list(variant["args"])
    if variant["sets"]:
        config = {}
        for assignment in variant["sets"]:
            pointer, _, value = assignment.partition("=")
            node = config
            parts = [p.replace("~1", "/").replace("~0", "~") for p in pointer.split("/")[1:]]
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            try:
                node[parts[-1]] = json.loads(value)
            except json.JSONDecodeError:
                node[parts[-1]] = value
        merge = out_dir / "overrides.json"
        merge.write_text(json.dumps(config, indent=1))
        extra += ["--merge", str(merge)]
    return extra


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("-o", "--out", default="eval", help="output directory")
    ap.add_argument("--graph", default=None, help="defaults to the scene's file type")
    ap.add_argument("--samples", type=int, default=1024, help="samples per variant")
    ap.add_argument("--reference", default=None,
                    help="reference image; rendered with --reference-args when absent")
    ap.add_argument("--reference-samples", type=int, default=0, help="default: 8x --samples")
    ap.add_argument("--reference-args", default="",
                    help='options for the reference render, quoted: "--renderer pt"')
    ap.add_argument("--metric", default="rmse", choices=("rmse", "relmse", "mae"))
    ap.add_argument("--warmup", type=int, default=0, help="drop samples below this from the fit")
    ap.add_argument("--grid-samples", type=int, default=0,
                    help="capture shown in the grid; default: the last")
    ap.add_argument("--exposure", type=float, default=0.0, help="0 auto-exposes on the reference")
    ap.add_argument("--train", type=int, default=0,
                    help="also measure each variant after N training frames, accumulation cleared")
    ap.add_argument("--no-report", action="store_true", help="skip report.html")
    args, rest = ap.parse_known_args()

    variants = parse_variants(rest)
    if not variants:
        sys.exit("no variants: pass -v NAME [options...] at least once")

    scene = pathlib.Path(args.scene).expanduser()
    graph = args.graph or GRAPH_FOR_SUFFIX.get(scene.suffix)
    if not graph:
        sys.exit(f"{scene.suffix}: no default graph, pass --graph")
    if args.samples < 1 or args.samples & (args.samples - 1):
        sys.exit(f"--samples {args.samples}: captures land on powers of two, pass one")
    out =pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    reference_samples = args.reference_samples or args.samples * 8
    if args.reference:
        reference_path = pathlib.Path(args.reference)
    else:
        run_dir = out / "reference"
        reference_path = run_dir / f"{reference_samples:05}.pfm"
        if not reference_path.exists():
            print(f"reference: {reference_samples} samples", flush=True)
            run_merian(graph, scene, run_dir, reference_samples,
                       shlex.split(args.reference_args) + ["--capture-power", "1",
                                                    "--capture-iteration", str(reference_samples)])
        if not reference_path.exists():
            sys.exit("reference was not written")
    reference = image_grid.load_image(reference_path)

    exposure = args.exposure
    if exposure <= 0.0:
        luminance = reference @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        exposure = 0.18 / float(np.exp(np.mean(np.log(np.maximum(luminance, 0.0) + 1e-4))))

    protocols = [("", 0)] + ([("trained", args.train)] if args.train > 0 else [])
    results, sections, table = {}, [], [["variant", "protocol", args.metric, "slope", "ms/frame"]]
    notes = [f"{scene.name}, {args.samples} spp per variant, reference {reference_samples} spp"]

    for suffix, train in protocols:
        curves, curves_time = [], []
        tiles = [(f"reference  {reference_samples} spp", reference_path)]
        for variant in variants:
            name = f"{variant['name']} {suffix}".strip()
            run_dir = out / name.replace(" ", "_").replace("/", "_")
            if not (run_dir / f"{args.samples:05}.pfm").exists():
                print(f"{name}: {args.samples} samples", flush=True)
                extra = variant_extra(variant, run_dir) + ["--print-times"]
                if train:
                    # recording starts there, and the same event clears the accumulation, so the
                    # measurement begins from a trained sampler with an empty estimate
                    extra += ["--capture-start", str(train)]
                run_merian(graph, scene, run_dir, args.samples, extra)
            shots = captures(run_dir)
            if not shots:
                print(f"  {name}: no captures, skipped")
                continue
            sample = image_grid.load_image(shots[max(shots)])
            if sample.shape != reference.shape:
                # a renderer that does not honour the scene's film resolution would otherwise be
                # compared against resampled content, which reads as a constant error
                print(f"  {name}: renders {sample.shape[1]}x{sample.shape[0]}, reference is "
                      f"{reference.shape[1]}x{reference.shape[0]} - not comparable, grid only")
                tiles.append((f"{name}  (size mismatch)", shots[max(shots)]))
                table.append([variant["name"], suffix or "cold", "-", "-", "-"])
                continue
            curve = {n: metrics(reference, image_grid.load_image(p)) for n, p in shots.items()}
            results[name] = curve
            points = sorted((n, curve[n][args.metric]) for n in curve if n > args.warmup)
            if not points:
                print(f"  {name}: every capture is at or below --warmup {args.warmup}, skipping")
                table.append([variant["name"], suffix or "cold", "-", "-", "-"])
                continue
            ms = render_ms(run_dir / "render.log")
            slope = fit_slope(points)
            note = "slope {:+.2f}\n{:.2f} ms/frame\nfinal {:.3g}".format(slope, ms, points[-1][1])
            curves.append((variant["name"], points, note))
            if ms == ms:
                curves_time.append((variant["name"], [(n * ms, v) for n, v in points], note))
            shown = args.grid_samples or max(shots)
            nearest = min(shots, key=lambda n: abs(n - shown))
            tiles.append((f"{variant['name']}  {nearest} spp", shots[nearest]))
            table.append([variant["name"], suffix or "cold", f"{points[-1][1]:.4g}",
                          f"{slope:+.2f}", f"{ms:.2f}"])
            print(f"  {name}: {args.metric} {points[0][1]:.3e} -> {points[-1][1]:.3e}, "
                  f"slope {slope:+.3f}, {ms:.2f} ms/frame")

        label = "after training" if suffix else "cold start"
        grid_path = out / f"grid{'_' + suffix if suffix else ''}.png"
        image_grid.write_grid([(t, str(p)) for t, p in tiles], grid_path, exposure=exposure,
                              title=f"{scene.stem}: {label}")
        print("wrote", grid_path)
        images = [grid_path]
        if curves:
            ideal, guide = ((-0.5, "slope -1/2") if args.metric in ("rmse", "mae")
                            else (-1.0, "slope -1"))
            samples_path = out / f"convergence{'_' + suffix if suffix else ''}.png"
            plot(samples_path, f"{scene.stem}: {args.metric} vs samples, {label}", curves,
                 "samples per pixel", ideal, guide)
            images.append(samples_path)
            if curves_time:
                time_path = out / f"convergence_time{'_' + suffix if suffix else ''}.png"
                plot(time_path, f"{scene.stem}: {args.metric} vs render time, {label}",
                     curves_time, "render time (ms in the render node)", ideal, guide)
                images.append(time_path)
        sections.append((label, images))

    (out / "metrics.json").write_text(json.dumps(results, indent=1))
    if not args.no_report:
        notes += [
            "A reference is itself an estimate: once a run approaches its noise the measured "
            "error stops falling, so read the rate well above that.",
            "A method that adapts beats the Monte Carlo rate while it is still learning; "
            "--warmup drops that phase from the fit, --train measures from a trained sampler.",
            "The guide line is the rate an unbiased estimator keeps. A curve that flattens "
            "onto a floor is converging to a different image than the reference.",
            "Equal samples is not equal cost: the ms/frame column and the second plot compare "
            "the same runs against render time.",
        ]
        report_html(out / "report.html", f"{scene.stem}: renderer comparison", sections, table,
                    notes)


if __name__ == "__main__":
    main()
