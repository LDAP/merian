#!/usr/bin/env python3
"""
Compares rendering methods on a scene against a reference, headless: an image grid,
convergence plots and report.html.

  compare_methods.py [options] scene [graph options] -v NAME [graph options] ...

A variant is a name and the graph options that distinguish it: a renderer, a mixin, a
graph option or a JSON pointer. The graph options before the first -v apply to every
variant and to the reference, those after -v NAME to that variant only:

  compare_methods.py scene.pbrt -v pt --renderer pt -v mcpg --renderer pt_mcpg
  compare_methods.py scene.glb --samples 256 --max-path-length 5 \\
      -v "p=0.3" --renderer pt_mcpg --set "/nodes/render/properties/x=0.3" \\
      -v "p=0.9" --renderer pt_mcpg --set "/nodes/render/properties/x=0.9"

Each variant captures its accumulated image at every power of two as PFM files. A run
whose last capture exists from the same configuration is skipped, so this is resumable
and a finished run can be re-plotted with another --metric without rendering again.

The reference renders the graph with the common options, resumable as with
render_reference.py, unless --reference names an image. A reference is itself an
estimate: give it several times the samples and read the rate well above its noise.
"""

import argparse
import base64
import functools
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lib import scene

# isort: split
import image_grid
import merian
import merian.graph as mg
import render_reference

_COLORS = [
    (255, 190, 70),
    (110, 200, 255),
    (170, 255, 150),
    (255, 120, 120),
    (220, 160, 255),
    (255, 235, 130),
]
_VARIANT_OPTIONS = ("-v", "--variant")


@dataclass
class _Variant:
    name: str
    config: mg.config.Config


def _render(context, description, samples, out_dir, train=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    with mg.Graph(context, description) as graph:
        graph.profiler.enabled = True
        graph.connect()
        if train:
            graph.run(iterations=train)
            graph.send_event("clear")
        counts = {2**k for k in range(samples.bit_length())} | {samples}
        graph.sink(scene.CAPTURE).capture_at([train + n - 1 for n in counts])
        graph.profiler.restart_report()
        graph.run(iterations=samples)
        graph.wait()
        for identifier, reason in graph.inactive_nodes.items():
            print(f"  not running: {identifier} {reason}")
        report = graph.profiler.report()
        captures = graph.sink(scene.CAPTURE).get_all()
        if len(captures) != len(counts):
            raise SystemExit(f"{out_dir}: captured {len(captures)} of {len(counts)}")
        for capture in captures:
            merian.io.save_image(
                out_dir / f"{capture.run_iteration - train + 1:05}.pfm",
                scene.rgb(capture.array),
            )
    return scene.render_ms(report.gpu)


def _captures(run_dir):
    return {int(p.stem): p for p in sorted(run_dir.glob("*.pfm")) if p.stem.isdigit()}


def _signature(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def _fit_slope(points):
    pts = [(n, v) for n, v in points if n > 0 and v > 0]
    if len(pts) < 3:
        return float("nan")
    xs, ys = np.log10(np.array(pts, dtype=np.float64)).T
    return float(np.polyfit(xs, ys, 1)[0])


def _plot(path, title, curves, xlabel, ideal, ideal_label) -> bool:
    width, height = 900, 500
    pad_l, pad_r, pad_t, pad_b = 82, 230, 46, 56
    font, small = image_grid.load_font(15), image_grid.load_font(13)
    image = Image.new("RGB", (width, height), (22, 22, 26))
    draw = ImageDraw.Draw(image)
    xs = [x for _, pts, _ in curves for x, _ in pts]
    ys = [y for _, pts, _ in curves for _, y in pts]
    x0, x1 = math.log10(min(xs)), math.log10(max(xs))
    if min(ys) <= 0.0:
        print(f"  skipping the plot for {title}: a metric reached zero")
        return False
    y0, y1 = math.log10(min(ys)) - 0.15, math.log10(max(ys)) + 0.15

    def sx(v):
        return pad_l + (width - pad_l - pad_r) * (math.log10(v) - x0) / max(
            x1 - x0, 1e-9
        )

    def sy(v):
        return pad_t + (height - pad_t - pad_b) * (y1 - math.log10(v)) / max(
            y1 - y0, 1e-9
        )

    for dec in range(math.floor(y0), math.ceil(y1) + 1):
        if y0 <= dec <= y1:
            y = sy(10.0**dec)
            draw.line([(pad_l, y), (width - pad_r, y)], fill=(58, 58, 66))
            draw.text((8, y - 8), f"1e{dec}", font=small, fill=(150, 150, 160))
    for dec in range(math.floor(x0), math.ceil(x1) + 1):
        if x0 <= dec <= x1:
            x = sx(10.0**dec)
            draw.line([(x, pad_t), (x, height - pad_b)], fill=(58, 58, 66))
            draw.text(
                (x - 14, height - pad_b + 6),
                f"1e{dec}",
                font=small,
                fill=(150, 150, 160),
            )

    ax, ay = curves[0][1][0]
    end = 10.0**x1
    draw.line(
        [(sx(ax), sy(ay)), (sx(end), sy(ay * (end / ax) ** ideal))],
        fill=(125, 125, 145),
    )
    draw.text(
        (sx(end) - 92, sy(ay * (end / ax) ** ideal) - 18),
        ideal_label,
        font=small,
        fill=(145, 145, 165),
    )

    for i, (name, pts, note) in enumerate(curves):
        color = _COLORS[i % len(_COLORS)]
        draw.line([(sx(n), sy(v)) for n, v in pts], fill=color, width=2)
        for n, v in pts:
            draw.ellipse(
                [sx(n) - 2.5, sy(v) - 2.5, sx(n) + 2.5, sy(v) + 2.5], fill=color
            )
        draw.text((width - pad_r + 12, pad_t + 8 + i * 62), name, font=font, fill=color)
        draw.multiline_text(
            (width - pad_r + 12, pad_t + 26 + i * 62),
            note,
            font=small,
            fill=(170, 170, 180),
            spacing=3,
        )

    draw.rectangle([pad_l, pad_t, width - pad_r, height - pad_b], outline=(95, 95, 105))
    draw.text((pad_l, 12), title, font=font, fill=(235, 235, 235))
    draw.text((pad_l, height - 24), xlabel, font=small, fill=(160, 160, 170))
    image.save(path)
    print("wrote", path)
    return True


def _report_html(path, title, sections, table, notes):
    def embed(image):
        data = base64.b64encode(Path(image).read_bytes()).decode()
        return f'<img src="data:image/png;base64,{data}">'

    rows = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in table[1:]
    )
    head = "".join(f"<th>{c}</th>" for c in table[0])
    body = "".join(
        f"<section><h2>{heading}</h2>{''.join(embed(i) for i in images)}</section>"
        for heading, images in sections
    )
    note_items = "".join(f"<li>{n}</li>" for n in notes[1:])
    Path(path).write_text(
        f"""<!doctype html>
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
""",
        encoding="utf-8",
    )
    print("wrote", path)


def _load_image(path: Path | str) -> np.ndarray:
    return scene.rgb(merian.io.load_image(path))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    parser.add_argument("scene", type=Path, help="a scene or a graph config (.json)")
    parser.add_argument(
        "-o", "--out", type=Path, default=Path("compare"), help="output directory"
    )
    parser.add_argument(
        "--samples", type=int, default=1024, help="iterations per variant"
    )
    parser.add_argument(
        "--reference", type=Path, help="the reference image (default: rendered)"
    )
    parser.add_argument(
        "--reference-samples",
        type=int,
        default=0,
        help="iterations of the reference (default: 8x --samples)",
    )
    parser.add_argument(
        "--reference-batch",
        type=int,
        default=4096,
        help="iterations per batch of the reference",
    )
    parser.add_argument("--metric", default="rmse", choices=scene.METRICS)
    parser.add_argument(
        "--warmup", type=int, default=0, help="drop iterations below this from the fit"
    )
    parser.add_argument(
        "--train",
        type=int,
        default=0,
        help="also measure each variant after N training iterations, accumulation "
        "cleared",
    )
    parser.add_argument(
        "--grid-samples",
        type=int,
        default=0,
        help="the capture shown in the grid (default: the last)",
    )
    parser.add_argument(
        "--exposure", type=float, default=0.0, help="0 auto-exposes on the reference"
    )
    parser.add_argument("--no-report", action="store_true", help="skip report.html")
    return parser


def _parse_args(
    argv: list[str] | None,
) -> tuple[argparse.Namespace, mg.config.Config, list[_Variant]]:
    argv = sys.argv[1:] if argv is None else argv
    first = next(
        (i for i, token in enumerate(argv) if token in _VARIANT_OPTIONS), len(argv)
    )
    common, rest = argv[:first], argv[first:]
    parser = _parser()
    args, reference = scene.parse(parser, common)
    variants: list[_Variant] = []
    while rest:
        if len(rest) == 1:
            parser.error(f"{rest[0]}: expected a NAME")
        name, rest = rest[1], rest[2:]
        end = next(
            (i for i, token in enumerate(rest) if token in _VARIANT_OPTIONS), len(rest)
        )
        options, rest = rest[:end], rest[end:]
        if misplaced := [
            token
            for token in options
            if token.partition("=")[0] in parser._option_string_actions
        ]:
            parser.error(
                f"{' '.join(misplaced)}: options of the comparison go before the "
                "first -v"
            )
        if any(variant.name == name for variant in variants):
            parser.error(f"variant '{name}' given twice")
        variants.append(_Variant(name, scene.parse(parser, [*common, *options])[1]))
    if not variants:
        parser.error("no variants: pass -v NAME [graph options] at least once")
    return args, reference, variants


class _Comparison:
    def __init__(self, args: argparse.Namespace, reference: mg.config.Config):
        self.args = args
        self.scene_path = args.scene
        self.out = args.out
        self.out.mkdir(parents=True, exist_ok=True)
        self.context = functools.cache(scene.context)
        self.reference_samples = args.reference_samples or args.samples * 8
        self.reference_path = self.render_reference(reference)
        self.reference_image = _load_image(self.reference_path)
        self.exposure = args.exposure
        if self.exposure <= 0.0:
            luminance = self.reference_image @ np.array(
                [0.2126, 0.7152, 0.0722], np.float32
            )
            self.exposure = 0.18 / float(
                np.exp(np.mean(np.log(np.maximum(luminance, 0.0) + 1e-4)))
            )
        self.results: dict[str, dict] = {}
        self.table = [["variant", "protocol", args.metric, "slope", "ms/frame"]]

    def render_reference(self, config: mg.config.Config) -> Path:
        if self.args.reference:
            return self.args.reference
        path = (
            self.out
            / f"reference_{self.reference_samples:05}_{_signature(config.json)}.pfm"
        )
        if not path.exists():
            print(f"reference: {self.reference_samples} iterations", flush=True)
            mean = render_reference.accumulate(
                path,
                self.reference_samples,
                self.args.reference_batch,
                config,
                self.context(),
            )
            merian.io.save_image(path, mean)
        return path

    def protocol(
        self, variants: list[_Variant], suffix: str, train: int
    ) -> tuple[str, list]:
        """Measures every variant after train frames and returns the report section."""
        args = self.args
        curves, curves_time = [], []
        tiles = [
            (f"reference  {self.reference_samples} iterations", self.reference_path)
        ]
        for variant in variants:
            measured = self.measure(variant, suffix, train)
            if measured is None:
                continue
            tile, points, note, ms = measured
            tiles.append(tile)
            if points is None:
                continue
            curves.append((variant.name, points, note))
            if not math.isnan(ms):
                curves_time.append(
                    (variant.name, [(n * ms, v) for n, v in points], note)
                )

        label = "after training" if suffix else "cold start"
        stem = self.scene_path.stem
        grid_path = self.out / f"grid{'_' + suffix if suffix else ''}.png"
        image_grid.write_grid(
            [(t, str(p)) for t, p in tiles],
            str(grid_path),
            exposure=self.exposure,
            title=f"{stem}: {label}",
        )
        print("wrote", grid_path)
        images = [grid_path]
        if curves:
            ideal, guide = (
                (-0.5, "slope -1/2")
                if args.metric in ("rmse", "mae")
                else (-1.0, "slope -1")
            )
            samples_path = self.out / f"convergence{'_' + suffix if suffix else ''}.png"
            if _plot(
                samples_path,
                f"{stem}: {args.metric} vs iterations, {label}",
                curves,
                "iterations",
                ideal,
                guide,
            ):
                images.append(samples_path)
            if curves_time:
                time_path = (
                    self.out / f"convergence_time{'_' + suffix if suffix else ''}.png"
                )
                if _plot(
                    time_path,
                    f"{stem}: {args.metric} vs render time, {label}",
                    curves_time,
                    "render time (ms in the render node)",
                    ideal,
                    guide,
                ):
                    images.append(time_path)
        return label, images

    def measure(self, variant: _Variant, suffix: str, train: int) -> tuple | None:
        """
        Returns the grid tile, fitted points, plot note and ms per frame, rendering
        unless the captures exist; no points if the variant is shown in the grid only.
        """
        args = self.args
        name = f"{variant.name} {suffix}".strip()
        run_dir = self.out / name.replace(" ", "_").replace("/", "_")
        signature = _signature(variant.config.json, train)
        signature_file = run_dir / "signature.txt"
        if not (
            (run_dir / f"{args.samples:05}.pfm").exists()
            and signature_file.exists()
            and signature_file.read_text() == signature
        ):
            print(f"{name}: {args.samples} iterations", flush=True)
            for stale in run_dir.glob("*.pfm"):
                stale.unlink()
            description = scene.headless(
                variant.config.description(), scene.ACCUMULATED
            )
            ms = _render(self.context(), description, args.samples, run_dir, train)
            (run_dir / "render_ms.txt").write_text(f"{ms}\n")
            signature_file.write_text(signature)
        shots = {n: p for n, p in _captures(run_dir).items() if n <= args.samples}
        if not shots:
            print(f"  {name}: no captures, skipped")
            return None
        sample = _load_image(shots[max(shots)])
        if sample.shape != self.reference_image.shape:
            print(
                f"  {name}: renders {sample.shape[1]}x{sample.shape[0]}, reference is "
                f"{self.reference_image.shape[1]}x{self.reference_image.shape[0]} "
                "- not comparable, grid only"
            )
            self.table.append([variant.name, suffix or "cold", "-", "-", "-"])
            return (
                (f"{name}  (size mismatch)", shots[max(shots)]),
                None,
                None,
                float("nan"),
            )
        curve = {
            n: scene.metrics(self.reference_image, _load_image(p))
            for n, p in shots.items()
        }
        self.results[name] = curve
        points = sorted((n, curve[n][args.metric]) for n in curve if n > args.warmup)
        if not points:
            print(
                f"  {name}: every capture is at or below --warmup {args.warmup}, "
                "skipping"
            )
            self.table.append([variant.name, suffix or "cold", "-", "-", "-"])
            return None
        ms_file = run_dir / "render_ms.txt"
        ms = float(ms_file.read_text()) if ms_file.exists() else float("nan")
        slope = _fit_slope(points)
        note = f"slope {slope:+.2f}\n{ms:.2f} ms/frame\nfinal {points[-1][1]:.3g}"
        shown = args.grid_samples or max(shots)
        nearest = min(shots, key=lambda n: abs(n - shown))
        self.table.append(
            [
                variant.name,
                suffix or "cold",
                f"{points[-1][1]:.4g}",
                f"{slope:+.2f}",
                f"{ms:.2f}",
            ]
        )
        print(
            f"  {name}: {args.metric} {points[0][1]:.3e} -> {points[-1][1]:.3e}, "
            f"slope {slope:+.3f}, {ms:.2f} ms/frame"
        )
        return (
            (f"{variant.name}  {nearest} iterations", shots[nearest]),
            points,
            note,
            ms,
        )


def main(argv: list[str] | None = None) -> int:
    args, reference, variants = _parse_args(argv)
    comparison = _Comparison(args, reference)
    protocols = [("", 0)] + ([("trained", args.train)] if args.train > 0 else [])
    sections = [
        comparison.protocol(variants, suffix, train) for suffix, train in protocols
    ]

    out = comparison.out
    (out / "metrics.json").write_text(json.dumps(comparison.results, indent=1))
    if not args.no_report:
        notes = [
            f"{comparison.scene_path.name}, {args.samples} iterations per variant, "
            f"reference {comparison.reference_samples} iterations",
            "A reference is itself an estimate: once a run approaches its noise the "
            "measured error stops falling, so read the rate well above that.",
            "A method that adapts beats the Monte Carlo rate while it is still "
            "learning; --warmup drops that phase from the fit, --train measures from a "
            "trained sampler.",
            "The guide line is the rate an unbiased estimator keeps. A curve that "
            "flattens onto a floor is converging to a different image than the "
            "reference.",
            "Equal iterations are not equal cost: the ms/frame column and the second "
            "plot compare the same runs against render time.",
        ]
        _report_html(
            out / "report.html",
            f"{comparison.scene_path.stem}: renderer comparison",
            sections,
            comparison.table,
            notes,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
