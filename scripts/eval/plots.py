#!/usr/bin/env python3
"""Metrics, labeled previews, image grids and convergence plots for renderer evaluations.

  plots.py <eval_dir> <out_dir>

Expects <eval_dir>/<scene>/{ref,cold_bsdf,cold_old,cold_new,pre_bsdf,pre_old,pre_new}/<iter>.pfm.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TECHNIQUES = [
    ("bsdf", "BSDF sampling", "#2a78d6"),
    ("old", "MCPG (single chain)", "#eb6834"),
    ("new", "MCPG (dual chain)", "#1baf7a"),
]
PROTOCOLS = [
    ("cold", "cold start (guiding trains while accumulating)"),
    ("pre", "after 1024 training frames"),
]

LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)
FONT = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"


def read_pfm(path):
    with open(path, "rb") as f:
        header = f.readline().rstrip()
        if header not in (b"PF", b"Pf"):
            raise ValueError(f"{path}: not a PFM file ({header!r})")
        channels = 3 if header == b"PF" else 1
        while True:
            line = f.readline()
            if not line.startswith(b"#"):
                break
        width, height = (int(v) for v in line.split())
        scale = float(f.readline().rstrip())
        data = np.fromfile(f, "<f4" if scale < 0 else ">f4", width * height * channels)
    return np.flipud(data.reshape(height, width, channels)).astype(np.float32)


def finite(image):
    return np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)


def tonemap(image, exposure):
    x = np.clip(image * exposure, 0.0, None)
    x = x / (1.0 + x)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def to_u8(image, exposure):
    return (np.clip(tonemap(image, exposure), 0, 1) * 255).astype(np.uint8)


def label(image_u8, text):
    img = Image.fromarray(image_u8)
    draw = ImageDraw.Draw(img, "RGBA")
    size = max(16, img.height // 40)
    font = ImageFont.truetype(str(FONT), size)
    pad = size // 2
    box = draw.textbbox((pad, pad), text, font=font)
    draw.rectangle((0, 0, box[2] + pad, box[3] + pad), fill=(0, 0, 0, 180))
    draw.text((pad, pad), text, fill=(255, 255, 255, 255), font=font)
    return np.asarray(img)


def save_labeled(image, exposure, text, path):
    Image.fromarray(label(to_u8(image, exposure), text)).save(path)


def metrics(ref, test, scale):
    r, t = ref * scale, test * scale
    d = t - r
    return {
        "rmse": float(np.sqrt(np.mean(d * d))),
        # Rousselle-style relative MSE on brightness-normalized images
        "rel_mse": float(np.mean(d * d / (r * r + 1e-2))),
    }


def style_axis(ax):
    ax.grid(True, which="major", color="#e8e7e4", linewidth=0.7)
    ax.grid(True, which="minor", color="#f3f2f0", linewidth=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c9c8c4")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.set_xticks([1, 4, 16, 64, 256, 1024, 4096])
    ax.set_xticklabels(["1", "4", "16", "64", "256", "1024", "4096"])


def plot_protocol(ax, runs, protocol, metric):
    for key, name, color in TECHNIQUES:
        curve = runs[f"{protocol}_{key}"]
        spp = np.array(sorted(curve))
        err = np.array([curve[s][metric] for s in spp])
        ax.loglog(spp, err, color=color, linewidth=1.8, marker="o", markersize=4, label=name)


def eval_scene(scene_dir, out_dir, results):
    scene = scene_dir.name
    ref_pfm = next(iter(sorted((scene_dir / "ref").glob("*.pfm"))))
    ref = finite(read_pfm(ref_pfm))
    lum = ref @ LUM
    scale = 1.0 / float(np.mean(lum))
    # photographic log-average key, so a bright window does not underexpose the interior
    exposure = 0.18 / float(np.exp(np.mean(np.log(lum + 1e-4))))
    ref_spp = 16 * int(ref_pfm.stem)
    save_labeled(ref, exposure, f"{scene} — reference, BSDF {ref_spp} spp",
                 out_dir / f"{scene}_reference.png")

    runs = {}
    for protocol, _ in PROTOCOLS:
        for key, name, _ in TECHNIQUES:
            run = f"{protocol}_{key}"
            curve = {}
            for pfm in sorted((scene_dir / run).glob("*.pfm")):
                curve[int(pfm.stem)] = metrics(ref, finite(read_pfm(pfm)), scale)
            runs[run] = curve
            print(scene, run, {s: round(v["rmse"], 4) for s, v in sorted(curve.items())})
            for spp in (1, 4):
                pfm = scene_dir / run / f"{spp:05}.pfm"
                if pfm.exists():
                    save_labeled(finite(read_pfm(pfm)), exposure,
                                 f"{scene} — {name}, {spp} spp ({protocol})",
                                 out_dir / f"{scene}_{run}_{spp}spp.png")
    results[scene] = {"exposure": exposure, "runs": runs}

    for protocol, subtitle in PROTOCOLS:
        fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=160)
        fig.patch.set_facecolor("#fcfcfb")
        ax.set_facecolor("#fcfcfb")
        plot_protocol(ax, runs, protocol, "rmse")
        ax.set_xlabel("samples per pixel", color="#0b0b0b")
        ax.set_ylabel("RMSE", color="#0b0b0b")
        ax.set_title(f"{scene} — {subtitle}", color="#0b0b0b", fontsize=11, loc="left")
        style_axis(ax)
        ax.legend(frameon=False, fontsize=9, loc="lower left")
        fig.tight_layout()
        fig.savefig(out_dir / f"{scene}_convergence_{protocol}.png",
                    facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    # comparison sheet: rows 1/4 spp, columns reference | BSDF | single chain | dual chain
    ref_tile = label(to_u8(ref, exposure), f"reference — {ref_spp} spp")
    for protocol, _ in PROTOCOLS:
        rows = []
        for spp in (1, 4):
            row = [ref_tile] + [
                label(to_u8(finite(read_pfm(scene_dir / f"{protocol}_{key}" / f"{spp:05}.pfm")),
                            exposure), f"{name} — {spp} spp")
                for key, name, _ in TECHNIQUES
            ]
            rows.append(np.concatenate(row, axis=1))
        Image.fromarray(np.concatenate(rows, axis=0)).save(
            out_dir / f"{scene}_sheet_{protocol}.png")


def evaluate(eval_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    scenes = [d for d in sorted(eval_dir.iterdir()) if any((d / "ref").glob("*.pfm"))]
    for scene_dir in scenes:
        eval_scene(scene_dir, out_dir, results)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=1))

    # combined small-multiples figure, one panel per scene
    for protocol, subtitle in PROTOCOLS:
        fig, axes = plt.subplots(1, len(scenes), figsize=(4.6 * len(scenes), 4.2), dpi=160)
        fig.patch.set_facecolor("#fcfcfb")
        for ax, scene_dir in zip(np.atleast_1d(axes), scenes):
            ax.set_facecolor("#fcfcfb")
            plot_protocol(ax, results[scene_dir.name]["runs"], protocol, "rmse")
            ax.set_title(scene_dir.name, color="#0b0b0b", fontsize=10, loc="left")
            ax.set_xlabel("spp", color="#52514e", fontsize=9)
            style_axis(ax)
        np.atleast_1d(axes)[0].set_ylabel("RMSE", color="#0b0b0b")
        np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8.5, loc="lower left")
        fig.suptitle(f"Convergence, {subtitle}", color="#0b0b0b", fontsize=11, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_dir / f"convergence_all_{protocol}.png",
                    facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    print("wrote", out_dir)


if __name__ == "__main__":
    evaluate(Path(sys.argv[1]), Path(sys.argv[2]))
