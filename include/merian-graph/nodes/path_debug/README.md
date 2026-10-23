# Path Debugger

Draws and inspects the paths a renderer records. The renderer records only while this node is
connected.

```
python examples/graph_run.py examples/pbrt.json scene.pbrt --mixin debug
```

## Controls

| Input | Action |
| --- | --- |
| Alt+click | select the surface under the cursor |
| Alt+scroll | resize a region |
| Alt+Backspace | clear the selection |
| click the LIVE badge | LIVE → FROZEN (hold the paths) → OFF |

Every binding uses one modifier, `keys/modifier`. The header holds what changes most: point or
region, how a region matches, and the path color. The tabs hold the rest: **Paths** (what to draw, filter, line style),
**View** (image under the paths, false color, sphere) and **Analysis** (sampling, BSDF check,
pixel, histograms). The image under the paths can also be chosen at the bottom left.

## Selection

- **point**: the paths of the clicked pixel that start at the clicked surface. The renderer then
  records only that pixel, every sample, unless a per-pixel image is shown. The panels need it.
- **region**: the paths that pass through, start inside or end inside a sphere around the clicked
  surface. It looks through glass.
- `anchor: surface point` follows the surface when the camera moves; a point behind another surface
  keeps no paths. `anchor: pixel` stays on the pixel.
- The filter narrows the selection: expression, contribution (finite or NaN/Inf), scatter events,
  material, technique.
- Every view shows the selection. With nothing selected, that is every path passing the filter.
- The per-pixel images, `image: filtered` and `false color: pixel statistic`, leave a point out:
  they show the filter and a region over the whole image.

Path expressions match a path's scatter events: `S G D U` specular, glossy, diffuse, unclassified;
`R T` reflection, transmission; `.` any; `<RD>` both; `[SG]` either; `* + ? {n} {n,m}`; a trailing
`N` ends at an NEE connection. For example `.*<T>.*` refracts somewhere, `.N` is direct light by
NEE, `.*S.+N` is NEE after a specular bounce.

## Recipes

| Goal | How |
| --- | --- |
| Typical paths | the defaults: paths resampled by contribution |
| Fireflies | `draw: brightest share`, then Paths › Drawn paths |
| Where a firefly's weight explodes | `color: throughput gain`: f·cos/pdf per scattered segment, red above 1 |
| NaN or Inf | filter `contribution: NaN or Inf` |
| Light through or into a spot | region, `passes through` or `ends inside` |
| Orbit around paths | FROZEN or `keep on camera move` |
| One kind of light: refracted, NEE, n bounces | the filter, `image: filtered` |
| Where light through a spot lands | region, `image: filtered` |
| Noisy pixels | `false color: pixel statistic` |
| Where light flows in the scene | `false color: scene heatmap` |
| Bias | the Error Plot node, or `examples/compare_methods.py` |

## Panels

Shown while a point is selected. Every panel bins the directions sampled at that point, and the
distributions share one scale. Hover a bin, on a panel or the sphere, for its values; View › sphere
draws any panel around the point.

| Panel | Shows |
| --- | --- |
| integrand (normalized) | f·L·cos as a density: what a zero-variance sampler samples |
| sampled directions | what the renderer sampled; `binned` picks all, scatter, BSDF only, guiding only or NEE only |
| recorded pdf | the pdf the renderer claims for those draws |
| BSDF pdf(), sample(), eval() | the material alone |
| z-score, density / pdf | sampled directions against `test against` |
| sampled / integrand | where sampling misses the integrand: blue undersampled, red wasted |
| second moment | where the variance comes from |
| incident radiance (color) | the light arriving at the point, the integrand over eval() |
| integrand, integrand (color), mean luminance, mean color | absolute values |

- BSDF sampling: `binned: BSDF only`, `test against: BSDF pdf()`.
- Guiding or any mixture: `binned: scatter`, `test against: recorded pdf`; the renderer records the
  mixture density.
- chi2/dof near 1 (green) passes. A bin at the edge of a partly occluded light reads low.
- Reused samples (ReSTIR) are tested with batch means; raise `batch frames` when asked to.

## BSDF check

`BSDF sample()` draws while it is shown, up to what a check needs. `check BSDF` tops the draws up
and verifies the material at the incident direction (camera ray or the picker): energy
conservation, reciprocity, pdf normalization, weight = eval / pdf, sampled pdf = pdf(),
finiteness, sampling integrates eval, `get_albedo()`.

- When `sample()` can reject, the pdf integrates to the share of valid draws, not to 1.
- A lobe too narrow for the integration grid reads n/a.
- `get_albedo()` is an approximation for denoiser demodulation: APPROX when far off, n/a with
  refraction, whose sample weights carry eta².

## Notes

- The renderer subsamples the record stream to fit its buffer (`record probability`); counts are a
  sample.
- ReSTIR PT: `debugger records: candidates` for the analysis, `shaded paths` to see reconnection and
  replay (no chi2).
- Segments behind geometry are tested against the gbuffer; glass counts as an occluder.
