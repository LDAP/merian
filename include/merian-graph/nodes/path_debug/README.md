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
region, how a region matches and its target, and the path color. The tabs hold the rest: **Paths**
(what to draw, filter, line style), **View** (image under the paths, false color, sphere) and
**Analysis** (sampling, BSDF check, pixel, histograms). The image under the paths and its exposure
are also at the bottom left.

## Selection

A recorded path holds a light path, from the camera to a light, for every light it reaches: each
NEE connection and each light it hits. The selection and the filter pick light paths; a path is
drawn up to its last selected light path, and the images add what the selected ones carry.

- **point**: the paths of the clicked pixel that start at the clicked surface. The renderer then
  records every sample of that pixel, which the panels need, and the other pixels only while a
  per-pixel image is shown.
- **region**: the light paths that hit, start at or end at a target: a sphere around the clicked
  surface, or the triangle, part (a mesh at one place), object, mesh or material the click sees,
  the environment for the sky. It looks through glass. The dropdown next to the target chooses one
  without a click.
- `anchor: surface point` follows the surface when the camera moves; a point behind another surface
  keeps no paths. `anchor: pixel` stays on the pixel.
- The filter narrows the selection: expression, contribution (finite or NaN/Inf), scatter events,
  technique.
- Every view shows the selection. With nothing selected, that is every path passing the filter.
- The per-pixel images, `image: filtered` and `false color: pixel statistic`, leave a point out:
  they show the filter and a region over the whole image.
- `image: filtered` shows what the selected light paths add to each pixel, the pixel statistic
  their variance and more. FROZEN, both stay on the surfaces while the camera moves.

Path expressions match the scatter events of a light path: `S G D U` specular, glossy, diffuse,
unclassified; `R T` reflection, transmission; `.` any; a quoted code for the technique that drew
the event's outgoing direction: `'b'` BSDF, `'g'` guiding, `'n'` NEE, `'c'` cosine, `'u'` uniform,
`'rrb'` `'rrn'` replayed BSDF and NEE, `'rec'` reconnection; `<D'g'>` all of; `[SG]` either;
`* + ? {n} {n,m}`. The last event's technique is how the light was reached. For example `L` is a
light the camera sees, `.` direct light, `.'n'` direct light by NEE, `.'b'` direct light a BSDF
sample hit, `.*<T>.*` refracts somewhere, `.*<D'g'>.*` guides a diffuse bounce.

## Recipes

| Goal | How |
| --- | --- |
| Typical paths | the defaults: paths resampled by contribution |
| Fireflies | `draw: brightest share`, then Paths › Drawn paths |
| Where a firefly's weight explodes | `color: BSDF·cos/pdf`: the weight a segment's direction was drawn with, red above 1 |
| How light takes on color | `color: transported color` |
| NaN or Inf | filter `contribution: NaN or Inf` |
| Light through or into a spot | region, `hits` or `ends at` |
| Light of one lamp | region, `ends at`, target `part`, click the lamp |
| Orbit around paths | FROZEN or `keep on camera move` |
| Only the image | `color: off` |
| One kind of light: refracted, NEE, n bounces | the filter, `image: filtered` |
| Where light through a spot lands | region, `image: filtered` |
| Noisy pixels, fireflies, pixels nothing lights | `false color: pixel statistic`: relative standard error, max / mean, zero fraction |
| Where light flows in the scene | `false color: scene heatmap`: contribution, incident radiance |
| Where NEE is wasted | `scene heatmap: NEE failed` |
| Bias | the Error Plot node, or `examples/compare_methods.py` |

## Panels

Shown while a point is selected. Every panel bins the directions sampled at that point, and the
distributions share one scale. Hover a bin, on a panel or the sphere, for its values; View › sphere
draws any panel around the point.

| Panel | Shows |
| --- | --- |
| integrand (normalized) | BSDF·L·cos normalized to integrate to 1: what a zero-variance sampler samples |
| sampled directions | what the renderer sampled; `binned` picks all, scatter, BSDF only, guiding only or NEE only |
| recorded pdf | the pdf the renderer claims for those draws |
| BSDF pdf(), sample(), eval() | the material alone |
| z-score, sampled / pdf | sampled directions against `test against` |
| sampled / integrand | where sampling misses the integrand: blue undersampled, red wasted |
| second moment | where the variance comes from |
| incident radiance (color) | the light arriving at the point, the integrand over eval() |
| integrand, integrand (color), mean luminance, mean color | absolute values |

- BSDF sampling: `binned: BSDF only`, `test against: BSDF pdf()`.
- Guiding or any mixture: `binned: scatter`, `test against: recorded pdf`; the renderer records the
  mixture's pdf.
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
- ReSTIR DI's reservoirs keep no light: its connections end on a sphere or the environment, no
  other target.
