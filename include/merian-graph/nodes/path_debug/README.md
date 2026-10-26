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
| Alt+scroll | resize a sphere |
| Alt+Backspace | clear the selection |
| click the LIVE badge | LIVE → FROZEN (hold the paths) → OFF |

Every binding uses one modifier, Preferences › modifier. The header holds the selection; the tabs
hold the rest: **Paths** (color, what to draw, line style, list), **Filter**, **View** (background,
false color, sphere, reference), **Analysis** (BSDF check, panels, sampling, statistics) and
**Preferences**. The image under the paths, its exposure and its reset are at the bottom left.

## Selection

A recorded path holds a light path, from the camera to a light, for every light it reaches: each
NEE connection and each light it hits. A path is drawn up to its last selected light path.

- **point**: the paths of the clicked pixel that start at the clicked surface. The renderer then
  records every sample of that pixel, which the panels need.
- **sphere, triangle, instance, node, mesh, material**: the light paths that hit, start at or end
  at a sphere around the click, or at what the click sees (an instance is a mesh placed at a
  node); the environment for the sky. It looks through glass. The dropdown next to the match picks
  a target without a click.
- **selection** below holds the ids, the radius, the anchor (`surface point` follows the surface
  when the camera moves, `pixel` stays) and the pixel.

## Filter

Expression, contribution (finite or NaN/Inf), scatter events and technique. It applies where it is
asked for: `Paths › filtered` (on by default), the `(filtered)` images and `View › Overlay ›
filtered`. The analysis never filters.

Path expressions match the scatter events of a light path, camera to light: `S G D U` specular,
glossy, diffuse, unclassified; `R T` reflection, transmission; `.` any; a quoted code for the
technique that drew the event's outgoing direction: `'b'` BSDF, `'g'` guiding, `'n'` NEE, `'c'`
cosine, `'u'` uniform, `'rrb'` `'rrn'` replayed BSDF and NEE, `'rec'` reconnection; `<D'g'>` all
of; `[SG]` either; `* + ? {n} {n,m}`. The last event's technique is how the light was reached;
`L` alone is a light the camera sees. Hover the field for examples.

## Images

| Image | Shows |
| --- | --- |
| source input | the renderer's output |
| current frame | this frame's recorded paths, per pixel |
| accumulated | the recorded paths over the frames, the default |
| (filtered) | what the region and the filter add |
| reference, difference, split | against the reference (View › Reference) |

FROZEN, the recorded images stay on the surfaces while the camera moves. View › Background draws the
scene's triangle edges (the default) where they leave gaps and over them where the scene is closer,
or its albedo, which hides the images behind it. When the header's record probability drops below
1, raise the renderer's `capture budget`.

## Recipes

| Goal | How |
| --- | --- |
| Typical paths | the defaults: paths resampled by contribution |
| Fireflies | `draw: brightest share`, then Paths › List |
| Where a firefly's weight explodes | `color: BSDF·cos/pdf`: red above 1 |
| How light takes on color | `color: transported color` |
| NaN or Inf | Filter › `contribution: NaN or Inf` |
| Light of one lamp | `instance`, `ends at`, click the lamp |
| Where light through a spot lands | `sphere`, `hits`, image `accumulated (filtered)` |
| Where random replay carries light | Filter `.*['rrb''rrn'].*`, image `accumulated (filtered)` |
| Noisy pixels, fireflies, black pixels | overlay `screen space`: relative standard error, max / mean, zero fraction |
| Where light flows in the scene | overlay `world space`: contribution, incident radiance |
| Where NEE is wasted | overlay `world space: NEE failed` |
| Is the BSDF valid, does sample() draw pdf() | Analysis › BSDF check |
| Does the renderer draw what it records | Analysis › Sampling, the z-score panels |
| Which bounces carry the light | Analysis › Statistics › light paths |
| Orbit around paths | `keep on camera move`, on by default; FROZEN also holds the images |
| Bias | the Error Plot node, or `examples/compare_methods.py` |

## Panels

Shown while a point is selected; every panel bins directions at the point. The columns are three
samplers: the renderer's continuations, its NEE, and the material's own `sample()`. Per column:
what it should follow, what it drew, the pdf it claims, and the z-score of the two (blue too few
draws, red too many). Hover a bin for its values; View › Sphere draws any panel around the point,
with `scale` as a polar plot.

- chi2/dof near 1 (green) passes. A mixture records the mixture's pdf, resampled NEE its realized
  pdf.
- NEE draws lights, not directions: lights behind each other (closed or stacked emitters) read red.
- Reused samples (ReSTIR) are tested with batch means; raise `batch frames` when asked to.

## BSDF check

`check BSDF` draws `sample()` and integrates the material at the incident direction (camera ray or
the disc): is it valid (energy, reciprocity, finite, weight = BSDF·cos/pdf), and does `sample()`
draw `pdf()`.

- The check tops the draws up to 1G over the next frames while the point is on screen; the
  verdicts update with them.
- `sample()`, `pdf()` and the check see one hit of the jittered camera until the next reset, so a
  textured material is checked at one texel of the pixel.
- The pdf integrates to the share of continuous draws, not to 1, when `sample()` can reject or
  draws a singular lobe (a smooth pane, a pass-through).
- A lobe too narrow for the integration grid reads n/a.
- `get_albedo()` is an approximation for denoiser demodulation: APPROX when far off, n/a with
  refraction, whose sample weights carry eta².

## Notes

- ReSTIR PT: `debugger records: candidates` for the analysis, `shaded paths` to see reconnection and
  replay.
- Segments behind geometry are tested against the gbuffer; glass counts as an occluder.
- ReSTIR DI's reservoirs keep no light: its connections end on a sphere or the environment, no
  other target.
