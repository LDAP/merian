## Bloom

Glare around bright light sources. Place it before tone mapping. `intensity` is the fraction of
light scattered into the glare, `falloff` the exponent of its `r^-falloff` tail, `size` the largest and `core` the smallest
glare radius as fractions of the image height, so the glare looks the same at every resolution.

### Inputs:

| Type  | Input name | Description       | Delay |
|-------|------------|-------------------|-------|
| Image | src        | linear HDR image  | no    |

### Outputs:

| Type  | Output name | Description             | Format/Resolution  | Persistent |
|-------|-------------|-------------------------|--------------------|------------|
| Image | out         | image with bloom        | like `src`         | no         |
