# Accumulation

Calculates accumulated irradiance and the first two moments.

- Includes a simple but effective firefly filter for real-time usage.
  Calculates percentiles within 8x8 clusters,
  then clamps the luminance by `upper_percentile + user_factor * (upper_percentile - lower_percentile)`

- Includes simple but effective "adaptive alpha reduction".
  For history values that are larger than `upper_percentile` or lower than `lower_percentile` alpha can be reduced dynamically.
  This helphs agains smearing, ghosting and fireflies in the history.

Inputs:

| Type   | Input ID | Input name   | Description                                                                            | Delay |
|--------|----------|--------------|----------------------------------------------------------------------------------------|-------|
| Image  | 0        | prev_accum   | feedback last `accum`                                                                  | 1     |
| Image  | 1        | prev_moments | feedback last `moments`                                                                | 1     |
|        |                                                                                                                   
| Image  | 2        | irr          | irradiance input (irradiance in `rgb`, sample count in `a`)                            | no    |
|        |                                                                                                                   
| Image  | 3        | mv           | motion vectors in `r` and `g` channel                                                  | no    |
| Image  | 4        | moments      | moments in `r` and `g` channel                                                         | no    |
|        |
| Buffer | 0        | gbuffer      | GBuffer (see `gbuffer.glsl.h`)                                                         | no    |
| Buffer | 1        | prev_gbuffer | previous GBuffer                                                                       | 1     |

Outputs:

| Type  | Output ID | Output name   | Description                                                 | Format/Resolution        | Persistent |
|-------|-----------|---------------|-------------------------------------------------------------|--------------------------|------------|
| Image | 0         | accum         | exp average of irradiance in `rgb`, history length in `a`   | rgba32f                  | no         |
| Image | 1         | moments_accum | exp average of moments in `rg`                              | rgba32f                  | no         |
