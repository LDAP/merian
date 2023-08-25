# Raytrace accumulation

Calculates accumulated irradiance and the first two moments.

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

| Type  | Output ID | Output name | Description                                                 | Format/Resolution        | Persistent |
|-------|-----------|-------------|-------------------------------------------------------------|--------------------------|------------|
| Image | 0         | accum       | exp average of irradiance in `rgb`, history length in `a`   | rgba32f                  | no         |
| Image | 1         | moments     | exp average of moments in `rg`                              | rgba32f                  | no         |
