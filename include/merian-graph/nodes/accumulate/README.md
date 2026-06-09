## Accumulation

Calculates accumulated irradiance and the first two moments.

- Includes a simple but effective firefly filter for real-time usage.
  Calculates percentiles within 8x8 clusters,
  then clamps the luminance by `upper_percentile + user_factor * (upper_percentile - lower_percentile)`

- Includes a simple but effective "adaptive alpha reduction".
  For history values that are larger than `upper_percentile` or lower than `lower_percentile` alpha can be reduced dynamically.
  This helps against smearing, ghosting and fireflies in the history.

The implementation of the "firefly filter" and "adaptive alpha reduction" follow
> Lucas Alber. (2024), Markov Chain Path Guiding for Real-Time Global Illumination and Single-Scattering, MSc Thesis, Karlsruhe Institute of Technology.

Inputs:

| Type       | Input name   | Description                                                                            | Delay |
|------------|--------------|----------------------------------------------------------------------------------------|-------|
| VkImageIn  | src          | irradiance in `rgb`, second moment `a`                                                 | no    |
| VkBufferIn | gbuffer      | GBuffer (see `gbuffer.glsl.h`)                                                         | no    |
| VkImageIn  | mv           | motion vectors in `r` and `g` channel                                                  | no    |
| VkImageIn  | prev_out     | feedback last `out`                                                                  | 1     |
| VkBufferIn | prev_gbuf    | previous GBuffer                                                                       | 1     |
| VkImageIn  | prev_history | feedback last `history`                                                                | 1     |

Outputs:

| Type       | Output name   | Description                                                 | Format/Resolution           | Persistent |
|------------|---------------|-------------------------------------------------------------|-----------------------------|------------|
| VkImageOut | out           | exp average of irradiance in `rgb`, second moment in `a`    | user defined or like irr    | no         |
| VkImageOut | history       | history length in `r`                                       | R32Sfloat                   | no         |

Events:

- `clear`: Sent in `process` if the accumulation buffer is reset
